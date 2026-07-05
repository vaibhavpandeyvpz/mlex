//! Gemma4 ("Gemma 3n" family) architecture: text decoder (sliding + global
//! attention mix, KV sharing across the tail layers, per-layer input
//! embeddings) plus, on checkpoints that carry a `vision_config` /
//! `audio_config`, an optional SigLIP-style vision tower (see [`vision`])
//! and an optional Conformer-style audio tower (see [`audio`]) whose
//! projected embeddings get spliced into the text embedding stream at
//! `image_token_id` / `audio_token_id` placeholder positions before the
//! decoder stack runs (video is handled as a sampled frame sequence
//! through the vision tower). Text-only checkpoints (no `vision_config` /
//! `audio_config` in `config.json`) are completely unaffected: the towers
//! stay `None` and `forward` behaves exactly as before.
//!
//! Weight paths keep their `language_model.model.*` checkpoint prefix
//! throughout (matching mlx-lm), so quantization overrides keyed the same
//! way in `config.json` line up without any renaming.

pub mod audio;
pub mod unified;
pub mod vision;

use serde_json::Value;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::media::audio::ProcessedAudio;
use crate::media::image::ProcessedImage;
use crate::nn::{Embedding, Linear, RmsNorm, WeightMap};
use crate::ops::{self, AttentionMask};
use crate::quant::Quantization;

use super::base::splice_media_features;
use super::cache::{KvCache, LayerCache};
use super::config::{get_bool, get_f32, get_i32, require_i32, text_config};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub head_dim: i32,
    pub global_head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub vocab_size_per_layer_input: i32,
    pub num_key_value_heads: i32,
    pub num_global_key_value_heads: Option<i32>,
    pub num_kv_shared_layers: i32,
    pub hidden_size_per_layer_input: i32,
    pub sliding_window: i32,
    pub attention_k_eq_v: bool,
    pub final_logit_softcapping: Option<f32>,
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<LayerType>,
    pub full_attention_rope_theta: f32,
    pub full_attention_partial_rotary_factor: f32,
    pub sliding_attention_rope_theta: f32,
}

impl Gemma4Config {
    pub fn from_json(root: &Value) -> Result<Self> {
        let cfg = text_config(root);
        let hidden_size = require_i32(cfg, "hidden_size")?;
        let num_hidden_layers = require_i32(cfg, "num_hidden_layers")?;
        let sliding_window_pattern = get_i32(cfg, "sliding_window_pattern", 5);

        let layer_types = match cfg.get("layer_types").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .map(|v| match v.as_str() {
                    Some("full_attention") => LayerType::Full,
                    _ => LayerType::Sliding,
                })
                .collect(),
            None => (0..num_hidden_layers)
                .map(|i| {
                    if (i + 1) % sliding_window_pattern == 0 {
                        LayerType::Full
                    } else {
                        LayerType::Sliding
                    }
                })
                .collect(),
        };

        let rope_params = cfg.get("rope_parameters");
        let full_rp = rope_params.and_then(|r| r.get("full_attention"));
        let sliding_rp = rope_params.and_then(|r| r.get("sliding_attention"));

        Ok(Gemma4Config {
            hidden_size,
            num_hidden_layers,
            num_attention_heads: require_i32(cfg, "num_attention_heads")?,
            head_dim: get_i32(cfg, "head_dim", 256),
            global_head_dim: get_i32(cfg, "global_head_dim", 512),
            rms_norm_eps: get_f32(cfg, "rms_norm_eps", 1e-6),
            vocab_size: require_i32(cfg, "vocab_size")?,
            vocab_size_per_layer_input: get_i32(cfg, "vocab_size_per_layer_input", 0),
            num_key_value_heads: get_i32(cfg, "num_key_value_heads", 1),
            num_global_key_value_heads: cfg
                .get("num_global_key_value_heads")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32),
            num_kv_shared_layers: get_i32(cfg, "num_kv_shared_layers", 0),
            hidden_size_per_layer_input: get_i32(cfg, "hidden_size_per_layer_input", 0),
            sliding_window: get_i32(cfg, "sliding_window", 512),
            attention_k_eq_v: get_bool(cfg, "attention_k_eq_v", false),
            final_logit_softcapping: cfg
                .get("final_logit_softcapping")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            tie_word_embeddings: get_bool(cfg, "tie_word_embeddings", true),
            layer_types,
            full_attention_rope_theta: full_rp
                .map(|r| get_f32(r, "rope_theta", 1_000_000.0))
                .unwrap_or(1_000_000.0),
            full_attention_partial_rotary_factor: full_rp
                .map(|r| get_f32(r, "partial_rotary_factor", 0.25))
                .unwrap_or(0.25),
            sliding_attention_rope_theta: sliding_rp
                .map(|r| get_f32(r, "rope_theta", 10_000.0))
                .unwrap_or(10_000.0),
        })
    }

    fn first_kv_shared_layer(&self) -> i32 {
        self.num_hidden_layers - self.num_kv_shared_layers
    }
}

/// RoPE variant: standard inverse-frequency rotation over the full head
/// dimension, or a "proportional" partial rotation with precomputed
/// per-pair frequencies (unrotated dims get an `inf` divisor, i.e. identity).
enum Rope {
    Standard { dims: i32, theta: f32 },
    Proportional { dims: i32, freqs: Array },
}

impl Rope {
    fn standard(dims: i32, theta: f32) -> Self {
        Rope::Standard { dims, theta }
    }

    fn proportional(dims: i32, theta: f32, partial_rotary_factor: f32) -> Self {
        let rotated_dims = ((dims as f32) * partial_rotary_factor) as i32;
        let n_pairs = (dims / 2) as usize;
        let n_rotated_pairs = (rotated_dims / 2) as usize;
        let mut freqs = Vec::with_capacity(n_pairs);
        for i in 0..n_rotated_pairs {
            let exponent = (2 * i) as f32 / dims as f32;
            freqs.push(theta.powf(exponent));
        }
        for _ in n_rotated_pairs..n_pairs {
            freqs.push(f32::INFINITY);
        }
        Rope::Proportional {
            dims,
            freqs: Array::from_slice(&freqs, &[n_pairs as i32]),
        }
    }

    fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
        match self {
            Rope::Standard { dims, theta } => {
                ops::rope(x, *dims, false, Some(*theta), 1.0, offset, None)
            }
            Rope::Proportional { dims, freqs } => {
                ops::rope(x, *dims, false, None, 1.0, offset, Some(freqs))
            }
        }
    }
}

/// RMSNorm with no learnable scale (used for the value-projection norm).
fn rms_norm_no_scale(x: &Array, eps: f32) -> Result<Array> {
    ops::rms_norm(x, None, eps)
}

struct Attention {
    q_proj: Linear,
    k_proj: Option<Linear>,
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    rope: Rope,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    is_sliding: bool,
    use_k_eq_v: bool,
    sliding_window: i32,
    eps: f32,
}

impl Attention {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &Gemma4Config, layer_idx: i32) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        let layer_type = cfg.layer_types[layer_idx as usize];
        let is_sliding = layer_type == LayerType::Sliding;
        let has_kv = layer_idx < cfg.first_kv_shared_layer();

        let head_dim = if layer_type == LayerType::Full {
            cfg.global_head_dim
        } else {
            cfg.head_dim
        };
        let use_k_eq_v = cfg.attention_k_eq_v && !is_sliding;
        let n_kv_heads = if use_k_eq_v {
            cfg.num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads)
        } else {
            cfg.num_key_value_heads
        };

        let (k_proj, v_proj, k_norm) = if has_kv {
            let k_proj = w.linear(&format!("{attn}.k_proj"))?;
            let v_proj = if use_k_eq_v {
                None
            } else {
                Some(w.linear(&format!("{attn}.v_proj"))?)
            };
            let k_norm = w.rms_norm(&format!("{attn}.k_norm"), cfg.rms_norm_eps)?;
            (Some(k_proj), v_proj, Some(k_norm))
        } else {
            (None, None, None)
        };

        let rope = if layer_type == LayerType::Full {
            Rope::proportional(
                head_dim,
                cfg.full_attention_rope_theta,
                cfg.full_attention_partial_rotary_factor,
            )
        } else {
            Rope::standard(head_dim, cfg.sliding_attention_rope_theta)
        };

        Ok(Attention {
            q_proj: w.linear(&format!("{attn}.q_proj"))?,
            k_proj,
            v_proj,
            o_proj: w.linear(&format!("{attn}.o_proj"))?,
            q_norm: w.rms_norm(&format!("{attn}.q_norm"), cfg.rms_norm_eps)?,
            k_norm,
            rope,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            is_sliding,
            use_k_eq_v,
            sliding_window: cfg.sliding_window,
            eps: cfg.rms_norm_eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Array,
        cache: Option<&mut KvCache>,
        shared_kv: Option<(Array, Array, i32)>,
    ) -> Result<(Array, Array, Array, i32)> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?;
        let q = ops::reshape(&q, &[b, l, self.n_heads, self.head_dim])?;
        let q = self.q_norm.forward(&q)?;
        let q = ops::transpose_axes(&q, &[0, 2, 1, 3])?;

        let (keys, values, offset) = if let Some((k, v, off)) = shared_kv {
            (k, v, off)
        } else {
            let cache = cache
                .ok_or_else(|| Error::Model("gemma4: KV-owning layer missing cache".into()))?;
            let k_proj = self.k_proj.as_ref().unwrap();
            let k = k_proj.forward(x)?;
            let k = ops::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim])?;
            let v = if self.use_k_eq_v {
                k.clone()
            } else {
                let v_proj = self.v_proj.as_ref().unwrap();
                let v = v_proj.forward(x)?;
                ops::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim])?
            };

            let offset = cache.offset();
            let k = self.k_norm.as_ref().unwrap().forward(&k)?;
            let k = ops::transpose_axes(&k, &[0, 2, 1, 3])?;
            let k = self.rope.apply(&k, offset)?;
            let v = rms_norm_no_scale(&v, self.eps)?;
            let v = ops::transpose_axes(&v, &[0, 2, 1, 3])?;
            let (k, v) = cache.update_and_fetch(k, v)?;
            (k, v, offset)
        };

        let q = self.rope.apply(&q, offset)?;

        let kv_len = keys.dim(-2);
        let out = if self.is_sliding {
            let mask = ops::sliding_window_mask(l, kv_len, offset, self.sliding_window, q.dtype())?;
            ops::scaled_dot_product_attention_masked(&q, &keys, &values, 1.0, &mask)?
        } else {
            let mask = if l == 1 {
                AttentionMask::None
            } else {
                AttentionMask::Causal
            };
            ops::scaled_dot_product_attention(&q, &keys, &values, 1.0, mask)?
        };
        let out = ops::transpose_axes(&out, &[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, l, -1])?;
        let out = self.o_proj.forward(&out)?;
        Ok((out, keys, values, offset))
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(w: &mut WeightMap, prefix: &str) -> Result<Self> {
        let mlp = format!("{prefix}.mlp");
        Ok(Mlp {
            gate_proj: w.linear(&format!("{mlp}.gate_proj"))?,
            up_proj: w.linear(&format!("{mlp}.up_proj"))?,
            down_proj: w.linear(&format!("{mlp}.down_proj"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = ops::gelu_tanh(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&ops::multiply(&gate, &up)?)
    }
}

struct Block {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    per_layer_input_gate: Option<Linear>,
    per_layer_projection: Option<Linear>,
    post_per_layer_input_norm: Option<RmsNorm>,
    layer_scalar: Array,
}

impl Block {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &Gemma4Config, layer_idx: i32) -> Result<Self> {
        let has_ple = cfg.hidden_size_per_layer_input > 0;
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) = if has_ple {
            (
                Some(w.linear(&format!("{prefix}.per_layer_input_gate"))?),
                Some(w.linear(&format!("{prefix}.per_layer_projection"))?),
                Some(w.rms_norm(
                    &format!("{prefix}.post_per_layer_input_norm"),
                    cfg.rms_norm_eps,
                )?),
            )
        } else {
            (None, None, None)
        };

        Ok(Block {
            self_attn: Attention::load(w, prefix, cfg, layer_idx)?,
            mlp: Mlp::load(w, prefix)?,
            input_layernorm: w.rms_norm(&format!("{prefix}.input_layernorm"), cfg.rms_norm_eps)?,
            post_attention_layernorm: w.rms_norm(
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_norm_eps,
            )?,
            pre_feedforward_layernorm: w.rms_norm(
                &format!("{prefix}.pre_feedforward_layernorm"),
                cfg.rms_norm_eps,
            )?,
            post_feedforward_layernorm: w.rms_norm(
                &format!("{prefix}.post_feedforward_layernorm"),
                cfg.rms_norm_eps,
            )?,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar: w.take(&format!("{prefix}.layer_scalar"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Array,
        cache: Option<&mut KvCache>,
        shared_kv: Option<(Array, Array, i32)>,
        per_layer_input: Option<&Array>,
    ) -> Result<(Array, Array, Array, i32)> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let (h, keys, values, offset) = self.self_attn.forward(&h, cache, shared_kv)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = ops::add(&residual, &h)?;

        let residual = h.clone();
        let m = self.pre_feedforward_layernorm.forward(&h)?;
        let m = self.mlp.forward(&m)?;
        let m = self.post_feedforward_layernorm.forward(&m)?;
        let mut h = ops::add(&residual, &m)?;

        if let (Some(gate_proj), Some(proj), Some(norm), Some(pli)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = h.clone();
            let gate = gate_proj.forward(&h)?;
            let gate = ops::gelu_tanh(&gate)?;
            let gate = ops::multiply(&gate, pli)?;
            let gate = proj.forward(&gate)?;
            let gate = norm.forward(&gate)?;
            h = ops::add(&residual, &gate)?;
        }

        h = ops::multiply(&h, &self.layer_scalar)?;
        Ok((h, keys, values, offset))
    }
}

/// The vision encoder backing a checkpoint's image/video input: either the
/// classic SigLIP-style transformer tower, or the encoder-free "unified"
/// patch embedder (see [`unified`]) some OptiQ checkpoints ship instead.
enum VisionEncoder {
    Tower(vision::VisionTower),
    Unified(unified::UnifiedVisionEmbedder),
}

impl VisionEncoder {
    fn forward(&self, pixel_values: &Array, patch_h: i32, patch_w: i32) -> Result<Array> {
        match self {
            VisionEncoder::Tower(t) => t.forward(pixel_values, patch_h, patch_w),
            VisionEncoder::Unified(u) => u.forward(pixel_values, patch_h, patch_w),
        }
    }
}

/// The audio encoder backing a checkpoint's audio input: either the
/// classic Conformer-style transformer tower (mel-spectrogram input), or
/// the encoder-free "unified" path (raw PCM windows projected directly).
enum AudioEncoder {
    Tower(audio::AudioTower),
    /// Raw samples per audio token (640 = 40 ms @ 16 kHz); no tower.
    Unified {
        samples_per_token: i32,
    },
}

/// Optional image support (`Some` only on checkpoints whose `config.json`
/// carries a `vision_config` sub-dict with a matching `vision_tower.*` /
/// `vision_embedder.*` / `embed_vision.*` weight set).
struct VisionSupport {
    encoder: VisionEncoder,
    embedder: vision::MultimodalEmbedder,
    patch_size: i32,
    max_soft_tokens: i32,
    pooling_kernel_size: i32,
    image_token_id: i32,
    boi_token_id: i32,
    eoi_token_id: i32,
}

/// Optional audio support (`Some` only on checkpoints whose `config.json`
/// carries an `audio_config` sub-dict with a matching `audio_tower.*` /
/// `embed_audio.*` weight set).
struct AudioSupport {
    encoder: AudioEncoder,
    embedder: vision::MultimodalEmbedder,
    audio_token_id: i32,
    boa_token_id: i32,
    eoa_token_id: i32,
}

/// Gemma4 causal language model: text decoder, plus optional vision and
/// audio towers for checkpoints that support image/audio input.
pub struct Gemma4Model {
    pub config: Gemma4Config,
    embed_tokens: Embedding,
    embed_tokens_per_layer: Option<Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<RmsNorm>,
    layers: Vec<Block>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    embed_scale: f32,
    embed_tokens_per_layer_scale: f32,
    per_layer_input_scale: f32,
    per_layer_projection_scale: f32,
    first_kv_shared_layer: i32,
    vision: Option<VisionSupport>,
    audio: Option<AudioSupport>,
    video_token_id_raw: i32,
}

impl Gemma4Model {
    pub fn load(mut weights: WeightMap, config_json: &Value) -> Result<Self> {
        let cfg = Gemma4Config::from_json(config_json)?;
        let prefix = "language_model.model";

        let embed_tokens = weights.embedding(&format!("{prefix}.embed_tokens"))?;

        let has_ple = cfg.hidden_size_per_layer_input > 0;
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if has_ple {
                (
                    Some(weights.embedding(&format!("{prefix}.embed_tokens_per_layer"))?),
                    Some(weights.linear(&format!("{prefix}.per_layer_model_projection"))?),
                    Some(weights.rms_norm(
                        &format!("{prefix}.per_layer_projection_norm"),
                        cfg.rms_norm_eps,
                    )?),
                )
            } else {
                (None, None, None)
            };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Block::load(
                &mut weights,
                &format!("{prefix}.layers.{i}"),
                &cfg,
                i,
            )?);
        }
        let norm = weights.rms_norm(&format!("{prefix}.norm"), cfg.rms_norm_eps)?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(weights.linear("language_model.lm_head")?)
        };

        let embed_scale = (cfg.hidden_size as f32).sqrt();
        let embed_tokens_per_layer_scale = (cfg.hidden_size_per_layer_input as f32).sqrt();
        let per_layer_input_scale = 2f32.powf(-0.5);
        let per_layer_projection_scale = (cfg.hidden_size as f32).powf(-0.5);
        let first_kv_shared_layer = cfg.first_kv_shared_layer();

        let vision = Self::load_vision(&mut weights, config_json)?;
        let audio = Self::load_audio(&mut weights, config_json)?;

        Ok(Gemma4Model {
            config: cfg,
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            embed_scale,
            embed_tokens_per_layer_scale,
            per_layer_input_scale,
            per_layer_projection_scale,
            first_kv_shared_layer,
            vision,
            audio,
            video_token_id_raw: get_i32(config_json, "video_token_id", 258884),
        })
    }

    /// Build the vision encoder iff `config.json` carries a `vision_config`
    /// sub-dict AND the checkpoint actually shipped the matching weights
    /// (a checkpoint could declare `vision_config` while being distributed
    /// text-only); returns `None` otherwise so text-only checkpoints load
    /// exactly as before. Two encoder shapes are recognized (see
    /// [`VisionEncoder`]): the classic SigLIP tower
    /// (`vision_tower.patch_embedder.*`), and the encoder-free "unified"
    /// patch embedder (`vision_embedder.patch_dense.*`) some OptiQ
    /// checkpoints ship instead - selected by whichever weights are
    /// actually present.
    fn load_vision(weights: &mut WeightMap, config_json: &Value) -> Result<Option<VisionSupport>> {
        let Some(vision_cfg_json) = config_json.get("vision_config") else {
            return Ok(None);
        };

        let (encoder, patch_size, max_soft_tokens, pooling_kernel_size, eps) =
            if weights.contains("vision_tower.patch_embedder.input_proj.weight") {
                let cfg = vision::VisionConfig::from_json(vision_cfg_json);
                let params = (
                    cfg.patch_size,
                    cfg.default_output_length,
                    cfg.pooling_kernel_size,
                    cfg.rms_norm_eps,
                );
                let tower = vision::VisionTower::load(weights, cfg)?;
                (
                    VisionEncoder::Tower(tower),
                    params.0,
                    params.1,
                    params.2,
                    params.3,
                )
            } else if weights.contains("vision_embedder.patch_dense.weight") {
                let cfg = unified::UnifiedVisionConfig::from_json(vision_cfg_json);
                let params = (
                    cfg.patch_size,
                    cfg.num_soft_tokens,
                    cfg.pooling_kernel_size,
                    cfg.rms_norm_eps,
                );
                let embedder = unified::UnifiedVisionEmbedder::load(weights, &cfg)?;
                (
                    VisionEncoder::Unified(embedder),
                    params.0,
                    params.1,
                    params.2,
                    params.3,
                )
            } else {
                return Ok(None);
            };

        let embedder = vision::MultimodalEmbedder::load(weights, "embed_vision", eps)?;

        let image_token_id = get_i32(config_json, "image_token_id", 258880);
        let boi_token_id = get_i32(config_json, "boi_token_id", 255999);
        let eoi_token_id = get_i32(config_json, "eoi_token_id", 258882);

        Ok(Some(VisionSupport {
            encoder,
            embedder,
            patch_size,
            max_soft_tokens,
            pooling_kernel_size,
            image_token_id,
            boi_token_id,
            eoi_token_id,
        }))
    }

    /// Build the audio encoder iff `config.json` carries an `audio_config`
    /// sub-dict AND the checkpoint actually shipped a matching weight set;
    /// returns `None` otherwise so text-only and image-only checkpoints
    /// load exactly as before (mirrors `load_vision`). Two encoder shapes
    /// are recognized (see [`AudioEncoder`]): the classic Conformer tower
    /// (`audio_tower.subsample_conv_projection.*`, mel-spectrogram input),
    /// and the encoder-free "unified" path (no tower weights at all - raw
    /// PCM windows feed `embed_audio` directly) some OptiQ checkpoints
    /// ship instead.
    fn load_audio(weights: &mut WeightMap, config_json: &Value) -> Result<Option<AudioSupport>> {
        let Some(audio_cfg_json) = config_json.get("audio_config") else {
            return Ok(None);
        };

        let encoder =
            if weights.contains("audio_tower.subsample_conv_projection.input_proj_linear.weight") {
                let audio_cfg = audio::AudioConfig::from_json(audio_cfg_json);
                AudioEncoder::Tower(audio::AudioTower::load(weights, audio_cfg)?)
            } else if weights.contains("embed_audio.embedding_projection.weight") {
                let samples_per_token = get_i32(audio_cfg_json, "audio_samples_per_token", 640);
                AudioEncoder::Unified { samples_per_token }
            } else {
                return Ok(None);
            };

        let eps = get_f32(audio_cfg_json, "rms_norm_eps", 1e-6);
        let embedder = vision::MultimodalEmbedder::load(weights, "embed_audio", eps)?;

        let audio_token_id = get_i32(config_json, "audio_token_id", 258881);
        let boa_token_id = get_i32(config_json, "boa_token_id", 256000);
        let eoa_token_id = get_i32(config_json, "eoa_token_id", 258883);

        Ok(Some(AudioSupport {
            encoder,
            embedder,
            audio_token_id,
            boa_token_id,
            eoa_token_id,
        }))
    }

    /// TEMP debug hook.
    pub fn debug_vision_forward(&self, image: &ProcessedImage) -> Result<Vec<f32>> {
        let vision = self.vision.as_ref().unwrap();
        let feats = vision
            .encoder
            .forward(&image.pixel_values, image.patch_h, image.patch_w)?;
        eprintln!("raw vision tower feats shape: {:?}", feats.shape());
        let raw = feats.to_vec_f32()?;
        let mean: f32 = raw.iter().sum::<f32>() / raw.len() as f32;
        let min = raw.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("raw vision feats: mean={mean} min={min} max={max}");
        let proj = vision.embedder.forward(&feats)?;
        proj.to_vec_f32()
    }

    /// Whether this checkpoint's vision tower was loaded (i.e. it can
    /// accept image input via [`Gemma4Model::forward_with_images`]).
    pub fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    /// `(patch_size, max_soft_tokens, pooling_kernel_size)` for
    /// [`crate::media::image::preprocess_image_bytes`], or `None` if this
    /// checkpoint has no vision support.
    pub fn image_processing_params(&self) -> Option<(i32, i32, i32)> {
        self.vision
            .as_ref()
            .map(|v| (v.patch_size, v.max_soft_tokens, v.pooling_kernel_size))
    }

    /// `(image_token_id, boi_token_id, eoi_token_id)`, or `None` if this
    /// checkpoint has no vision support.
    pub fn image_token_ids(&self) -> Option<(u32, u32, u32)> {
        self.vision.as_ref().map(|v| {
            (
                v.image_token_id as u32,
                v.boi_token_id as u32,
                v.eoi_token_id as u32,
            )
        })
    }

    /// Whether this checkpoint's audio tower was loaded (i.e. it can
    /// accept audio input via [`Gemma4Model::forward_with_media`]).
    pub fn supports_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// `(audio_token_id, boa_token_id, eoa_token_id)`, or `None` if this
    /// checkpoint has no audio support.
    pub fn audio_token_ids(&self) -> Option<(u32, u32, u32)> {
        self.audio.as_ref().map(|a| {
            (
                a.audio_token_id as u32,
                a.boa_token_id as u32,
                a.eoa_token_id as u32,
            )
        })
    }

    /// Raw PCM samples per audio token for the encoder-free "unified"
    /// audio path, or `None` if this checkpoint either has no audio
    /// support or uses the classic mel-spectrogram Conformer tower
    /// (which has its own chunking in [`crate::media::audio`]).
    pub fn audio_samples_per_token(&self) -> Option<i32> {
        match self.audio.as_ref()?.encoder {
            AudioEncoder::Unified { samples_per_token } => Some(samples_per_token),
            AudioEncoder::Tower(_) => None,
        }
    }

    /// `video_token_id` from `config.json`, or `None` when the checkpoint
    /// has no vision tower (video reuses the vision path per frame).
    pub fn video_token_id(&self) -> Option<u32> {
        self.vision.as_ref().map(|_| self.video_token_id_raw as u32)
    }

    /// Only layers that own their own KV cache need a slot; KV-shared tail
    /// layers reuse an earlier layer's keys/values.
    pub fn num_cache_layers(&self) -> usize {
        self.first_kv_shared_layer.max(0) as usize
    }

    pub fn new_caches(&self) -> Vec<LayerCache> {
        (0..self.num_cache_layers())
            .map(|_| LayerCache::new_attention())
            .collect()
    }

    fn slice_layer_index(x: &Array, index: i32) -> Result<Array> {
        let shape = x.shape();
        let (b, s, _, h) = (shape[0], shape[1], shape[2], shape[3]);
        let sliced = ops::slice(x, &[0, 0, index, 0], &[b, s, index + 1, h])?;
        ops::reshape(&sliced, &[b, s, h])
    }

    pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
        let embeddings = self.embed_tokens.forward(input_ids)?;
        let h = ops::scale_by(&embeddings, self.embed_scale)?;
        self.forward_from_embeds(input_ids, h, caches)
    }

    /// Same as [`Gemma4Model::forward`], but splices `images`' projected
    /// vision features into the text embedding stream at `image_token_id`
    /// placeholder positions before running the decoder stack (order
    /// preserving across multiple images; each image's soft-token count
    /// must exactly match the number of `image_token_id` placeholders
    /// `input_ids` holds for it, produced by expanding the chat template's
    /// single-placeholder-per-image convention beforehand - see
    /// `crate::generate::Session::encode_chat_with_media`).
    pub fn forward_with_images(
        &self,
        input_ids: &Array,
        images: &[ProcessedImage],
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        self.forward_with_media(input_ids, images, &[], caches)
    }

    /// Same as [`Gemma4Model::forward`], but splices `images`' projected
    /// vision features in at `image_token_id` placeholder positions and
    /// `audios`' projected audio features at `audio_token_id` placeholder
    /// positions (each modality order-preserving; per-clip soft-token
    /// counts must exactly match the placeholder expansion done by
    /// `crate::generate::Session::encode_chat_with_media`). Video frames
    /// arrive here as ordinary entries of `images`.
    pub fn forward_with_media(
        &self,
        input_ids: &Array,
        images: &[ProcessedImage],
        audios: &[ProcessedAudio],
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        let embeddings = self.embed_tokens.forward(input_ids)?;
        let mut h = ops::scale_by(&embeddings, self.embed_scale)?;

        if !images.is_empty() {
            let vision = self.vision.as_ref().ok_or_else(|| {
                Error::Model("gemma4: model has no vision support (no vision_config)".into())
            })?;
            let mut all_features = Vec::with_capacity(images.len());
            for image in images {
                let features =
                    vision
                        .encoder
                        .forward(&image.pixel_values, image.patch_h, image.patch_w)?;
                let features = vision.embedder.forward(&features)?;
                all_features.push(features);
            }
            h = splice_media_features(&h, input_ids, all_features, vision.image_token_id, "image")?;
        }

        if !audios.is_empty() {
            let audio = self.audio.as_ref().ok_or_else(|| {
                Error::Model("gemma4: model has no audio support (no audio_config)".into())
            })?;
            let mut all_features = Vec::new();
            for clip in audios {
                for chunk in &clip.chunks {
                    let features = match &audio.encoder {
                        AudioEncoder::Tower(tower) => {
                            let features = tower.forward(chunk)?;
                            audio.embedder.forward(&features)?
                        }
                        AudioEncoder::Unified { .. } => {
                            // `chunk` is `[n_frames, samples_per_token]` raw PCM
                            // windows (see `crate::media::audio::preprocess_audio_bytes_raw`);
                            // no tower, straight into `embed_audio`.
                            let features = audio.embedder.forward(chunk)?;
                            ops::expand_dims(&features, 0)?
                        }
                    };
                    all_features.push(features);
                }
            }
            h = splice_media_features(&h, input_ids, all_features, audio.audio_token_id, "audio")?;
        }

        self.forward_from_embeds(input_ids, h, caches)
    }

    /// Shared decode stack: `h` is the already-computed (and, for
    /// multimodal turns, already media-fused) initial hidden state;
    /// `input_ids` is still needed for the per-layer input embedding table
    /// lookup (Gemma3n-style), which is keyed on token ids regardless of
    /// media fusion.
    fn forward_from_embeds(
        &self,
        input_ids: &Array,
        mut h: Array,
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        let per_layer_inputs: Vec<Option<Array>> = if self.config.hidden_size_per_layer_input > 0 {
            let ple = self
                .embed_tokens_per_layer
                .as_ref()
                .unwrap()
                .forward(input_ids)?;
            let ple = ops::scale_by(&ple, self.embed_tokens_per_layer_scale)?;
            let shape = ple.shape();
            let ple = ops::reshape(
                &ple,
                &[
                    shape[0],
                    shape[1],
                    self.config.num_hidden_layers,
                    self.config.hidden_size_per_layer_input,
                ],
            )?;

            let proj = self
                .per_layer_model_projection
                .as_ref()
                .unwrap()
                .forward(&h)?;
            let proj = ops::scale_by(&proj, self.per_layer_projection_scale)?;
            let pshape = proj.shape();
            let proj = ops::reshape(
                &proj,
                &[
                    pshape[0],
                    pshape[1],
                    self.config.num_hidden_layers,
                    self.config.hidden_size_per_layer_input,
                ],
            )?;
            let proj = self
                .per_layer_projection_norm
                .as_ref()
                .unwrap()
                .forward(&proj)?;

            let combined = ops::scale_by(&ops::add(&proj, &ple)?, self.per_layer_input_scale)?;
            (0..self.layers.len())
                .map(|i| Self::slice_layer_index(&combined, i as i32).map(Some))
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..self.layers.len()).map(|_| None).collect()
        };

        // `intermediates[i]` holds the (keys, values, offset) computed by
        // layer `i`, used both by that layer's own forward pass bookkeeping
        // and by any later layer that shares this layer's KV cache.
        let mut intermediates: Vec<Option<(Array, Array, i32)>> = vec![None; self.layers.len()];
        let mut cache_iter = caches.iter_mut();

        for (idx, layer) in self.layers.iter().enumerate() {
            let idx_i32 = idx as i32;
            let per_layer_input = per_layer_inputs[idx].as_ref();

            let (h_out, keys, values, offset) = if idx_i32 < self.first_kv_shared_layer {
                let cache = cache_iter
                    .next()
                    .ok_or_else(|| {
                        Error::Model("gemma4: not enough KV caches for owning layers".into())
                    })?
                    .as_attention()?;
                layer.forward(&h, Some(cache), None, per_layer_input)?
            } else {
                let layer_type = self.config.layer_types[idx];
                // Mirrors mlx-lm's `kvs_by_type` map, which keeps getting
                // overwritten while scanning 0..first_kv_shared_layer, i.e.
                // ends up pointing at the LAST KV-owning layer of this type.
                let source = (0..self.first_kv_shared_layer as usize)
                    .rev()
                    .find(|&j| self.config.layer_types[j] == layer_type)
                    .ok_or_else(|| {
                        Error::Model(format!("gemma4: layer {idx} has no earlier KV source"))
                    })?;
                let (sk, sv, soff) = intermediates[source]
                    .clone()
                    .ok_or_else(|| Error::Model(format!("gemma4: KV source {source} not ready")))?;
                layer.forward(&h, None, Some((sk, sv, soff)), per_layer_input)?
            };

            intermediates[idx] = Some((keys, values, offset));
            h = h_out;
        }

        h = self.norm.forward(&h)?;

        let mut logits = match &self.lm_head {
            Some(head) => head.forward(&h)?,
            None => self.embed_tokens.as_linear(&h)?,
        };
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = ops::scale_by(&ops::tanh(&ops::scale_by(&logits, 1.0 / cap)?)?, cap)?;
        }
        Ok(logits)
    }
}

/// Drop weights outside the towers this crate supports; keep
/// `language_model.*`, `vision_tower.*`/`embed_vision.*` and
/// `audio_tower.*`/`embed_audio.*` untouched since layer/quantization
/// paths are matched against them verbatim.
pub fn sanitize(weights: &mut WeightMap) {
    weights.rename_keys(|k| {
        if k.starts_with("language_model.")
            || k.starts_with("vision_tower.")
            || k.starts_with("vision_embedder.")
            || k.starts_with("embed_vision.")
            || k.starts_with("audio_tower.")
            || k.starts_with("embed_audio.")
        {
            Some(k.to_string())
        } else {
            None
        }
    });
}

pub fn parse_quantization(config_json: &Value) -> Result<Quantization> {
    Quantization::from_config(config_json)
}
