//! Qwen3.5 / Qwen3.6 hybrid architecture: most decoder layers use the
//! linear-time `GatedDeltaNet` recurrence (see `gated_delta.rs`), with a
//! plain GQA self-attention layer every `full_attention_interval` layers.
//! Also covers the MoE variant (Qwen3.6-A3B) via `SparseMoeBlock`.
//!
//! Checkpoints whose `config.json` carries a `vision_config` sub-dict (and
//! ship the matching `vision_tower.*` weights - see [`vision`]) additionally
//! accept image input: [`Qwen35Model::forward_with_media`] splices the
//! vision tower's projected patch features into the text embedding stream
//! at `image_token_id` placeholder positions before the decoder stack runs
//! (video frames arrive as ordinary entries of `images`, one vision-tower
//! pass per frame). Text-only checkpoints are unaffected: the tower stays
//! `None` and `forward` behaves exactly as before.

pub mod vision;

use serde_json::Value;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::media::image::ProcessedImage;
use crate::nn::{Embedding, Linear, RmsNorm, WeightMap};
use crate::ops::{self, AttentionMask};
use crate::quant::Quantization;

use super::base::{attention_mask_for, merge_heads, splice_media_features, RopeConfig};
use super::cache::{GatedDeltaCache, LayerCache};
use super::config::{get_bool, get_f32, get_i32, require_i32, text_config};
use super::gated_delta::{GatedDeltaConfig, GatedDeltaNet};
use super::moe::SparseMoeBlock;
use vision::{Qwen35VisionConfig, Qwen35VisionTower};

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub full_attention_interval: i32,

    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,

    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub decoder_sparse_step: i32,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
    pub norm_topk_prob: bool,
}

impl Qwen35Config {
    pub fn from_json(cfg: &Value) -> Result<Self> {
        let cfg = text_config(cfg);
        let hidden_size = require_i32(cfg, "hidden_size")?;
        let num_attention_heads = require_i32(cfg, "num_attention_heads")?;
        let head_dim = get_i32(cfg, "head_dim", hidden_size / num_attention_heads);
        let rope = cfg.get("rope_parameters");
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(100_000.0);
        let partial_rotary_factor = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.25);

        Ok(Qwen35Config {
            hidden_size,
            num_hidden_layers: require_i32(cfg, "num_hidden_layers")?,
            // Absent on checkpoints where every layer is MoE (e.g. Qwen3.6-A3B),
            // since `Mlp::load` (the dense fallback) is then never reached.
            intermediate_size: get_i32(cfg, "intermediate_size", 0),
            num_attention_heads,
            num_key_value_heads: get_i32(cfg, "num_key_value_heads", num_attention_heads),
            head_dim,
            rms_norm_eps: get_f32(cfg, "rms_norm_eps", 1e-6),
            vocab_size: require_i32(cfg, "vocab_size")?,
            tie_word_embeddings: get_bool(cfg, "tie_word_embeddings", false),
            attention_bias: get_bool(cfg, "attention_bias", false),
            rope_theta,
            partial_rotary_factor,
            full_attention_interval: get_i32(cfg, "full_attention_interval", 4),
            linear_num_value_heads: get_i32(cfg, "linear_num_value_heads", 64),
            linear_num_key_heads: get_i32(cfg, "linear_num_key_heads", 16),
            linear_key_head_dim: get_i32(cfg, "linear_key_head_dim", 192),
            linear_value_head_dim: get_i32(cfg, "linear_value_head_dim", 128),
            linear_conv_kernel_dim: get_i32(cfg, "linear_conv_kernel_dim", 4),
            num_experts: get_i32(cfg, "num_experts", 0),
            num_experts_per_tok: get_i32(cfg, "num_experts_per_tok", 0),
            decoder_sparse_step: get_i32(cfg, "decoder_sparse_step", 1),
            moe_intermediate_size: get_i32(cfg, "moe_intermediate_size", 0),
            shared_expert_intermediate_size: get_i32(cfg, "shared_expert_intermediate_size", 0),
            norm_topk_prob: get_bool(cfg, "norm_topk_prob", true),
        })
    }

    fn is_linear_layer(&self, layer_idx: i32) -> bool {
        (layer_idx + 1) % self.full_attention_interval != 0
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: RopeConfig,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &Qwen35Config) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        let rope_dims = ((cfg.head_dim as f32) * cfg.partial_rotary_factor) as i32;
        Ok(Attention {
            q_proj: w.linear(&format!("{attn}.q_proj"))?,
            k_proj: w.linear(&format!("{attn}.k_proj"))?,
            v_proj: w.linear(&format!("{attn}.v_proj"))?,
            o_proj: w.linear(&format!("{attn}.o_proj"))?,
            q_norm: w.rms_norm(&format!("{attn}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: w.rms_norm(&format!("{attn}.k_norm"), cfg.rms_norm_eps)?,
            rope: RopeConfig::new(rope_dims, cfg.rope_theta),
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &Array,
        mask: AttentionMask,
        cache: &mut super::cache::KvCache,
    ) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q_out = self.q_proj.forward(x)?;
        let q_out = ops::reshape(&q_out, &[b, l, self.n_heads, 2 * self.head_dim])?;
        let parts = ops::split(&q_out, 2, -1)?;
        let (queries, gate) = (&parts[0], &parts[1]);
        let gate = ops::reshape(gate, &[b, l, self.n_heads * self.head_dim])?;

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let k = ops::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim])?;
        let v = ops::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim])?;

        let queries = self.q_norm.forward(queries)?;
        let queries = ops::transpose_axes(&queries, &[0, 2, 1, 3])?;
        let k = self.k_norm.forward(&k)?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3])?;

        let offset = cache.offset();
        let queries = self.rope.apply(&queries, offset)?;
        let k = self.rope.apply(&k, offset)?;
        let (k, v) = cache.update_and_fetch(k, v)?;

        let out = ops::scaled_dot_product_attention(&queries, &k, &v, self.scale, mask)?;
        let out = merge_heads(&out, b, l)?;
        let gated = ops::multiply(&out, &ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(w: &mut WeightMap, prefix: &str, hidden: i32, intermediate: i32) -> Result<Self> {
        let _ = (hidden, intermediate);
        Ok(Mlp {
            gate_proj: w.linear(&format!("{prefix}.gate_proj"))?,
            up_proj: w.linear(&format!("{prefix}.up_proj"))?,
            down_proj: w.linear(&format!("{prefix}.down_proj"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&ops::multiply(&gate, &up)?)
    }
}

enum Mixer {
    Linear(GatedDeltaNet),
    Attention(Attention),
}

enum FeedForward {
    Dense(Mlp),
    Moe(SparseMoeBlock),
}

struct Block {
    mixer: Mixer,
    ff: FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Block {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &Qwen35Config, layer_idx: i32) -> Result<Self> {
        let is_linear = cfg.is_linear_layer(layer_idx);
        let mixer = if is_linear {
            let gd_cfg = GatedDeltaConfig {
                num_v_heads: cfg.linear_num_value_heads,
                num_k_heads: cfg.linear_num_key_heads,
                head_k_dim: cfg.linear_key_head_dim,
                head_v_dim: cfg.linear_value_head_dim,
                conv_kernel_size: cfg.linear_conv_kernel_dim,
                rms_norm_eps: cfg.rms_norm_eps,
            };
            Mixer::Linear(GatedDeltaNet::load(
                w,
                &format!("{prefix}.linear_attn"),
                &gd_cfg,
            )?)
        } else {
            Mixer::Attention(Attention::load(w, prefix, cfg)?)
        };

        let use_moe = cfg.num_experts > 0 && (layer_idx + 1) % cfg.decoder_sparse_step == 0;
        let ff = if use_moe {
            FeedForward::Moe(SparseMoeBlock::load(w, &format!("{prefix}.mlp"), cfg)?)
        } else {
            FeedForward::Dense(Mlp::load(
                w,
                &format!("{prefix}.mlp"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?)
        };

        Ok(Block {
            mixer,
            ff,
            input_layernorm: w.rms_norm(&format!("{prefix}.input_layernorm"), cfg.rms_norm_eps)?,
            post_attention_layernorm: w.rms_norm(
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask, cache: &mut LayerCache) -> Result<Array> {
        let normed = self.input_layernorm.forward(x)?;
        let r = match &self.mixer {
            Mixer::Linear(m) => m.forward(&normed, cache.as_gated_delta()?)?,
            Mixer::Attention(m) => m.forward(&normed, mask, cache.as_attention()?)?,
        };
        let h = ops::add(x, &r)?;
        let ff_in = self.post_attention_layernorm.forward(&h)?;
        let ff_out = match &self.ff {
            FeedForward::Dense(m) => m.forward(&ff_in)?,
            FeedForward::Moe(m) => m.forward(&ff_in)?,
        };
        ops::add(&h, &ff_out)
    }

    fn is_linear(&self) -> bool {
        matches!(self.mixer, Mixer::Linear(_))
    }
}

/// Optional image support (`Some` only on checkpoints whose `config.json`
/// carries a `vision_config` sub-dict with a matching `vision_tower.*`
/// weight set).
struct VisionSupport {
    tower: Qwen35VisionTower,
    image_token_id: i32,
    vision_start_token_id: i32,
    vision_end_token_id: i32,
    video_token_id: i32,
}

pub struct Qwen35Model {
    pub config: Qwen35Config,
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    vision: Option<VisionSupport>,
}

impl Qwen35Model {
    pub fn load(mut weights: WeightMap, config_json: &Value) -> Result<Self> {
        let cfg = Qwen35Config::from_json(config_json)?;

        let embed_tokens = weights.embedding("language_model.model.embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Block::load(
                &mut weights,
                &format!("language_model.model.layers.{i}"),
                &cfg,
                i,
            )?);
        }
        let norm = weights.rms_norm("language_model.model.norm", cfg.rms_norm_eps)?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(weights.linear("language_model.lm_head")?)
        };

        let vision = Self::load_vision(&mut weights, config_json)?;

        Ok(Qwen35Model {
            config: cfg,
            embed_tokens,
            layers,
            norm,
            lm_head,
            vision,
        })
    }

    /// Build the vision tower iff `config.json` carries a `vision_config`
    /// sub-dict AND the checkpoint actually shipped the matching
    /// `vision_tower.*` weights (a checkpoint could declare `vision_config`
    /// while being distributed text-only); returns `None` otherwise so
    /// text-only checkpoints load exactly as before.
    fn load_vision(weights: &mut WeightMap, config_json: &Value) -> Result<Option<VisionSupport>> {
        let Some(vision_cfg_json) = config_json.get("vision_config") else {
            return Ok(None);
        };
        if !vision::has_vision_weights(weights) {
            return Ok(None);
        }

        let vision_cfg = Qwen35VisionConfig::from_json(vision_cfg_json);
        let tower = Qwen35VisionTower::load(weights, vision_cfg)?;

        Ok(Some(VisionSupport {
            tower,
            image_token_id: get_i32(config_json, "image_token_id", 151655),
            vision_start_token_id: get_i32(config_json, "vision_start_token_id", 151652),
            vision_end_token_id: get_i32(config_json, "vision_end_token_id", 151653),
            video_token_id: get_i32(config_json, "video_token_id", 151656),
        }))
    }

    /// Whether this checkpoint's vision tower was loaded (i.e. it can
    /// accept image input via [`Qwen35Model::forward_with_media`]).
    pub fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    /// `(patch_size, max_soft_tokens, spatial_merge_size)` for
    /// [`crate::media::image::preprocess_image_bytes`], or `None` if this
    /// checkpoint has no vision support.
    pub fn image_processing_params(&self) -> Option<(i32, i32, i32)> {
        self.vision.as_ref().map(|v| {
            let cfg = v.tower.config();
            (cfg.patch_size, 1280, cfg.spatial_merge_size)
        })
    }

    /// `(image_token_id, vision_start_token_id, vision_end_token_id)`, or
    /// `None` if this checkpoint has no vision support. Reuses the
    /// `(image_token_id, boi_token_id, eoi_token_id)` shape every
    /// multimodal architecture in this crate exposes: Qwen3.5's
    /// `vision_start`/`vision_end` tokens play the same "wrap the expanded
    /// placeholder span" role as Gemma4's `boi`/`eoi`.
    pub fn image_token_ids(&self) -> Option<(u32, u32, u32)> {
        self.vision.as_ref().map(|v| {
            (
                v.image_token_id as u32,
                v.vision_start_token_id as u32,
                v.vision_end_token_id as u32,
            )
        })
    }

    /// `video_token_id` from `config.json`, or `None` when the checkpoint
    /// has no vision tower (video reuses the vision path per frame).
    pub fn video_token_id(&self) -> Option<u32> {
        self.vision.as_ref().map(|v| v.video_token_id as u32)
    }

    pub fn new_caches(&self) -> Vec<LayerCache> {
        self.layers
            .iter()
            .map(|l| {
                if l.is_linear() {
                    LayerCache::GatedDelta(GatedDeltaCache::new())
                } else {
                    LayerCache::new_attention()
                }
            })
            .collect()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
        let h = self.embed_tokens.forward(input_ids)?;
        self.forward_from_embeds(input_ids, h, caches)
    }

    /// Same as [`Qwen35Model::forward`], but splices `images`' projected
    /// vision-tower features into the text embedding stream at
    /// `image_token_id` placeholder positions before running the decoder
    /// stack (order preserving across multiple images; each image's
    /// soft-token count must exactly match the number of `image_token_id`
    /// placeholders `input_ids` holds for it, produced by expanding the
    /// chat template's single-placeholder-per-image convention beforehand -
    /// see `crate::generate::Session::encode_chat_with_media`).
    pub fn forward_with_images(
        &self,
        input_ids: &Array,
        images: &[ProcessedImage],
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        self.forward_with_media(input_ids, images, caches)
    }

    /// Same as [`Qwen35Model::forward_with_images`]; Qwen3.5-VL has no
    /// audio tower, so this simply ignores audio (kept as a distinct name
    /// to mirror the generic [`crate::models::Model::forward_with_media`]
    /// dispatch signature other multimodal architectures use).
    pub fn forward_with_media(
        &self,
        input_ids: &Array,
        images: &[ProcessedImage],
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;

        if !images.is_empty() {
            let vision = self.vision.as_ref().ok_or_else(|| {
                Error::Model("qwen3.5: model has no vision support (no vision_config)".into())
            })?;
            let mut all_features = Vec::with_capacity(images.len());
            for image in images {
                let features =
                    vision
                        .tower
                        .forward(&image.pixel_values, image.patch_h, image.patch_w)?;
                all_features.push(features);
            }
            h = splice_media_features(&h, input_ids, all_features, vision.image_token_id, "image")?;
        }

        self.forward_from_embeds(input_ids, h, caches)
    }

    fn forward_from_embeds(
        &self,
        input_ids: &Array,
        mut h: Array,
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        let seq_len = input_ids.dim(1);
        let mask = attention_mask_for(seq_len);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask, cache)?;
        }
        h = self.norm.forward(&h)?;

        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }
}

/// Normalize checkpoint weight keys. Qwen3.5/3.6 checkpoints (dense or MoE,
/// with or without a vision tower) always ship the language model under a
/// `language_model.*` prefix and, when present, the vision tower under a
/// bare `vision_tower.*` prefix (an `optiq_vision.safetensors` sidecar on
/// OptiQ checkpoints); keep both, drop everything else (MTP
/// speculative-decoding heads), and fold fused MoE expert weights into the
/// `switch_mlp.*` layout `SparseMoeBlock` expects.
pub fn sanitize(weights: &mut WeightMap, num_hidden_layers: i32, num_experts: i32) {
    weights.rename_keys(|k| {
        if k.starts_with("vision_tower.") {
            Some(k.to_string())
        } else if k.starts_with("language_model.") && !k.contains("mtp.") {
            Some(k.to_string())
        } else {
            None
        }
    });

    if num_experts <= 0 {
        return;
    }

    for l in 0..num_hidden_layers {
        let prefix = format!("language_model.model.layers.{l}.mlp");
        // Fused gate_up_proj (stacked experts): [E, 2*I, H] -> split in half.
        if let Some(gate_up) = weights.take_optional(&format!("{prefix}.experts.gate_up_proj")) {
            let mid = gate_up.dim(-2) / 2;
            let shape = gate_up.shape();
            if let Ok(gate) = ops::slice(&gate_up, &[0, 0, 0], &[shape[0], mid, shape[2]]) {
                weights.insert(format!("{prefix}.switch_mlp.gate_proj.weight"), gate);
            }
            if let Ok(up) = ops::slice(&gate_up, &[0, mid, 0], &[shape[0], shape[1], shape[2]]) {
                weights.insert(format!("{prefix}.switch_mlp.up_proj.weight"), up);
            }
        }
        if let Some(down) = weights.take_optional(&format!("{prefix}.experts.down_proj")) {
            weights.insert(format!("{prefix}.switch_mlp.down_proj.weight"), down);
        }
        // Per-expert separate weights: stack into [E, out, in].
        for name in ["gate_proj", "up_proj", "down_proj"] {
            if weights.contains(&format!("{prefix}.experts.0.{name}.weight")) {
                let mut expert_weights = Vec::new();
                let mut e = 0;
                while let Some(w) =
                    weights.take_optional(&format!("{prefix}.experts.{e}.{name}.weight"))
                {
                    expert_weights.push(w);
                    e += 1;
                }
                let refs: Vec<&Array> = expert_weights.iter().collect();
                if let Ok(stacked) = ops::stack_axis(&refs, 0) {
                    weights.insert(format!("{prefix}.switch_mlp.{name}.weight"), stacked);
                }
            }
        }
    }
}

pub fn parse_quantization(config_json: &Value) -> Result<Quantization> {
    Quantization::from_config(config_json)
}

pub fn model_error(model_type: &str) -> Error {
    Error::Model(format!("unsupported qwen3.5 variant '{model_type}'"))
}
