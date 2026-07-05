//! DharaAR architecture (`mlx-community/dhara-250m-OptiQ-4bit`): a
//! LLaMA3-style GQA transformer plus "Canon layer" causal depthwise
//! convolutions at up to four positions per block (A/B/C/D, from "Physics
//! of Language Models: Part 4.1"), post-RoPE per-head QK-norm (opposite
//! order from Qwen3's pre-RoPE `q_norm`/`k_norm`), and a final logit
//! softcap. See `modeling_dhara_ar.py` in the downloaded checkpoint for the
//! reference implementation this ports.
//!
//! Only the autoregressive decode path is implemented (matches
//! `Session::generate`/`generate_cached`); the model's
//! `generate_diffusion`/`generate_self_spec`
//! modes use a fundamentally different bidirectional block-causal decode
//! loop and are out of scope.

use serde_json::Value;

use crate::array::Array;
use crate::error::Result;
use crate::nn::{Embedding, Linear, RmsNorm, WeightMap};
use crate::ops::{self, AttentionMask};
use crate::quant::Quantization;

use super::base::{attention_mask_for, merge_heads, split_heads, RopeConfig};
use super::cache::{DharaCache, LayerCache};
use super::config::{get_bool, get_f32, get_i32, get_str, require_i32};

#[derive(Debug, Clone)]
pub struct DharaConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub use_qk_norm: bool,
    pub use_logit_softcap: bool,
    pub logit_softcap: f32,
    pub canon_set: String,
    pub canon_kernel: i32,
    pub canon_residual: bool,
    pub canon_activation: bool,
}

impl DharaConfig {
    pub fn from_json(cfg: &Value) -> Result<Self> {
        let hidden_size = require_i32(cfg, "hidden_size")?;
        let num_attention_heads = require_i32(cfg, "num_attention_heads")?;
        let head_dim = get_i32(cfg, "head_dim", hidden_size / num_attention_heads);
        Ok(DharaConfig {
            hidden_size,
            num_hidden_layers: require_i32(cfg, "num_hidden_layers")?,
            intermediate_size: require_i32(cfg, "intermediate_size")?,
            num_attention_heads,
            num_key_value_heads: get_i32(cfg, "num_key_value_heads", num_attention_heads),
            head_dim,
            rms_norm_eps: get_f32(cfg, "rms_norm_eps", 1e-6),
            vocab_size: require_i32(cfg, "vocab_size")?,
            rope_theta: get_f32(cfg, "rope_theta", 100_000.0),
            tie_word_embeddings: get_bool(cfg, "tie_word_embeddings", true),
            use_qk_norm: get_bool(cfg, "use_qk_norm", true),
            use_logit_softcap: get_bool(cfg, "use_logit_softcap", true),
            logit_softcap: get_f32(cfg, "logit_softcap", 30.0),
            canon_set: get_str(cfg, "canon_set").unwrap_or("ABCD").to_string(),
            canon_kernel: get_i32(cfg, "canon_kernel", 4),
            canon_residual: get_bool(cfg, "canon_residual", true),
            canon_activation: get_bool(cfg, "canon_activation", false),
        })
    }
}

/// Causal depthwise 1D conv ("Canon layer"): the trailing `kernel-1` inputs
/// are kept in `DharaCache` so incremental decode matches a full forward
/// exactly (no zero-padding on cached steps).
struct CanonLayer {
    weight: Array,
    dim: i32,
    kernel: i32,
    residual: bool,
    activation: bool,
}

impl CanonLayer {
    fn load(w: &mut WeightMap, path: &str, dim: i32, cfg: &DharaConfig) -> Result<Self> {
        Ok(CanonLayer {
            weight: w.take(&format!("{path}.conv.weight"))?,
            dim,
            kernel: cfg.canon_kernel,
            residual: cfg.canon_residual,
            activation: cfg.canon_activation,
        })
    }

    fn forward(&self, x: &Array, state: &mut Option<Array>) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let n_keep = self.kernel - 1;

        let prev = state
            .take()
            .unwrap_or(ops::zeros(&[b, n_keep, self.dim], x.dtype())?);
        let conv_input = ops::concatenate(&[&prev, x], 1)?;
        let total_len = conv_input.dim(1);
        *state = Some(ops::contiguous(&ops::slice(
            &conv_input,
            &[0, total_len - n_keep, 0],
            &[b, total_len, self.dim],
        )?)?);

        let mut out = ops::conv1d(&conv_input, &self.weight, 1, 0, 1, self.dim)?;
        if self.activation {
            out = ops::silu(&out)?;
        }
        if self.residual {
            out = ops::add(x, &out)?;
        }
        Ok(out)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    canon_b_q: Option<CanonLayer>,
    canon_b_k: Option<CanonLayer>,
    canon_b_v: Option<CanonLayer>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    rope: RopeConfig,
    n_heads: i32,
    n_kv_heads: i32,
    scale: f32,
}

impl Attention {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &DharaConfig) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let has_canon_b = cfg.canon_set.contains('B');
        Ok(Attention {
            q_proj: w.linear(&format!("{attn}.q_proj"))?,
            k_proj: w.linear(&format!("{attn}.k_proj"))?,
            v_proj: w.linear(&format!("{attn}.v_proj"))?,
            o_proj: w.linear(&format!("{attn}.o_proj"))?,
            canon_b_q: has_canon_b
                .then(|| CanonLayer::load(w, &format!("{attn}.canon_b_q"), q_dim, cfg))
                .transpose()?,
            canon_b_k: has_canon_b
                .then(|| CanonLayer::load(w, &format!("{attn}.canon_b_k"), kv_dim, cfg))
                .transpose()?,
            canon_b_v: has_canon_b
                .then(|| CanonLayer::load(w, &format!("{attn}.canon_b_v"), kv_dim, cfg))
                .transpose()?,
            q_norm: cfg
                .use_qk_norm
                .then(|| w.rms_norm(&format!("{attn}.q_norm"), cfg.rms_norm_eps))
                .transpose()?,
            k_norm: cfg
                .use_qk_norm
                .then(|| w.rms_norm(&format!("{attn}.k_norm"), cfg.rms_norm_eps))
                .transpose()?,
            rope: RopeConfig::new(cfg.head_dim, cfg.rope_theta),
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask, cache: &mut DharaCache) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let mut v = self.v_proj.forward(x)?;

        if let Some(canon) = &self.canon_b_q {
            q = canon.forward(&q, &mut cache.canon_b_q)?;
        }
        if let Some(canon) = &self.canon_b_k {
            k = canon.forward(&k, &mut cache.canon_b_k)?;
        }
        if let Some(canon) = &self.canon_b_v {
            v = canon.forward(&v, &mut cache.canon_b_v)?;
        }

        let q = split_heads(&q, b, l, self.n_heads)?;
        let k = split_heads(&k, b, l, self.n_kv_heads)?;
        let v = split_heads(&v, b, l, self.n_kv_heads)?;

        let offset = cache.attn.offset();
        let mut q = self.rope.apply(&q, offset)?;
        let mut k = self.rope.apply(&k, offset)?;

        // Post-RoPE QK-norm (opposite order from Qwen3).
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }

        let (k, v) = cache.attn.update_and_fetch(k, v)?;

        let out = ops::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;
        let out = merge_heads(&out, b, l)?;
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    canon_d: Option<CanonLayer>,
}

impl Mlp {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &DharaConfig) -> Result<Self> {
        let mlp = format!("{prefix}.mlp");
        let has_canon_d = cfg.canon_set.contains('D');
        Ok(Mlp {
            gate_proj: w.linear(&format!("{mlp}.gate_proj"))?,
            up_proj: w.linear(&format!("{mlp}.up_proj"))?,
            down_proj: w.linear(&format!("{mlp}.down_proj"))?,
            canon_d: has_canon_d
                .then(|| CanonLayer::load(w, &format!("{mlp}.canon_d"), cfg.intermediate_size, cfg))
                .transpose()?,
        })
    }

    fn forward(&self, x: &Array, state: &mut Option<Array>) -> Result<Array> {
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let mut inter = ops::multiply(&gate, &up)?;
        if let Some(canon) = &self.canon_d {
            inter = canon.forward(&inter, state)?;
        }
        self.down_proj.forward(&inter)
    }
}

struct Block {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    canon_a: Option<CanonLayer>,
    canon_c: Option<CanonLayer>,
}

impl Block {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &DharaConfig) -> Result<Self> {
        Ok(Block {
            self_attn: Attention::load(w, prefix, cfg)?,
            mlp: Mlp::load(w, prefix, cfg)?,
            input_layernorm: w.rms_norm(&format!("{prefix}.input_layernorm"), cfg.rms_norm_eps)?,
            post_attention_layernorm: w.rms_norm(
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_norm_eps,
            )?,
            canon_a: cfg
                .canon_set
                .contains('A')
                .then(|| CanonLayer::load(w, &format!("{prefix}.canon_a"), cfg.hidden_size, cfg))
                .transpose()?,
            canon_c: cfg
                .canon_set
                .contains('C')
                .then(|| CanonLayer::load(w, &format!("{prefix}.canon_c"), cfg.hidden_size, cfg))
                .transpose()?,
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask, cache: &mut DharaCache) -> Result<Array> {
        let mut normed = self.input_layernorm.forward(x)?;
        if let Some(canon) = &self.canon_a {
            normed = canon.forward(&normed, &mut cache.canon_a)?;
        }
        let h = ops::add(x, &self.self_attn.forward(&normed, mask, cache)?)?;

        let mut normed2 = self.post_attention_layernorm.forward(&h)?;
        if let Some(canon) = &self.canon_c {
            normed2 = canon.forward(&normed2, &mut cache.canon_c)?;
        }
        let out = ops::add(&h, &self.mlp.forward(&normed2, &mut cache.canon_d)?)?;
        Ok(out)
    }
}

/// DharaAR causal language model.
pub struct DharaModel {
    pub config: DharaConfig,
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
}

impl DharaModel {
    pub fn load(mut weights: WeightMap, config_json: &Value) -> Result<Self> {
        let cfg = DharaConfig::from_json(config_json)?;

        let embed_tokens = weights.embedding("model.embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Block::load(
                &mut weights,
                &format!("model.layers.{i}"),
                &cfg,
            )?);
        }
        let norm = weights.rms_norm("model.norm", cfg.rms_norm_eps)?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(weights.linear("lm_head")?)
        };

        Ok(DharaModel {
            config: cfg,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn new_caches(&self) -> Vec<LayerCache> {
        (0..self.num_layers())
            .map(|_| LayerCache::new_dhara())
            .collect()
    }

    pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        let seq_len = input_ids.dim(1);
        let mask = attention_mask_for(seq_len);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask, cache.as_dhara()?)?;
        }
        h = self.norm.forward(&h)?;

        let mut logits = match &self.lm_head {
            Some(head) => head.forward(&h)?,
            None => self.embed_tokens.as_linear(&h)?,
        };

        if self.config.use_logit_softcap && self.config.logit_softcap > 0.0 {
            let cap = self.config.logit_softcap;
            logits = ops::scale_by(&ops::tanh(&ops::scale_by(&logits, 1.0 / cap)?)?, cap)?;
        }

        Ok(logits)
    }
}

/// Drop non-weight buffers (`rotary_emb.inv_freq` is recomputed, not
/// loaded) and the untied `lm_head.weight` when embeddings are tied.
pub fn sanitize(weights: &mut WeightMap, tie_word_embeddings: bool) {
    weights.rename_keys(|k| {
        if k.ends_with("rotary_emb.inv_freq") {
            None
        } else if tie_word_embeddings && k == "lm_head.weight" {
            None
        } else {
            Some(k.to_string())
        }
    });
}

pub fn parse_quantization(config_json: &Value) -> Result<Quantization> {
    Quantization::from_config(config_json)
}
