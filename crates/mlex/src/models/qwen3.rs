//! Qwen3 (dense, first-generation) architecture: GQA with per-head q/k RMSNorm.

use serde_json::Value;

use crate::array::Array;
use crate::error::Result;
use crate::nn::{Embedding, Linear, RmsNorm, WeightMap};
use crate::ops::{self, AttentionMask};
use crate::quant::Quantization;

use super::base::{attention_mask_for, merge_heads, split_heads, RopeConfig};
use super::cache::{KvCache, LayerCache};
use super::config::{get_bool, get_f32, get_i32, require_i32};

#[derive(Debug, Clone)]
pub struct Qwen3Config {
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
}

impl Qwen3Config {
    pub fn from_json(cfg: &Value) -> Result<Self> {
        let hidden_size = require_i32(cfg, "hidden_size")?;
        let num_attention_heads = require_i32(cfg, "num_attention_heads")?;
        let head_dim = get_i32(cfg, "head_dim", hidden_size / num_attention_heads);
        Ok(Qwen3Config {
            hidden_size,
            num_hidden_layers: require_i32(cfg, "num_hidden_layers")?,
            intermediate_size: require_i32(cfg, "intermediate_size")?,
            num_attention_heads,
            num_key_value_heads: get_i32(cfg, "num_key_value_heads", num_attention_heads),
            head_dim,
            rms_norm_eps: get_f32(cfg, "rms_norm_eps", 1e-6),
            vocab_size: require_i32(cfg, "vocab_size")?,
            rope_theta: get_f32(cfg, "rope_theta", 1_000_000.0),
            tie_word_embeddings: get_bool(cfg, "tie_word_embeddings", true),
        })
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
    fn load(w: &mut WeightMap, prefix: &str, cfg: &Qwen3Config) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        Ok(Attention {
            q_proj: w.linear(&format!("{attn}.q_proj"))?,
            k_proj: w.linear(&format!("{attn}.k_proj"))?,
            v_proj: w.linear(&format!("{attn}.v_proj"))?,
            o_proj: w.linear(&format!("{attn}.o_proj"))?,
            q_norm: w.rms_norm(&format!("{attn}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: w.rms_norm(&format!("{attn}.k_norm"), cfg.rms_norm_eps)?,
            rope: RopeConfig::new(cfg.head_dim, cfg.rope_theta),
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask, cache: &mut KvCache) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = split_heads(&q, b, l, self.n_heads)?;
        let k = split_heads(&k, b, l, self.n_kv_heads)?;
        let v = split_heads(&v, b, l, self.n_kv_heads)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let offset = cache.offset();
        let q = self.rope.apply(&q, offset)?;
        let k = self.rope.apply(&k, offset)?;
        let (k, v) = cache.update_and_fetch(k, v)?;

        let out = ops::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;
        let out = merge_heads(&out, b, l)?;
        let _ = self.head_dim;
        self.o_proj.forward(&out)
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
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&ops::multiply(&gate, &up)?)
    }
}

struct Block {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Block {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &Qwen3Config) -> Result<Self> {
        Ok(Block {
            self_attn: Attention::load(w, prefix, cfg)?,
            mlp: Mlp::load(w, prefix)?,
            input_layernorm: w.rms_norm(&format!("{prefix}.input_layernorm"), cfg.rms_norm_eps)?,
            post_attention_layernorm: w.rms_norm(
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask, cache: &mut KvCache) -> Result<Array> {
        let h = ops::add(
            x,
            &self
                .self_attn
                .forward(&self.input_layernorm.forward(x)?, mask, cache)?,
        )?;
        let out = ops::add(
            &h,
            &self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&h)?)?,
        )?;
        Ok(out)
    }
}

/// Qwen3 causal language model.
pub struct Qwen3Model {
    pub config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
}

impl Qwen3Model {
    pub fn load(mut weights: WeightMap, config_json: &Value) -> Result<Self> {
        let cfg = Qwen3Config::from_json(config_json)?;

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

        Ok(Qwen3Model {
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
            .map(|_| LayerCache::new_attention())
            .collect()
    }

    /// Run one forward pass. `input_ids` has shape `[B, L]`.
    pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        let seq_len = input_ids.dim(1);
        let mask = attention_mask_for(seq_len);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask, cache.as_attention()?)?;
        }
        h = self.norm.forward(&h)?;

        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embed_tokens.as_linear(&h),
        }
    }
}

/// Weight-key normalization applied before loading (drop the untied
/// `lm_head.weight` from checkpoints saved with `tie_word_embeddings`, and
/// strip the top-level `model.` prefix mismatch is handled by the callers).
pub fn sanitize(weights: &mut WeightMap, tie_word_embeddings: bool) {
    if tie_word_embeddings {
        weights.rename_keys(|k| {
            if k == "lm_head.weight" {
                None
            } else {
                Some(k.to_string())
            }
        });
    }
}

pub fn parse_quantization(config_json: &Value) -> Result<Quantization> {
    Quantization::from_config(config_json)
}
