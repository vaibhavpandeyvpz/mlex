//! NemotronH: a hybrid architecture interleaving Mamba2 SSM layers, plain
//! GQA self-attention layers, and gate-free (`relu2`) MLP layers, per a
//! `hybrid_override_pattern` string (`M` / `*` / `-`, MoE `E` not covered
//! by the currently supported checkpoints).

use serde_json::Value;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::nn::{Embedding, Linear, RmsNorm, WeightMap};
use crate::ops::{self, AttentionMask};
use crate::quant::Quantization;

use super::base::{attention_mask_for, merge_heads, split_heads};
use super::cache::{GatedDeltaCache, KvCache, LayerCache};
use super::config::{get_bool, get_f32, get_i32, get_str, require_i32};
use super::mamba2::{Mamba2Config, Mamba2Mixer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    Mamba,
    Attention,
    Mlp,
}

#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub hidden_size: i32,
    pub vocab_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub attention_bias: bool,
    pub mamba_num_heads: i32,
    pub mamba_head_dim: i32,
    pub mamba_proj_bias: bool,
    pub ssm_state_size: i32,
    pub conv_kernel: i32,
    pub n_groups: i32,
    pub mlp_bias: bool,
    pub layer_norm_eps: f32,
    pub use_conv_bias: bool,
    pub time_step_min: f32,
    pub time_step_max: f32,
    pub tie_word_embeddings: bool,
    pub layers: Vec<BlockType>,
}

impl NemotronConfig {
    pub fn from_json(cfg: &Value) -> Result<Self> {
        let pattern = get_str(cfg, "hybrid_override_pattern")
            .ok_or_else(|| Error::Config("nemotron_h: missing hybrid_override_pattern".into()))?;
        let layers = pattern
            .chars()
            .map(|c| match c {
                'M' => Ok(BlockType::Mamba),
                '*' => Ok(BlockType::Attention),
                '-' => Ok(BlockType::Mlp),
                other => Err(Error::Config(format!(
                    "nemotron_h: unsupported layer type '{other}' in hybrid_override_pattern (MoE 'E' not yet implemented)"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;

        let hidden_size = require_i32(cfg, "hidden_size")?;
        let num_attention_heads = require_i32(cfg, "num_attention_heads")?;
        let layer_norm_eps = get_f32(
            cfg,
            "layer_norm_epsilon",
            get_f32(cfg, "rms_norm_eps", 1e-5),
        );

        Ok(NemotronConfig {
            hidden_size,
            vocab_size: require_i32(cfg, "vocab_size")?,
            intermediate_size: require_i32(cfg, "intermediate_size")?,
            num_attention_heads,
            num_key_value_heads: get_i32(cfg, "num_key_value_heads", num_attention_heads),
            head_dim: get_i32(cfg, "head_dim", hidden_size / num_attention_heads),
            attention_bias: get_bool(cfg, "attention_bias", false),
            mamba_num_heads: require_i32(cfg, "mamba_num_heads")?,
            mamba_head_dim: require_i32(cfg, "mamba_head_dim")?,
            mamba_proj_bias: get_bool(cfg, "mamba_proj_bias", false),
            ssm_state_size: require_i32(cfg, "ssm_state_size")?,
            conv_kernel: get_i32(cfg, "conv_kernel", 4),
            n_groups: get_i32(cfg, "n_groups", 1),
            mlp_bias: get_bool(cfg, "mlp_bias", false),
            layer_norm_eps,
            use_conv_bias: get_bool(cfg, "use_conv_bias", true),
            // mlx-lm's `ModelArgs.time_step_limit` (the actual dt-clipping
            // bound used by `ssm_update`) is a distinct field from the
            // `time_step_min`/`time_step_max`/`time_step_floor` values that
            // show up in NemotronH `config.json` files; those are HF
            // training-time hyperparameters the mlx-lm dataclass doesn't
            // even parse, so unless a checkpoint sets `time_step_limit`
            // explicitly, no clipping is applied beyond softplus's implicit
            // non-negativity.
            time_step_min: cfg
                .get("time_step_limit")
                .and_then(|v| v.get(0))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(0.0),
            time_step_max: cfg
                .get("time_step_limit")
                .and_then(|v| v.get(1))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(f32::INFINITY),
            tie_word_embeddings: get_bool(cfg, "tie_word_embeddings", false),
            layers,
        })
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: i32,
    n_kv_heads: i32,
    scale: f32,
}

impl Attention {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &NemotronConfig) -> Result<Self> {
        let m = format!("{prefix}.mixer");
        Ok(Attention {
            q_proj: w.linear(&format!("{m}.q_proj"))?,
            k_proj: w.linear(&format!("{m}.k_proj"))?,
            v_proj: w.linear(&format!("{m}.v_proj"))?,
            o_proj: w.linear(&format!("{m}.o_proj"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask, cache: &mut KvCache) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = split_heads(&self.q_proj.forward(x)?, b, l, self.n_heads)?;
        let k = split_heads(&self.k_proj.forward(x)?, b, l, self.n_kv_heads)?;
        let v = split_heads(&self.v_proj.forward(x)?, b, l, self.n_kv_heads)?;

        // NemotronH attention layers carry no positional embedding.
        let (k, v) = cache.update_and_fetch(k, v)?;
        let out = ops::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;
        let out = merge_heads(&out, b, l)?;
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(w: &mut WeightMap, prefix: &str) -> Result<Self> {
        let m = format!("{prefix}.mixer");
        Ok(Mlp {
            up_proj: w.linear(&format!("{m}.up_proj"))?,
            down_proj: w.linear(&format!("{m}.down_proj"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.down_proj
            .forward(&ops::relu2(&self.up_proj.forward(x)?)?)
    }
}

enum Mixer {
    Mamba(Mamba2Mixer),
    Attention(Attention),
    Mlp(Mlp),
}

struct Block {
    mixer: Mixer,
    norm: RmsNorm,
}

impl Block {
    fn load(
        w: &mut WeightMap,
        prefix: &str,
        cfg: &NemotronConfig,
        block_type: BlockType,
    ) -> Result<Self> {
        let mixer = match block_type {
            BlockType::Mamba => {
                let mamba_cfg = Mamba2Config {
                    num_heads: cfg.mamba_num_heads,
                    head_dim: cfg.mamba_head_dim,
                    n_groups: cfg.n_groups,
                    state_size: cfg.ssm_state_size,
                    conv_kernel: cfg.conv_kernel,
                    proj_bias: cfg.mamba_proj_bias,
                    conv_bias: cfg.use_conv_bias,
                    norm_eps: cfg.layer_norm_eps,
                    time_step_min: cfg.time_step_min,
                    time_step_max: cfg.time_step_max,
                };
                Mixer::Mamba(Mamba2Mixer::load(
                    w,
                    &format!("{prefix}.mixer"),
                    &mamba_cfg,
                )?)
            }
            BlockType::Attention => Mixer::Attention(Attention::load(w, prefix, cfg)?),
            BlockType::Mlp => Mixer::Mlp(Mlp::load(w, prefix)?),
        };
        Ok(Block {
            mixer,
            norm: w.rms_norm(&format!("{prefix}.norm"), cfg.layer_norm_eps)?,
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask, cache: &mut LayerCache) -> Result<Array> {
        let normed = self.norm.forward(x)?;
        let out = match &self.mixer {
            Mixer::Mamba(m) => m.forward(&normed, cache.as_gated_delta()?)?,
            Mixer::Attention(m) => m.forward(&normed, mask, cache.as_attention()?)?,
            Mixer::Mlp(m) => m.forward(&normed)?,
        };
        ops::add(x, &out)
    }

    fn cache_kind(&self) -> LayerCache {
        match self.mixer {
            Mixer::Mamba(_) => LayerCache::GatedDelta(GatedDeltaCache::new()),
            Mixer::Attention(_) => LayerCache::new_attention(),
            Mixer::Mlp(_) => LayerCache::new_attention(),
        }
    }
}

pub struct NemotronModel {
    pub config: NemotronConfig,
    embeddings: Embedding,
    layers: Vec<Block>,
    norm_f: RmsNorm,
    lm_head: Option<Linear>,
}

impl NemotronModel {
    pub fn load(mut weights: WeightMap, config_json: &Value) -> Result<Self> {
        let cfg = NemotronConfig::from_json(config_json)?;

        let embeddings = weights.embedding("backbone.embeddings")?;
        let mut layers = Vec::with_capacity(cfg.layers.len());
        for (i, block_type) in cfg.layers.iter().enumerate() {
            layers.push(Block::load(
                &mut weights,
                &format!("backbone.layers.{i}"),
                &cfg,
                *block_type,
            )?);
        }
        let norm_f = weights.rms_norm("backbone.norm_f", cfg.layer_norm_eps)?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(weights.linear("lm_head")?)
        };

        Ok(NemotronModel {
            config: cfg,
            embeddings,
            layers,
            norm_f,
            lm_head,
        })
    }

    pub fn new_caches(&self) -> Vec<LayerCache> {
        self.layers.iter().map(Block::cache_kind).collect()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Debug helper: run the forward pass but return `(mean(abs), std)` of
    /// the hidden state after every layer, for comparison against the
    /// Python reference during numerical debugging.
    pub fn debug_layer_stats(&self, input_ids: &Array) -> Result<Vec<(f32, f32)>> {
        let mut h = self.embeddings.forward(input_ids)?;
        let seq_len = input_ids.dim(1);
        let mask = attention_mask_for(seq_len);
        let mut caches = self.new_caches();
        let mut stats = Vec::new();
        {
            let v = h.to_vec_f32()?;
            let n = v.len() as f32;
            let mean_abs = v.iter().map(|x| x.abs()).sum::<f32>() / n;
            stats.push((mean_abs, -1.0f32));
        }
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask, cache)?;
            let v = h.to_vec_f32()?;
            let n = v.len() as f32;
            let mean_abs = v.iter().map(|x| x.abs()).sum::<f32>() / n;
            let mean = v.iter().sum::<f32>() / n;
            let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            stats.push((mean_abs, var.sqrt()));
        }
        Ok(stats)
    }

    pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
        let mut h = self.embeddings.forward(input_ids)?;
        let seq_len = input_ids.dim(1);
        let mask = attention_mask_for(seq_len);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask, cache)?;
        }
        h = self.norm_f.forward(&h)?;

        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embeddings.as_linear(&h),
        }
    }
}

/// Drop MTP speculative-decoding weights; everything else in NemotronH
/// checkpoints is already at its final key (no vision tower, no prefix
/// remapping needed).
pub fn sanitize(weights: &mut WeightMap) {
    weights.rename_keys(|k| {
        if k.starts_with("mtp.") {
            None
        } else {
            Some(k.to_string())
        }
    });
}

pub fn parse_quantization(config_json: &Value) -> Result<Quantization> {
    Quantization::from_config(config_json)
}
