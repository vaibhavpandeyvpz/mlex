//! Gemma4 SigLIP-style ("dense", `vision_config.model_type ==
//! "gemma4_vision"`) vision tower + multimodal embedder.
//!
//! Ported from the reference implementation's
//! `gemma4/{vision.rs,vision_config.rs,vision_rope.rs}`, adapted to our
//! `Array`/`ops`/`nn` idioms and simplified for the single-image (no
//! cross-image batching, no padding) inference path this crate needs: we
//! run the encoder over exactly the real patch grid (no padded canvas), so
//! the bidirectional attention needs no mask (`AttentionMask::None`) and
//! the reference's padding-aware "average pool by absolute patch position"
//! machinery collapses to a plain grid-reshape average pool, which is
//! numerically identical for the unpadded/unbatched case.
//!
//! Weight key layout (matching the checkpoint verbatim, no renaming):
//!   `vision_tower.patch_embedder.{input_proj.weight, position_embedding_table}`
//!   `vision_tower.encoder.layers.{i}.{input_layernorm,...}.weight`
//!   `vision_tower.encoder.layers.{i}.self_attn.{q,k,v,o}_proj.(linear.)weight`
//!   `vision_tower.encoder.layers.{i}.self_attn.{q,k}_norm.weight`
//!   `vision_tower.encoder.layers.{i}.mlp.{gate,up,down}_proj.(linear.)weight`
//!   `embed_vision.embedding_projection.*`

use serde_json::Value;

use crate::array::{Array, Dtype};
use crate::error::Result;
use crate::nn::{Linear, RmsNorm, WeightMap};
use crate::ops;

/// Parsed `vision_config` sub-dict of a Gemma4 multimodal checkpoint.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub patch_size: i32,
    pub position_embedding_size: i32,
    pub default_output_length: i32,
    pub pooling_kernel_size: i32,
    pub use_clipped_linears: bool,
    pub rope_theta: f32,
}

impl VisionConfig {
    pub fn from_json(cfg: &Value) -> Self {
        let gi = |k: &str, d: i32| {
            cfg.get(k)
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(d)
        };
        let gf = |k: &str, d: f32| {
            cfg.get(k)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(d)
        };
        let gb = |k: &str, d: bool| cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(d);
        let rope_theta = cfg
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(100.0) as f32;

        VisionConfig {
            hidden_size: gi("hidden_size", 768),
            intermediate_size: gi("intermediate_size", 3072),
            num_hidden_layers: gi("num_hidden_layers", 16),
            num_attention_heads: gi("num_attention_heads", 12),
            num_key_value_heads: gi("num_key_value_heads", 12),
            head_dim: gi("head_dim", 64),
            rms_norm_eps: gf("rms_norm_eps", 1e-6),
            patch_size: gi("patch_size", 16),
            position_embedding_size: gi("position_embedding_size", 10240),
            default_output_length: gi("default_output_length", 280),
            pooling_kernel_size: gi("pooling_kernel_size", 3),
            use_clipped_linears: gb("use_clipped_linears", false),
            rope_theta,
        }
    }
}

/// A dense `Linear` with optional pre/post activation clipping.
///
/// The Gemma4 vision tower's checkpoints (`use_clipped_linears: true`) wrap
/// every attention/MLP projection with static clip bounds recorded as
/// scalar tensors alongside a plain dense weight (`{path}.linear.weight` +
/// `{path}.{input,output}_{min,max}`), bounding activation range around an
/// otherwise-float path. Distinct from `quant::DynamicRangeParams`
/// (per-tensor int8 dequantization) - here the weight itself stays dense;
/// only the activations passing through are clamped.
pub(crate) struct ClippedLinear {
    linear: Linear,
    clip: Option<(f32, f32, f32, f32)>,
}

impl ClippedLinear {
    pub(crate) fn load(w: &mut WeightMap, path: &str, use_clipping: bool) -> Result<Self> {
        let linear = if w.contains(&format!("{path}.linear.weight")) {
            w.linear(&format!("{path}.linear"))?
        } else {
            w.linear(path)?
        };
        let clip = if use_clipping && w.contains(&format!("{path}.input_min")) {
            Some((
                w.take(&format!("{path}.input_min"))?.item_f32()?,
                w.take(&format!("{path}.input_max"))?.item_f32()?,
                w.take(&format!("{path}.output_min"))?.item_f32()?,
                w.take(&format!("{path}.output_max"))?.item_f32()?,
            ))
        } else {
            None
        };
        if std::env::var("MLEX_DEBUG_VISION").is_ok() {
            eprintln!("ClippedLinear {path}: clip={clip:?}");
        }
        Ok(ClippedLinear { linear, clip })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        match self.clip {
            Some((input_min, input_max, output_min, output_max)) => {
                let x = ops::clip(x, input_min, input_max)?;
                let out = self.linear.forward(&x)?;
                ops::clip(&out, output_min, output_max)
            }
            None => self.linear.forward(x),
        }
    }
}

fn rms_norm_no_scale(x: &Array, eps: f32) -> Result<Array> {
    ops::rms_norm(x, None, eps)
}

/// Slice `x`'s last axis to `[start, stop)`, keeping every other axis whole.
fn slice_last_axis(x: &Array, start: i32, stop: i32) -> Result<Array> {
    let shape = x.shape();
    let ndim = shape.len();
    let mut lo = vec![0; ndim];
    let mut hi = shape.clone();
    lo[ndim - 1] = start;
    hi[ndim - 1] = stop;
    ops::slice(x, &lo, &hi)
}

/// `[x1, x2] -> [-x2, x1]` (splits the last axis in half).
fn rotate_half(x: &Array) -> Result<Array> {
    let last = x.dim(-1);
    let half = last / 2;
    let x1 = slice_last_axis(x, 0, half)?;
    let x2 = slice_last_axis(x, half, last)?;
    let neg_x2 = ops::negative(&x2)?;
    ops::concatenate(&[&neg_x2, &x1], -1)
}

/// 2D RoPE over `[B, L, H, D]` inputs, splitting the head dimension into an
/// x-axis half and a y-axis half, each rotated by its own axis's position.
fn apply_vision_rope_2d(x: &Array, positions: &Array, base_frequency: f32) -> Result<Array> {
    let head_dim = x.dim(-1);
    let channels_per_dim = 2 * (head_dim / 4);
    let half_per_dim = channels_per_dim / 2;
    let in_dtype = x.dtype();

    let exps: Vec<f32> = (0..half_per_dim)
        .map(|i| (2 * i) as f32 / channels_per_dim as f32)
        .collect();
    let timescale: Vec<f32> = exps.iter().map(|&e| base_frequency.powf(e)).collect();
    let timescale_arr = Array::from_slice(&timescale, &[half_per_dim]);

    let mut parts = Vec::with_capacity(2);
    for d in 0..2i32 {
        let x_part = slice_last_axis(x, d * channels_per_dim, (d + 1) * channels_per_dim)?;
        let pos_d = slice_last_axis(positions, d, d + 1)?;
        let pos_f = ops::astype(&pos_d, Dtype::Float32)?;
        let sinusoid = ops::divide(&pos_f, &timescale_arr)?;
        let cos_d = ops::cos(&sinusoid)?;
        let sin_d = ops::sin(&sinusoid)?;
        let cos_d = ops::concatenate(&[&cos_d, &cos_d], -1)?;
        let sin_d = ops::concatenate(&[&sin_d, &sin_d], -1)?;
        let cos_d = ops::expand_dims(&ops::astype(&cos_d, in_dtype)?, 2)?;
        let sin_d = ops::expand_dims(&ops::astype(&sin_d, in_dtype)?, 2)?;
        let rotated = rotate_half(&x_part)?;
        let y = ops::add(
            &ops::multiply(&x_part, &cos_d)?,
            &ops::multiply(&rotated, &sin_d)?,
        )?;
        parts.push(y);
    }
    ops::concatenate(&[&parts[0], &parts[1]], -1)
}

struct VisionAttention {
    q_proj: ClippedLinear,
    k_proj: ClippedLinear,
    v_proj: ClippedLinear,
    o_proj: ClippedLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    v_norm_eps: f32,
    rope_theta: f32,
}

impl VisionAttention {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        let clip = cfg.use_clipped_linears;
        Ok(VisionAttention {
            q_proj: ClippedLinear::load(w, &format!("{attn}.q_proj"), clip)?,
            k_proj: ClippedLinear::load(w, &format!("{attn}.k_proj"), clip)?,
            v_proj: ClippedLinear::load(w, &format!("{attn}.v_proj"), clip)?,
            o_proj: ClippedLinear::load(w, &format!("{attn}.o_proj"), clip)?,
            q_norm: w.rms_norm(&format!("{attn}.q_norm"), cfg.rms_norm_eps)?,
            k_norm: w.rms_norm(&format!("{attn}.k_norm"), cfg.rms_norm_eps)?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            v_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
        })
    }

    fn forward(&self, x: &Array, positions: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?;
        let q = ops::reshape(&q, &[b, l, self.n_heads, self.head_dim])?;
        let q = self.q_norm.forward(&q)?;

        let k = self.k_proj.forward(x)?;
        let k = ops::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim])?;
        let k = self.k_norm.forward(&k)?;

        let v = self.v_proj.forward(x)?;
        let v = ops::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim])?;
        let v = rms_norm_no_scale(&v, self.v_norm_eps)?;

        let q = apply_vision_rope_2d(&q, positions, self.rope_theta)?;
        let k = apply_vision_rope_2d(&k, positions, self.rope_theta)?;

        let q = ops::transpose_axes(&q, &[0, 2, 1, 3])?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3])?;

        // Bidirectional, unpadded (single image, no batching) -> no mask.
        let out = ops::scaled_dot_product_attention(&q, &k, &v, 1.0, ops::AttentionMask::None)?;
        let out = ops::transpose_axes(&out, &[0, 2, 1, 3])?;
        let out = ops::reshape(&out, &[b, l, -1])?;
        self.o_proj.forward(&out)
    }
}

struct VisionMlp {
    gate_proj: ClippedLinear,
    up_proj: ClippedLinear,
    down_proj: ClippedLinear,
}

impl VisionMlp {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let mlp = format!("{prefix}.mlp");
        let clip = cfg.use_clipped_linears;
        Ok(VisionMlp {
            gate_proj: ClippedLinear::load(w, &format!("{mlp}.gate_proj"), clip)?,
            up_proj: ClippedLinear::load(w, &format!("{mlp}.up_proj"), clip)?,
            down_proj: ClippedLinear::load(w, &format!("{mlp}.down_proj"), clip)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = ops::gelu_tanh(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&ops::multiply(&gate, &up)?)
    }
}

struct VisionBlock {
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl VisionBlock {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        Ok(VisionBlock {
            self_attn: VisionAttention::load(w, prefix, cfg)?,
            mlp: VisionMlp::load(w, prefix, cfg)?,
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
        })
    }

    fn forward(&self, x: &Array, positions: &Array) -> Result<Array> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, positions)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let h = ops::add(x, &attn_out)?;

        let normed_h = self.pre_feedforward_layernorm.forward(&h)?;
        let ffw_out = self.mlp.forward(&normed_h)?;
        let ffw_out = self.post_feedforward_layernorm.forward(&ffw_out)?;
        ops::add(&h, &ffw_out)
    }
}

/// Patchify pixels + additive learned 2D position embedding.
struct PatchEmbedder {
    input_proj: Linear,
    /// `[2, position_embedding_size, hidden_size]`.
    position_embedding_table: Array,
    patch_size: i32,
    position_embedding_size: i32,
    hidden_size: i32,
}

impl PatchEmbedder {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        Ok(PatchEmbedder {
            input_proj: w.linear(&format!("{prefix}.input_proj"))?,
            position_embedding_table: w.take(&format!("{prefix}.position_embedding_table"))?,
            patch_size: cfg.patch_size,
            position_embedding_size: cfg.position_embedding_size,
            hidden_size: cfg.hidden_size,
        })
    }

    /// `pixel_values`: `[1, 3, H, W]` -> `[1, patch_h*patch_w, C*p*p]`
    /// projected to hidden_size, normalized to `2*(x-0.5)` before projection
    /// (matches the reference patch embedder; the image-preprocessing step
    /// only rescales to `[0, 1]`, this final normalization happens here).
    fn patchify(&self, pixel_values: &Array) -> Result<Array> {
        let shape = pixel_values.shape();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let p = self.patch_size;
        let (ph, pw) = (h / p, w / p);

        let patches = ops::reshape(pixel_values, &[b, c, ph, p, pw, p])?;
        let patches = ops::transpose_axes(&patches, &[0, 2, 4, 3, 5, 1])?;
        let patches = ops::reshape(&patches, &[b, ph * pw, c * p * p])?;

        let half = ops::astype(&Array::scalar_f32(0.5), patches.dtype())?;
        let patches = ops::subtract(&patches, &half)?;
        let patches = ops::scale_by(&patches, 2.0)?;

        let target_dtype = self.input_proj.weight_dtype(patches.dtype());
        let patches = ops::astype(&patches, target_dtype)?;
        self.input_proj.forward(&patches)
    }

    /// Additive learned position embedding for a `patch_h x patch_w` grid
    /// (row-major, matching `patchify`'s patch ordering): gathers row `x`
    /// from plane 0 and row `y` from plane 1 of the position table and sums
    /// them (equivalent to the reference's one-hot-then-matmul, without the
    /// padding-mask machinery this crate's unpadded single-image path
    /// doesn't need).
    fn position_embeddings(&self, patch_h: i32, patch_w: i32, dtype: Dtype) -> Result<Array> {
        let num_patches = (patch_h * patch_w) as usize;
        let mut x_idx = Vec::with_capacity(num_patches);
        let mut y_idx = Vec::with_capacity(num_patches);
        for row in 0..patch_h {
            for col in 0..patch_w {
                x_idx.push(col as u32);
                y_idx.push(row as u32);
            }
        }
        let x_arr = Array::from_slice(&x_idx, &[num_patches as i32]);
        let y_arr = Array::from_slice(&y_idx, &[num_patches as i32]);

        let pes = self.position_embedding_size;
        let hidden = self.hidden_size;
        let plane_x = ops::reshape(
            &ops::slice(
                &self.position_embedding_table,
                &[0, 0, 0],
                &[1, pes, hidden],
            )?,
            &[pes, hidden],
        )?;
        let plane_y = ops::reshape(
            &ops::slice(
                &self.position_embedding_table,
                &[1, 0, 0],
                &[2, pes, hidden],
            )?,
            &[pes, hidden],
        )?;

        let pe_x = ops::take_axis(&plane_x, &x_arr, 0)?;
        let pe_y = ops::take_axis(&plane_y, &y_arr, 0)?;
        let pe = ops::add(&pe_x, &pe_y)?;
        let pe = ops::astype(&pe, dtype)?;
        ops::expand_dims(&pe, 0)
    }

    fn forward(&self, pixel_values: &Array, patch_h: i32, patch_w: i32) -> Result<Array> {
        let hidden_states = self.patchify(pixel_values)?;
        let pe = self.position_embeddings(patch_h, patch_w, hidden_states.dtype())?;
        ops::add(&hidden_states, &pe)
    }
}

fn debug_stats(label: &str, x: &Array) {
    if let Ok(v) = x.to_vec_f32() {
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "[vision debug] {label}: shape={:?} mean={mean} min={min} max={max}",
            x.shape()
        );
    }
}

/// Host-computed `(x, y)` patch positions for a `patch_h x patch_w` grid,
/// shaped `[1, patch_h*patch_w, 2]` int32 (row-major, matching `patchify`).
fn build_positions(patch_h: i32, patch_w: i32) -> Array {
    let num_patches = (patch_h * patch_w) as usize;
    let mut data = Vec::with_capacity(num_patches * 2);
    for row in 0..patch_h {
        for col in 0..patch_w {
            data.push(col);
            data.push(row);
        }
    }
    Array::from_slice(&data, &[1, num_patches as i32, 2])
}

/// Top-level SigLIP-style vision encoder: patch embed -> N transformer
/// blocks (bidirectional attention, 2D RoPE) -> average-pool down by
/// `pooling_kernel_size` per spatial axis.
pub struct VisionTower {
    config: VisionConfig,
    patch_embedder: PatchEmbedder,
    layers: Vec<VisionBlock>,
}

impl VisionTower {
    pub fn load(w: &mut WeightMap, cfg: VisionConfig) -> Result<Self> {
        let patch_embedder = PatchEmbedder::load(w, "vision_tower.patch_embedder", &cfg)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(VisionBlock::load(
                w,
                &format!("vision_tower.encoder.layers.{i}"),
                &cfg,
            )?);
        }
        Ok(VisionTower {
            config: cfg,
            patch_embedder,
            layers,
        })
    }

    /// Run the encoder over one image's `[1, 3, H, W]` pixel tensor
    /// (`patch_h = H / patch_size`, `patch_w = W / patch_size`, both
    /// divisible by `pooling_kernel_size`). Returns
    /// `[1, num_soft_tokens, hidden_size]`.
    pub fn forward(&self, pixel_values: &Array, patch_h: i32, patch_w: i32) -> Result<Array> {
        let debug = std::env::var("MLEX_DEBUG_VISION").is_ok();
        let mut h = self
            .patch_embedder
            .forward(pixel_values, patch_h, patch_w)?;
        if debug {
            debug_stats("patch_embed", &h);
        }
        let positions = build_positions(patch_h, patch_w);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &positions)?;
            if debug {
                debug_stats(&format!("layer_{i}"), &h);
            }
        }

        let k = self.config.pooling_kernel_size;
        let hidden = self.config.hidden_size;
        let (ph2, pw2) = (patch_h / k, patch_w / k);
        let h = ops::reshape(&h, &[1, ph2, k, pw2, k, hidden])?;
        let h = ops::mean_axes(&h, &[2, 4], false)?;
        let h = ops::reshape(&h, &[1, ph2 * pw2, hidden])?;

        let scale = (hidden as f32).sqrt();
        ops::scale_by(&h, scale)
    }
}

/// Projects vision-tower output into the text model's embedding space:
/// parameter-free RMSNorm, then a (possibly quantized) linear projection.
pub struct MultimodalEmbedder {
    projection: Linear,
    eps: f32,
}

impl MultimodalEmbedder {
    pub fn load(w: &mut WeightMap, prefix: &str, eps: f32) -> Result<Self> {
        Ok(MultimodalEmbedder {
            projection: w.linear(&format!("{prefix}.embedding_projection"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let normed = rms_norm_no_scale(x, self.eps)?;
        self.projection.forward(&normed)
    }
}
