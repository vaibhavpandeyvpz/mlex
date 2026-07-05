//! Qwen3.5-VL vision tower: a dynamic-resolution ViT (patch embed + learned,
//! bilinearly-interpolated absolute position embedding + 2D-RoPE attention
//! blocks) followed by a spatial-merge projector that reduces the patch grid
//! by `spatial_merge_size^2` and projects into the text model's embedding
//! space. Reuses [`crate::media::image::ProcessedImage`] (the same
//! resize-to-a-patch-grid preprocessing Gemma4's vision tower uses) and
//! [`super::super::base::splice_media_features`] to fuse the resulting soft
//! tokens into the prompt at `image_token_id` placeholder positions.
//!
//! Checkpoint weights ship `patch_embed.proj.weight` as a `[out, T, kH, kW,
//! in]` 3D-conv kernel (`T` = `temporal_patch_size`, always 2) even though
//! this crate only ever feeds it a single static frame per image/video
//! frame; since the image processor doesn't duplicate the frame across the
//! temporal axis, the effective 2D kernel is the *sum* of the `T` temporal
//! slices (folding the duplicated-frame convention into the weight once at
//! load time instead of doubling the input).

use serde_json::Value;

use crate::array::Array;
use crate::error::Result;
use crate::nn::{LayerNorm, Linear, WeightMap};
use crate::ops::{self, AttentionMask};

use super::super::config::get_i32;

const VISION_ROPE_THETA: f32 = 10000.0;
const VISION_LN_EPS: f32 = 1e-6;

#[derive(Debug, Clone)]
pub struct Qwen35VisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_heads: i32,
    pub num_layers: i32,
    pub patch_size: i32,
    pub spatial_merge_size: i32,
    pub out_hidden_size: i32,
    pub num_position_embeddings: i32,
}

impl Qwen35VisionConfig {
    pub fn from_json(cfg: &Value) -> Self {
        Qwen35VisionConfig {
            hidden_size: get_i32(cfg, "hidden_size", 768),
            intermediate_size: get_i32(cfg, "intermediate_size", 3072),
            num_heads: get_i32(cfg, "num_heads", 12),
            num_layers: cfg
                .get("depth")
                .and_then(|v| v.as_i64())
                .or_else(|| cfg.get("num_hidden_layers").and_then(|v| v.as_i64()))
                .unwrap_or(12) as i32,
            patch_size: get_i32(cfg, "patch_size", 16),
            spatial_merge_size: get_i32(cfg, "spatial_merge_size", 2),
            out_hidden_size: get_i32(cfg, "out_hidden_size", 1024),
            num_position_embeddings: get_i32(cfg, "num_position_embeddings", 1024),
        }
    }
}

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    attn_proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

impl VisionBlock {
    fn load(weights: &mut WeightMap, prefix: &str) -> Result<Self> {
        Ok(VisionBlock {
            norm1: weights.layer_norm(&format!("{prefix}.norm1"), VISION_LN_EPS)?,
            norm2: weights.layer_norm(&format!("{prefix}.norm2"), VISION_LN_EPS)?,
            qkv: weights.linear(&format!("{prefix}.attn.qkv"))?,
            attn_proj: weights.linear(&format!("{prefix}.attn.proj"))?,
            fc1: weights.linear(&format!("{prefix}.mlp.linear_fc1"))?,
            fc2: weights.linear(&format!("{prefix}.mlp.linear_fc2"))?,
        })
    }

    /// `x`: `[seq, hidden]`. `cos`/`sin`: `[seq, head_dim]` (already
    /// duplicated to the full head width - see
    /// [`Qwen35VisionTower::rotary_cos_sin`]).
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<Array> {
        let seq = x.dim(0);

        let normed = self.norm1.forward(x)?;
        let qkv = self.qkv.forward(&normed)?;
        let qkv = ops::reshape(&qkv, &[seq, 3, num_heads, head_dim])?;
        let qkv = ops::transpose_axes(&qkv, &[1, 0, 2, 3])?;
        let q = ops::squeeze_axis(
            &ops::slice(&qkv, &[0, 0, 0, 0], &[1, seq, num_heads, head_dim])?,
            0,
        )?;
        let k = ops::squeeze_axis(
            &ops::slice(&qkv, &[1, 0, 0, 0], &[2, seq, num_heads, head_dim])?,
            0,
        )?;
        let v = ops::squeeze_axis(
            &ops::slice(&qkv, &[2, 0, 0, 0], &[3, seq, num_heads, head_dim])?,
            0,
        )?;

        let q = apply_vision_rope(&q, cos, sin)?;
        let k = apply_vision_rope(&k, cos, sin)?;

        // [seq, heads, dim] -> [1, heads, seq, dim] for fused SDPA.
        let to_attn = |x: &Array| -> Result<Array> {
            let x = ops::transpose_axes(x, &[1, 0, 2])?;
            ops::expand_dims(&x, 0)
        };
        let q = to_attn(&q)?;
        let k = to_attn(&k)?;
        let v = to_attn(&v)?;

        let scale = (head_dim as f32).powf(-0.5);
        let attn = ops::scaled_dot_product_attention(&q, &k, &v, scale, AttentionMask::None)?;
        // [1, heads, seq, dim] -> [seq, heads*dim]
        let attn = ops::transpose_axes(&attn, &[0, 2, 1, 3])?;
        let attn = ops::reshape(&attn, &[seq, num_heads * head_dim])?;
        let attn = self.attn_proj.forward(&attn)?;
        let x = ops::add(x, &attn)?;

        let normed = self.norm2.forward(&x)?;
        let hidden = self.fc1.forward(&normed)?;
        let hidden = ops::gelu_tanh(&hidden)?;
        let hidden = self.fc2.forward(&hidden)?;
        ops::add(&x, &hidden)
    }
}

/// Reduces the merged `spatial_merge_size x spatial_merge_size` patch
/// blocks into a single token and projects to the text model's hidden
/// size: `LayerNorm -> reshape/transpose merge -> Linear -> GELU -> Linear`.
struct SpatialMerger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    merge: i32,
}

impl SpatialMerger {
    fn load(weights: &mut WeightMap, prefix: &str, merge: i32) -> Result<Self> {
        Ok(SpatialMerger {
            norm: weights.layer_norm(&format!("{prefix}.norm"), VISION_LN_EPS)?,
            fc1: weights.linear(&format!("{prefix}.linear_fc1"))?,
            fc2: weights.linear(&format!("{prefix}.linear_fc2"))?,
            merge,
        })
    }

    /// `x`: `[patch_h * patch_w, hidden]` in row-major `(h, w)` order.
    fn forward(&self, x: &Array, patch_h: i32, patch_w: i32) -> Result<Array> {
        let hidden = x.dim(1);
        let normed = self.norm.forward(x)?;

        let merge = self.merge;
        let h_block = patch_h / merge;
        let w_block = patch_w / merge;
        let reshaped = ops::reshape(&normed, &[h_block, merge, w_block, merge, hidden])?;
        let transposed = ops::transpose_axes(&reshaped, &[0, 2, 1, 3, 4])?;
        let merged = ops::reshape(&transposed, &[h_block * w_block, merge * merge * hidden])?;

        let hidden = self.fc1.forward(&merged)?;
        let hidden = ops::gelu_tanh(&hidden)?;
        self.fc2.forward(&hidden)
    }
}

pub struct Qwen35VisionTower {
    config: Qwen35VisionConfig,
    patch_embed: Linear,
    pos_embed: Array,
    blocks: Vec<VisionBlock>,
    merger: SpatialMerger,
    base_grid: i32,
}

impl Qwen35VisionTower {
    pub fn load(weights: &mut WeightMap, config: Qwen35VisionConfig) -> Result<Self> {
        let prefix = "vision_tower";

        let pe_weight = weights.take(&format!("{prefix}.patch_embed.proj.weight"))?;
        let pe_bias = weights.take_optional(&format!("{prefix}.patch_embed.proj.bias"));
        // Collapse the `[out, T, kH, kW, in]` conv weight into a plain
        // `[out, kH*kW*in]` linear projection (see module docs).
        let pe_2d = if pe_weight.ndim() == 5 {
            ops::sum_axes(&pe_weight, &[1], false)?
        } else {
            pe_weight
        };
        let out_dim = pe_2d.dim(0);
        let in_dim = pe_2d.dim(1) * pe_2d.dim(2) * pe_2d.dim(3);
        let pe_flat = ops::reshape(&pe_2d, &[out_dim, in_dim])?;
        let patch_embed = Linear::Dense(crate::nn::DenseLinear {
            weight: pe_flat,
            bias: pe_bias,
        });

        let pos_embed = weights.take(&format!("{prefix}.pos_embed.weight"))?;
        let base_grid = (config.num_position_embeddings as f64).sqrt().round() as i32;

        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            blocks.push(VisionBlock::load(weights, &format!("{prefix}.blocks.{i}"))?);
        }

        let merger = SpatialMerger::load(
            weights,
            &format!("{prefix}.merger"),
            config.spatial_merge_size,
        )?;

        Ok(Qwen35VisionTower {
            config,
            patch_embed,
            pos_embed,
            blocks,
            merger,
            base_grid,
        })
    }

    /// Non-overlapping patch extraction: `[1, 3, H, W]` (`H = patch_h *
    /// patch_size`, `W = patch_w * patch_size`) -> `[patch_h * patch_w,
    /// patch_size * patch_size * 3]` in row-major `(h, w)` patch order with
    /// each patch's pixels flattened `(py, px, channel)` - matching the
    /// checkpoint's collapsed patch-embed weight layout.
    fn patchify(
        pixel_values: &Array,
        patch_h: i32,
        patch_w: i32,
        patch_size: i32,
    ) -> Result<Array> {
        let split = ops::reshape(pixel_values, &[3, patch_h, patch_size, patch_w, patch_size])?;
        let transposed = ops::transpose_axes(&split, &[1, 3, 2, 4, 0])?;
        ops::reshape(
            &transposed,
            &[patch_h * patch_w, patch_size * patch_size * 3],
        )
    }

    /// Position embeddings for a `patch_h x patch_w` grid, bilinearly
    /// interpolated from the checkpoint's fixed `base_grid x base_grid`
    /// table when the requested grid differs (dynamic-resolution images
    /// rarely land exactly on the base grid).
    fn position_embeddings(&self, patch_h: i32, patch_w: i32) -> Result<Array> {
        if patch_h == self.base_grid && patch_w == self.base_grid {
            return Ok(self.pos_embed.clone());
        }
        let hidden = self.pos_embed.dim(1);
        let grid = ops::reshape(&self.pos_embed, &[self.base_grid, self.base_grid, hidden])?;
        bilinear_interpolate_grid(&grid, patch_h, patch_w)
    }

    /// `cos`/`sin` for 2D vision RoPE over a `patch_h x patch_w` grid in
    /// row-major order, each `[num_patches, head_dim]` (concatenated
    /// height/width frequency bands, duplicated to the full head width to
    /// match the `x*cos + rotate_half(x)*sin` application in
    /// [`apply_vision_rope`]).
    fn rotary_cos_sin(&self, patch_h: i32, patch_w: i32, head_dim: i32) -> Result<(Array, Array)> {
        let rope_dim = head_dim / 2; // split evenly between h and w bands
        let half = rope_dim / 2;
        let max_grid = patch_h.max(patch_w);

        let mut inv_freq = Vec::with_capacity(half as usize);
        for i in 0..half {
            let exp = (2 * i) as f32 / rope_dim as f32;
            inv_freq.push(1.0 / VISION_ROPE_THETA.powf(exp));
        }
        let inv_freq = Array::from_slice(&inv_freq, &[half]);
        let positions = ops::arange(0.0, max_grid as f64, 1.0, crate::array::Dtype::Float32)?;
        let positions = ops::reshape(&positions, &[max_grid, 1])?;
        let inv_freq_row = ops::reshape(&inv_freq, &[1, half])?;
        // freqs_table[p, i] = p * inv_freq[i], shape [max_grid, half]
        let freqs_table = ops::multiply(
            &ops::broadcast_to(&positions, &[max_grid, half])?,
            &ops::broadcast_to(&inv_freq_row, &[max_grid, half])?,
        )?;

        let mut h_ids = Vec::with_capacity((patch_h * patch_w) as usize);
        let mut w_ids = Vec::with_capacity((patch_h * patch_w) as usize);
        for h in 0..patch_h {
            for w in 0..patch_w {
                h_ids.push(h);
                w_ids.push(w);
            }
        }
        let h_ids = Array::from_slice(&h_ids, &[(patch_h * patch_w) as i32]);
        let w_ids = Array::from_slice(&w_ids, &[(patch_h * patch_w) as i32]);

        let h_freqs = ops::take_axis(&freqs_table, &h_ids, 0)?;
        let w_freqs = ops::take_axis(&freqs_table, &w_ids, 0)?;
        // [num_patches, rope_dim]
        let freqs = ops::concatenate(&[&h_freqs, &w_freqs], -1)?;

        let cos = ops::cos(&freqs)?;
        let sin = ops::sin(&freqs)?;
        // Duplicate to the full head width (rotate_half convention).
        let cos = ops::concatenate(&[&cos, &cos], -1)?;
        let sin = ops::concatenate(&[&sin, &sin], -1)?;
        Ok((cos, sin))
    }

    /// `pixel_values`: `[1, 3, H, W]` float32 in `[0, 1]` (Gemma4-style
    /// [`ProcessedImage`] convention: no mean/std normalization applied by
    /// preprocessing - the tower applies it here, matching the checkpoint's
    /// `image_mean=image_std=0.5` processor config).
    pub fn forward(&self, pixel_values: &Array, patch_h: i32, patch_w: i32) -> Result<Array> {
        let normalized = ops::add_scalar(&ops::scale_by(pixel_values, 2.0)?, -1.0)?;
        let patches = Self::patchify(&normalized, patch_h, patch_w, self.config.patch_size)?;

        let mut h = self.patch_embed.forward(&patches)?;
        let pos = self.position_embeddings(patch_h, patch_w)?;
        h = ops::add(&h, &ops::astype(&pos, h.dtype())?)?;

        let head_dim = self.config.hidden_size / self.config.num_heads;
        let (cos, sin) = self.rotary_cos_sin(patch_h, patch_w, head_dim)?;
        let cos = ops::astype(&cos, h.dtype())?;
        let sin = ops::astype(&sin, h.dtype())?;

        for block in &self.blocks {
            h = block.forward(&h, &cos, &sin, self.config.num_heads, head_dim)?;
        }

        let merged = self.merger.forward(&h, patch_h, patch_w)?;
        ops::expand_dims(&merged, 0)
    }

    pub fn config(&self) -> &Qwen35VisionConfig {
        &self.config
    }
}

/// Apply 2D vision RoPE to `x` (`[seq, heads, head_dim]`) given `cos`/`sin`
/// (`[seq, head_dim]`), broadcasting over the heads axis.
fn apply_vision_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let seq = x.dim(0);
    let heads = x.dim(1);
    let head_dim = x.dim(2);
    let cos = ops::expand_dims(cos, 1)?;
    let sin = ops::expand_dims(sin, 1)?;
    let cos = ops::broadcast_to(&cos, &[seq, heads, head_dim])?;
    let sin = ops::broadcast_to(&sin, &[seq, heads, head_dim])?;

    let half = head_dim / 2;
    let x1 = ops::slice(x, &[0, 0, 0], &[seq, heads, half])?;
    let x2 = ops::slice(x, &[0, 0, half], &[seq, heads, head_dim])?;
    let rotated = ops::concatenate(&[&ops::negative(&x2)?, &x1], -1)?;

    ops::add(&ops::multiply(x, &cos)?, &ops::multiply(&rotated, &sin)?)
}

/// Separable bilinear interpolation of a `[base, base, dim]` grid to
/// `[target_h, target_w, dim]`, `align_corners=false` semantics (matching
/// `torch.nn.functional.interpolate`'s default): built as two small
/// interpolation weight matrices (`[target, base]`) applied via matmul
/// along each spatial axis in turn, rather than a general-purpose resize
/// primitive this crate otherwise has no need for.
fn bilinear_interpolate_grid(grid: &Array, target_h: i32, target_w: i32) -> Result<Array> {
    let base_h = grid.dim(0);
    let base_w = grid.dim(1);
    let dim = grid.dim(2);

    let interp_matrix = |target: i32, base: i32| -> Array {
        let scale = base as f64 / target as f64;
        let mut weights = vec![0f32; (target * base) as usize];
        for t in 0..target {
            let src = ((t as f64 + 0.5) * scale - 0.5)
                .max(0.0)
                .min((base - 1) as f64);
            let lo = src.floor() as i32;
            let hi = (lo + 1).min(base - 1);
            let frac = (src - lo as f64) as f32;
            weights[(t * base + lo) as usize] += 1.0 - frac;
            if hi != lo {
                weights[(t * base + hi) as usize] += frac;
            } else {
                weights[(t * base + lo) as usize] += frac;
            }
        }
        Array::from_slice(&weights, &[target, base])
    };

    let h_matrix = interp_matrix(target_h, base_h);
    let w_matrix = interp_matrix(target_w, base_w);

    // Interpolate along H: [target_h, base_h] @ [base_h, base_w * dim] -> [target_h, base_w * dim]
    let grid_flat = ops::reshape(grid, &[base_h, base_w * dim])?;
    let h_interp = ops::matmul(&h_matrix, &grid_flat)?;
    let h_interp = ops::reshape(&h_interp, &[target_h, base_w, dim])?;

    // Interpolate along W: transpose so W is leading, matmul, transpose back.
    let h_interp_t = ops::transpose_axes(&h_interp, &[1, 0, 2])?;
    let h_interp_t_flat = ops::reshape(&h_interp_t, &[base_w, target_h * dim])?;
    let w_interp = ops::matmul(&w_matrix, &h_interp_t_flat)?;
    let w_interp = ops::reshape(&w_interp, &[target_w, target_h, dim])?;
    let result = ops::transpose_axes(&w_interp, &[1, 0, 2])?;
    ops::reshape(&result, &[target_h * target_w, dim])
}

/// Whether `weights` has the tensors [`Qwen35VisionTower::load`] needs.
pub fn has_vision_weights(weights: &WeightMap) -> bool {
    weights.contains("vision_tower.patch_embed.proj.weight")
}
