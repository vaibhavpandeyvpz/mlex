//! Gemma4 "unified" encoder-free multimodal path
//! (`vision_config.model_type == "gemma4_unified_vision"` /
//! `audio_config.model_type == "gemma4_unified_audio"`).
//!
//! Distinct from the SigLIP-style tower in [`super::vision`] and the
//! Conformer-style tower in [`super::audio`]: there is no transformer
//! encoder at all. Raw pixel patches (and, for audio, raw PCM windows) are
//! projected straight into the text embedding space by a couple of small
//! linear layers, then fused through the same `embed_vision`/`embed_audio`
//! projection (`vision::MultimodalEmbedder`) every Gemma4 checkpoint uses.
//! This keeps quantized checkpoints small (no encoder weights to ship) at
//! the cost of the encoder's spatial/temporal mixing.
//!
//! Weight key layout (matching the checkpoint verbatim, no renaming; these
//! live in a sidecar `optiq_vision.safetensors` shard for OptiQ
//! checkpoints, transparently merged into the same weight map by
//! `crate::weights::load_all`):
//!   `vision_embedder.{patch_ln1,patch_dense,patch_ln2,pos_norm}.{weight,bias}`
//!   `vision_embedder.pos_embedding`
//!   `embed_vision.embedding_projection.*`, `embed_audio.embedding_projection.*`

use serde_json::Value;

use crate::array::Array;
use crate::error::Result;
use crate::nn::{LayerNorm, Linear, WeightMap};
use crate::ops;

/// Parsed `vision_config` sub-dict of a `gemma4_unified` checkpoint.
#[derive(Debug, Clone)]
pub struct UnifiedVisionConfig {
    /// Pixel-grid patch size used by the resize math (16).
    pub patch_size: i32,
    /// Pooling kernel size (3); `model_patch_size = patch_size *
    /// pooling_kernel_size` is the raw pixel block side length fed
    /// directly into the patch embedder (no separate pooling step - the
    /// "pooling" is baked into patchifying at the larger block size).
    pub pooling_kernel_size: i32,
    /// Embedding width inside the vision embedder (== text hidden_size).
    pub mm_embed_dim: i32,
    /// Number of rows in the 2D positional-embedding table.
    pub mm_posemb_size: i32,
    /// Maximum soft tokens (patches) per image after resize.
    pub num_soft_tokens: i32,
    /// Epsilon for the embedder LayerNorms.
    pub rms_norm_eps: f32,
}

impl UnifiedVisionConfig {
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
        UnifiedVisionConfig {
            patch_size: gi("patch_size", 16),
            pooling_kernel_size: gi("pooling_kernel_size", 3),
            mm_embed_dim: gi("mm_embed_dim", 3840),
            mm_posemb_size: gi("mm_posemb_size", 1120),
            num_soft_tokens: gi("num_soft_tokens", 280),
            rms_norm_eps: gf("rms_norm_eps", 1e-6),
        }
    }

    /// Raw pixel block side length fed into the patch embedder.
    pub fn model_patch_size(&self) -> i32 {
        self.patch_size * self.pooling_kernel_size
    }
}

/// Encoder-free vision embedder: `patch_ln1 -> patch_dense -> patch_ln2 ->
/// + 2D positional embedding -> pos_norm`. Patches are raw (unnormalized
/// beyond the `[0, 1]` rescale already done by
/// `crate::media::image::preprocess_image_bytes`) `model_patch_size x
/// model_patch_size` pixel blocks, flattened per-patch in `(row, col,
/// channel)` order.
pub struct UnifiedVisionEmbedder {
    model_patch_size: i32,
    pooling_kernel_size: i32,
    patch_ln1: LayerNorm,
    patch_dense: Linear,
    patch_ln2: LayerNorm,
    /// `[mm_posemb_size, 2, mm_embed_dim]`.
    pos_embedding: Array,
    pos_norm: LayerNorm,
}

impl UnifiedVisionEmbedder {
    pub fn load(w: &mut WeightMap, cfg: &UnifiedVisionConfig) -> Result<Self> {
        let prefix = "vision_embedder";
        let eps = cfg.rms_norm_eps;
        Ok(UnifiedVisionEmbedder {
            model_patch_size: cfg.model_patch_size(),
            pooling_kernel_size: cfg.pooling_kernel_size,
            patch_ln1: w.layer_norm(&format!("{prefix}.patch_ln1"), eps)?,
            patch_dense: w.linear(&format!("{prefix}.patch_dense"))?,
            patch_ln2: w.layer_norm(&format!("{prefix}.patch_ln2"), eps)?,
            pos_embedding: w.take(&format!("{prefix}.pos_embedding"))?,
            pos_norm: w.layer_norm(&format!("{prefix}.pos_norm"), eps)?,
        })
    }

    /// `pixel_values`: `[1, 3, H, W]` (channel-first, `[0, 1]`-rescaled,
    /// `H`/`W` divisible by `model_patch_size` - guaranteed by
    /// `crate::media::image::preprocess_image_bytes`'s alignment).
    /// `patch_h`/`patch_w` are the caller's `patch_size`-granularity grid
    /// (`H / patch_size`, `W / patch_size`); this divides them down by
    /// `pooling_kernel_size` to get the `model_patch_size`-granularity
    /// grid the raw patchify + position ids use.
    ///
    /// Returns `[1, n, mm_embed_dim]`, `n = (patch_h / k) * (patch_w / k)`.
    pub fn forward(&self, pixel_values: &Array, patch_h: i32, patch_w: i32) -> Result<Array> {
        let k = self.pooling_kernel_size;
        let (grid_h, grid_w) = (patch_h / k, patch_w / k);
        let patches = self.patchify(pixel_values, grid_h, grid_w)?;
        let positions = build_positions(grid_h, grid_w);

        let hidden = self.patch_ln1.forward(&patches)?;
        let hidden = self.patch_dense.forward(&hidden)?;
        let hidden = self.patch_ln2.forward(&hidden)?;
        let hidden = self.add_position_embeddings(&hidden, &positions)?;
        let hidden = self.pos_norm.forward(&hidden)?;
        ops::expand_dims(&hidden, 0)
    }

    /// `[1, 3, H, W] -> [grid_h*grid_w, model_patch_size^2 * 3]`, patch
    /// order row-major over `(grid_h, grid_w)`, flatten order `(py, px,
    /// c)` within a patch - mirrors `super::vision::PatchEmbedder::patchify`
    /// but at `model_patch_size` granularity and with no `2*(x-0.5)`
    /// rescale (the raw `[0, 1]` values feed straight into `patch_ln1`,
    /// which is a learned affine LayerNorm and normalizes them itself).
    fn patchify(&self, pixel_values: &Array, grid_h: i32, grid_w: i32) -> Result<Array> {
        let shape = pixel_values.shape();
        let c = shape[1];
        let p = self.model_patch_size;

        let patches = ops::reshape(pixel_values, &[1, c, grid_h, p, grid_w, p])?;
        let patches = ops::transpose_axes(&patches, &[0, 2, 4, 3, 5, 1])?;
        let patches = ops::reshape(&patches, &[grid_h * grid_w, p * p * c])?;

        let target_dtype = self.patch_dense.weight_dtype(patches.dtype());
        ops::astype(&patches, target_dtype)
    }

    /// `hidden[i] += pos_embedding[x_i, 0] + pos_embedding[y_i, 1]`.
    fn add_position_embeddings(&self, hidden: &Array, positions: &Array) -> Result<Array> {
        let dtype = hidden.dtype();
        let n = positions.dim(0);
        let d = self.pos_embedding.dim(2);
        let pes = self.pos_embedding.dim(0);

        let x_idx = ops::reshape(&ops::slice(positions, &[0, 0], &[n, 1])?, &[n])?;
        let y_idx = ops::reshape(&ops::slice(positions, &[0, 1], &[n, 2])?, &[n])?;

        let plane_x = ops::reshape(
            &ops::slice(&self.pos_embedding, &[0, 0, 0], &[pes, 1, d])?,
            &[pes, d],
        )?;
        let plane_y = ops::reshape(
            &ops::slice(&self.pos_embedding, &[0, 1, 0], &[pes, 2, d])?,
            &[pes, d],
        )?;

        let pe_x = ops::astype(&ops::take_axis(&plane_x, &x_idx, 0)?, dtype)?;
        let pe_y = ops::astype(&ops::take_axis(&plane_y, &y_idx, 0)?, dtype)?;
        let pe = ops::add(&pe_x, &pe_y)?;
        ops::add(hidden, &pe)
    }
}

/// Host-computed `(x, y)` patch positions for a `grid_h x grid_w` grid,
/// shaped `[n, 2]` int32 (row-major, matching `UnifiedVisionEmbedder::patchify`).
fn build_positions(grid_h: i32, grid_w: i32) -> Array {
    let n = (grid_h * grid_w) as usize;
    let mut data = Vec::with_capacity(n * 2);
    for row in 0..grid_h {
        for col in 0..grid_w {
            data.push(col);
            data.push(row);
        }
    }
    Array::from_slice(&data, &[grid_h * grid_w, 2])
}
