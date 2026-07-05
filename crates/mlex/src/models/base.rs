//! Shared helpers used by every model architecture.

use crate::array::{Array, Dtype};
use crate::error::{Error, Result};
use crate::ops::{self, AttentionMask};

/// Decide the attention mask mode for one forward pass, mirroring mlx-lm's
/// `create_attention_mask`: no mask is needed when decoding a single new
/// token against an existing cache (the new token can attend to everything
/// already cached), otherwise use a causal mask.
pub fn attention_mask_for(seq_len: i32) -> AttentionMask {
    if seq_len == 1 {
        AttentionMask::None
    } else {
        AttentionMask::Causal
    }
}

/// RoPE configuration for one attention layer.
#[derive(Debug, Clone, Copy)]
pub struct RopeConfig {
    pub dims: i32,
    pub base: f32,
    pub traditional: bool,
    /// Linear scaling factor (`rope_scaling: {"type": "linear", "factor": f}`).
    /// MLX's fused RoPE kernel applies `1/scale` to position indices, i.e.
    /// passing `scale = 1/factor` stretches the effective context.
    pub scale: f32,
}

impl RopeConfig {
    pub fn new(dims: i32, base: f32) -> Self {
        RopeConfig {
            dims,
            base,
            traditional: false,
            scale: 1.0,
        }
    }

    pub fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
        ops::rope(
            x,
            self.dims,
            self.traditional,
            Some(self.base),
            self.scale,
            offset,
            None,
        )
    }
}

/// Reshape `[B, L, H*D]` into `[B, H, L, D]` for attention.
pub fn split_heads(x: &Array, batch: i32, seq: i32, heads: i32) -> Result<Array> {
    let reshaped = ops::reshape(x, &[batch, seq, heads, -1])?;
    ops::transpose_axes(&reshaped, &[0, 2, 1, 3])
}

/// Reshape `[B, H, L, D]` back into `[B, L, H*D]`.
pub fn merge_heads(x: &Array, batch: i32, seq: i32) -> Result<Array> {
    let t = ops::transpose_axes(x, &[0, 2, 1, 3])?;
    ops::reshape(&t, &[batch, seq, -1])
}

/// Repeat KV heads along the head axis to match the query head count (GQA).
pub fn repeat_kv_heads(x: &Array, n_repeats: i32) -> Result<Array> {
    if n_repeats == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let (b, h, l, d) = (shape[0], shape[1], shape[2], shape[3]);
    let expanded = ops::expand_dims(x, 2)?;
    let broadcasted = ops::broadcast_to(&expanded, &[b, h, n_repeats, l, d])?;
    ops::reshape(&broadcasted, &[b, h * n_repeats, l, d])
}

/// Splice `features` (one `[1, N_i, hidden]` tensor per media item, in
/// prompt order) into `h` at the positions where `input_ids` equals
/// `placeholder_token_id`, erroring on any placeholder/feature count
/// mismatch. Shared by every architecture's image/audio fusion path
/// (Gemma4's classic + unified towers, Qwen3.5-VL's vision tower, ...).
pub fn splice_media_features(
    h: &Array,
    input_ids: &Array,
    mut features: Vec<Array>,
    placeholder_token_id: i32,
    modality: &str,
) -> Result<Array> {
    let features = if features.len() == 1 {
        features.remove(0)
    } else {
        let refs: Vec<&Array> = features.iter().collect();
        ops::concatenate(&refs, 1)?
    };
    let features = ops::astype(&features, h.dtype())?;

    let placeholder = ops::astype(&Array::scalar_i32(placeholder_token_id), input_ids.dtype())?;
    let mask = ops::equal(input_ids, &placeholder)?;
    let mask_count_arr = ops::sum_axes(
        &ops::reshape(&ops::astype(&mask, Dtype::Int32)?, &[-1])?,
        &[0],
        false,
    )?;
    let mask_count = mask_count_arr.item_f32()? as i32;
    let feature_count = features.dim(1);
    if mask_count != feature_count {
        return Err(Error::Model(format!(
            "{modality} token count ({mask_count}) does not match {modality} feature count \
             ({feature_count}); check that {modality} placeholder expansion produced the right number of tokens"
        )));
    }

    let mask_expanded = ops::broadcast_to(&ops::expand_dims(&mask, -1)?, &h.shape())?;
    masked_scatter(h, &mask_expanded, &features)
}

/// Replace positions where `mask` is true with values from `source` (in
/// mask-order), keeping `input`'s values everywhere else. Out-of-range
/// indices (positions before the first `true`, which would cumsum to -1)
/// are clamped into `[0, source_size)` instead of wrapping via modulo -
/// equivalent here since those lookups are always discarded by the
/// `where_cond` below (only `true` positions ever select `aligned`).
pub fn masked_scatter(input: &Array, mask: &Array, source: &Array) -> Result<Array> {
    let input_shape = input.shape();
    let mask_flat = ops::reshape(&ops::astype(mask, Dtype::Int32)?, &[-1])?;
    let input_flat = ops::reshape(input, &[-1])?;
    let source_flat = ops::reshape(source, &[-1])?;
    let source_size = source_flat.dim(0);

    let idx = ops::subtract(&ops::cumsum(&mask_flat, 0)?, &Array::scalar_i32(1))?;
    let idx = ops::maximum(&idx, &Array::scalar_i32(0))?;
    let idx = ops::minimum(&idx, &Array::scalar_i32((source_size - 1).max(0)))?;
    let idx = ops::astype(&idx, Dtype::UInt32)?;

    let aligned = ops::take(&source_flat, &idx)?;
    let mask_bool = ops::astype(&mask_flat, Dtype::Bool)?;
    let result = ops::where_cond(&mask_bool, &aligned, &input_flat)?;
    ops::reshape(&result, &input_shape)
}
