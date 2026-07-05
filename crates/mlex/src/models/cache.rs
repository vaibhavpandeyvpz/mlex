//! Per-layer caches used during incremental generation.
//!
//! Two kinds exist: the standard quadratic-attention `KvCache` (growing
//! keys/values) and `GatedDeltaCache` for linear-attention layers
//! (Qwen3.5/3.6 `GatedDeltaNet`), which instead carries a fixed-size causal
//! conv window plus a recurrent `[B, H, Dv, Dk]` state matrix. `LayerCache`
//! lets `Model::forward` treat both uniformly across architectures.

use crate::array::Array;
use crate::error::{Error, Result};
use crate::ops;

/// Growing key/value cache for one attention layer.
///
/// Concatenates on every step; simple and correct. A pre-allocated,
/// amortized-growth version (as mlx-lm's `KVCache` does) is a drop-in
/// optimization that can replace this without touching call sites.
#[derive(Default, Clone)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn offset(&self) -> i32 {
        self.keys.as_ref().map(|k| k.dim(-2)).unwrap_or(0)
    }

    /// Append `keys`/`values` (shape `[B, H, L, D]`) and return the full
    /// cache contents seen so far.
    pub fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array)> {
        let (full_k, full_v) = match (self.keys.take(), self.values.take()) {
            (Some(ok), Some(ov)) => (
                ops::concatenate(&[&ok, &keys], -2)?,
                ops::concatenate(&[&ov, &values], -2)?,
            ),
            _ => (keys, values),
        };
        self.keys = Some(full_k.clone());
        self.values = Some(full_v.clone());
        Ok((full_k, full_v))
    }
}

/// Recurrent state for one `GatedDeltaNet` (linear-attention) layer.
#[derive(Default, Clone)]
pub struct GatedDeltaCache {
    /// Trailing `kernel_size - 1` conv input frames, `[B, K-1, conv_dim]`.
    pub conv_state: Option<Array>,
    /// Recurrent delta-rule state, `[B, Hv, Dv, Dk]` (kept in f32).
    pub recur_state: Option<Array>,
}

impl GatedDeltaCache {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Per-layer state for a DharaAR decoder block: one attention `KvCache`
/// plus up to four causal-conv "Canon layer" states (positions A/B(q,k,v)/C/D
/// — see `models::dhara`), each a trailing `[B, kernel-1, dim]` window of
/// prior inputs, `None` until the first token has been fed through.
#[derive(Default, Clone)]
pub struct DharaCache {
    pub attn: KvCache,
    pub canon_a: Option<Array>,
    pub canon_b_q: Option<Array>,
    pub canon_b_k: Option<Array>,
    pub canon_b_v: Option<Array>,
    pub canon_c: Option<Array>,
    pub canon_d: Option<Array>,
}

impl DharaCache {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Either flavor of per-layer cache a model architecture may need.
///
/// Cloning is O(1) per `Array` field (a refcount bump on MLX's
/// `shared_ptr`-backed buffer, not a deep copy - see `Array::clone` in
/// `array.rs`), so cloning a whole `Vec<LayerCache>` to fork a cached
/// prefix (see `prompt_cache.rs`) costs O(num_layers), not O(cache size).
#[derive(Clone)]
pub enum LayerCache {
    Attention(KvCache),
    GatedDelta(GatedDeltaCache),
    Dhara(DharaCache),
}

impl LayerCache {
    pub fn new_attention() -> Self {
        LayerCache::Attention(KvCache::new())
    }

    pub fn new_gated_delta() -> Self {
        LayerCache::GatedDelta(GatedDeltaCache::new())
    }

    pub fn new_dhara() -> Self {
        LayerCache::Dhara(DharaCache::new())
    }

    pub fn as_attention(&mut self) -> Result<&mut KvCache> {
        match self {
            LayerCache::Attention(c) => Ok(c),
            _ => Err(Error::Model(
                "expected an attention cache, found a different cache kind".into(),
            )),
        }
    }

    pub fn as_gated_delta(&mut self) -> Result<&mut GatedDeltaCache> {
        match self {
            LayerCache::GatedDelta(c) => Ok(c),
            _ => Err(Error::Model(
                "expected a gated-delta cache, found a different cache kind".into(),
            )),
        }
    }

    pub fn as_dhara(&mut self) -> Result<&mut DharaCache> {
        match self {
            LayerCache::Dhara(c) => Ok(c),
            _ => Err(Error::Model(
                "expected a dhara cache, found a different cache kind".into(),
            )),
        }
    }
}
