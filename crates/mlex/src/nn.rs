//! Neural-net building blocks that understand quantized checkpoints.

use std::collections::HashMap;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::ops;
use crate::quant::{QuantParams, Quantization};

/// All tensors of a checkpoint plus its quantization config; modules pull
/// their parameters out of this store by canonical path.
pub struct WeightMap {
    tensors: HashMap<String, Array>,
    quant: Quantization,
}

impl WeightMap {
    pub fn new(tensors: HashMap<String, Array>, quant: Quantization) -> Self {
        WeightMap { tensors, quant }
    }

    pub fn quantization(&self) -> &Quantization {
        &self.quant
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Remove and return a raw tensor, erroring with the path when missing.
    pub fn take(&mut self, key: &str) -> Result<Array> {
        self.tensors
            .remove(key)
            .ok_or_else(|| Error::Model(format!("missing weight: {key}")))
    }

    pub fn take_optional(&mut self, key: &str) -> Option<Array> {
        self.tensors.remove(key)
    }

    /// Insert or overwrite a tensor (used by `sanitize` steps that
    /// synthesize new keys, e.g. splitting/stacking fused MoE weights).
    pub fn insert(&mut self, key: String, value: Array) {
        self.tensors.insert(key, value);
    }

    /// Rename keys with a mapping function (used by model `sanitize` steps).
    pub fn rename_keys(&mut self, f: impl Fn(&str) -> Option<String>) {
        let keys: Vec<String> = self.tensors.keys().cloned().collect();
        for key in keys {
            if let Some(new_key) = f(&key) {
                if new_key != key {
                    let value = self.tensors.remove(&key).unwrap();
                    self.tensors.insert(new_key, value);
                }
            } else {
                self.tensors.remove(&key);
            }
        }
    }

    /// Apply the same normalization to per-layer quantization override keys.
    pub fn normalize_quant_keys(&mut self, f: impl Fn(&str) -> Option<String>) {
        let entries: Vec<(String, crate::quant::LayerOverride)> =
            self.quant.per_layer.drain().collect();
        for (key, value) in entries {
            let new_key = f(&key).unwrap_or(key);
            self.quant.per_layer.insert(new_key, value);
        }
    }

    /// Load a linear layer at `path`, choosing dense vs quantized from the
    /// checkpoint contents and config (mlx-lm `class_predicate` semantics).
    pub fn linear(&mut self, path: &str) -> Result<Linear> {
        let has_scales = self.contains(&format!("{path}.scales"));
        let params = self.quant.resolve(path, has_scales);
        let weight = self.take(&format!("{path}.weight"))?;
        let bias = self.take_optional(&format!("{path}.bias"));

        match params {
            Some(q) if has_scales => {
                let scales = self.take(&format!("{path}.scales"))?;
                let biases = self.take_optional(&format!("{path}.biases"));
                Ok(Linear::Quantized(QuantizedLinear {
                    weight,
                    scales,
                    biases,
                    params: q,
                    bias,
                }))
            }
            _ => Ok(Linear::Dense(DenseLinear { weight, bias })),
        }
    }

    /// Load an embedding table (dense or quantized).
    pub fn embedding(&mut self, path: &str) -> Result<Embedding> {
        let has_scales = self.contains(&format!("{path}.scales"));
        let params = self.quant.resolve(path, has_scales);
        let weight = self.take(&format!("{path}.weight"))?;

        match params {
            Some(q) if has_scales => {
                let scales = self.take(&format!("{path}.scales"))?;
                let biases = self.take_optional(&format!("{path}.biases"));
                Ok(Embedding::Quantized {
                    weight,
                    scales,
                    biases,
                    params: q,
                })
            }
            _ => Ok(Embedding::Dense { weight }),
        }
    }

    /// Load an RMSNorm weight vector.
    pub fn rms_norm(&mut self, path: &str, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: self.take(&format!("{path}.weight"))?,
            eps,
        })
    }

    /// Load a (weight, bias) LayerNorm.
    pub fn layer_norm(&mut self, path: &str, eps: f32) -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: self.take(&format!("{path}.weight"))?,
            bias: self.take_optional(&format!("{path}.bias")),
            eps,
        })
    }
}

/// Dense (float) linear layer.
pub struct DenseLinear {
    pub weight: Array,
    pub bias: Option<Array>,
}

/// Quantized linear layer backed by fused `quantized_matmul`.
pub struct QuantizedLinear {
    pub weight: Array,
    pub scales: Array,
    pub biases: Option<Array>,
    pub params: QuantParams,
    pub bias: Option<Array>,
}

/// A linear layer as stored in the checkpoint: dense or quantized.
pub enum Linear {
    Dense(DenseLinear),
    Quantized(QuantizedLinear),
}

impl Linear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let out = match self {
            Linear::Dense(l) => {
                let wt = ops::swapaxes(&l.weight, -1, -2)?;
                ops::matmul(x, &wt)?
            }
            Linear::Quantized(l) => ops::quantized_matmul(
                x,
                &l.weight,
                &l.scales,
                l.biases.as_ref(),
                true,
                l.params.group_size,
                l.params.bits,
                l.params.mode,
            )?,
        };
        match self.bias() {
            Some(b) => ops::add(&out, b),
            None => Ok(out),
        }
    }

    fn bias(&self) -> Option<&Array> {
        match self {
            Linear::Dense(l) => l.bias.as_ref(),
            Linear::Quantized(l) => l.bias.as_ref(),
        }
    }

    /// Output feature count (rows of the stored weight matrix).
    pub fn output_dims(&self) -> i32 {
        match self {
            Linear::Dense(l) => l.weight.dim(0),
            Linear::Quantized(l) => l.weight.dim(0),
        }
    }

    /// The dtype an input must be cast to before [`Linear::forward`] (only
    /// meaningful for [`Linear::Dense`]; quantized weights accept any
    /// floating input dtype via `quantized_matmul`, so this returns the
    /// input dtype unchanged in that case).
    pub fn weight_dtype(&self, input_dtype: crate::array::Dtype) -> crate::array::Dtype {
        match self {
            Linear::Dense(l) => l.weight.dtype(),
            Linear::Quantized(_) => input_dtype,
        }
    }
}

/// Token embedding table, optionally quantized, usable as a tied LM head.
pub enum Embedding {
    Dense {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        params: QuantParams,
    },
}

impl Embedding {
    /// Look up embeddings for integer token ids.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        match self {
            Embedding::Dense { weight } => ops::take_axis(weight, ids, 0),
            Embedding::Quantized {
                weight,
                scales,
                biases,
                params,
            } => {
                // Gather packed rows then dequantize just those rows.
                let w = ops::take_axis(weight, ids, 0)?;
                let s = ops::take_axis(scales, ids, 0)?;
                let b = match biases {
                    Some(b) => Some(ops::take_axis(b, ids, 0)?),
                    None => None,
                };
                ops::dequantize(
                    &w,
                    &s,
                    b.as_ref(),
                    params.group_size,
                    params.bits,
                    params.mode,
                )
            }
        }
    }

    /// Use the (transposed) embedding table as the LM head projection.
    pub fn as_linear(&self, x: &Array) -> Result<Array> {
        match self {
            Embedding::Dense { weight } => {
                let wt = ops::swapaxes(weight, -1, -2)?;
                ops::matmul(x, &wt)
            }
            Embedding::Quantized {
                weight,
                scales,
                biases,
                params,
            } => ops::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                true,
                params.group_size,
                params.bits,
                params.mode,
            ),
        }
    }
}

/// RMS normalization with a learned scale.
pub struct RmsNorm {
    pub weight: Array,
    pub eps: f32,
}

impl RmsNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        ops::rms_norm(x, Some(&self.weight), self.eps)
    }
}

/// Standard (mean/variance) layer normalization with an affine weight and
/// optional bias.
pub struct LayerNorm {
    pub weight: Array,
    pub bias: Option<Array>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        ops::layer_norm(x, Some(&self.weight), self.bias.as_ref(), self.eps)
    }
}
