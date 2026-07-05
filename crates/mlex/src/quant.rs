//! Quantization configuration parsing and resolution.
//!
//! Follows mlx-lm semantics: `config.json` carries a `quantization` object
//! with global defaults (`group_size`, `bits`, optional `mode`) plus
//! per-layer overrides keyed by module path. A layer is quantized when
//! either an override exists for its path or a `{path}.scales` tensor is
//! present in the checkpoint.

use std::collections::HashMap;

use serde_json::Value;

use crate::array::{Array, Dtype};
use crate::error::{Error, Result};
use crate::ops::{self, QuantMode};

/// Quantization parameters for one layer (or the global default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantParams {
    pub group_size: i32,
    pub bits: i32,
    pub mode: QuantMode,
}

/// Per-layer override: quantize with specific params, or skip quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerOverride {
    Params(QuantParams),
    Skip,
}

/// Parsed `quantization` section of a model config.
#[derive(Debug, Clone, Default)]
pub struct Quantization {
    pub default: Option<QuantParams>,
    pub per_layer: HashMap<String, LayerOverride>,
}

impl Quantization {
    /// Parse from the root config JSON. Checks `quantization` first, then
    /// `quantization_config` (kept in sync by mlx-lm on save).
    pub fn from_config(config: &Value) -> Result<Self> {
        let section = config
            .get("quantization")
            .or_else(|| config.get("quantization_config"));
        let Some(section) = section.and_then(|s| s.as_object()) else {
            return Ok(Quantization::default());
        };

        let default_mode = section
            .get("mode")
            .and_then(|m| m.as_str())
            .map(QuantMode::parse)
            .transpose()?
            .unwrap_or(QuantMode::Affine);
        let default_group = section.get("group_size").and_then(|v| v.as_i64());
        let default_bits = section.get("bits").and_then(|v| v.as_i64());

        let default = match (default_group, default_bits) {
            (Some(g), Some(b)) => Some(QuantParams {
                group_size: g as i32,
                bits: b as i32,
                mode: default_mode,
            }),
            _ => None,
        };

        let mut per_layer = HashMap::new();
        for (key, value) in section {
            match value {
                Value::Object(obj) => {
                    let group_size = obj
                        .get("group_size")
                        .and_then(|v| v.as_i64())
                        .or(default_group)
                        .ok_or_else(|| {
                            Error::Config(format!(
                                "layer '{key}' quant override missing group_size"
                            ))
                        })? as i32;
                    let bits = obj
                        .get("bits")
                        .and_then(|v| v.as_i64())
                        .or(default_bits)
                        .ok_or_else(|| {
                            Error::Config(format!("layer '{key}' quant override missing bits"))
                        })? as i32;
                    let mode = obj
                        .get("mode")
                        .and_then(|m| m.as_str())
                        .map(QuantMode::parse)
                        .transpose()?
                        .unwrap_or(default_mode);
                    per_layer.insert(
                        key.clone(),
                        LayerOverride::Params(QuantParams {
                            group_size,
                            bits,
                            mode,
                        }),
                    );
                }
                Value::Bool(false) => {
                    per_layer.insert(key.clone(), LayerOverride::Skip);
                }
                _ => {} // group_size / bits / mode scalars handled above
            }
        }

        Ok(Quantization { default, per_layer })
    }

    /// Does `weight_map` carry per-tensor dynamic-range int8 weights at
    /// `path` (`{path}.input_min`/`input_max`/`output_min`/`output_max`
    /// sitting alongside `{path}.weight`, as opposed to `.scales`)? Used to
    /// pick between [`crate::nn::WeightMap::linear`]'s group-affine path
    /// and [`dequantize_dynamic_int8`] when loading multimodal tower
    /// weights (Gemma4 vision/audio).
    pub fn is_dynamic_range_int8(weight_map: &crate::nn::WeightMap, path: &str) -> bool {
        weight_map.contains(&format!("{path}.output_min"))
            && weight_map.contains(&format!("{path}.output_max"))
    }

    /// Whether any quantization is configured at all.
    pub fn is_quantized(&self) -> bool {
        self.default.is_some() || !self.per_layer.is_empty()
    }

    /// Resolve quantization for the module at `path`, mirroring the mlx-lm
    /// `class_predicate`:
    /// - a per-layer override wins (params or skip),
    /// - otherwise quantize with defaults iff `{path}.scales` exists,
    /// - `path` may be probed with alternative prefixes by the caller.
    pub fn resolve(&self, path: &str, has_scales: bool) -> Option<QuantParams> {
        match self.per_layer.get(path) {
            Some(LayerOverride::Params(p)) => Some(*p),
            Some(LayerOverride::Skip) => None,
            None => {
                if has_scales {
                    self.default
                } else {
                    None
                }
            }
        }
    }
}

/// Per-tensor dynamic-range int8 quantization parameters (Gemma4
/// vision/audio tower weights). Distinct from the group-affine scheme
/// above: one scale/zero-point pair for the *whole* weight tensor rather
/// than per-group, plus separately-recorded input-activation range
/// (unused for pure weight-only dequantization, kept for completeness /
/// potential future activation-quantization support).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicRangeParams {
    pub input_min: f32,
    pub input_max: f32,
    pub output_min: f32,
    pub output_max: f32,
}

/// Dequantize a per-tensor dynamic-range int8 weight: maps the full signed
/// int8 range `[-128, 127]` affinely onto `[output_min, output_max]`.
pub fn dequantize_dynamic_int8(weight_i8: &Array, params: DynamicRangeParams) -> Result<Array> {
    let scale = (params.output_max - params.output_min) / 255.0;
    let shifted = ops::add(
        &ops::astype(weight_i8, Dtype::Float32)?,
        &Array::scalar_f32(128.0),
    )?;
    let scaled = ops::scale_by(&shifted, scale)?;
    ops::add(&scaled, &Array::scalar_f32(params.output_min))
}

/// Quantize a dense f32 weight tensor into the same per-tensor
/// dynamic-range int8 scheme [`dequantize_dynamic_int8`] reads back, using
/// the tensor's own min/max as the output range. Used by tests to check
/// the encode/decode round-trip; real checkpoints ship pre-quantized.
pub fn quantize_dynamic_int8(weight: &[f32]) -> (Vec<i8>, DynamicRangeParams) {
    let min = weight.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scale = ((max - min) / 255.0).max(f32::EPSILON);
    let q: Vec<i8> = weight
        .iter()
        .map(|&w| (((w - min) / scale).round() - 128.0).clamp(-128.0, 127.0) as i8)
        .collect();
    (
        q,
        DynamicRangeParams {
            input_min: min,
            input_max: max,
            output_min: min,
            output_max: max,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_int8_round_trip_is_close() {
        let original = vec![-2.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, 3.5];
        let (q, params) = quantize_dynamic_int8(&original);
        let q_arr = Array::from_slice(
            &q.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            &[original.len() as i32],
        );
        let q_arr = ops::astype(&q_arr, Dtype::Int8).unwrap();
        let dequantized = dequantize_dynamic_int8(&q_arr, params).unwrap();
        let out = dequantized.to_vec_f32().unwrap();
        let tolerance = (params.output_max - params.output_min) / 255.0 + 1e-4;
        for (a, b) in original.iter().zip(out.iter()) {
            assert!((a - b).abs() <= tolerance, "{a} vs {b} (tol {tolerance})");
        }
    }

    #[test]
    fn quantization_from_config_defaults_when_absent() {
        let cfg = serde_json::json!({});
        let q = Quantization::from_config(&cfg).unwrap();
        assert!(!q.is_quantized());
        assert!(q.default.is_none());
    }

    #[test]
    fn quantization_from_config_parses_global_defaults() {
        let cfg = serde_json::json!({"quantization": {"group_size": 64, "bits": 4}});
        let q = Quantization::from_config(&cfg).unwrap();
        assert!(q.is_quantized());
        let params = q.default.unwrap();
        assert_eq!(params.group_size, 64);
        assert_eq!(params.bits, 4);
        assert_eq!(params.mode, QuantMode::Affine);
    }

    #[test]
    fn quantization_per_layer_skip_override() {
        let cfg = serde_json::json!({
            "quantization": {"group_size": 64, "bits": 4, "model.layers.0.mlp": false}
        });
        let q = Quantization::from_config(&cfg).unwrap();
        assert_eq!(q.resolve("model.layers.0.mlp", true), None);
        assert!(q.resolve("model.layers.1.mlp", true).is_some());
    }

    #[test]
    fn quantization_per_layer_params_override() {
        let cfg = serde_json::json!({
            "quantization": {
                "group_size": 64, "bits": 4,
                "model.layers.0.mlp": {"group_size": 32, "bits": 8}
            }
        });
        let q = Quantization::from_config(&cfg).unwrap();
        let params = q.resolve("model.layers.0.mlp", true).unwrap();
        assert_eq!(params.group_size, 32);
        assert_eq!(params.bits, 8);
    }

    #[test]
    fn resolve_without_scales_and_without_override_is_none() {
        let cfg = serde_json::json!({"quantization": {"group_size": 64, "bits": 4}});
        let q = Quantization::from_config(&cfg).unwrap();
        assert_eq!(q.resolve("model.layers.0.mlp", false), None);
    }
}
