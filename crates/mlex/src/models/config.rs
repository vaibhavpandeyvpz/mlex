//! Common config parsing helpers shared across architectures.

use serde_json::Value;

use crate::error::{Error, Result};

pub fn get_str<'a>(cfg: &'a Value, key: &str) -> Option<&'a str> {
    cfg.get(key).and_then(|v| v.as_str())
}

pub fn get_i32(cfg: &Value, key: &str, default: i32) -> i32 {
    cfg.get(key)
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(default)
}

pub fn require_i32(cfg: &Value, key: &str) -> Result<i32> {
    cfg.get(key)
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .ok_or_else(|| Error::Config(format!("missing required integer field '{key}'")))
}

pub fn get_f32(cfg: &Value, key: &str, default: f32) -> f32 {
    cfg.get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(default)
}

pub fn get_bool(cfg: &Value, key: &str, default: bool) -> bool {
    cfg.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

/// Read `config.json`, following the `text_config` nesting used by
/// multimodal checkpoints (Gemma4, Qwen3.5-VL, ...) when the requested key
/// is not present at the top level.
pub fn text_config(cfg: &Value) -> &Value {
    cfg.get("text_config").unwrap_or(cfg)
}
