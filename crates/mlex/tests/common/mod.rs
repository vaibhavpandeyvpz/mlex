//! Shared model-discovery + measured-memory size-gating helper for every
//! integration test suite in this crate.
//!
//! Rather than guessing CI-eligibility from disk size (unreliable - actual
//! runtime footprint tracks disk size closely for plain transformers but
//! runs substantially higher for architectures like NemotronH whose
//! Mamba/GatedDelta recurrences upcast to f32), this measures real peak
//! resident memory once per model directory (load + one short forward
//! pass, `getrusage(RUSAGE_SELF, ...)` after), caches the result in
//! `target/mlex-test-cache/model-memory.json`, and classifies against
//! `MLEX_MAX_MODEL_GB` (default 5.0).
//!
//! Env vars:
//! - `MLEX_MODELS_DIR` (default `<repo_root>/models`): HF-cache-layout
//!   directory to scan (`models--<org>--<name>/snapshots/<rev>/`).
//! - `MLEX_MAX_MODEL_GB` (default `5.0`): peak-RSS ceiling for `ci_safe`.
//! - `MLEX_INCLUDE_LARGE_MODELS=1`: also return `local_only` models from
//!   [`registry`] (CI never sets this).

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One discovered, classified model checkpoint.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub dir: PathBuf,
    pub repo_id: String,
    pub family: String,
    pub weights_bytes: u64,
    pub peak_rss_bytes: Option<u64>,
    pub ci_safe: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct MemoryCache {
    /// Keyed by `"{dir}|{weights_bytes}"` so a re-download/resize
    /// invalidates the cached measurement.
    entries: HashMap<String, u64>,
}

fn repo_root() -> PathBuf {
    // crates/mlex -> crates -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("mlex lives at <repo>/crates/mlex")
        .to_path_buf()
}

fn models_dir() -> PathBuf {
    std::env::var("MLEX_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("models"))
}

fn max_model_gb() -> f64 {
    std::env::var("MLEX_MAX_MODEL_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5.0)
}

fn include_large_models() -> bool {
    std::env::var("MLEX_INCLUDE_LARGE_MODELS").as_deref() == Ok("1")
}

fn cache_path() -> PathBuf {
    repo_root()
        .join("target")
        .join("mlex-test-cache")
        .join("model-memory.json")
}

fn load_cache() -> MemoryCache {
    fs::read_to_string(cache_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_cache(cache: &MemoryCache) {
    if let Some(parent) = cache_path().parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        let _ = fs::write(cache_path(), json);
    }
}

/// Total size on disk of every `*.safetensors` file directly inside `dir`
/// (following symlinks, as the HF cache layout uses them).
fn safetensors_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .filter_map(|e| fs::metadata(e.path()).ok())
        .map(|m| m.len())
        .sum()
}

/// Peak resident set size (bytes) of *this* process so far, via
/// `getrusage(RUSAGE_SELF, ...)`. On macOS, `ru_maxrss` is already in
/// bytes (unlike Linux, where it's kilobytes).
fn current_peak_rss_bytes() -> u64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        #[cfg(target_os = "macos")]
        {
            usage.ru_maxrss as u64
        }
        #[cfg(not(target_os = "macos"))]
        {
            usage.ru_maxrss as u64 * 1024
        }
    }
}

/// Measure peak RSS for loading `dir` + running one short forward pass, by
/// spawning a fresh subprocess (so measurements don't accumulate across
/// models within one `cargo test` process). Requires the `measure_memory`
/// example binary to be built (`cargo build --release --example
/// measure_memory -p mlex`); falls back to `None` (treated as `local_only`)
/// if the example can't be run.
fn measure_peak_rss(dir: &Path) -> Option<u64> {
    let exe = repo_root()
        .join("target")
        .join("release")
        .join("examples")
        .join("measure_memory");
    let exe = if exe.exists() {
        exe
    } else {
        repo_root()
            .join("target")
            .join("debug")
            .join("examples")
            .join("measure_memory")
    };
    if !exe.exists() {
        return None;
    }
    let output = std::process::Command::new(exe).arg(dir).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

fn model_type_of(config_path: &Path) -> Option<String> {
    let text = fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    json.get("model_type")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// Discover every locally-downloaded model under `MLEX_MODELS_DIR`,
/// measuring (and caching) real peak memory as needed, and classifying
/// each as CI-safe or local-only against `MLEX_MAX_MODEL_GB`.
///
/// By default only `ci_safe` models are returned; set
/// `MLEX_INCLUDE_LARGE_MODELS=1` to also get `local_only` ones.
pub fn registry() -> Vec<ModelInfo> {
    let root = models_dir();
    let max_bytes = (max_model_gb() * 1024.0 * 1024.0 * 1024.0) as u64;
    let mut cache = load_cache();
    let mut cache_dirty = false;
    let mut out = Vec::new();

    let Ok(top) = fs::read_dir(&root) else {
        return out;
    };
    for entry in top.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("models--") {
            continue;
        }
        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = fs::read_dir(&snapshots_dir) else {
            continue;
        };
        for snap in snapshots.filter_map(|e| e.ok()) {
            let dir = snap.path();
            let config_path = dir.join("config.json");
            if !config_path.exists() {
                continue;
            }
            let family = model_type_of(&config_path).unwrap_or_else(|| "unknown".to_string());
            let weights_bytes = safetensors_bytes(&dir);
            let repo_id = name.trim_start_matches("models--").replacen("--", "/", 1);

            // Cheap pre-filter: something more than 2x the ceiling on disk
            // has no realistic path to fitting in memory - skip measuring.
            let ci_safe;
            let peak_rss_bytes;
            if weights_bytes > max_bytes.saturating_mul(2) {
                ci_safe = false;
                peak_rss_bytes = None;
            } else {
                let cache_key = format!("{}|{}", dir.display(), weights_bytes);
                let measured = match cache.entries.get(&cache_key) {
                    Some(&bytes) => Some(bytes),
                    None => {
                        let m = measure_peak_rss(&dir);
                        if let Some(bytes) = m {
                            cache.entries.insert(cache_key, bytes);
                            cache_dirty = true;
                        }
                        m
                    }
                };
                peak_rss_bytes = measured;
                ci_safe = measured.map(|rss| rss <= max_bytes).unwrap_or(false);
            }

            out.push(ModelInfo {
                dir,
                repo_id,
                family,
                weights_bytes,
                peak_rss_bytes,
                ci_safe,
            });
        }
    }

    if cache_dirty {
        save_cache(&cache);
    }

    if include_large_models() {
        out
    } else {
        out.into_iter().filter(|m| m.ci_safe).collect()
    }
}

/// [`registry`] filtered to one architecture family (`config.json`'s
/// `model_type`, e.g. `"qwen3_5"`, `"dhara_ar"`).
pub fn registry_for_family(family: &str) -> Vec<ModelInfo> {
    registry()
        .into_iter()
        .filter(|m| m.family == family)
        .collect()
}

/// True if `dir`'s `config.json` advertises the given modality capability
/// (e.g. `"vision_config"`, `"audio_config"`, `"image_token_id"`) - used to
/// gate multi-modal test cases without a hardcoded model list.
/// Tensor-name prefix a checkpoint must actually ship weights under for a
/// given `config.json` sub-dict to be a *real* (not just declared)
/// capability - mirrors `Gemma4Model::load_vision`/`load_audio`'s own
/// weight-presence gate, so config-only "unified" checkpoints that declare
/// `vision_config`/`audio_config` for architecture-class metadata but were
/// distributed text-only (no tower weights) aren't mistaken for capable
/// multimodal checkpoints.
/// Any one of these tensor-name prefixes present is enough to call `key`'s
/// capability "real" (not just declared): either the classic transformer
/// tower, or the encoder-free "unified" path's patch/window embedder -
/// `embed_vision.`/`embed_audio.` alone is the simplest and most reliable
/// signal (both encoder shapes always route through it - see
/// `Gemma4Model::load_vision`/`load_audio`), listed alongside the more
/// specific tower/embedder prefixes for clarity.
fn required_weight_prefixes(key: &str) -> Option<&'static [&'static str]> {
    match key {
        "vision_config" => Some(&["vision_tower.", "vision_embedder.", "embed_vision."]),
        "audio_config" => Some(&["audio_tower.", "embed_audio."]),
        _ => None,
    }
}

/// True iff `config.json` declares `key` *and* (when `key` maps to known
/// weight prefixes) the checkpoint's safetensors actually carry matching
/// weights - a config-only declaration alone isn't enough (some OptiQ
/// checkpoints declare `vision_config`/`audio_config` for architecture-class
/// metadata while shipping the unified encoder in a sidecar shard, or
/// nothing at all).
pub fn has_capability(dir: &Path, key: &str) -> bool {
    let Ok(text) = fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    if json.get(key).is_none() {
        return false;
    }
    match required_weight_prefixes(key) {
        Some(prefixes) => prefixes.iter().any(|p| any_tensor_has_prefix(dir, p)),
        None => true,
    }
}

/// Scan every `*.safetensors` shard under `dir` (sharded index, single
/// file, and any sidecar shard the index doesn't mention - mirrors
/// `crate::weights::load_all`'s file discovery) for a tensor name starting
/// with `prefix`, without loading tensor data.
fn any_tensor_has_prefix(dir: &Path, prefix: &str) -> bool {
    let mut files: Vec<String> = Vec::new();
    let index_path = dir.join("model.safetensors.index.json");
    if let Ok(text) = fs::read_to_string(&index_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(weight_map) = json.get("weight_map").and_then(|v| v.as_object()) {
                files.extend(
                    weight_map
                        .values()
                        .filter_map(|v| v.as_str().map(String::from)),
                );
                files.sort();
                files.dedup();
            }
        }
    } else {
        files.push("model.safetensors".to_string());
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for name in entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str().map(String::from))
        {
            if name.ends_with(".safetensors") && !files.contains(&name) {
                files.push(name);
            }
        }
    }

    files
        .iter()
        .any(|file| safetensors_header_has_prefix(&dir.join(file), prefix))
}

/// Read one safetensors file's JSON header and check for a tensor name
/// starting with `prefix`.
fn safetensors_header_has_prefix(path: &Path, prefix: &str) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    if bytes.len() < 8 {
        return false;
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + header_len {
        return false;
    }
    let Ok(header) = serde_json::from_slice::<serde_json::Value>(&bytes[8..8 + header_len]) else {
        return false;
    };
    let Some(obj) = header.as_object() else {
        return false;
    };
    obj.keys()
        .filter(|k| *k != "__metadata__")
        .any(|k| k.starts_with(prefix))
}
