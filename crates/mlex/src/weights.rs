//! Checkpoint loading: single-file and sharded safetensors.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;

use crate::array::Array;
use crate::error::{check, install_error_handler, Error, Result};
use crate::stream::cpu_stream;

/// Load every tensor from all `*.safetensors` shards under `model_dir`.
///
/// Uses `model.safetensors.index.json` when present, otherwise loads the
/// single `model.safetensors` file. Additionally loads any OTHER
/// `*.safetensors` file sitting in `model_dir` that the index doesn't
/// mention (e.g. OptiQ checkpoints ship vision/audio tower weights in a
/// sidecar `optiq_vision.safetensors` that `model.safetensors.index.json`
/// never references) - discovered generically by directory listing rather
/// than hardcoding that filename, so any future sidecar shard works too.
pub fn load_all(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let mut files: Vec<String> = if index_path.exists() {
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)
            .map_err(|e| Error::Config(format!("bad safetensors index: {e}")))?;
        let map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| Error::Config("safetensors index missing weight_map".into()))?;
        let mut names: Vec<String> = map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        names.sort();
        names.dedup();
        names
    } else {
        vec!["model.safetensors".to_string()]
    };

    if let Ok(entries) = std::fs::read_dir(model_dir) {
        let mut extra: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str().map(String::from))
            .filter(|name| name.ends_with(".safetensors") && !files.contains(name))
            .collect();
        extra.sort();
        files.extend(extra);
    }

    let mut all = HashMap::new();
    for file in files {
        let path = model_dir.join(&file);
        if !path.exists() {
            return Err(Error::Model(format!(
                "checkpoint shard not found: {}",
                path.display()
            )));
        }
        load_file(&path, &mut all)?;
    }
    Ok(all)
}

/// Load one safetensors file, inserting tensors into `out`.
pub fn load_file(path: &Path, out: &mut HashMap<String, Array>) -> Result<()> {
    install_error_handler();
    let c_path =
        CString::new(path.to_str().ok_or_else(|| {
            Error::Model(format!("non-utf8 checkpoint path: {}", path.display()))
        })?)
        .unwrap();

    unsafe {
        let mut tensors = crate::sys::mlx_map_string_to_array_new();
        let mut metadata = crate::sys::mlx_map_string_to_string_new();
        let status = crate::sys::mlx_load_safetensors(
            &mut tensors,
            &mut metadata,
            c_path.as_ptr(),
            cpu_stream(),
        );
        crate::sys::mlx_map_string_to_string_free(metadata);
        if status != 0 {
            crate::sys::mlx_map_string_to_array_free(tensors);
            check(status)?;
        }

        let it = crate::sys::mlx_map_string_to_array_iterator_new(tensors);
        loop {
            let mut key: *const std::ffi::c_char = std::ptr::null();
            let mut value = Array::new_handle();
            let status =
                crate::sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value, it);
            if status != 0 {
                // End of iteration.
                crate::sys::mlx_array_free(value);
                break;
            }
            let name = CStr::from_ptr(key).to_string_lossy().into_owned();
            let arr = Array::from_raw(value);
            // `mlx_load_safetensors` produces arrays backed by the `Load`
            // primitive, which only has a CPU eval kernel. Materialize each
            // tensor immediately so later GPU ops never have to evaluate a
            // graph with a dangling `Load` node.
            arr.eval()?;
            out.insert(name, arr);
        }
        crate::sys::mlx_map_string_to_array_iterator_free(it);
        crate::sys::mlx_map_string_to_array_free(tensors);
    }
    Ok(())
}
