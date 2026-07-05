//! Raw FFI bindings to Apple MLX via the official `mlx-c` C API.
//!
//! Everything in this module is `unsafe` and mirrors the C API exactly.
//! It is private to this crate; the rest of `mlex` builds a safe, idiomatic
//! interface on top of it (see [`crate::array`], [`crate::ops`], ...).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(dead_code)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Absolute path of the `mlx.metallib` produced by the CMake build, if any.
/// Useful for colocating the metallib next to a distributed binary.
pub const BUILD_METALLIB_PATH: Option<&str> = option_env!("MLEX_METALLIB_PATH");
