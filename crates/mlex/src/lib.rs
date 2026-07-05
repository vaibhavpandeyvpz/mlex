//! `mlex`: a safe, idiomatic Rust runtime for running MLX language models —
//! including the full range of quantized checkpoints produced by the MLX
//! community (affine 2/3/4/5/6/8-bit, mxfp4, mxfp8, nvfp4, and mixed
//! per-layer precision such as OptiQ or Google QAT exports).
//!
//! Layering:
//! - [`array`] / [`ops`] / [`stream`]: thin safe wrappers over the private
//!   `sys` module (raw `mlx-c` FFI bindings, built from vendored MLX/mlx-c
//!   C++ sources via `build.rs`).
//! - [`quant`]: parses the `quantization` section of `config.json` and
//!   resolves per-layer bit-widths, mirroring mlx-lm's loader semantics.
//! - [`nn`] / [`weights`]: generic building blocks (linear, embedding,
//!   norm) that transparently load dense or quantized weights.
//! - [`models`]: concrete architectures (Qwen3, Qwen3.5 (+MoE), Gemma4).
//! - [`tokenizer`] / [`sampling`] / [`generate`]: text I/O and the
//!   generation loop shared by every architecture.

pub mod array;
pub mod error;
pub mod generate;
pub mod media;
pub mod models;
pub mod nn;
pub mod ops;
pub mod prompt_cache;
pub mod quant;
pub mod reasoning;
pub mod sampling;
pub mod stream;
pub mod streaming;
mod sys;
pub mod tokenizer;
pub mod tools;
pub mod weights;

pub use array::{Array, Dtype};
pub use error::{Error, Result};
