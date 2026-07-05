//! Media preprocessing shared by multi-modal architectures: turning raw
//! image/audio/video bytes into the tensors a vision/audio tower expects.
//!
//! Kept independent of any one architecture (`crate::models::gemma4`, and
//! future VLMs) since the resize/patchify/frame-extraction math here is
//! largely model-family-agnostic; per-architecture code only supplies the
//! numeric parameters (patch size, pooling, token budget, ...).

pub mod audio;
pub mod image;
pub mod video;
