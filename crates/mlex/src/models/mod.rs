//! Concrete model architectures and the loader that dispatches between them
//! based on `config.json`'s `model_type`.

pub mod base;
pub mod cache;
pub mod config;
pub mod dhara;
pub mod gated_delta;
pub mod gemma4;
pub mod mamba2;
pub mod moe;
pub mod nemotron;
pub mod qwen2;
pub mod qwen3;
pub mod qwen3_5;

use std::path::Path;

use serde_json::Value;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::media::audio::ProcessedAudio;
use crate::media::image::ProcessedImage;
use crate::nn::WeightMap;
use crate::quant::Quantization;
use crate::weights;
use cache::LayerCache;

/// A loaded, ready-to-run causal language model.
///
/// New architectures are added as variants here; `forward` dispatches to
/// the concrete implementation. Keeping one enum (rather than a trait
/// object) avoids `dyn`-safety friction around the differing per-layer
/// cache types each architecture needs.
pub enum Model {
    Qwen2(qwen2::Qwen2Model),
    Qwen3(qwen3::Qwen3Model),
    Qwen35(qwen3_5::Qwen35Model),
    Gemma4(gemma4::Gemma4Model),
    NemotronH(nemotron::NemotronModel),
    Dhara(dhara::DharaModel),
}

impl Model {
    /// Load a model directory (expects `config.json` + safetensors shards).
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_json: Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .map_err(|e| Error::Config(format!("bad config.json: {e}")))?;

        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("qwen3");

        let tensors = weights::load_all(model_dir)?;
        let quant = Quantization::from_config(&config_json)?;
        let mut weight_map = WeightMap::new(tensors, quant);

        match model_type {
            "qwen2" => {
                let tie = config_json
                    .get("tie_word_embeddings")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                qwen2::sanitize(&mut weight_map, tie);
                let model = qwen2::Qwen2Model::load(weight_map, &config_json)?;
                Ok(Model::Qwen2(model))
            }
            "qwen3" => {
                let tie = config_json
                    .get("tie_word_embeddings")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                qwen3::sanitize(&mut weight_map, tie);
                let model = qwen3::Qwen3Model::load(weight_map, &config_json)?;
                Ok(Model::Qwen3(model))
            }
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => {
                gemma4::sanitize(&mut weight_map);
                let model = gemma4::Gemma4Model::load(weight_map, &config_json)?;
                Ok(Model::Gemma4(model))
            }
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => {
                let text_cfg = config::text_config(&config_json);
                let num_hidden_layers = config::require_i32(text_cfg, "num_hidden_layers")?;
                let num_experts = config::get_i32(text_cfg, "num_experts", 0);
                qwen3_5::sanitize(&mut weight_map, num_hidden_layers, num_experts);
                let model = qwen3_5::Qwen35Model::load(weight_map, &config_json)?;
                Ok(Model::Qwen35(model))
            }
            "nemotron_h" => {
                nemotron::sanitize(&mut weight_map);
                let model = nemotron::NemotronModel::load(weight_map, &config_json)?;
                Ok(Model::NemotronH(model))
            }
            "llama" => {
                // Vanilla Llama-style GQA checkpoints (e.g. MiniCPM5) are
                // structurally identical to our Qwen2 implementation
                // (RoPE, SwiGLU MLP, RMSNorm, optional qkv bias, optional
                // tied lm_head) - reuse it rather than duplicating code.
                let tie = config_json
                    .get("tie_word_embeddings")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                qwen2::sanitize(&mut weight_map, tie);
                let model = qwen2::Qwen2Model::load(weight_map, &config_json)?;
                Ok(Model::Qwen2(model))
            }
            "dhara_ar" => {
                let tie = config_json
                    .get("tie_word_embeddings")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                dhara::sanitize(&mut weight_map, tie);
                let model = dhara::DharaModel::load(weight_map, &config_json)?;
                Ok(Model::Dhara(model))
            }
            other => Err(Error::Model(format!(
                "unsupported model_type '{other}' (supported: qwen2, qwen3, qwen3_5, qwen3_5_moe, gemma4, gemma4_unified, nemotron_h, llama, dhara_ar)"
            ))),
        }
    }

    pub fn new_caches(&self) -> Vec<LayerCache> {
        match self {
            Model::Qwen2(m) => m.new_caches(),
            Model::Qwen3(m) => m.new_caches(),
            Model::Qwen35(m) => m.new_caches(),
            Model::Gemma4(m) => m.new_caches(),
            Model::NemotronH(m) => m.new_caches(),
            Model::Dhara(m) => m.new_caches(),
        }
    }

    /// Run one forward pass over `input_ids` (`[B, L]`), returning logits
    /// (`[B, L, vocab]`).
    pub fn forward(&self, input_ids: &Array, caches: &mut [LayerCache]) -> Result<Array> {
        match self {
            Model::Qwen2(m) => m.forward(input_ids, caches),
            Model::Qwen3(m) => m.forward(input_ids, caches),
            Model::Qwen35(m) => m.forward(input_ids, caches),
            Model::Gemma4(m) => m.forward(input_ids, caches),
            Model::NemotronH(m) => m.forward(input_ids, caches),
            Model::Dhara(m) => m.forward(input_ids, caches),
        }
    }

    /// Which tool-call output convention this architecture's chat template
    /// uses (see `crate::tools`).
    pub fn tool_call_format(&self) -> crate::tools::ToolCallFormat {
        use crate::tools::ToolCallFormat;
        match self {
            Model::Gemma4(_) => ToolCallFormat::Gemma,
            Model::Qwen2(_) | Model::Qwen3(_) | Model::Qwen35(_) | Model::NemotronH(_) => {
                ToolCallFormat::Hermes
            }
            Model::Dhara(_) => ToolCallFormat::Hermes,
        }
    }

    /// Debug-only helper (see `NemotronModel::debug_layer_stats`).
    pub fn debug_nemotron_layer_stats(&self, input_ids: &Array) -> Result<Vec<(f32, f32)>> {
        match self {
            Model::NemotronH(m) => m.debug_layer_stats(input_ids),
            _ => Err(Error::Model(
                "debug_nemotron_layer_stats: not a NemotronH model".into(),
            )),
        }
    }

    /// Whether this model was loaded with image support (a `vision_config`
    /// in `config.json` plus matching vision tower weights).
    pub fn supports_images(&self) -> bool {
        match self {
            Model::Gemma4(m) => m.supports_images(),
            Model::Qwen35(m) => m.supports_images(),
            _ => false,
        }
    }

    /// `(patch_size, max_soft_tokens, pooling_kernel_size)` for
    /// [`crate::media::image::preprocess_image_bytes`], or `None` if this
    /// model has no image support.
    pub fn image_processing_params(&self) -> Option<(i32, i32, i32)> {
        match self {
            Model::Gemma4(m) => m.image_processing_params(),
            Model::Qwen35(m) => m.image_processing_params(),
            _ => None,
        }
    }

    /// `(image_token_id, boi_token_id, eoi_token_id)`, or `None` if this
    /// model has no image support.
    pub fn image_token_ids(&self) -> Option<(u32, u32, u32)> {
        match self {
            Model::Gemma4(m) => m.image_token_ids(),
            Model::Qwen35(m) => m.image_token_ids(),
            _ => None,
        }
    }

    /// TEMP debug hook.
    pub fn debug_vision_forward(&self, image: &ProcessedImage) -> Result<Vec<f32>> {
        match self {
            Model::Gemma4(m) => m.debug_vision_forward(image),
            _ => Err(Error::Model("nope".into())),
        }
    }

    /// Whether this model was loaded with audio support (an `audio_config`
    /// in `config.json` plus matching audio tower weights).
    pub fn supports_audio(&self) -> bool {
        match self {
            Model::Gemma4(m) => m.supports_audio(),
            _ => false,
        }
    }

    /// `(audio_token_id, boa_token_id, eoa_token_id)`, or `None` if this
    /// model has no audio support.
    pub fn audio_token_ids(&self) -> Option<(u32, u32, u32)> {
        match self {
            Model::Gemma4(m) => m.audio_token_ids(),
            _ => None,
        }
    }

    /// Raw PCM samples per audio token for the encoder-free "unified"
    /// audio path (see `crate::media::audio::preprocess_audio_bytes_raw`),
    /// or `None` if this model has no audio support or uses the classic
    /// mel-spectrogram tower instead.
    pub fn audio_samples_per_token(&self) -> Option<i32> {
        match self {
            Model::Gemma4(m) => m.audio_samples_per_token(),
            _ => None,
        }
    }

    /// The chat template's video placeholder token id, or `None` if this
    /// model has no vision support (video frames reuse the vision tower).
    pub fn video_token_id(&self) -> Option<u32> {
        match self {
            Model::Gemma4(m) => m.video_token_id(),
            Model::Qwen35(m) => m.video_token_id(),
            _ => None,
        }
    }

    /// Run one forward pass over `input_ids` (`[B, L]`), splicing `images`'
    /// projected vision features in at `image_token_id` placeholder
    /// positions before the decoder stack. Errors if this model has no
    /// image support.
    pub fn forward_with_images(
        &self,
        input_ids: &Array,
        images: &[ProcessedImage],
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        self.forward_with_media(input_ids, images, &[], caches)
    }

    /// Run one forward pass over `input_ids` (`[B, L]`), splicing image
    /// and/or audio features in at their placeholder positions before the
    /// decoder stack (video frames arrive as ordinary `images` entries).
    /// Errors if a modality is supplied that this model doesn't support.
    pub fn forward_with_media(
        &self,
        input_ids: &Array,
        images: &[ProcessedImage],
        audios: &[ProcessedAudio],
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        match self {
            Model::Gemma4(m) => m.forward_with_media(input_ids, images, audios, caches),
            Model::Qwen35(m) => {
                if !audios.is_empty() {
                    return Err(Error::Model(
                        "qwen3.5: model has no audio support (no audio_config)".into(),
                    ));
                }
                m.forward_with_media(input_ids, images, caches)
            }
            _ => Err(Error::Model(
                "forward_with_media: model has no multimodal support".into(),
            )),
        }
    }
}
