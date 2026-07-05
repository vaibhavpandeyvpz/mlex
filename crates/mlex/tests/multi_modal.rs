//! Multi-modal (image) chat against a real, locally-downloaded Gemma4
//! checkpoint + a real sample image. Skips gracefully (printing a clear
//! reason, not failing the suite) when no CI-safe model with vision
//! support is available under `MLEX_MODELS_DIR` - mirrors the "no CI-safe
//! model could do X" skip pattern used throughout `tool_calling.rs`.
//!
//! Run against the memory-constrained single-model directory:
//!   MLEX_MODELS_DIR=/tmp/mlex-gemma4-only cargo test --release -p mlex \
//!     --test multi_modal -- --test-threads=1 --nocapture

mod common;

use std::path::PathBuf;

use mlex::generate::{GenerateOptions, Session};
use mlex::sampling::SamplingConfig;
use mlex::tokenizer::ChatMessage;

fn sample_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("samples")
        .join(name)
}

fn greedy_opts(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_tokens,
        sampling: SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Loose "not degenerate" sanity check for generated text: non-empty and
/// not just one token repeated over and over. Mirrors the soft heuristics
/// `generate_basic.rs`/`tool_calling.rs` use rather than matching exact
/// wording (model output for a given prompt is not something this suite
/// should pin down verbatim).
fn looks_non_degenerate(text: &str, generated_ids: &[u32]) -> bool {
    if text.trim().is_empty() {
        return false;
    }
    if generated_ids.len() < 6 {
        return true;
    }
    let unique_words: std::collections::HashSet<&str> = text.split_whitespace().collect();
    unique_words.len() >= 3
}

#[test]
fn gemma4_describes_a_real_image() {
    let models = common::registry_for_family("gemma4");
    let Some(model) = models
        .iter()
        .find(|m| common::has_capability(&m.dir, "vision_config"))
    else {
        println!(
            "[multi_modal] skip: no CI-safe gemma4 model with vision_config found under MLEX_MODELS_DIR \
             ({:?}); set MLEX_MODELS_DIR to a directory containing e.g. mlx-community/gemma-4-E2B-it-qat-4bit \
             to run this test.",
            std::env::var("MLEX_MODELS_DIR")
        );
        return;
    };

    let session = Session::load(&model.dir).expect("failed to load gemma4 vision model");

    let image_bytes =
        std::fs::read(sample_path("image1.jpg")).expect("failed to read samples/image1.jpg");
    let messages = vec![ChatMessage::user_with_image(
        "Describe this image in one short sentence.",
        image_bytes,
    )];

    let (prompt_ids, media) = session
        .encode_chat_with_media(&messages)
        .expect("encode_chat_with_media failed");
    assert!(
        !media.images.is_empty(),
        "expected at least one processed image"
    );
    assert!(
        media.images[0].num_soft_tokens > 0,
        "processed image has zero soft tokens"
    );
    println!(
        "[multi_modal] image1.jpg -> patch grid {}x{}, {} soft tokens, prompt length {} tokens",
        media.images[0].patch_h,
        media.images[0].patch_w,
        media.images[0].num_soft_tokens,
        prompt_ids.len()
    );

    let mut caches = session.debug_new_caches();
    let mut text = String::new();
    let generated = session
        .generate_with_media(&prompt_ids, &media, &mut caches, greedy_opts(64), |tok| {
            text.push_str(&tok.text);
            true
        })
        .expect("generate_with_media failed");

    println!("=== gemma4 image description ===\n{text}\n=================================");

    assert!(
        !generated.is_empty(),
        "{}: no tokens generated for image prompt",
        model.repo_id
    );
    assert!(
        generated.iter().all(|&id| id != u32::MAX),
        "{}: generation produced a sentinel/invalid token id",
        model.repo_id
    );
    assert!(
        looks_non_degenerate(&text, &generated),
        "{}: generated text looks degenerate: {text:?}",
        model.repo_id
    );
}

/// Real audio transcription against the same multimodal checkpoint: sends
/// `samples/audio.mp3` (a deterministic speech recording from
/// samplelib.com) with a transcription prompt. Unlike image description,
/// transcription accuracy is meaningfully checkable: the clip's known
/// transcript is "Welcome to samplelib.com, a free online resource for
/// downloading sample files...", so we assert the output mentions at
/// least some of the recording's distinctive vocabulary rather than just
/// "is non-empty".
#[test]
fn gemma4_transcribes_a_real_audio_clip() {
    let models = common::registry_for_family("gemma4");
    let Some(model) = models
        .iter()
        .find(|m| common::has_capability(&m.dir, "audio_config"))
    else {
        println!("[multi_modal] skip: no CI-safe gemma4 model with audio_config found under MLEX_MODELS_DIR");
        return;
    };

    let session = Session::load(&model.dir).expect("failed to load gemma4 audio model");
    if !session.supports_audio() {
        println!(
            "[multi_modal] skip: {} declares audio_config but shipped no audio tower weights",
            model.repo_id
        );
        return;
    }

    let audio_bytes =
        std::fs::read(sample_path("audio.mp3")).expect("failed to read samples/audio.mp3");
    let messages = vec![ChatMessage::user_with_audio(
        "Transcribe this audio.",
        audio_bytes,
    )];

    let (prompt_ids, media) = session
        .encode_chat_with_media(&messages)
        .expect("encode_chat_with_media failed");
    assert!(
        !media.audios.is_empty(),
        "expected at least one processed audio clip"
    );
    let soft_tokens = media.audios[0].num_soft_tokens();
    assert!(soft_tokens > 0, "processed audio has zero soft tokens");
    println!(
        "[multi_modal] audio.mp3 -> {} mel chunk(s), {} soft tokens, prompt length {} tokens",
        media.audios[0].chunks.len(),
        soft_tokens,
        prompt_ids.len()
    );

    let mut caches = session.debug_new_caches();
    let mut text = String::new();
    let generated = session
        .generate_with_media(&prompt_ids, &media, &mut caches, greedy_opts(128), |tok| {
            text.push_str(&tok.text);
            true
        })
        .expect("generate_with_media failed for audio");

    println!("=== gemma4 audio transcription ===\n{text}\n===================================");

    assert!(
        !generated.is_empty(),
        "{}: no tokens generated for audio prompt",
        model.repo_id
    );
    assert!(
        looks_non_degenerate(&text, &generated),
        "{}: generated text looks degenerate: {text:?}",
        model.repo_id
    );
    // The clip's actual content is deterministic speech; a working audio
    // path must pick up at least some of its distinctive vocabulary.
    let lower = text.to_lowercase();
    let markers = ["samplelib", "sample", "download", "file", "format", "test"];
    let hits = markers.iter().filter(|m| lower.contains(*m)).count();
    assert!(
        hits >= 2,
        "{}: transcription does not resemble the known clip content (matched {hits}/6 markers): {text:?}",
        model.repo_id
    );
}

/// Real video description: uniform frames from `samples/video1.mp4`
/// through the vision tower, spliced as a frame sequence.
#[test]
fn gemma4_describes_a_real_video() {
    let models = common::registry_for_family("gemma4");
    let Some(model) = models
        .iter()
        .find(|m| common::has_capability(&m.dir, "vision_config"))
    else {
        println!("[multi_modal] skip: no CI-safe gemma4 vision model found under MLEX_MODELS_DIR");
        return;
    };

    let session = Session::load(&model.dir).expect("failed to load gemma4 vision model");

    let video_bytes =
        std::fs::read(sample_path("video1.mp4")).expect("failed to read samples/video1.mp4");
    let messages = vec![ChatMessage::user_with_video(
        "Describe what happens in this video in one or two sentences.",
        video_bytes,
    )];

    let (prompt_ids, media) = session
        .encode_chat_with_media(&messages)
        .expect("encode_chat_with_media failed");
    assert!(
        !media.images.is_empty(),
        "expected at least one extracted video frame"
    );
    println!(
        "[multi_modal] video1.mp4 -> {} frames, {} soft tokens/frame, prompt length {} tokens",
        media.images.len(),
        media.images[0].num_soft_tokens,
        prompt_ids.len()
    );

    let mut caches = session.debug_new_caches();
    let mut text = String::new();
    let generated = session
        .generate_with_media(&prompt_ids, &media, &mut caches, greedy_opts(96), |tok| {
            text.push_str(&tok.text);
            true
        })
        .expect("generate_with_media failed for video");

    println!("=== gemma4 video description ===\n{text}\n=================================");

    assert!(
        !generated.is_empty(),
        "{}: no tokens generated for video prompt",
        model.repo_id
    );
    assert!(
        looks_non_degenerate(&text, &generated),
        "{}: generated text looks degenerate: {text:?}",
        model.repo_id
    );
}

/// Multi-turn conversation with a multimodal first turn: the image is
/// preprocessed and spliced once, then a text-only follow-up reuses the
/// persistent prompt cache (only the new suffix runs through the model).
#[test]
fn gemma4_multi_turn_conversation_with_an_image() {
    let models = common::registry_for_family("gemma4");
    let Some(model) = models
        .iter()
        .find(|m| common::has_capability(&m.dir, "vision_config"))
    else {
        println!("[multi_modal] skip: no CI-safe gemma4 vision model found under MLEX_MODELS_DIR");
        return;
    };

    let session = Session::load(&model.dir).expect("failed to load gemma4 vision model");

    let image_bytes =
        std::fs::read(sample_path("image1.jpg")).expect("failed to read samples/image1.jpg");
    let mut messages = vec![ChatMessage::user_with_image(
        "Describe this image in one short sentence.",
        image_bytes,
    )];
    let first = session
        .generate_cached(&messages, None, greedy_opts(48), |_| true)
        .expect("first (image) turn failed")
        .text;
    println!("=== turn 1 (image) ===\n{first}");
    messages.push(ChatMessage::assistant(&first));

    messages.push(ChatMessage::user(
        "What main colors are in the image you just described?",
    ));
    let second = session
        .generate_cached(&messages, None, greedy_opts(48), |_| true)
        .expect("second (text follow-up) turn failed")
        .text;
    println!("=== turn 2 (follow-up) ===\n{second}\n=======================");
    messages.push(ChatMessage::assistant(&second));

    assert!(
        !first.trim().is_empty(),
        "first multimodal turn produced no text"
    );
    assert!(
        !second.trim().is_empty(),
        "text follow-up after an image turn produced no text"
    );
    assert_eq!(
        messages.len(),
        4,
        "expected user/assistant/user/assistant transcript"
    );
}

/// A text-only prompt against the same vision-capable checkpoint must still
/// work exactly as before (the vision tower must not perturb the text-only
/// path at all).
#[test]
fn gemma4_vision_model_still_handles_text_only_prompts() {
    let models = common::registry_for_family("gemma4");
    let Some(model) = models
        .iter()
        .find(|m| common::has_capability(&m.dir, "vision_config"))
    else {
        println!("[multi_modal] skip: no CI-safe gemma4 vision model found under MLEX_MODELS_DIR");
        return;
    };

    let session = Session::load(&model.dir).expect("failed to load gemma4 vision model");
    let messages = vec![ChatMessage::user("Count from one to three.")];
    let prompt_ids = session.encode_chat(&messages).expect("encode_chat failed");

    let mut caches = session.debug_new_caches();
    let mut text = String::new();
    let generated = session
        .generate_with_caches(&prompt_ids, &mut caches, greedy_opts(16), |tok| {
            text.push_str(&tok.text);
            true
        })
        .expect("text-only generation failed on a vision-capable checkpoint");

    println!("=== gemma4 text-only (vision-capable checkpoint) ===\n{text}\n=====================================");
    assert!(
        !generated.is_empty(),
        "{}: no tokens generated for text-only prompt",
        model.repo_id
    );
}
