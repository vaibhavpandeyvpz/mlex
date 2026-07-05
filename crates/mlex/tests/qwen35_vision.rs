//! Multi-modal (image) chat against a real, locally-downloaded Qwen3.5-VL
//! checkpoint (OptiQ checkpoints ship a `vision_tower.*` sidecar) + a real
//! sample image. Skips gracefully (printing a clear reason, not failing the
//! suite) when no CI-safe model with vision support is available under
//! `MLEX_MODELS_DIR` - mirrors `multi_modal.rs`'s Gemma4 coverage.
//!
//! Run against the memory-constrained single-model directory:
//!   MLEX_MODELS_DIR=/tmp/mlex-qwen35-08b-only cargo test --release -p mlex \
//!     --test qwen35_vision -- --test-threads=1 --nocapture

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
        enable_thinking: Some(false),
        ..Default::default()
    }
}

/// Loose "not degenerate" sanity check for generated text: non-empty and
/// not just one token repeated over and over.
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

fn find_vision_qwen35() -> Option<common::ModelInfo> {
    common::registry_for_family("qwen3_5")
        .into_iter()
        .find(|m| common::has_capability(&m.dir, "vision_config"))
}

#[test]
fn qwen35_describes_a_real_image() {
    let Some(model) = find_vision_qwen35() else {
        println!(
            "[qwen35_vision] skip: no CI-safe qwen3_5 model with vision_config found under MLEX_MODELS_DIR \
             ({:?}); set MLEX_MODELS_DIR to a directory containing e.g. mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
             to run this test.",
            std::env::var("MLEX_MODELS_DIR")
        );
        return;
    };

    let session = Session::load(&model.dir).expect("failed to load qwen3.5 vision model");
    assert!(
        session.supports_images(),
        "{}: declares vision_config but has no vision tower loaded",
        model.repo_id
    );

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
        "[qwen35_vision] image1.jpg -> patch grid {}x{}, {} soft tokens, prompt length {} tokens",
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

    println!("=== qwen3.5 image description ===\n{text}\n==================================");

    assert!(
        !generated.is_empty(),
        "{}: no tokens generated for image prompt",
        model.repo_id
    );
    assert!(
        looks_non_degenerate(&text, &generated),
        "{}: generated text looks degenerate: {text:?}",
        model.repo_id
    );
}

/// A text-only prompt against the same vision-capable checkpoint must still
/// work exactly as before (the vision tower must not perturb the text-only
/// path at all).
#[test]
fn qwen35_vision_model_still_handles_text_only_prompts() {
    let Some(model) = find_vision_qwen35() else {
        println!(
            "[qwen35_vision] skip: no CI-safe qwen3_5 vision model found under MLEX_MODELS_DIR"
        );
        return;
    };

    let session = Session::load(&model.dir).expect("failed to load qwen3.5 vision model");
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

    println!("=== qwen3.5 text-only (vision-capable checkpoint) ===\n{text}\n======================================");
    assert!(
        !generated.is_empty(),
        "{}: no tokens generated for text-only prompt",
        model.repo_id
    );
}

/// Multi-turn conversation with a multimodal first turn: the image is
/// preprocessed and spliced once, then a text-only follow-up reuses the
/// persistent prompt cache (only the new suffix runs through the model).
#[test]
fn qwen35_multi_turn_conversation_with_an_image() {
    let Some(model) = find_vision_qwen35() else {
        println!(
            "[qwen35_vision] skip: no CI-safe qwen3_5 vision model found under MLEX_MODELS_DIR"
        );
        return;
    };

    let session = Session::load(&model.dir).expect("failed to load qwen3.5 vision model");

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
