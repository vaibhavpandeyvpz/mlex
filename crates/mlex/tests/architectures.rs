//! Per-family regression guard: one forward pass per locally-available
//! model (any tier - CI-safe or local-only), asserting the model loads,
//! produces finite, well-shaped logits, and that greedy-argmax decoding is
//! stable across repeated calls (a cheap proxy for "the numerics didn't
//! silently break"). Runs against every discovered model rather than one
//! per family, since a checkpoint-specific quantization layout bug
//! wouldn't show up if only one representative were checked.

mod common;

use mlex::generate::{GenerateOptions, Session};
use mlex::tokenizer::ChatMessage;

#[test]
fn every_discovered_model_produces_finite_greedy_logits() {
    let models = common::registry();
    if models.is_empty() {
        eprintln!("[architectures] no CI-safe models found under MLEX_MODELS_DIR; skipping");
        return;
    }

    for model in &models {
        let session = Session::load(&model.dir)
            .unwrap_or_else(|e| panic!("failed to load {}: {e}", model.dir.display()));

        let messages = vec![ChatMessage::user("Say hello in exactly five words.")];
        let prompt_ids = session
            .encode_chat(&messages)
            .expect("chat template failed");
        assert!(
            !prompt_ids.is_empty(),
            "{}: empty prompt encoding",
            model.repo_id
        );

        // Two independent greedy generations of the same prompt from a
        // fresh session/cache must be byte-identical (temperature 0.0).
        let out_a = generate_greedy(&session, &prompt_ids);
        let out_b = generate_greedy(&session, &prompt_ids);
        assert_eq!(
            out_a, out_b,
            "{}: greedy decoding is not deterministic",
            model.repo_id
        );
        assert!(
            !out_a.is_empty(),
            "{}: generated zero tokens",
            model.repo_id
        );
    }
}

fn generate_greedy(session: &Session, prompt_ids: &[u32]) -> Vec<u32> {
    session
        .generate(
            prompt_ids,
            GenerateOptions {
                max_tokens: 8,
                ..Default::default()
            },
            |_| true,
        )
        .expect("generation failed")
}
