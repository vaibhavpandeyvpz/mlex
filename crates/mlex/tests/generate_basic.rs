//! Single-turn `Session::generate()`: deterministic (`temperature: 0.0`)
//! output is non-empty, respects `max_tokens`, and reports EOS correctly.
//! Run once per CI-safe model (see `tests/common/mod.rs`).

mod common;

use mlex::generate::{GenerateOptions, Session};
use mlex::tokenizer::ChatMessage;

#[test]
fn greedy_generation_is_nonempty_and_respects_max_tokens() {
    let models = common::registry();
    if models.is_empty() {
        eprintln!("[generate_basic] no CI-safe models found; skipping");
        return;
    }

    for model in &models {
        let session = Session::load(&model.dir).expect("load failed");
        let messages = vec![ChatMessage::user("Count from one to three.")];
        let prompt_ids = session.encode_chat(&messages).expect("template failed");

        let max_tokens = 16;
        let mut streamed = Vec::new();
        let generated = session
            .generate(
                &prompt_ids,
                GenerateOptions {
                    max_tokens,
                    ..Default::default()
                },
                |tok| {
                    streamed.push(tok.id);
                    true
                },
            )
            .expect("generation failed");

        assert!(
            !generated.is_empty(),
            "{}: no tokens generated",
            model.repo_id
        );
        assert!(
            generated.len() <= max_tokens,
            "{}: generated {} tokens, exceeding max_tokens={max_tokens}",
            model.repo_id,
            generated.len()
        );
        assert_eq!(
            generated, streamed,
            "{}: streamed ids don't match returned ids",
            model.repo_id
        );
    }
}

#[test]
fn on_token_can_stop_generation_early() {
    let models = common::registry();
    if models.is_empty() {
        return;
    }
    let model = &models[0];
    let session = Session::load(&model.dir).expect("load failed");
    let prompt_ids = session
        .encode_chat(&[ChatMessage::user("Tell me a long story.")])
        .unwrap();

    let mut count = 0;
    let generated = session
        .generate(
            &prompt_ids,
            GenerateOptions {
                max_tokens: 100,
                ..Default::default()
            },
            |_| {
                count += 1;
                count < 3
            },
        )
        .expect("generation failed");

    assert_eq!(
        generated.len(),
        3,
        "stopping via on_token's return value should cut generation short"
    );
}

#[test]
fn eos_token_stops_generation() {
    let models = common::registry();
    if models.is_empty() {
        return;
    }
    let model = &models[0];
    let session = Session::load(&model.dir).expect("load failed");
    let prompt_ids = session
        .encode_chat(&[ChatMessage::user("Say just \"hi\" and stop.")])
        .unwrap();

    let mut saw_finished = false;
    let generated = session
        .generate(
            &prompt_ids,
            GenerateOptions {
                max_tokens: 200,
                ..Default::default()
            },
            |tok| {
                if tok.finished {
                    saw_finished = true;
                }
                true
            },
        )
        .expect("generation failed");

    // Either we hit max_tokens or EOS; if generation ended before
    // max_tokens, it must be because EOS fired.
    if generated.len() < 200 {
        assert!(
            saw_finished,
            "{}: generation ended early without an EOS token",
            model.repo_id
        );
    }
}
