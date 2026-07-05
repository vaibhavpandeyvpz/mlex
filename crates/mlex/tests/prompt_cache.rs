//! Core correctness contract for `Session::generate_cached`'s stateless
//! prompt-cache pool: incremental generation across two calls that share a
//! growing message-list prefix must produce byte-identical text to a full
//! from-scratch `Session::generate()` call over the fully-rendered
//! multi-turn prompt, for every CI-safe model. This proves cache reuse is
//! numerically transparent, not just "fast".

mod common;

use mlex::generate::{GenerateOptions, Session};
use mlex::sampling::SamplingConfig;
use mlex::tokenizer::ChatMessage;

/// Greedy sampling with thinking explicitly disabled: this suite compares
/// `generate_cached`'s output against a from-scratch full recompute via
/// raw `Session::generate` + plain token decode (no reasoning-splitting
/// post-processing), so any checkpoint that spontaneously opens a
/// `<think>` span (e.g. Qwen3, MiniCPM5) would otherwise make the two
/// paths diverge for reasons unrelated to prompt-cache correctness.
fn greedy_opts(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_tokens,
        sampling: SamplingConfig::default(),
        enable_thinking: Some(false),
        ..Default::default()
    }
}

#[test]
fn generate_cached_matches_full_recompute_on_turn_two() {
    let models = common::registry();
    if models.is_empty() {
        eprintln!("[prompt_cache] no CI-safe models found; skipping");
        return;
    }

    for model in &models {
        let session = Session::load(&model.dir).expect("load failed");

        // Path A: two stateless `generate_cached` calls, passing the full
        // (growing) message list each time - exactly as a real caller
        // would, with no session handle in between.
        let turn1_messages = vec![ChatMessage::user("My favorite color is blue.")];
        let turn1_reply = session
            .generate_cached(&turn1_messages, None, greedy_opts(6), |_| true)
            .unwrap()
            .text;

        let turn2_messages = vec![
            ChatMessage::user("My favorite color is blue."),
            ChatMessage::assistant(&turn1_reply),
            ChatMessage::user("What is 2+2?"),
        ];
        let turn2_via_cache = session
            .generate_cached(&turn2_messages, None, greedy_opts(6), |_| true)
            .unwrap()
            .text;

        // Path B: full recompute - render the exact same 2-turn transcript
        // from scratch and generate once over it. `enable_thinking:
        // Some(false)` must match `greedy_opts` above so both paths render
        // the same prompt (some checkpoints open an unprompted `<think>`
        // span when the key is left undefined).
        let prompt = session
            .tokenizer()
            .apply_chat_template_full(&turn2_messages, true, None, Some(false))
            .unwrap();
        let prompt_ids = session.tokenizer().encode(&prompt).unwrap();
        let full_recompute_ids = session
            .generate(&prompt_ids, greedy_opts(6), |_| true)
            .unwrap();
        let full_recompute_text = session.tokenizer().decode(&full_recompute_ids).unwrap();

        assert_eq!(
            turn2_via_cache, full_recompute_text,
            "{}: generate_cached's cached turn 2 diverged from full recompute",
            model.repo_id
        );
    }
}

#[test]
fn generate_cached_with_an_unrelated_first_message_starts_a_fresh_lineage() {
    let models = common::registry();
    if models.is_empty() {
        return;
    }
    let model = &models[0];
    let session = Session::load(&model.dir).expect("load failed");

    let remember = vec![ChatMessage::user("Remember the number 7.")];
    session
        .generate_cached(&remember, None, greedy_opts(4), |_| true)
        .unwrap();

    // No explicit "reset" exists in a stateless API - a brand-new,
    // unrelated message list simply misses the pool and starts cold,
    // producing the same output a fresh session's `generate()` would.
    let say_hi = vec![ChatMessage::user("Say hi.")];
    let reply = session
        .generate_cached(&say_hi, None, greedy_opts(4), |_| true)
        .unwrap()
        .text;

    // `enable_thinking: Some(false)` must match `greedy_opts` above so both
    // paths render the same prompt.
    let fresh_prompt = session
        .tokenizer()
        .apply_chat_template_full(&say_hi, true, None, Some(false))
        .unwrap();
    let fresh_ids = session.tokenizer().encode(&fresh_prompt).unwrap();
    let fresh_ids = session
        .generate(&fresh_ids, greedy_opts(4), |_| true)
        .unwrap();
    let fresh_text = session.tokenizer().decode(&fresh_ids).unwrap();

    assert_eq!(
        reply, fresh_text,
        "{}: an unrelated message list should match a fresh session",
        model.repo_id
    );
}

#[test]
fn generate_cached_supports_three_turns() {
    let models = common::registry();
    if models.is_empty() {
        return;
    }
    let model = &models[0];
    let session = Session::load(&model.dir).expect("load failed");

    let mut messages = vec![ChatMessage::user("Turn one.")];
    let reply1 = session
        .generate_cached(&messages, None, greedy_opts(4), |_| true)
        .unwrap()
        .text;
    messages.push(ChatMessage::assistant(&reply1));

    messages.push(ChatMessage::user("Turn two."));
    let reply2 = session
        .generate_cached(&messages, None, greedy_opts(4), |_| true)
        .unwrap()
        .text;
    messages.push(ChatMessage::assistant(&reply2));

    messages.push(ChatMessage::user("Turn three."));
    let reply3 = session
        .generate_cached(&messages, None, greedy_opts(4), |_| true)
        .unwrap()
        .text;
    messages.push(ChatMessage::assistant(&reply3));

    assert_eq!(messages.len(), 6, "expected 3 user + 3 assistant turns");
}

/// Two independent, unrelated calls sharing only a common system-style
/// prefix should both benefit from the pool - the defining property a
/// single-lineage `Conversation` handle could never have.
#[test]
fn generate_cached_serves_two_independent_callers_sharing_a_prefix() {
    let models = common::registry();
    if models.is_empty() {
        return;
    }
    let model = &models[0];
    let session = Session::load(&model.dir).expect("load failed");

    let shared_prefix = vec![ChatMessage::user("You are a helpful assistant. Say hi.")];
    let first_reply = session
        .generate_cached(&shared_prefix, None, greedy_opts(4), |_| true)
        .unwrap()
        .text;

    // A second, unrelated "caller" sends the exact same first message
    // again (e.g. two different users hitting the same cached system
    // prompt) - this must be numerically identical to the first call,
    // whether served from the pool or recomputed.
    let second_reply = session
        .generate_cached(&shared_prefix, None, greedy_opts(4), |_| true)
        .unwrap()
        .text;

    assert_eq!(
        first_reply, second_reply,
        "{}: repeating the same prompt should be deterministic",
        model.repo_id
    );
}
