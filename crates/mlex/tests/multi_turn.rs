//! Multi-turn `Session::generate_cached` produces coherent, context-aware
//! replies when the caller grows the message list itself between calls.
//! Assertions are loose (structural: non-empty, turn count) rather than
//! exact-match, since small CI-safe models sampled at low temperature
//! aren't guaranteed to reliably echo back facts from earlier turns - the
//! numerically-exact contract is `prompt_cache.rs`'s job; this suite is
//! about the stateless chat API surface behaving sensibly end to end.

mod common;

use mlex::generate::{GenerateOptions, Session};
use mlex::sampling::SamplingConfig;
use mlex::tokenizer::ChatMessage;

/// Greedy sampling with thinking explicitly disabled: this suite asserts
/// non-empty, structural replies within a small token budget, and some
/// checkpoints (e.g. MiniCPM5) spontaneously open a `<think>` span even
/// without an explicit request - which a small `max_tokens` budget can
/// truncate before the model ever closes it, leaving the post-reasoning
/// `text` empty. `enable_thinking: false` renders a pre-closed empty
/// `<think></think>` span so every checkpoint answers directly.
fn greedy_opts(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_tokens,
        sampling: SamplingConfig::default(),
        enable_thinking: Some(false),
        ..Default::default()
    }
}

#[test]
fn multi_turn_conversation_grows_message_history_in_order() {
    let models = common::registry();
    if models.is_empty() {
        eprintln!("[multi_turn] no CI-safe models found; skipping");
        return;
    }

    for model in &models {
        let session = Session::load(&model.dir).expect("load failed");

        let mut messages = vec![ChatMessage::user("Hi, my name is Alex.")];
        let reply1 = session
            .generate_cached(&messages, None, greedy_opts(16), |_| true)
            .unwrap()
            .text;
        assert!(!reply1.is_empty(), "{}: empty first reply", model.repo_id);
        messages.push(ChatMessage::assistant(&reply1));

        messages.push(ChatMessage::user("What's 10 minus 3?"));
        let reply2 = session
            .generate_cached(&messages, None, greedy_opts(16), |_| true)
            .unwrap()
            .text;
        assert!(!reply2.is_empty(), "{}: empty second reply", model.repo_id);
        messages.push(ChatMessage::assistant(&reply2));

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[2].role, "user");
        assert_eq!(messages[3].role, "assistant");
        assert_eq!(messages[1].text(), reply1);
        assert_eq!(messages[3].text(), reply2);
    }
}

#[test]
fn tool_result_turn_is_recorded_with_role_tool() {
    let models = common::registry();
    if models.is_empty() {
        return;
    }
    let model = &models[0];
    let session = Session::load(&model.dir).expect("load failed");

    let mut messages = vec![ChatMessage::user("What's the weather?")];
    let reply = session
        .generate_cached(&messages, None, greedy_opts(8), |_| true)
        .unwrap()
        .text;
    messages.push(ChatMessage::assistant(&reply));
    messages.push(ChatMessage::tool_result("call_0", "{\"temp_c\": 21}"));

    let tool_msg = messages.last().unwrap();
    assert_eq!(tool_msg.role, "tool");
    assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_0"));
    assert_eq!(tool_msg.text(), "{\"temp_c\": 21}");
}
