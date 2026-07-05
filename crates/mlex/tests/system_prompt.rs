//! System instructions: a leading `role: "system"` message is a first-class
//! part of the chat template contract (every supported family's template
//! special-cases `messages[0]['role'] in ['system', 'developer']`, the same
//! convention as the OpenAI/Anthropic APIs), not something bolted on for
//! one architecture. These tests verify both that the template actually
//! renders the system content and that generation still works when a
//! leading system turn is present, without assuming every tiny checkpoint
//! will obey an arbitrary style instruction perfectly.

mod common;

use mlex::generate::{GenerateOptions, Session};
use mlex::sampling::SamplingConfig;
use mlex::tokenizer::ChatMessage;

#[test]
fn chat_template_renders_the_system_message_content() {
    let models = common::registry();
    if models.is_empty() {
        eprintln!("[system_prompt] no CI-safe models found; skipping");
        return;
    }

    let marker = "TEST-SYSTEM-INSTRUCTION-MARKER-71828";
    for model in &models {
        let session = Session::load(&model.dir).expect("load failed");
        let messages = vec![
            ChatMessage::system(marker),
            ChatMessage::user("What's the capital of France?"),
        ];
        let rendered = session
            .tokenizer()
            .apply_chat_template(&messages, true)
            .unwrap_or_else(|e| panic!("{}: template rendering failed: {e}", model.repo_id));

        assert!(
            rendered.contains(marker),
            "{}: rendered prompt doesn't contain the system message content:\n{rendered}",
            model.repo_id
        );
    }
}

#[test]
fn generation_with_a_leading_system_message_is_nonempty() {
    let models = common::registry();
    if models.is_empty() {
        eprintln!("[system_prompt] no CI-safe models found; skipping");
        return;
    }

    for model in &models {
        let session = Session::load(&model.dir).expect("load failed");
        let messages = vec![
            ChatMessage::system("You are a terse assistant. Answer in one short sentence."),
            ChatMessage::user("What's the capital of France?"),
        ];
        let opts = GenerateOptions {
            max_tokens: 16,
            sampling: SamplingConfig::default(),
            ..Default::default()
        };
        let text = session
            .generate_cached(&messages, None, opts, |_| true)
            .unwrap()
            .text;

        assert!(
            !text.trim().is_empty(),
            "{}: generation with a leading system message produced empty output",
            model.repo_id
        );
    }
}
