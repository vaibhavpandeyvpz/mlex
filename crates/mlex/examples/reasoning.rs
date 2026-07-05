//! Demonstrates `enable_thinking` + `reasoning_budget_tokens` plus the
//! streamed `kind` tagging that separates reasoning from the final answer
//! live (dim-rendered here), on any checkpoint whose chat template
//! supports "thinking" mode (Qwen3/3.5/3.6, Gemma4, MiniCPM5, NemotronH).
//!
//! `cargo run --release --example reasoning -- <model_dir>`

use std::path::PathBuf;

use mlex::generate::{GenerateOptions, Session};
use mlex::streaming::TokenKind;
use mlex::tokenizer::ChatMessage;

fn main() {
    let model_dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: reasoning_smoke <model_dir>"),
    );
    let session = Session::load(&model_dir).expect("load failed");

    let messages = vec![ChatMessage::user("What is 12 * 7? Think step by step.")];
    let options = GenerateOptions {
        max_tokens: 80,
        enable_thinking: Some(true),
        reasoning_budget_tokens: Some(30),
        ..Default::default()
    };

    let mut kinds_seen = std::collections::HashSet::new();
    let reply = session
        .generate_cached(&messages, None, options, |tok| {
            kinds_seen.insert(format!("{:?}", tok.kind));
            match tok.kind {
                TokenKind::Reasoning => print!("\x1b[2m{}\x1b[0m", tok.text),
                _ => print!("{}", tok.text),
            }
            true
        })
        .expect("generate failed");

    println!("\n--- resolved ---");
    println!("text: {:?}", reply.text);
    println!("reasoning: {:?}", reply.reasoning);
    println!("usage: {:?}", reply.usage);
    println!("kinds seen while streaming: {kinds_seen:?}");
}
