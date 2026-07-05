//! Loads a model directory, runs one short forward pass, and prints its
//! peak resident-set size (bytes) to stdout. Used by
//! `tests/common/mod.rs`'s model registry to measure real (not guessed)
//! per-model memory footprint for CI size-gating; run in a fresh
//! subprocess per model so measurements never accumulate.
//!
//! Usage: `measure_memory <model_dir>`

use std::path::PathBuf;

use mlex::generate::{GenerateOptions, Session};
use mlex::tokenizer::ChatMessage;

fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        #[cfg(target_os = "macos")]
        {
            usage.ru_maxrss as u64
        }
        #[cfg(not(target_os = "macos"))]
        {
            usage.ru_maxrss as u64 * 1024
        }
    }
}

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .expect("usage: measure_memory <model_dir>");
    let model_dir = PathBuf::from(model_dir);

    let session = match Session::load(&model_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to load model: {e}");
            std::process::exit(1);
        }
    };

    let messages = vec![ChatMessage::user("Say hello.")];
    let prompt_ids = session.encode_chat(&messages).unwrap_or_else(|_| vec![0]);
    let _ = session.generate(
        &prompt_ids,
        GenerateOptions {
            max_tokens: 4,
            ..Default::default()
        },
        |_| true,
    );

    println!("{}", peak_rss_bytes());
}
