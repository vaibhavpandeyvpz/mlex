//! Minimal end-to-end smoke test: `cargo run --example generate -- <model_dir>`

use std::path::PathBuf;
use std::time::Instant;

use mlex::generate::{GenerateOptions, Session};
use mlex::tokenizer::ChatMessage;

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .expect("usage: generate <model_dir>");
    let model_dir = PathBuf::from(model_dir);

    println!("[example] loading {}", model_dir.display());
    let t0 = Instant::now();
    let session = Session::load(&model_dir).expect("failed to load model");
    println!("[example] loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let messages = vec![ChatMessage::user("Say hello in exactly five words.")];
    let prompt_ids = session
        .encode_chat(&messages)
        .expect("chat template failed");
    println!("[example] prompt tokens: {}", prompt_ids.len());
    println!("[example] prompt ids: {prompt_ids:?}");

    if let Ok(override_ids) = std::env::var("MLEX_DEBUG_LOGITS") {
        use mlex::array::Array;
        let ids: Vec<u32> = if override_ids == "1" {
            prompt_ids.clone()
        } else {
            override_ids
                .split(',')
                .map(|s| s.parse().unwrap())
                .collect()
        };
        let arr = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let mut caches = session.debug_new_caches();
        let logits = session
            .debug_forward(&arr, &mut caches)
            .expect("forward failed");
        let shape = logits.shape();
        println!("[debug] logits shape: {shape:?}");
        let last = logits.to_vec_f32().expect("extract");
        let vocab = shape[2] as usize;
        let seq = shape[1] as usize;
        let last_row = &last[(seq - 1) * vocab..seq * vocab];
        let mut idx: Vec<usize> = (0..vocab).collect();
        idx.sort_by(|&a, &b| last_row[b].partial_cmp(&last_row[a]).unwrap());
        println!("[debug] top5 ids: {:?}", &idx[..5]);
        println!(
            "[debug] top5 vals: {:?}",
            idx[..5].iter().map(|&i| last_row[i]).collect::<Vec<_>>()
        );

        if std::env::var("MLEX_DEBUG_LAYERS").is_ok() {
            let stats = session
                .debug_nemotron_layer_stats(&arr)
                .expect("layer stats failed");
            for (i, (mean_abs, std)) in stats.iter().enumerate() {
                println!("[layer {i}] mean_abs={mean_abs:.6} std={std:.6}");
            }
        }
        return;
    }

    let t1 = Instant::now();
    let mut text = String::new();
    session
        .generate(
            &prompt_ids,
            GenerateOptions {
                max_tokens: 40,
                ..Default::default()
            },
            |tok| {
                text.push_str(&tok.text);
                print!("{}", tok.text);
                use std::io::Write;
                std::io::stdout().flush().ok();
                true
            },
        )
        .expect("generation failed");
    println!(
        "\n[example] generated in {:.2}s",
        t1.elapsed().as_secs_f32()
    );
}
