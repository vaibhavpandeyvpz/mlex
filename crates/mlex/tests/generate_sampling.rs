//! Seeded `temperature > 0` sampling reproducibility, and that top-k/top-p
//! filtering actually changes output relative to unfiltered sampling.

mod common;

use mlex::generate::{GenerateOptions, Session};
use mlex::sampling::SamplingConfig;
use mlex::tokenizer::ChatMessage;

#[test]
fn seeded_sampling_is_reproducible_end_to_end() {
    let models = common::registry();
    if models.is_empty() {
        eprintln!("[generate_sampling] no CI-safe models found; skipping");
        return;
    }

    for model in &models {
        let session = Session::load(&model.dir).expect("load failed");
        let prompt_ids = session
            .encode_chat(&[ChatMessage::user("Tell me a fun fact.")])
            .unwrap();

        let sampling = SamplingConfig {
            temperature: 0.8,
            top_p: 0.95,
            top_k: Some(40),
            seed: Some(1234),
        };
        let opts = GenerateOptions {
            max_tokens: 12,
            sampling,
            ..Default::default()
        };

        let out_a = session.generate(&prompt_ids, opts, |_| true).unwrap();
        let out_b = session.generate(&prompt_ids, opts, |_| true).unwrap();

        assert_eq!(
            out_a, out_b,
            "{}: same seed must produce identical sampled ids",
            model.repo_id
        );
    }
}

#[test]
fn different_seeds_can_diverge() {
    let models = common::registry();
    if models.is_empty() {
        return;
    }
    let model = &models[0];
    let session = Session::load(&model.dir).expect("load failed");
    let prompt_ids = session
        .encode_chat(&[ChatMessage::user("Tell me a fun fact.")])
        .unwrap();

    // Sample with a handful of distinct seeds rather than just one pair.
    //
    // Investigated a real failure here where seeds 1 and 2 alone produced
    // byte-identical 20-token output on `gemma-4-e2b-it-OptiQ-4bit`
    // (`common::registry()[0]` on this checkpoint set). Root-caused via
    // `crates/mlex/src/sampling.rs`'s RNG and the OptiQ dequant path
    // (`nn.rs`/`quant.rs`) rather than assuming a bug:
    //   - The RNG *is* threading through correctly: probing seeds
    //     1/2/3/42/999/123456 on this exact model+prompt shows seeds 3 and
    //     123456 *do* diverge (at token 20 and token 14 respectively) -
    //     if the seed were being ignored (e.g. sampling silently
    //     collapsing to argmax), every seed would produce identical
    //     output with no exceptions.
    //   - The top-1 logit margin for the very first sampled token is
    //     genuinely enormous on this checkpoint (~24.5 vs. ~10.4 for the
    //     runner-up, softmax(T=1) mass > 0.999999 on "Here" - the obvious
    //     opener for "tell me a fun fact") - this is a real, extremely
    //     peaked distribution, not NaN/Inf/garbage.
    //   - Comparing against `gemma-4-E2B-it-qat-4bit` (same architecture,
    //     same prompt, same decoding params) shows a *much* flatter
    //     distribution that diverges within 1-3 tokens across the same
    //     seed set. OptiQ's mixed-precision recipe averages ~5.26 bits/
    //     weight (`optiq_metadata.json`: `target_bpw: 5.0`) vs. QAT's
    //     uniform 4-bit, i.e. OptiQ is the *higher*-fidelity/closer-to-
    //     bf16 quantization of the two. A higher-fidelity checkpoint
    //     being *more* confident (not less) on a canonical completion,
    //     while a cruder one is flattened by extra quantization noise, is
    //     the expected direction if this is a genuine model/checkpoint
    //     property rather than a dequantization bug - a bug would be far
    //     more likely to *corrupt* the ranking or blow up to inf/NaN than
    //     to consistently reproduce a plausible, coherent top token.
    //
    // So: real, checkpoint-specific near-degenerate confidence on this
    // particular prompt's opening token, not a broken RNG or a logit-
    // scale bug - widen the seed set instead of asserting on a single
    // arbitrary pair, which was always somewhat likely to collide by
    // chance for the initial part of a plausible sequence like this.
    let seeds = [1u64, 2, 3, 42, 999, 123456];
    let outputs: Vec<Vec<u32>> = seeds
        .iter()
        .map(|&seed| {
            let opts = GenerateOptions {
                max_tokens: 20,
                sampling: SamplingConfig {
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: None,
                    seed: Some(seed),
                },
                ..Default::default()
            };
            session.generate(&prompt_ids, opts, |_| true).unwrap()
        })
        .collect();

    let all_identical = outputs.windows(2).all(|w| w[0] == w[1]);
    assert!(
        !all_identical,
        "{}: {} different seeds all produced identical output - seed likely isn't threading through",
        model.repo_id,
        seeds.len()
    );
}

#[test]
fn greedy_temperature_zero_matches_across_runs() {
    let models = common::registry();
    if models.is_empty() {
        return;
    }
    let model = &models[0];
    let session = Session::load(&model.dir).expect("load failed");
    let prompt_ids = session
        .encode_chat(&[ChatMessage::user("Say hi.")])
        .unwrap();

    let opts = GenerateOptions {
        max_tokens: 10,
        sampling: SamplingConfig::default(),
        ..Default::default()
    };
    let out_a = session.generate(&prompt_ids, opts, |_| true).unwrap();
    let out_b = session.generate(&prompt_ids, opts, |_| true).unwrap();
    assert_eq!(out_a, out_b);
}
