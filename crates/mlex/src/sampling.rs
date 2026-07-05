//! Token sampling strategies applied to the final-step logits.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::array::Array;
use crate::error::Result;
use crate::ops;

/// Sampling configuration for one generation call.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<i32>,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: None,
        }
    }
}

/// Stateful sampler holding the RNG across a generation session.
pub struct Sampler {
    config: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Sampler { config, rng }
    }

    /// Sample one token id from `logits` (shape `[vocab]`, last-step only).
    pub fn sample(&mut self, logits: &Array) -> Result<u32> {
        if self.config.temperature <= 0.0 {
            let idx = ops::argmax_axis(logits, -1, false)?;
            return idx.item_u32();
        }

        let scaled = ops::scale_by(logits, 1.0 / self.config.temperature)?;
        let mut probs = softmax_to_vec(&scaled)?;

        if let Some(k) = self.config.top_k {
            top_k_filter(&mut probs, k as usize);
        }
        if self.config.top_p < 1.0 {
            top_p_filter(&mut probs, self.config.top_p);
        }

        let total: f32 = probs.iter().sum();
        let mut draw = self.rng.gen::<f32>() * total;
        for (i, p) in probs.iter().enumerate() {
            draw -= p;
            if draw <= 0.0 {
                return Ok(i as u32);
            }
        }
        Ok((probs.len().max(1) - 1) as u32)
    }
}

fn softmax_to_vec(logits: &Array) -> Result<Vec<f32>> {
    let probs = ops::softmax_axis(logits, -1, true)?;
    probs.to_vec_f32()
}

fn top_k_filter(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let keep: std::collections::HashSet<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *p = 0.0;
        }
    }
}

fn top_p_filter(probs: &mut [f32], top_p: f32) {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        return;
    }
    let mut cumulative = 0.0;
    let mut cutoff = indexed.len();
    for (rank, (_, p)) in indexed.iter().enumerate() {
        cumulative += p / total;
        if cumulative >= top_p {
            cutoff = rank + 1;
            break;
        }
    }
    let keep: std::collections::HashSet<usize> =
        indexed.iter().take(cutoff).map(|(i, _)| *i).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *p = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_picks_argmax() {
        let logits = Array::from_slice(&[0.1f32, 0.9, 0.3, -0.2], &[4]);
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        });
        assert_eq!(sampler.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn greedy_sampling_is_deterministic_across_calls() {
        let logits = Array::from_slice(&[1.0f32, 5.0, 2.0], &[3]);
        let mut a = Sampler::new(SamplingConfig::default());
        let mut b = Sampler::new(SamplingConfig::default());
        assert_eq!(a.sample(&logits).unwrap(), b.sample(&logits).unwrap());
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let logits = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4]);
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            seed: Some(42),
        };
        let mut a = Sampler::new(cfg);
        let mut b = Sampler::new(cfg);
        let seq_a: Vec<u32> = (0..10).map(|_| a.sample(&logits).unwrap()).collect();
        let seq_b: Vec<u32> = (0..10).map(|_| b.sample(&logits).unwrap()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn top_k_filter_keeps_only_k_largest() {
        let mut probs = vec![0.1, 0.4, 0.2, 0.3];
        top_k_filter(&mut probs, 2);
        let nonzero: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(nonzero, vec![1, 3]);
    }

    #[test]
    fn top_k_filter_noop_when_k_covers_all() {
        let mut probs = vec![0.1, 0.4, 0.2, 0.3];
        let original = probs.clone();
        top_k_filter(&mut probs, 10);
        assert_eq!(probs, original);
    }

    #[test]
    fn top_k_filter_noop_when_k_is_zero() {
        let mut probs = vec![0.1, 0.4, 0.2, 0.3];
        let original = probs.clone();
        top_k_filter(&mut probs, 0);
        assert_eq!(probs, original);
    }

    #[test]
    fn top_p_filter_keeps_smallest_prefix_reaching_mass() {
        let mut probs = vec![0.5, 0.3, 0.15, 0.05];
        top_p_filter(&mut probs, 0.8);
        let nonzero: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .collect();
        // 0.5 + 0.3 = 0.8 >= 0.8 cutoff, so only the top 2 survive.
        assert_eq!(nonzero, vec![0, 1]);
    }

    #[test]
    fn top_p_filter_keeps_everything_at_top_p_one() {
        let mut probs = vec![0.5, 0.3, 0.15, 0.05];
        let original = probs.clone();
        top_p_filter(&mut probs, 1.0);
        assert_eq!(probs, original);
    }

    #[test]
    fn sampling_config_default_is_greedy() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_p, 1.0);
        assert!(cfg.top_k.is_none());
    }
}
