//! Sparse mixture-of-experts feed-forward block (`SwitchGLU` +
//! `Qwen3NextSparseMoeBlock` in mlx-lm), used by the Qwen3.6-A3B MoE
//! checkpoints. Ports the ops-based (non-pre-sorted) gather-matmul path;
//! `sorted_indices` is always passed as `false` to `gather_mm`/`gather_qmm`
//! since sorting by expert id is purely a locality optimization for very
//! large batches, not required for correctness.

use crate::array::Array;
use crate::error::Result;
use crate::nn::WeightMap;
use crate::ops;
use crate::quant::QuantParams;

use super::qwen3_5::Qwen35Config;

/// One (fused) expert-batched linear layer: dense `[E, out, in]` weight or
/// its quantized packed equivalent, dispatched per-token via `gather_mm`.
enum SwitchLinear {
    Dense {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        params: QuantParams,
    },
}

impl SwitchLinear {
    fn load(w: &mut WeightMap, path: &str) -> Result<Self> {
        let has_scales = w.contains(&format!("{path}.scales"));
        let params = w.quantization().resolve(path, has_scales);
        let weight = w.take(&format!("{path}.weight"))?;
        match params {
            Some(q) if has_scales => {
                let scales = w.take(&format!("{path}.scales"))?;
                let biases = w.take_optional(&format!("{path}.biases"));
                Ok(SwitchLinear::Quantized {
                    weight,
                    scales,
                    biases,
                    params: q,
                })
            }
            _ => Ok(SwitchLinear::Dense { weight }),
        }
    }

    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        match self {
            SwitchLinear::Dense { weight } => {
                let wt = ops::swapaxes(weight, -1, -2)?;
                ops::gather_mm(x, &wt, indices)
            }
            SwitchLinear::Quantized {
                weight,
                scales,
                biases,
                params,
            } => ops::gather_qmm(
                x,
                weight,
                scales,
                biases.as_ref(),
                None,
                Some(indices),
                true,
                params.group_size,
                params.bits,
                params.mode,
                false,
            ),
        }
    }
}

struct SwitchGlu {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGlu {
    fn load(w: &mut WeightMap, prefix: &str) -> Result<Self> {
        Ok(SwitchGlu {
            gate_proj: SwitchLinear::load(w, &format!("{prefix}.gate_proj"))?,
            up_proj: SwitchLinear::load(w, &format!("{prefix}.up_proj"))?,
            down_proj: SwitchLinear::load(w, &format!("{prefix}.down_proj"))?,
        })
    }

    /// `x`: `[B, S, H]`, `indices`: `[B, S, K]` selected expert ids.
    /// Returns `[B, S, K, H]`.
    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, s, h) = (shape[0], shape[1], shape[2]);
        let x = ops::reshape(x, &[b, s, 1, 1, h])?;

        let x_up = self.up_proj.forward(&x, indices)?;
        let x_gate = self.gate_proj.forward(&x, indices)?;
        let activated = ops::multiply(&ops::silu(&x_gate)?, &x_up)?;
        let out = self.down_proj.forward(&activated, indices)?;
        let out_shape = out.shape();
        // Drop the singleton axis introduced by `reshape(..., [B, S, 1, 1, H])`.
        ops::reshape(
            &out,
            &[
                out_shape[0],
                out_shape[1],
                out_shape[2],
                *out_shape.last().unwrap(),
            ],
        )
    }
}

struct SharedExpert {
    gate_proj: crate::nn::Linear,
    up_proj: crate::nn::Linear,
    down_proj: crate::nn::Linear,
}

impl SharedExpert {
    fn load(w: &mut WeightMap, prefix: &str) -> Result<Self> {
        Ok(SharedExpert {
            gate_proj: w.linear(&format!("{prefix}.gate_proj"))?,
            up_proj: w.linear(&format!("{prefix}.up_proj"))?,
            down_proj: w.linear(&format!("{prefix}.down_proj"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&ops::multiply(&gate, &up)?)
    }
}

/// `Qwen3NextSparseMoeBlock`: top-k routed experts plus one always-on
/// "shared" expert whose output is scaled by a learned per-token gate.
pub struct SparseMoeBlock {
    gate: crate::nn::Linear,
    switch_mlp: SwitchGlu,
    shared_expert: SharedExpert,
    shared_expert_gate: crate::nn::Linear,
    top_k: i32,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub fn load(w: &mut WeightMap, prefix: &str, cfg: &Qwen35Config) -> Result<Self> {
        Ok(SparseMoeBlock {
            gate: w.linear(&format!("{prefix}.gate"))?,
            switch_mlp: SwitchGlu::load(w, &format!("{prefix}.switch_mlp"))?,
            shared_expert: SharedExpert::load(w, &format!("{prefix}.shared_expert"))?,
            shared_expert_gate: w.linear(&format!("{prefix}.shared_expert_gate"))?,
            top_k: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gates = self.gate.forward(x)?;
        let gates = ops::softmax_axis(&gates, -1, true)?;

        let k = self.top_k;
        let n_experts = gates.dim(-1);
        let part = ops::argpartition_axis(&gates, n_experts - k, -1)?;
        let shape = part.shape();
        let inds = ops::slice(&part, &[0, 0, n_experts - k], &shape)?;
        let scores = ops::take_along_axis(&gates, &inds, -1)?;
        let scores = if self.norm_topk_prob {
            let sum = ops::sum_axes(&scores, &[-1], true)?;
            ops::divide(&scores, &sum)?
        } else {
            scores
        };

        let y = self.switch_mlp.forward(x, &inds)?; // [B, S, K, H]
        let weighted = ops::multiply(&y, &ops::expand_dims(&scores, -1)?)?;
        let y = ops::sum_axes(&weighted, &[-2], false)?; // [B, S, H]

        let shared_y = self.shared_expert.forward(x)?;
        let gate = ops::sigmoid(&self.shared_expert_gate.forward(x)?)?;
        let shared_y = ops::multiply(&gate, &shared_y)?;

        ops::add(&y, &shared_y)
    }
}
