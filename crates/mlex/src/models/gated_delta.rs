//! `GatedDeltaNet`: the linear-attention layer used by Qwen3.5/3.6 in place
//! of quadratic self-attention on most decoder layers. Ports mlx-lm's
//! `gated_delta.py` + `Qwen3NextGatedDeltaNet`/`GatedDeltaNet` ops-based
//! reference path (no fused Metal kernel): a short causal depthwise
//! convolution feeds a per-timestep delta-rule recurrence.
//!
//! Batch size is assumed to be 1 (single-sequence generation), so the
//! `mask`-based padding handling present in the Python reference (used for
//! batched training) is omitted.

use crate::array::{Array, Dtype};
use crate::error::Result;
use crate::nn::{Linear, WeightMap};
use crate::ops;

use super::cache::GatedDeltaCache;

pub struct GatedDeltaNet {
    conv1d_weight: Array,
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    dt_bias: Array,
    a_log: Array,
    norm_weight: Array,
    norm_eps: f32,
    out_proj: Linear,

    num_v_heads: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    key_dim: i32,
    value_dim: i32,
    conv_dim: i32,
    conv_kernel_size: i32,
}

pub struct GatedDeltaConfig {
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_size: i32,
    pub rms_norm_eps: f32,
}

impl GatedDeltaNet {
    pub fn load(w: &mut WeightMap, prefix: &str, cfg: &GatedDeltaConfig) -> Result<Self> {
        let key_dim = cfg.head_k_dim * cfg.num_k_heads;
        let value_dim = cfg.head_v_dim * cfg.num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;

        Ok(GatedDeltaNet {
            conv1d_weight: w.take(&format!("{prefix}.conv1d.weight"))?,
            in_proj_qkv: w.linear(&format!("{prefix}.in_proj_qkv"))?,
            in_proj_z: w.linear(&format!("{prefix}.in_proj_z"))?,
            in_proj_b: w.linear(&format!("{prefix}.in_proj_b"))?,
            in_proj_a: w.linear(&format!("{prefix}.in_proj_a"))?,
            dt_bias: w.take(&format!("{prefix}.dt_bias"))?,
            a_log: w.take(&format!("{prefix}.A_log"))?,
            norm_weight: w.take(&format!("{prefix}.norm.weight"))?,
            norm_eps: cfg.rms_norm_eps,
            out_proj: w.linear(&format!("{prefix}.out_proj"))?,
            num_v_heads: cfg.num_v_heads,
            num_k_heads: cfg.num_k_heads,
            head_k_dim: cfg.head_k_dim,
            head_v_dim: cfg.head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: cfg.conv_kernel_size,
        })
    }

    pub fn forward(&self, inputs: &Array, cache: &mut GatedDeltaCache) -> Result<Array> {
        let shape = inputs.shape();
        let (b, s) = (shape[0], shape[1]);

        let qkv = self.in_proj_qkv.forward(inputs)?;
        let z = self.in_proj_z.forward(inputs)?;
        let z = ops::reshape(&z, &[b, s, self.num_v_heads, self.head_v_dim])?;
        let bt = self.in_proj_b.forward(inputs)?;
        let at = self.in_proj_a.forward(inputs)?;

        let n_keep = self.conv_kernel_size - 1;
        let conv_state = cache
            .conv_state
            .take()
            .unwrap_or(ops::zeros(&[b, n_keep, self.conv_dim], inputs.dtype())?);
        let conv_input = ops::concatenate(&[&conv_state, &qkv], 1)?;
        let total_len = conv_input.dim(1);
        cache.conv_state = Some(ops::contiguous(&ops::slice(
            &conv_input,
            &[0, total_len - n_keep, 0],
            &[b, total_len, self.conv_dim],
        )?)?);

        let conv_out = ops::silu(&ops::conv1d(
            &conv_input,
            &self.conv1d_weight,
            1,
            0,
            1,
            self.conv_dim,
        )?)?;

        let parts = ops::split_sections(&conv_out, &[self.key_dim, 2 * self.key_dim], -1)?;
        let q = ops::reshape(&parts[0], &[b, s, self.num_k_heads, self.head_k_dim])?;
        let k = ops::reshape(&parts[1], &[b, s, self.num_k_heads, self.head_k_dim])?;
        let v = ops::reshape(&parts[2], &[b, s, self.num_v_heads, self.head_v_dim])?;

        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q = ops::scale_by(&ops::rms_norm(&q, None, 1e-6)?, inv_scale * inv_scale)?;
        let k = ops::scale_by(&ops::rms_norm(&k, None, 1e-6)?, inv_scale)?;

        let repeat_factor = self.num_v_heads / self.num_k_heads;
        let (q, k) = if repeat_factor > 1 {
            (
                ops::repeat_axis(&q, repeat_factor, 2)?,
                ops::repeat_axis(&k, repeat_factor, 2)?,
            )
        } else {
            (q, k)
        };

        // Recurrence runs in f32 for numerical stability, matching the
        // Python reference (`state` is created as float32).
        let q = ops::astype(&q, Dtype::Float32)?;
        let k = ops::astype(&k, Dtype::Float32)?;
        let v = ops::astype(&v, Dtype::Float32)?;
        let beta = ops::sigmoid(&ops::astype(&bt, Dtype::Float32)?)?;
        let a32 = ops::astype(&at, Dtype::Float32)?;
        let a_log32 = ops::astype(&self.a_log, Dtype::Float32)?;
        let dt_bias32 = ops::astype(&self.dt_bias, Dtype::Float32)?;
        // g = exp(-exp(A_log) * softplus(a + dt_bias)), shape [B, S, Hv].
        let g = {
            let inner = ops::softplus(&ops::add(&a32, &dt_bias32)?)?;
            let decay_rate = ops::exp(&a_log32)?;
            ops::exp(&ops::negative(&ops::multiply(&decay_rate, &inner)?)?)?
        };

        let mut state = cache.recur_state.take().unwrap_or(ops::zeros(
            &[b, self.num_v_heads, self.head_v_dim, self.head_k_dim],
            Dtype::Float32,
        )?);

        let mut ys = Vec::with_capacity(s as usize);
        for t in 0..s {
            let qt = ops::reshape(
                &ops::slice(
                    &q,
                    &[0, t, 0, 0],
                    &[b, t + 1, self.num_v_heads, self.head_k_dim],
                )?,
                &[b, self.num_v_heads, self.head_k_dim],
            )?;
            let kt = ops::reshape(
                &ops::slice(
                    &k,
                    &[0, t, 0, 0],
                    &[b, t + 1, self.num_v_heads, self.head_k_dim],
                )?,
                &[b, self.num_v_heads, self.head_k_dim],
            )?;
            let vt = ops::reshape(
                &ops::slice(
                    &v,
                    &[0, t, 0, 0],
                    &[b, t + 1, self.num_v_heads, self.head_v_dim],
                )?,
                &[b, self.num_v_heads, self.head_v_dim],
            )?;
            let gt = ops::reshape(
                &ops::slice(&g, &[0, t, 0], &[b, t + 1, self.num_v_heads])?,
                &[b, self.num_v_heads],
            )?;
            let betat = ops::reshape(
                &ops::slice(&beta, &[0, t, 0], &[b, t + 1, self.num_v_heads])?,
                &[b, self.num_v_heads],
            )?;

            let decay = ops::reshape(&gt, &[b, self.num_v_heads, 1, 1])?;
            state = ops::multiply(&state, &decay)?;
            let kt_row = ops::expand_dims(&kt, 2)?; // [B, Hv, 1, Dk]
            let kv_mem = ops::sum_axes(&ops::multiply(&state, &kt_row)?, &[-1], false)?; // [B, Hv, Dv]
            let delta = ops::multiply(
                &ops::subtract(&vt, &kv_mem)?,
                &ops::expand_dims(&betat, -1)?,
            )?; // [B,Hv,Dv]
            let update = ops::multiply(&kt_row, &ops::expand_dims(&delta, -1)?)?; // [B,Hv,Dv,Dk]
            state = ops::add(&state, &update)?;
            let qt_row = ops::expand_dims(&qt, 2)?; // [B, Hv, 1, Dk]
            let yt = ops::sum_axes(&ops::multiply(&state, &qt_row)?, &[-1], false)?; // [B, Hv, Dv]
            ys.push(ops::astype(&yt, inputs.dtype())?);
        }
        cache.recur_state = Some(state);

        let y_refs: Vec<&Array> = ys.iter().collect();
        let y = ops::stack_axis(&y_refs, 1)?; // [B, S, Hv, Dv]

        // RMSNormGated: normalize then gate with silu(z), all done in f32.
        let y32 = ops::astype(&y, Dtype::Float32)?;
        let normed = ops::rms_norm(&y32, None, self.norm_eps)?;
        let normed = ops::multiply(&normed, &ops::astype(&self.norm_weight, Dtype::Float32)?)?;
        let gated = ops::multiply(&ops::silu(&ops::astype(&z, Dtype::Float32)?)?, &normed)?;
        let out = ops::astype(&gated, inputs.dtype())?;
        let out = ops::reshape(&out, &[b, s, self.value_dim])?;

        self.out_proj.forward(&out)
    }
}
