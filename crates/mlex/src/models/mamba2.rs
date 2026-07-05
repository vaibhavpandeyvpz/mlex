//! Mamba2 SSM mixer used by NemotronH's hybrid layers. Ports mlx-lm's
//! `NemotronHMamba2Mixer` + `ssm.py` reference math as a straightforward
//! per-timestep recurrence (mathematically equivalent to the chunked
//! "SSD" formulation `ssm_attn` uses for training-time parallelism, and to
//! the fused Metal `ssm_kernel` used for single-token decode) rather than
//! porting the chunked matrix algorithm — same pattern as `gated_delta.rs`.
//!
//! Batch size is assumed to be 1 (single-sequence generation).

use crate::array::{Array, Dtype};
use crate::error::Result;
use crate::nn::{Linear, WeightMap};
use crate::ops;

use super::cache::GatedDeltaCache;

pub struct Mamba2Config {
    pub num_heads: i32,
    pub head_dim: i32,
    pub n_groups: i32,
    pub state_size: i32,
    pub conv_kernel: i32,
    pub proj_bias: bool,
    pub conv_bias: bool,
    pub norm_eps: f32,
    pub time_step_min: f32,
    pub time_step_max: f32,
}

pub struct Mamba2Mixer {
    conv1d_weight: Array,
    conv1d_bias: Option<Array>,
    in_proj: Linear,
    dt_bias: Array,
    a_log: Array,
    d: Array,
    norm_weight: Array,
    norm_eps: f32,
    out_proj: Linear,

    num_heads: i32,
    head_dim: i32,
    n_groups: i32,
    state_size: i32,
    conv_kernel: i32,
    intermediate_size: i32,
    conv_dim: i32,
    time_step_min: f32,
    time_step_max: f32,
}

impl Mamba2Mixer {
    pub fn load(w: &mut WeightMap, prefix: &str, cfg: &Mamba2Config) -> Result<Self> {
        let intermediate_size = cfg.num_heads * cfg.head_dim;
        let conv_dim = intermediate_size + 2 * cfg.n_groups * cfg.state_size;

        let conv1d_bias = if cfg.conv_bias {
            Some(w.take(&format!("{prefix}.conv1d.bias"))?)
        } else {
            None
        };

        Ok(Mamba2Mixer {
            conv1d_weight: w.take(&format!("{prefix}.conv1d.weight"))?,
            conv1d_bias,
            in_proj: w.linear(&format!("{prefix}.in_proj"))?,
            dt_bias: w.take(&format!("{prefix}.dt_bias"))?,
            a_log: w.take(&format!("{prefix}.A_log"))?,
            d: w.take(&format!("{prefix}.D"))?,
            norm_weight: w.take(&format!("{prefix}.norm.weight"))?,
            norm_eps: cfg.norm_eps,
            out_proj: w.linear(&format!("{prefix}.out_proj"))?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            n_groups: cfg.n_groups,
            state_size: cfg.state_size,
            conv_kernel: cfg.conv_kernel,
            intermediate_size,
            conv_dim,
            time_step_min: cfg.time_step_min,
            time_step_max: cfg.time_step_max,
        })
    }

    pub fn forward(&self, hidden_states: &Array, cache: &mut GatedDeltaCache) -> Result<Array> {
        let shape = hidden_states.shape();
        let (b, s) = (shape[0], shape[1]);

        let projected = self.in_proj.forward(hidden_states)?;
        let parts = ops::split_sections(
            &projected,
            &[
                self.intermediate_size,
                self.intermediate_size + self.conv_dim,
            ],
            -1,
        )?;
        let (gate, conv_input, dt_raw) = (&parts[0], &parts[1], &parts[2]);

        let n_keep = self.conv_kernel - 1;
        let conv_state = cache.conv_state.take().unwrap_or(ops::zeros(
            &[b, n_keep, self.conv_dim],
            hidden_states.dtype(),
        )?);
        let padded = ops::concatenate(&[&conv_state, conv_input], 1)?;
        let total_len = padded.dim(1);
        cache.conv_state = Some(ops::contiguous(&ops::slice(
            &padded,
            &[0, total_len - n_keep, 0],
            &[b, total_len, self.conv_dim],
        )?)?);

        let mut conv_out = ops::conv1d(&padded, &self.conv1d_weight, 1, 0, 1, self.conv_dim)?;
        if let Some(bias) = &self.conv1d_bias {
            conv_out = ops::add(&conv_out, bias)?;
        }
        let conv_out = ops::silu(&conv_out)?;

        let gs = self.n_groups * self.state_size;
        let parts = ops::split_sections(
            &conv_out,
            &[self.intermediate_size, self.intermediate_size + gs],
            -1,
        )?;
        let (x, bmat, cmat) = (&parts[0], &parts[1], &parts[2]);

        let x = ops::reshape(x, &[b, s, self.num_heads, self.head_dim])?;
        let bmat = ops::reshape(bmat, &[b, s, self.n_groups, self.state_size])?;
        let cmat = ops::reshape(cmat, &[b, s, self.n_groups, self.state_size])?;

        // f32 recurrence for numerical stability, matching the Python
        // reference's fused-kernel accumulation precision.
        let x = ops::astype(&x, Dtype::Float32)?;
        let bmat = ops::astype(&bmat, Dtype::Float32)?;
        let cmat = ops::astype(&cmat, Dtype::Float32)?;
        let dt_raw = ops::astype(dt_raw, Dtype::Float32)?;
        let dt_bias32 = ops::astype(&self.dt_bias, Dtype::Float32)?;
        let a_log32 = ops::astype(&self.a_log, Dtype::Float32)?;
        let d32 = ops::astype(&self.d, Dtype::Float32)?;

        // dt = clip(softplus(dt_raw + dt_bias), min, max), shape [B, S, H].
        let dt = {
            let sp = ops::softplus(&ops::add(&dt_raw, &dt_bias32)?)?;
            let lo = ops::maximum(&sp, &Array::scalar_f32(self.time_step_min))?;
            ops::minimum(&lo, &Array::scalar_f32(self.time_step_max))?
        };
        let neg_a = ops::negative(&ops::exp(&a_log32)?)?; // [H]

        let repeat_factor = self.num_heads / self.n_groups;
        let (bmat, cmat) = if repeat_factor > 1 {
            (
                ops::repeat_axis(&bmat, repeat_factor, 2)?,
                ops::repeat_axis(&cmat, repeat_factor, 2)?,
            )
        } else {
            (bmat, cmat)
        };

        let mut state = cache.recur_state.take().unwrap_or(ops::zeros(
            &[b, self.num_heads, self.head_dim, self.state_size],
            Dtype::Float32,
        )?);

        let mut ys = Vec::with_capacity(s as usize);
        for t in 0..s {
            let xt = ops::reshape(
                &ops::slice(
                    &x,
                    &[0, t, 0, 0],
                    &[b, t + 1, self.num_heads, self.head_dim],
                )?,
                &[b, self.num_heads, self.head_dim],
            )?;
            let bt = ops::reshape(
                &ops::slice(
                    &bmat,
                    &[0, t, 0, 0],
                    &[b, t + 1, self.num_heads, self.state_size],
                )?,
                &[b, self.num_heads, self.state_size],
            )?;
            let ct = ops::reshape(
                &ops::slice(
                    &cmat,
                    &[0, t, 0, 0],
                    &[b, t + 1, self.num_heads, self.state_size],
                )?,
                &[b, self.num_heads, self.state_size],
            )?;
            let dtt = ops::reshape(
                &ops::slice(&dt, &[0, t, 0], &[b, t + 1, self.num_heads])?,
                &[b, self.num_heads],
            )?;

            let da = ops::exp(&ops::multiply(&dtt, &neg_a)?)?; // [B, H]
            let da = ops::reshape(&da, &[b, self.num_heads, 1, 1])?;

            let dbx = {
                let x_col = ops::expand_dims(&xt, -1)?; // [B,H,Dh,1]
                let dt_col = ops::reshape(&dtt, &[b, self.num_heads, 1, 1])?;
                let b_row = ops::expand_dims(&bt, 2)?; // [B,H,1,Ds]
                ops::multiply(&ops::multiply(&x_col, &dt_col)?, &b_row)?
            };
            state = ops::add(&ops::multiply(&state, &da)?, &dbx)?;

            let c_row = ops::expand_dims(&ct, 2)?; // [B,H,1,Ds]
            let y_ssm = ops::sum_axes(&ops::multiply(&state, &c_row)?, &[-1], false)?; // [B,H,Dh]
            let d_row = ops::reshape(&d32, &[1, self.num_heads, 1])?;
            let yt = ops::add(&y_ssm, &ops::multiply(&xt, &d_row)?)?;
            ys.push(yt);
        }
        cache.recur_state = Some(state);

        let y_refs: Vec<&Array> = ys.iter().collect();
        let y = ops::stack_axis(&y_refs, 1)?; // [B, S, H, Dh]
        let y = ops::reshape(&y, &[b, s, self.intermediate_size])?;

        // MambaRMSNormGated: swiglu(gate, y) then grouped RMSNorm.
        let gate32 = ops::astype(gate, Dtype::Float32)?;
        let gated = ops::multiply(&ops::silu(&gate32)?, &y)?;
        let group_size = self.intermediate_size / self.n_groups;
        let n_groups_flat = self.intermediate_size / group_size;
        let grouped = ops::reshape(&gated, &[b, s, n_groups_flat, group_size])?;
        let normed = ops::rms_norm(&grouped, None, self.norm_eps)?;
        let normed = ops::reshape(&normed, &[b, s, self.intermediate_size])?;
        let normed = ops::multiply(&normed, &ops::astype(&self.norm_weight, Dtype::Float32)?)?;
        let out = ops::astype(&normed, hidden_states.dtype())?;

        self.out_proj.forward(&out)
    }
}
