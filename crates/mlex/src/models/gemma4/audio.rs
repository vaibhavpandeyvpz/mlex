//! Gemma4 Conformer-style audio tower (`audio_config.model_type ==
//! "gemma4_audio"`).
//!
//! Implemented for the single-clip inference path this crate needs: instead
//! of a blocked/chunked local-attention kernel (an efficiency trick for
//! streaming/batched inference), we run plain full attention with an
//! additive banded mask - the chunked mask reduces to "attend to self plus
//! the previous `attention_context_left - 2` positions", which the dense
//! mask expresses exactly, and the Transformer-XL relative-shift collapses
//! to a direct gather of the per-distance relative-position bias.
//!
//! Structure per Conformer block: half-step FFN -> local self-attention
//! with sinusoidal relative position embeddings, per-dim query scaling and
//! logit softcapping -> GLU + causal depthwise conv module -> half-step
//! FFN -> output RMSNorm. Front-end: two stride-2 Conv2D subsampling
//! layers over the `[T, 128]` log-mel spectrogram + a linear projection.
//!
//! Weight key layout (matching the checkpoint verbatim, no renaming):
//!   `audio_tower.subsample_conv_projection.layer{0,1}.{conv,norm}.weight`
//!   `audio_tower.subsample_conv_projection.input_proj_linear.weight`
//!   `audio_tower.layers.{i}.feed_forward{1,2}.{pre,post}_layer_norm.weight`
//!   `audio_tower.layers.{i}.feed_forward{1,2}.ffw_layer_{1,2}.(linear.)weight`
//!   `audio_tower.layers.{i}.self_attn.{q,k,v}_proj.(linear.)weight`
//!   `audio_tower.layers.{i}.self_attn.{post.(linear.)weight, relative_k_proj.weight, per_dim_scale}`
//!   `audio_tower.layers.{i}.norm_{pre_attn,post_attn,out}.weight`
//!   `audio_tower.layers.{i}.lconv1d.{pre_layer_norm,linear_start,depthwise_conv1d,conv_norm,linear_end}`
//!   `audio_tower.output_proj.{weight,bias}`
//!   `embed_audio.embedding_projection.weight`

use serde_json::Value;

use crate::array::{Array, Dtype};
use crate::error::Result;
use crate::nn::{Linear, RmsNorm, WeightMap};
use crate::ops;

use super::vision::ClippedLinear;

/// Parsed `audio_config` sub-dict of a Gemma4 multimodal checkpoint.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub conv_kernel_size: i32,
    pub rms_norm_eps: f32,
    pub attention_context_left: i32,
    pub attention_logit_cap: f32,
    pub attention_invalid_logits_value: f32,
    pub residual_weight: f32,
    pub use_clipped_linears: bool,
}

impl AudioConfig {
    pub fn from_json(cfg: &Value) -> Self {
        let gi = |k: &str, d: i32| {
            cfg.get(k)
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(d)
        };
        let gf = |k: &str, d: f32| {
            cfg.get(k)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(d)
        };
        let gb = |k: &str, d: bool| cfg.get(k).and_then(|v| v.as_bool()).unwrap_or(d);
        AudioConfig {
            hidden_size: gi("hidden_size", 1024),
            num_hidden_layers: gi("num_hidden_layers", 12),
            num_attention_heads: gi("num_attention_heads", 8),
            conv_kernel_size: gi("conv_kernel_size", 5),
            rms_norm_eps: gf("rms_norm_eps", 1e-6),
            attention_context_left: gi("attention_context_left", 13),
            attention_logit_cap: gf("attention_logit_cap", 50.0),
            attention_invalid_logits_value: gf("attention_invalid_logits_value", -1e9),
            residual_weight: gf("residual_weight", 0.5),
            use_clipped_linears: gb("use_clipped_linears", false),
        }
    }

    fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Maximum past distance a query attends to (self excluded); the
    /// reference's `max_past_horizon = attention_context_left - 1`.
    fn max_past(&self) -> i32 {
        self.attention_context_left - 1
    }
}

/// LayerNorm over the last axis, learned scale, no bias (the SSCP conv
/// norm; `ggml_norm` + weight-mul in the reference).
fn layer_norm_no_bias(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let xf = ops::astype(x, Dtype::Float32)?;
    let mean = ops::mean_axes(&xf, &[-1], true)?;
    let centered = ops::subtract(&xf, &mean)?;
    let var = ops::mean_axes(&ops::square(&centered)?, &[-1], true)?;
    let inv_std = ops::rsqrt(&ops::add_scalar(&var, eps)?)?;
    let normed = ops::multiply(&centered, &inv_std)?;
    let normed = ops::astype(&normed, x.dtype())?;
    ops::multiply(&normed, weight)
}

/// One half-step feed-forward module: RMSNorm -> up (silu) -> down ->
/// RMSNorm; caller adds `residual_weight *` this to the residual stream.
struct HalfStepFfn {
    pre_layer_norm: RmsNorm,
    ffw_layer_1: ClippedLinear,
    ffw_layer_2: ClippedLinear,
    post_layer_norm: RmsNorm,
}

impl HalfStepFfn {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &AudioConfig) -> Result<Self> {
        let clip = cfg.use_clipped_linears;
        Ok(HalfStepFfn {
            pre_layer_norm: w.rms_norm(&format!("{prefix}.pre_layer_norm"), cfg.rms_norm_eps)?,
            ffw_layer_1: ClippedLinear::load(w, &format!("{prefix}.ffw_layer_1"), clip)?,
            ffw_layer_2: ClippedLinear::load(w, &format!("{prefix}.ffw_layer_2"), clip)?,
            post_layer_norm: w.rms_norm(&format!("{prefix}.post_layer_norm"), cfg.rms_norm_eps)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.pre_layer_norm.forward(x)?;
        let h = ops::silu(&self.ffw_layer_1.forward(&h)?)?;
        let h = self.ffw_layer_2.forward(&h)?;
        self.post_layer_norm.forward(&h)
    }
}

/// Local self-attention with sinusoidal relative position embeddings.
struct AudioAttention {
    norm_pre_attn: RmsNorm,
    q_proj: ClippedLinear,
    k_proj: ClippedLinear,
    v_proj: ClippedLinear,
    post: ClippedLinear,
    norm_post_attn: RmsNorm,
    relative_k_proj: Linear,
    /// `softplus(per_dim_scale)`, precomputed at load (`[head_dim]`).
    per_dim_scale: Array,
    n_heads: i32,
    head_dim: i32,
    max_past: i32,
    logit_cap: f32,
    invalid_logit: f32,
}

impl AudioAttention {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &AudioConfig) -> Result<Self> {
        let attn = format!("{prefix}.self_attn");
        let clip = cfg.use_clipped_linears;
        let per_dim_scale_raw = w.take(&format!("{attn}.per_dim_scale"))?;
        let per_dim_scale = ops::softplus(&ops::astype(&per_dim_scale_raw, Dtype::Float32)?)?;
        Ok(AudioAttention {
            norm_pre_attn: w.rms_norm(&format!("{prefix}.norm_pre_attn"), cfg.rms_norm_eps)?,
            q_proj: ClippedLinear::load(w, &format!("{attn}.q_proj"), clip)?,
            k_proj: ClippedLinear::load(w, &format!("{attn}.k_proj"), clip)?,
            v_proj: ClippedLinear::load(w, &format!("{attn}.v_proj"), clip)?,
            post: ClippedLinear::load(w, &format!("{attn}.post"), clip)?,
            norm_post_attn: w.rms_norm(&format!("{prefix}.norm_post_attn"), cfg.rms_norm_eps)?,
            relative_k_proj: w.linear(&format!("{attn}.relative_k_proj"))?,
            per_dim_scale,
            n_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            max_past: cfg.max_past(),
            logit_cap: cfg.attention_logit_cap,
            invalid_logit: cfg.attention_invalid_logits_value,
        })
    }

    /// Sinusoidal relative position embedding table, one row per distance
    /// `d = 0..=max_past` (`[R, hidden]`, matching the reference's
    /// `pos_emb` fill with position values `max_past..0` reordered by
    /// distance).
    fn build_rel_pos_table(&self, dtype: Dtype) -> Result<Array> {
        let hidden = (self.n_heads * self.head_dim) as usize;
        let num_timescales = hidden / 2;
        let log_timescale_increment = 10000f32.ln() / (num_timescales as f32 - 1.0).max(1.0);
        let r = (self.max_past + 1) as usize;
        let mut data = vec![0f32; r * hidden];
        for d in 0..r {
            for i in 0..num_timescales {
                let inv_ts = (-(i as f32) * log_timescale_increment).exp();
                let scaled = d as f32 * inv_ts;
                data[d * hidden + i] = scaled.sin();
                data[d * hidden + i + num_timescales] = scaled.cos();
            }
        }
        let table = Array::from_slice(&data, &[r as i32, hidden as i32]);
        ops::astype(&table, dtype)
    }

    /// Banded additive attention mask: `0` where `0 <= i - j < max_past`,
    /// `invalid_logit` elsewhere (`[1, 1, T, T]`). Matches the reference's
    /// blocked causal mask condition `gk <= gq && (gq - gk) < max_past`.
    fn build_mask(&self, t: i32) -> Array {
        let tu = t as usize;
        let mut data = vec![self.invalid_logit; tu * tu];
        for i in 0..tu {
            let lo = i.saturating_sub(self.max_past as usize - 1);
            for j in lo..=i {
                data[i * tu + j] = 0.0;
            }
        }
        Array::from_slice(&data, &[1, 1, t, t])
    }

    /// Gather indices mapping each `(i, j)` score position to its
    /// relative-distance bias row `clamp(i - j, 0, max_past)`
    /// (`[1, 1, T, T]` int32; out-of-band positions are masked anyway).
    fn build_distance_indices(&self, t: i32) -> Array {
        let tu = t as usize;
        let mut data = vec![0i32; tu * tu];
        for i in 0..tu {
            for j in 0..tu {
                data[i * tu + j] = (i as i32 - j as i32).clamp(0, self.max_past);
            }
        }
        Array::from_slice(&data, &[1, 1, t, t])
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, t) = (shape[0], shape[1]);

        let h = self.norm_pre_attn.forward(x)?;
        let q = self.q_proj.forward(&h)?;
        let k = self.k_proj.forward(&h)?;
        let v = self.v_proj.forward(&h)?;

        let q = ops::transpose_axes(
            &ops::reshape(&q, &[b, t, self.n_heads, self.head_dim])?,
            &[0, 2, 1, 3],
        )?;
        let k = ops::transpose_axes(
            &ops::reshape(&k, &[b, t, self.n_heads, self.head_dim])?,
            &[0, 2, 1, 3],
        )?;
        let v = ops::transpose_axes(
            &ops::reshape(&v, &[b, t, self.n_heads, self.head_dim])?,
            &[0, 2, 1, 3],
        )?;

        // Query scaling: (1/sqrt(d)) / softplus(0) * softplus(per_dim_scale);
        // key scaling: softplus(1) / softplus(0) (reference's q_scale/k_scale).
        let ln2 = std::f32::consts::LN_2;
        let q_scale = (1.0 / (self.head_dim as f32).sqrt()) / ln2;
        let k_scale = 1f32.exp().ln_1p() / ln2;
        let per_dim = ops::astype(&self.per_dim_scale, q.dtype())?;
        let q = ops::multiply(&ops::scale_by(&q, q_scale)?, &per_dim)?;
        let k = ops::scale_by(&k, k_scale)?;

        // Content scores + per-distance relative position bias.
        let kt = ops::swapaxes(&k, -1, -2)?;
        let scores = ops::matmul(&q, &kt)?; // [B, H, T, T]

        let rel_table = self.build_rel_pos_table(q.dtype())?; // [R, hidden]
        let k_rel = self.relative_k_proj.forward(&rel_table)?; // [R, hidden]
        let r = self.max_past + 1;
        let k_rel = ops::transpose_axes(
            &ops::reshape(&k_rel, &[r, self.n_heads, self.head_dim])?,
            &[1, 0, 2],
        )?;
        let k_rel = ops::expand_dims(&k_rel, 0)?; // [1, H, R, D]
        let qrel = ops::matmul(&q, &ops::swapaxes(&k_rel, -1, -2)?)?; // [B, H, T, R]

        let idx = self.build_distance_indices(t);
        let idx = ops::broadcast_to(&idx, &[b, self.n_heads, t, t])?;
        let bias = ops::take_along_axis(&qrel, &idx, -1)?; // [B, H, T, T]
        let scores = ops::add(&scores, &bias)?;

        // Logit softcapping, then the banded local-attention mask.
        let cap = self.logit_cap;
        let scores = ops::scale_by(&ops::tanh(&ops::scale_by(&scores, 1.0 / cap)?)?, cap)?;
        let mask = ops::astype(&self.build_mask(t), scores.dtype())?;
        let scores = ops::add(&scores, &mask)?;

        let attn = ops::softmax_axis(&scores, -1, true)?;
        let out = ops::matmul(&attn, &v)?; // [B, H, T, D]
        let out = ops::reshape(&ops::transpose_axes(&out, &[0, 2, 1, 3])?, &[b, t, -1])?;
        let out = self.post.forward(&out)?;
        self.norm_post_attn.forward(&out)
    }
}

/// GLU + causal depthwise convolution module.
struct LightConv1d {
    pre_layer_norm: RmsNorm,
    linear_start: ClippedLinear,
    /// `[C, K, 1]` depthwise conv weight (MLX conv1d layout).
    depthwise_conv1d: Array,
    conv_norm: RmsNorm,
    linear_end: ClippedLinear,
    kernel_size: i32,
    channels: i32,
}

impl LightConv1d {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &AudioConfig) -> Result<Self> {
        let conv = format!("{prefix}.lconv1d");
        let clip = cfg.use_clipped_linears;
        Ok(LightConv1d {
            pre_layer_norm: w.rms_norm(&format!("{conv}.pre_layer_norm"), cfg.rms_norm_eps)?,
            linear_start: ClippedLinear::load(w, &format!("{conv}.linear_start"), clip)?,
            depthwise_conv1d: w.take(&format!("{conv}.depthwise_conv1d.weight"))?,
            conv_norm: w.rms_norm(&format!("{conv}.conv_norm"), cfg.rms_norm_eps)?,
            linear_end: ClippedLinear::load(w, &format!("{conv}.linear_end"), clip)?,
            kernel_size: cfg.conv_kernel_size,
            channels: cfg.hidden_size,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, t) = (shape[0], shape[1]);

        let h = self.pre_layer_norm.forward(x)?;
        let h = self.linear_start.forward(&h)?; // [B, T, 2C]

        // GLU: first half gated by sigmoid of the second half.
        let c = self.channels;
        let first = ops::slice(&h, &[0, 0, 0], &[b, t, c])?;
        let second = ops::slice(&h, &[0, 0, c], &[b, t, 2 * c])?;
        let h = ops::multiply(&first, &ops::sigmoid(&second)?)?;

        // Causal depthwise conv: left-pad by kernel_size - 1.
        let pad = ops::zeros(&[b, self.kernel_size - 1, c], h.dtype())?;
        let padded = ops::concatenate(&[&pad, &h], 1)?;
        let weight = ops::astype(&self.depthwise_conv1d, h.dtype())?;
        let h = ops::conv1d(&padded, &weight, 1, 0, 1, c)?; // [B, T, C]

        let h = self.conv_norm.forward(&h)?;
        let h = ops::silu(&h)?;
        self.linear_end.forward(&h)
    }
}

/// One Conformer block.
struct AudioBlock {
    feed_forward1: HalfStepFfn,
    self_attn: AudioAttention,
    lconv1d: LightConv1d,
    feed_forward2: HalfStepFfn,
    norm_out: RmsNorm,
    residual_weight: f32,
}

impl AudioBlock {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &AudioConfig) -> Result<Self> {
        Ok(AudioBlock {
            feed_forward1: HalfStepFfn::load(w, &format!("{prefix}.feed_forward1"), cfg)?,
            self_attn: AudioAttention::load(w, prefix, cfg)?,
            lconv1d: LightConv1d::load(w, prefix, cfg)?,
            feed_forward2: HalfStepFfn::load(w, &format!("{prefix}.feed_forward2"), cfg)?,
            norm_out: w.rms_norm(&format!("{prefix}.norm_out"), cfg.rms_norm_eps)?,
            residual_weight: cfg.residual_weight,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let ff1 = self.feed_forward1.forward(x)?;
        let residual = ops::add(x, &ops::scale_by(&ff1, self.residual_weight)?)?;

        let attn = self.self_attn.forward(&residual)?;
        let residual = ops::add(&residual, &attn)?;

        let conv = self.lconv1d.forward(&residual)?;
        let residual = ops::add(&residual, &conv)?;

        let ff2 = self.feed_forward2.forward(&residual)?;
        let residual = ops::add(&residual, &ops::scale_by(&ff2, self.residual_weight)?)?;

        self.norm_out.forward(&residual)
    }
}

/// Subsampling conv front-end: two stride-2 Conv2D layers (LayerNorm +
/// ReLU after each) over the `[T, n_mels]` spectrogram treated as a
/// 1-channel image, flattened and projected to `hidden_size`.
struct SubsampleConvProjection {
    conv0: Array,
    norm0: Array,
    conv1: Array,
    norm1: Array,
    input_proj_linear: ClippedLinear,
    eps: f32,
}

impl SubsampleConvProjection {
    fn load(w: &mut WeightMap, prefix: &str, cfg: &AudioConfig) -> Result<Self> {
        Ok(SubsampleConvProjection {
            conv0: w.take(&format!("{prefix}.layer0.conv.weight"))?,
            norm0: w.take(&format!("{prefix}.layer0.norm.weight"))?,
            conv1: w.take(&format!("{prefix}.layer1.conv.weight"))?,
            norm1: w.take(&format!("{prefix}.layer1.norm.weight"))?,
            input_proj_linear: ClippedLinear::load(
                w,
                &format!("{prefix}.input_proj_linear"),
                cfg.use_clipped_linears,
            )?,
            eps: cfg.rms_norm_eps,
        })
    }

    /// `[1, T, n_mels]` -> `[1, T', hidden]` where `T' = ((T-1)/2+1)` twice.
    fn forward(&self, mel: &Array) -> Result<Array> {
        let shape = mel.shape();
        let (b, t, f) = (shape[0], shape[1], shape[2]);
        let x = ops::reshape(mel, &[b, t, f, 1])?;
        let x = ops::astype(&x, self.conv0.dtype())?;

        let x = ops::conv2d(&x, &self.conv0, (2, 2), (1, 1), (1, 1), 1)?;
        let x = layer_norm_no_bias(&x, &self.norm0, self.eps)?;
        let x = ops::maximum(&x, &ops::astype(&Array::scalar_f32(0.0), x.dtype())?)?;

        let x = ops::conv2d(&x, &self.conv1, (2, 2), (1, 1), (1, 1), 1)?;
        let x = layer_norm_no_bias(&x, &self.norm1, self.eps)?;
        let x = ops::maximum(&x, &ops::astype(&Array::scalar_f32(0.0), x.dtype())?)?;

        // [B, T', F', C] -> [B, T', F'*C] (channel fastest, matching the
        // reference's `[freq, time, ch] -> [ch*freq, time]` flatten).
        let out_shape = x.shape();
        let (t2, f2, c2) = (out_shape[1], out_shape[2], out_shape[3]);
        let x = ops::reshape(&x, &[b, t2, f2 * c2])?;
        self.input_proj_linear.forward(&x)
    }
}

fn debug_stats(label: &str, x: &Array) {
    if let Ok(v) = x.to_vec_f32() {
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "[audio debug] {label}: shape={:?} mean={mean} min={min} max={max}",
            x.shape()
        );
    }
}

/// Top-level Conformer audio encoder: subsampling conv front-end -> N
/// Conformer blocks -> output projection to the multimodal width.
pub struct AudioTower {
    subsample_conv_projection: SubsampleConvProjection,
    layers: Vec<AudioBlock>,
    output_proj: Linear,
}

impl AudioTower {
    pub fn load(w: &mut WeightMap, cfg: AudioConfig) -> Result<Self> {
        let subsample_conv_projection =
            SubsampleConvProjection::load(w, "audio_tower.subsample_conv_projection", &cfg)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            layers.push(AudioBlock::load(
                w,
                &format!("audio_tower.layers.{i}"),
                &cfg,
            )?);
        }
        let output_proj = w.linear("audio_tower.output_proj")?;
        Ok(AudioTower {
            subsample_conv_projection,
            layers,
            output_proj,
        })
    }

    /// Run the encoder over one chunk's `[1, T, n_mels]` log-mel tensor,
    /// returning `[1, T', output_proj_dims]`.
    pub fn forward(&self, mel: &Array) -> Result<Array> {
        let debug = std::env::var("MLEX_DEBUG_AUDIO").is_ok();
        let mut h = self.subsample_conv_projection.forward(mel)?;
        if debug {
            debug_stats("subsample_conv_projection", &h);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if debug {
                debug_stats(&format!("layer_{i}"), &h);
            }
        }
        self.output_proj.forward(&h)
    }
}
