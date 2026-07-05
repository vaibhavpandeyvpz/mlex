//! Safe wrappers over the mlx-c operation set used by the model runtimes.

use std::ffi::CString;
use std::ptr;

use crate::array::{Array, Dtype};
use crate::error::{check, Result};
use crate::stream::stream;

macro_rules! unary_op {
    ($name:ident, $ffi:ident) => {
        pub fn $name(a: &Array) -> Result<Array> {
            let mut res = Array::new_handle();
            unsafe {
                check(crate::sys::$ffi(&mut res, a.raw, stream()))?;
            }
            Ok(Array::from_raw(res))
        }
    };
}

macro_rules! binary_op {
    ($name:ident, $ffi:ident) => {
        pub fn $name(a: &Array, b: &Array) -> Result<Array> {
            let mut res = Array::new_handle();
            unsafe {
                check(crate::sys::$ffi(&mut res, a.raw, b.raw, stream()))?;
            }
            Ok(Array::from_raw(res))
        }
    };
}

unary_op!(exp, mlx_exp);
unary_op!(sigmoid, mlx_sigmoid);
unary_op!(tanh, mlx_tanh);
unary_op!(rsqrt, mlx_rsqrt);
unary_op!(sqrt, mlx_sqrt);
unary_op!(square, mlx_square);
unary_op!(erf, mlx_erf);
unary_op!(negative, mlx_negative);
unary_op!(abs, mlx_abs);
unary_op!(log, mlx_log);
unary_op!(sin, mlx_sin);
unary_op!(cos, mlx_cos);
unary_op!(log1p, mlx_log1p);

binary_op!(add, mlx_add);
binary_op!(subtract, mlx_subtract);
binary_op!(multiply, mlx_multiply);
binary_op!(divide, mlx_divide);
binary_op!(matmul, mlx_matmul);
binary_op!(maximum, mlx_maximum);
binary_op!(minimum, mlx_minimum);
binary_op!(power, mlx_power);
binary_op!(greater, mlx_greater);
binary_op!(greater_equal, mlx_greater_equal);
binary_op!(less, mlx_less);
binary_op!(less_equal, mlx_less_equal);
binary_op!(equal, mlx_equal);
binary_op!(logaddexp, mlx_logaddexp);
binary_op!(logical_and, mlx_logical_and);

pub fn astype(a: &Array, dtype: Dtype) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_astype(
            &mut res,
            a.raw,
            dtype.to_raw(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_reshape(
            &mut res,
            a.raw,
            shape.as_ptr(),
            shape.len(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn transpose_axes(a: &Array, axes: &[i32]) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_transpose_axes(
            &mut res,
            a.raw,
            axes.as_ptr(),
            axes.len(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn swapaxes(a: &Array, axis1: i32, axis2: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_swapaxes(
            &mut res,
            a.raw,
            axis1,
            axis2,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn expand_dims(a: &Array, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_expand_dims(&mut res, a.raw, axis, stream()))?;
    }
    Ok(Array::from_raw(res))
}

pub fn squeeze_axis(a: &Array, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_squeeze_axis(
            &mut res,
            a.raw,
            axis,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn flatten_from(a: &Array, start_axis: i32) -> Result<Array> {
    let shape = a.shape();
    let ndim = shape.len() as i32;
    let start = if start_axis < 0 {
        start_axis + ndim
    } else {
        start_axis
    } as usize;
    let mut new_shape: Vec<i32> = shape[..start].to_vec();
    new_shape.push(shape[start..].iter().product());
    reshape(a, &new_shape)
}

pub fn slice(a: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    let strides = vec![1i32; start.len()];
    slice_strided(a, start, stop, &strides)
}

pub fn slice_strided(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_slice(
            &mut res,
            a.raw,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn slice_update(src: &Array, update: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    let strides = vec![1i32; start.len()];
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_slice_update(
            &mut res,
            src.raw,
            update.raw,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        let vec = crate::sys::mlx_vector_array_new();
        for a in arrays {
            check(crate::sys::mlx_vector_array_append_value(vec, a.raw))?;
        }
        let status = crate::sys::mlx_concatenate_axis(&mut res, vec, axis, stream());
        crate::sys::mlx_vector_array_free(vec);
        check(status)?;
    }
    Ok(Array::from_raw(res))
}

pub fn split(a: &Array, num_splits: i32, axis: i32) -> Result<Vec<Array>> {
    unsafe {
        let mut vec = crate::sys::mlx_vector_array_new();
        let status = crate::sys::mlx_split(&mut vec, a.raw, num_splits, axis, stream());
        if status != 0 {
            crate::sys::mlx_vector_array_free(vec);
            check(status)?;
        }
        let out = collect_vector(vec)?;
        crate::sys::mlx_vector_array_free(vec);
        Ok(out)
    }
}

pub fn split_sections(a: &Array, indices: &[i32], axis: i32) -> Result<Vec<Array>> {
    unsafe {
        let mut vec = crate::sys::mlx_vector_array_new();
        let status = crate::sys::mlx_split_sections(
            &mut vec,
            a.raw,
            indices.as_ptr(),
            indices.len(),
            axis,
            stream(),
        );
        if status != 0 {
            crate::sys::mlx_vector_array_free(vec);
            check(status)?;
        }
        let out = collect_vector(vec)?;
        crate::sys::mlx_vector_array_free(vec);
        Ok(out)
    }
}

unsafe fn collect_vector(vec: crate::sys::mlx_vector_array) -> Result<Vec<Array>> {
    let n = crate::sys::mlx_vector_array_size(vec);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut item = Array::new_handle();
        check(crate::sys::mlx_vector_array_get(&mut item, vec, i))?;
        out.push(Array::from_raw(item));
    }
    Ok(out)
}

pub fn stack_axis(arrays: &[&Array], axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        let vec = crate::sys::mlx_vector_array_new();
        for a in arrays {
            check(crate::sys::mlx_vector_array_append_value(vec, a.raw))?;
        }
        let status = crate::sys::mlx_stack_axis(&mut res, vec, axis, stream());
        crate::sys::mlx_vector_array_free(vec);
        check(status)?;
    }
    Ok(Array::from_raw(res))
}

pub fn repeat_axis(a: &Array, repeats: i32, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_repeat_axis(
            &mut res,
            a.raw,
            repeats,
            axis,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Numerically stable `softplus(x) = log(1 + exp(x))`.
pub fn softplus(x: &Array) -> Result<Array> {
    let zero = Array::scalar_f32(0.0);
    let relu = maximum(x, &zero)?;
    let neg_abs = negative(&abs(x)?)?;
    add(&relu, &log1p(&exp(&neg_abs)?)?)
}

/// Depthwise-capable 1D convolution. `input` is `[B, L, C_in]`, `weight` is
/// `[C_out, K, C_in / groups]` (MLX layout).
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_conv1d(
            &mut res,
            input.raw,
            weight.raw,
            stride,
            padding,
            dilation,
            groups,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// 2D convolution. `input` is `[B, H, W, C_in]`, `weight` is
/// `[C_out, KH, KW, C_in / groups]` (MLX layout).
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &Array,
    weight: &Array,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_conv2d(
            &mut res,
            input.raw,
            weight.raw,
            stride.0,
            stride.1,
            padding.0,
            padding.1,
            dilation.0,
            dilation.1,
            groups,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn softmax_axis(a: &Array, axis: i32, precise: bool) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_softmax_axis(
            &mut res,
            a.raw,
            axis,
            precise,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn argmax_axis(a: &Array, axis: i32, keepdims: bool) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_argmax_axis(
            &mut res,
            a.raw,
            axis,
            keepdims,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn take(a: &Array, indices: &Array) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_take(&mut res, a.raw, indices.raw, stream()))?;
    }
    Ok(Array::from_raw(res))
}

pub fn take_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_take_axis(
            &mut res,
            a.raw,
            indices.raw,
            axis,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_take_along_axis(
            &mut res,
            a.raw,
            indices.raw,
            axis,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_put_along_axis(
            &mut res,
            a.raw,
            indices.raw,
            values.raw,
            axis,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn topk_axis(a: &Array, k: i32, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_topk_axis(
            &mut res,
            a.raw,
            k,
            axis,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn argpartition_axis(a: &Array, kth: i32, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_argpartition_axis(
            &mut res,
            a.raw,
            kth,
            axis,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn argsort_axis(a: &Array, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_argsort_axis(
            &mut res,
            a.raw,
            axis,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn sort_axis(a: &Array, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_sort_axis(&mut res, a.raw, axis, stream()))?;
    }
    Ok(Array::from_raw(res))
}

pub fn cumsum(a: &Array, axis: i32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_cumsum(
            &mut res,
            a.raw,
            axis,
            false,
            true,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn mean_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_mean_axes(
            &mut res,
            a.raw,
            axes.as_ptr(),
            axes.len(),
            keepdims,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn sum_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_sum_axes(
            &mut res,
            a.raw,
            axes.as_ptr(),
            axes.len(),
            keepdims,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn max_axis(a: &Array, axis: i32, keepdims: bool) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_max_axis(
            &mut res,
            a.raw,
            axis,
            keepdims,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn broadcast_to(a: &Array, shape: &[i32]) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_broadcast_to(
            &mut res,
            a.raw,
            shape.as_ptr(),
            shape.len(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn where_cond(cond: &Array, a: &Array, b: &Array) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_where(
            &mut res,
            cond.raw,
            a.raw,
            b.raw,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_zeros(
            &mut res,
            shape.as_ptr(),
            shape.len(),
            dtype.to_raw(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn ones(shape: &[i32], dtype: Dtype) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_ones(
            &mut res,
            shape.as_ptr(),
            shape.len(),
            dtype.to_raw(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn full_f32(shape: &[i32], value: f32, dtype: Dtype) -> Result<Array> {
    let scalar = astype(&Array::scalar_f32(value), dtype)?;
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_full(
            &mut res,
            shape.as_ptr(),
            shape.len(),
            scalar.raw,
            dtype.to_raw(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn arange(start: f64, stop: f64, step: f64, dtype: Dtype) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_arange(
            &mut res,
            start,
            stop,
            step,
            dtype.to_raw(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

pub fn contiguous(a: &Array) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_contiguous(&mut res, a.raw, false, stream()))?;
    }
    Ok(Array::from_raw(res))
}

fn some_int(v: i32) -> crate::sys::mlx_optional_int {
    crate::sys::mlx_optional_int {
        value: v,
        has_value: true,
    }
}

fn none_dtype() -> crate::sys::mlx_optional_dtype {
    crate::sys::mlx_optional_dtype {
        value: 0,
        has_value: false,
    }
}

fn null_array() -> crate::sys::mlx_array {
    crate::sys::mlx_array {
        ctx: ptr::null_mut(),
    }
}

/// Quantization mode strings accepted by MLX (`affine`, `mxfp4`, `mxfp8`, `nvfp4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Affine,
    Mxfp4,
    Mxfp8,
    Nvfp4,
}

impl QuantMode {
    pub fn as_str(self) -> &'static str {
        match self {
            QuantMode::Affine => "affine",
            QuantMode::Mxfp4 => "mxfp4",
            QuantMode::Mxfp8 => "mxfp8",
            QuantMode::Nvfp4 => "nvfp4",
        }
    }

    pub fn parse(s: &str) -> crate::error::Result<Self> {
        match s {
            "affine" => Ok(QuantMode::Affine),
            "mxfp4" => Ok(QuantMode::Mxfp4),
            "mxfp8" => Ok(QuantMode::Mxfp8),
            "nvfp4" => Ok(QuantMode::Nvfp4),
            other => Err(crate::error::Error::Config(format!(
                "unsupported quantization mode '{other}'"
            ))),
        }
    }
}

/// Fused quantized matmul: `x @ dequant(w).T` when `transpose` is true.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
) -> Result<Array> {
    let mode_c = CString::new(mode.as_str()).unwrap();
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_quantized_matmul(
            &mut res,
            x.raw,
            w.raw,
            scales.raw,
            biases.map_or_else(null_array, |b| b.raw),
            transpose,
            some_int(group_size),
            some_int(bits),
            mode_c.as_ptr(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Gathered (dense) matmul for MoE expert dispatch: `x[i] @ w[rhs_indices[i]]`.
pub fn gather_mm(x: &Array, w: &Array, rhs_indices: &Array) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_gather_mm(
            &mut res,
            x.raw,
            w.raw,
            null_array(),
            rhs_indices.raw,
            false,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Gathered quantized matmul for MoE expert dispatch.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
    sorted_indices: bool,
) -> Result<Array> {
    let mode_c = CString::new(mode.as_str()).unwrap();
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_gather_qmm(
            &mut res,
            x.raw,
            w.raw,
            scales.raw,
            biases.map_or_else(null_array, |b| b.raw),
            lhs_indices.map_or_else(null_array, |i| i.raw),
            rhs_indices.map_or_else(null_array, |i| i.raw),
            transpose,
            some_int(group_size),
            some_int(bits),
            mode_c.as_ptr(),
            sorted_indices,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Dequantize packed weights back to floats (used for embeddings / tied heads).
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
) -> Result<Array> {
    let mode_c = CString::new(mode.as_str()).unwrap();
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_dequantize(
            &mut res,
            w.raw,
            scales.raw,
            biases.map_or_else(null_array, |b| b.raw),
            some_int(group_size),
            some_int(bits),
            mode_c.as_ptr(),
            null_array(),
            none_dtype(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Quantize float weights; returns `(w_packed, scales, biases?)`.
pub fn quantize(
    w: &Array,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
) -> Result<(Array, Array, Option<Array>)> {
    let mode_c = CString::new(mode.as_str()).unwrap();
    unsafe {
        let mut vec = crate::sys::mlx_vector_array_new();
        let status = crate::sys::mlx_quantize(
            &mut vec,
            w.raw,
            some_int(group_size),
            some_int(bits),
            mode_c.as_ptr(),
            null_array(),
            stream(),
        );
        if status != 0 {
            crate::sys::mlx_vector_array_free(vec);
            check(status)?;
        }
        let mut parts = collect_vector(vec)?;
        crate::sys::mlx_vector_array_free(vec);
        let biases = if parts.len() > 2 {
            Some(parts.remove(2))
        } else {
            None
        };
        let scales = parts.remove(1);
        let packed = parts.remove(0);
        Ok((packed, scales, biases))
    }
}

/// Fast fused RMSNorm.
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_fast_rms_norm(
            &mut res,
            x.raw,
            weight.map_or_else(null_array, |w| w.raw),
            eps,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Fast fused LayerNorm (mean/variance over the last axis, affine weight
/// and bias, both optional).
pub fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
) -> Result<Array> {
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_fast_layer_norm(
            &mut res,
            x.raw,
            weight.map_or_else(null_array, |w| w.raw),
            bias.map_or_else(null_array, |b| b.raw),
            eps,
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Fast fused RoPE.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&Array>,
) -> Result<Array> {
    let mut res = Array::new_handle();
    let base_opt = crate::sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    unsafe {
        check(crate::sys::mlx_fast_rope(
            &mut res,
            x.raw,
            dims,
            traditional,
            base_opt,
            scale,
            offset,
            freqs.map_or_else(null_array, |f| f.raw),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Mask mode for [`scaled_dot_product_attention`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    /// No mask (used for single-token decode).
    None,
    /// Causal mask computed inside the kernel.
    Causal,
}

/// Fast fused attention.
pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttentionMask,
) -> Result<Array> {
    let mask_mode = CString::new(match mask {
        AttentionMask::None => "",
        AttentionMask::Causal => "causal",
    })
    .unwrap();
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_fast_scaled_dot_product_attention(
            &mut res,
            queries.raw,
            keys.raw,
            values.raw,
            scale,
            mask_mode.as_ptr(),
            null_array(),
            null_array(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Fast fused attention with an explicit additive mask array.
pub fn scaled_dot_product_attention_masked(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: &Array,
) -> Result<Array> {
    let mask_mode = CString::new("").unwrap();
    let mut res = Array::new_handle();
    unsafe {
        check(crate::sys::mlx_fast_scaled_dot_product_attention(
            &mut res,
            queries.raw,
            keys.raw,
            values.raw,
            scale,
            mask_mode.as_ptr(),
            mask.raw,
            null_array(),
            stream(),
        ))?;
    }
    Ok(Array::from_raw(res))
}

/// Build an additive float mask of shape `[seq_len, kv_len]` for sliding
/// window attention: query position `offset + i` may attend to key position
/// `j` iff `0 <= (offset + i) - j < window`. Returns `0.0` where allowed and
/// a large negative value elsewhere, matching mlx-lm's windowed causal mask.
pub fn sliding_window_mask(
    seq_len: i32,
    kv_len: i32,
    offset: i32,
    window: i32,
    dtype: Dtype,
) -> Result<Array> {
    let rows = reshape(
        &arange(offset as f64, (offset + seq_len) as f64, 1.0, Dtype::Int32)?,
        &[seq_len, 1],
    )?;
    let cols = reshape(
        &arange(0.0, kv_len as f64, 1.0, Dtype::Int32)?,
        &[1, kv_len],
    )?;
    let diff = subtract(&rows, &cols)?;
    let not_future = greater_equal(&diff, &Array::scalar_i32(0))?;
    let within_window = less(&diff, &Array::scalar_i32(window))?;
    let allowed = logical_and(&not_future, &within_window)?;
    let neg_inf = full_f32(&[seq_len, kv_len], f32::NEG_INFINITY, dtype)?;
    let zeros_mask = zeros(&[seq_len, kv_len], dtype)?;
    where_cond(&allowed, &zeros_mask, &neg_inf)
}

/// SiLU activation: `x * sigmoid(x)`.
pub fn silu(x: &Array) -> Result<Array> {
    multiply(x, &sigmoid(x)?)
}

/// GELU (tanh approximation), matching `gelu_pytorch_tanh` used by Gemma.
/// Squared ReLU: `relu(x)^2` (NemotronH's `mlp_hidden_act`).
pub fn relu2(x: &Array) -> Result<Array> {
    let zero = Array::scalar_f32(0.0);
    square(&maximum(x, &zero)?)
}

pub fn gelu_tanh(x: &Array) -> Result<Array> {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = astype(&Array::scalar_f32(0.797_884_6), x.dtype())?;
    let coeff = astype(&Array::scalar_f32(0.044_715), x.dtype())?;
    let half = astype(&Array::scalar_f32(0.5), x.dtype())?;
    let one = astype(&Array::scalar_f32(1.0), x.dtype())?;
    let x3 = multiply(&multiply(x, x)?, x)?;
    let inner = multiply(&add(x, &multiply(&coeff, &x3)?)?, &c)?;
    let t = tanh(&inner)?;
    multiply(&multiply(&half, x)?, &add(&one, &t)?)
}

/// Exact GELU using erf, matching `nn.GELU` default.
pub fn gelu_erf(x: &Array) -> Result<Array> {
    let half = astype(&Array::scalar_f32(0.5), x.dtype())?;
    let one = astype(&Array::scalar_f32(1.0), x.dtype())?;
    let inv_sqrt2 = astype(
        &Array::scalar_f32(std::f32::consts::FRAC_1_SQRT_2),
        x.dtype(),
    )?;
    let e = erf(&multiply(x, &inv_sqrt2)?)?;
    multiply(&multiply(&half, x)?, &add(&one, &e)?)
}

/// Multiply by a scalar constant, preserving dtype.
pub fn scale_by(x: &Array, value: f32) -> Result<Array> {
    let s = astype(&Array::scalar_f32(value), x.dtype())?;
    multiply(x, &s)
}

/// Add a scalar constant, preserving dtype.
pub fn add_scalar(x: &Array, value: f32) -> Result<Array> {
    let s = astype(&Array::scalar_f32(value), x.dtype())?;
    add(x, &s)
}

/// Clamp every element of `x` to `[lo, hi]`, preserving dtype.
pub fn clip(x: &Array, lo: f32, hi: f32) -> Result<Array> {
    let lo = astype(&Array::scalar_f32(lo), x.dtype())?;
    let hi = astype(&Array::scalar_f32(hi), x.dtype())?;
    let clamped = maximum(x, &lo)?;
    minimum(&clamped, &hi)
}
