use std::ffi::c_void;
use std::fmt;

use crate::error::{check, install_error_handler, Result};
use crate::stream::stream;

/// Element dtype of an [`Array`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Complex64,
}

impl Dtype {
    pub(crate) fn to_raw(self) -> crate::sys::mlx_dtype {
        use crate::sys::*;
        match self {
            Dtype::Bool => mlx_dtype__MLX_BOOL,
            Dtype::UInt8 => mlx_dtype__MLX_UINT8,
            Dtype::UInt16 => mlx_dtype__MLX_UINT16,
            Dtype::UInt32 => mlx_dtype__MLX_UINT32,
            Dtype::UInt64 => mlx_dtype__MLX_UINT64,
            Dtype::Int8 => mlx_dtype__MLX_INT8,
            Dtype::Int16 => mlx_dtype__MLX_INT16,
            Dtype::Int32 => mlx_dtype__MLX_INT32,
            Dtype::Int64 => mlx_dtype__MLX_INT64,
            Dtype::Float16 => mlx_dtype__MLX_FLOAT16,
            Dtype::Float32 => mlx_dtype__MLX_FLOAT32,
            Dtype::Float64 => mlx_dtype__MLX_FLOAT64,
            Dtype::BFloat16 => mlx_dtype__MLX_BFLOAT16,
            Dtype::Complex64 => mlx_dtype__MLX_COMPLEX64,
        }
    }

    pub(crate) fn from_raw(raw: crate::sys::mlx_dtype) -> Self {
        match raw {
            crate::sys::mlx_dtype__MLX_BOOL => Dtype::Bool,
            crate::sys::mlx_dtype__MLX_UINT8 => Dtype::UInt8,
            crate::sys::mlx_dtype__MLX_UINT16 => Dtype::UInt16,
            crate::sys::mlx_dtype__MLX_UINT32 => Dtype::UInt32,
            crate::sys::mlx_dtype__MLX_UINT64 => Dtype::UInt64,
            crate::sys::mlx_dtype__MLX_INT8 => Dtype::Int8,
            crate::sys::mlx_dtype__MLX_INT16 => Dtype::Int16,
            crate::sys::mlx_dtype__MLX_INT32 => Dtype::Int32,
            crate::sys::mlx_dtype__MLX_INT64 => Dtype::Int64,
            crate::sys::mlx_dtype__MLX_FLOAT16 => Dtype::Float16,
            crate::sys::mlx_dtype__MLX_FLOAT32 => Dtype::Float32,
            crate::sys::mlx_dtype__MLX_FLOAT64 => Dtype::Float64,
            crate::sys::mlx_dtype__MLX_BFLOAT16 => Dtype::BFloat16,
            crate::sys::mlx_dtype__MLX_COMPLEX64 => Dtype::Complex64,
            other => panic!("unknown mlx dtype {other}"),
        }
    }
}

/// An owned, lazily-evaluated MLX array.
///
/// Cheap to clone metadata-wise is not offered on purpose: `Array` owns the
/// underlying `mlx_array` handle and frees it on drop. Use [`Array::clone`]
/// (deep handle copy, shares the same lazy buffer internally) when needed.
pub struct Array {
    pub(crate) raw: crate::sys::mlx_array,
}

// mlx_array handles are internally reference counted std::shared_ptr-backed
// objects; MLX itself synchronizes evaluation through its scheduler.
unsafe impl Send for Array {}
unsafe impl Sync for Array {}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe {
            crate::sys::mlx_array_free(self.raw);
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        install_error_handler();
        unsafe {
            let mut out = crate::sys::mlx_array_new();
            crate::sys::mlx_array_set(&mut out, self.raw);
            Array { raw: out }
        }
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}

impl Array {
    pub(crate) fn from_raw(raw: crate::sys::mlx_array) -> Self {
        Array { raw }
    }

    /// Allocate the output handle used by mlx-c out-parameters.
    pub(crate) fn new_handle() -> crate::sys::mlx_array {
        install_error_handler();
        unsafe { crate::sys::mlx_array_new() }
    }

    /// Create an array by copying `data` with the given shape.
    pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Self {
        install_error_handler();
        let expected: i64 = shape.iter().map(|&d| d as i64).product();
        assert_eq!(
            data.len() as i64,
            expected,
            "data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        unsafe {
            let raw = crate::sys::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len() as i32,
                T::DTYPE.to_raw(),
            );
            Array { raw }
        }
    }

    /// Create a rank-0 (scalar) float32 array.
    pub fn scalar_f32(value: f32) -> Self {
        Self::from_slice(&[value], &[])
    }

    /// Create a rank-0 (scalar) int32 array.
    pub fn scalar_i32(value: i32) -> Self {
        Self::from_slice(&[value], &[])
    }

    pub fn ndim(&self) -> usize {
        unsafe { crate::sys::mlx_array_ndim(self.raw) }
    }

    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let ndim = crate::sys::mlx_array_ndim(self.raw);
            let ptr = crate::sys::mlx_array_shape(self.raw);
            std::slice::from_raw_parts(ptr, ndim).to_vec()
        }
    }

    pub fn dim(&self, axis: i32) -> i32 {
        let shape = self.shape();
        let ndim = shape.len() as i32;
        let axis = if axis < 0 { axis + ndim } else { axis };
        shape[axis as usize]
    }

    pub fn size(&self) -> usize {
        self.shape().iter().map(|&d| d as usize).product()
    }

    pub fn dtype(&self) -> Dtype {
        unsafe { Dtype::from_raw(crate::sys::mlx_array_dtype(self.raw)) }
    }

    /// Force evaluation of this array (MLX is lazy).
    pub fn eval(&self) -> Result<()> {
        unsafe { check(crate::sys::mlx_array_eval(self.raw)) }
    }

    /// Extract a scalar float (evaluates if needed). Works on any float dtype.
    pub fn item_f32(&self) -> Result<f32> {
        let mut out: f32 = 0.0;
        unsafe {
            let arr = crate::ops::astype(self, Dtype::Float32)?;
            check(crate::sys::mlx_array_item_float32(&mut out, arr.raw))?;
        }
        Ok(out)
    }

    /// Extract a scalar u32 (evaluates if needed).
    pub fn item_u32(&self) -> Result<u32> {
        let mut out: u32 = 0;
        unsafe {
            let arr = crate::ops::astype(self, Dtype::UInt32)?;
            check(crate::sys::mlx_array_item_uint32(&mut out, arr.raw))?;
        }
        Ok(out)
    }

    /// Copy the contents out as `f32` values (converting dtype if needed).
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let as_f32 = crate::ops::astype(self, Dtype::Float32)?;
        as_f32.eval()?;
        unsafe {
            let ptr = crate::sys::mlx_array_data_float32(as_f32.raw);
            if ptr.is_null() {
                return Err(crate::error::Error::Mlx(
                    "null data pointer reading array".into(),
                ));
            }
            Ok(std::slice::from_raw_parts(ptr, as_f32.size()).to_vec())
        }
    }

    /// Copy the contents out as `u32` values (must already be uint32).
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        let arr = crate::ops::astype(self, Dtype::UInt32)?;
        arr.eval()?;
        unsafe {
            let ptr = crate::sys::mlx_array_data_uint32(arr.raw);
            if ptr.is_null() {
                return Err(crate::error::Error::Mlx(
                    "null data pointer reading array".into(),
                ));
            }
            Ok(std::slice::from_raw_parts(ptr, arr.size()).to_vec())
        }
    }
}

/// Rust element types that map onto MLX dtypes for [`Array::from_slice`].
pub trait ArrayElement {
    const DTYPE: Dtype;
}

impl ArrayElement for f32 {
    const DTYPE: Dtype = Dtype::Float32;
}
impl ArrayElement for u32 {
    const DTYPE: Dtype = Dtype::UInt32;
}
impl ArrayElement for i32 {
    const DTYPE: Dtype = Dtype::Int32;
}
impl ArrayElement for u8 {
    const DTYPE: Dtype = Dtype::UInt8;
}
impl ArrayElement for bool {
    const DTYPE: Dtype = Dtype::Bool;
}

/// Synchronize the default stream, blocking until queued work completes.
pub fn synchronize() -> Result<()> {
    unsafe { check(crate::sys::mlx_synchronize(stream())) }
}
