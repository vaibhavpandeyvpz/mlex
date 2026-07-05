//! Image preprocessing matching Gemma4's vision processor (HF
//! `preprocessor_config.json`: `do_rescale=true` (÷255), `do_normalize=false`
//! (mean=0, std=1) - i.e. resize + rescale to `[0, 1]` only; the model's own
//! patch embedder applies the `2*(x-0.5)` normalization).
//!
//! Resize policy: a "smart_resize" style used by several vision processors:
//! round each side to the nearest multiple of `patch_size *
//! pooling_kernel_size`, then only rescale (preserving aspect ratio) if the
//! rounded pixel count falls outside `[min_tokens, max_tokens] *
//! (patch_size * pooling_kernel_size)^2` - i.e. most naturally-sized photos
//! pass through close to their native resolution instead of always being
//! stretched to fill the token budget (an "always fill the budget" policy
//! instead causes small photos to be upscaled ~1.5x, blurring detail and
//! degrading vision quality).

use image::imageops::FilterType;
use image::{GenericImageView, RgbImage};

use crate::array::Array;
use crate::error::{Error, Result};

/// A resized, patch-grid-aligned image ready for a vision tower.
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// `[1, 3, H, W]` float32, values in `[0, 1]` (channel-first, no
    /// mean/std normalization - matches Gemma4's processor config).
    pub pixel_values: Array,
    /// Patch grid height (`H / patch_size`).
    pub patch_h: i32,
    /// Patch grid width (`W / patch_size`).
    pub patch_w: i32,
    /// Soft tokens this image expands to after pooling:
    /// `patch_h * patch_w / pooling_kernel_size^2`.
    pub num_soft_tokens: i32,
}

/// Gemma4 vision's hardcoded soft-token budget - not read from
/// `config.json`, so we hardcode it here too.
pub const MIN_SOFT_TOKENS: i32 = 40;

/// Decode `data` (JPEG/PNG/...) and resize it per Gemma4's smart-resize
/// policy: [`MIN_SOFT_TOKENS`]..=`max_soft_tokens` worth of `patch_size`x
/// `patch_size` patches (grouped into `pooling_kernel_size`x
/// `pooling_kernel_size` pooling blocks).
pub fn preprocess_image_bytes(
    data: &[u8],
    patch_size: i32,
    max_soft_tokens: i32,
    pooling_kernel_size: i32,
) -> Result<ProcessedImage> {
    let img = image::load_from_memory(data)
        .map_err(|e| Error::Model(format!("failed to decode image: {e}")))?;
    let (orig_w, orig_h) = img.dimensions();
    let (target_w, target_h) = compute_target_size(
        orig_h,
        orig_w,
        patch_size,
        MIN_SOFT_TOKENS,
        max_soft_tokens,
        pooling_kernel_size,
    );

    let resized = if target_h == orig_h && target_w == orig_w {
        img.to_rgb8()
    } else {
        // Bilinear resize (not a higher-order filter), matching Gemma4's
        // expected image resize algorithm.
        img.resize_exact(target_w, target_h, FilterType::Triangle)
            .to_rgb8()
    };

    let pixel_values = rgb_to_chw_array(&resized);
    let patch_h = target_h as i32 / patch_size;
    let patch_w = target_w as i32 / patch_size;
    let num_soft_tokens = (patch_h * patch_w) / (pooling_kernel_size * pooling_kernel_size);

    Ok(ProcessedImage {
        pixel_values,
        patch_h,
        patch_w,
        num_soft_tokens,
    })
}

/// "smart_resize" style size calculation: round both
/// sides up to the nearest multiple of `align_size` (`patch_size *
/// pooling_kernel_size`) first: if that lands within `[min_pixels,
/// max_pixels]`, use it as-is (near-native resolution, no big up/downscale);
/// otherwise scale by `sqrt(area_ratio)` and re-align (floor when shrinking
/// to fit under `max_pixels`, ceil when growing to clear `min_pixels`).
fn compute_target_size(
    height: u32,
    width: u32,
    patch_size: i32,
    min_soft_tokens: i32,
    max_soft_tokens: i32,
    pooling_kernel_size: i32,
) -> (u32, u32) {
    let align = (patch_size * pooling_kernel_size) as f64;
    let patch_area = align * align;
    let min_pixels = min_soft_tokens as f64 * patch_area;
    let max_pixels = max_soft_tokens as f64 * patch_area;

    let (h, w) = (height as f64, width as f64);
    let round_by = |x: f64| -> f64 { (x / align).round() * align };
    let floor_by = |x: f64| -> f64 { (x / align).floor() * align };
    let ceil_by = |x: f64| -> f64 { (x / align).ceil() * align };

    let mut h_bar = round_by(h).max(align);
    let mut w_bar = round_by(w).max(align);

    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels).sqrt();
        h_bar = floor_by(h / beta).max(align);
        w_bar = floor_by(w / beta).max(align);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels / (h * w)).sqrt();
        h_bar = ceil_by(h * beta).max(align);
        w_bar = ceil_by(w * beta).max(align);
    }

    (w_bar as u32, h_bar as u32)
}

/// Rescale an RGB image to `[0, 1]` and lay it out channel-first as
/// `[1, 3, H, W]` float32 (no mean/std normalization).
fn rgb_to_chw_array(rgb: &RgbImage) -> Array {
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let mut chw = vec![0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3usize {
                chw[c * height * width + y * width + x] = pixel[c] as f32 / 255.0;
            }
        }
    }
    Array::from_slice(&chw, &[1, 3, height as i32, width as i32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_target_size_is_a_multiple_of_side_mult() {
        // patch_size=16, pooling=3 -> align=48.
        let (w, h) = compute_target_size(1080, 1920, 16, 40, 280, 3);
        assert_eq!(w % 48, 0);
        assert_eq!(h % 48, 0);
        assert!(w > 0 && h > 0);
    }

    #[test]
    fn num_soft_tokens_never_exceeds_budget() {
        let (w, h) = compute_target_size(4000, 3000, 16, 40, 280, 3);
        let patch_h = h as i32 / 16;
        let patch_w = w as i32 / 16;
        let soft = (patch_h * patch_w) / 9;
        assert!(soft <= 280, "soft={soft}");
    }

    #[test]
    fn a_naturally_sized_photo_stays_near_native_resolution() {
        // 640x426 (samples/image1.jpg) rounds to within the [40, 280]
        // soft-token budget already, so it should NOT be upscaled to fill
        // the budget (a prior, incorrect port of a different reference's
        // "always fill the budget" policy stretched this to 960x624).
        let (w, h) = compute_target_size(426, 640, 16, 40, 280, 3);
        assert_eq!((w, h), (624, 432));
    }
}
