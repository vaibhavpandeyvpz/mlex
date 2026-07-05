//! Video preprocessing: uniform frame sampling via `ffmpeg`, with each
//! extracted frame reusing the image path (`crate::media::image`).
//!
//! Frames are extracted with a file-path input and file-based PNG output
//! (`ffmpeg -i <path> -vf fps=N <dir>/frame_%04d.png`) - deliberately no
//! stdin/stdout piping, which is fragile for long clips (SIGPIPE-prone
//! for pipe-based video feeders on clips longer than a couple of seconds
//! on this machine).
//!
//! Sampling policy: ~1 frame per second, capped at [`MAX_FRAMES`] frames
//! (uniformly spread over the clip when the cap kicks in). Gemma4 has no
//! published video-specific processor config; a higher uncapped frame
//! rate would produce far too many soft tokens (256 per frame) for a small
//! on-device model's context, so we follow the widely-used 1 fps / fixed
//! cap convention instead.

use std::path::Path;
use std::process::Command;

use crate::error::{Error, Result};

/// Maximum number of frames sampled from one clip.
pub const MAX_FRAMES: usize = 8;

/// Extract up to [`MAX_FRAMES`] uniformly-sampled frames from an encoded
/// video (mp4/webm/...), returning each frame's encoded PNG bytes in
/// temporal order. The caller feeds these through
/// `crate::media::image::preprocess_image_bytes` like any other image.
pub fn extract_video_frames(data: &[u8]) -> Result<Vec<Vec<u8>>> {
    let dir = std::env::temp_dir().join(format!(
        "mlex-video-{}-{:x}",
        std::process::id(),
        data.as_ptr() as usize ^ data.len()
    ));
    std::fs::create_dir_all(&dir)?;
    let result = extract_in_dir(data, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn extract_in_dir(data: &[u8], dir: &Path) -> Result<Vec<Vec<u8>>> {
    let input = dir.join("input.bin");
    std::fs::write(&input, data)?;

    let duration = probe_duration(&input)?;
    // ~1 fps, reduced when the clip is long enough that 1 fps would
    // exceed the frame cap.
    let fps = if duration > MAX_FRAMES as f64 {
        MAX_FRAMES as f64 / duration
    } else {
        1.0
    };

    let pattern = dir.join("frame_%04d.png");
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(&input)
        .args(["-vf", &format!("fps={fps:.6}")])
        .args(["-frames:v", &MAX_FRAMES.to_string()])
        .arg(&pattern)
        .output()
        .map_err(|e| {
            Error::Model(format!(
                "failed to run ffmpeg for video decoding (is ffmpeg on PATH?): {e}"
            ))
        })?;
    if !output.status.success() {
        return Err(Error::Model(format!(
            "ffmpeg failed to extract video frames: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let mut frames = Vec::new();
    for i in 1..=MAX_FRAMES {
        let path = dir.join(format!("frame_{i:04}.png"));
        if !path.exists() {
            break;
        }
        frames.push(std::fs::read(&path)?);
    }
    if frames.is_empty() {
        return Err(Error::Model(
            "video produced no frames (empty or undecodable clip)".into(),
        ));
    }
    Ok(frames)
}

/// Container duration in seconds via `ffprobe` (0.0 when unavailable,
/// which falls back to 1 fps sampling capped by `-frames:v`).
fn probe_duration(input: &Path) -> Result<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
        ])
        .arg(input)
        .output()
        .map_err(|e| {
            Error::Model(format!(
                "failed to run ffprobe for video probing (is ffprobe on PATH?): {e}"
            ))
        })?;
    if !output.status.success() {
        return Err(Error::Model(format!(
            "ffprobe failed on video input: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0))
}
