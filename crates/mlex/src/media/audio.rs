//! Audio preprocessing matching Gemma4's audio feature extractor:
//! 16 kHz mono PCM in, `[T, 128]` log-mel spectrogram frames out.
//!
//! Parameters (verified against this model family's published
//! configuration): `sample_rate=16000`, `n_fft=512`,
//! `window_len=320` (20 ms periodic Hann, zero-padded to the FFT size),
//! `hop=160` (10 ms), `n_mels=128` (HTK scale, `fmin=0`, `fmax=8000`, no
//! Slaney area normalization), magnitude (not power) spectrum, natural log
//! with a `1e-3` floor, semicausal left padding of `window_len / 2`, split
//! into 30-second chunks (each chunk is run through the audio tower
//! separately, matching the model's attention-context training length).
//!
//! Decoding: WAV (PCM16 / float32) is parsed natively; every other
//! container/codec (mp3, mp4, flac, ...) is decoded by shelling out to
//! `ffmpeg` with a temp-file input (never stdin piping) and raw
//! `f32le/16k/mono` output.

use std::io::Write;
use std::process::Command;

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use crate::array::Array;
use crate::error::{Error, Result};

/// Sample rate the Gemma4 audio front-end expects (16 kHz mono).
pub const AUDIO_SAMPLE_RATE: u32 = 16_000;
/// FFT size.
const N_FFT: usize = 512;
/// Hann window length (20 ms @ 16 kHz), zero-padded to [`N_FFT`].
const WINDOW_LEN: usize = 320;
/// Hop length (10 ms @ 16 kHz).
const HOP: usize = 160;
/// Mel filterbank size.
const N_MELS: usize = 128;
/// Log-mel floor.
const MEL_FLOOR: f64 = 1e-3;
/// Chunk length in samples (30 s, the model's per-pass context limit).
const CHUNK_SAMPLES: usize = 30 * AUDIO_SAMPLE_RATE as usize;

/// A preprocessed audio clip ready for a Gemma4 audio encoder. Two shapes,
/// selected by which encoder the checkpoint loaded (see
/// `crate::models::gemma4::AudioEncoder`):
/// - mel-spectrogram Conformer tower: `chunks` holds one `[1, T_i, 128]`
///   log-mel tensor per 30-second chunk, and each chunk's soft-token count
///   is subsampled by the tower's two stride-2 convolutions.
/// - encoder-free "unified" path: `chunks` holds a single `[n, S]` raw PCM
///   window tensor (`S` = samples per audio token), one soft token per row.
#[derive(Debug, Clone)]
pub struct ProcessedAudio {
    pub chunks: Vec<Array>,
    /// Per-chunk frame count (`chunks[i].dim(1)` for mel, `chunks[i].dim(0)`
    /// for raw windows).
    pub frames_per_chunk: Vec<i32>,
    /// Whether `chunks`/`frames_per_chunk` hold raw PCM windows (unified
    /// path, one soft token per row, no subsampling) rather than
    /// mel-spectrogram frames (Conformer tower path).
    pub raw: bool,
}

impl ProcessedAudio {
    /// Total soft tokens this clip expands to: for the mel-spectrogram
    /// path, after the audio tower's two stride-2 subsampling convolutions
    /// (`O = (I - 1) / 2 + 1`, twice); for the raw-window "unified" path,
    /// one soft token per window (no subsampling), summed over chunks.
    pub fn num_soft_tokens(&self) -> i32 {
        if self.raw {
            self.frames_per_chunk.iter().sum()
        } else {
            self.frames_per_chunk
                .iter()
                .map(|&t| subsampled_len(t))
                .sum()
        }
    }
}

/// Output length of the audio tower's two stride-2 (kernel 3, pad 1)
/// subsampling convolutions for `t` input mel frames.
pub fn subsampled_len(t: i32) -> i32 {
    let mut n = t;
    for _ in 0..2 {
        n = (n - 1) / 2 + 1;
    }
    n
}

/// Decode `data` (WAV/MP3/...) to 16 kHz mono f32 PCM, then compute
/// per-chunk log-mel spectrograms.
pub fn preprocess_audio_bytes(data: &[u8]) -> Result<ProcessedAudio> {
    let pcm = decode_audio_bytes(data)?;
    if pcm.is_empty() {
        return Err(Error::Model("audio clip decoded to zero samples".into()));
    }

    let mut chunks = Vec::new();
    let mut frames_per_chunk = Vec::new();
    for chunk in pcm.chunks(CHUNK_SAMPLES) {
        let mel = log_mel_spectrogram(chunk);
        let t = (mel.len() / N_MELS) as i32;
        if t == 0 {
            continue;
        }
        chunks.push(Array::from_slice(&mel, &[1, t, N_MELS as i32]));
        frames_per_chunk.push(t);
    }
    if chunks.is_empty() {
        return Err(Error::Model(
            "audio clip too short to produce any mel frames".into(),
        ));
    }
    Ok(ProcessedAudio {
        chunks,
        frames_per_chunk,
        raw: false,
    })
}

/// Decode `data` (WAV/MP3/...) to 16 kHz mono f32 PCM, then build the
/// encoder-free "unified" path's raw-window frame tensor: zero-pad right
/// to a multiple of `samples_per_token`, reshape to `[n_frames,
/// samples_per_token]`, no scaling/normalization (samples pass through
/// untouched). One frame = one audio soft token; there is no tower and no
/// subsampling, unlike the mel-spectrogram Conformer path.
pub fn preprocess_audio_bytes_raw(data: &[u8], samples_per_token: i32) -> Result<ProcessedAudio> {
    let pcm = decode_audio_bytes(data)?;
    if pcm.is_empty() {
        return Err(Error::Model("audio clip decoded to zero samples".into()));
    }
    let spt = samples_per_token.max(1) as usize;

    let n = pcm.len();
    let pad = (spt - (n % spt)) % spt;
    let n_frames = (n + pad) / spt;

    let mut frames = pcm;
    frames.resize(n + pad, 0.0);

    let tensor = Array::from_slice(&frames, &[n_frames as i32, spt as i32]);
    Ok(ProcessedAudio {
        chunks: vec![tensor],
        frames_per_chunk: vec![n_frames as i32],
        raw: true,
    })
}

/// Decode encoded audio bytes to 16 kHz mono f32 PCM: native RIFF/WAVE
/// parsing when possible, `ffmpeg` (temp-file input, raw f32le output)
/// for everything else.
pub fn decode_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        if let Some(pcm) = decode_wav(data)? {
            return Ok(pcm);
        }
        // Fall through to ffmpeg for WAVs we can't handle natively
        // (wrong sample rate, exotic codec, ...).
    }
    decode_with_ffmpeg(data)
}

/// Parse a RIFF/WAVE byte stream. Returns `Ok(None)` when the file is a
/// valid WAV that needs resampling/transcoding (handled by ffmpeg instead).
fn decode_wav(bytes: &[u8]) -> Result<Option<Vec<f32>>> {
    let mut audio_format = 0u16;
    let mut num_channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data: Option<&[u8]> = None;

    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start
            .checked_add(chunk_size)
            .ok_or_else(|| Error::Model("corrupt WAV: chunk size overflow".into()))?;
        if body_end > bytes.len() {
            return Err(Error::Model("corrupt WAV: truncated chunk".into()));
        }
        let body = &bytes[body_start..body_end];
        if chunk_id == b"fmt " {
            if body.len() < 16 {
                return Err(Error::Model("corrupt WAV: fmt chunk too short".into()));
            }
            audio_format = u16::from_le_bytes([body[0], body[1]]);
            num_channels = u16::from_le_bytes([body[2], body[3]]);
            sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
        } else if chunk_id == b"data" {
            data = Some(body);
        }
        pos = body_end + (chunk_size & 1);
    }

    let Some(data) = data else {
        return Err(Error::Model("corrupt WAV: missing data chunk".into()));
    };
    if num_channels == 0 || sample_rate != AUDIO_SAMPLE_RATE {
        return Ok(None); // let ffmpeg resample/downmix
    }

    let interleaved: Vec<f32> = match (audio_format, bits_per_sample) {
        (1, 16) => data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        _ => return Ok(None),
    };

    let channels = num_channels as usize;
    if channels == 1 {
        return Ok(Some(interleaved));
    }
    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for frame in 0..frames {
        let base = frame * channels;
        let sum: f32 = interleaved[base..base + channels].iter().sum();
        mono.push(sum / channels as f32);
    }
    Ok(Some(mono))
}

/// Decode arbitrary encoded audio via `ffmpeg`, writing the input to a
/// temp file first (file-path input, never stdin piping) and reading raw
/// `f32le` 16 kHz mono PCM from stdout.
fn decode_with_ffmpeg(data: &[u8]) -> Result<Vec<f32>> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "mlex-audio-{}-{:x}.bin",
        std::process::id(),
        data.len()
    ));
    {
        let mut f = std::fs::File::create(&path)?;
        f.write_all(data)?;
    }
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"])
        .output();
    let _ = std::fs::remove_file(&path);
    let output = output.map_err(|e| {
        Error::Model(format!(
            "failed to run ffmpeg for audio decoding (is ffmpeg on PATH?): {e}"
        ))
    })?;
    if !output.status.success() {
        return Err(Error::Model(format!(
            "ffmpeg failed to decode audio: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Gemma4 log-mel spectrogram for one <=30s chunk of 16 kHz mono PCM.
/// Returns `t * N_MELS` values, frame-major (`out[t * N_MELS + m]`).
fn log_mel_spectrogram(chunk: &[f32]) -> Vec<f32> {
    // Semicausal left padding + right padding to match the expected frame
    // count: unfold(size=window_len + 1, step=hop) over the left-padded
    // waveform.
    let pad_left = WINDOW_LEN / 2;
    let n_with_left = chunk.len() + pad_left;
    if n_with_left < WINDOW_LEN + 1 {
        return Vec::new();
    }
    let pt_frames = (n_with_left - (WINDOW_LEN + 1)) / HOP + 1;
    let n_padded_needed = (pt_frames - 1) * HOP + N_FFT;
    let total_pad = n_padded_needed.saturating_sub(chunk.len()).max(pad_left);
    let mut padded = vec![0f32; total_pad + chunk.len()];
    padded[pad_left..pad_left + chunk.len()].copy_from_slice(chunk);

    // Standard periodic Hann window of WINDOW_LEN, zero-padded to N_FFT.
    let mut hann = vec![0f32; N_FFT];
    for (i, w) in hann.iter_mut().enumerate().take(WINDOW_LEN) {
        *w = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / WINDOW_LEN as f32).cos();
    }

    let filters = mel_filterbank();
    let n_bins = N_FFT / 2 + 1;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let n_frames = ((padded.len() - N_FFT) / HOP + 1).min(pt_frames);
    let mut out = vec![0f32; n_frames * N_MELS];
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];
    let mut magnitude = vec![0f32; n_bins];
    for t in 0..n_frames {
        let offset = t * HOP;
        for j in 0..N_FFT {
            buf[j] = Complex32::new(hann[j] * padded[offset + j], 0.0);
        }
        fft.process(&mut buf);
        for (j, m) in magnitude.iter_mut().enumerate() {
            *m = buf[j].norm();
        }
        for m in 0..N_MELS {
            let mut sum = 0f64;
            for (j, &mag) in magnitude.iter().enumerate() {
                sum += mag as f64 * filters[m * n_bins + j] as f64;
            }
            out[t * N_MELS + m] = sum.max(MEL_FLOOR).ln() as f32;
        }
    }
    out
}

/// Triangular mel filterbank: HTK mel scale, `fmin=0`, `fmax=sr/2`, no
/// Slaney area normalization. `N_MELS x (N_FFT/2 + 1)`, filter-major.
fn mel_filterbank() -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1;
    let fmax = AUDIO_SAMPLE_RATE as f64 / 2.0;
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);

    let m_lo = hz_to_mel(0.0);
    let m_hi = hz_to_mel(fmax);
    let hz_pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();

    let bin_hz_step = AUDIO_SAMPLE_RATE as f64 / N_FFT as f64;
    let mut out = vec![0f32; N_MELS * n_bins];
    for m in 0..N_MELS {
        let (f_left, f_center, f_right) = (hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]);
        let denom_l = (f_center - f_left).max(1e-30);
        let denom_r = (f_right - f_center).max(1e-30);
        for (k, o) in out[m * n_bins..(m + 1) * n_bins].iter_mut().enumerate() {
            let f = k as f64 * bin_hz_step;
            let w = if f >= f_left && f <= f_center {
                (f - f_left) / denom_l
            } else if f > f_center && f <= f_right {
                (f_right - f) / denom_r
            } else {
                0.0
            };
            *o = w as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampled_len_matches_two_stride2_convs() {
        // O = (I - 1)/2 + 1, applied twice.
        assert_eq!(subsampled_len(1), 1);
        assert_eq!(subsampled_len(4), 1);
        assert_eq!(subsampled_len(100), 25);
        assert_eq!(subsampled_len(3000), 750);
    }

    #[test]
    fn mel_filterbank_rows_are_valid_triangles() {
        let filters = mel_filterbank();
        let n_bins = N_FFT / 2 + 1;
        let mut nonempty = 0;
        for m in 0..N_MELS {
            let row = &filters[m * n_bins..(m + 1) * n_bins];
            if row.iter().sum::<f32>() > 0.0 {
                nonempty += 1;
            }
            assert!(row.iter().all(|&w| (0.0..=1.0).contains(&w)));
        }
        // The lowest few filters legitimately span less than one FFT bin
        // (triangle narrower than 31.25 Hz) and come out all-zero -
        // identical to the reference filterbank. Everything else must be
        // a real triangle.
        assert!(
            nonempty >= N_MELS - 8,
            "only {nonempty}/{N_MELS} mel filters are nonzero"
        );
    }

    #[test]
    fn spectrogram_frame_count_matches_pytorch_unfold() {
        // 1 second of a 440 Hz tone: pt_frames = (16000 + 160 - 321)/160 + 1 = 99.
        let pcm: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let mel = log_mel_spectrogram(&pcm);
        assert_eq!(mel.len() / N_MELS, 99);
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn silence_hits_the_mel_floor() {
        let pcm = vec![0f32; 16000];
        let mel = log_mel_spectrogram(&pcm);
        let floor = (MEL_FLOOR as f32).ln();
        assert!(mel.iter().all(|&v| (v - floor).abs() < 1e-4));
    }

    #[test]
    fn decode_wav_pcm16_mono_roundtrip() {
        let samples: [i16; 4] = [0, 16384, -32768, 32767];
        let mut data = Vec::new();
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);

        let pcm = decode_audio_bytes(&wav).unwrap();
        assert_eq!(pcm.len(), 4);
        assert_eq!(pcm[0], 0.0);
        assert_eq!(pcm[1], 0.5);
        assert_eq!(pcm[2], -1.0);
    }
}
