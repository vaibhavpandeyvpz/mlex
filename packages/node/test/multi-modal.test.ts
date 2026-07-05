import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { MlexModel } from "../index.js";
import { hasCapability, registry } from "./models.js";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..", "..");
const sample = (name: string) => join(REPO_ROOT, "samples", name);

// Multimodal generation through a vision + audio capable checkpoint's
// towers: raw media bytes ride along on the message as
// `images`/`audios`/`videos` Buffers, get preprocessed + encoded in Rust,
// and their soft tokens are spliced into the prompt at the placeholder
// positions. Mirrors `crates/mlex/tests/multi_modal.rs`.
//
// Requiring both `vision_config` and `audio_config` naturally selects a
// Gemma4 checkpoint today (the only architecture with an audio tower);
// other vision-only families (e.g. Qwen3.5-VL) are exercised by
// `multi-modal-multi-turn.test.ts` and the image-only case here isn't
// gated on audio.
const mmModel = registry().find(
  (m) =>
    hasCapability(m.dir, "vision_config") &&
    hasCapability(m.dir, "audio_config"),
);

describe.skipIf(!mmModel)("multi-modal generation (vision/audio/video)", () => {
  let model: MlexModel;

  beforeAll(async () => {
    model = await MlexModel.load(mmModel!.dir);
  }, 120_000);

  it("reports image and audio support for the multimodal checkpoint", () => {
    expect(model.supportsImages()).toBe(true);
    expect(model.supportsAudio()).toBe(true);
  });

  it("describes samples/image1.jpg", async () => {
    const { text: reply } = await model.generate(
      [
        {
          role: "user",
          content: "Describe this image in one short sentence.",
          images: [readFileSync(sample("image1.jpg"))],
        },
      ],
      { maxTokens: 64, temperature: 0 },
    );
    console.log(`[multi-modal] image description: ${reply}`);
    expect(reply.trim().length).toBeGreaterThan(10);
    // Coherent-English sanity check rather than exact-content assertion
    // (small quantized VLMs are imprecise at identification).
    expect(reply.toLowerCase()).toMatch(/image|photo|picture|depicts|shows/);
  }, 180_000);

  it("transcribes samples/audio.mp3 recognizably", async () => {
    const { text: reply } = await model.generate(
      [
        {
          role: "user",
          content: "Transcribe this audio.",
          audios: [readFileSync(sample("audio.mp3"))],
        },
      ],
      { maxTokens: 96, temperature: 0 },
    );
    console.log(`[multi-modal] audio transcription: ${reply}`);
    // The clip is the deterministic samplelib.com speech recording, so
    // the transcription is meaningfully checkable (same bar as the Rust
    // test).
    const lower = reply.toLowerCase();
    const hits = ["sample", "download", "digital", "format", "file"].filter(
      (w) => lower.includes(w),
    );
    expect(hits.length).toBeGreaterThanOrEqual(2);
  }, 300_000);

  it("describes samples/video1.mp4 from uniformly sampled frames", async () => {
    const { text: reply } = await model.generate(
      [
        {
          role: "user",
          content: "Describe what happens in this video in one sentence.",
          videos: [readFileSync(sample("video1.mp4"))],
        },
      ],
      { maxTokens: 64, temperature: 0 },
    );
    console.log(`[multi-modal] video description: ${reply}`);
    expect(reply.trim().length).toBeGreaterThan(10);
    expect(reply.toLowerCase()).toMatch(
      /video|shows|frames?|scene|person|animal|horse/,
    );
  }, 300_000);
});

describe.skipIf(mmModel)(
  "multi-modal generation (no capable model available)",
  () => {
    it("skips: no CI-safe vision+audio checkpoint under MLEX_MODELS_DIR", () => {
      expect(mmModel).toBeUndefined();
    });
  },
);
