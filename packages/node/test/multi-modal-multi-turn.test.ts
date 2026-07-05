import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { MlexModel } from "../index.js";
import { hasCapability, registry } from "./models.js";

const REPO_ROOT = resolve(import.meta.dirname, "..", "..", "..");
const sample = (name: string) => join(REPO_ROOT, "samples", name);

// Multi-turn conversation where the first turn carries an image: the
// vision tower runs once for that turn, its features land in the cache
// pool, and the text-only follow-up (caller grows the message array
// itself - see `multi-turn.test.ts`) reuses it. Mirrors the Rust
// `gemma4_multi_turn_conversation_with_an_image` test. Any vision-capable
// family works (Gemma4's classic/unified towers, Qwen3.5-VL's tower, ...).
const mmModel = registry().find((m) => hasCapability(m.dir, "vision_config"));

describe.skipIf(!mmModel)("multi-modal multi-turn conversation", () => {
  it("answers a text follow-up about an image sent in a previous turn", async () => {
    const model = await MlexModel.load(mmModel!.dir);

    const messages: Array<{
      role: string;
      content: string;
      images?: Buffer[];
    }> = [
      {
        role: "user",
        content: "Describe this image in one short sentence.",
        images: [readFileSync(sample("image1.jpg"))],
      },
    ];
    const { text: first } = await model.generate(messages, {
      maxTokens: 48,
      temperature: 0,
    });
    console.log(`[multi-turn] turn 1 (image): ${first}`);
    expect(first.trim().length).toBeGreaterThan(10);
    messages.push({ role: "assistant", content: first });

    messages.push({
      role: "user",
      content: "What main colors are in the image you just described?",
    });
    const { text: second } = await model.generate(messages, {
      maxTokens: 48,
      temperature: 0,
    });
    console.log(`[multi-turn] turn 2 (follow-up): ${second}`);
    expect(second.trim().length).toBeGreaterThan(10);
    messages.push({ role: "assistant", content: second });

    // user/assistant/user/assistant transcript
    expect(messages.map((m) => m.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
  }, 300_000);
});

describe.skipIf(mmModel)(
  "multi-modal multi-turn (no capable model available)",
  () => {
    it("skips: no CI-safe vision-capable checkpoint under MLEX_MODELS_DIR", () => {
      expect(mmModel).toBeUndefined();
    });
  },
);
