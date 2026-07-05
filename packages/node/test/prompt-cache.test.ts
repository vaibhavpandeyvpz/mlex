import { describe, expect, it } from "vitest";
import { MlexModel } from "../index.js";
import { registry } from "./models.js";

// `MlexModel.generate`'s internal prompt-cache pool is stateless from the
// caller's perspective (no session/conversation handle - see
// `multi-turn.test.ts`): these tests prove cache reuse is numerically
// transparent by comparing an incrementally-grown call against a full
// from-scratch recompute of the same rendered transcript.
describe("stateless prompt caching", () => {
  const models = registry();

  it.skipIf(models.length === 0)(
    "turn 2 via the cache pool matches a full recompute over the rendered transcript",
    async () => {
      for (const model of models) {
        const m = await MlexModel.load(model.dir);

        const messages: { role: string; content: string }[] = [
          { role: "user", content: "My favorite color is blue." },
        ];
        const { text: turn1Reply } = await m.generate(messages, {
          maxTokens: 6,
          temperature: 0,
        });
        messages.push({ role: "assistant", content: turn1Reply });
        messages.push({ role: "user", content: "What is 2+2?" });
        const { text: turn2ViaCache } = await m.generate(messages, {
          maxTokens: 6,
          temperature: 0,
        });

        // Full recompute: a fresh model instance (empty pool) generating
        // once over the exact same rendered transcript.
        const fresh = await MlexModel.load(model.dir);
        const { text: fullRecompute } = await fresh.generate(messages, {
          maxTokens: 6,
          temperature: 0,
        });

        expect(turn2ViaCache).toBe(fullRecompute);
      }
    },
  );

  it.skipIf(models.length === 0)(
    "an unrelated first message starts a fresh (uncached) lineage, not a broken one",
    async () => {
      const model = models[0];
      const m = await MlexModel.load(model.dir);

      await m.generate([{ role: "user", content: "Remember 7." }], {
        maxTokens: 4,
        temperature: 0,
      });

      // No explicit reset exists in a stateless API - an unrelated message
      // list simply misses the pool and is computed cold, matching a
      // brand-new model instance's output for the same prompt.
      const sayHi = [{ role: "user", content: "Say hi." }];
      const { text: reply } = await m.generate(sayHi, {
        maxTokens: 4,
        temperature: 0,
      });

      const fresh = await MlexModel.load(model.dir);
      const { text: freshReply } = await fresh.generate(sayHi, {
        maxTokens: 4,
        temperature: 0,
      });

      expect(reply).toBe(freshReply);
    },
  );

  if (models.length === 0) {
    it("skips: no CI-safe models found under MLEX_MODELS_DIR", () => {
      expect(true).toBe(true);
    });
  }
});
