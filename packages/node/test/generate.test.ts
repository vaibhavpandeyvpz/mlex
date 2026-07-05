import { describe, expect, it } from "vitest";
import { MlexModel } from "../index.js";
import { registry } from "./models.js";

describe("MlexModel.generate", () => {
  const models = registry();

  it.skipIf(models.length === 0)(
    "produces non-empty, bounded output",
    async () => {
      for (const model of models) {
        const m = await MlexModel.load(model.dir);
        const streamedText: string[] = [];
        const { text: reply } = await m.generate(
          [{ role: "user", content: "Count from one to three." }],
          // `enableThinking: false` keeps this a structural "non-empty
          // output" check rather than a reasoning test: some checkpoints
          // (e.g. MiniCPM5) spontaneously open a `<think>` span even
          // without an explicit request, which this small `maxTokens`
          // budget can truncate before the model closes it, leaving the
          // post-reasoning `text` empty.
          { maxTokens: 16, enableThinking: false },
          (err, tok) => {
            expect(err).toBeNull();
            // Some checkpoints (e.g. Gemma4) spontaneously wrap part of
            // their raw output in a reasoning/"thinking" span even without
            // `enableThinking` requested - `reply.text` already has that
            // span split out (see `reply.reasoning`), so only `"text"`-kind
            // streamed pieces should be expected to reassemble into it.
            if (tok.kind === "text") {
              streamedText.push(tok.text);
            }
          },
        );
        expect(reply.length).toBeGreaterThan(0);
        expect(streamedText.join("")).toBe(reply);
      }
    },
  );

  it.skipIf(models.length === 0)(
    "greedy (temperature 0) generation is deterministic",
    async () => {
      const model = models[0];
      const m = await MlexModel.load(model.dir);
      const messages = [{ role: "user", content: "Say hi." }];
      const { text: a } = await m.generate(messages, {
        maxTokens: 8,
        temperature: 0,
      });
      const { text: b } = await m.generate(messages, {
        maxTokens: 8,
        temperature: 0,
      });
      expect(a).toBe(b);
    },
  );

  // A leading `role: "system"` message is a first-class part of the chat
  // template contract (every family's template special-cases
  // `messages[0].role in ['system', 'developer']`), not something bolted
  // on for one architecture. This test exercises the full generation path
  // with a system turn without assuming every tiny checkpoint will obey an
  // arbitrary style instruction perfectly.
  it.skipIf(models.length === 0)(
    "generation with a leading system message is non-empty",
    async () => {
      for (const model of models) {
        const m = await MlexModel.load(model.dir);
        const messages = [
          {
            role: "system",
            content: "You are a terse assistant. Answer in one short sentence.",
          },
          { role: "user", content: "What's the capital of France?" },
        ];
        const { text } = await m.generate(messages, {
          maxTokens: 16,
          temperature: 0,
          enableThinking: false,
        });
        expect(text.trim().length).toBeGreaterThan(0);
      }
    },
  );

  if (models.length === 0) {
    it("skips: no CI-safe models found under MLEX_MODELS_DIR", () => {
      expect(true).toBe(true);
    });
  }
});
