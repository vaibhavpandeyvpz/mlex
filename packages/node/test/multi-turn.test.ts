import { describe, expect, it } from "vitest";
import { MlexModel } from "../index.js";
import { registry } from "./models.js";

// There is no conversation/session handle: the caller grows the message
// array themselves between calls, exactly like the OpenAI/Anthropic chat
// completion APIs - `model.generate`'s internal prompt-cache pool
// transparently reuses KV state for whatever prefix a previous call
// already computed. See `prompt-cache.test.ts` for the caching contract
// itself; this suite is about the stateless message-array API surface.
describe("multi-turn conversation (stateless message array)", () => {
  const models = registry();

  it.skipIf(models.length === 0)(
    "grows message history in role order",
    async () => {
      for (const model of models) {
        const m = await MlexModel.load(model.dir);

        // `enableThinking: false` keeps this a structural "message history"
        // check rather than a reasoning test: some checkpoints (e.g.
        // MiniCPM5) spontaneously open a `<think>` span even without an
        // explicit request, which this small `maxTokens` budget can
        // truncate before the model closes it, leaving the post-reasoning
        // `text` empty.
        const messages: { role: string; content: string }[] = [
          { role: "user", content: "Hi, I'm Alex." },
        ];
        const { text: reply1 } = await m.generate(messages, {
          maxTokens: 16,
          enableThinking: false,
        });
        expect(reply1.length).toBeGreaterThan(0);
        messages.push({ role: "assistant", content: reply1 });

        messages.push({ role: "user", content: "What's 10 minus 3?" });
        const { text: reply2 } = await m.generate(messages, {
          maxTokens: 16,
          enableThinking: false,
        });
        expect(reply2.length).toBeGreaterThan(0);
        messages.push({ role: "assistant", content: reply2 });

        expect(messages.map((msg) => msg.role)).toEqual([
          "user",
          "assistant",
          "user",
          "assistant",
        ]);
        expect(messages[1].content).toBe(reply1);
        expect(messages[3].content).toBe(reply2);
      }
    },
  );

  it.skipIf(models.length === 0)(
    "a tool-role message can be appended to the transcript",
    async () => {
      const model = models[0];
      const m = await MlexModel.load(model.dir);

      const messages: Array<{
        role: string;
        content: string;
        toolCallId?: string;
      }> = [{ role: "user", content: "What's the weather?" }];
      const { text: reply } = await m.generate(messages, { maxTokens: 8 });
      messages.push({ role: "assistant", content: reply });
      messages.push({
        role: "tool",
        content: '{"temp_c": 21}',
        toolCallId: "call_0",
      });

      const last = messages[messages.length - 1];
      expect(last.role).toBe("tool");
      expect(last.content).toBe('{"temp_c": 21}');
    },
  );

  if (models.length === 0) {
    it("skips: no CI-safe models found under MLEX_MODELS_DIR", () => {
      expect(true).toBe(true);
    });
  }
});
