import { describe, expect, it } from "vitest";
import { MlexModel } from "../index.js";
import { registry } from "./models.js";

const weatherTool = {
  name: "get_weather",
  description: "Get the current weather for a location",
  parameters: {
    type: "object",
    properties: { location: { type: "string", description: "City name" } },
    required: ["location"],
  },
};

describe("tool calling", () => {
  const models = registry();

  it.skipIf(models.length === 0)(
    "generate(messages, { tools }) resolves with text and a toolCalls array",
    async () => {
      let anyCallSeen = false;
      for (const model of models) {
        const m = await MlexModel.load(model.dir);
        const messages = [
          {
            role: "user",
            content:
              "What's the weather in Paris? You must respond only by calling the get_weather tool.",
          },
        ];
        const result = await m.generate(messages, {
          maxTokens: 64,
          tools: [weatherTool],
        });
        expect(typeof result.text).toBe("string");
        expect(Array.isArray(result.toolCalls)).toBe(true);
        if (result.toolCalls.length > 0) {
          anyCallSeen = true;
          expect(result.toolCalls[0].name).toBe("get_weather");
          expect(() =>
            JSON.parse(result.toolCalls[0].argumentsJson),
          ).not.toThrow();
        }
      }
      if (!anyCallSeen) {
        // Model-quality-dependent (small CI-safe models aren't guaranteed to
        // reliably call tools for any given prompt) - logged, not a hard
        // failure. See `crates/mlex/tests/tool_calling.rs` docs.
        console.warn(
          "[toolCalling] no CI-safe model emitted a parseable tool call for this prompt",
        );
      }
    },
  );

  if (models.length === 0) {
    it("skips: no CI-safe models found under MLEX_MODELS_DIR", () => {
      expect(true).toBe(true);
    });
  }
});
