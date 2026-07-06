# mlex

**Run your favorite LLMs at blazing-fast speeds on Apple Silicon — from Node.js.**

`mlex` is a native Node.js addon (prebuilt, no compiler required) wrapping [Apple MLX](https://github.com/ml-explore/mlx) for local LLM inference on the GPU via Metal. It loads MLX-format checkpoints straight from a Hugging Face Hub-style model directory and gives you a single OpenAI-style `generate` call, with streaming, system prompts, tool calling, and multi-modal (image/audio/video) input built in.

Under the hood, this package wraps the [`mlex`](https://crates.io/crates/mlex) Rust crate.

## Features

- **No Python, no system MLX install.** Everything needed to run MLX models is compiled into the native addon.
- **Broad quantization support.** Dense bf16/fp16, affine 2/3/4/5/6/8-bit at any group size, `mxfp4`, `mxfp8`, `nvfp4`, and mixed per-layer precision checkpoints such as **OptiQ** or Google **QAT** exports.
- **Wide architecture coverage.** Qwen2/Qwen3/Qwen3.5 (dense + MoE + vision-capable variants), Gemma4 (text + multi-modal), NemotronH, DharaAR, and vanilla-Llama-shaped checkpoints (e.g. MiniCPM5).
- **Multi-modal.** Attach images on vision-capable Qwen3.5 or Gemma4 checkpoints, and attach audio/video on Gemma4 checkpoints with the corresponding towers.
- **System prompts.** A leading `{ role: 'system' }` message is rendered by every supported chat template, exactly like the OpenAI/Anthropic system role.
- **Reasoning / "thinking".** Opt into native "thinking" mode on the families that support it, with an optional token budget and the reasoning span split out of `text` automatically.
- **Usage metrics.** Every reply includes an OpenAI/Anthropic-style `usage` block (`promptTokens`, `cachedTokens`, `completionTokens`).
- **Tool calling.** Pass OpenAI-style function schemas; get parsed tool calls back out of the reply.
- **Streaming.** Optional `onToken` callback fires once per generated token.
- **Stateless, automatic prompt caching.** No session handle to manage — grow the `messages` array yourself between calls (like the OpenAI/Anthropic APIs) and a transparent prompt-cache pool reuses KV state for whatever prefix was already computed.

## Installation

```bash
npm install mlex
```

Prebuilt binaries are published for `aarch64-apple-darwin` only. `mlex` requires macOS on Apple Silicon — MLX's Metal backend doesn't run on Intel Macs, so there's no x86_64 build.

## Quickstart

```js
import { MlexModel } from "mlex";

const model = await MlexModel.load("./models/Qwen3-0.6B-4bit");

const messages = [{ role: "user", content: "Say hello in five words." }];
const { text } = await model.generate(messages);
console.log(text);
```

`generate` always resolves with `{ text, toolCalls, usage, reasoning }` (`toolCalls` is an empty array unless you pass `tools` — see [Tool calling](#tool-calling) below; `reasoning` is `undefined` unless the model emitted a recognized reasoning span — see [Reasoning](#reasoning--thinking) below).

### System prompts

A leading `{ role: 'system', content: ... }` message is rendered by every supported chat template (Qwen, Gemma4, NemotronH, DharaAR, ...) exactly like OpenAI/Anthropic's system role — put instructions there rather than folding them into the first user turn:

```js
const messages = [
  {
    role: "system",
    content: "You are a terse assistant. Answer in one sentence.",
  },
  { role: "user", content: "What's the capital of France?" },
];
const { text } = await model.generate(messages);
```

### Streaming tokens

```js
const { text } = await model.generate(messages, undefined, (err, token) => {
  if (!err) process.stdout.write(token.text);
});
```

Each streamed `token` also carries a `kind` — `'text'`, `'reasoning'`, or `'toolCall'` — so you can route reasoning, the final answer, and in-progress tool-call syntax to separate UI regions live, instead of waiting for the resolved `{ text, reasoning, toolCalls }` once generation finishes (mirroring OpenAI/Anthropic's typed streaming deltas). It's best-effort at token granularity — a marker split across two tokens is only classified correctly once its second half arrives:

```js
let liveAnswer = "";
let liveThinking = "";
await model.generate(messages, { enableThinking: true }, (err, token) => {
  if (err) return;
  if (token.kind === "reasoning") liveThinking += token.text;
  else if (token.kind === "text") liveAnswer += token.text;
  // token.kind === 'toolCall' -> raw, not-yet-parsed tool-call syntax
});
```

### Multi-turn conversations

There's no conversation/session object — just grow the `messages` array yourself and call `generate` again. The shared prefix (system prompt, earlier turns) is served from an internal cache automatically:

```js
const messages = [{ role: "user", content: "What's the capital of France?" }];
const { text } = await model.generate(messages);

messages.push({ role: "assistant", content: text });
messages.push({ role: "user", content: "What's its population?" });
const { text: reply2 } = await model.generate(messages);
```

### Sampling options

```js
const { text } = await model.generate(messages, {
  maxTokens: 512,
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
  seed: 42,
});
```

### Reasoning / "thinking"

Qwen3/3.5/3.6, Gemma4, MiniCPM5, and NemotronH checkpoints support an opt-in "thinking" mode via their chat template's `enable_thinking` variable. Set `options.enableThinking` to turn it on; any reasoning span the model emits (`<think>...</think>` or Gemma4's `<|channel>thought...<channel|>`) is stripped out of `text` automatically and returned separately as `reasoning`. `options.reasoningBudgetTokens` caps how long that span may run — once the budget is hit, generation is force-moved to the final answer, mirroring Anthropic's extended-thinking `budget_tokens`:

```js
const { text, reasoning } = await model.generate(messages, {
  enableThinking: true,
  reasoningBudgetTokens: 256,
});
if (reasoning) console.log("[thinking]", reasoning);
console.log(text);

// Round-trip the reasoning back into history on the next turn, matching
// templates that special-case `reasoningContent`:
messages.push({
  role: "assistant",
  content: text,
  reasoningContent: reasoning,
});
```

Leaving `enableThinking` unset keeps the template's own default, which for every family above means reasoning is off.

### Usage metrics

Every `generate` call reports token usage, mirroring the `usage` block OpenAI/Anthropic return alongside a chat completion:

```js
const { usage } = await model.generate(messages);
console.log(usage.promptTokens, usage.cachedTokens, usage.completionTokens);
```

`cachedTokens` is how many of `promptTokens` were served from the internal prompt-cache pool (an exact-prefix hit) rather than recomputed this call.

### Tool calling

Pass `tools` as part of the same `options` object — there's no separate method:

```js
const tools = [
  {
    name: "get_weather",
    description: "Get the current weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
];

const messages = [{ role: "user", content: "What's the weather in Paris?" }];
const { text, toolCalls } = await model.generate(messages, { tools });

for (const call of toolCalls) {
  console.log(`model wants to call ${call.name} with ${call.argumentsJson}`);
}
```

Feed a tool's result back in as a `{ role: 'tool', toolCallId, content }` message (and record the assistant's `toolCalls` on its turn) to continue the conversation across a tool round-trip.

### Multi-modal input

```js
import { readFileSync } from "node:fs";

if (model.supportsImages()) {
  const messages = [
    {
      role: "user",
      content: "Describe this image.",
      images: [readFileSync("photo.jpg")],
    },
  ];
  const { text } = await model.generate(messages);
  console.log(text);
}
```

`model.supportsImages()` is true on image-capable Qwen3.5 and Gemma4 checkpoints. `audios` and `videos` are additionally supported on Gemma4 checkpoints with the corresponding towers; `model.supportsAudio()` gates audio input, and video frames are routed through the vision tower. All supported media types can be attached to the same message alongside text.

## API reference

### `MlexModel`

| Member           | Signature                                                                          | Description                                                                                                                                                                                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MlexModel.load` | `(modelPath: string) => Promise<MlexModel>`                                        | Load a model directory (`config.json`, safetensors shards, `tokenizer.json`, optional `chat_template.jinja`).                                                                                                                                                       |
| `generate`       | `(messages, options?, onToken?) => Promise<{ text, toolCalls, usage, reasoning }>` | Generate a reply to the full message transcript. Pass `options.tools` to enable tool calling — `toolCalls` is empty otherwise. Pass `options.enableThinking` to enable reasoning — `reasoning` is `undefined` otherwise (unless the model reasons unconditionally). |
| `supportsImages` | `() => boolean`                                                                    | Whether the loaded checkpoint accepts image/video input.                                                                                                                                                                                                            |
| `supportsAudio`  | `() => boolean`                                                                    | Whether the loaded checkpoint accepts audio input.                                                                                                                                                                                                                  |

### Types

```ts
interface JsChatMessage {
  role: string; // "system" | "user" | "assistant" | "tool"
  content: string;
  images?: Buffer[];
  audios?: Buffer[];
  videos?: Buffer[];
  toolCallId?: string; // set on a `role: "tool"` reply
  toolCalls?: JsToolCall[]; // set on an assistant turn that issued calls
  reasoningContent?: string; // round-trips a prior reply's `reasoning`
}

interface JsGenerateOptions {
  maxTokens?: number; // default 256
  temperature?: number; // default 0 (greedy)
  topP?: number; // default 1.0
  topK?: number;
  seed?: number;
  tools?: JsTool[]; // enables tool calling for this call
  enableThinking?: boolean; // opt into "thinking" mode, where supported
  reasoningBudgetTokens?: number; // cap on the reasoning span's length
}

interface JsToken {
  id: number;
  text: string;
  finished: boolean;
  kind: "text" | "reasoning" | "toolCall";
}

interface JsTool {
  name: string;
  description?: string;
  parameters: object; // JSON Schema
}

interface JsToolCall {
  id: string;
  name: string;
  argumentsJson: string; // JSON-encoded arguments object
}

interface JsUsage {
  promptTokens: number;
  cachedTokens: number;
  completionTokens: number;
}

interface JsGenerateResult {
  text: string;
  toolCalls: JsToolCall[]; // empty unless `options.tools` was passed and used
  usage: JsUsage;
  reasoning?: string; // present if the model emitted a recognized reasoning span
}
```

Full type definitions ship in `index.d.ts`.

## Supported architectures

| `model_type` in `config.json`                                    | Family               | Notes                                                         |
| ---------------------------------------------------------------- | -------------------- | ------------------------------------------------------------- |
| `qwen2`, `llama`                                                 | Qwen2 / Llama-shaped | Also covers MiniCPM5 and similar vanilla-GQA checkpoints      |
| `qwen3`                                                          | Qwen3                | Dense, with QK-norm                                           |
| `qwen3_5`, `qwen3_5_moe` (+ `_text` variants)                    | Qwen3.5              | Dense, Mixture-of-Experts, and vision-capable variants        |
| `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` | Gemma4               | Text-only, unified, and multi-modal (vision + audio) variants |
| `nemotron_h`                                                     | NemotronH            | Hybrid Mamba2 / GatedDelta / attention layers                 |
| `dhara_ar`                                                       | DharaAR              | Canon convolution layers, post-RoPE QK-norm, logit softcap    |

See the [top-level project README](https://github.com/vaibhavpandeyvpz/mlex#readme) for the full picture, including the Rust API and how to build from source.

## License

MIT
