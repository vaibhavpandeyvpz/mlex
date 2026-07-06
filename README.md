# mlex

**Run your favorite LLMs at blazing-fast speeds on Apple Silicon.**

`mlex` is a from-scratch Rust runtime for [Apple MLX](https://github.com/ml-explore/mlx) — no Python, no system MLX install, no PyTorch. Point it at a model directory from the Hugging Face Hub, and it loads straight onto the GPU through Metal, ready to chat, call tools, and see/hear. It ships as a native Rust crate ([`mlex`](https://crates.io/crates/mlex)) and a native Node.js package ([`mlex.js`](https://www.npmjs.com/package/mlex.js)), both built from the same core.

Why another inference runtime? Because Apple Silicon deserves a first-class, dependency-free way to run modern quantized LLMs — not a Python environment shimmed through FFI, but a real static binary that loads a model and starts generating tokens in milliseconds.

## Highlights

- 🚀 **Blazing fast, GPU-accelerated.** Built directly on MLX's Metal backend — the same engine behind `mlx-lm`, minus the Python interpreter.
- 📦 **Zero runtime dependencies.** `mlx`/`mlx-c` are vendored and statically linked; `cargo build` (or `npm install`, which ships prebuilt binaries) is all it takes.
- 🧠 **Broad architecture support.** Qwen2, Qwen3, Qwen3.5 (dense + MoE + vision-capable variants), Gemma4, NemotronH, DharaAR, and any vanilla-Llama-shaped checkpoint (e.g. MiniCPM5) — see the [full table](#supported-models) below.
- 🔢 **Every quantization scheme that matters.** Dense bf16/fp16, affine 2/3/4/5/6/8-bit at any group size, `mxfp4`, `mxfp8`, `nvfp4`, and mixed per-layer precision checkpoints like **OptiQ** or Google **QAT** exports — not a hardcoded allowlist, but a direct read of `config.json`'s `quantization` section.
- 🖼️ **Multi-modal.** Image input works on vision-capable Qwen3.5 checkpoints, while Gemma4 checkpoints with vision/audio towers additionally accept audio clips and video (uniformly sampled into frames) right alongside text, in the same turn.
- 💬 **System prompts.** A leading `role: "system"` message is a first-class part of every supported chat template, the same convention as the OpenAI/Anthropic APIs.
- 🧩 **Reasoning / "thinking".** Opt into native "thinking" mode on the families that support it, with an optional token budget and the reasoning span split out of the final answer automatically.
- 🛠️ **Tool calling.** OpenAI-style function schemas in, parsed tool calls out — Hermes-style JSON and Gemma's native format both supported.
- ⚡ **Automatic, stateless prompt caching.** No session handle to manage: pass the full conversation on every call (OpenAI/Anthropic-style) and a transparent cache pool reuses KV state for whatever prefix was already computed, even across independent calls sharing a system prompt.
- 🧵 **Streaming.** Every generation entry point supports per-token callbacks.

## Supported models

Any checkpoint below works across _every_ quantization scheme `mlex` supports (bf16/fp16, affine 2–8 bit, mxfp4/mxfp8/nvfp4, OptiQ, QAT, ...) — quantization is resolved dynamically from `config.json`, not hardcoded per model.

| `model_type`                                                     | Family               | Notes                                                             |
| ---------------------------------------------------------------- | -------------------- | ----------------------------------------------------------------- |
| `qwen2`, `llama`                                                 | Qwen2 / Llama-shaped | Also covers MiniCPM5 and similar vanilla-GQA checkpoints          |
| `qwen3`                                                          | Qwen3                | Dense, with QK-norm                                               |
| `qwen3_5`, `qwen3_5_moe` (+ `_text` variants)                    | Qwen3.5              | Dense, Mixture-of-Experts, and vision-capable variants            |
| `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` | Gemma4               | Text-only, unified, and multi-modal (**vision + audio**) variants |
| `nemotron_h`                                                     | NemotronH            | Hybrid Mamba2 / GatedDelta / attention layers                     |
| `dhara_ar`                                                       | DharaAR              | Canon convolution layers, post-RoPE QK-norm, logit softcap        |

Multi-modal support is currently split by family: image input works on Qwen3.5 checkpoints that ship vision weights, while Gemma4 checkpoints with vision/audio tower weights support image, audio, and video input (e.g. `mlx-community/gemma-4-e2b-it-OptiQ-4bit`).

## Project layout

This is a Cargo + npm monorepo with a single Rust crate and one native Node.js package built from it:

```
mlex/
├── crates/
│   ├── mlex/         # the runtime — published to crates.io as `mlex`
│   └── mlex-node/     # NAPI-RS glue, compiles `mlex` into a Node native addon (not published)
└── packages/
    └── node/          # the npm package — published as `mlex.js`, wraps mlex-node's addon
```

`mlex` used to depend on a separate `mlex-sys` crate for the raw `mlx-c` FFI bindings; that's since been folded in as a private module so there's exactly one Rust crate to `cargo add`. See [`crates/mlex/README.md`](crates/mlex/README.md) and [`crates/mlex-node/README.md`](crates/mlex-node/README.md) for the details, and [`packages/node/README.md`](packages/node/README.md) for the npm package.

## Quickstart

### Rust

```bash
cargo add mlex
```

```rust
use mlex::generate::{GenerateOptions, Session};
use mlex::tokenizer::ChatMessage;
use std::path::Path;

fn main() -> mlex::Result<()> {
    let session = Session::load(Path::new("./models/Qwen3-0.6B-4bit"))?;
    let messages = vec![ChatMessage::user("Say hello in five words.")];

    let reply = session.generate_cached(&messages, None, GenerateOptions::default(), |_token| true)?;

    println!("{}", reply.text);
    Ok(())
}
```

### Node.js / TypeScript

```bash
npm install mlex.js
```

```js
import { MlexModel } from "mlex.js";

const model = await MlexModel.load("./models/Qwen3-0.6B-4bit");
const messages = [{ role: "user", content: "Say hello in five words." }];
const { text } = await model.generate(messages);
console.log(text);
```

`generate` is the single entry point on the JS side: it always resolves `{ text, toolCalls }`, and there's no separate session/conversation object — tools are just another `options` field. Multi-turn conversations, streaming, system prompts, sampling controls, tool calling, and multi-modal input all work the same way in both languages — see the crate/package READMEs linked above for the full API with examples, or jump straight to a few below.

### System prompts

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

### Multi-turn, with streaming

```js
const messages = [{ role: "user", content: "What's the capital of France?" }];
const { text } = await model.generate(messages, undefined, (err, tok) => {
  if (!err) process.stdout.write(tok.text);
});

messages.push({ role: "assistant", content: text });
messages.push({ role: "user", content: "What's its population?" });
await model.generate(messages);
```

### Tool calling

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

const { text, toolCalls } = await model.generate(
  [{ role: "user", content: "What's the weather in Paris?" }],
  { tools },
);
```

### Reasoning / "thinking"

Qwen3/3.5/3.6, Gemma4, MiniCPM5, and NemotronH checkpoints support an opt-in "thinking" mode. Turn it on and optionally cap how long it may run (mirroring Anthropic's extended-thinking `budget_tokens`) — the reasoning span is split out of the reply automatically, in both languages:

```js
const { text, reasoning } = await model.generate(messages, {
  enableThinking: true,
  reasoningBudgetTokens: 256,
});
if (reasoning) console.log("[thinking]", reasoning);
console.log(text);
```

```rust
let options = GenerateOptions { enable_thinking: Some(true), reasoning_budget_tokens: Some(256), ..Default::default() };
let reply = session.generate_cached(&messages, None, options, |_| true)?;
if let Some(reasoning) = &reply.reasoning {
    println!("[thinking] {reasoning}");
}
println!("{}", reply.text);
```

### Multi-modal (image/audio/video)

```js
import { readFileSync } from "node:fs";

const model = await MlexModel.load("./models/gemma-4-e2b-it-OptiQ-4bit");
if (model.supportsImages()) {
  const messages = [
    {
      role: "user",
      content: "What is in this photo?",
      images: [readFileSync("photo.jpg")],
    },
  ];
  const { text } = await model.generate(messages);
  console.log(text);
}
```

## Building from source

Requirements (macOS on Apple Silicon):

- Xcode Command Line Tools (`xcode-select --install`) — provides a C++17 toolchain and `libclang` for `bindgen`.
- [CMake](https://cmake.org/) — `brew install cmake`.
- Rust (stable) — [rustup.rs](https://rustup.rs).
- Node.js ≥ 18, if building the npm package.

```bash
git clone https://github.com/vaibhavpandeyvpz/mlex.git
cd mlex

# Rust
cargo build --release -p mlex

# Node.js
cd packages/node
npm install
npm run build
npm test
```

The `mlx`/`mlx-c` C++ sources are vendored under `crates/mlex/vendor/` (pinned commits documented in `crates/mlex/vendor/PINS.md`) and built once via CMake by `crates/mlex/build.rs`; no separate MLX installation is needed.

## Testing & CI

- **Rust:** `cargo test -p mlex` — unit tests plus integration suites covering generation, sampling, multi-turn/prompt-caching, tool calling, multi-modal input, and full architecture regression.
- **Node.js:** `npm test` in `packages/node` — a mirrored Vitest suite exercising the same surface through the native addon.
- Both suites auto-discover locally downloaded model checkpoints (Hugging Face hub cache layout) and gate which ones run in CI by **measured peak resident memory** (not disk size) against a 5 GB ceiling, so the exact same tests run unattended on GitHub Actions `macos-latest` runners against small models, and locally against anything you've downloaded — set `MLEX_INCLUDE_LARGE_MODELS=1` to also exercise larger, local-only checkpoints.
- `.github/workflows/ci.yml` builds, tests, and computes coverage (`cargo-llvm-cov` + `vitest --coverage`) on every push/PR.
- `.github/workflows/publish.yml` publishes `mlex` to crates.io and npm whenever a `v*.*.*` tag is pushed.

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

Issues and pull requests are welcome. New architectures typically only need a new module under `crates/mlex/src/models/` plus a dispatch arm in `crates/mlex/src/models/mod.rs`; new quantization schemes live in `crates/mlex/src/quant.rs`.
