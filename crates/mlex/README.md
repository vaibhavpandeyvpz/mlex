# `mlex`

**Run your favorite LLMs at blazing-fast speeds on Apple Silicon.**

`mlex` is a safe, idiomatic Rust runtime built directly on [Apple MLX](https://github.com/ml-explore/mlx) (via the official `mlx-c` C API, vendored and statically linked — no system MLX install, no Python required). It loads MLX-format checkpoints straight off the Hugging Face Hub layout and runs them on the GPU through Metal, with first-class support for every quantization scheme the MLX community actually ships, tool calling, and multi-modal (image/audio/video) input.

This crate is the engine underneath the [`mlex` npm package](https://www.npmjs.com/package/mlex) (Node.js bindings), but it's a complete, self-contained Rust library on its own.

## Features

- **Zero system dependencies at runtime.** `mlx` and `mlx-c` are vendored, pinned, and built from source via `build.rs` (CMake). `cargo build` on a fresh checkout is all it takes.
- **Broad quantization support.** Dense bf16/fp16, affine 2/3/4/5/6/8-bit at any group size, `mxfp4`, `mxfp8`, `nvfp4`, and mixed per-layer precision checkpoints such as **OptiQ** or Google **QAT** exports.
- **Wide architecture coverage.** Qwen2/Qwen3/Qwen3.5 (dense + MoE + vision-capable variants), Gemma4 (text + multi-modal), NemotronH (hybrid Mamba2/attention), DharaAR, and vanilla-Llama-shaped checkpoints (e.g. MiniCPM5) — see the table below.
- **Multi-modal.** Image input works on vision-capable Qwen3.5 checkpoints, while Gemma4 checkpoints with vision/audio towers additionally accept audio clips and video (uniformly sampled into frames) alongside text.
- **System prompts.** A leading `role: "system"` message is rendered by every supported chat template, exactly like the OpenAI/Anthropic system role.
- **Reasoning / "thinking".** Opt into Qwen3/3.5/3.6, Gemma4, MiniCPM5, and NemotronH's native "thinking" mode, with an optional token budget and the reasoning span automatically split out of the final answer.
- **Tool calling.** Render OpenAI-style tool/function schemas into the model's native chat template and parse tool calls back out of the reply (Hermes-style JSON and Gemma's native key/value format).
- **Stateless, automatic prompt caching.** [`Session::generate_cached`] takes the full message transcript on every call (mirroring the OpenAI/Anthropic chat APIs) and transparently reuses KV cache state for whatever prefix a previous call already computed — no session handle to manage, no explicit cache invalidation.
- **Streaming.** Every generation entry point accepts an `on_token` callback invoked once per generated token.

## Supported architectures

| `model_type`                                                     | Family               | Notes                                                         |
| ---------------------------------------------------------------- | -------------------- | ------------------------------------------------------------- |
| `qwen2`, `llama`                                                 | Qwen2 / Llama-shaped | Also covers MiniCPM5 and similar vanilla-GQA checkpoints      |
| `qwen3`                                                          | Qwen3                | Dense, with QK-norm                                           |
| `qwen3_5`, `qwen3_5_moe` (+ `_text` variants)                    | Qwen3.5              | Dense, Mixture-of-Experts, and vision-capable variants        |
| `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` | Gemma4               | Text-only, unified, and multi-modal (vision + audio) variants |
| `nemotron_h`                                                     | NemotronH            | Hybrid Mamba2 / GatedDelta / attention layers                 |
| `dhara_ar`                                                       | DharaAR              | Canon convolution layers, post-RoPE QK-norm, logit softcap    |

Any checkpoint using one of the quantization schemes above works for every architecture where the underlying ops are wired up — this isn't a per-model allowlist, it's a function of `config.json`'s `quantization` section plus the tensors actually present in the checkpoint.

## Installation

```bash
cargo add mlex
```

Building from source requires:

- A C++17 toolchain (Xcode Command Line Tools on macOS).
- CMake (`brew install cmake`).
- `libclang` for `bindgen` (ships with Xcode Command Line Tools).

The vendored `mlx`/`mlx-c` C++ sources under `vendor/` are built once per `target/` via CMake and statically linked; subsequent builds are incremental like any other Rust crate.

> `mlex` targets macOS on Apple Silicon only. MLX's Metal backend doesn't run on Intel Macs, so x86_64 isn't a supported target.

## Quickstart

```rust
use mlex::generate::{GenerateOptions, Session};
use mlex::tokenizer::ChatMessage;
use std::path::Path;

fn main() -> mlex::Result<()> {
    let session = Session::load(Path::new("./models/Qwen3-0.6B-4bit"))?;

    let messages = vec![ChatMessage::user("Say hello in five words.")];
    let reply = session.generate_cached(
        &messages,
        None,
        GenerateOptions::default(),
        |_token| true, // return `false` to stop generation early
    )?;

    println!("{}", reply.text);
    Ok(())
}
```

### Streaming tokens

```rust
let reply = session.generate_cached(&messages, None, GenerateOptions::default(), |tok| {
    print!("{}", tok.text);
    true
})?;
```

### System prompts

A leading `role: "system"` message is a first-class part of the chat template contract — every supported family's template special-cases `messages[0]` being `system` (or `developer`), the same convention as the OpenAI/Anthropic APIs, rather than something that only works on one architecture:

```rust
let messages = vec![
    ChatMessage::system("You are a terse assistant. Answer in one sentence."),
    ChatMessage::user("What's the capital of France?"),
];
let reply = session.generate_cached(&messages, None, GenerateOptions::default(), |_| true)?;
```

### Multi-turn conversations

There is no session/conversation handle to manage — like the OpenAI and Anthropic chat completion APIs, you grow the message list yourself and pass the _full_ transcript on every call. An internal prompt-cache pool transparently reuses KV state for whatever prefix was already computed (including across independent calls that happen to share a prefix, e.g. a common system prompt):

```rust
let mut messages = vec![ChatMessage::user("What's the capital of France?")];
let reply = session.generate_cached(&messages, None, GenerateOptions::default(), |_| true)?;

messages.push(ChatMessage::assistant(reply.text));
messages.push(ChatMessage::user("What's its population?"));
let reply2 = session.generate_cached(&messages, None, GenerateOptions::default(), |_| true)?;
```

### Sampling

`temperature`, `top_p`, `top_k`, and `seed` are all first-class knobs on [`sampling::SamplingConfig`]:

```rust
use mlex::generate::GenerateOptions;
use mlex::sampling::SamplingConfig;

let options = GenerateOptions {
    max_tokens: 512,
    sampling: SamplingConfig { temperature: 0.7, top_p: 0.95, top_k: Some(40), seed: Some(42) },
    ..Default::default()
};
```

### Reasoning / "thinking"

Qwen3/3.5/3.6, Gemma4, MiniCPM5, and NemotronH checkpoints support an opt-in "thinking" mode via their chat template's `enable_thinking` variable. Set [`GenerateOptions::enable_thinking`] to turn it on; any reasoning span the model emits (`<think>...</think>` or Gemma4's `<|channel>thought...<channel|>`) is stripped out of `reply.text` automatically and returned separately as `reply.reasoning`:

```rust
let options = GenerateOptions {
    enable_thinking: Some(true),
    // Cap how long the model may spend reasoning before it's force-closed
    // and moved on to the final answer (mirrors Anthropic's `budget_tokens`).
    reasoning_budget_tokens: Some(256),
    ..Default::default()
};
let reply = session.generate_cached(&messages, None, options, |_| true)?;

if let Some(reasoning) = &reply.reasoning {
    println!("[thinking] {reasoning}");
}
println!("{}", reply.text);

// Round-trip the reasoning back into history on the next turn, matching
// templates that special-case `message.reasoning_content`:
messages.push(ChatMessage::assistant_with_reasoning(reply.text, reply.reasoning.unwrap_or_default()));
```

`enable_thinking: None` (the default) leaves the template's own default in place, which for every family above means reasoning is off — so existing code that doesn't set it keeps behaving exactly as before.

### Streaming reasoning and tool calls separately

Every streamed [`generate::GeneratedToken`] carries a `kind` (`Text`, `Reasoning`, or `ToolCall`) so a UI can render each live, mirroring OpenAI/Anthropic's typed streaming deltas rather than only being able to split them apart from the resolved [`generate::GenerateReply`] once generation finishes:

```rust
use mlex::streaming::TokenKind;

session.generate_cached(&messages, None, options, |tok| {
    match tok.kind {
        TokenKind::Reasoning => print!("\x1b[2m{}\x1b[0m", tok.text), // dim
        TokenKind::Text => print!("{}", tok.text),
        TokenKind::ToolCall => {} // raw, not-yet-parsed syntax; use `reply.tool_calls` once finished
    }
    true
})?;
```

Classification is marker-based and best-effort at token granularity — a marker straddling two tokens is only classified correctly once its second half arrives.

### Tool calling

```rust
use mlex::tools::{Tool, ToolFunction};
use serde_json::json;

let tools = vec![Tool {
    kind: "function".into(),
    function: ToolFunction {
        name: "get_weather".into(),
        description: Some("Get the current weather for a city".into()),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    },
}];

let messages = vec![ChatMessage::user("What's the weather in Paris?")];
let reply = session.generate_cached(
    &messages,
    Some(&tools),
    GenerateOptions::default(),
    |_| true,
)?;

for call in reply.tool_calls {
    println!("model wants to call {} with {}", call.name, call.arguments);
}
```

Feed a tool's result back in as a `role: "tool"` message (see `ChatMessage`'s `tool_call_id` field) and call `generate_cached` again to continue the conversation.

### Multi-modal input

```rust
use mlex::tokenizer::ChatMessage;

let image_bytes = std::fs::read("photo.jpg")?;
let messages = vec![ChatMessage::user_with_image("Describe this image.", image_bytes)];

if session.supports_images() {
    let reply = session.generate_cached(&messages, None, GenerateOptions::default(), |_| true)?;
    println!("{}", reply.text);
}
```

`Session::supports_images()` is true on image-capable Qwen3.5 and Gemma4 checkpoints. `ChatMessage::user_with_audio` and `ChatMessage::user_with_video` are additionally supported on Gemma4 checkpoints with audio/vision towers; `Session::supports_audio()` gates the audio path, and video is uniformly sampled into frames and processed through the same vision tower as still images.

## API surface

- [`generate::Session`] — load a model directory, generate/stream completions, all cache-aware via `generate_cached`.
- [`tokenizer::ChatMessage`] / [`tokenizer::ContentPart`] — multi-part (text/image/audio/video) chat turns.
- [`tools`] — `Tool`, `ToolCall`, `ToolCallFormat`, and `parse_tool_calls` for tool-calling support.
- [`sampling::SamplingConfig`] — temperature/top-p/top-k/seed controls.
- [`models`] — the `Model` enum and per-architecture implementations, if you need lower-level access (custom caches, raw forward passes).
- [`quant`] — quantization config parsing, if you're inspecting or building checkpoints.

Run `cargo doc --open -p mlex` for the full generated API reference.

## Testing

```bash
cargo test -p mlex
```

Integration tests auto-discover any model checkpoints under `MLEX_MODELS_DIR` (default `<repo>/models`, in Hugging Face hub cache layout) and gate on measured peak memory (`MLEX_MAX_MODEL_GB`, default 5 GB) so the same suite runs unattended in CI against small models and, locally, against anything you've downloaded — set `MLEX_INCLUDE_LARGE_MODELS=1` to also exercise larger local-only checkpoints. See the top-level [project README](https://github.com/vaibhavpandeyvpz/mlex#readme) for the full test/CI story.

## License

MIT
