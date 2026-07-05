#![deny(clippy::all)]

use std::path::PathBuf;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use serde_json::Value as JsonValue;

use mlex::generate::{GenerateOptions, Session};
use mlex::sampling::SamplingConfig;
use mlex::tokenizer::{AudioContent, ChatMessage, ContentPart, ImageContent, VideoContent};
use mlex::tools::{Tool, ToolCall, ToolFunction};

/// One chat turn, mirroring the OpenAI-style `{role, content}` shape used
/// by every JS chat API, plus optional media attachments (raw encoded
/// file bytes) for multimodal models: `images` (JPEG/PNG/...), `audios`
/// (WAV/MP3/...), `videos` (MP4/WebM/..., uniformly sampled into frames
/// through the vision tower).
///
/// There is no separate conversation/session handle - like the OpenAI and
/// Anthropic chat APIs, callers pass the *full* running transcript on
/// every call (growing this array themselves between calls) and
/// `MlexModel.generate`'s internal prompt-cache pool transparently reuses
/// KV state for whatever prefix was already computed by a previous call.
#[napi(object)]
pub struct JsChatMessage {
    pub role: String,
    pub content: String,
    pub images: Option<Vec<Buffer>>,
    pub audios: Option<Vec<Buffer>>,
    pub videos: Option<Vec<Buffer>>,
    /// Set on a `role: "tool"` turn: which previously-issued call this
    /// result answers.
    pub tool_call_id: Option<String>,
    /// Set on an assistant turn that issued tool calls (as returned by a
    /// prior `generate` call's `toolCalls`, when `options.tools` was
    /// passed).
    pub tool_calls: Option<Vec<JsToolCall>>,
    /// Set on an assistant turn to round-trip its reasoning/"thinking"
    /// content (as returned by a prior `generate` call's `reasoning`)
    /// back into history, matching templates that special-case it
    /// (Gemma4/Qwen-style).
    pub reasoning_content: Option<String>,
}

impl JsChatMessage {
    fn into_core(self) -> Result<ChatMessage> {
        let mut content = vec![ContentPart::Text(self.content)];
        for img in self.images.into_iter().flatten() {
            content.push(ContentPart::Image(ImageContent {
                bytes: img.to_vec(),
            }));
        }
        for aud in self.audios.into_iter().flatten() {
            content.push(ContentPart::Audio(AudioContent {
                bytes: aud.to_vec(),
            }));
        }
        for vid in self.videos.into_iter().flatten() {
            content.push(ContentPart::Video(VideoContent {
                bytes: vid.to_vec(),
            }));
        }
        let tool_calls = self
            .tool_calls
            .into_iter()
            .flatten()
            .map(JsToolCall::into_core)
            .collect::<Result<Vec<_>>>()?;
        Ok(ChatMessage {
            role: self.role,
            content,
            tool_calls,
            tool_call_id: self.tool_call_id,
            reasoning_content: self.reasoning_content,
        })
    }
}

/// An OpenAI-style tool ("function") declaration, passed via
/// `options.tools` to `MlexModel.generate`.
#[napi(object)]
pub struct JsTool {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema for the function's arguments, as a JSON string (e.g.
    /// `'{"type": "object", "properties": {...}}'`).
    pub parameters_json: String,
}

impl JsTool {
    fn into_core(self) -> Result<Tool> {
        let parameters: JsonValue = serde_json::from_str(&self.parameters_json)
            .map_err(|e| Error::from_reason(format!("invalid parameters_json: {e}")))?;
        Ok(Tool {
            kind: "function".to_string(),
            function: ToolFunction {
                name: self.name,
                description: self.description,
                parameters,
            },
        })
    }
}

/// A parsed tool call recovered from model output.
#[napi(object)]
#[derive(Clone)]
pub struct JsToolCall {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments object.
    pub arguments_json: String,
}

impl From<ToolCall> for JsToolCall {
    fn from(c: ToolCall) -> Self {
        JsToolCall {
            id: c.id,
            name: c.name,
            arguments_json: c.arguments.to_string(),
        }
    }
}

impl JsToolCall {
    fn into_core(self) -> Result<ToolCall> {
        let arguments: JsonValue = serde_json::from_str(&self.arguments_json)
            .map_err(|e| Error::from_reason(format!("invalid arguments_json: {e}")))?;
        Ok(ToolCall {
            id: self.id,
            name: self.name,
            arguments,
        })
    }
}

/// Sampling + length controls for a single `generate` call, plus an
/// optional tool ("function") declaration list.
///
/// Passing `tools` renders an OpenAI-style function schema into the chat
/// template and parses any tool calls back out of the reply (populating
/// [`JsGenerateResult::tool_calls`]) - there is no separate
/// `generateWithTools` method, this is the only entry point.
#[napi(object)]
pub struct JsGenerateOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub seed: Option<f64>,
    pub tools: Option<Vec<JsTool>>,
    /// Opt into "thinking" mode on models whose chat template supports it
    /// (Qwen3/3.5/3.6, Gemma4, MiniCPM5, NemotronH, ...). Leaving this
    /// unset keeps the template's own default, which for all of those is
    /// reasoning off.
    pub enable_thinking: Option<bool>,
    /// Cap, in tokens, on how long the model may spend inside a detected
    /// reasoning span before it's force-closed and generation moves on to
    /// the final answer - mirroring Anthropic's extended-thinking
    /// `budget_tokens`. Unset means no cap.
    pub reasoning_budget_tokens: Option<u32>,
}

impl JsGenerateOptions {
    /// Splits into `(tools, core_options)`; `tools` is handled separately
    /// from [`mlex::generate::GenerateOptions`] since it isn't a sampling
    /// parameter.
    fn into_parts(self) -> Result<(Option<Vec<Tool>>, GenerateOptions)> {
        let tools = self
            .tools
            .map(|ts| {
                ts.into_iter()
                    .map(JsTool::into_core)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let core_options = GenerateOptions {
            max_tokens: self.max_tokens.unwrap_or(256) as usize,
            sampling: SamplingConfig {
                temperature: self.temperature.unwrap_or(0.0) as f32,
                top_p: self.top_p.unwrap_or(1.0) as f32,
                top_k: self.top_k.map(|k| k as i32),
                seed: self.seed.map(|s| s as u64),
            },
            enable_thinking: self.enable_thinking,
            reasoning_budget_tokens: self.reasoning_budget_tokens.map(|b| b as usize),
        };
        Ok((tools, core_options))
    }
}

/// One streamed token, passed to the optional `onToken` callback of
/// [`MlexModel::generate`].
#[napi(object)]
pub struct JsToken {
    pub id: u32,
    pub text: String,
    pub finished: bool,
    /// Which span this token belongs to - `"text"` (the final answer),
    /// `"reasoning"` (inside a "thinking" span), or `"toolCall"` (raw,
    /// not-yet-parsed tool-call syntax) - mirroring OpenAI/Anthropic's
    /// typed streaming deltas so a UI can render each separately live,
    /// rather than only being able to split them apart from the final
    /// `{ text, reasoning, toolCalls }` once generation finishes.
    /// Best-effort at token granularity - see `mlex::streaming`.
    pub kind: String,
}

fn token_kind_str(k: mlex::streaming::TokenKind) -> &'static str {
    match k {
        mlex::streaming::TokenKind::Text => "text",
        mlex::streaming::TokenKind::Reasoning => "reasoning",
        mlex::streaming::TokenKind::ToolCall => "toolCall",
    }
}

/// A loaded MLX language model, ready to chat.
///
/// There is no session/conversation handle to manage: like the OpenAI and
/// Anthropic chat completion APIs, every call passes the full message
/// list, and an internal, always-on prompt-cache pool transparently
/// reuses KV state for whatever prefix (if any) a previous call already
/// computed - including from a *different* logical conversation that
/// happens to share a prefix (e.g. a common system prompt).
///
/// ```js
/// import { MlexModel } from 'mlex';
///
/// const model = await MlexModel.load('./models/Qwen3-0.6B-4bit');
/// const messages = [{ role: 'user', content: 'Say hi in five words.' }];
/// const { text } = await model.generate(messages, { maxTokens: 64 });
/// console.log(text);
///
/// // Continue the conversation by growing `messages` yourself and
/// // calling `generate` again - the shared prefix is served from cache.
/// messages.push({ role: 'assistant', content: text });
/// messages.push({ role: 'user', content: "What's 2+2?" });
/// const { text: reply2 } = await model.generate(messages, { maxTokens: 64 });
/// ```
#[napi]
pub struct MlexModel {
    session: Arc<Session>,
}

#[napi]
impl MlexModel {
    /// Load a model directory containing `config.json`, safetensors
    /// shards, `tokenizer.json`, and (optionally) `chat_template.jinja`.
    ///
    /// Supports dense bf16/fp16 checkpoints as well as every quantization
    /// scheme MLX ships (affine 2-8 bit at any group size, mxfp4, mxfp8,
    /// nvfp4) and mixed per-layer precision checkpoints such as OptiQ or
    /// Google QAT exports, wherever the underlying architecture is wired up.
    #[napi(factory)]
    pub async fn load(model_path: String) -> Result<Self> {
        let session = napi::bindgen_prelude::spawn_blocking(move || {
            Session::load(&PathBuf::from(model_path)).map_err(|e| Error::from_reason(e.to_string()))
        })
        .await
        .map_err(|e| Error::from_reason(format!("load task panicked: {e}")))??;
        Ok(MlexModel {
            session: Arc::new(session),
        })
    }

    /// Generate a reply to `messages` (the full transcript so far - see
    /// the type-level docs for how caching works across calls).
    ///
    /// Passing `options.tools` renders an OpenAI-style function schema
    /// into the chat template and parses any tool calls back out of the
    /// reply into the result's `toolCalls` (empty when no tools were
    /// passed, or none were called). When `onToken` is provided it is
    /// invoked once per generated token (streaming); the returned Promise
    /// always resolves once generation finishes.
    ///
    /// Passing `options.enableThinking` opts into "thinking" mode on
    /// models whose chat template supports it; any resulting reasoning
    /// span is stripped out of `text` and returned separately as
    /// `reasoning` (`options.reasoningBudgetTokens` caps how long that
    /// span may run before it's force-closed).
    #[napi]
    pub async fn generate(
        &self,
        messages: Vec<JsChatMessage>,
        options: Option<JsGenerateOptions>,
        on_token: Option<ThreadsafeFunction<JsToken, ()>>,
    ) -> Result<JsGenerateResult> {
        let (tools, core_options) = options
            .map(JsGenerateOptions::into_parts)
            .transpose()?
            .unwrap_or_default();

        let session = self.session.clone();
        let core_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(JsChatMessage::into_core)
            .collect::<Result<_>>()?;

        let reply = napi::bindgen_prelude::spawn_blocking(move || {
            session
                .generate_cached(&core_messages, tools.as_deref(), core_options, |tok| {
                    if let Some(cb) = &on_token {
                        cb.call(
                            Ok(JsToken {
                                id: tok.id,
                                text: tok.text,
                                finished: tok.finished,
                                kind: token_kind_str(tok.kind).to_string(),
                            }),
                            ThreadsafeFunctionCallMode::NonBlocking,
                        );
                    }
                    true
                })
                .map_err(|e| Error::from_reason(e.to_string()))
        })
        .await
        .map_err(|e| Error::from_reason(format!("generate task panicked: {e}")))??;

        Ok(JsGenerateResult {
            text: reply.text,
            tool_calls: reply.tool_calls.into_iter().map(JsToolCall::from).collect(),
            usage: JsUsage::from(reply.usage),
            reasoning: reply.reasoning,
        })
    }

    /// Whether this checkpoint accepts image (and video-as-frames) input.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.session.supports_images()
    }

    /// Whether this checkpoint accepts audio input.
    #[napi]
    pub fn supports_audio(&self) -> bool {
        self.session.supports_audio()
    }
}

/// The resolved value of [`MlexModel::generate`].
#[napi(object)]
pub struct JsGenerateResult {
    pub text: String,
    /// Tool calls parsed out of `text`; empty unless `options.tools` was
    /// passed and the model actually issued one or more calls.
    pub tool_calls: Vec<JsToolCall>,
    /// Token accounting for this call, mirroring the `usage` block
    /// OpenAI/Anthropic return alongside a chat completion.
    pub usage: JsUsage,
    /// Extracted reasoning/"thinking" content, if the model emitted a
    /// recognized reasoning span - present regardless of whether
    /// `options.enableThinking` was explicitly set, since some
    /// checkpoints reason unconditionally. `text` never includes this
    /// span; pass it back via `reasoningContent` on the next assistant
    /// turn to preserve it in multi-turn history.
    pub reasoning: Option<String>,
}

/// Token accounting for one [`MlexModel::generate`] call.
#[napi(object)]
pub struct JsUsage {
    /// Total input tokens for this call's fully-rendered prompt (the sum
    /// of `cachedTokens` and however many had to be freshly computed).
    pub prompt_tokens: u32,
    /// How many of `promptTokens` were served from the internal
    /// prompt-cache pool (an exact-prefix hit) rather than run through
    /// the model this call.
    pub cached_tokens: u32,
    /// Tokens generated in this call's reply.
    pub completion_tokens: u32,
}

impl From<mlex::generate::Usage> for JsUsage {
    fn from(u: mlex::generate::Usage) -> Self {
        JsUsage {
            prompt_tokens: u.prompt_tokens as u32,
            cached_tokens: u.cached_tokens as u32,
            completion_tokens: u.completion_tokens as u32,
        }
    }
}
