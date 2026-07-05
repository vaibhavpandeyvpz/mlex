//! Tokenizer + chat-template handling, matching the Hugging Face
//! `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` trio
//! shipped alongside every mlx-community checkpoint.

use std::path::Path;

use minijinja::value::Value as JinjaValue;
use minijinja::Environment;
use serde_json::Value;
use tokenizers::Tokenizer as HfTokenizer;

use crate::error::{Error, Result};
use crate::tools::{Tool, ToolCall};

/// One part of a (possibly multi-modal) chat message's content.
#[derive(Debug, Clone)]
pub enum ContentPart {
    Text(String),
    Image(ImageContent),
    Audio(AudioContent),
    Video(VideoContent),
}

/// A single image attachment (raw encoded bytes - JPEG/PNG/...; decoded and
/// preprocessed later by `crate::media::image`).
#[derive(Debug, Clone)]
pub struct ImageContent {
    pub bytes: Vec<u8>,
}

/// A single audio attachment (raw encoded bytes - WAV/MP3/...; decoded and
/// preprocessed later by `crate::media::audio`).
#[derive(Debug, Clone)]
pub struct AudioContent {
    pub bytes: Vec<u8>,
}

/// A single video attachment (raw encoded bytes - MP4/WebM/...; frames
/// extracted later by `crate::media::video` and fed through the image
/// path).
#[derive(Debug, Clone)]
pub struct VideoContent {
    pub bytes: Vec<u8>,
}

/// One turn of a chat conversation.
///
/// `content` is a list of parts (text and/or media) rather than a plain
/// string, so a single turn can interleave free text with images (and,
/// eventually, audio/video). The common text-only case stays ergonomic via
/// [`ChatMessage::user`]/[`ChatMessage::system`]/[`ChatMessage::assistant`]
/// (each produces a single `ContentPart::Text`) and the [`ChatMessage::text`]
/// accessor (concatenates every `Text` part, ignoring media).
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    pub role: String,
    pub content: Vec<ContentPart>,
    /// Set on assistant turns that contain tool calls (rendered into the
    /// template's `message.tool_calls`).
    pub tool_calls: Vec<ToolCall>,
    /// Set on `role: "tool"` turns: which call this result answers.
    pub tool_call_id: Option<String>,
    /// Set on assistant turns whose reasoning/"thinking" span (see
    /// `crate::reasoning`) should round-trip back into multi-turn history
    /// (rendered into the template's `message.reasoning_content`, matched
    /// by Gemma4/Qwen-style templates that special-case it).
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: vec![ContentPart::Text(content.into())],
            ..Default::default()
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "system".into(),
            content: vec![ContentPart::Text(content.into())],
            ..Default::default()
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "assistant".into(),
            content: vec![ContentPart::Text(content.into())],
            ..Default::default()
        }
    }

    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        ChatMessage {
            role: "assistant".into(),
            content: vec![ContentPart::Text(content.into())],
            tool_calls,
            ..Default::default()
        }
    }

    /// An assistant turn carrying its reasoning/"thinking" content
    /// alongside the final answer, so it round-trips into the next turn's
    /// history exactly as [`Session::generate_cached`] split it out (see
    /// `crate::reasoning::split_reasoning`).
    pub fn assistant_with_reasoning(
        content: impl Into<String>,
        reasoning_content: impl Into<String>,
    ) -> Self {
        ChatMessage {
            role: "assistant".into(),
            content: vec![ContentPart::Text(content.into())],
            reasoning_content: Some(reasoning_content.into()),
            ..Default::default()
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: "tool".into(),
            content: vec![ContentPart::Text(content.into())],
            tool_call_id: Some(call_id.into()),
            ..Default::default()
        }
    }

    /// A user turn pairing free text with one image (raw encoded bytes,
    /// e.g. a `.jpg`/`.png` file's contents). The chat template renders
    /// this as a single `<|image|>`-style placeholder that
    /// `Session::encode_chat_with_media` later expands into the model's
    /// `boi + image_token × num_soft_tokens + eoi` span.
    pub fn user_with_image(text: impl Into<String>, image_bytes: impl Into<Vec<u8>>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: vec![
                ContentPart::Text(text.into()),
                ContentPart::Image(ImageContent {
                    bytes: image_bytes.into(),
                }),
            ],
            ..Default::default()
        }
    }

    /// A user turn pairing free text with one audio clip (raw encoded
    /// bytes, e.g. a `.wav`/`.mp3` file's contents). The chat template
    /// renders this as a single `<|audio|>`-style placeholder that
    /// `Session::encode_chat_with_media` later expands into the model's
    /// `boa + audio_token × num_soft_tokens + eoa` span.
    pub fn user_with_audio(text: impl Into<String>, audio_bytes: impl Into<Vec<u8>>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: vec![
                ContentPart::Text(text.into()),
                ContentPart::Audio(AudioContent {
                    bytes: audio_bytes.into(),
                }),
            ],
            ..Default::default()
        }
    }

    /// A user turn pairing free text with one video clip (raw encoded
    /// bytes, e.g. an `.mp4` file's contents). The chat template renders
    /// this as a single `<|video|>`-style placeholder that
    /// `Session::encode_chat_with_media` later expands into one
    /// `boi + image_token × N + eoi` span per sampled frame.
    pub fn user_with_video(text: impl Into<String>, video_bytes: impl Into<Vec<u8>>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: vec![
                ContentPart::Text(text.into()),
                ContentPart::Video(VideoContent {
                    bytes: video_bytes.into(),
                }),
            ],
            ..Default::default()
        }
    }

    /// Concatenation of every `Text` part (media parts contribute nothing);
    /// the common accessor for the plain-text case.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .concat()
    }

    /// Every image attached to this message, in order.
    pub fn images(&self) -> impl Iterator<Item = &ImageContent> {
        self.content.iter().filter_map(|p| match p {
            ContentPart::Image(i) => Some(i),
            _ => None,
        })
    }

    /// Every audio clip attached to this message, in order.
    pub fn audios(&self) -> impl Iterator<Item = &AudioContent> {
        self.content.iter().filter_map(|p| match p {
            ContentPart::Audio(a) => Some(a),
            _ => None,
        })
    }

    /// Every video clip attached to this message, in order.
    pub fn videos(&self) -> impl Iterator<Item = &VideoContent> {
        self.content.iter().filter_map(|p| match p {
            ContentPart::Video(v) => Some(v),
            _ => None,
        })
    }

    /// True if this message carries any non-text content part.
    pub fn has_media(&self) -> bool {
        self.content
            .iter()
            .any(|p| !matches!(p, ContentPart::Text(_)))
    }
}

/// Render `content` the way the Jinja context expects: a plain string when
/// it's a single text part (matches every existing text-only template's
/// `message['content'] is string` branch byte-for-byte), otherwise a
/// sequence of `{"type": ..., ...}` part objects (matches the
/// `message['content'] is sequence` branch multimodal templates use).
fn content_to_json(content: &[ContentPart]) -> Value {
    if let [ContentPart::Text(t)] = content {
        return Value::String(t.clone());
    }
    Value::Array(
        content
            .iter()
            .map(|p| match p {
                ContentPart::Text(t) => serde_json::json!({"type": "text", "text": t}),
                ContentPart::Image(_) => serde_json::json!({"type": "image"}),
                ContentPart::Audio(_) => serde_json::json!({"type": "audio"}),
                ContentPart::Video(_) => serde_json::json!({"type": "video"}),
            })
            .collect(),
    )
}

pub struct Tokenizer {
    inner: HfTokenizer,
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    eos_token_ids: Vec<u32>,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("failed to load tokenizer.json: {e}")))?;

        let tokenizer_config: Value = model_dir
            .join("tokenizer_config.json")
            .exists()
            .then(|| std::fs::read_to_string(model_dir.join("tokenizer_config.json")))
            .transpose()?
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| Error::Tokenizer(format!("bad tokenizer_config.json: {e}")))?
            .unwrap_or(Value::Null);

        let chat_template = model_dir
            .join("chat_template.jinja")
            .exists()
            .then(|| std::fs::read_to_string(model_dir.join("chat_template.jinja")))
            .transpose()?
            .or_else(|| {
                tokenizer_config
                    .get("chat_template")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            });

        let bos_token = extract_special_token(&tokenizer_config, "bos_token");
        let eos_token = extract_special_token(&tokenizer_config, "eos_token");

        let mut eos_token_ids = Vec::new();
        if let Some(t) = &eos_token {
            if let Some(id) = inner.token_to_id(t) {
                eos_token_ids.push(id);
            }
        }
        // generation_config.json / config.json commonly list eos_token_id
        // as a scalar or list; callers can extend this via `add_eos_id`.

        Ok(Tokenizer {
            inner,
            chat_template,
            bos_token,
            eos_token,
            eos_token_ids,
        })
    }

    pub fn add_eos_id(&mut self, id: u32) {
        if !self.eos_token_ids.contains(&id) {
            self.eos_token_ids.push(id);
        }
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Decode a single new token, useful for incremental streaming
    /// (does not attempt to merge partial multi-byte UTF-8 sequences).
    pub fn decode_piece(&self, id: u32) -> Result<String> {
        self.decode(&[id])
    }

    /// Same as [`Tokenizer::decode`], but keeps special tokens as literal
    /// text instead of stripping them.
    ///
    /// Some chat templates render reasoning/tool-call delimiters (e.g.
    /// Gemma4's `<|channel>`/`<channel|>`, `<|tool_call>`/`<tool_call|>`)
    /// as *special* vocabulary entries (`added_tokens_decoder[...].special
    /// == true`), which [`Tokenizer::decode`]'s `skip_special_tokens`
    /// unconditionally strips - silently deleting the very markers
    /// `crate::reasoning`/`crate::streaming`/`crate::tools` need to see to
    /// detect a reasoning or tool-call span at all (other families, e.g.
    /// Qwen's `<think>`/`<tool_call>`, ship the same markers as
    /// *ordinary*, non-special vocabulary entries instead, so this
    /// distinction is purely a per-checkpoint tokenizer detail, not an
    /// architecture one). This method is what those modules decode
    /// through internally so marker detection works uniformly across
    /// both conventions.
    pub fn decode_raw(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Same as [`Tokenizer::decode_raw`], for a single token.
    pub fn decode_piece_raw(&self, id: u32) -> Result<String> {
        self.decode_raw(&[id])
    }

    /// Render the chat template for `messages`, returning the prompt text
    /// to feed into [`Tokenizer::encode`].
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.apply_chat_template_with_tools(messages, add_generation_prompt, None)
    }

    /// Same as [`Tokenizer::apply_chat_template`], additionally threading a
    /// `tools` list into the jinja context (rendered as the OpenAI-style
    /// `[{"type": "function", "function": {...}}, ...]` shape every
    /// downloaded checkpoint's template expects).
    pub fn apply_chat_template_with_tools(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: Option<&[Tool]>,
    ) -> Result<String> {
        self.apply_chat_template_full(messages, add_generation_prompt, tools, None)
    }

    /// Same as [`Tokenizer::apply_chat_template_with_tools`], additionally
    /// threading `enable_thinking` into the jinja context - the variable
    /// Qwen3/3.5/3.6, Gemma4, MiniCPM5, and NemotronH templates check to
    /// decide whether to open a reasoning span. `None` omits the key
    /// entirely, leaving it `Undefined` (the template's own default).
    /// Note that several of these templates only special-case
    /// `enable_thinking` when it's explicitly `false` (forcing a
    /// pre-closed `<think></think>`) and otherwise open a reasoning span
    /// unprompted - i.e. `Undefined` does *not* reliably mean "off" here.
    /// Callers that want reasoning disabled by default should pass an
    /// explicit `Some(false)` rather than relying on `None`; see
    /// [`crate::generate::Session::encode_chat_with_media_full`], which
    /// does exactly that.
    pub fn apply_chat_template_full(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: Option<&[Tool]>,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let template_src = self.chat_template.as_deref().ok_or_else(|| {
            Error::Template("model has no chat_template.jinja / chat_template config".into())
        })?;
        // HF chat templates sometimes wrap the assistant-content span in
        // Jinja2's `{% generation %}...{% endgeneration %}` extension tag
        // (used to build training loss masks) - not part of standard Jinja
        // and unsupported by minijinja. It's a pure no-op for rendering
        // purposes, so strip it before parsing.
        let template_src = strip_generation_tags(template_src);

        let mut env = Environment::new();
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        // HF chat templates lean on Python string/list methods (startswith,
        // strip, join, ...); minijinja-contrib's pycompat shim maps these
        // onto minijinja's native `unknown_method_callback`.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<(), minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_function("strftime_now", |fmt: String| -> String {
            let _ = fmt;
            String::new()
        });
        // Some HF templates (e.g. MiniCPM5) call `tojson(ensure_ascii=False)`
        // - a Python `json.dumps` kwarg minijinja's builtin `tojson` filter
        // doesn't accept. Override it with one that takes (and ignores) the
        // kwarg; `serde_json` never escapes non-ASCII to `\uXXXX` either way,
        // so behavior matches `ensure_ascii=False` regardless.
        env.add_filter(
            "tojson",
            |value: JinjaValue,
             _kwargs: minijinja::value::Kwargs|
             -> std::result::Result<String, minijinja::Error> {
                serde_json::to_string(&value).map_err(|e| {
                    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
                })
            },
        );
        env.add_template("chat", &template_src)
            .map_err(|e| Error::Template(e.to_string()))?;
        let tmpl = env.get_template("chat").unwrap();

        let messages_value: Vec<JinjaValue> = messages
            .iter()
            .map(|m| {
                let tool_calls: Vec<Value> = m
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        })
                    })
                    .collect();
                JinjaValue::from_serialize(serde_json::json!({
                    "role": m.role,
                    "content": content_to_json(&m.content),
                    "tool_calls": tool_calls,
                    "tool_call_id": m.tool_call_id,
                    "reasoning_content": m.reasoning_content,
                }))
            })
            .collect();

        // Templates commonly do `{{ tools|length }}` *without* an `is
        // defined`/truthy guard first, relying on `tools` being wholly
        // absent (Undefined, which `length` treats as empty) rather than
        // explicitly `None` (which `length` raises on) when no tools are
        // passed - so omit the key entirely rather than passing `None`.
        let base_context = minijinja::context! {
            messages => messages_value,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone().unwrap_or_default(),
        };
        let context_with_tools = match tools {
            Some(ts) => {
                let tools_value: Vec<JinjaValue> = ts
                    .iter()
                    .map(|t| JinjaValue::from_serialize(serde_json::to_value(t).unwrap()))
                    .collect();
                minijinja::context! { tools => tools_value, ..base_context }
            }
            None => base_context,
        };
        // Same Undefined-vs-None reasoning as `tools` above: templates
        // guard with `enable_thinking is defined`, so omit the key when
        // the caller didn't ask for a specific value.
        let full_context = match enable_thinking {
            Some(v) => minijinja::context! { enable_thinking => v, ..context_with_tools },
            None => context_with_tools,
        };

        let rendered = tmpl
            .render(full_context)
            .map_err(|e| Error::Template(e.to_string()))?;
        Ok(rendered)
    }
}

/// Strip HF's non-standard `{% generation %}` / `{% endgeneration %}`
/// markers (with any leading `-`/trailing `-` whitespace-control variants).
fn strip_generation_tags(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    loop {
        let Some(start) = rest.find("{%") else {
            out.push_str(rest);
            break;
        };
        let Some(rel_end) = rest[start..].find("%}") else {
            out.push_str(rest);
            break;
        };
        let end = start + rel_end + 2;
        let tag_body = rest[start + 2..end - 2].trim().trim_matches('-').trim();
        if tag_body == "generation" || tag_body == "endgeneration" {
            out.push_str(&rest[..start]);
        } else {
            out.push_str(&rest[..end]);
        }
        rest = &rest[end..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_generation_tags() {
        let src = "a{% generation %}b{% endgeneration %}c{% if x %}d{% endif %}";
        let stripped = strip_generation_tags(src);
        assert_eq!(stripped, "abc{% if x %}d{% endif %}");
    }

    #[test]
    fn strips_generation_tags_with_whitespace_control() {
        let src = "a{%- generation -%}b{%- endgeneration -%}c";
        let stripped = strip_generation_tags(src);
        assert_eq!(stripped, "abc");
    }

    fn dummy_tokenizer_with_template(template: &str) -> Tokenizer {
        // Minimal `HfTokenizer` covering just the bytes needed for tests
        // that only exercise `apply_chat_template` (never `encode`).
        let inner = HfTokenizer::from_bytes(
            br#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#,
        )
        .unwrap();
        Tokenizer {
            inner,
            chat_template: Some(template.to_string()),
            bos_token: Some("<bos>".into()),
            eos_token: Some("<eos>".into()),
            eos_token_ids: Vec::new(),
        }
    }

    #[test]
    fn renders_simple_template() {
        let tok = dummy_tokenizer_with_template(
            "{%- for m in messages %}{{ m.role }}: {{ m.content }}\n{%- endfor %}",
        );
        let messages = vec![ChatMessage::user("hi"), ChatMessage::assistant("hello")];
        let rendered = tok.apply_chat_template(&messages, false).unwrap();
        assert_eq!(rendered, "user: hiassistant: hello");
    }

    #[test]
    fn renders_tools_into_context() {
        let tok = dummy_tokenizer_with_template(
            "{%- if tools %}TOOLS:{% for t in tools %}{{ t.function.name }},{% endfor %}{%- endif %}",
        );
        let tools = vec![Tool::new("get_weather", "desc", serde_json::json!({}))];
        let rendered = tok
            .apply_chat_template_with_tools(&[ChatMessage::user("hi")], false, Some(&tools))
            .unwrap();
        assert_eq!(rendered, "TOOLS:get_weather,");
    }

    #[test]
    fn missing_template_errors() {
        let mut tok = dummy_tokenizer_with_template("x");
        tok.chat_template = None;
        assert!(tok
            .apply_chat_template(&[ChatMessage::user("hi")], false)
            .is_err());
    }
}

fn extract_special_token(cfg: &Value, key: &str) -> Option<String> {
    match cfg.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}
