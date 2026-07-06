//! High-level generation loop: prompt in, tokens out.

use std::path::Path;
use std::sync::Mutex;

use serde_json::Value;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::media::audio::{preprocess_audio_bytes, preprocess_audio_bytes_raw, ProcessedAudio};
use crate::media::image::{preprocess_image_bytes, ProcessedImage};
use crate::media::video::extract_video_frames;
use crate::models::Model;
use crate::ops;
use crate::prompt_cache::PromptCachePool;
use crate::reasoning::{self, ReasoningBudget};
use crate::sampling::{Sampler, SamplingConfig};
use crate::streaming::{StreamClassifier, TokenKind};
use crate::tokenizer::ContentPart;
use crate::tokenizer::{ChatMessage, Tokenizer};
use crate::tools::{Tool, ToolCall, ToolCallFormat};

/// A loaded model + tokenizer pair, ready to generate.
///
/// Prompt caching is stateless from the caller's perspective (mirroring
/// the OpenAI/Anthropic chat APIs): [`Session::generate_cached`] takes the
/// *full* message list on every call rather than a session handle, and an
/// internal [`PromptCachePool`] transparently reuses KV state for whatever
/// prefix (if any) a previous call already computed - see
/// `crate::prompt_cache` for the pool's eviction/matching semantics.
pub struct Session {
    model: Model,
    tokenizer: Tokenizer,
    prompt_cache: Mutex<PromptCachePool>,
}

/// One step of streamed generation.
pub struct GeneratedToken {
    pub id: u32,
    pub text: String,
    pub finished: bool,
    /// Which span this token belongs to (plain text, reasoning, or a
    /// raw, not-yet-parsed tool-call span) - see [`crate::streaming`].
    /// Best-effort: a marker straddling two tokens is still detected,
    /// but only once its second half arrives.
    pub kind: TokenKind,
}

/// Token accounting for one [`Session::generate_cached`] call, mirroring
/// the `usage` block OpenAI/Anthropic return alongside a chat completion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Total input tokens for this call's fully-rendered prompt (the sum
    /// of `cached_tokens` and however many had to be freshly computed).
    pub prompt_tokens: usize,
    /// How many of `prompt_tokens` were served from the prompt-cache pool
    /// (an exact-prefix hit) rather than run through the model this call.
    pub cached_tokens: usize,
    /// Tokens generated in this call's reply.
    pub completion_tokens: usize,
}

/// Why one [`Session::generate_cached`] call stopped generating,
/// mirroring OpenAI's `finish_reason` / Anthropic's `stop_reason`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token - a natural end of
    /// turn.
    #[default]
    Stop,
    /// [`GenerateOptions::max_tokens`] was exhausted before the model
    /// finished its reply.
    Length,
    /// The reply issued one or more tool calls (see
    /// [`GenerateReply::tool_calls`]).
    ToolCalls,
    /// The caller's `on_token` callback stopped generation early by
    /// returning `false`.
    Aborted,
}

/// Classify why generation stopped, given what the decode loop produced.
/// Tool calls take precedence (the natural "end" of a tool-calling turn,
/// whether or not an eos token followed), then a trailing eos token,
/// then an explicit caller abort; anything else means the token budget
/// ran out.
fn classify_finish(
    generated: &[u32],
    eos_ids: &[u32],
    has_tool_calls: bool,
    aborted: bool,
) -> FinishReason {
    if has_tool_calls {
        FinishReason::ToolCalls
    } else if generated.last().is_some_and(|id| eos_ids.contains(id)) {
        FinishReason::Stop
    } else if aborted {
        FinishReason::Aborted
    } else {
        FinishReason::Length
    }
}

/// Result of one [`Session::generate_cached`] call.
#[derive(Debug, Clone, Default)]
pub struct GenerateReply {
    /// The final answer text, with any reasoning span (see `reasoning`)
    /// already stripped out.
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    /// Extracted reasoning/"thinking" content, if the model emitted a
    /// recognized reasoning span (`<think>...</think>` or Gemma4's
    /// `<|channel>thought...<channel|>`) - present regardless of whether
    /// `enable_thinking` was explicitly requested, since some checkpoints
    /// reason unconditionally. See [`crate::reasoning`].
    pub reasoning: Option<String>,
    /// Why generation stopped - a natural end of turn, the token budget,
    /// a tool call, or a caller-initiated abort.
    pub finish_reason: FinishReason,
}

/// Generation parameters for a single call.
#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    /// Opt into a model's "thinking" mode via its chat template's
    /// `enable_thinking` variable (Qwen3/3.5/3.6, Gemma4, MiniCPM5,
    /// NemotronH, ...; see `crate::reasoning`). `None` leaves the
    /// template's own default in place - every template above defaults
    /// reasoning to off when the variable is left undefined, so `None`
    /// and `Some(false)` behave the same for those checkpoints.
    pub enable_thinking: Option<bool>,
    /// Cap, in tokens, on how long the model may spend inside a detected
    /// reasoning span (`<think>...</think>` or Gemma4's
    /// `<|channel>thought...<channel|>`) before it is force-closed and
    /// generation moves on to the final answer - mirroring Anthropic's
    /// extended-thinking `budget_tokens`. `None` means no cap. Has no
    /// effect if the model never opens a recognized reasoning span.
    pub reasoning_budget_tokens: Option<usize>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        GenerateOptions {
            max_tokens: 256,
            sampling: SamplingConfig::default(),
            enable_thinking: None,
            reasoning_budget_tokens: None,
        }
    }
}

/// Some checkpoints stop generation on more than one token id (e.g. a
/// dedicated end-of-turn token in addition to the tokenizer's primary
/// `eos_token`). `Tokenizer::load` only registers the latter, so this
/// folds in every `eos_token_id` (scalar or list form) declared in
/// `config.json` (top-level or nested `text_config`) and
/// `generation_config.json` - without this, checkpoints whose model
/// actually prefers an alternate stop id keep generating past the
/// intended end of the turn until `max_tokens` is hit.
fn register_extra_eos_ids(model_dir: &Path, tokenizer: &mut Tokenizer) {
    let read_json = |name: &str| -> Option<Value> {
        let path = model_dir.join(name);
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    };
    let collect_ids = |v: &Value, out: &mut Vec<u32>| match v {
        Value::Number(n) => {
            if let Some(id) = n.as_u64() {
                out.push(id as u32);
            }
        }
        Value::Array(items) => {
            for item in items {
                if let Some(id) = item.as_u64() {
                    out.push(id as u32);
                }
            }
        }
        _ => {}
    };

    let mut ids = Vec::new();
    if let Some(config) = read_json("config.json") {
        if let Some(v) = config.get("eos_token_id") {
            collect_ids(v, &mut ids);
        }
        if let Some(v) = config
            .get("text_config")
            .and_then(|t| t.get("eos_token_id"))
        {
            collect_ids(v, &mut ids);
        }
    }
    if let Some(gen_config) = read_json("generation_config.json") {
        if let Some(v) = gen_config.get("eos_token_id") {
            collect_ids(v, &mut ids);
        }
    }
    for id in ids {
        tokenizer.add_eos_id(id);
    }
}

impl Session {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model = Model::load(model_dir)?;
        let mut tokenizer = Tokenizer::load(model_dir)?;
        register_extra_eos_ids(model_dir, &mut tokenizer);
        Ok(Session {
            model,
            tokenizer,
            prompt_cache: Mutex::new(PromptCachePool::with_defaults()),
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The tool-call output convention this model's chat template uses.
    pub fn tool_call_format(&self) -> crate::tools::ToolCallFormat {
        self.model.tool_call_format()
    }

    /// Whether the loaded model can accept image attachments.
    pub fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    /// Whether the loaded model can accept audio attachments.
    pub fn supports_audio(&self) -> bool {
        self.model.supports_audio()
    }

    /// Test/debug hook: fresh per-layer caches for this model.
    pub fn debug_new_caches(&self) -> Vec<crate::models::cache::LayerCache> {
        self.model.new_caches()
    }

    /// Test/debug hook: per-layer hidden state stats (NemotronH only).
    pub fn debug_nemotron_layer_stats(&self, input_ids: &Array) -> Result<Vec<(f32, f32)>> {
        self.model.debug_nemotron_layer_stats(input_ids)
    }

    /// TEMP debug hook.
    pub fn debug_image_features(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let (patch_size, max_soft_tokens, pooling_kernel_size) =
            self.model.image_processing_params().unwrap();
        let img = preprocess_image_bytes(bytes, patch_size, max_soft_tokens, pooling_kernel_size)?;
        self.model.debug_vision_forward(&img)
    }

    /// Test/debug hook: run one raw forward pass.
    pub fn debug_forward(
        &self,
        input_ids: &Array,
        caches: &mut [crate::models::cache::LayerCache],
    ) -> Result<Array> {
        self.model.forward(input_ids, caches)
    }

    /// Render `messages` through the model's chat template and tokenize.
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let prompt = self.tokenizer.apply_chat_template(messages, true)?;
        self.tokenizer.encode(&prompt)
    }

    /// Same as [`Session::encode_chat`], but also preprocesses any media
    /// attached to `messages` and expands each rendered placeholder into
    /// its model-specific span, matching the number of soft tokens the
    /// corresponding tower will actually produce:
    /// - `<|image|>` -> `boi + image_token × N + eoi`,
    /// - `<|audio|>` -> `boa + audio_token × N + eoa`,
    /// - `<|video|>` -> one `boi + image_token × N + eoi` span per
    ///   uniformly-sampled frame (video reuses the vision tower).
    ///
    /// Returns `(expanded_prompt_ids, media)`; `media` is empty (and no
    /// tower work happens) for prompts with no attachments, including on
    /// models with no multimodal support at all.
    pub fn encode_chat_with_media(
        &self,
        messages: &[ChatMessage],
    ) -> Result<(Vec<u32>, MediaInputs)> {
        self.encode_chat_with_media_tools(messages, None)
    }

    /// Same as [`Session::encode_chat_with_media`], additionally threading
    /// a `tools` list into the chat template (mirroring
    /// [`crate::tokenizer::Tokenizer::apply_chat_template_with_tools`]).
    pub fn encode_chat_with_media_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[crate::tools::Tool]>,
    ) -> Result<(Vec<u32>, MediaInputs)> {
        self.encode_chat_with_media_full(messages, tools, None)
    }

    /// Same as [`Session::encode_chat_with_media_tools`], additionally
    /// threading `enable_thinking` into the chat template (see
    /// [`GenerateOptions::enable_thinking`] /
    /// [`crate::tokenizer::Tokenizer::apply_chat_template_full`]).
    pub fn encode_chat_with_media_full(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[crate::tools::Tool]>,
        enable_thinking: Option<bool>,
    ) -> Result<(Vec<u32>, MediaInputs)> {
        let (ids, media, _pending_reasoning) =
            self.encode_chat_with_media_full_inner(messages, tools, enable_thinking)?;
        Ok((ids, media))
    }

    /// Same as [`Session::encode_chat_with_media_full`], but additionally
    /// returns whether the rendered prompt itself already opened an
    /// unclosed reasoning span (see [`reasoning::pending_marker`]) - used
    /// internally by [`Session::generate_cached`] to correctly classify/
    /// extract reasoning on checkpoints (Qwen3/3.5/3.6, NemotronH) whose
    /// template bakes the open marker into the generation prompt rather
    /// than letting the model generate it.
    fn encode_chat_with_media_full_inner(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[crate::tools::Tool]>,
        enable_thinking: Option<bool>,
    ) -> Result<(Vec<u32>, MediaInputs, Option<(&'static str, &'static str)>)> {
        // Reasoning is opt-in: several hybrid-thinking checkpoints (Qwen3/
        // 3.5/3.6, MiniCPM5) only special-case `enable_thinking` in their
        // template when it's explicitly `false` and otherwise open a
        // `<think>` span unprompted, so leaving the key entirely undefined
        // (rather than explicitly forcing it off) would silently turn
        // reasoning "on by default" for those families. Default `None` to
        // `false` here so callers get a direct answer unless they opt in
        // with `Some(true)`.
        let enable_thinking = enable_thinking.or(Some(false));
        let prompt =
            self.tokenizer
                .apply_chat_template_full(messages, true, tools, enable_thinking)?;
        let pending_reasoning = reasoning::pending_marker(&prompt);
        let base_ids = self.tokenizer.encode(&prompt)?;

        let parts: Vec<&ContentPart> = messages.iter().flat_map(|m| m.content.iter()).collect();
        let has_images = parts
            .iter()
            .any(|p| matches!(p, ContentPart::Image(_) | ContentPart::Video(_)));
        let has_audio = parts.iter().any(|p| matches!(p, ContentPart::Audio(_)));
        if !has_images && !has_audio {
            return Ok((base_ids, MediaInputs::default(), pending_reasoning));
        }

        let image_params = if has_images {
            let params = self.model.image_processing_params().ok_or_else(|| {
                Error::Model(
                    "images/videos were attached but this model has no vision support (no vision_config)".into(),
                )
            })?;
            let ids = self.model.image_token_ids().expect(
                "image_processing_params() returned Some implies image_token_ids() does too",
            );
            Some((params, ids))
        } else {
            None
        };
        let audio_ids = if has_audio {
            Some(self.model.audio_token_ids().ok_or_else(|| {
                Error::Model(
                    "audio was attached but this model has no audio support (no audio_config)"
                        .into(),
                )
            })?)
        } else {
            None
        };
        let video_token_id = self.model.video_token_id();

        // Preprocess in content-part order (which matches placeholder
        // order in the rendered prompt): standalone images, per-video
        // frame groups, audio clips - each into its own per-type queue.
        let mut image_queue: Vec<ProcessedImage> = Vec::new();
        let mut video_queue: Vec<Vec<ProcessedImage>> = Vec::new();
        let mut audio_queue: Vec<ProcessedAudio> = Vec::new();
        for part in &parts {
            match part {
                ContentPart::Image(img) => {
                    let ((patch, max_soft, pool), _) = image_params.unwrap();
                    image_queue.push(preprocess_image_bytes(&img.bytes, patch, max_soft, pool)?);
                }
                ContentPart::Video(vid) => {
                    let ((patch, max_soft, pool), _) = image_params.unwrap();
                    let frames = extract_video_frames(&vid.bytes)?;
                    let mut processed = Vec::with_capacity(frames.len());
                    for frame in &frames {
                        processed.push(preprocess_image_bytes(frame, patch, max_soft, pool)?);
                    }
                    video_queue.push(processed);
                }
                ContentPart::Audio(aud) => {
                    let processed = match self.model.audio_samples_per_token() {
                        Some(spt) => preprocess_audio_bytes_raw(&aud.bytes, spt)?,
                        None => preprocess_audio_bytes(&aud.bytes)?,
                    };
                    audio_queue.push(processed);
                }
                ContentPart::Text(_) => {}
            }
        }

        // Walk the rendered token stream, expanding each placeholder and
        // recording the media in placeholder order (the fusion pass fills
        // placeholder positions sequentially with the concatenated
        // per-modality features, so ordering must match exactly).
        let mut media = MediaInputs::default();
        let mut image_iter = image_queue.into_iter();
        let mut video_iter = video_queue.into_iter();
        let mut audio_iter = audio_queue.into_iter();
        let mut expanded = Vec::with_capacity(base_ids.len() * 2);
        for &t in &base_ids {
            if let Some(((_, _, _), (image_token_id, boi, eoi))) = image_params {
                if t == image_token_id {
                    if let Some(img) = image_iter.next() {
                        push_image_span(
                            &mut expanded,
                            img.num_soft_tokens,
                            image_token_id,
                            boi,
                            eoi,
                        );
                        media.images.push(img);
                        continue;
                    }
                } else if video_token_id == Some(t) {
                    if let Some(frames) = video_iter.next() {
                        for frame in frames {
                            push_image_span(
                                &mut expanded,
                                frame.num_soft_tokens,
                                image_token_id,
                                boi,
                                eoi,
                            );
                            media.images.push(frame);
                        }
                        continue;
                    }
                }
            }
            if let Some((audio_token_id, boa, eoa)) = audio_ids {
                if t == audio_token_id {
                    if let Some(clip) = audio_iter.next() {
                        expanded.push(boa);
                        for _ in 0..clip.num_soft_tokens() {
                            expanded.push(audio_token_id);
                        }
                        expanded.push(eoa);
                        media.audios.push(clip);
                        continue;
                    }
                }
            }
            expanded.push(t);
        }

        Ok((expanded, media, pending_reasoning))
    }

    /// Generate up to `options.max_tokens` tokens continuing `prompt_ids`,
    /// invoking `on_token` for each generated token (stop early by
    /// returning `false`). Returns the full generated id sequence.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        options: GenerateOptions,
        on_token: impl FnMut(GeneratedToken) -> bool,
    ) -> Result<Vec<u32>> {
        let mut caches = self.model.new_caches();
        self.generate_with_caches(prompt_ids, &mut caches, options, on_token)
    }

    /// Same as [`Session::generate`] but reuses (and mutates) an existing
    /// set of per-layer caches, only running the forward pass over
    /// `new_prompt_ids` (the *new* suffix, not the whole conversation so
    /// far) before decoding. This is the primitive [`Session::generate_cached`]
    /// builds its prompt-cache pool on top of.
    pub fn generate_with_caches(
        &self,
        new_prompt_ids: &[u32],
        caches: &mut [crate::models::cache::LayerCache],
        options: GenerateOptions,
        on_token: impl FnMut(GeneratedToken) -> bool,
    ) -> Result<Vec<u32>> {
        let mut sampler = Sampler::new(options.sampling);
        let prompt_arr = Array::from_slice(new_prompt_ids, &[1, new_prompt_ids.len() as i32]);
        let logits = self.model.forward(&prompt_arr, caches)?;
        let next = self.sample_last(&logits, &mut sampler)?;
        self.decode_loop(next, caches, sampler, options, None, on_token)
    }

    /// Same as [`Session::generate`], but the prefill forward pass splices
    /// `media`'s image/audio features in at their placeholder positions
    /// (fresh caches, single-shot).
    pub fn generate_media(
        &self,
        prompt_ids: &[u32],
        media: &MediaInputs,
        options: GenerateOptions,
        on_token: impl FnMut(GeneratedToken) -> bool,
    ) -> Result<Vec<u32>> {
        let mut caches = self.model.new_caches();
        self.generate_with_media(prompt_ids, media, &mut caches, options, on_token)
    }

    /// Same as [`Session::generate_with_caches`], but the prefill forward
    /// pass splices `media`'s image/audio features in at their placeholder
    /// positions (see [`Session::encode_chat_with_media`]). Pass an empty
    /// `media` for a text-only prompt - equivalent to
    /// [`Session::generate_with_caches`].
    pub fn generate_with_media(
        &self,
        new_prompt_ids: &[u32],
        media: &MediaInputs,
        caches: &mut [crate::models::cache::LayerCache],
        options: GenerateOptions,
        on_token: impl FnMut(GeneratedToken) -> bool,
    ) -> Result<Vec<u32>> {
        self.generate_with_media_inner(new_prompt_ids, media, caches, options, None, on_token)
    }

    /// Same as [`Session::generate_with_media`], but additionally accepts
    /// `pending_reasoning` - the `(open, close)` marker pair the caller
    /// already knows the prompt opened (unclosed) at its very end, per
    /// [`reasoning::pending_marker`]. Used internally by
    /// [`Session::generate_cached`] so the decode loop's
    /// [`StreamClassifier`]/[`ReasoningBudget`] treat generation as
    /// already inside that reasoning span from its very first token,
    /// matching checkpoints (Qwen3/3.5/3.6, NemotronH) whose chat template
    /// bakes the open marker into the generation prompt instead of
    /// leaving the model to generate it.
    fn generate_with_media_inner(
        &self,
        new_prompt_ids: &[u32],
        media: &MediaInputs,
        caches: &mut [crate::models::cache::LayerCache],
        options: GenerateOptions,
        pending_reasoning: Option<(&'static str, &'static str)>,
        on_token: impl FnMut(GeneratedToken) -> bool,
    ) -> Result<Vec<u32>> {
        let mut sampler = Sampler::new(options.sampling);
        let prompt_arr = Array::from_slice(new_prompt_ids, &[1, new_prompt_ids.len() as i32]);
        let logits = if media.is_empty() {
            self.model.forward(&prompt_arr, caches)?
        } else {
            self.model
                .forward_with_media(&prompt_arr, &media.images, &media.audios, caches)?
        };
        let next = self.sample_last(&logits, &mut sampler)?;
        self.decode_loop(next, caches, sampler, options, pending_reasoning, on_token)
    }

    /// Shared token-by-token decode loop used by both
    /// [`Session::generate_with_caches`] and [`Session::generate_with_media`]
    /// once the prefill forward pass has produced the first sampled token.
    ///
    /// When `options.reasoning_budget_tokens` is set, tracks generated text
    /// against a [`ReasoningBudget`] and, the moment it's exceeded inside
    /// an open reasoning span, teacher-forces that span's close marker's
    /// tokens through the model (updating `caches` exactly as if the model
    /// had generated them) before resuming normal sampling - moving
    /// generation over to the final answer instead of letting reasoning
    /// run unbounded.
    fn decode_loop(
        &self,
        mut next: u32,
        caches: &mut [crate::models::cache::LayerCache],
        mut sampler: Sampler,
        options: GenerateOptions,
        pending_reasoning: Option<(&'static str, &'static str)>,
        mut on_token: impl FnMut(GeneratedToken) -> bool,
    ) -> Result<Vec<u32>> {
        let eos_ids = self.tokenizer.eos_token_ids();
        let mut budget = options.reasoning_budget_tokens.map(ReasoningBudget::new);
        let mut classifier = StreamClassifier::new(self.tool_call_format());
        if let Some(pair @ (_, close)) = pending_reasoning {
            classifier.seed_reasoning(close);
            if let Some(b) = budget.as_mut() {
                b.seed_open(pair);
            }
        }

        let mut generated = Vec::with_capacity(options.max_tokens);
        for _ in 0..options.max_tokens {
            let finished = eos_ids.contains(&next);
            let text = self.tokenizer.decode_piece(next)?;
            // Marker detection (reasoning/tool-call spans) runs against a
            // *raw* decode that keeps special tokens as literal text -
            // some checkpoints (e.g. Gemma4) implement these markers as
            // special vocabulary entries that `text` (skip_special_tokens
            // decode, used for display) silently strips - see
            // `Tokenizer::decode_raw`.
            let raw_text = self.tokenizer.decode_piece_raw(next)?;
            generated.push(next);
            let kind = classifier.classify(&raw_text);
            let keep_going = on_token(GeneratedToken {
                id: next,
                text: stream_display_text(&classifier, text.clone()),
                finished,
                kind,
            });
            if finished || !keep_going {
                break;
            }

            let forced_close = budget.as_mut().and_then(|b| b.observe(&raw_text));
            if let Some(close_marker) = forced_close {
                let close_ids = self.tokenizer.encode(close_marker).unwrap_or_default();
                if close_ids.is_empty() {
                    let next_arr = Array::from_slice(&[next], &[1, 1]);
                    let logits = self.model.forward(&next_arr, caches)?;
                    next = self.sample_last(&logits, &mut sampler)?;
                    continue;
                }
                let close_arr = Array::from_slice(&close_ids, &[1, close_ids.len() as i32]);
                let logits = self.model.forward(&close_arr, caches)?;
                let mut stopped = false;
                for &id in &close_ids {
                    generated.push(id);
                    let piece = self.tokenizer.decode_piece(id)?;
                    let raw_piece = self.tokenizer.decode_piece_raw(id)?;
                    let kind = classifier.classify(&raw_piece);
                    if !on_token(GeneratedToken {
                        id,
                        text: stream_display_text(&classifier, piece),
                        finished: false,
                        kind,
                    }) {
                        stopped = true;
                        break;
                    }
                }
                if stopped {
                    break;
                }
                next = self.sample_last(&logits, &mut sampler)?;
                continue;
            }

            let next_arr = Array::from_slice(&[next], &[1, 1]);
            let logits = self.model.forward(&next_arr, caches)?;
            next = self.sample_last(&logits, &mut sampler)?;
        }

        Ok(generated)
    }

    pub(crate) fn new_caches(&self) -> Vec<crate::models::cache::LayerCache> {
        self.model.new_caches()
    }

    /// Stateless, cache-aware chat completion: render + encode the *full*
    /// `messages` transcript (mirroring how OpenAI/Anthropic's APIs take
    /// the whole conversation on every call, not a delta), look up the
    /// longest cached prefix of it in this session's [`PromptCachePool`],
    /// run only the uncached suffix (and any not-yet-fed media) through
    /// the model, then store the extended prefix back into the pool.
    ///
    /// Two independent calls that happen to share a prefix (the common
    /// case: the next turn of the same conversation, but also just two
    /// unrelated calls sharing a system prompt) both benefit - there is no
    /// caller-held session handle, so nothing needs to be reset when
    /// switching to an unrelated conversation; it simply misses the pool
    /// and starts cold, exactly like a fresh prompt would.
    pub fn generate_cached(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: GenerateOptions,
        mut on_token: impl FnMut(GeneratedToken) -> bool,
    ) -> Result<GenerateReply> {
        let (full_ids, media, pending_reasoning) =
            self.encode_chat_with_media_full_inner(messages, tools, options.enable_thinking)?;

        let (mut caches, fed_len, fed_images, fed_audios) = {
            let mut pool = self.prompt_cache.lock().unwrap();
            match pool.find_longest_prefix(&full_ids) {
                Some((entry, shared)) => (entry.caches, shared, entry.fed_images, entry.fed_audios),
                None => (self.new_caches(), 0, 0, 0),
            }
        };

        let new_suffix = &full_ids[fed_len..];
        let new_media = MediaInputs {
            images: media.images[fed_images.min(media.images.len())..].to_vec(),
            audios: media.audios[fed_audios.min(media.audios.len())..].to_vec(),
        };

        let mut generated_ids = Vec::new();
        let mut aborted = false;
        let ids = self.generate_with_media_inner(
            new_suffix,
            &new_media,
            &mut caches,
            options,
            pending_reasoning,
            |tok| {
                generated_ids.push(tok.id);
                let keep_going = on_token(tok);
                if !keep_going {
                    aborted = true;
                }
                keep_going
            },
        )?;
        debug_assert_eq!(ids, generated_ids);

        let usage = Usage {
            prompt_tokens: full_ids.len(),
            cached_tokens: fed_len,
            completion_tokens: generated_ids.len(),
        };

        let mut cached_ids = full_ids;
        cached_ids.extend_from_slice(&generated_ids);
        self.prompt_cache.lock().unwrap().insert_or_update(
            cached_ids,
            caches,
            media.images.len(),
            media.audios.len(),
            false,
        );

        // Decode without stripping special tokens (see
        // `Tokenizer::decode_raw`) so reasoning/tool-call markers survive
        // on checkpoints that implement them as special vocabulary
        // entries - except the eos token itself, which carries no
        // content and would otherwise leak its literal spelling (e.g.
        // `<end_of_turn>`) into the reply.
        let eos_ids = self.tokenizer.eos_token_ids();
        let content_ids: Vec<u32> = generated_ids
            .iter()
            .copied()
            .filter(|id| !eos_ids.contains(id))
            .collect();
        let raw_text = self.tokenizer.decode_raw(&content_ids)?;
        // If the prompt itself already opened a reasoning span (see
        // `pending_reasoning` above), the model's generated text never
        // contains the literal open marker - splice it back on so
        // `split_reasoning` still finds and extracts the span.
        let (reasoning, text) = match pending_reasoning {
            Some((open, _)) => reasoning::split_reasoning(&format!("{open}{raw_text}")),
            None => reasoning::split_reasoning(&raw_text),
        };
        let format = self.tool_call_format();
        let (text, calls) = if matches!(format, ToolCallFormat::None) {
            (text, Vec::new())
        } else {
            let calls = crate::tools::parse_tool_calls(&text, format);
            // Keep `text` and `tool_calls` separate (OpenAI/Anthropic
            // style) rather than leaving raw call syntax in the reply.
            (crate::tools::strip_tool_calls(&text, format), calls)
        };
        let finish_reason =
            classify_finish(&generated_ids, eos_ids, !calls.is_empty(), aborted);

        Ok(GenerateReply {
            text,
            tool_calls: calls,
            usage,
            reasoning,
            finish_reason,
        })
    }

    fn sample_last(&self, logits: &Array, sampler: &mut Sampler) -> Result<u32> {
        let shape = logits.shape();
        let seq_len = shape[1];
        let last = ops::slice(logits, &[0, seq_len - 1, 0], &[shape[0], seq_len, shape[2]])?;
        let last = ops::reshape(&last, &[shape[2]])?;
        sampler.sample(&last)
    }
}

/// Preprocessed media accompanying one encoded prompt, in placeholder
/// order (video frames appear as ordinary `images` entries, one per
/// sampled frame). Produced by [`Session::encode_chat_with_media`] and
/// consumed by [`Session::generate_with_media`].
#[derive(Debug, Clone, Default)]
pub struct MediaInputs {
    pub images: Vec<ProcessedImage>,
    pub audios: Vec<ProcessedAudio>,
}

impl MediaInputs {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audios.is_empty()
    }
}

/// Display text for one streamed token, with marker remnants suppressed.
///
/// When [`StreamClassifier::classify`] just completed a reasoning or
/// tool-call marker on this token, whatever display text the token
/// carries is (part of) the marker's own spelling, not content - e.g.
/// Qwen's `<think>`/`</think>` (non-special tokens, so their display
/// decode keeps the literal text) or the trailing `thought` of Gemma4's
/// `<|channel>thought` open marker (the `<|channel>` special token
/// display-decodes empty, but `thought` is an ordinary word token).
/// Streaming consumers should see clean span content - mirroring how
/// OpenAI/Anthropic's typed deltas never include the wire markers - so
/// suppress the remnant. The containment check keeps genuine content
/// safe: it only ever applies to the single marker-completing token.
fn stream_display_text(classifier: &StreamClassifier, text: String) -> String {
    match classifier.last_marker() {
        Some(marker) => {
            let trimmed = text.trim();
            if !trimmed.is_empty() && marker.contains(trimmed) {
                String::new()
            } else {
                text
            }
        }
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: &[u32] = &[2, 106];

    #[test]
    fn finish_stop_on_trailing_eos() {
        assert_eq!(
            classify_finish(&[5, 9, 2], EOS, false, false),
            FinishReason::Stop
        );
    }

    #[test]
    fn finish_tool_calls_takes_precedence_over_eos() {
        assert_eq!(
            classify_finish(&[5, 9, 2], EOS, true, false),
            FinishReason::ToolCalls
        );
    }

    #[test]
    fn finish_length_when_no_eos_and_not_aborted() {
        assert_eq!(
            classify_finish(&[5, 9, 7], EOS, false, false),
            FinishReason::Length
        );
    }

    #[test]
    fn finish_aborted_when_callback_stopped_early() {
        assert_eq!(
            classify_finish(&[5, 9, 7], EOS, false, true),
            FinishReason::Aborted
        );
    }

    #[test]
    fn finish_empty_generation_without_abort_is_length() {
        assert_eq!(classify_finish(&[], EOS, false, false), FinishReason::Length);
    }
}

/// Append one image's `boi + image_token × num_soft_tokens + eoi` span.
fn push_image_span(
    out: &mut Vec<u32>,
    num_soft_tokens: i32,
    image_token_id: u32,
    boi: u32,
    eoi: u32,
) {
    out.push(boi);
    for _ in 0..num_soft_tokens {
        out.push(image_token_id);
    }
    out.push(eoi);
}
