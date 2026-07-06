//! Reasoning ("thinking") span detection, extraction, and budget
//! enforcement.
//!
//! Several model families shipped as `mlx-community` checkpoints (Qwen3,
//! Qwen3.5, Qwen3.6, Gemma4, MiniCPM5, NemotronH, ...) support an opt-in
//! "thinking" mode - toggled via the chat template's `enable_thinking`
//! variable (see [`crate::generate::GenerateOptions::enable_thinking`]) -
//! where the model prefixes its reply with a delimited reasoning span
//! before the actual answer. Two delimiter conventions show up across
//! these templates:
//! - Qwen-style: plain-text `<think>...</think>` tags.
//! - Gemma4-style: `<|channel>thought` ... `<channel|>` tags.
//!
//! This module is deliberately architecture-agnostic: it just looks for
//! whichever marker pair shows up first in the generated text, rather
//! than hardcoding per-model-family logic.

/// `(open, close)` marker pairs, checked in order; the first `open`
/// marker found in the generated text fixes which `close` marker
/// terminates its reasoning span. Also consumed by
/// `crate::streaming::StreamClassifier` to tag live-streamed tokens.
pub(crate) const MARKER_PAIRS: &[(&str, &str)] =
    &[("<think>", "</think>"), ("<|channel>thought", "<channel|>")];

/// Split a raw generated reply into `(reasoning, answer)`.
///
/// If none of the known opening markers are present, `reasoning` is
/// `None` and `answer` is `text` unchanged. If an opening marker is
/// present but never closed (e.g. generation was cut off by `max_tokens`
/// mid-thought, or a budget force-close's tokens are still pending),
/// the entire remainder after the opening marker is treated as
/// reasoning and `answer` is whatever text preceded it.
pub fn split_reasoning(text: &str) -> (Option<String>, String) {
    for (open, close) in MARKER_PAIRS {
        if let Some(open_at) = text.find(open) {
            let before = &text[..open_at];
            let after_open = &text[open_at + open.len()..];
            return match after_open.find(close) {
                Some(close_at) => {
                    let reasoning = after_open[..close_at].trim().to_string();
                    let mut answer = String::from(before);
                    answer.push_str(after_open[close_at + close.len()..].trim_start());
                    (Some(reasoning), answer.trim().to_string())
                }
                None => (
                    Some(after_open.trim().to_string()),
                    before.trim().to_string(),
                ),
            };
        }
    }
    (None, text.to_string())
}

/// Detect whether a *rendered chat-template prompt* (i.e. the text fed to
/// the model, not its output) already opens an unclosed reasoning span at
/// its very end.
///
/// Several `enable_thinking`-style templates (Qwen3/3.5/3.6, NemotronH)
/// bake the opening marker into the generation prompt itself - e.g.
/// Qwen3.5's template ends `add_generation_prompt` with
/// `'<|im_start|>assistant\n<think>\n'` rather than letting the model
/// generate `<think>` itself. On those checkpoints the model's *generated*
/// text starts already inside the reasoning span and never contains the
/// literal open marker, so [`split_reasoning`]/`StreamClassifier` would
/// otherwise never detect it (Gemma4's template, by contrast, leaves the
/// open marker for the model to generate itself when thinking is
/// enabled - see its template's `add_generation_prompt` block - so it
/// needs no such treatment).
///
/// Returns the `(open, close)` pair whose `open` marker the prompt ends
/// with (after trimming trailing whitespace), so callers can seed
/// [`ReasoningBudget`] / `crate::streaming::StreamClassifier` as if that
/// marker had just been generated, and prepend it back before calling
/// [`split_reasoning`] on the model's actual output.
pub(crate) fn pending_marker(prompt: &str) -> Option<(&'static str, &'static str)> {
    let trimmed = prompt.trim_end();
    MARKER_PAIRS
        .iter()
        .find(|(open, _)| trimmed.ends_with(open))
        .copied()
}

/// Tracks generated text against a token budget for the *reasoning* span
/// only, so [`crate::generate::Session`]'s decode loop can force it closed
/// once the budget is exhausted - mirroring Anthropic's extended-thinking
/// `budget_tokens`: once the budget runs out mid-thought, generation is
/// cut over to the final answer rather than left to ramble indefinitely.
pub struct ReasoningBudget {
    budget: usize,
    buffer: String,
    open_close: Option<(&'static str, &'static str)>,
    closed: bool,
    tokens_since_open: usize,
}

impl ReasoningBudget {
    pub fn new(budget: usize) -> Self {
        ReasoningBudget {
            budget,
            buffer: String::new(),
            open_close: None,
            closed: false,
            tokens_since_open: 0,
        }
    }

    /// Seed the budget as if `pair.0` (the open marker) had already been
    /// observed - used when the reasoning span was opened by the *prompt*
    /// itself rather than by generated text (see
    /// [`pending_marker`]), so tokens generated from the very first one
    /// still count against the budget.
    pub(crate) fn seed_open(&mut self, pair: (&'static str, &'static str)) {
        if !self.closed && self.open_close.is_none() {
            self.open_close = Some(pair);
        }
    }

    /// Feed one newly generated token's decoded text. Returns the close
    /// marker to force-inject the first time the budget is exceeded while
    /// still inside an (unclosed) reasoning span; after that, this always
    /// returns `None` (a budget only ever fires once per generation).
    pub fn observe(&mut self, piece: &str) -> Option<&'static str> {
        if self.closed {
            return None;
        }
        self.buffer.push_str(piece);
        if self.open_close.is_none() {
            for pair in MARKER_PAIRS {
                if self.buffer.contains(pair.0) {
                    self.open_close = Some(*pair);
                    break;
                }
            }
        }
        let (_, close) = self.open_close?;
        if self.buffer.contains(close) {
            // Model closed the span itself before hitting the budget.
            self.closed = true;
            return None;
        }
        self.tokens_since_open += 1;
        if self.tokens_since_open > self.budget {
            self.closed = true;
            return Some(close);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_markers_passes_through_unchanged() {
        let (reasoning, answer) = split_reasoning("just a plain answer");
        assert_eq!(reasoning, None);
        assert_eq!(answer, "just a plain answer");
    }

    #[test]
    fn extracts_qwen_style_think_tags() {
        let (reasoning, answer) =
            split_reasoning("<think>\nlet me work this out\n</think>\n\nthe answer is 4");
        assert_eq!(reasoning.as_deref(), Some("let me work this out"));
        assert_eq!(answer, "the answer is 4");
    }

    #[test]
    fn extracts_gemma4_channel_style() {
        let (reasoning, answer) =
            split_reasoning("<|channel>thought\nhmm\n<channel|>final answer here");
        assert_eq!(reasoning.as_deref(), Some("hmm"));
        assert_eq!(answer, "final answer here");
    }

    #[test]
    fn unclosed_span_is_all_reasoning() {
        let (reasoning, answer) = split_reasoning("<think>\nstill thinking with no end in sight");
        assert_eq!(
            reasoning.as_deref(),
            Some("still thinking with no end in sight")
        );
        assert_eq!(answer, "");
    }

    #[test]
    fn budget_fires_once_after_threshold_tokens_inside_span() {
        let mut budget = ReasoningBudget::new(3);
        assert_eq!(budget.observe("<think>"), None);
        assert_eq!(budget.observe("a"), None);
        assert_eq!(budget.observe("b"), None);
        assert_eq!(budget.observe("c"), Some("</think>"));
        // Fires only once.
        assert_eq!(budget.observe("d"), None);
    }

    #[test]
    fn budget_does_not_fire_if_model_closes_span_itself() {
        let mut budget = ReasoningBudget::new(100);
        assert_eq!(budget.observe("<think>"), None);
        assert_eq!(budget.observe("quick"), None);
        assert_eq!(budget.observe("</think>"), None);
        for _ in 0..200 {
            assert_eq!(budget.observe("more text"), None);
        }
    }

    #[test]
    fn pending_marker_detects_qwen_style_generation_prompt() {
        // Qwen3.5/NemotronH-style templates bake the open marker into the
        // generation prompt itself instead of letting the model generate
        // it.
        let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        assert_eq!(pending_marker(prompt), Some(("<think>", "</think>")));
    }

    #[test]
    fn pending_marker_is_none_when_thinking_disabled_and_already_closed() {
        let prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        assert_eq!(pending_marker(prompt), None);
    }

    #[test]
    fn pending_marker_is_none_for_gemma_style_prompt_that_leaves_marker_to_model() {
        let prompt = "<|turn>model\n";
        assert_eq!(pending_marker(prompt), None);
    }

    #[test]
    fn pending_marker_is_none_for_plain_prompt() {
        assert_eq!(pending_marker("<|im_start|>assistant\n"), None);
    }

    #[test]
    fn budget_seed_open_lets_a_pending_span_count_from_the_first_token() {
        let mut budget = ReasoningBudget::new(2);
        budget.seed_open(("<think>", "</think>"));
        assert_eq!(budget.observe("a"), None);
        assert_eq!(budget.observe("b"), None);
        assert_eq!(budget.observe("c"), Some("</think>"));
    }

    #[test]
    fn budget_seed_open_is_a_noop_once_a_span_is_already_tracked() {
        let mut budget = ReasoningBudget::new(100);
        assert_eq!(budget.observe("<think>"), None);
        // A second seed (e.g. a defensive call site) must not reset the
        // in-progress span's marker pair.
        budget.seed_open(("<|channel>thought", "<channel|>"));
        assert_eq!(budget.observe("hmm"), None);
        assert_eq!(budget.observe("</think>"), None);
    }

    #[test]
    fn budget_ignores_generation_with_no_reasoning_span() {
        let mut budget = ReasoningBudget::new(1);
        for _ in 0..50 {
            assert_eq!(budget.observe("no markers here "), None);
        }
    }
}
