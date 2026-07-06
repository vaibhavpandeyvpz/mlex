//! Live classification of streamed tokens into reasoning/text/tool-call
//! spans, mirroring the typed streaming deltas OpenAI (`reasoning`,
//! `content`, `function_call_arguments`) and Anthropic (`thinking_delta`,
//! `text_delta`, `input_json_delta`) emit - so a UI can render "thinking",
//! the final answer, and in-progress tool-call syntax in separate regions
//! *while* generation is still running, instead of only being able to
//! split them apart from [`crate::generate::GenerateReply`] once
//! generation has finished.
//!
//! Detection is marker-based and format-aware (reasoning markers are the
//! same architecture-agnostic set `crate::reasoning` uses; tool-call
//! markers depend on the model's [`crate::tools::ToolCallFormat`]), and
//! - like [`crate::reasoning::ReasoningBudget`] - necessarily a best
//! effort at token granularity: a marker split awkwardly across two
//! tokens is still detected (classification looks at a trailing text
//! window, not a single token in isolation), but the token containing
//! the boundary is attributed to whichever span it completes.

use crate::reasoning::MARKER_PAIRS;
use crate::tools::ToolCallFormat;

/// Which part of a reply a streamed token belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// Ordinary reply text.
    Text,
    /// Inside a reasoning/"thinking" span (`crate::reasoning`).
    Reasoning,
    /// Inside a tool-call span, in the model's raw, not-yet-parsed
    /// output convention (Hermes JSON or Gemma's native key/value
    /// syntax - see `crate::tools`).
    ToolCall,
}

/// Longest marker byte length across every (reasoning + tool-call)
/// open/close pair - the classifier only needs to retain this many
/// trailing bytes to reliably detect a marker straddling two tokens.
const TAIL_CAP: usize = 32;

/// Stateful, incremental classifier: feed it each newly generated
/// token's decoded text (in order) via [`StreamClassifier::classify`].
pub struct StreamClassifier {
    tool_open: &'static str,
    tool_close: &'static str,
    state: TokenKind,
    reasoning_close: &'static str,
    tail: String,
    last_marker: Option<&'static str>,
}

impl StreamClassifier {
    pub fn new(tool_format: ToolCallFormat) -> Self {
        let (tool_open, tool_close) = match tool_format {
            ToolCallFormat::Hermes => ("<tool_call>", "</tool_call>"),
            // The parser's full open marker is `<|tool_call>call:`, but the
            // classifier keys on the special token alone: `call` and `:`
            // arrive as separate ordinary tokens after it, and waiting for
            // them would misclassify those tokens (and everything until the
            // trailing window catches up) as text.
            ToolCallFormat::Gemma => ("<|tool_call>", "<tool_call|>"),
            ToolCallFormat::None => ("", ""),
        };
        StreamClassifier {
            tool_open,
            tool_close,
            state: TokenKind::Text,
            reasoning_close: "",
            tail: String::new(),
            last_marker: None,
        }
    }

    /// The open/close marker string the most recent [`StreamClassifier::classify`]
    /// call completed, if any. Lets the caller suppress a marker's own
    /// (display-decoded) remnant from user-facing streamed text - e.g.
    /// Gemma4's `<|channel>thought` open marker spans a special token
    /// (display-empty) plus the ordinary word `thought`, which would
    /// otherwise leak into the reasoning stream.
    pub fn last_marker(&self) -> Option<&'static str> {
        self.last_marker
    }

    /// Seed the classifier as if `open` (a reasoning marker) had already
    /// been observed and its text already emitted - used when the model's
    /// chat-template prompt itself opens the reasoning span (see
    /// `crate::reasoning::pending_marker`), so the model's *generated*
    /// text - which never contains the literal open marker on those
    /// checkpoints - is still classified as `Reasoning` from its first
    /// token.
    pub(crate) fn seed_reasoning(&mut self, close: &'static str) {
        self.state = TokenKind::Reasoning;
        self.reasoning_close = close;
        self.tail.clear();
    }

    /// Classify one newly generated token's decoded text, returning
    /// which span it belongs to and updating internal state for the
    /// next call.
    pub fn classify(&mut self, piece: &str) -> TokenKind {
        self.last_marker = None;
        self.tail.push_str(piece);
        if self.tail.len() > TAIL_CAP {
            let mut cut = self.tail.len() - TAIL_CAP;
            while !self.tail.is_char_boundary(cut) {
                cut += 1;
            }
            self.tail.drain(..cut);
        }

        match self.state {
            TokenKind::Text => {
                if let Some((open, close)) = MARKER_PAIRS
                    .iter()
                    .find(|(open, _)| self.tail.contains(open))
                {
                    self.state = TokenKind::Reasoning;
                    self.reasoning_close = close;
                    self.last_marker = Some(open);
                    self.tail.clear();
                    return TokenKind::Reasoning;
                }
                if !self.tool_open.is_empty() && self.tail.contains(self.tool_open) {
                    self.state = TokenKind::ToolCall;
                    self.last_marker = Some(self.tool_open);
                    self.tail.clear();
                    return TokenKind::ToolCall;
                }
                TokenKind::Text
            }
            TokenKind::Reasoning => {
                if self.tail.contains(self.reasoning_close) {
                    self.state = TokenKind::Text;
                    self.last_marker = Some(self.reasoning_close);
                    self.tail.clear();
                }
                TokenKind::Reasoning
            }
            TokenKind::ToolCall => {
                if self.tail.contains(self.tool_close) {
                    self.state = TokenKind::Text;
                    self.last_marker = Some(self.tool_close);
                    self.tail.clear();
                }
                TokenKind::ToolCall
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify_all(format: ToolCallFormat, pieces: &[&str]) -> Vec<TokenKind> {
        let mut c = StreamClassifier::new(format);
        pieces.iter().map(|p| c.classify(p)).collect()
    }

    #[test]
    fn plain_text_stays_text() {
        let kinds = classify_all(ToolCallFormat::None, &["hello", " ", "world"]);
        assert!(kinds.iter().all(|k| *k == TokenKind::Text));
    }

    #[test]
    fn reasoning_span_is_tagged_and_reverts_to_text() {
        let kinds = classify_all(
            ToolCallFormat::None,
            &["<think>", "hmm", "</think>", "the answer", " is 4"],
        );
        assert_eq!(
            kinds,
            vec![
                TokenKind::Reasoning,
                TokenKind::Reasoning,
                TokenKind::Reasoning,
                TokenKind::Text,
                TokenKind::Text,
            ]
        );
    }

    #[test]
    fn marker_split_across_two_tokens_is_still_detected() {
        // "<think>" arrives as "<thi" + "nk>" - neither half alone
        // contains the full marker, but the classifier's trailing
        // window sees it once both pieces have been observed.
        let kinds = classify_all(
            ToolCallFormat::None,
            &["<thi", "nk>", "reasoning here", "</think>", "answer"],
        );
        assert_eq!(kinds[1], TokenKind::Reasoning);
        assert_eq!(kinds[2], TokenKind::Reasoning);
        assert_eq!(kinds[3], TokenKind::Reasoning);
        assert_eq!(kinds[4], TokenKind::Text);
    }

    #[test]
    fn seed_reasoning_tags_generation_from_the_first_token() {
        // Mirrors checkpoints (Qwen3.5, NemotronH) whose chat template
        // bakes the open marker into the generation prompt, so it's never
        // present in the model's own generated text.
        let mut c = StreamClassifier::new(ToolCallFormat::None);
        c.seed_reasoning("</think>");
        let kinds: Vec<_> = ["let", " me", " think", "</think>", "answer"]
            .iter()
            .map(|p| c.classify(p))
            .collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Reasoning,
                TokenKind::Reasoning,
                TokenKind::Reasoning,
                TokenKind::Reasoning,
                TokenKind::Text,
            ]
        );
    }

    #[test]
    fn hermes_tool_call_span_is_tagged() {
        let kinds = classify_all(
            ToolCallFormat::Hermes,
            &[
                "let me check.",
                "<tool_call>",
                r#"{"name": "x"}"#,
                "</tool_call>",
                "done",
            ],
        );
        assert_eq!(
            kinds,
            vec![
                TokenKind::Text,
                TokenKind::ToolCall,
                TokenKind::ToolCall,
                TokenKind::ToolCall,
                TokenKind::Text
            ]
        );
    }

    #[test]
    fn gemma_tool_call_span_is_tagged() {
        let kinds = classify_all(
            ToolCallFormat::Gemma,
            &[
                "<|tool_call>call:",
                "get_weather{city:\"Paris\"}",
                "<tool_call|>",
                "ok",
            ],
        );
        assert_eq!(
            kinds,
            vec![
                TokenKind::ToolCall,
                TokenKind::ToolCall,
                TokenKind::ToolCall,
                TokenKind::Text
            ]
        );
    }

    #[test]
    fn none_format_never_enters_tool_call_state() {
        let kinds = classify_all(
            ToolCallFormat::None,
            &["<tool_call>", "still just text", "</tool_call>"],
        );
        assert!(kinds.iter().all(|k| *k == TokenKind::Text));
    }
}
