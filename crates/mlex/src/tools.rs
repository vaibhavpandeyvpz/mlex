//! Tool ("function") calling: types passed into chat-template rendering,
//! plus parsers that recover structured `ToolCall`s from raw generated
//! text for the two output conventions seen across the supported model
//! families:
//!
//! - **Hermes-style JSON** (Qwen2/2.5/3/3.5/3.6, NemotronH):
//!   `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`.
//! - **Gemma-native** key/value macros (Gemma4):
//!   `<|tool_call>call:NAME{key:value,...}<tool_call|>`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One callable function's JSON-schema description (OpenAI `function` shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "default_params")]
    pub parameters: Value,
}

fn default_params() -> Value {
    serde_json::json!({"type": "object", "properties": {}})
}

/// A tool declaration, OpenAI-style `{"type": "function", "function": {...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "default_tool_type")]
    pub kind: String,
    pub function: ToolFunction,
}

fn default_tool_type() -> String {
    "function".to_string()
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Tool {
            kind: "function".to_string(),
            function: ToolFunction {
                name: name.into(),
                description: Some(description.into()),
                parameters,
            },
        }
    }
}

/// A single parsed tool invocation from model output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Which of the two tool-call output conventions a model family emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    Hermes,
    Gemma,
    /// Family has no documented tool-calling convention; parsing is a no-op.
    None,
}

/// Extract every tool call found in `text`, in order of appearance.
pub fn parse_tool_calls(text: &str, format: ToolCallFormat) -> Vec<ToolCall> {
    match format {
        ToolCallFormat::Hermes => parse_hermes(text),
        ToolCallFormat::Gemma => parse_gemma(text),
        ToolCallFormat::None => Vec::new(),
    }
}

fn parse_hermes(text: &str) -> Vec<ToolCall> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut calls = Vec::new();
    let mut rest = text;
    let mut idx = 0usize;
    while let Some(start) = rest.find(OPEN) {
        let after_open = &rest[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            break;
        };
        let payload = after_open[..end].trim();
        if let Ok(v) = serde_json::from_str::<Value>(payload) {
            let name = v
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or_default()
                .to_string();
            let arguments = v.get("arguments").cloned().unwrap_or(Value::Null);
            if !name.is_empty() {
                calls.push(ToolCall {
                    id: format!("call_{idx}"),
                    name,
                    arguments,
                });
                idx += 1;
            }
        }
        rest = &after_open[end + CLOSE.len()..];
    }
    calls
}

/// Best-effort Gemma-native parser. Handles scalar/string/list argument
/// values (the shapes the `format_argument` jinja macro emits); deeply
/// nested structures are not supported.
fn parse_gemma(text: &str) -> Vec<ToolCall> {
    const OPEN: &str = "<|tool_call>call:";
    const CLOSE: &str = "<tool_call|>";
    let mut calls = Vec::new();
    let mut rest = text;
    let mut idx = 0usize;
    while let Some(start) = rest.find(OPEN) {
        let after_open = &rest[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            break;
        };
        let body = &after_open[..end];
        if let Some((name, call)) = parse_gemma_call(body) {
            calls.push(ToolCall {
                id: format!("call_{idx}"),
                name,
                arguments: call,
            });
            idx += 1;
        }
        rest = &after_open[end + CLOSE.len()..];
    }
    calls
}

fn parse_gemma_call(body: &str) -> Option<(String, Value)> {
    let brace_start = body.find('{')?;
    let name = body[..brace_start].trim().to_string();
    let brace_end = body.rfind('}')?;
    let args_body = &body[brace_start + 1..brace_end];

    let mut obj = serde_json::Map::new();
    for pair in split_top_level(args_body, ',') {
        let Some(colon) = pair.find(':') else {
            continue;
        };
        let key = pair[..colon].trim().to_string();
        let raw_value = pair[colon + 1..].trim();
        obj.insert(key, parse_gemma_value(raw_value));
    }
    Some((name, Value::Object(obj)))
}

fn parse_gemma_value(raw: &str) -> Value {
    if raw.starts_with('[') && raw.ends_with(']') {
        let inner = &raw[1..raw.len() - 1];
        let items = split_top_level(inner, ',')
            .into_iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| parse_gemma_scalar(s.trim()))
            .collect();
        Value::Array(items)
    } else {
        parse_gemma_scalar(raw)
    }
}

fn parse_gemma_scalar(raw: &str) -> Value {
    let unquoted = raw.trim().trim_matches('"').trim_matches('\'');
    if let Ok(i) = unquoted.parse::<i64>() {
        Value::from(i)
    } else if let Ok(f) = unquoted.parse::<f64>() {
        Value::from(f)
    } else if unquoted == "true" || unquoted == "false" {
        Value::from(unquoted == "true")
    } else {
        Value::from(unquoted.to_string())
    }
}

/// Split `s` on `sep`, but not inside `[...]` brackets (one level deep,
/// sufficient for the simple scalar/list grammar Gemma emits).
fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            c if c == sep && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermes_single_call() {
        let text = r#"sure, let me check.<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#;
        let calls = parse_tool_calls(text, ToolCallFormat::Hermes);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "Paris");
    }

    #[test]
    fn hermes_multiple_calls() {
        let text = r#"<tool_call>{"name": "a", "arguments": {}}</tool_call>text<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>"#;
        let calls = parse_tool_calls(text, ToolCallFormat::Hermes);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments["x"], 1);
    }

    #[test]
    fn hermes_no_call() {
        let calls = parse_tool_calls("just plain text", ToolCallFormat::Hermes);
        assert!(calls.is_empty());
    }

    #[test]
    fn hermes_malformed_json_is_skipped() {
        let text = "<tool_call>{not json}</tool_call>";
        let calls = parse_tool_calls(text, ToolCallFormat::Hermes);
        assert!(calls.is_empty());
    }

    #[test]
    fn gemma_scalar_and_string_args() {
        let text = "<|tool_call>call:get_weather{location:\"Paris\",units:celsius}<tool_call|>";
        let calls = parse_tool_calls(text, ToolCallFormat::Gemma);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "Paris");
        assert_eq!(calls[0].arguments["units"], "celsius");
    }

    #[test]
    fn gemma_list_args() {
        let text = "<|tool_call>call:sum{values:[1,2,3]}<tool_call|>";
        let calls = parse_tool_calls(text, ToolCallFormat::Gemma);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["values"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn none_format_is_noop() {
        let calls = parse_tool_calls(
            "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>",
            ToolCallFormat::None,
        );
        assert!(calls.is_empty());
    }

    #[test]
    fn tool_serializes_openai_shape() {
        let tool = Tool::new(
            "get_weather",
            "Get weather for a location",
            serde_json::json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            }),
        );
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], "get_weather");
    }
}
