use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::CoreError;

/// Standard LLM response with thought/content/summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    /// Full reasoning/thinking process
    pub thought: String,
    /// Structured output content
    pub content: Value,
    /// Concise summary for L1 session
    pub summary: String,
    /// Finish reason: stop, tool_calls, length, etc.
    pub finish_reason: String,
    /// Tool calls if LLM requested tools
    pub tool_calls: Vec<ToolCall>,
}

/// A tool call from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Parse LLM response from raw JSON string
pub fn parse_response(raw: &str) -> Result<LLMResponse, CoreError> {
    match serde_json::from_str::<Value>(raw) {
        Ok(val) => parse_response_value(&val),
        Err(_) => {
            // Try extracting JSON from markdown code fence
            if let Some(json_str) = extract_json_from_markdown(raw) {
                match serde_json::from_str::<Value>(&json_str) {
                    Ok(val) => parse_response_value(&val),
                    Err(e) => Err(CoreError::InvalidJsonLd {
                        message: format!("Failed to parse LLM response: {}", e),
                    }),
                }
            } else {
                Err(CoreError::InvalidJsonLd {
                    message: "LLM response is not valid JSON".to_string(),
                })
            }
        }
    }
}

fn parse_response_value(val: &Value) -> Result<LLMResponse, CoreError> {
    let thought = val
        .get("thought")
        .or_else(|| val.get("reasoning"))
        .or_else(|| val.get("thinking"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let content = val.get("content").cloned().unwrap_or(Value::Null);

    let summary = val
        .get("summary")
        .or_else(|| val.get("meta").and_then(|m| m.get("summary")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = val
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop")
        .to_string();

    let tool_calls = parse_tool_calls(val);

    Ok(LLMResponse {
        thought,
        content,
        summary,
        finish_reason,
        tool_calls,
    })
}

fn parse_tool_calls(val: &Value) -> Vec<ToolCall> {
    let calls = val
        .get("tool_calls")
        .or_else(|| val.get("tools"))
        .or_else(|| val.get("functions"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    calls
        .iter()
        .filter_map(|c| {
            let id = c
                .get("id")
                .or_else(|| c.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("tool_0")
                .to_string();
            let name = c
                .get("name")
                .or_else(|| c.get("function"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args = c
                .get("arguments")
                .or_else(|| c.get("input"))
                .or_else(|| c.get("params"))
                .cloned()
                .unwrap_or(Value::Null);

            if name.is_empty() {
                None
            } else {
                Some(ToolCall {
                    id,
                    name,
                    arguments: args,
                })
            }
        })
        .collect()
}

/// Extract JSON from markdown code fences
fn extract_json_from_markdown(text: &str) -> Option<String> {
    let re = regex::Regex::new(r"```(?:json)?\s*\n([\s\S]*?)```").ok()?;
    if let Some(cap) = re.captures(text) {
        let candidate = cap[1].trim().to_string();
        // Validate it's parseable
        if serde_json::from_str::<Value>(&candidate).is_ok() {
            return Some(candidate);
        }
    }
    // Try finding a JSON object directly: { ... }
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            let candidate = text[start..=end].to_string();
            if serde_json::from_str::<Value>(&candidate).is_ok() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Parse tool call results from LLM response
pub fn parse_tool_results(raw: &str) -> Result<Vec<(String, Value)>, CoreError> {
    let val: Value = serde_json::from_str(raw).map_err(|e| CoreError::InvalidJsonLd {
        message: format!("Failed to parse tool results: {}", e),
    })?;

    let results = val
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let tool_id = item
                        .get("tool_call_id")
                        .or_else(|| item.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let output = item
                        .get("output")
                        .or_else(|| item.get("result"))
                        .or_else(|| item.get("content"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    if tool_id.is_empty() {
                        None
                    } else {
                        Some((tool_id, output))
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_response() {
        let raw = r#"{
            "thought": "I need to analyze the data",
            "content": {"result": "success", "value": 42},
            "summary": "Analysis complete"
        }"#;

        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.thought, "I need to analyze the data");
        assert_eq!(resp.content["result"], "success");
        assert_eq!(resp.summary, "Analysis complete");
    }

    #[test]
    fn test_parse_with_tool_calls() {
        let raw = r#"{
            "thought": "Let me read the file first",
            "content": null,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "file_read",
                    "arguments": {"path": "/tmp/test.txt"}
                }
            ]
        }"#;

        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "file_read");
    }

    #[test]
    fn test_extract_json_from_markdown() {
        let text = "Some text\n```json\n{\"key\": \"value\"}\n```\nmore text";
        let extracted = extract_json_from_markdown(text);
        assert!(extracted.is_some());
        assert!(extracted.unwrap().contains("key"));
    }

    #[test]
    fn test_parse_minimal_response() {
        let raw = r#"{"content": "hello"}"#;
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.content, "hello");
        assert!(resp.thought.is_empty());
    }
}
