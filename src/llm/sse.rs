use serde_json::Value;

use crate::llm::stream_types::{
    ContentBlock, ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent,
    MessageDeltaEvent, MessageStartEvent, StreamEvent, Usage,
};

#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, SseError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(frame) = self.next_frame() {
            if let Some(event) = parse_frame(&frame)? {
                events.push(event);
            }
        }

        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, SseError> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        let trailing = std::mem::take(&mut self.buffer);
        match parse_frame(&String::from_utf8_lossy(&trailing))? {
            Some(event) => Ok(vec![event]),
            None => Ok(Vec::new()),
        }
    }

    fn next_frame(&mut self) -> Option<String> {
        let separator = self
            .buffer
            .windows(2)
            .position(|window| window == b"\n\n")
            .map(|position| (position, 2))
            .or_else(|| {
                self.buffer
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|position| (position, 4))
            })?;

        let (position, separator_len) = separator;
        let frame = self
            .buffer
            .drain(..position + separator_len)
            .collect::<Vec<_>>();
        let frame_len = frame.len().saturating_sub(separator_len);
        Some(String::from_utf8_lossy(&frame[..frame_len]).into_owned())
    }
}

#[derive(Debug, Clone)]
pub struct SseError(pub String);

impl std::fmt::Display for SseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SSE Error: {}", self.0)
    }
}

impl std::error::Error for SseError {}

pub fn parse_frame(frame: &str) -> Result<Option<StreamEvent>, SseError> {
    let trimmed = frame.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let mut data_lines = Vec::new();
    let mut event_name: Option<&str> = None;

    for line in trimmed.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(name) = line.strip_prefix("event:") {
            event_name = Some(name.trim());
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }

    if matches!(event_name, Some("ping")) {
        return Ok(None);
    }

    if data_lines.is_empty() {
        return Ok(None);
    }

    let payload = data_lines.join("\n");
    if payload == "[DONE]" {
        return Ok(None);
    }

    let json: Value = match serde_json::from_str(&payload) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(
                "Failed to parse SSE payload as JSON: {} - payload: {}",
                e,
                &payload[..payload.len().min(200)]
            );
            return Ok(None);
        }
    };

    // DeepSeek Responses API streams use semantic events (type: "response.*") and
    // terminate with response.completed / response.incomplete / response.failed —
    // they never emit a trailing `data: [DONE]` frame.
    let event_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if event_type.starts_with("response.") {
        return parse_responses_api_event(&json);
    }

    parse_openai_stream_event(&json)
}

fn parse_openai_stream_event(json: &Value) -> Result<Option<StreamEvent>, SseError> {
    let event_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");

    if event_type == "ping" {
        return Ok(None);
    }

    if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
        if choices.is_empty() {
            return Ok(None);
        }

        let choice = &choices[0];
        let index = choice.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                if !content.is_empty() {
                    return Ok(Some(StreamEvent::ContentBlockDelta(
                        ContentBlockDeltaEvent {
                            index,
                            delta: ContentBlockDelta::TextDelta {
                                text: content.to_string(),
                            },
                        },
                    )));
                }
            }

            if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                if !reasoning.is_empty() {
                    return Ok(Some(StreamEvent::ContentBlockDelta(
                        ContentBlockDeltaEvent {
                            index,
                            delta: ContentBlockDelta::ThinkingDelta {
                                thinking: reasoning.to_string(),
                            },
                        },
                    )));
                }
            }

            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                if let Some(tc) = tool_calls.first() {
                    let tc_index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                    let function = tc.get("function");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let arguments = function
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    return Ok(Some(StreamEvent::ContentBlockDelta(
                        ContentBlockDeltaEvent {
                            index: tc_index,
                            delta: ContentBlockDelta::ToolCallDelta {
                                id,
                                name,
                                arguments,
                            },
                        },
                    )));
                }
            }
        }

        if let Some(finish_reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            if !finish_reason.is_empty() {
                return Ok(Some(StreamEvent::MessageDelta(MessageDeltaEvent {
                    finish_reason: Some(finish_reason.to_string()),
                    usage: None,
                })));
            }
        }
    }

    if let Some(usage) = json.get("usage") {
        let prompt_tokens = usage
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let completion_tokens = usage
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let total_tokens = usage
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        return Ok(Some(StreamEvent::MessageDelta(MessageDeltaEvent {
            finish_reason: None,
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }),
        })));
    }

    if let Some(id) = json.get("id").and_then(|v| v.as_str()) {
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        return Ok(Some(StreamEvent::MessageStart(MessageStartEvent {
            id: Some(id.to_string()),
            model,
            role: "assistant".to_string(),
        })));
    }

    Ok(None)
}

/// Parse a single DeepSeek / OpenAI Responses API stream event.
///
/// Responses API streams are made of semantic events (`response.created`,
/// `response.output_text.delta`, `response.function_call_arguments.delta`, …)
/// and end with `response.completed` / `response.incomplete` / `response.failed`.
/// They are translated into the same internal [`StreamEvent`] vocabulary used by
/// the chat-completions path so downstream consumers need no knowledge of the
/// wire format.
fn parse_responses_api_event(json: &Value) -> Result<Option<StreamEvent>, SseError> {
    let event_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match event_type {
        "response.created" => {
            let response = json.get("response");
            let id = response
                .and_then(|r| r.get("id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let model = response
                .and_then(|r| r.get("model"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            Ok(Some(StreamEvent::MessageStart(MessageStartEvent {
                id,
                model,
                role: "assistant".to_string(),
            })))
        }
        "response.output_item.added" => {
            let item_type = json
                .get("item")
                .and_then(|i| i.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let index = json
                .get("output_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            match item_type {
                "function_call" | "custom_tool_call" => {
                    let id = json
                        .get("item")
                        .and_then(|i| i.get("call_id"))
                        .or_else(|| json.get("item").and_then(|i| i.get("id")))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = json
                        .get("item")
                        .and_then(|i| i.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    Ok(Some(StreamEvent::ContentBlockStart(
                        ContentBlockStartEvent {
                            index,
                            content_block: ContentBlock::ToolUse { id, name },
                        },
                    )))
                }
                _ => Ok(None),
            }
        }
        "response.output_text.delta" => {
            let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return Ok(None);
            }
            let index = json
                .get("output_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            Ok(Some(StreamEvent::ContentBlockDelta(
                ContentBlockDeltaEvent {
                    index,
                    delta: ContentBlockDelta::TextDelta {
                        text: delta.to_string(),
                    },
                },
            )))
        }
        "response.reasoning_text.delta" => {
            let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return Ok(None);
            }
            Ok(Some(StreamEvent::ContentBlockDelta(
                ContentBlockDeltaEvent {
                    index: 0,
                    delta: ContentBlockDelta::ThinkingDelta {
                        thinking: delta.to_string(),
                    },
                },
            )))
        }
        "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
            let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return Ok(None);
            }
            let index = json
                .get("output_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            Ok(Some(StreamEvent::ContentBlockDelta(
                ContentBlockDeltaEvent {
                    index,
                    delta: ContentBlockDelta::ToolCallDelta {
                        id: None,
                        name: None,
                        arguments: Some(delta.to_string()),
                    },
                },
            )))
        }
        "response.completed" | "response.incomplete" | "response.failed" => {
            let response = json.get("response");
            let status = response
                .and_then(|r| r.get("status"))
                .and_then(|v| v.as_str())
                .unwrap_or(event_type.trim_start_matches("response."));
            let has_tool_calls = response
                .and_then(|r| r.get("output"))
                .and_then(|v| v.as_array())
                .map(|items| {
                    items.iter().any(|item| {
                        matches!(
                            item.get("type").and_then(|v| v.as_str()),
                            Some("function_call") | Some("custom_tool_call")
                        )
                    })
                })
                .unwrap_or(false);
            let finish_reason = match status {
                "completed" if has_tool_calls => "tool_calls".to_string(),
                "completed" => "stop".to_string(),
                "incomplete" => "length".to_string(),
                _ => "error".to_string(),
            };
            let usage = response.and_then(|r| r.get("usage")).map(|u| Usage {
                prompt_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                completion_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0)
                    as u32,
                total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            });
            Ok(Some(StreamEvent::MessageDelta(MessageDeltaEvent {
                finish_reason: Some(finish_reason),
                usage,
            })))
        }
        _ => Ok(None),
    }
}

#[derive(Debug, Default)]
pub struct IncrementalJsonParser {
    buffer: String,
    in_string: bool,
    escape_next: bool,
    brace_depth: i32,
    bracket_depth: i32,
    last_check_pos: usize,
}

impl IncrementalJsonParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &str) -> Option<Value> {
        self.buffer.push_str(chunk);
        self.try_parse()
    }

    fn try_parse(&mut self) -> Option<Value> {
        let chars: Vec<char> = self.buffer.chars().collect();

        for i in self.last_check_pos..chars.len() {
            let c = chars[i];

            if self.escape_next {
                self.escape_next = false;
                self.last_check_pos = i + 1;
                continue;
            }

            match c {
                '\\' if self.in_string => {
                    self.escape_next = true;
                }
                '"' => {
                    self.in_string = !self.in_string;
                }
                '{' if !self.in_string => {
                    self.brace_depth += 1;
                }
                '}' if !self.in_string => {
                    self.brace_depth -= 1;
                    if self.brace_depth == 0 && self.bracket_depth == 0 {
                        let candidate = self.buffer.clone();
                        if let Ok(v) = serde_json::from_str::<Value>(&candidate) {
                            return Some(v);
                        }
                    }
                }
                '[' if !self.in_string => {
                    self.bracket_depth += 1;
                }
                ']' if !self.in_string => {
                    self.bracket_depth -= 1;
                }
                _ => {}
            }
            self.last_check_pos = i + 1;
        }

        if self.brace_depth == 0 && self.bracket_depth == 0 && !self.buffer.is_empty() {
            if let Ok(v) = serde_json::from_str::<Value>(&self.buffer) {
                return Some(v);
            }
        }

        None
    }

    pub fn finish(&mut self) -> Option<Value> {
        if self.buffer.is_empty() {
            return None;
        }
        serde_json::from_str(&self.buffer).ok()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.in_string = false;
        self.escape_next = false;
        self.brace_depth = 0;
        self.bracket_depth = 0;
        self.last_check_pos = 0;
    }
}

#[derive(Debug, Default)]
pub struct StreamingFieldParser {
    #[allow(dead_code)]
    thought_parser: IncrementalJsonParser,
    content_parser: IncrementalJsonParser,
    #[allow(dead_code)]
    summary_parser: IncrementalJsonParser,
    #[allow(dead_code)]
    current_field: Option<String>,
    thought: String,
    content: String,
    summary: String,
}

impl StreamingFieldParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_text(&mut self, text: &str) {
        self.content.push_str(text);
    }

    pub fn push_thinking(&mut self, thinking: &str) {
        self.thought.push_str(thinking);
    }

    pub fn push_json(&mut self, partial: &str) -> Option<Value> {
        self.content_parser.push(partial)
    }

    pub fn get_thought(&self) -> Option<String> {
        if self.thought.is_empty() {
            None
        } else {
            Some(self.thought.clone())
        }
    }

    pub fn get_content(&self) -> String {
        self.content.clone()
    }

    pub fn get_summary(&self) -> Option<String> {
        if self.summary.is_empty() {
            None
        } else {
            Some(self.summary.clone())
        }
    }

    pub fn parse_structured_content(&mut self) -> Option<(Option<String>, String, Option<String>)> {
        let content = self.content.trim();
        if content.is_empty() {
            return None;
        }

        if let Ok(parsed) = serde_json::from_str::<Value>(content) {
            let thought = parsed
                .get("thought")
                .or_else(|| parsed.get("reasoning"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let content_str = parsed
                .get("content")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    other => Some(other.to_string()),
                })
                .unwrap_or_else(|| content.to_string());

            let summary = parsed
                .get("summary")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            return Some((thought, content_str, summary));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::stream_types::StreamAccumulator;

    #[test]
    fn test_sse_parser_single_frame() {
        let frame =
            concat!("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n");

        let event = parse_frame(frame).expect("frame should parse");
        assert!(event.is_some());
        if let Some(StreamEvent::ContentBlockDelta(e)) = event {
            assert_eq!(
                e.delta,
                ContentBlockDelta::TextDelta {
                    text: "Hello".to_string()
                }
            );
        } else {
            panic!("Expected ContentBlockDelta");
        }
    }

    #[test]
    fn test_sse_parser_chunked() {
        let mut parser = SseParser::new();
        let first = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel";
        let second = b"lo\"}}]}\n\n";

        assert!(parser
            .push(first)
            .expect("first chunk should buffer")
            .is_empty());
        let events = parser.push(second).expect("second chunk should parse");

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: 0,
                delta: ContentBlockDelta::TextDelta {
                    text: "Hello".to_string()
                },
            })
        );
    }

    #[test]
    fn test_sse_parser_ignores_done() {
        let mut parser = SseParser::new();
        let payload = "data: [DONE]\n\n";

        let events = parser
            .push(payload.as_bytes())
            .expect("parser should succeed");
        assert!(events.is_empty());
    }

    #[test]
    fn test_sse_parser_reasoning_content() {
        let frame = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Thinking...\"}}]}\n\n"
        );

        let event = parse_frame(frame).expect("frame should parse");
        assert!(event.is_some());
        if let Some(StreamEvent::ContentBlockDelta(e)) = event {
            assert_eq!(
                e.delta,
                ContentBlockDelta::ThinkingDelta {
                    thinking: "Thinking...".to_string()
                }
            );
        } else {
            panic!("Expected ThinkingDelta");
        }
    }

    #[test]
    fn test_incremental_json_parser() {
        let mut parser = IncrementalJsonParser::new();

        assert!(parser.push(r#"{"key": "#).is_none());
        let result = parser.push(r#""value"}"#);

        assert!(result.is_some());
        let json = result.unwrap();
        assert_eq!(json["key"], "value");
    }

    #[test]
    fn test_responses_api_text_stream() {
        let created = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"deepseek-v4-flash\",\"status\":\"in_progress\"},\"sequence_number\":0}\n\n"
        );
        let delta = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\",\"sequence_number\":1}\n\n"
        );
        let done = concat!(
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}},\"sequence_number\":2}\n\n"
        );

        let mut parser = SseParser::new();
        let mut events = parser.push(created.as_bytes()).unwrap();
        events.extend(parser.push(delta.as_bytes()).unwrap());
        events.extend(parser.push(done.as_bytes()).unwrap());

        let mut acc = StreamAccumulator::new();
        for e in &events {
            acc.process_event(e);
        }
        assert_eq!(acc.get_text(), "Hello");
        assert_eq!(acc.finish_reason.as_deref(), Some("stop"));
        let usage = acc.usage.as_ref().expect("usage should be set");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_responses_api_no_done_terminator() {
        // A response stream must not depend on data: [DONE]; response.completed is the terminator.
        let frame = concat!(
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n"
        );
        let event = parse_frame(frame).expect("frame should parse");
        assert!(matches!(
            event,
            Some(StreamEvent::MessageDelta(MessageDeltaEvent {
                finish_reason: Some(ref r),
                ..
            })) if r == "stop"
        ));
    }

    #[test]
    fn test_responses_api_reasoning_delta() {
        let frame = concat!(
            "data: {\"type\":\"response.reasoning_text.delta\",\"item_id\":\"rs_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Let me think...\",\"sequence_number\":1}\n\n"
        );
        let event = parse_frame(frame).expect("frame should parse");
        if let Some(StreamEvent::ContentBlockDelta(e)) = event {
            assert_eq!(
                e.delta,
                ContentBlockDelta::ThinkingDelta {
                    thinking: "Let me think...".to_string()
                }
            );
        } else {
            panic!("Expected ThinkingDelta");
        }
    }

    #[test]
    fn test_responses_api_tool_call_stream() {
        let added = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"file_read\",\"arguments\":\"\"},\"sequence_number\":2}\n\n"
        );
        let args_delta = concat!(
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":1,\"delta\":\"{\\\"path\\\":\\\"/tmp/a.txt\\\"}\",\"sequence_number\":3}\n\n"
        );
        let done = concat!(
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"file_read\",\"arguments\":\"{\\\"path\\\":\\\"/tmp/a.txt\\\"}\"}],\"usage\":{\"input_tokens\":20,\"output_tokens\":3,\"total_tokens\":23}},\"sequence_number\":4}\n\n"
        );

        let mut parser = SseParser::new();
        let mut events = parser.push(added.as_bytes()).unwrap();
        events.extend(parser.push(args_delta.as_bytes()).unwrap());
        events.extend(parser.push(done.as_bytes()).unwrap());

        let mut acc = StreamAccumulator::new();
        for e in &events {
            acc.process_event(e);
        }

        let tool_calls = acc.get_tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].0, "call_1");
        assert_eq!(tool_calls[0].1, "file_read");
        assert_eq!(tool_calls[0].2["path"], "/tmp/a.txt");
        assert_eq!(acc.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn test_responses_api_custom_tool_call_delta() {
        let added = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"ctc_1\",\"type\":\"custom_tool_call\",\"name\":\"apply_patch\",\"input\":\"\"},\"sequence_number\":1}\n\n"
        );
        let input_delta = concat!(
            "data: {\"type\":\"response.custom_tool_call_input.delta\",\"item_id\":\"ctc_1\",\"output_index\":0,\"delta\":\"[patch body]\",\"sequence_number\":2}\n\n"
        );

        let mut parser = SseParser::new();
        let mut events = parser.push(added.as_bytes()).unwrap();
        events.extend(parser.push(input_delta.as_bytes()).unwrap());

        let mut acc = StreamAccumulator::new();
        for e in &events {
            acc.process_event(e);
        }
        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0].name, "apply_patch");
        assert_eq!(acc.tool_calls[0].arguments, "[patch body]");
    }

    #[test]
    fn test_responses_api_incomplete_sets_length() {
        let frame = concat!(
            "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":100,\"total_tokens\":105}}}\n\n"
        );
        let event = parse_frame(frame).expect("frame should parse");
        assert!(matches!(
            event,
            Some(StreamEvent::MessageDelta(MessageDeltaEvent {
                finish_reason: Some(ref r),
                ..
            })) if r == "length"
        ));
    }

    #[test]
    fn test_responses_api_failed_sets_error() {
        let frame = concat!(
            "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"server_error\"},\"output\":[],\"usage\":null}}\n\n"
        );
        let event = parse_frame(frame).expect("frame should parse");
        assert!(matches!(
            event,
            Some(StreamEvent::MessageDelta(MessageDeltaEvent {
                finish_reason: Some(ref r),
                ..
            })) if r == "error"
        ));
    }
}
