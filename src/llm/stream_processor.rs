use std::collections::VecDeque;

use reqwest::Response;
use serde_json::Value;
use tracing::debug;

use crate::llm::response_parser::ToolCall;
use crate::llm::sse::{SseDialect, SseError, SseErrorKind, SseParser, SseTerminalOutcome};
use crate::llm::stream_types::{
    ContentBlock, ContentBlockDelta, StreamAccumulator, StreamEvent, StreamResponse, Usage,
};

pub type StreamCallback = Box<dyn Fn(&StreamEvent) + Send + Sync>;

#[derive(Debug, Default)]
pub struct StreamingProcessor {
    parser: SseParser,
    accumulator: StreamAccumulator,
    pending: VecDeque<StreamEvent>,
    done: bool,
    terminal_error: Option<SseErrorKind>,
    message_started: bool,
    current_block_index: u32,
    tool_call_states: Vec<ToolCallState>,
}

#[derive(Debug, Clone, Default)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
}

/// Content-free telemetry for a provider stream event.
///
/// `StreamEvent`'s derived `Debug` representation includes assistant text,
/// reasoning and partial tool arguments.  Those payloads belong in the
/// explicitly configured interaction capture, not normal tracing.  Keep this
/// projection deliberately limited to fixed classifications and scalar sizes.
#[derive(Debug, PartialEq, Eq)]
struct StreamEventMetadata {
    event_kind: &'static str,
    block_index: Option<u32>,
    payload_kind: Option<&'static str>,
    payload_bytes: usize,
    message_id_present: bool,
    model_bytes: usize,
    role_kind: Option<&'static str>,
    tool_call_id_present: bool,
    tool_name_bytes: usize,
    finish_reason_kind: Option<&'static str>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

impl StreamEventMetadata {
    fn new(event_kind: &'static str) -> Self {
        Self {
            event_kind,
            block_index: None,
            payload_kind: None,
            payload_bytes: 0,
            message_id_present: false,
            model_bytes: 0,
            role_kind: None,
            tool_call_id_present: false,
            tool_name_bytes: 0,
            finish_reason_kind: None,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
        }
    }

    fn from_event(event: &StreamEvent) -> Self {
        match event {
            StreamEvent::MessageStart(event) => {
                let mut metadata = Self::new("message_start");
                metadata.message_id_present = event.id.is_some();
                metadata.model_bytes = event.model.as_ref().map_or(0, String::len);
                metadata.role_kind = Some(match event.role.as_str() {
                    "assistant" => "assistant",
                    "user" => "user",
                    "system" => "system",
                    "tool" => "tool",
                    _ => "other",
                });
                metadata
            }
            StreamEvent::MessageDelta(event) => {
                let mut metadata = Self::new("message_delta");
                metadata.finish_reason_kind =
                    event.finish_reason.as_deref().map(classify_finish_reason);
                if let Some(usage) = event.usage.as_ref() {
                    metadata.prompt_tokens = Some(usage.prompt_tokens);
                    metadata.completion_tokens = Some(usage.completion_tokens);
                    metadata.total_tokens = Some(usage.total_tokens);
                }
                metadata
            }
            StreamEvent::ContentBlockStart(event) => {
                let mut metadata = Self::new("content_block_start");
                metadata.block_index = Some(event.index);
                match &event.content_block {
                    ContentBlock::Text { text } => {
                        metadata.payload_kind = Some("text");
                        metadata.payload_bytes = text.len();
                    }
                    ContentBlock::ToolUse { id, name } => {
                        metadata.payload_kind = Some("tool_use");
                        metadata.tool_call_id_present = !id.is_empty();
                        metadata.tool_name_bytes = name.len();
                    }
                    ContentBlock::Thinking { thinking } => {
                        metadata.payload_kind = Some("thinking");
                        metadata.payload_bytes = thinking.len();
                    }
                }
                metadata
            }
            StreamEvent::ContentBlockDelta(event) => {
                let mut metadata = Self::new("content_block_delta");
                metadata.block_index = Some(event.index);
                match &event.delta {
                    ContentBlockDelta::TextDelta { text } => {
                        metadata.payload_kind = Some("text");
                        metadata.payload_bytes = text.len();
                    }
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        metadata.payload_kind = Some("tool_input_json");
                        metadata.payload_bytes = partial_json.len();
                    }
                    ContentBlockDelta::ThinkingDelta { thinking } => {
                        metadata.payload_kind = Some("thinking");
                        metadata.payload_bytes = thinking.len();
                    }
                    ContentBlockDelta::ToolCallDelta {
                        id,
                        name,
                        arguments,
                    } => {
                        metadata.payload_kind = Some("tool_call");
                        metadata.tool_call_id_present =
                            id.as_deref().is_some_and(|id| !id.is_empty());
                        metadata.tool_name_bytes = name.as_ref().map_or(0, String::len);
                        metadata.payload_bytes = arguments.as_ref().map_or(0, String::len);
                    }
                }
                metadata
            }
            StreamEvent::ContentBlockStop(event) => {
                let mut metadata = Self::new("content_block_stop");
                metadata.block_index = Some(event.index);
                metadata
            }
            StreamEvent::MessageStop(_) => Self::new("message_stop"),
        }
    }
}

fn classify_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "stop",
        "length" | "max_tokens" => "length",
        "tool_call" | "tool_calls" | "function_call" => "tool_call",
        "content_filter" => "content_filter",
        _ => "other",
    }
}

impl StreamingProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_dialect(dialect: SseDialect) -> Self {
        Self {
            parser: SseParser::with_dialect(dialect),
            ..Self::default()
        }
    }

    pub fn push_chunk(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, SseError> {
        let events = self.parser.push(chunk)?;

        for event in &events {
            self.process_event(event);
        }
        self.sync_terminal_outcome();

        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, SseError> {
        let events = self.parser.finish()?;

        for event in &events {
            self.process_event(event);
        }
        self.sync_terminal_outcome();

        Ok(events)
    }

    fn sync_terminal_outcome(&mut self) {
        match self.parser.terminal_outcome() {
            Some(SseTerminalOutcome::Completed) => {
                // The parser emits MessageStop for a successful protocol
                // terminal, and `process_event` owns the public done flag.
            }
            Some(SseTerminalOutcome::Failed(kind)) => self.terminal_error = Some(kind),
            None => {}
        }
    }

    fn process_event(&mut self, event: &StreamEvent) {
        self.accumulator.process_event(event);
        debug!(
            metadata = ?StreamEventMetadata::from_event(event),
            "stream event processed"
        );

        match event {
            StreamEvent::MessageStart(_) => {
                self.message_started = true;
            }
            StreamEvent::ContentBlockStart(e) => {
                self.current_block_index = e.index;
                if let ContentBlock::ToolUse { id, name } = &e.content_block {
                    while self.tool_call_states.len() <= e.index as usize {
                        self.tool_call_states.push(ToolCallState::default());
                    }
                    self.tool_call_states[e.index as usize].id = id.clone();
                    self.tool_call_states[e.index as usize].name = name.clone();
                }
            }
            StreamEvent::ContentBlockDelta(e) => {
                if let ContentBlockDelta::ToolCallDelta {
                    id,
                    name,
                    arguments,
                } = &e.delta
                {
                    let idx = e.index as usize;
                    while self.tool_call_states.len() <= idx {
                        self.tool_call_states.push(ToolCallState::default());
                    }
                    if let Some(i) = id {
                        self.tool_call_states[idx].id = i.clone();
                    }
                    if let Some(n) = name {
                        self.tool_call_states[idx].name = n.clone();
                    }
                    if let Some(a) = arguments {
                        self.tool_call_states[idx].arguments.push_str(a);
                    }
                }
            }
            StreamEvent::MessageStop(_) => {
                self.done = true;
            }
            _ => {}
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    fn take_terminal_error(&mut self) -> Option<SseError> {
        self.terminal_error.take().map(SseError::new)
    }

    pub fn get_accumulator(&self) -> &StreamAccumulator {
        &self.accumulator
    }

    pub fn into_response(self) -> StreamResponse {
        let content = self.accumulator.get_text();
        let thought = if self.accumulator.thinking.is_empty() {
            None
        } else {
            Some(self.accumulator.thinking.clone())
        };

        let tool_calls: Vec<crate::llm::response_parser::ToolCall> = self
            .accumulator
            .tool_calls
            .iter()
            .filter(|tc| !tc.name.is_empty())
            .map(|tc| crate::llm::response_parser::ToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: serde_json::from_str(&tc.arguments).unwrap_or(Value::Null),
            })
            .collect();

        let finish_reason = self
            .accumulator
            .finish_reason
            .clone()
            .unwrap_or_else(|| "stop".to_string());

        let mut response = StreamResponse {
            thought,
            content,
            summary: None,
            tool_calls,
            finish_reason,
            usage: self.accumulator.usage.clone(),
        };

        if response.thought.is_none() {
            if let Some((thought, content, summary)) =
                Self::parse_structured_content_static(&response.content)
            {
                response.thought = thought;
                response.content = content;
                response.summary = summary;
            }
        }

        response
    }

    fn parse_structured_content_static(
        content: &str,
    ) -> Option<(Option<String>, String, Option<String>)> {
        let content = content.trim();
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

pub struct MessageStream {
    response: Response,
    processor: StreamingProcessor,
    #[allow(dead_code)]
    buffer: Vec<u8>,
}

impl MessageStream {
    /// Compatibility constructor for Chat Completions streams. Production
    /// endpoint dispatch must use [`Self::with_dialect`] explicitly.
    pub fn new(response: Response) -> Self {
        Self::with_dialect(response, SseDialect::ChatCompletions)
    }

    pub fn with_dialect(response: Response, dialect: SseDialect) -> Self {
        Self {
            response,
            processor: StreamingProcessor::with_dialect(dialect),
            buffer: Vec::new(),
        }
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, SseError> {
        loop {
            if let Some(event) = self.processor.pending.pop_front() {
                return Ok(Some(event));
            }

            if let Some(error) = self.processor.take_terminal_error() {
                return Err(error);
            }

            if self.processor.is_done() {
                let remaining = self.processor.finish()?;
                self.processor.pending.extend(remaining);
                if let Some(event) = self.processor.pending.pop_front() {
                    return Ok(Some(event));
                }
                return Ok(None);
            }

            let chunk = self
                .response
                .chunk()
                .await
                .map_err(|error| SseError::from_response_body(&error))?;

            match chunk {
                Some(bytes) => {
                    let events = self.processor.push_chunk(&bytes)?;
                    self.processor.pending.extend(events);
                }
                None => {
                    let remaining = self.processor.finish()?;
                    self.processor.pending.extend(remaining);
                    if let Some(event) = self.processor.pending.pop_front() {
                        return Ok(Some(event));
                    }
                    if let Some(error) = self.processor.take_terminal_error() {
                        return Err(error);
                    }
                    if self.processor.is_done() {
                        return Ok(None);
                    }
                    // A clean HTTP body EOF is not an LLM protocol terminal.
                    // Chat requires `[DONE]`; Responses requires one of its
                    // explicit terminal events. Never execute a merely partial
                    // assistant/tool payload as a completed model decision.
                    return Err(SseError::new(SseErrorKind::Protocol));
                }
            }
        }
    }

    pub async fn collect_all(&mut self) -> Result<StreamResponse, SseError> {
        while self.next_event().await?.is_some() {}
        Ok(std::mem::take(&mut self.processor).into_response())
    }

    pub async fn collect_with_callback<F>(
        &mut self,
        mut callback: F,
    ) -> Result<StreamResponse, SseError>
    where
        F: FnMut(&StreamEvent),
    {
        while let Some(event) = self.next_event().await? {
            callback(&event);
        }
        Ok(std::mem::take(&mut self.processor).into_response())
    }
}

pub struct StreamingResponseBuilder {
    thought: String,
    content: String,
    summary: String,
    tool_calls: Vec<ToolCall>,
    finish_reason: String,
    usage: Option<Usage>,
}

impl StreamingResponseBuilder {
    pub fn new() -> Self {
        Self {
            thought: String::new(),
            content: String::new(),
            summary: String::new(),
            tool_calls: Vec::new(),
            finish_reason: "stop".to_string(),
            usage: None,
        }
    }

    pub fn with_thought(mut self, thought: &str) -> Self {
        self.thought.push_str(thought);
        self
    }

    pub fn with_content(mut self, content: &str) -> Self {
        self.content.push_str(content);
        self
    }

    pub fn with_summary(mut self, summary: &str) -> Self {
        self.summary.push_str(summary);
        self
    }

    pub fn with_tool_call(mut self, tool_call: ToolCall) -> Self {
        self.tool_calls.push(tool_call);
        self
    }

    pub fn with_finish_reason(mut self, reason: &str) -> Self {
        self.finish_reason = reason.to_string();
        self
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn build(self) -> StreamResponse {
        StreamResponse {
            thought: if self.thought.is_empty() {
                None
            } else {
                Some(self.thought)
            },
            content: self.content,
            summary: if self.summary.is_empty() {
                None
            } else {
                Some(self.summary)
            },
            tool_calls: self.tool_calls,
            finish_reason: self.finish_reason,
            usage: self.usage,
        }
    }
}

impl Default for StreamingResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_processor_text() {
        let mut processor = StreamingProcessor::new();

        let chunk1 = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n";
        let chunk2 = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" World\"}}]}\n\n";

        processor.push_chunk(chunk1).unwrap();
        processor.push_chunk(chunk2).unwrap();

        let acc = processor.get_accumulator();
        assert_eq!(acc.get_text(), "Hello World");
    }

    #[test]
    fn test_streaming_processor_thinking() {
        let mut processor = StreamingProcessor::new();

        let chunk = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Thinking...\"}}]}\n\n";

        processor.push_chunk(chunk).unwrap();

        let acc = processor.get_accumulator();
        assert_eq!(acc.thinking, "Thinking...");
    }

    #[test]
    fn stream_event_metadata_never_contains_provider_payloads() {
        let body_events = [
            (
                StreamEvent::ContentBlockStart(crate::llm::stream_types::ContentBlockStartEvent {
                    index: 0,
                    content_block: ContentBlock::Text {
                        text: "secret-initial-text".to_string(),
                    },
                }),
                "secret-initial-text",
            ),
            (
                StreamEvent::ContentBlockStart(crate::llm::stream_types::ContentBlockStartEvent {
                    index: 1,
                    content_block: ContentBlock::Thinking {
                        thinking: "secret-initial-thinking".to_string(),
                    },
                }),
                "secret-initial-thinking",
            ),
            (
                StreamEvent::ContentBlockDelta(crate::llm::stream_types::ContentBlockDeltaEvent {
                    index: 2,
                    delta: ContentBlockDelta::TextDelta {
                        text: "secret-text-delta".to_string(),
                    },
                }),
                "secret-text-delta",
            ),
            (
                StreamEvent::ContentBlockDelta(crate::llm::stream_types::ContentBlockDeltaEvent {
                    index: 3,
                    delta: ContentBlockDelta::ThinkingDelta {
                        thinking: "secret-thinking-delta".to_string(),
                    },
                }),
                "secret-thinking-delta",
            ),
            (
                StreamEvent::ContentBlockDelta(crate::llm::stream_types::ContentBlockDeltaEvent {
                    index: 4,
                    delta: ContentBlockDelta::InputJsonDelta {
                        partial_json: "secret-partial-json".to_string(),
                    },
                }),
                "secret-partial-json",
            ),
        ];
        for (event, secret) in body_events {
            let rendered = format!("{:?}", StreamEventMetadata::from_event(&event));
            assert!(!rendered.contains(secret));
        }

        let tool_id = "call-secret-id";
        let tool_name = "secret_tool_name";
        let arguments = r#"{"token":"secret-tool-argument"}"#;
        let event =
            StreamEvent::ContentBlockDelta(crate::llm::stream_types::ContentBlockDeltaEvent {
                index: 7,
                delta: ContentBlockDelta::ToolCallDelta {
                    id: Some(tool_id.to_string()),
                    name: Some(tool_name.to_string()),
                    arguments: Some(arguments.to_string()),
                },
            });

        let metadata = StreamEventMetadata::from_event(&event);
        assert_eq!(metadata.event_kind, "content_block_delta");
        assert_eq!(metadata.block_index, Some(7));
        assert_eq!(metadata.payload_kind, Some("tool_call"));
        assert_eq!(metadata.payload_bytes, arguments.len());
        assert_eq!(metadata.tool_name_bytes, tool_name.len());
        assert!(metadata.tool_call_id_present);

        let rendered = format!("{metadata:?}");
        assert!(!rendered.contains(tool_id));
        assert!(!rendered.contains(tool_name));
        assert!(!rendered.contains(arguments));

        let start = StreamEvent::MessageStart(crate::llm::stream_types::MessageStartEvent {
            id: Some("secret-message-id".to_string()),
            model: Some("secret-provider-model".to_string()),
            role: "secret-provider-role".to_string(),
        });
        let rendered = format!("{:?}", StreamEventMetadata::from_event(&start));
        assert!(!rendered.contains("secret-message-id"));
        assert!(!rendered.contains("secret-provider-model"));
        assert!(!rendered.contains("secret-provider-role"));

        let finish = StreamEvent::MessageDelta(crate::llm::stream_types::MessageDeltaEvent {
            finish_reason: Some("secret-finish-reason".to_string()),
            usage: None,
        });
        let rendered = format!("{:?}", StreamEventMetadata::from_event(&finish));
        assert!(!rendered.contains("secret-finish-reason"));
        assert!(rendered.contains("other"));
    }

    #[test]
    fn test_streaming_response_builder() {
        let response = StreamingResponseBuilder::new()
            .with_thought("Let me think")
            .with_content("The answer is 42")
            .with_summary("Calculated answer")
            .with_finish_reason("stop")
            .build();

        assert_eq!(response.thought, Some("Let me think".to_string()));
        assert_eq!(response.content, "The answer is 42");
        assert_eq!(response.summary, Some("Calculated answer".to_string()));
        assert_eq!(response.finish_reason, "stop");
    }
}
