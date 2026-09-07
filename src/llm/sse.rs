use serde_json::Value;

use crate::llm::stream_types::{
    ContentBlock, ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent,
    MessageDeltaEvent, MessageStartEvent, MessageStopEvent, StreamEvent, Usage,
};
use crate::CoreError;

const STREAM_FAILURE_PREFIX: &str = "llm_stream_failure:";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SseTerminalOutcome {
    Completed,
    Failed(SseErrorKind),
}

/// Wire protocol selected by the request endpoint.
///
/// Chat Completions and Responses deliberately have different terminal
/// markers.  Keeping the dialect on the stateful parser prevents a response
/// from being accepted merely because its payload happens to resemble the
/// other endpoint's protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseDialect {
    ChatCompletions,
    Responses,
}

#[derive(Debug)]
pub struct SseParser {
    buffer: Vec<u8>,
    terminal: Option<SseTerminalOutcome>,
    message_started: bool,
    dialect: SseDialect,
}

impl Default for SseParser {
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            terminal: None,
            message_started: false,
            // Compatibility default for existing direct parser users.  The
            // production gateway always constructs the parser with the
            // endpoint's explicit dialect.
            dialect: SseDialect::ChatCompletions,
        }
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_dialect(dialect: SseDialect) -> Self {
        Self {
            dialect,
            ..Self::default()
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, SseError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(frame) = self.next_frame() {
            self.consume_frame(&frame, &mut events)?;
        }

        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, SseError> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        let trailing = std::mem::take(&mut self.buffer);
        let mut events = Vec::new();
        self.consume_frame(&String::from_utf8_lossy(&trailing), &mut events)?;
        Ok(events)
    }

    pub(crate) const fn terminal_outcome(&self) -> Option<SseTerminalOutcome> {
        self.terminal
    }

    fn consume_frame(
        &mut self,
        frame: &str,
        events: &mut Vec<StreamEvent>,
    ) -> Result<(), SseError> {
        let parsed = parse_frame_events(frame, Some(self.dialect))?;
        let has_protocol_material = parsed.message_start.is_some()
            || !parsed.events.is_empty()
            || parsed.terminal.is_some();
        if self.terminal.is_some() {
            if has_protocol_material {
                return Err(SseError::new(SseErrorKind::Protocol));
            }
            return Ok(());
        }

        if !self.message_started {
            if let Some(start) = parsed.message_start {
                events.push(StreamEvent::MessageStart(start));
                self.message_started = true;
            }
        }
        events.extend(parsed.events);
        if let Some(terminal) = parsed.terminal {
            self.terminal = Some(terminal);
        }
        Ok(())
    }

    fn next_frame(&mut self) -> Option<String> {
        let lf_separator = self
            .buffer
            .windows(2)
            .position(|window| window == b"\n\n")
            .map(|position| (position, 2));
        let crlf_separator = self
            .buffer
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|position| (position, 4));

        // A stream is allowed to contain either line ending convention. Pick
        // the earliest delimiter on the wire. Preferring LF merely because an
        // LF delimiter exists later in the buffer can merge two frames when a
        // CRLF frame is followed by an LF frame.
        let separator = match (lf_separator, crlf_separator) {
            (Some(lf), Some(crlf)) => Some(if lf.0 <= crlf.0 { lf } else { crlf }),
            (Some(lf), None) => Some(lf),
            (None, Some(crlf)) => Some(crlf),
            (None, None) => None,
        }?;

        let (position, separator_len) = separator;
        let frame = self
            .buffer
            .drain(..position + separator_len)
            .collect::<Vec<_>>();
        let frame_len = frame.len().saturating_sub(separator_len);
        Some(String::from_utf8_lossy(&frame[..frame_len]).into_owned())
    }
}

/// Stable, payload-free classification for failures while consuming a model
/// stream. Provider text, reasoning and tool arguments must never be embedded
/// in this value because it is propagated through normal logs and lifecycle
/// events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseErrorKind {
    Protocol,
    OutputTokenLimit,
    ProviderIncomplete,
    ProviderFailed,
    TransportTimeout,
    TransportConnect,
    TransportDecode,
    TransportBody,
    TransportOther,
    ResponseHookRetryUnsupported,
    ResponseHookRejected,
}

impl SseErrorKind {
    pub const fn error_class(self) -> &'static str {
        match self {
            Self::Protocol => "stream_protocol",
            Self::OutputTokenLimit => "output_token_limit",
            Self::ProviderIncomplete => "stream_provider_incomplete",
            Self::ProviderFailed => "stream_provider_failed",
            Self::TransportTimeout => "stream_transport_timeout",
            Self::TransportConnect => "stream_transport_connect",
            Self::TransportDecode => "stream_transport_decode",
            Self::TransportBody => "stream_transport_body",
            Self::TransportOther => "stream_transport",
            Self::ResponseHookRetryUnsupported => "stream_response_hook_retry_unsupported",
            Self::ResponseHookRejected => "stream_response_hook_rejected",
        }
    }

    const fn safe_description(self) -> &'static str {
        match self {
            Self::Protocol => "provider SSE protocol violation",
            Self::OutputTokenLimit => "provider stream exhausted its output token budget",
            Self::ProviderIncomplete => "provider response stream ended incomplete",
            Self::ProviderFailed => "provider reported a failed response stream",
            Self::TransportTimeout => "provider stream timed out",
            Self::TransportConnect => "provider stream connection failed",
            Self::TransportDecode => "provider response body encoding could not be decoded",
            Self::TransportBody => "provider response body ended unexpectedly",
            Self::TransportOther => "provider stream transport failed",
            Self::ResponseHookRetryUnsupported => {
                "stream response hook requested an unsupported replay"
            }
            Self::ResponseHookRejected => "stream response was rejected by hook policy",
        }
    }

    /// Whether replaying the model request is generally safe and useful. The
    /// AgentRunner does not execute streamed tool calls before the terminal
    /// marker, so transport/provider failures can be retried without replaying
    /// a tool side effect. Budget/protocol/policy failures are deterministic.
    pub const fn retryable(self) -> Option<bool> {
        match self {
            Self::TransportTimeout
            | Self::TransportConnect
            | Self::TransportDecode
            | Self::TransportBody
            | Self::TransportOther
            | Self::ProviderFailed => Some(true),
            Self::Protocol
            | Self::OutputTokenLimit
            | Self::ProviderIncomplete
            | Self::ResponseHookRetryUnsupported
            | Self::ResponseHookRejected => Some(false),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseError {
    kind: SseErrorKind,
}

impl SseError {
    pub const fn new(kind: SseErrorKind) -> Self {
        Self { kind }
    }

    pub const fn kind(&self) -> SseErrorKind {
        self.kind
    }

    pub const fn error_class(&self) -> &'static str {
        self.kind.error_class()
    }

    pub const fn retryable(&self) -> Option<bool> {
        self.kind.retryable()
    }

    /// Convert to the repository-wide error type without embedding provider
    /// payloads and without losing the stable stream failure class.
    pub fn into_core_error(self) -> CoreError {
        CoreError::Internal {
            message: format!(
                "{STREAM_FAILURE_PREFIX}{}: {}",
                self.error_class(),
                self.kind.safe_description()
            ),
        }
    }

    pub(crate) fn from_response_body(error: &reqwest::Error) -> Self {
        // `Response::bytes_stream` wraps every underlying HTTP-body failure
        // in reqwest's Decode category. Inspect the causal chain before that
        // broad wrapper so a truncated Content-Length/chunked response is not
        // mislabeled as a character/content-encoding problem. The cause text
        // is used only for classification and is never emitted to logs.
        let mut cause = std::error::Error::source(error);
        let mut ended_early = false;
        while let Some(current) = cause {
            if current
                .downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::UnexpectedEof)
            {
                ended_early = true;
                break;
            }
            let description = current.to_string().to_ascii_lowercase();
            if description.contains("unexpected eof")
                || description.contains("end of file before message length reached")
                || description.contains("connection closed before message completed")
            {
                ended_early = true;
                break;
            }
            cause = current.source();
        }
        let kind = if error.is_timeout() {
            SseErrorKind::TransportTimeout
        } else if error.is_connect() {
            SseErrorKind::TransportConnect
        } else if error.is_body() || ended_early {
            SseErrorKind::TransportBody
        } else if error.is_decode() {
            SseErrorKind::TransportDecode
        } else {
            SseErrorKind::TransportOther
        };
        tracing::warn!(
            error_class = kind.error_class(),
            http_status = error.status().map(|status| status.as_u16()),
            "Provider stream body read failed"
        );
        Self::new(kind)
    }
}

pub(crate) fn stream_core_error_class(error: &CoreError) -> Option<&'static str> {
    let CoreError::Internal { message } = error else {
        return None;
    };
    let class = message
        .strip_prefix(STREAM_FAILURE_PREFIX)?
        .split(':')
        .next()?;
    match class {
        "stream_protocol" => Some("stream_protocol"),
        "output_token_limit" => Some("output_token_limit"),
        "stream_provider_incomplete" => Some("stream_provider_incomplete"),
        "stream_provider_failed" => Some("stream_provider_failed"),
        "stream_transport_timeout" => Some("stream_transport_timeout"),
        "stream_transport_connect" => Some("stream_transport_connect"),
        "stream_transport_decode" => Some("stream_transport_decode"),
        "stream_transport_body" => Some("stream_transport_body"),
        "stream_transport" => Some("stream_transport"),
        "stream_response_hook_retry_unsupported" => Some("stream_response_hook_retry_unsupported"),
        "stream_response_hook_rejected" => Some("stream_response_hook_rejected"),
        _ => None,
    }
}

pub(crate) fn stream_core_error_retryable(error: &CoreError) -> Option<bool> {
    match stream_core_error_class(error)? {
        "stream_transport_timeout"
        | "stream_transport_connect"
        | "stream_transport_decode"
        | "stream_transport_body"
        | "stream_transport"
        | "stream_provider_failed" => Some(true),
        "stream_protocol"
        | "output_token_limit"
        | "stream_provider_incomplete"
        | "stream_response_hook_retry_unsupported"
        | "stream_response_hook_rejected" => Some(false),
        _ => None,
    }
}

impl std::fmt::Display for SseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SSE error [{}]: {}",
            self.error_class(),
            self.kind.safe_description()
        )
    }
}

impl std::error::Error for SseError {}

#[derive(Debug, Default)]
struct ParsedSseFrame {
    message_start: Option<MessageStartEvent>,
    events: Vec<StreamEvent>,
    terminal: Option<SseTerminalOutcome>,
}

/// Compatibility parser for callers that consume one event at a time. The
/// stateful [`SseParser`] uses `parse_frame_events` directly so a single Chat
/// Completions frame can carry content, usage, finish metadata and multiple
/// parallel tool-call deltas without dropping any of them.
pub fn parse_frame(frame: &str) -> Result<Option<StreamEvent>, SseError> {
    // Compatibility helper for callers parsing an isolated frame without
    // request context. Stateful/production parsing must use `SseParser`,
    // whose dialect is fixed by the endpoint before any bytes are consumed.
    let parsed = parse_frame_events(frame, None)?;
    if let Some(SseTerminalOutcome::Failed(kind)) = parsed.terminal {
        return Err(SseError::new(kind));
    }
    Ok(parsed
        .events
        .into_iter()
        .next()
        .or_else(|| parsed.message_start.map(StreamEvent::MessageStart)))
}

fn parse_frame_events(
    frame: &str,
    dialect: Option<SseDialect>,
) -> Result<ParsedSseFrame, SseError> {
    let trimmed = frame.trim();
    if trimmed.is_empty() {
        return Ok(ParsedSseFrame::default());
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
        return Ok(ParsedSseFrame::default());
    }

    if data_lines.is_empty() {
        return Ok(ParsedSseFrame::default());
    }

    let payload = data_lines.join("\n");
    if payload == "[DONE]" {
        if dialect == Some(SseDialect::Responses) {
            return Err(SseError::new(SseErrorKind::Protocol));
        }
        return Ok(ParsedSseFrame {
            events: vec![StreamEvent::MessageStop(MessageStopEvent)],
            terminal: Some(SseTerminalOutcome::Completed),
            ..ParsedSseFrame::default()
        });
    }

    let json: Value = match serde_json::from_str(&payload) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                error_class = SseErrorKind::Protocol.error_class(),
                payload_bytes = payload.len(),
                error_line = e.line(),
                error_column = e.column(),
                "Provider SSE data frame was not valid JSON"
            );
            return Err(SseError::new(SseErrorKind::Protocol));
        }
    };

    // DeepSeek Responses API streams use semantic events (type: "response.*") and
    // terminate with response.completed / response.incomplete / response.failed —
    // they never emit a trailing `data: [DONE]` frame.
    let event_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if event_type.starts_with("response.") {
        if dialect == Some(SseDialect::ChatCompletions) {
            return Err(SseError::new(SseErrorKind::Protocol));
        }
        return parse_responses_api_frame(&json);
    }

    if dialect == Some(SseDialect::Responses) {
        return Err(SseError::new(SseErrorKind::Protocol));
    }

    parse_openai_stream_event(&json)
}

fn parse_openai_stream_event(json: &Value) -> Result<ParsedSseFrame, SseError> {
    let event_type = json.get("type").and_then(|v| v.as_str()).unwrap_or("");

    if event_type == "ping" {
        return Ok(ParsedSseFrame::default());
    }

    let message_start = json
        .get("id")
        .and_then(Value::as_str)
        .map(|id| MessageStartEvent {
            id: Some(id.to_string()),
            model: json
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_string),
            role: "assistant".to_string(),
        });
    let mut events = Vec::new();

    if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
        for choice in choices {
            let index = choice.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

            if let Some(delta) = choice.get("delta") {
                if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                    if !content.is_empty() {
                        events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                            index,
                            delta: ContentBlockDelta::TextDelta {
                                text: content.to_string(),
                            },
                        }));
                    }
                }

                if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                    if !reasoning.is_empty() {
                        events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                            index,
                            delta: ContentBlockDelta::ThinkingDelta {
                                thinking: reasoning.to_string(),
                            },
                        }));
                    }
                }

                if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    for tool_call in tool_calls {
                        let tool_index =
                            tool_call.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                        let id = tool_call
                            .get("id")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        let function = tool_call.get("function");
                        let name = function
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        let arguments = function
                            .and_then(|function| function.get("arguments"))
                            .and_then(Value::as_str)
                            .map(str::to_string);

                        events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                            index: tool_index,
                            delta: ContentBlockDelta::ToolCallDelta {
                                id,
                                name,
                                arguments,
                            },
                        }));
                    }
                }
            }

            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
                if !finish_reason.is_empty() {
                    events.push(StreamEvent::MessageDelta(MessageDeltaEvent {
                        finish_reason: Some(finish_reason.to_string()),
                        usage: None,
                    }));
                }
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

        events.push(StreamEvent::MessageDelta(MessageDeltaEvent {
            finish_reason: None,
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }),
        }));
    }

    Ok(ParsedSseFrame {
        message_start,
        events,
        terminal: None,
    })
}

/// Parse a single DeepSeek / OpenAI Responses API stream event.
///
/// Responses API streams are made of semantic events (`response.created`,
/// `response.output_text.delta`, `response.function_call_arguments.delta`, …)
/// and end with `response.completed` / `response.incomplete` / `response.failed`.
/// They are translated into the same internal [`StreamEvent`] vocabulary used by
/// the chat-completions path so downstream consumers need no knowledge of the
/// wire format.
fn parse_responses_api_frame(json: &Value) -> Result<ParsedSseFrame, SseError> {
    let event_type = json.get("type").and_then(Value::as_str).unwrap_or("");
    if !matches!(
        event_type,
        "response.completed" | "response.incomplete" | "response.failed"
    ) {
        return parse_responses_api_event(json).map(|event| match event {
            Some(StreamEvent::MessageStart(start)) => ParsedSseFrame {
                message_start: Some(start),
                ..ParsedSseFrame::default()
            },
            event => ParsedSseFrame {
                events: event.into_iter().collect(),
                ..ParsedSseFrame::default()
            },
        });
    }

    let response = json.get("response");
    let response_id = response
        .and_then(|value| value.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let response_model = response
        .and_then(|value| value.get("model"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let message_start = response_id.map(|id| MessageStartEvent {
        id: Some(id),
        model: response_model,
        role: "assistant".to_string(),
    });
    let has_tool_calls = response
        .and_then(|value| value.get("output"))
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items.iter().any(|item| {
                matches!(
                    item.get("type").and_then(Value::as_str),
                    Some("function_call") | Some("custom_tool_call")
                )
            })
        });
    let usage = response
        .and_then(|value| value.get("usage"))
        .map(|usage| Usage {
            prompt_tokens: usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
            completion_tokens: usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
            total_tokens: usage
                .get("total_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
        });

    let (finish_reason, terminal) = match event_type {
        "response.completed" => (
            if has_tool_calls { "tool_calls" } else { "stop" },
            SseTerminalOutcome::Completed,
        ),
        "response.incomplete" => {
            let reason = response
                .and_then(|value| value.get("incomplete_details"))
                .and_then(|value| value.get("reason"))
                .and_then(Value::as_str);
            let kind = if reason == Some("max_output_tokens") {
                SseErrorKind::OutputTokenLimit
            } else {
                SseErrorKind::ProviderIncomplete
            };
            ("length", SseTerminalOutcome::Failed(kind))
        }
        "response.failed" => (
            "error",
            SseTerminalOutcome::Failed(SseErrorKind::ProviderFailed),
        ),
        _ => unreachable!("terminal Responses event checked above"),
    };
    let mut events = vec![StreamEvent::MessageDelta(MessageDeltaEvent {
        finish_reason: Some(finish_reason.to_string()),
        usage,
    })];
    if terminal == SseTerminalOutcome::Completed {
        events.push(StreamEvent::MessageStop(MessageStopEvent));
    }
    Ok(ParsedSseFrame {
        message_start,
        events,
        terminal: Some(terminal),
    })
}

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
                    let item = json.get("item");
                    let id = if item_type == "function_call" {
                        // Responses function calls have a distinct output item
                        // id and protocol call_id. Never substitute the former
                        // for the latter: doing so breaks the provider's exact
                        // assistant/tool-result pairing contract.
                        item.and_then(|item| item.get("call_id"))
                            .and_then(|value| value.as_str())
                            .filter(|call_id| !call_id.is_empty())
                            .ok_or_else(|| SseError::new(SseErrorKind::Protocol))?
                            .to_string()
                    } else {
                        item.and_then(|item| item.get("id"))
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .to_string()
                    };
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
        // Terminal Responses events require both payload events and a
        // protocol outcome, so they are handled by `parse_responses_api_frame`.
        "response.completed" | "response.incomplete" | "response.failed" => {
            Err(SseError::new(SseErrorKind::Protocol))
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
    fn mixed_frame_delimiters_use_the_earliest_wire_boundary() {
        let payload = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"}}]}\r\n\r\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"}}]}\n\n"
        );
        let mut parser = SseParser::new();
        let events = parser.push(payload.as_bytes()).expect("mixed SSE frames");

        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                delta: ContentBlockDelta::TextDelta { text },
                ..
            }) if text == "first"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                delta: ContentBlockDelta::TextDelta { text },
                ..
            }) if text == "second"
        ));
    }

    #[test]
    fn malformed_data_frame_fails_closed_without_retaining_payload() {
        let secret = "TOP_SECRET_PROVIDER_FRAGMENT";
        let payload = format!("data: {{not-json:{secret}}}\n\n");
        let mut parser = SseParser::new();
        let error = parser
            .push(payload.as_bytes())
            .expect_err("malformed provider data must not be silently discarded");

        assert_eq!(error.kind(), SseErrorKind::Protocol);
        assert_eq!(error.error_class(), "stream_protocol");
        let rendered = format!("{error:?} {error}");
        assert!(!rendered.contains(secret));
    }

    #[test]
    fn chat_done_is_the_success_terminal() {
        let mut parser = SseParser::new();
        let payload = "data: [DONE]\n\n";

        let events = parser
            .push(payload.as_bytes())
            .expect("parser should succeed");
        assert_eq!(events, vec![StreamEvent::MessageStop(MessageStopEvent)]);
        assert_eq!(
            parser.terminal_outcome(),
            Some(SseTerminalOutcome::Completed)
        );
    }

    #[test]
    fn explicit_dialect_rejects_the_other_protocol_terminal() {
        let mut responses = SseParser::with_dialect(SseDialect::Responses);
        let done_error = responses
            .push(b"data: [DONE]\n\n")
            .expect_err("Responses must not accept the Chat terminal");
        assert_eq!(done_error.kind(), SseErrorKind::Protocol);

        let mut chat = SseParser::with_dialect(SseDialect::ChatCompletions);
        let completed_error = chat
            .push(
                b"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[]}}\n\n",
            )
            .expect_err("Chat must not accept a Responses terminal");
        assert_eq!(completed_error.kind(), SseErrorKind::Protocol);
    }

    #[test]
    fn one_chat_frame_preserves_every_parallel_raw_tool_call_id() {
        let first_id = " Provider/CALL:Raw#A ";
        let second_id = "provider/call:raw#b";
        let frame = format!(
            "data: {{\"id\":\"response-raw\",\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":{first_id:?},\"function\":{{\"name\":\"first_tool\",\"arguments\":\"{{}}\"}}}},{{\"index\":1,\"id\":{second_id:?},\"function\":{{\"name\":\"second_tool\",\"arguments\":\"{{\\\"value\\\":2}}\"}}}}]}},\"finish_reason\":\"tool_calls\"}}],\"usage\":{{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}}}\n\n"
        );
        let mut parser = SseParser::new();
        let mut events = parser.push(frame.as_bytes()).expect("parallel tool frame");
        events.extend(parser.push(b"data: [DONE]\n\n").expect("chat terminal"));

        let mut accumulator = StreamAccumulator::new();
        for event in &events {
            accumulator.process_event(event);
        }
        let calls = accumulator.get_tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, first_id);
        assert_eq!(calls[0].1, "first_tool");
        assert_eq!(calls[1].0, second_id);
        assert_eq!(calls[1].1, "second_tool");
        assert_eq!(calls[1].2["value"], 2);
        assert!(matches!(events.last(), Some(StreamEvent::MessageStop(_))));
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

        let mut parser = SseParser::with_dialect(SseDialect::Responses);
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
        assert!(matches!(events.last(), Some(StreamEvent::MessageStop(_))));
        assert_eq!(
            parser.terminal_outcome(),
            Some(SseTerminalOutcome::Completed)
        );
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
        let raw_call_id = " provider/CALL:Raw#01 ";
        let added = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"fc_must_not_replace_call_id\",\"type\":\"function_call\",\"call_id\":\" provider/CALL:Raw#01 \",\"name\":\"file_read\",\"arguments\":\"\"},\"sequence_number\":2}\n\n"
        );
        let args_delta = concat!(
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"output_index\":1,\"delta\":\"{\\\"path\\\":\\\"/tmp/a.txt\\\"}\",\"sequence_number\":3}\n\n"
        );
        let done = format!(
            "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_2\",\"status\":\"completed\",\"output\":[{{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":{raw_call_id:?},\"name\":\"file_read\",\"arguments\":\"{{\\\"path\\\":\\\"/tmp/a.txt\\\"}}\"}}],\"usage\":{{\"input_tokens\":20,\"output_tokens\":3,\"total_tokens\":23}}}},\"sequence_number\":4}}\n\n"
        );

        let mut parser = SseParser::with_dialect(SseDialect::Responses);
        let mut events = parser.push(added.as_bytes()).unwrap();
        events.extend(parser.push(args_delta.as_bytes()).unwrap());
        events.extend(parser.push(done.as_bytes()).unwrap());

        let mut acc = StreamAccumulator::new();
        for e in &events {
            acc.process_event(e);
        }

        let tool_calls = acc.get_tool_calls();
        assert_eq!(tool_calls.len(), 1);
        // `item.id` and `call_id` intentionally differ. The parser must carry
        // the provider's raw call_id byte-for-byte; it must not trim,
        // normalize, synthesize or replace it with the output item id.
        assert_eq!(tool_calls[0].0, raw_call_id);
        assert_eq!(tool_calls[0].1, "file_read");
        assert_eq!(tool_calls[0].2["path"], "/tmp/a.txt");
        assert_eq!(acc.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn responses_function_call_without_call_id_fails_instead_of_using_item_id() {
        let frame = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_is_not_a_call_id\",\"type\":\"function_call\",\"name\":\"file_read\",\"arguments\":\"\"}}\n\n"
        );
        let error = parse_frame(frame).expect_err("missing provider call_id must fail closed");
        assert_eq!(error.kind(), SseErrorKind::Protocol);
        assert!(!format!("{error:?} {error}").contains("fc_is_not_a_call_id"));
    }

    #[test]
    fn test_responses_api_custom_tool_call_delta() {
        let added = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"ctc_1\",\"type\":\"custom_tool_call\",\"name\":\"apply_patch\",\"input\":\"\"},\"sequence_number\":1}\n\n"
        );
        let input_delta = concat!(
            "data: {\"type\":\"response.custom_tool_call_input.delta\",\"item_id\":\"ctc_1\",\"output_index\":0,\"delta\":\"[patch body]\",\"sequence_number\":2}\n\n"
        );

        let mut parser = SseParser::with_dialect(SseDialect::Responses);
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
    fn responses_api_incomplete_is_a_safe_terminal_failure() {
        let frame = concat!(
            "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":100,\"total_tokens\":105}}}\n\n"
        );
        let error = parse_frame(frame).expect_err("incomplete is not success");
        assert_eq!(error.kind(), SseErrorKind::OutputTokenLimit);
        assert_eq!(error.error_class(), "output_token_limit");
        assert_eq!(error.retryable(), Some(false));
    }

    #[test]
    fn responses_api_failed_is_a_safe_retryable_terminal_failure() {
        let frame = concat!(
            "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"server_error\"},\"output\":[],\"usage\":null}}\n\n"
        );
        let error = parse_frame(frame).expect_err("failed is not success");
        assert_eq!(error.kind(), SseErrorKind::ProviderFailed);
        assert_eq!(error.error_class(), "stream_provider_failed");
        assert_eq!(error.retryable(), Some(true));
        assert!(!format!("{error:?} {error}").contains("server_error"));
    }
}
