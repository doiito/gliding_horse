use std::collections::VecDeque;

use reqwest::Response;
use serde_json::Value;
use tracing::debug;

use crate::llm::response_parser::ToolCall;
use crate::llm::sse::{SseError, SseParser};
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

impl StreamingProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_chunk(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, SseError> {
        let events = self.parser.push(chunk)?;

        for event in &events {
            self.process_event(event);
        }

        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, SseError> {
        let events = self.parser.finish()?;

        for event in &events {
            self.process_event(event);
        }

        Ok(events)
    }

    fn process_event(&mut self, event: &StreamEvent) {
        self.accumulator.process_event(event);

        match event {
            StreamEvent::MessageStart(e) => {
                self.message_started = true;
                debug!("Stream message started: id={:?}, model={:?}", e.id, e.model);
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
            StreamEvent::MessageDelta(e) => {
                if let Some(ref finish_reason) = e.finish_reason {
                    debug!("Stream message delta: finish_reason={}", finish_reason);
                }
            }
            StreamEvent::MessageStop(_) => {
                self.done = true;
                debug!("Stream message stopped");
            }
            _ => {}
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
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
    pub fn new(response: Response) -> Self {
        Self {
            response,
            processor: StreamingProcessor::new(),
            buffer: Vec::new(),
        }
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, SseError> {
        loop {
            if let Some(event) = self.processor.pending.pop_front() {
                return Ok(Some(event));
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
                .map_err(|e| SseError(e.to_string()))?;

            match chunk {
                Some(bytes) => {
                    let events = self.processor.push_chunk(&bytes)?;
                    self.processor.pending.extend(events);
                }
                None => {
                    let remaining = self.processor.finish()?;
                    self.processor.pending.extend(remaining);
                    if self.processor.pending.is_empty() {
                        return Ok(None);
                    }
                }
            }
        }
    }

    pub async fn collect_all(&mut self) -> Result<StreamResponse, SseError> {
        while let Some(event) = self.next_event().await? {
            debug!("Collected stream event: {:?}", event);
        }
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
