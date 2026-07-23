use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Default for Usage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStartEvent {
    pub id: Option<String>,
    pub model: Option<String>,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageDeltaEvent {
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentBlockStartEvent {
    pub index: u32,
    pub content_block: ContentBlock,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentBlockDeltaEvent {
    pub index: u32,
    pub delta: ContentBlockDelta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentBlockStopEvent {
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStopEvent;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String },
    Thinking { thinking: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentBlockDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    ToolCallDelta {
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamEvent {
    MessageStart(MessageStartEvent),
    MessageDelta(MessageDeltaEvent),
    ContentBlockStart(ContentBlockStartEvent),
    ContentBlockDelta(ContentBlockDeltaEvent),
    ContentBlockStop(ContentBlockStopEvent),
    MessageStop(MessageStopEvent),
}

#[derive(Debug, Clone, Default)]
pub struct StreamAccumulator {
    pub text_blocks: Vec<String>,
    pub thinking: String,
    pub tool_calls: Vec<ToolCallState>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    pub model: Option<String>,
    pub message_id: Option<String>,
    current_block_index: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct ToolCallState {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::MessageStart(e) => {
                self.message_id = e.id.clone();
                self.model = e.model.clone();
            }
            StreamEvent::ContentBlockStart(e) => {
                self.current_block_index = Some(e.index);
                match &e.content_block {
                    ContentBlock::Text { .. } => {
                        while self.text_blocks.len() <= e.index as usize {
                            self.text_blocks.push(String::new());
                        }
                    }
                    ContentBlock::ToolUse { id, name } => {
                        while self.tool_calls.len() <= e.index as usize {
                            self.tool_calls.push(ToolCallState::default());
                        }
                        self.tool_calls[e.index as usize].id = id.clone();
                        self.tool_calls[e.index as usize].name = name.clone();
                    }
                    ContentBlock::Thinking { .. } => {}
                }
            }
            StreamEvent::ContentBlockDelta(e) => match &e.delta {
                ContentBlockDelta::TextDelta { text } => {
                    let idx = e.index as usize;
                    while self.text_blocks.len() <= idx {
                        self.text_blocks.push(String::new());
                    }
                    self.text_blocks[idx].push_str(text);
                }
                ContentBlockDelta::ThinkingDelta { thinking } => {
                    self.thinking.push_str(thinking);
                }
                ContentBlockDelta::InputJsonDelta { partial_json } => {
                    let idx = e.index as usize;
                    while self.tool_calls.len() <= idx {
                        self.tool_calls.push(ToolCallState::default());
                    }
                    self.tool_calls[idx].arguments.push_str(partial_json);
                }
                ContentBlockDelta::ToolCallDelta {
                    id,
                    name,
                    arguments,
                } => {
                    let idx = e.index as usize;
                    while self.tool_calls.len() <= idx {
                        self.tool_calls.push(ToolCallState::default());
                    }
                    if let Some(i) = id {
                        self.tool_calls[idx].id = i.clone();
                    }
                    if let Some(n) = name {
                        self.tool_calls[idx].name = n.clone();
                    }
                    if let Some(a) = arguments {
                        self.tool_calls[idx].arguments.push_str(a);
                    }
                }
            },
            StreamEvent::MessageDelta(e) => {
                self.finish_reason = e.finish_reason.clone();
                if let Some(ref usage) = e.usage {
                    self.usage = Some(usage.clone());
                }
            }
            StreamEvent::ContentBlockStop(_) | StreamEvent::MessageStop(_) => {}
        }
    }

    pub fn get_text(&self) -> String {
        self.text_blocks.join("")
    }

    pub fn get_tool_calls(&self) -> Vec<(String, String, Value)> {
        self.tool_calls
            .iter()
            .filter(|tc| !tc.name.is_empty())
            .map(|tc| {
                let args: Value = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                (tc.id.clone(), tc.name.clone(), args)
            })
            .collect()
    }

    pub fn is_tool_call(&self) -> bool {
        self.finish_reason.as_deref() == Some("tool_calls")
            || !self.tool_calls.iter().any(|tc| !tc.name.is_empty())
    }
}

#[derive(Debug, Clone)]
pub struct StreamResponse {
    pub thought: Option<String>,
    pub content: String,
    pub summary: Option<String>,
    pub tool_calls: Vec<crate::llm::response_parser::ToolCall>,
    pub finish_reason: String,
    pub usage: Option<Usage>,
}

impl From<StreamAccumulator> for StreamResponse {
    fn from(acc: StreamAccumulator) -> Self {
        let tool_calls: Vec<crate::llm::response_parser::ToolCall> = acc
            .tool_calls
            .iter()
            .filter(|tc| !tc.name.is_empty())
            .map(|tc| crate::llm::response_parser::ToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: serde_json::from_str(&tc.arguments).unwrap_or(Value::Null),
            })
            .collect();

        let content = acc.get_text();
        let thought = if acc.thinking.is_empty() {
            None
        } else {
            Some(acc.thinking)
        };

        Self {
            thought,
            content,
            summary: None,
            tool_calls,
            finish_reason: acc.finish_reason.unwrap_or_else(|| "stop".to_string()),
            usage: acc.usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_text() {
        let mut acc = StreamAccumulator::new();
        acc.process_event(&StreamEvent::ContentBlockStart(ContentBlockStartEvent {
            index: 0,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        }));
        acc.process_event(&StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "Hello".to_string(),
            },
        }));
        acc.process_event(&StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: " World".to_string(),
            },
        }));

        assert_eq!(acc.get_text(), "Hello World");
    }

    #[test]
    fn test_accumulator_tool_call() {
        let mut acc = StreamAccumulator::new();
        acc.process_event(&StreamEvent::ContentBlockStart(ContentBlockStartEvent {
            index: 0,
            content_block: ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "file_read".to_string(),
            },
        }));
        acc.process_event(&StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            index: 0,
            delta: ContentBlockDelta::InputJsonDelta {
                partial_json: r#"{"path":"#.to_string(),
            },
        }));
        acc.process_event(&StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            index: 0,
            delta: ContentBlockDelta::InputJsonDelta {
                partial_json: r#""/tmp/test.txt"}"#.to_string(),
            },
        }));

        let tool_calls = acc.get_tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].0, "call_1");
        assert_eq!(tool_calls[0].1, "file_read");
    }

    #[test]
    fn test_accumulator_thinking() {
        let mut acc = StreamAccumulator::new();
        acc.process_event(&StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            index: 0,
            delta: ContentBlockDelta::ThinkingDelta {
                thinking: "Let me think...".to_string(),
            },
        }));
        acc.process_event(&StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            index: 0,
            delta: ContentBlockDelta::ThinkingDelta {
                thinking: " Step 1".to_string(),
            },
        }));

        assert_eq!(acc.thinking, "Let me think... Step 1");
    }
}
