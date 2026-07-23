use std::sync::Arc;

use serde_json::Value;

use crate::gateway::unified_gateway::{ChatCompletionResponse, ChatMessage, UnifiedGateway};
use crate::llm::message::Message;
use crate::llm::response_parser::{self, LLMResponse, ToolCall};
use crate::CoreError;

/// High-level LLM client
///
/// Wraps UnifiedGateway with message construction, response parsing,
/// and standard prompt template injection.
pub struct LLMClient {
    gateway: Arc<UnifiedGateway>,
    default_model: String,
}

impl LLMClient {
    pub fn new(gateway: Arc<UnifiedGateway>) -> Self {
        let model = gateway.default_model().to_string();
        Self {
            gateway,
            default_model: model,
        }
    }

    pub fn with_model(gateway: Arc<UnifiedGateway>, model: &str) -> Self {
        Self {
            gateway,
            default_model: model.to_string(),
        }
    }

    pub fn set_default_model(&mut self, model: &str) {
        self.default_model = model.to_string();
    }

    /// Send chat messages and get parsed response
    pub async fn chat(&self, messages: Vec<Message>) -> Result<LLMResponse, CoreError> {
        self.chat_with_model(&self.default_model, messages).await
    }

    /// Send chat messages with specific model
    pub async fn chat_with_model(
        &self,
        model: &str,
        messages: Vec<Message>,
    ) -> Result<LLMResponse, CoreError> {
        let chat_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            })
            .collect();

        let response = self.gateway.chat_with_model(model, chat_messages).await?;

        Self::parse_completion_response(&response)
    }

    /// Send chat with tool definitions
    pub async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Vec<Value>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<LLMResponse, CoreError> {
        let chat_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            })
            .collect();

        let response = self
            .gateway
            .chat_with_params(
                model,
                chat_messages,
                temperature,
                max_tokens,
                Some(tools),
                None,
            )
            .await?;

        Self::parse_completion_response(&response)
    }

    /// Parse gateway response into LLMResponse
    fn parse_completion_response(
        response: &ChatCompletionResponse,
    ) -> Result<LLMResponse, CoreError> {
        let choice = response
            .choices
            .first()
            .ok_or_else(|| CoreError::Internal {
                message: "Empty LLM response choices".to_string(),
            })?;

        let content = choice.message.content.as_deref().unwrap_or("");
        let finish_reason = choice
            .finish_reason
            .as_deref()
            .unwrap_or("stop")
            .to_string();

        // Parse content JSON for structured thought/content/summary
        let mut parsed = response_parser::parse_response(content).unwrap_or_else(|_| {
            let fallback = response_parser::parse_response("{}").expect("{} is valid JSON");
            if let Ok(val) = serde_json::from_str::<Value>(content) {
                if val.get("content").is_some() || val.get("thought").is_some() {
                    return response_parser::parse_response(content)
                        .expect("parse_response already validated above");
                }
            }
            fallback
        });
        parsed.finish_reason = finish_reason;

        // Extract tool calls from response message (overrides any in content)
        if let Some(ref tool_calls) = choice.message.tool_calls {
            parsed.tool_calls = tool_calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null),
                })
                .collect();
        }

        Ok(parsed)
    }

    /// Health check via gateway
    pub async fn health_check(&self) -> Result<bool, CoreError> {
        self.gateway.health_check().await
    }

    /// Send chat request with JSON format constraint
    pub async fn chat_with_json_format(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Vec<Value>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<LLMResponse, CoreError> {
        let chat_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            })
            .collect();

        let response = self
            .gateway
            .chat_with_params(
                model,
                chat_messages,
                temperature,
                max_tokens,
                Some(tools),
                None,
            )
            .await?;

        Self::parse_completion_response(&response)
    }

    /// Get default model name
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// Get gateway reference (for direct access to rate limiter, cache, etc.)
    pub fn gateway(&self) -> &Arc<UnifiedGateway> {
        &self.gateway
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::unified_gateway::ResponseToolCall;

    #[test]
    fn test_parse_tool_calls_from_response() {
        let response = ChatCompletionResponse {
            id: Some("test".to_string()),
            choices: vec![crate::gateway::unified_gateway::Choice {
                index: 0,
                message: crate::gateway::unified_gateway::ResponseMessage {
                    role: "assistant".to_string(),
                    content: Some(r#"{"thought":"","content":"Using tool"}"#.to_string()),
                    reasoning_content: None,
                    tool_calls: Some(vec![ResponseToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: crate::gateway::unified_gateway::ResponseToolCallFunction {
                            name: "file_read".to_string(),
                            arguments: r#"{"path":"/tmp/test.txt"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
        };

        let result = LLMClient::parse_completion_response(&response).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "file_read");
        assert_eq!(result.finish_reason, "tool_calls");
    }
}
