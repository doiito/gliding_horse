use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::config::settings::GatewaySettings;
use crate::llm::stream_processor::MessageStream;
use crate::CoreError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallPayload {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: i32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ResponseToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub struct UnifiedGateway {
    base_url: RwLock<String>,
    api_key: RwLock<String>,
    client: Client,
    model_mapping: RwLock<HashMap<String, String>>,
    default_model: RwLock<String>,
    #[allow(dead_code)]
    timeout_seconds: u64,
    max_retries: u32,
    retry_base_ms: u64,
    /// When enabled, requests for Responses-API-capable models (deepseek-v4-flash)
    /// are sent to `{base_url}/v1/responses`; all other models keep using
    /// `/v1/chat/completions`.
    use_responses_api: RwLock<bool>,
}

impl UnifiedGateway {
    pub fn new(settings: &GatewaySettings) -> Result<Self, CoreError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(settings.timeout_seconds))
            .build()
            .map_err(|e| CoreError::Internal {
                message: format!("Failed to create HTTP client: {}", e),
            })?;

        Ok(Self {
            base_url: RwLock::new(settings.base_url.trim_end_matches('/').to_string()),
            api_key: RwLock::new(settings.api_key.clone()),
            client,
            model_mapping: RwLock::new(settings.model_mapping.clone()),
            default_model: RwLock::new(settings.default_model.clone()),
            timeout_seconds: settings.timeout_seconds,
            max_retries: settings.max_retries,
            retry_base_ms: settings.retry_base_ms,
            use_responses_api: RwLock::new(settings.use_responses_api),
        })
    }

    pub fn default_model(&self) -> String {
        self.default_model.read().unwrap().clone()
    }

    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<ChatCompletionResponse, CoreError> {
        let model = self.get_model("default");
        let sanitized = Self::sanitize_tool_messages(messages);
        self.chat_with_model(&model, sanitized).await
    }

    pub async fn chat_with_model(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<ChatCompletionResponse, CoreError> {
        let sanitized = Self::sanitize_tool_messages(messages);
        if self.should_use_responses_api(model) {
            let url = format!("{}/v1/responses", self.base_url.read().unwrap());
            let body = Self::build_responses_body(model, &sanitized, None, None, None, None, false);
            return self.send_responses_request(&url, body).await;
        }
        let url = format!("{}/v1/chat/completions", self.base_url.read().unwrap());
        let body = serde_json::json!({
            "model": model,
            "messages": sanitized,
        });
        self.send_request(&url, body).await
    }

    pub async fn chat_with_params(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
    ) -> Result<ChatCompletionResponse, CoreError> {
        let messages = Self::sanitize_tool_messages(messages);
        // Pre-validate messages: check for empty content that might cause 400 errors
        for (i, msg) in messages.iter().enumerate() {
            if msg.content.trim().is_empty() && msg.role != "assistant" {
                warn!(
                    msg_idx = i, role = %msg.role,
                    "Message has empty content — this may cause 400 errors from the LLM API"
                );
            }
        }

        if self.should_use_responses_api(model) {
            let url = format!("{}/v1/responses", self.base_url.read().unwrap());
            let body = Self::build_responses_body(
                model,
                &messages,
                temperature,
                max_tokens,
                tools,
                tool_choice,
                false,
            );
            return self.send_responses_request(&url, body).await;
        }

        let url = format!("{}/v1/chat/completions", self.base_url.read().unwrap());
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
        });
        if let Some(temp) = temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = max_tokens {
            body["max_tokens"] = serde_json::json!(tokens);
        }
        if let Some(t) = tools {
            body["tools"] = serde_json::json!(t);
            body["tool_choice"] = Self::parse_tool_choice(tool_choice.unwrap_or("auto"));
        }
        self.send_request(&url, body).await
    }

    /// Serialize a tool_choice string into the JSON value the API expects.
    /// `"auto"` / `"none"` / `"required"` stay as plain strings, while a JSON
    /// object string (e.g. `{"type":"function","name":"get_weather"}`) is parsed
    /// into an object instead of being double-quoted.
    fn parse_tool_choice(tool_choice: &str) -> Value {
        if tool_choice.trim_start().starts_with('{') {
            if let Ok(v) = serde_json::from_str::<Value>(tool_choice) {
                return v;
            }
        }
        serde_json::json!(tool_choice)
    }

    async fn send_request(
        &self,
        url: &str,
        body: Value,
    ) -> Result<ChatCompletionResponse, CoreError> {
        self.send_with_retry(url, body, |json| {
            serde_json::from_value(json.clone()).map_err(|e| CoreError::Internal {
                message: format!("Failed to parse LLM response JSON: {}", e),
            })
        })
        .await
    }

    async fn send_responses_request(
        &self,
        url: &str,
        body: Value,
    ) -> Result<ChatCompletionResponse, CoreError> {
        self.send_with_retry(url, body, Self::parse_responses_response)
            .await
    }

    async fn send_with_retry<F>(
        &self,
        url: &str,
        body: Value,
        parse: F,
    ) -> Result<ChatCompletionResponse, CoreError>
    where
        F: Fn(&Value) -> Result<ChatCompletionResponse, CoreError>,
    {
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let backoff = Duration::from_millis(self.retry_base_ms * u64::pow(2, attempt - 1));
                tokio::time::sleep(backoff).await;
                debug!(attempt, "Retrying LLM API call");
            }

            let req_body = body.clone();
            let req = self
                .client
                .post(url)
                .header(
                    "Authorization",
                    format!("Bearer {}", self.api_key.read().unwrap()),
                )
                .header("Content-Type", "application/json")
                .json(&req_body);

            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let response_text = match resp.text().await {
                            Ok(t) => t,
                            Err(e) => {
                                warn!(error = %e, "Failed to read LLM response body");
                                last_error = Some(CoreError::Internal {
                                    message: format!("Failed to read response body: {}", e),
                                });
                                continue;
                            }
                        };
                        let json: Value = match serde_json::from_str(&response_text) {
                            Ok(v) => v,
                            Err(e) => {
                                warn!(error = %e, response_len = response_text.len(), "Failed to parse LLM response");
                                last_error = Some(CoreError::Internal {
                                    message: format!(
                                        "Failed to parse LLM response: {} (response length: {})",
                                        e,
                                        response_text.len()
                                    ),
                                });
                                continue;
                            }
                        };
                        match parse(&json) {
                            Ok(result) => {
                                info!(
                                    model = %body["model"],
                                    usage = ?result.usage.as_ref().map(|u| u.total_tokens),
                                    "LLM API call successful"
                                );
                                return Ok(result);
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to convert LLM response");
                                last_error = Some(e);
                            }
                        }
                    } else {
                        let text = resp.text().await.unwrap_or_default();
                        // Embed a preview of the request body into the error message
                        // for debugging 4xx errors directly from the TUI / result display.
                        let req_body_str =
                            serde_json::to_string_pretty(&req_body).unwrap_or_default();
                        let req_preview: String = req_body_str.chars().take(8000).collect();
                        warn!(status = %status, body = %text, req_preview = %req_preview, "LLM API error");
                        last_error = Some(CoreError::Internal {
                            message: format!(
                                "LLM API error ({}): {}\nrequest_preview(8k)={}",
                                status, text, req_preview
                            ),
                        });
                        if status.is_client_error() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "LLM API request failed");
                    last_error = Some(CoreError::Internal {
                        message: format!("LLM API request failed: {}", e),
                    });
                }
            }
        }

        Err(last_error.unwrap_or_else(|| CoreError::Internal {
            message: "LLM API call failed after all retries".to_string(),
        }))
    }

    pub fn set_base_url(&self, url: String) {
        *self.base_url.write().unwrap() = url.trim_end_matches('/').to_string();
    }

    pub fn set_api_key(&self, key: String) {
        *self.api_key.write().unwrap() = key;
    }

    pub fn set_default_model(&self, model: String) {
        *self.default_model.write().unwrap() = model.clone();
        self.model_mapping
            .write()
            .unwrap()
            .insert("default".to_string(), model);
    }

    pub fn set_model_mapping(&self, task_type: String, model: String) {
        self.model_mapping.write().unwrap().insert(task_type, model);
    }

    pub fn get_model(&self, task_type: &str) -> String {
        let mapping = self.model_mapping.read().unwrap();
        mapping
            .get(task_type)
            .or_else(|| mapping.get("default"))
            .cloned()
            .unwrap_or_else(|| self.default_model.read().unwrap().clone())
    }

    /// Toggle Responses API usage at runtime.
    pub fn set_use_responses_api(&self, enabled: bool) {
        *self.use_responses_api.write().unwrap() = enabled;
    }

    fn should_use_responses_api(&self, model: &str) -> bool {
        *self.use_responses_api.read().unwrap() && Self::is_responses_capable_model(model)
    }

    /// Only `deepseek-v4-flash` supports the Responses API today;
    /// `deepseek-v4-pro` keeps using chat completions until DeepSeek enables it.
    fn is_responses_capable_model(model: &str) -> bool {
        let m = model.to_lowercase();
        m == "deepseek-v4-flash" || m.starts_with("deepseek-v4-flash-")
    }

    fn build_responses_body(
        model: &str,
        messages: &[ChatMessage],
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
        stream: bool,
    ) -> Value {
        let (instructions, input_items) = Self::responses_input_items(messages);
        let mut body = serde_json::json!({
            "model": model,
            "input": input_items,
            "stream": stream,
        });
        if let Some(inst) = instructions {
            body["instructions"] = serde_json::json!(inst);
        }
        if let Some(temp) = temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = max_tokens {
            body["max_output_tokens"] = serde_json::json!(tokens);
        }
        if let Some(t) = tools {
            body["tools"] = serde_json::json!(Self::convert_responses_tools(t));
            body["tool_choice"] = Self::parse_tool_choice(tool_choice.unwrap_or("auto"));
        }
        body
    }

    /// Chat-completions tool definitions nest the function under `"function"`,
    /// but the Responses API expects `name`/`description`/`parameters` flattened
    /// onto the tool object. Non-function tools (web_search, custom) pass through.
    fn convert_responses_tools(tools: Vec<Value>) -> Vec<Value> {
        tools
            .into_iter()
            .map(|tool| {
                if tool.get("type").and_then(|v| v.as_str()) == Some("function") {
                    if let Some(func) = tool.get("function").filter(|f| f.is_object()) {
                        let mut out = serde_json::Map::new();
                        out.insert("type".to_string(), Value::String("function".to_string()));
                        for key in ["name", "description", "parameters", "strict"] {
                            if let Some(v) = func.get(key) {
                                out.insert(key.to_string(), v.clone());
                            }
                        }
                        return Value::Object(out);
                    }
                }
                tool
            })
            .collect()
    }

    /// Convert chat-completions messages into Responses API input items.
    /// The first non-empty system message becomes `instructions` (treated as the
    /// first system message by DeepSeek); tool messages become
    /// `function_call_output` items, and assistant tool calls become
    /// `function_call` items following their assistant message.
    fn responses_input_items(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
        let mut instructions: Option<String> = None;
        let mut items: Vec<Value> = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    if instructions.is_none() && !msg.content.is_empty() {
                        instructions = Some(msg.content.clone());
                    } else {
                        items.push(Self::responses_message_item("system", &msg.content));
                    }
                }
                "developer" => items.push(Self::responses_message_item("developer", &msg.content)),
                "user" => items.push(Self::responses_message_item("user", &msg.content)),
                "assistant" => {
                    items.push(Self::responses_message_item("assistant", &msg.content));
                    if let Some(tool_calls) = &msg.tool_calls {
                        for tc in tool_calls {
                            items.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }));
                        }
                    }
                }
                "tool" => {
                    items.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id.clone().unwrap_or_default(),
                        "output": msg.content,
                    }));
                }
                _ => items.push(Self::responses_message_item("user", &msg.content)),
            }
        }

        (instructions, items)
    }

    fn responses_message_item(role: &str, text: &str) -> Value {
        let block_type = if role == "assistant" {
            "output_text"
        } else {
            "input_text"
        };
        serde_json::json!({
            "type": "message",
            "role": role,
            "content": [{"type": block_type, "text": text}],
        })
    }

    /// Convert a `/v1/responses` response object into the internal
    /// [`ChatCompletionResponse`] shape so downstream callers stay unchanged.
    fn parse_responses_response(json: &Value) -> Result<ChatCompletionResponse, CoreError> {
        let id = json
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let output = json
            .get("output")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut text_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ResponseToolCall> = Vec::new();

        for item in &output {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(blocks) = item.get("content").and_then(|v| v.as_array()) {
                        for block in blocks {
                            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                text_parts.push(t.to_string());
                            }
                        }
                    }
                }
                Some("reasoning") => {
                    if let Some(blocks) = item.get("content").and_then(|v| v.as_array()) {
                        for block in blocks {
                            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                reasoning_parts.push(t.to_string());
                            }
                        }
                    }
                }
                Some("function_call") => {
                    tool_calls.push(ResponseToolCall {
                        id: item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        call_type: "function".to_string(),
                        function: ResponseToolCallFunction {
                            name: item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            arguments: item
                                .get("arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        },
                    });
                }
                Some("custom_tool_call") => {
                    tool_calls.push(ResponseToolCall {
                        id: item
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        call_type: "custom".to_string(),
                        function: ResponseToolCallFunction {
                            name: item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            arguments: item
                                .get("input")
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                        },
                    });
                }
                _ => {}
            }
        }

        let finish_reason = if tool_calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        let usage = json.get("usage").map(|u| Usage {
            prompt_tokens: u
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: u
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        });

        Ok(ChatCompletionResponse {
            id,
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: if text_parts.is_empty() {
                        None
                    } else {
                        Some(text_parts.join(""))
                    },
                    reasoning_content: if reasoning_parts.is_empty() {
                        None
                    } else {
                        Some(reasoning_parts.join(""))
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage,
        })
    }

    pub async fn health_check(&self) -> Result<bool, CoreError> {
        let url = format!("{}/v1/models", self.base_url.read().unwrap());
        match self
            .client
            .get(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.read().unwrap()),
            )
            .send()
            .await
        {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Sanitize messages to avoid OpenAI/DeepSeek API error:
    /// "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"
    fn sanitize_tool_messages(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        crate::core::context_compressor::ContextWindowManager::remove_orphaned_tool_messages(
            messages,
        )
    }

    pub fn supports_native_reasoning(&self, model: &str) -> bool {
        let model_lower = model.to_lowercase();

        if model_lower.contains("deepseek-r1") || model_lower.contains("deepseek-reasoning") {
            return true;
        }

        if model_lower.starts_with("o1-")
            || model_lower.starts_with("o3-")
            || model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
        {
            return true;
        }

        if model_lower.contains("gemini") && model_lower.contains("thinking") {
            return true;
        }

        false
    }

    pub async fn stream_chat_with_params(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
    ) -> Result<MessageStream, CoreError> {
        if self.should_use_responses_api(model) {
            let url = format!("{}/v1/responses", self.base_url.read().unwrap());
            let body = Self::build_responses_body(
                model,
                &messages,
                temperature,
                max_tokens,
                tools,
                tool_choice,
                true,
            );
            return self.send_stream_request(&url, body).await;
        }

        let url = format!("{}/v1/chat/completions", self.base_url.read().unwrap());
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
        });
        if let Some(temp) = temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = max_tokens {
            body["max_tokens"] = serde_json::json!(tokens);
        }
        if let Some(t) = tools {
            body["tools"] = serde_json::json!(t);
            body["tool_choice"] = Self::parse_tool_choice(tool_choice.unwrap_or("auto"));
        }

        self.send_stream_request(&url, body).await
    }

    async fn send_stream_request(
        &self,
        url: &str,
        body: Value,
    ) -> Result<MessageStream, CoreError> {
        let req = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.read().unwrap()),
            )
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body);

        let response = req.send().await.map_err(|e| CoreError::Internal {
            message: format!("Stream request failed: {}", e),
        })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(CoreError::Internal {
                message: format!("Stream API error ({}): {}", status, text),
            });
        }

        info!(model = %body["model"], "Stream request started");
        Ok(MessageStream::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_mapping() {
        let settings = GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "sk-test".to_string(),
            default_model: "deepseek-v4-flash".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: HashMap::from([
                ("planning".to_string(), "deepseek-v4-pro".to_string()),
                ("default".to_string(), "deepseek-v4-flash".to_string()),
            ]),
        };

        let gateway = UnifiedGateway::new(&settings).unwrap();
        assert_eq!(gateway.get_model("planning"), "deepseek-v4-pro");
        assert_eq!(gateway.get_model("unknown"), "deepseek-v4-flash");
    }

    #[test]
    fn test_runtime_updates() {
        let settings = GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "sk-test".to_string(),
            default_model: "deepseek-v4-flash".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: HashMap::from([(
                "default".to_string(),
                "deepseek-v4-flash".to_string(),
            )]),
        };

        let gateway = UnifiedGateway::new(&settings).unwrap();

        // test updating model at runtime
        gateway.set_default_model("deepseek-v4-pro".to_string());
        assert_eq!(gateway.get_model("default"), "deepseek-v4-pro");

        // test updating API key at runtime
        gateway.set_api_key("sk-new-key".to_string());
        assert_eq!(*gateway.api_key.read().unwrap(), "sk-new-key");

        // test updating base URL at runtime
        gateway.set_base_url("https://api.new-endpoint.com".to_string());
        assert_eq!(
            *gateway.base_url.read().unwrap(),
            "https://api.new-endpoint.com"
        );
    }

    #[test]
    fn test_build_responses_body_converts_messages() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Let me check.".to_string(),
                name: None,
                tool_calls: Some(vec![ToolCallPayload {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "file_read".to_string(),
                        arguments: r#"{"path":"/tmp/a.txt"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "found 42".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ];

        let body = UnifiedGateway::build_responses_body(
            "deepseek-v4-flash",
            &messages,
            Some(0.7),
            Some(2048),
            None,
            None,
            false,
        );

        assert_eq!(body["model"], "deepseek-v4-flash");
        assert_eq!(body["stream"], false);
        assert_eq!(body["instructions"], "You are a helpful assistant.");
        assert_eq!(body["max_output_tokens"], 2048);
        assert_eq!(body["temperature"].as_f64().unwrap() as f32, 0.7);
        assert!(body.get("max_tokens").is_none());

        let input = body["input"].as_array().expect("input is an array");
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "Hi");
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["name"], "file_read");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_1");
        assert_eq!(input[3]["output"], "found 42");
    }

    #[test]
    fn test_parse_responses_response_extracts_output() {
        let json = serde_json::json!({
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "I think..."}]},
                {"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "The answer is 42", "annotations": []}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "search", "arguments": "{\"q\":\"rust\"}"}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
            "store": false
        });

        let response = UnifiedGateway::parse_responses_response(&json).unwrap();
        assert_eq!(response.id.as_deref(), Some("resp_1"));
        assert_eq!(response.choices.len(), 1);
        let message = &response.choices[0].message;
        assert_eq!(message.content.as_deref(), Some("The answer is 42"));
        assert_eq!(message.reasoning_content.as_deref(), Some("I think..."));
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("tool_calls"));

        let calls = message.tool_calls.as_ref().expect("tool calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_9");
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[0].function.arguments, r#"{"q":"rust"}"#);

        let usage = response.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 25);
        assert_eq!(usage.total_tokens, 125);
    }

    #[test]
    fn test_parse_responses_response_plain_text() {
        let json = serde_json::json!({
            "id": "resp_2",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello world"}]}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });
        let response = UnifiedGateway::parse_responses_response(&json).unwrap();
        assert_eq!(response.choices[0].message.content.as_deref(), Some("hello world"));
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(response.choices[0].message.tool_calls.is_none());
    }

    #[test]
    fn test_responses_routing_only_for_capable_models() {
        let settings = GatewaySettings {
            base_url: "https://api.deepseek.com".to_string(),
            api_key: "sk-test".to_string(),
            default_model: "deepseek-v4-flash".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            retry_base_ms: 500,
            use_responses_api: true,
            model_mapping: HashMap::new(),
        };
        let gateway = UnifiedGateway::new(&settings).unwrap();

        assert!(gateway.should_use_responses_api("deepseek-v4-flash"));
        assert!(!gateway.should_use_responses_api("deepseek-v4-pro"));
        assert!(!gateway.should_use_responses_api("gpt-4o"));

        gateway.set_use_responses_api(false);
        assert!(!gateway.should_use_responses_api("deepseek-v4-flash"));
    }

    #[test]
    fn test_parse_tool_choice_handles_string_and_object() {
        assert_eq!(UnifiedGateway::parse_tool_choice("auto"), serde_json::json!("auto"));
        assert_eq!(UnifiedGateway::parse_tool_choice("none"), serde_json::json!("none"));

        let obj = UnifiedGateway::parse_tool_choice(r#"{"type":"function","name":"get_weather"}"#);
        assert!(obj.is_object());
        assert_eq!(obj["type"], "function");
        assert_eq!(obj["name"], "get_weather");
    }

    #[test]
    fn test_convert_responses_tools_flattens_function() {
        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }),
            serde_json::json!({"type": "web_search"}),
        ];
        let converted = UnifiedGateway::convert_responses_tools(tools);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0]["type"], "function");
        assert_eq!(converted[0]["name"], "get_weather");
        assert_eq!(converted[0]["description"], "Get weather");
        assert!(converted[0].get("function").is_none());
        assert_eq!(converted[1]["type"], "web_search");
    }

    // Live integration tests hit a real external provider. The provider's
    // streaming connection can be dropped transiently (rate-limit, network
    // blip) even when HTTP status is 2xx, which surfaces as a transport decode
    // error mid-body. Retry a bounded number of times with backoff before
    // failing so the suite stays deterministic; logic assertions are unchanged.
    async fn collect_stream_retrying(
        gateway: &UnifiedGateway,
        model: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<crate::llm::StreamResponse, String> {
        let mut last_err: Option<String> = None;
        for attempt in 0..3 {
            match gateway
                .stream_chat_with_params(model, messages.clone(), None, None, None, None)
                .await
            {
                Ok(mut stream) => match stream.collect_all().await {
                    Ok(resp) => return Ok(resp),
                    Err(e) => last_err = Some(format!("stream decode: {e}")),
                },
                Err(e) => last_err = Some(format!("request: {e}")),
            }
            let backoff = Duration::from_millis(400 * (1 << attempt));
            tokio::time::sleep(backoff).await;
        }
        Err(last_err.unwrap_or_else(|| "unknown error".to_string()))
    }

    fn live_gateway() -> Option<UnifiedGateway> {
        let api_key = std::env::var("DEEPSEEK_API_KEY").ok()?;
        let settings = GatewaySettings {
            base_url: "https://api.deepseek.com".to_string(),
            api_key,
            default_model: "deepseek-v4-flash".to_string(),
            timeout_seconds: 60,
            max_retries: 0,
            retry_base_ms: 500,
            use_responses_api: true,
            model_mapping: HashMap::new(),
        };
        Some(UnifiedGateway::new(&settings).unwrap())
    }

    #[tokio::test]
    async fn test_responses_api_live_non_streaming() {
        let Some(gateway) = live_gateway() else {
            eprintln!("skipping live test: DEEPSEEK_API_KEY not set");
            return;
        };
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Reply with exactly: PONG".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "ping".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        let response = gateway
            .chat_with_model("deepseek-v4-flash", messages)
            .await
            .unwrap();
        assert_eq!(
            response.choices[0].message.content.as_deref().map(str::trim),
            Some("PONG")
        );
        assert!(response.usage.is_some());
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn test_responses_api_live_streaming() {
        let Some(gateway) = live_gateway() else {
            eprintln!("skipping live test: DEEPSEEK_API_KEY not set");
            return;
        };
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Count from 1 to 3, one number per line.".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let response = collect_stream_retrying(&gateway, "deepseek-v4-flash", messages)
            .await
            .expect("live streaming should succeed within bounded retries");
        assert!(response.content.contains('1'));
        assert!(response.content.contains('3'));
        assert!(!response.content.is_empty());
    }

    #[tokio::test]
    async fn test_responses_api_live_tool_call() {
        let Some(gateway) = live_gateway() else {
            eprintln!("skipping live test: DEEPSEEK_API_KEY not set");
            return;
        };
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "You have exactly one available action: call get_weather for Shanghai. You must call it."
                .to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        })];
        let response = gateway
            .chat_with_params(
                "deepseek-v4-flash",
                messages,
                None,
                None,
                Some(tools),
                Some("auto"),
            )
            .await
            .unwrap();
        let tool_calls = response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("expected a function call");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        let args: Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
        assert!(args.get("city").is_some());
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    }
}
