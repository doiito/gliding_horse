use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::config::settings::GatewaySettings;
use crate::gateway::{RateLimiter, ResponseCache};
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

/// Transport metadata for one completed gateway request.  It deliberately
/// excludes request/response bodies so callers can correlate execution traces
/// without duplicating potentially sensitive model payloads in logs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatewayCallMetadata {
    pub endpoint: String,
    pub attempts: u32,
    pub cache_hit: bool,
    pub latency_ms: u64,
    pub http_status: Option<u16>,
    pub provider_response_id: Option<String>,
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
    rate_limiter: RateLimiter,
    response_cache: ResponseCache,
    response_cache_enabled: AtomicBool,
    /// Explicit endpoint selection. When enabled, requests are sent to
    /// `{base_url}/v1/responses`; model capability is owned by the configured
    /// provider rather than a kernel-side model-name allowlist.
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
            rate_limiter: RateLimiter::default(),
            response_cache: ResponseCache::default(),
            response_cache_enabled: AtomicBool::new(false),
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
        self.chat_with_params_traced(model, messages, temperature, max_tokens, tools, tool_choice)
            .await
            .map(|(response, _)| response)
    }

    /// Same request as [`Self::chat_with_params`], with transport metadata for
    /// durable task tracing.  The metadata never contains model payloads.
    pub async fn chat_with_params_traced(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
    ) -> Result<(ChatCompletionResponse, GatewayCallMetadata), CoreError> {
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
            return self.send_responses_request_traced(&url, body).await;
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
        self.send_request_traced(&url, body).await
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
        self.send_request_traced(url, body)
            .await
            .map(|(response, _)| response)
    }

    async fn send_request_traced(
        &self,
        url: &str,
        body: Value,
    ) -> Result<(ChatCompletionResponse, GatewayCallMetadata), CoreError> {
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
        self.send_responses_request_traced(url, body)
            .await
            .map(|(response, _)| response)
    }

    async fn send_responses_request_traced(
        &self,
        url: &str,
        body: Value,
    ) -> Result<(ChatCompletionResponse, GatewayCallMetadata), CoreError> {
        self.send_with_retry(url, body, Self::parse_responses_response)
            .await
    }

    async fn send_with_retry<F>(
        &self,
        url: &str,
        body: Value,
        parse: F,
    ) -> Result<(ChatCompletionResponse, GatewayCallMetadata), CoreError>
    where
        F: Fn(&Value) -> Result<ChatCompletionResponse, CoreError>,
    {
        let started_at = Instant::now();
        let endpoint = if url.ends_with("/v1/responses") {
            "responses"
        } else {
            "chat_completions"
        }
        .to_string();
        let mut last_error = None;
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("default");
        self.rate_limiter.acquire(model, 1).await;

        let cache_key = (self.response_cache_enabled.load(Ordering::Relaxed)
            && body.get("tools").is_none())
        .then(|| ResponseCache::build_request_key(url, &body));
        if let Some(cached) = cache_key
            .as_deref()
            .and_then(|key| self.response_cache.get(key))
        {
            return parse(&cached).map(|response| {
                let provider_response_id = response.id.clone();
                (
                    response,
                    GatewayCallMetadata {
                        endpoint,
                        attempts: 0,
                        cache_hit: true,
                        latency_ms: started_at.elapsed().as_millis().min(u128::from(u64::MAX))
                            as u64,
                        http_status: None,
                        provider_response_id,
                    },
                )
            });
        }
        let mut retry_delay = None;

        for attempt in 0..=self.max_retries {
            if let Some(delay) = retry_delay.take() {
                tokio::time::sleep(delay).await;
                debug!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    "Retrying LLM API call"
                );
            }

            let req = self
                .client
                .post(url)
                .header(
                    "Authorization",
                    format!("Bearer {}", self.api_key.read().unwrap()),
                )
                .header("Content-Type", "application/json")
                .json(&body);

            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let response_text = resp.text().await.map_err(|e| {
                            warn!(error = %e, "Failed to read LLM response body");
                            CoreError::Internal {
                                message: format!("Failed to read response body: {e}"),
                            }
                        })?;
                        let json: Value = match serde_json::from_str(&response_text) {
                            Ok(v) => v,
                            Err(e) => {
                                warn!(error = %e, response_len = response_text.len(), "Failed to parse LLM response");
                                return Err(CoreError::Internal {
                                    message: format!(
                                        "Failed to parse LLM response: {} (response length: {})",
                                        e,
                                        response_text.len()
                                    ),
                                });
                            }
                        };
                        match parse(&json) {
                            Ok(result) => {
                                info!(
                                    model = %body["model"],
                                    usage = ?result.usage.as_ref().map(|u| u.total_tokens),
                                    "LLM API call successful"
                                );
                                if let Some(key) = &cache_key {
                                    self.response_cache.set(key, json);
                                }
                                let provider_response_id = result.id.clone();
                                return Ok((
                                    result,
                                    GatewayCallMetadata {
                                        endpoint,
                                        attempts: attempt.saturating_add(1),
                                        cache_hit: false,
                                        latency_ms: started_at
                                            .elapsed()
                                            .as_millis()
                                            .min(u128::from(u64::MAX))
                                            as u64,
                                        http_status: Some(status.as_u16()),
                                        provider_response_id,
                                    },
                                ));
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to convert LLM response");
                                return Err(e);
                            }
                        }
                    } else {
                        let retry_after = Self::retry_after(resp.headers());
                        let retryable = Self::is_retryable_status(status);
                        warn!(status = %status, retryable, "LLM API returned an error status");
                        last_error = Some(CoreError::Internal {
                            message: format!("LLM API error ({status})"),
                        });
                        if !retryable || attempt == self.max_retries {
                            break;
                        }
                        retry_delay =
                            Some(retry_after.unwrap_or_else(|| self.retry_backoff(attempt)));
                    }
                }
                Err(e) => {
                    warn!(error = %e, "LLM API request failed");
                    last_error = Some(CoreError::Internal {
                        message: format!("LLM API request failed: {}", e),
                    });
                    if attempt < self.max_retries {
                        retry_delay = Some(self.retry_backoff(attempt));
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| CoreError::Internal {
            message: "LLM API call failed after all retries".to_string(),
        }))
    }

    fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
        let value = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
        if let Ok(seconds) = value.trim().parse::<u64>() {
            return Some(Duration::from_secs(seconds.min(300)));
        }
        let timestamp = chrono::DateTime::parse_from_rfc2822(value).ok()?;
        let delay = timestamp.signed_duration_since(chrono::Utc::now());
        (delay.num_milliseconds() > 0)
            .then(|| Duration::from_millis(delay.num_milliseconds().min(300_000) as u64))
    }

    fn is_retryable_status(status: reqwest::StatusCode) -> bool {
        status == reqwest::StatusCode::REQUEST_TIMEOUT
            || status == reqwest::StatusCode::TOO_MANY_REQUESTS
            || status.is_server_error()
    }

    fn retry_backoff(&self, attempt: u32) -> Duration {
        const MAX_BACKOFF_MS: u64 = 30_000;
        let multiplier = 1u64.checked_shl(attempt.min(20)).unwrap_or(u64::MAX);
        let base = self
            .retry_base_ms
            .saturating_mul(multiplier)
            .min(MAX_BACKOFF_MS);
        let jitter_bound = (base / 4).max(1);
        let jitter = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| u64::from(duration.subsec_nanos()) % jitter_bound)
            .unwrap_or(0);
        Duration::from_millis(base.saturating_add(jitter).min(MAX_BACKOFF_MS))
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

    /// Enable deterministic response caching explicitly. It remains disabled
    /// by default and requests containing tools are never cached.
    pub fn set_response_cache_enabled(&self, enabled: bool) {
        self.response_cache_enabled
            .store(enabled, Ordering::Relaxed);
        if !enabled {
            self.response_cache.clear();
        }
    }

    fn should_use_responses_api(&self, model: &str) -> bool {
        let _ = model;
        *self.use_responses_api.read().unwrap()
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
                            arguments: item.get("input").map(|v| v.to_string()).unwrap_or_default(),
                        },
                    });
                }
                _ => {}
            }
        }

        // DeepSeek's `max_output_tokens` is a shared budget for reasoning +
        // final output. When reasoning consumes the entire budget, the response
        // is marked incomplete with no `message` block at all — surface that
        // explicitly instead of silently returning `content: None`, which
        // downstream callers would misreport as a generic "No response content".
        if text_parts.is_empty() && tool_calls.is_empty() {
            let status = json.get("status").and_then(|v| v.as_str());
            let reason = json
                .get("incomplete_details")
                .and_then(|d| d.get("reason"))
                .and_then(|v| v.as_str());
            if status == Some("incomplete") && reason == Some("max_output_tokens") {
                return Err(CoreError::Internal {
                    message: format!(
                        "Responses API response incomplete: max_output_tokens reached with \
                         reasoning consuming the full budget; no final text was produced"
                    ),
                });
            }
        }

        let finish_reason = if tool_calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        let usage = json.get("usage").map(|u| Usage {
            prompt_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            completion_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
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
            "stream_options": {"include_usage": true},
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
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("default");
        self.rate_limiter.acquire(model, 1).await;
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
            return Err(CoreError::Internal {
                message: format!("Stream API error ({status})"),
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
    fn retry_policy_includes_transient_client_statuses_only() {
        assert!(UnifiedGateway::is_retryable_status(
            reqwest::StatusCode::REQUEST_TIMEOUT
        ));
        assert!(UnifiedGateway::is_retryable_status(
            reqwest::StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(UnifiedGateway::is_retryable_status(
            reqwest::StatusCode::BAD_GATEWAY
        ));
        assert!(!UnifiedGateway::is_retryable_status(
            reqwest::StatusCode::BAD_REQUEST
        ));
        assert!(!UnifiedGateway::is_retryable_status(
            reqwest::StatusCode::UNAUTHORIZED
        ));
    }

    #[test]
    fn retry_after_seconds_is_bounded() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "999".parse().unwrap());
        assert_eq!(
            UnifiedGateway::retry_after(&headers),
            Some(Duration::from_secs(300))
        );
    }

    #[test]
    fn gateway_debug_redacts_api_key() {
        let settings = GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "super-secret".to_string(),
            default_model: "model".to_string(),
            timeout_seconds: 30,
            max_retries: 1,
            retry_base_ms: 10,
            use_responses_api: false,
            model_mapping: HashMap::new(),
        };
        let debug = format!("{settings:?}");
        assert!(!debug.contains("super-secret"));
        assert!(debug.contains("[REDACTED]"));
    }

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
        assert_eq!(
            response.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );

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
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("hello world")
        );
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(response.choices[0].message.tool_calls.is_none());
    }

    #[test]
    fn test_responses_routing_follows_explicit_endpoint_setting() {
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
        assert!(gateway.should_use_responses_api("deepseek-v4-pro"));
        assert!(gateway.should_use_responses_api("gpt-4o"));

        gateway.set_use_responses_api(false);
        assert!(!gateway.should_use_responses_api("deepseek-v4-flash"));
    }

    #[test]
    fn test_parse_tool_choice_handles_string_and_object() {
        assert_eq!(
            UnifiedGateway::parse_tool_choice("auto"),
            serde_json::json!("auto")
        );
        assert_eq!(
            UnifiedGateway::parse_tool_choice("none"),
            serde_json::json!("none")
        );

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
    #[cfg(feature = "live-tests")]
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

    #[cfg(feature = "live-tests")]
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

    #[cfg(feature = "live-tests")]
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
            response.choices[0]
                .message
                .content
                .as_deref()
                .map(str::trim),
            Some("PONG")
        );
        assert!(response.usage.is_some());
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[cfg(feature = "live-tests")]
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

    #[cfg(feature = "live-tests")]
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
        assert_eq!(
            response.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }
}
