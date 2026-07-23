use std::sync::Arc;

use serde_json::Value;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::batch::error::BatchError;
use crate::batch::prompt::DynamicPromptEngine;
use crate::batch::types::{
    BatchAgentConfig, BatchMetrics, ExtractionResult, PromptContext, WindowEntry,
};
use crate::batch::validator::OutputValidator;
use crate::gateway::unified_gateway::{ChatMessage, UnifiedGateway};

pub struct ExtractorPipeline {
    gateway: Arc<UnifiedGateway>,
    prompt_engine: Arc<DynamicPromptEngine>,
    validator: OutputValidator,
    metrics: Arc<std::sync::Mutex<BatchMetrics>>,
}

impl ExtractorPipeline {
    pub fn new(
        gateway: Arc<UnifiedGateway>,
        prompt_engine: Arc<DynamicPromptEngine>,
        metrics: Arc<std::sync::Mutex<BatchMetrics>>,
    ) -> Self {
        Self {
            gateway,
            prompt_engine,
            validator: OutputValidator::new(),
            metrics,
        }
    }

    pub async fn extract(
        &self,
        config: &BatchAgentConfig,
        window_entries: Vec<WindowEntry>,
        context: &PromptContext,
    ) -> Result<ExtractionResult, BatchError> {
        let batch_id = format!("batch_{}", Uuid::new_v4().hyphenated());
        if window_entries.is_empty() {
            return Err(BatchError::Internal {
                message: "Cannot extract from empty window".to_string(),
            });
        }

        let system_prompt = self
            .prompt_engine
            .build_system_prompt(config, &window_entries, context)
            .await?;

        let extraction = self
            .call_llm_with_retry(config, &system_prompt, &window_entries)
            .await?;

        let result = ExtractionResult {
            batch_id,
            extracted_at: chrono::Utc::now(),
            entities: extraction.entities,
            relations: extraction.relations,
            intent: extraction.intent,
            key_decisions: extraction.key_decisions,
            context_summary: extraction.context_summary,
            llm_calls: extraction.llm_calls,
            tokens_consumed: extraction.tokens_consumed,
            confidence_scores: extraction.confidence_scores,
            raw_response: extraction.raw_response,
        };

        // Record metrics
        if let Ok(mut m) = self.metrics.lock() {
            m.record_success(
                result.tokens_consumed,
                result.entities.len(),
                result.relations.len(),
            );
        }

        info!(
            batch = %result.batch_id,
            entities = %result.entities.len(),
            relations = %result.relations.len(),
            intent = ?result.intent.as_ref().map(|i| &i.intent_type),
            tokens = %result.tokens_consumed,
            "Extraction completed"
        );

        Ok(result)
    }

    async fn call_llm_with_retry(
        &self,
        config: &BatchAgentConfig,
        system_prompt: &str,
        window_entries: &[WindowEntry],
    ) -> Result<ExtractionResult, BatchError> {
        let max_retries = config.max_retries.max(1);
        let mut last_error = String::new();

        for attempt in 1..=max_retries {
            let user_content = window_entries
                .iter()
                .map(|e| {
                    format!(
                        "[{}] {}: {}",
                        e.role,
                        e.timestamp.format("%H:%M:%S"),
                        e.content
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            let messages = vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_content,
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ];

            debug!(
                attempt = %attempt,
                max_retries = %max_retries,
                "LLM extraction call"
            );

            let model = config.model.as_deref().unwrap_or("default");
            let response = self
                .gateway
                .chat_with_params(
                    model,
                    messages,
                    config.temperature,
                    None, // max_tokens
                    None, // tools
                    None, // tool_choice
                )
                .await;

            match response {
                Ok(reply) => {
                    let content = reply.choices[0].message.content.as_deref().unwrap_or("");
                    match self.parse_and_validate(config, content) {
                        Ok(result) => return Ok(result),
                        Err(e) => {
                            last_error = format!("Validation failed: {}", e);
                            warn!(
                                attempt = %attempt,
                                error = %last_error,
                                "Extraction validation failed, retrying"
                            );
                        }
                    }
                }
                Err(e) => {
                    last_error = format!("LLM call failed: {}", e);
                    error!(
                        attempt = %attempt,
                        error = %last_error,
                        "LLM extraction failed"
                    );
                    if attempt < max_retries {
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            500 * attempt as u64,
                        ))
                        .await;
                    }
                }
            }
        }

        // Record failure in metrics
        if let Ok(mut m) = self.metrics.lock() {
            m.record_failure();
        }

        Err(BatchError::ExtractionFailed {
            attempts: max_retries,
            message: last_error,
        })
    }

    fn parse_and_validate(
        &self,
        config: &BatchAgentConfig,
        content: &str,
    ) -> Result<ExtractionResult, BatchError> {
        // Try to extract JSON from the response (it might be wrapped in markdown code blocks)
        let json_str = extract_json(content).ok_or_else(|| BatchError::ValidationFailed {
            message: "No valid JSON found in LLM response".to_string(),
        })?;

        let json: Value =
            serde_json::from_str(&json_str).map_err(|e| BatchError::ValidationFailed {
                message: format!("Invalid JSON output: {}", e),
            })?;

        let mut result = self.validator.validate_llm_json_output(&json, config)?;
        result.raw_response = Some(content.to_string());

        Ok(result)
    }
}

/// Extract the first JSON object or array from text, handling markdown code blocks.
fn extract_json(text: &str) -> Option<String> {
    // Try to find JSON in ```json ... ``` blocks first
    if let Some(start) = text.find("```json") {
        let after_start = &text[start + 7..];
        if let Some(end) = after_start.find("```") {
            let candidate = after_start[..end].trim();
            if candidate.starts_with('{') || candidate.starts_with('[') {
                return Some(candidate.to_string());
            }
        }
    }

    // Try to find a standalone JSON object
    if let Some(start) = text.find('{') {
        let mut depth = 0;
        let mut in_string = false;
        let mut escaped = false;
        for (i, c) in text[start..].char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match c {
                '\\' if in_string => escaped = true,
                '"' => in_string = !in_string,
                '{' if !in_string => depth += 1,
                '}' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[start..=start + i].to_string());
                    }
                }
                _ => {}
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_from_code_block() {
        let text = r#"Some text before
```json
{"entities": [{"name": "test", "entity_type": "onto:BusinessEntity", "confidence": 0.9}], "relations": [], "context_summary": "test"}
```
Some text after"#;
        let result = extract_json(text);
        assert!(result.is_some());
        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(parsed.get("entities").is_some());
    }

    #[test]
    fn test_extract_json_standalone() {
        let text = r#"The extracted data is:
{"entities": [{"name": "Project X", "entity_type": "onto:BusinessDomain", "confidence": 0.95}], "relations": [], "context_summary": "Discussion about Project X"}
That's all."#;
        let result = extract_json(text);
        assert!(result.is_some());
        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(parsed["entities"][0]["name"].as_str().unwrap(), "Project X");
    }

    #[test]
    fn test_extract_json_no_json() {
        let text = "This text contains no JSON at all.";
        assert!(extract_json(text).is_none());
    }

    #[test]
    fn test_extract_json_nested() {
        let text = "Result: {\"outer\": {\"inner\": [1, 2, 3]}, \"done\": true}";
        let result = extract_json(text);
        assert!(result.is_some());
        let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(parsed.get("outer").is_some());
        assert!(parsed.get("done").and_then(|v| v.as_bool()).unwrap());
    }
}
