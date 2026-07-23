use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use super::ontology::OntologyManager;
use super::rdf_mapper::RdfMapper;
use super::store::KnowledgeGraphStore;
use super::types::{LLMExtractionOutput, RdfMappingResult};

pub struct KnowledgeExtractor {
    ontology: OntologyManager,
    store: KnowledgeGraphStore,
    api_url: String,
    api_key: String,
    model: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequestMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequestBody {
    model: String,
    messages: Vec<ChatRequestMessage>,
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

impl KnowledgeExtractor {
    pub fn new(
        ontology: OntologyManager,
        store: KnowledgeGraphStore,
        api_url: String,
        api_key: String,
        model: String,
    ) -> Self {
        Self {
            ontology,
            store,
            api_url,
            api_key,
            model,
        }
    }

    pub fn build_extraction_prompt(text: &str, vocabulary: &str) -> String {
        format!(
            r#"You are a knowledge graph extraction expert. Extract entities and relations from the given text.

{vocabulary}

## Output Format Requirements

Output strictly in the following JSON format, without any additional text, explanations, or markdown code block markers:

{{
  "nodes": [
    {{
      "id": "Unique entity identifier (English/pinyin, no spaces)",
      "node_type": "Use the entity type IRI listed above",
      "label": "Entity name",
      "properties": {{}}
    }}
  ],
  "edges": [
    {{
      "source": "Source entity id",
      "target": "Target entity id",
      "relation": "Use the relation IRI listed above",
      "properties": {{}}
    }}
  ]
}}

## Rules
1. All fields in nodes and edges must not be empty
2. Extract at least one entity (node)
3. type and relation fields must use the IRIs listed above
4. id must use English or pinyin, not Chinese
5. properties can be an empty object {{}}

## Text to Extract

{text}"#
        )
    }

    async fn call_llm(&self, prompt: &str) -> Result<String, String> {
        let client = Client::builder()
            .build()
            .map_err(|e| format!("failed to create HTTP client: {}", e))?;

        let (url, body) = self.chat_request(prompt);

        debug!(model = %self.model, url = %url, "calling LLM API for knowledge extraction");

        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("LLM API request failed: {}", e))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("LLM API returned error ({}): {}", status, text));
        }

        let response_text = resp
            .text()
            .await
            .map_err(|e| format!("failed to read LLM response: {}", e))?;

        let chat_resp: ChatCompletionResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                format!(
                    "failed to parse LLM response JSON: {} (raw response: {})",
                    e,
                    truncate_str(&response_text, 200)
                )
            })?;

        let choice = chat_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| "no choices in LLM response".to_string())?;

        choice
            .message
            .content
            .ok_or_else(|| "LLM response content is empty".to_string())
    }

    fn chat_request(&self, prompt: &str) -> (String, ChatRequestBody) {
        (
            chat_completions_url(&self.api_url),
            ChatRequestBody {
                model: self.model.clone(),
                messages: vec![ChatRequestMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }],
                temperature: 0.1,
            },
        )
    }

    pub fn validate_extraction(json_str: &str) -> Result<LLMExtractionOutput, String> {
        let cleaned = clean_json_response(json_str);

        let output: LLMExtractionOutput = serde_json::from_str(&cleaned).map_err(|e| {
            format!(
                "JSON parse failed: {} (first 200 chars: {})",
                e,
                truncate_str(&cleaned, 200)
            )
        })?;

        if output.nodes.is_empty() {
            return Err("extraction result requires at least one entity (node)".to_string());
        }

        for (i, node) in output.nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                return Err(format!("entity {} id is empty", i + 1));
            }
            if node.node_type.trim().is_empty() {
                return Err(format!("entity {} type is empty", i + 1));
            }
            if node.label.trim().is_empty() {
                return Err(format!("entity {} label is empty", i + 1));
            }
        }

        for (i, edge) in output.edges.iter().enumerate() {
            if edge.source.trim().is_empty() {
                return Err(format!("edge {} source is empty", i + 1));
            }
            if edge.target.trim().is_empty() {
                return Err(format!("edge {} target is empty", i + 1));
            }
            if edge.relation.trim().is_empty() {
                return Err(format!("edge {} relation is empty", i + 1));
            }
        }

        Ok(output)
    }

    /// Extract knowledge asynchronously.
    ///
    /// This method is called from async tool handlers. It must not create or
    /// block a Tokio runtime internally, otherwise calls from an existing
    /// runtime panic with "Cannot start a runtime from within a runtime".
    pub async fn extract(
        &self,
        text: &str,
        domain: Option<&str>,
    ) -> Result<RdfMappingResult, String> {
        let vocab = self.ontology.get_vocabulary(domain);
        let vocabulary = self.ontology.format_vocabulary_for_prompt(&vocab);
        let base_prompt = Self::build_extraction_prompt(text, &vocabulary);

        let mut last_error = String::new();
        let mut current_prompt = base_prompt;

        for attempt in 1..=3 {
            debug!(attempt, "knowledge extraction attempt");

            let llm_result = self.call_llm(&current_prompt).await;

            let raw_response = match llm_result {
                Ok(resp) => resp,
                Err(e) => {
                    warn!(attempt, error = %e, "LLM API call failed");
                    last_error = e;
                    if attempt < 3 {
                        current_prompt = format!(
                            "{}\n\n---\nLast call failed, error: {}\nPlease retry.",
                            Self::build_extraction_prompt(text, &vocabulary),
                            last_error
                        );
                    }
                    continue;
                }
            };

            match Self::validate_extraction(&raw_response) {
                Ok(extraction) => {
                    let graph = self.store.default_graph();
                    let result = RdfMapper::map_extraction(&extraction, graph);

                    self.store.write_quads(&result.quads, graph)?;

                    debug!(
                        entities = result.entity_count,
                        relations = result.relation_count,
                        quads = result.quads.len(),
                        "knowledge extraction complete and written to store"
                    );

                    return Ok(result);
                }
                Err(e) => {
                    warn!(attempt, error = %e, "extraction validation failed");
                    last_error = e;
                    if attempt < 3 {
                        current_prompt = format!(
                            "{}\n\n---\nLast extraction validation failed, error: {}\nLLM raw output:\n{}\nPlease correct and re-output.",
                            Self::build_extraction_prompt(text, &vocabulary),
                            last_error,
                            truncate_str(&raw_response, 500)
                        );
                    }
                }
            }
        }

        Err(format!(
            "knowledge extraction failed after 3 attempts, last error: {}",
            last_error
        ))
    }

    pub fn ontology(&self) -> &OntologyManager {
        &self.ontology
    }

    pub fn store(&self) -> &KnowledgeGraphStore {
        &self.store
    }
}

/// Normalize an OpenAI-compatible base URL to its chat-completions endpoint.
///
/// Callers may provide a server root, a `/v1` base, or an already complete
/// endpoint. The official default (`https://api.openai.com/v1`) must retain
/// the `/v1` segment.
fn chat_completions_url(api_url: &str) -> String {
    let base = api_url.trim().trim_end_matches('/');
    if base.ends_with("/chat/completions") {
        base.to_string()
    } else if base.ends_with("/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    }
}

fn clean_json_response(input: &str) -> String {
    let trimmed = input.trim();

    if trimmed.starts_with("```json") {
        let without_start = trimmed.trim_start_matches("```json").trim();
        if let Some(pos) = without_start.rfind("```") {
            return without_start[..pos].trim().to_string();
        }
        return without_start.trim().to_string();
    }

    if trimmed.starts_with("```") {
        let without_start = trimmed.trim_start_matches("```").trim();
        if let Some(pos) = without_start.rfind("```") {
            return without_start[..pos].trim().to_string();
        }
        return without_start.trim().to_string();
    }

    if let Some(start) = trimmed.find('{') {
        let mut depth = 0i32;
        for (i, c) in trimmed[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return trimmed[start..start + i + 1].to_string();
                    }
                }
                _ => {}
            }
        }
    }

    trimmed.to_string()
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...(truncated)", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_clean_json_response_plain() {
        let input = r#"{"nodes": [], "edges": []}"#;
        assert_eq!(clean_json_response(input), input);
    }

    #[test]
    fn test_clean_json_response_markdown_block() {
        let input = "```json\n{\"nodes\": [], \"edges\": []}\n```";
        assert_eq!(clean_json_response(input), r#"{"nodes": [], "edges": []}"#);
    }

    #[test]
    fn test_clean_json_response_with_prefix() {
        let input = "Here is the result:\n{\"nodes\": [], \"edges\": []}\nDone.";
        assert_eq!(clean_json_response(input), r#"{"nodes": [], "edges": []}"#);
    }

    #[test]
    fn test_validate_extraction_empty_nodes() {
        let json = r#"{"nodes": [], "edges": []}"#;
        let result = KnowledgeExtractor::validate_extraction(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires at least one entity"));
    }

    #[test]
    fn test_validate_extraction_empty_node_id() {
        let json = r#"{"nodes": [{"id": "", "node_type": "T", "label": "L", "properties": {}}], "edges": []}"#;
        let result = KnowledgeExtractor::validate_extraction(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("id is empty"));
    }

    #[test]
    fn test_validate_extraction_empty_edge_source() {
        let json = r#"{"nodes": [{"id": "a", "node_type": "T", "label": "L", "properties": {}}], "edges": [{"source": "", "target": "b", "relation": "R", "properties": {}}]}"#;
        let result = KnowledgeExtractor::validate_extraction(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("source is empty"));
    }

    #[test]
    fn test_validate_extraction_valid() {
        let json = r#"{"nodes": [{"id": "a", "node_type": "https://agentos.ontology/core/Person", "label": "Alice", "properties": {}}], "edges": [{"source": "a", "target": "b", "relation": "https://agentos.ontology/business/worksFor", "properties": {}}]}"#;
        let result = KnowledgeExtractor::validate_extraction(json);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.nodes.len(), 1);
        assert_eq!(output.edges.len(), 1);
    }

    #[test]
    fn test_validate_extraction_invalid_json() {
        let json = "not json at all";
        let result = KnowledgeExtractor::validate_extraction(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("JSON parse failed"));
    }

    #[test]
    fn test_build_extraction_prompt() {
        let vocab = "## Available entity types\n- IRI: https://agentos.ontology/core/Person | name: Person | Represents a person";
        let prompt = KnowledgeExtractor::build_extraction_prompt("test text", vocab);
        assert!(prompt.contains("knowledge graph extraction expert"));
        assert!(prompt.contains("https://agentos.ontology/core/Person"));
        assert!(prompt.contains("test text"));
        assert!(prompt.contains("nodes"));
        assert!(prompt.contains("edges"));
    }

    #[test]
    fn test_chat_completions_url_normalization() {
        assert_eq!(
            chat_completions_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://proxy.example/"),
            "https://proxy.example/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://proxy.example/v1/chat/completions"),
            "https://proxy.example/v1/chat/completions"
        );
    }

    #[test]
    fn test_extract_request_uses_v1_endpoint_without_network_fixture() {
        let extractor = KnowledgeExtractor::new(
            OntologyManager::new(),
            KnowledgeGraphStore::new().unwrap(),
            "https://proxy.example/v1".to_string(),
            "test-key".to_string(),
            "test-model".to_string(),
        );
        let (url, body) = extractor.chat_request("Alice is a person");
        assert_eq!(url, "https://proxy.example/v1/chat/completions");
        assert_eq!(body.model, "test-model");
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].content, "Alice is a person");
    }

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_long() {
        let result = truncate_str("abcdefghij", 5);
        assert_eq!(result, "abcde...(truncated)");
    }

    #[test]
    fn test_clean_json_response_nested() {
        let input = r#"prefix {"nodes": [{"id": "a"}], "edges": [{"source": "a"}]} suffix"#;
        let cleaned = clean_json_response(input);
        let parsed: HashMap<String, serde_json::Value> = serde_json::from_str(&cleaned).unwrap();
        assert!(parsed.contains_key("nodes"));
    }
}
