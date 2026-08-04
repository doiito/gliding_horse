use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info, warn};

use crate::gateway::unified_gateway::{ChatMessage, UnifiedGateway};
use crate::memory::hyperspace_store::HyperspaceStore;
use crate::skill_graph::discovery::SkillDiscoveryEngine;
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::*;
use crate::tools::skill_registry::{SkillMeta, SkillRegistry};
use crate::CoreError;

const SKILL_CREATOR_SYSTEM_PROMPT: &str = r#"You are a Skill definition generator. Based on the user's natural language description, generate a Skill definition that conforms to the following JSON-LD specification.

## Output Format

You must output a valid JSON object containing the following fields:

```json
{
  "name": "skill_name (lowercase, underscore-separated)",
  "description": "One sentence describing what this Skill does",
  "category": "Category: file|network|ai|execution|validation|data|meta|system",
  "security_level": "Security level: low|normal|high|critical",
  "allowed_roles": ["PA","DA","CA","AA"],
  "input_schema": {
    "type": "object",
    "properties": { ... },
    "required": [...]
  },
  "output_schema": {
    "type": "object",
    "properties": { ... }
  },
  "steps": [
    { "step_id": "step_1", "order": 1, "action": "Step description" }
  ],
  "tags": ["tag1", "tag2"],
  "what": "What this Skill does",
  "why": "Why this Skill is needed",
  "approach": "Implementation approach overview"
}
```

## Rules

1. name must be lowercase letters and underscores, e.g. `web_search`, `code_review`
2. input_schema follows JSON Schema specification, required fields must be listed
3. Each input property must have type and description
4. output_schema must contain key fields of the execution result
5. steps must contain at least 1 step, describing the execution flow
6. security_level rules:
   - low: read-only operations, no side effects
   - normal: involves network requests or file reads
   - high: involves file writes or data modification
   - critical: involves code execution or system operations
7. allowed_roles set according to security level:
   - low: ["PA","DA","CA","AA"]
   - normal: ["PA","DA","CA","AA"]
   - high: ["DA","CA"]
   - critical: ["DA"]
8. tags must contain at least 2 tags
9. what/why/approach must be concise and clear

Output only JSON, do not output other content."#;

const MARKDOWN_CONVERTER_SYSTEM_PROMPT: &str = r#"You are a Markdown Skill converter. Convert Markdown-format Skill descriptions provided by the user into JSON-LD Skill definitions conforming to this system's specification.

## Input Format

A Markdown Skill typically contains:
- Title (# Skill Name)
- Description paragraphs
- Parameter list (table or list)
- Tool/dependency description
- Step description

## Output Format

Same JSON format as Skill Creator:

```json
{
  "name": "skill_name",
  "description": "Description",
  "category": "Category",
  "security_level": "Security level",
  "allowed_roles": [...],
  "input_schema": { ... },
  "output_schema": { ... },
  "steps": [ ... ],
  "tags": [...],
  "what": "...",
  "why": "...",
  "approach": "..."
}
```

## Conversion Rules

1. Extract skill name from Markdown title
2. Extract what/why from description paragraphs
3. Build input_schema from parameter list
4. Determine security_level and category from tool list
5. Build steps from step descriptions
6. If Markdown mentions tools (such as read_file, write_file, bash, etc.), annotate them in tags
7. Ensure the generated JSON is complete and valid

Output only JSON, do not output other content."#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSkillRequest {
    pub description: String,
    pub skill_name_hint: Option<String>,
    pub category_hint: Option<String>,
    pub security_level_override: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertMarkdownRequest {
    pub markdown_content: String,
    pub source_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatedSkill {
    pub skill_iri: String,
    pub name: String,
    pub graph_node: SkillGraphNode,
    pub registry_meta: SkillMeta,
    pub json_ld: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCreatorConfig {
    pub output_dir: PathBuf,
    pub auto_register: bool,
    pub validate_before_register: bool,
    pub default_security_level: String,
}

impl Default for SkillCreatorConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./skills"),
            auto_register: true,
            validate_before_register: true,
            default_security_level: "normal".to_string(),
        }
    }
}

pub struct SkillCreator {
    gateway: Arc<UnifiedGateway>,
    graph_store: Arc<SkillGraphStore>,
    skill_registry: Arc<SkillRegistry>,
    vector_store: Option<Arc<HyperspaceStore>>,
    config: SkillCreatorConfig,
}

impl SkillCreator {
    pub fn new(
        gateway: Arc<UnifiedGateway>,
        graph_store: Arc<SkillGraphStore>,
        skill_registry: Arc<SkillRegistry>,
        config: SkillCreatorConfig,
    ) -> Self {
        Self {
            gateway,
            graph_store,
            skill_registry,
            vector_store: None,
            config,
        }
    }

    /// Attach the application-owned semantic store for immediate indexing of
    /// created skills.  Without it creation still registers the skill graph,
    /// and a later application startup rebuild can populate the index.
    pub fn with_vector_store(mut self, vector_store: Arc<HyperspaceStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    async fn index_created_skill(&self, created: &CreatedSkill) -> Result<(), CoreError> {
        if let Some(vector_store) = &self.vector_store {
            SkillDiscoveryEngine::new(self.graph_store.clone())
                .with_vector_store(vector_store.clone())
                .index_skill(&created.graph_node)
                .await?;
        }
        Ok(())
    }

    async fn index_created_skill_or_compensate(
        &self,
        created: &CreatedSkill,
    ) -> Result<(), CoreError> {
        if let Err(index_error) = self.index_created_skill(created).await {
            if !self.config.auto_register {
                return Err(index_error);
            }
            let registry_removed = self.skill_registry.unregister_skill(&created.skill_iri);
            let graph_result = self.graph_store.remove_skill(&created.skill_iri);
            if !registry_removed || graph_result.is_err() {
                return Err(CoreError::StorageError {
                    message: format!(
                        "Skill vector indexing failed ({index_error}); compensation failed (registry_removed={registry_removed}, graph_error={})",
                        graph_result.err().map(|error| error.to_string()).unwrap_or_else(|| "none".to_string()),
                    ),
                });
            }
            return Err(index_error);
        }
        Ok(())
    }

    fn persist_definition_artifact(&self, created: &CreatedSkill) {
        if self.config.output_dir.exists()
            || std::fs::create_dir_all(&self.config.output_dir).is_ok()
        {
            let skill_dir = self.config.output_dir.join(&created.name);
            if std::fs::create_dir_all(&skill_dir).is_ok() {
                let jsonld_path = skill_dir.join("skill.jsonld");
                let jsonld_str = serde_json::to_string_pretty(&created.json_ld).unwrap_or_default();
                if let Err(error) = std::fs::write(&jsonld_path, jsonld_str) {
                    warn!(path = %jsonld_path.display(), %error, "Failed to write skill.jsonld");
                } else {
                    debug!(path = %jsonld_path.display(), "skill.jsonld written");
                }
            }
        }
    }

    pub async fn create_from_description(
        &self,
        request: CreateSkillRequest,
    ) -> Result<CreatedSkill, CoreError> {
        info!(description = %request.description, "Starting Skill creation from natural language description");

        let user_message = if let Some(hint) = &request.skill_name_hint {
            format!(
                "Please create a Skill: {}\n\nSuggested name: {}\nSuggested category: {}",
                request.description,
                hint,
                request.category_hint.as_deref().unwrap_or("auto-detect")
            )
        } else {
            format!("Please create a Skill: {}", request.description)
        };

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: SKILL_CREATOR_SYSTEM_PROMPT.to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_message,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let response = self
            .gateway
            .chat(messages)
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("LLM call failed: {}", e),
            })?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| CoreError::Internal {
                message: "LLM returned empty content".to_string(),
            })?;

        let skill_def = self.parse_llm_response(&content)?;
        let created =
            self.build_and_register(skill_def, request.security_level_override.as_deref())?;
        self.index_created_skill_or_compensate(&created).await?;
        self.persist_definition_artifact(&created);

        info!(skill_iri = %created.skill_iri, "Skill creation complete");
        Ok(created)
    }

    pub async fn convert_from_markdown(
        &self,
        request: ConvertMarkdownRequest,
    ) -> Result<CreatedSkill, CoreError> {
        info!(source = ?request.source_path, "Starting Skill conversion from Markdown");

        let user_message = format!(
            "Please convert the following Markdown-format Skill description to a JSON-LD Skill definition:\n\n```markdown\n{}\n```",
            request.markdown_content
        );

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: MARKDOWN_CONVERTER_SYSTEM_PROMPT.to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_message,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let response = self
            .gateway
            .chat(messages)
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("LLM call failed: {}", e),
            })?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| CoreError::Internal {
                message: "LLM returned empty content".to_string(),
            })?;

        let skill_def = self.parse_llm_response(&content)?;
        let created = self.build_and_register(skill_def, None)?;
        self.index_created_skill_or_compensate(&created).await?;
        self.persist_definition_artifact(&created);

        info!(skill_iri = %created.skill_iri, "Markdown Skill conversion complete");
        Ok(created)
    }

    pub fn convert_markdown_static(content: &str) -> Result<SkillDefinition, CoreError> {
        let mut def = SkillDefinition::default();
        let mut in_code_block = false;
        let mut current_section = String::new();

        for line in content.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("```") {
                in_code_block = !in_code_block;
                continue;
            }

            if in_code_block {
                continue;
            }

            if trimmed.starts_with("# ") {
                def.name = trimmed
                    .trim_start_matches("# ")
                    .trim()
                    .to_lowercase()
                    .replace(' ', "_")
                    .replace(|c: char| !c.is_alphanumeric() && c != '_', "");
                def.description = format!("{} Skill", trimmed.trim_start_matches("# ").trim());
            } else if trimmed.starts_with("## ") {
                current_section = trimmed.trim_start_matches("## ").trim().to_lowercase();
            } else if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
                let item = trimmed
                    .trim_start_matches("- ")
                    .trim_start_matches("* ")
                    .trim();
                if current_section.contains("parameter") {
                    if let Some((name, desc)) =
                        item.split_once(':').or_else(|| item.split_once(' '))
                    {
                        def.input_properties.push(InputProperty {
                            name: name.trim().to_string(),
                            prop_type: "string".to_string(),
                            description: desc.trim().to_string(),
                            required: !desc.contains("optional"),
                        });
                    }
                } else if current_section.contains("step") {
                    let order = (def.steps.len() + 1) as u32;
                    def.steps
                        .push(SkillStep::new(&format!("step_{}", order), order, item));
                } else if current_section.contains("tag") {
                    def.tags.push(item.to_string());
                }
            } else if !trimmed.is_empty() && def.what.is_empty() {
                def.what = trimmed.to_string();
            }
        }

        if def.name.is_empty() {
            def.name = "unnamed_skill".to_string();
        }
        if def.what.is_empty() {
            def.what = def.description.clone();
        }
        if def.why.is_empty() {
            def.why = format!("Auto-converted from Markdown: {}", def.description);
        }
        if def.approach.is_empty() {
            def.approach = "Execute according to steps".to_string();
        }
        if def.tags.is_empty() {
            def.tags.push("converted".to_string());
            def.tags.push("markdown".to_string());
        }

        Ok(def)
    }

    pub fn create_from_definition(
        &self,
        def: SkillDefinition,
        security_level_override: Option<&str>,
    ) -> Result<CreatedSkill, CoreError> {
        let created = self.build_and_register(def, security_level_override)?;
        self.persist_definition_artifact(&created);
        Ok(created)
    }

    fn parse_llm_response(&self, content: &str) -> Result<SkillDefinition, CoreError> {
        let json_str = content
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let parsed: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| CoreError::ValidationFailed {
                message: format!("Failed to parse JSON returned by LLM: {}", e),
            })?;

        let name = parsed["name"]
            .as_str()
            .unwrap_or("unnamed_skill")
            .to_lowercase()
            .replace(' ', "_")
            .replace(|c: char| !c.is_alphanumeric() && c != '_', "");

        let description = parsed["description"].as_str().unwrap_or("").to_string();
        let category = parsed["category"].as_str().unwrap_or("system").to_string();
        let security_level = parsed["security_level"]
            .as_str()
            .unwrap_or("normal")
            .to_string();

        let allowed_roles = parsed["allowed_roles"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec!["DA".to_string()]);

        let input_schema = parsed
            .get("input_schema")
            .cloned()
            .unwrap_or(json!({"type":"object","properties":{},"required":[]}));

        let output_schema = parsed
            .get("output_schema")
            .cloned()
            .unwrap_or(json!({"type":"object","properties":{}}));

        let steps = parsed["steps"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .map(|(i, s)| {
                        SkillStep::new(
                            s["step_id"].as_str().unwrap_or(&format!("step_{}", i + 1)),
                            (i + 1) as u32,
                            s["action"].as_str().unwrap_or(""),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        let tags = parsed["tags"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec!["auto-generated".to_string()]);

        let what = parsed["what"].as_str().unwrap_or(&description).to_string();
        let why = parsed["why"].as_str().unwrap_or("").to_string();
        let approach = parsed["approach"].as_str().unwrap_or("").to_string();

        Ok(SkillDefinition {
            name,
            description,
            category,
            security_level,
            allowed_roles,
            input_schema,
            output_schema,
            steps,
            tags,
            what,
            why,
            approach,
            input_properties: vec![],
        })
    }

    fn build_and_register(
        &self,
        def: SkillDefinition,
        security_level_override: Option<&str>,
    ) -> Result<CreatedSkill, CoreError> {
        let security_level = security_level_override
            .unwrap_or(&def.security_level)
            .trim()
            .to_ascii_lowercase();

        if self.config.validate_before_register {
            self.validate_definition(&def, &security_level)?;
        }

        let operation_risk = match security_level.as_str() {
            "low" => 0.1,
            "normal" => 0.3,
            "high" => 0.7,
            "critical" => 0.9,
            // Reached only when validation is explicitly disabled. Preserve a
            // conservative risk value rather than treating an unknown string
            // as safe.
            _ => 0.9,
        };
        let security_info = SkillSecurityInfo::new(SkillSource::UserDefined)
            .with_trust_level(TrustLevel::Untrusted)
            .with_risk_score(operation_risk);

        let skill_iri = format!("iri://skills/{}", def.name);
        if self.config.auto_register
            && (self.graph_store.get_skill(&skill_iri).is_some()
                || self.skill_registry.get_skill(&skill_iri).is_some())
        {
            return Err(CoreError::ValidationFailed {
                message: format!("Skill already exists and cannot be overwritten: {skill_iri}"),
            });
        }

        let mut w2h = Skill5W2H::new(&def.what, &def.why);
        w2h.how.approach = def.approach.clone();
        if let Some(primary_role) = def.allowed_roles.first() {
            w2h.who.role_name = primary_role.clone();
            w2h.who.required_agent_role = Some(primary_role.clone());
        }
        w2h.when.applicable_phases = match def.category.as_str() {
            "validation" => vec!["Check".to_string()],
            "meta" | "system" => vec!["Plan".to_string(), "Do".to_string()],
            _ => vec!["Do".to_string()],
        };

        let mut graph_node = SkillGraphNode::new(&skill_iri, &def.name, &def.description)
            .with_node_type(SkillNodeType::Atomic)
            .with_5w2h(w2h)
            .with_security_info(security_info);

        for tag in &def.tags {
            graph_node = graph_node.with_tag(tag);
        }

        if !def.steps.is_empty() {
            let mut content = SkillContent {
                summary: def.description.clone(),
                steps: def.steps.clone(),
                validation: None,
            };
            if !def.steps.is_empty() {
                content.validation = Some(SkillValidation {
                    method: "output_schema_check".to_string(),
                    success_condition: "Output conforms to output_schema definition".to_string(),
                });
            }
            graph_node = graph_node.with_content(content);
        }

        let json_ld = graph_node.to_json_ld();

        let registry_meta = SkillMeta {
            skill_iri: skill_iri.clone(),
            name: def.name.clone(),
            description: def.description.clone(),
            version: "1.0.0".to_string(),
            category: def.category.clone(),
            security_level: security_level.clone(),
            allowed_roles: def.allowed_roles.clone(),
            input_schema: def.input_schema.clone(),
            output_schema: def.output_schema.clone(),
            compiled_template: Self::build_compiled_template(&def.input_schema),
            signature: None,
            signature_algorithm: None,
            input_mapping: Self::build_input_mapping(&def.name, &def.input_schema)
                .into_iter()
                .collect(),
            output_mapping: Self::build_output_mapping(&def.name, &def.output_schema)
                .into_iter()
                .collect(),
            // A generated definition is useful to graph/discovery workflows,
            // but no ToolExecutor handler is generated in this path. Keep it
            // out of role-based executable skill lists until an explicit,
            // governed activation flow installs a handler.
            skill_types: vec!["definition_only".to_string()],
        };

        if self.config.auto_register {
            self.graph_store.register_skill(graph_node.clone())?;
            self.skill_registry.register_skill(registry_meta.clone());
            debug!(skill_iri = %skill_iri, "Skill registered to GraphStore and SkillRegistry");
        }

        Ok(CreatedSkill {
            skill_iri,
            name: def.name.clone(),
            graph_node,
            registry_meta,
            json_ld,
        })
    }

    /// Validate the static, untrusted input boundary before it reaches the
    /// Graph/Registry. This deliberately does not claim to be a full graph
    /// verifier: relationship conflicts and execution permissions require a
    /// typed governance proposal and are checked elsewhere.
    fn validate_definition(
        &self,
        def: &SkillDefinition,
        security_level: &str,
    ) -> Result<(), CoreError> {
        if def.name.trim().is_empty() {
            return Err(CoreError::ValidationFailed {
                message: "Skill name must not be empty".to_string(),
            });
        }
        if def.description.trim().is_empty()
            || def.what.trim().is_empty()
            || def.why.trim().is_empty()
            || def.approach.trim().is_empty()
        {
            return Err(CoreError::ValidationFailed {
                message: "Skill description and 5W2H what/why/approach must not be empty"
                    .to_string(),
            });
        }
        if !matches!(security_level, "low" | "normal" | "high" | "critical") {
            return Err(CoreError::ValidationFailed {
                message: format!("Unsupported security level '{security_level}'"),
            });
        }
        if def.allowed_roles.is_empty()
            || def
                .allowed_roles
                .iter()
                .any(|role| !matches!(role.as_str(), "PA" | "DA" | "CA" | "AA"))
        {
            return Err(CoreError::ValidationFailed {
                message: "Skill must declare one or more supported roles (PA, DA, CA, AA)"
                    .to_string(),
            });
        }
        if !def.input_schema.is_object() || !def.output_schema.is_object() {
            return Err(CoreError::ValidationFailed {
                message: "Input and output schemas must be JSON objects".to_string(),
            });
        }
        if def.steps.iter().any(|step| step.action.trim().is_empty()) {
            return Err(CoreError::ValidationFailed {
                message: "Skill steps must have a non-empty action".to_string(),
            });
        }
        Ok(())
    }

    fn build_compiled_template(input_schema: &serde_json::Value) -> String {
        let props = input_schema.get("properties").and_then(|p| p.as_object());

        let Some(props) = props else {
            return "{}".to_string();
        };

        let mut template = serde_json::Map::new();
        for (key, value) in props {
            let default = value.get("default");
            if let Some(d) = default {
                template.insert(key.clone(), d.clone());
            } else {
                let prop_type = value
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("string");
                match prop_type {
                    "string" => {
                        template.insert(key.clone(), json!("___"));
                    }
                    "number" | "integer" => {
                        template.insert(key.clone(), json!(0));
                    }
                    "boolean" => {
                        template.insert(key.clone(), json!(false));
                    }
                    "array" => {
                        template.insert(key.clone(), json!([]));
                    }
                    "object" => {
                        template.insert(key.clone(), json!({}));
                    }
                    _ => {
                        template.insert(key.clone(), json!("___"));
                    }
                }
            }
        }

        serde_json::Value::Object(template).to_string()
    }

    fn build_input_mapping(
        skill_name: &str,
        input_schema: &serde_json::Value,
    ) -> Vec<(String, String)> {
        let props = input_schema.get("properties").and_then(|p| p.as_object());

        let Some(props) = props else {
            return Vec::new();
        };

        props
            .keys()
            .map(|k| (k.clone(), format!("iri://schema/{}/{}", skill_name, k)))
            .collect()
    }

    fn build_output_mapping(
        skill_name: &str,
        output_schema: &serde_json::Value,
    ) -> Vec<(String, String)> {
        let props = output_schema.get("properties").and_then(|p| p.as_object());

        let Some(props) = props else {
            return Vec::new();
        };

        props
            .keys()
            .map(|k| {
                (
                    k.clone(),
                    format!("iri://schema/{}/output/{}", skill_name, k),
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    pub category: String,
    #[serde(default = "default_security_level")]
    pub security_level: String,
    #[serde(default = "default_allowed_roles")]
    pub allowed_roles: Vec<String>,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
    #[serde(default)]
    pub steps: Vec<SkillStep>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub what: String,
    #[serde(default)]
    pub why: String,
    #[serde(default)]
    pub approach: String,
    #[serde(default)]
    pub input_properties: Vec<InputProperty>,
}

fn default_security_level() -> String {
    "normal".to_string()
}
fn default_allowed_roles() -> Vec<String> {
    vec!["DA".to_string()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputProperty {
    pub name: String,
    pub prop_type: String,
    pub description: String,
    pub required: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embedding_service::EmbeddingService;

    struct FailingEmbeddingService;

    #[async_trait::async_trait]
    impl EmbeddingService for FailingEmbeddingService {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, String> {
            Err("intentional embedding failure".to_string())
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    fn test_gateway() -> Arc<UnifiedGateway> {
        Arc::new(
            UnifiedGateway::new(&crate::config::GatewaySettings {
                base_url: "http://localhost:3000".to_string(),
                api_key: "test".to_string(),
                default_model: "test".to_string(),
                timeout_seconds: 30,
                max_retries: 1,
                retry_base_ms: 500,
                use_responses_api: false,
                model_mapping: Default::default(),
            })
            .unwrap(),
        )
    }

    fn valid_definition(name: &str) -> SkillDefinition {
        SkillDefinition {
            name: name.to_string(),
            description: "Test skill".to_string(),
            category: "system".to_string(),
            security_level: "low".to_string(),
            allowed_roles: vec!["DA".to_string()],
            input_schema: json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}),
            output_schema: json!({"type":"object","properties":{"result":{"type":"string"}}}),
            steps: vec![SkillStep::new("step_1", 1, "Execute test")],
            tags: vec!["test".to_string()],
            what: "Test".to_string(),
            why: "Verify".to_string(),
            approach: "Direct execution".to_string(),
            input_properties: vec![],
        }
    }

    #[test]
    fn test_convert_markdown_static() {
        let markdown = r#"# Web Search

Search the internet for information

## Parameters

- query: Search keywords
- max_results: Maximum results (optional)

## Steps

- Build search URL
- Send HTTP request
- Parse returned results
- Format output

## Tags

- search
- web"#;

        let def = SkillCreator::convert_markdown_static(markdown).unwrap();

        assert_eq!(def.name, "web_search");
        assert!(def.description.contains("Web Search"));
        assert_eq!(def.input_properties.len(), 2);
        assert!(def.input_properties[0].required);
        assert!(!def.input_properties[1].required);
        assert_eq!(def.steps.len(), 4);
        assert!(def.tags.contains(&"search".to_string()));
    }

    #[test]
    fn test_build_compiled_template() {
        let input_schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        });

        let template = SkillCreator::build_compiled_template(&input_schema);
        let parsed: serde_json::Value = serde_json::from_str(&template).unwrap();

        assert_eq!(parsed["query"], "___");
        assert_eq!(parsed["limit"], 10);
    }

    #[test]
    fn test_parse_llm_response() {
        let config = SkillCreatorConfig::default();
        let gateway = Arc::new(
            UnifiedGateway::new(&crate::config::GatewaySettings {
                base_url: "http://localhost:3000".to_string(),
                api_key: "test".to_string(),
                default_model: "test".to_string(),
                timeout_seconds: 30,
                max_retries: 1,
                retry_base_ms: 500,
                use_responses_api: false,
                model_mapping: Default::default(),
            })
            .unwrap(),
        );
        let graph_store = Arc::new(SkillGraphStore::new());
        let registry = Arc::new(SkillRegistry::new());
        let creator = SkillCreator::new(gateway, graph_store, registry, config);

        let llm_response = r#"```json
{
  "name": "code_review",
  "description": "Review code quality and provide improvement suggestions",
  "category": "validation",
  "security_level": "low",
  "allowed_roles": ["CA"],
  "input_schema": {
    "type": "object",
    "properties": {
      "code": {"type": "string", "description": "Code to review"},
      "language": {"type": "string", "description": "Programming language"}
    },
    "required": ["code"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "issues": {"type": "array"},
      "score": {"type": "number"}
    }
  },
  "steps": [
    {"step_id": "step_1", "order": 1, "action": "Parse code structure"},
    {"step_id": "step_2", "order": 2, "action": "Check code conventions"},
    {"step_id": "step_3", "order": 3, "action": "Generate review report"}
  ],
  "tags": ["code", "review", "quality"],
  "what": "Review code quality",
  "why": "Ensure code meets conventions and best practices",
  "approach": "Static analysis + rule matching"
}
```"#;

        let def = creator.parse_llm_response(llm_response).unwrap();

        assert_eq!(def.name, "code_review");
        assert_eq!(def.category, "validation");
        assert_eq!(def.security_level, "low");
        assert_eq!(def.allowed_roles, vec!["CA"]);
        assert_eq!(def.steps.len(), 3);
        assert_eq!(def.tags.len(), 3);
    }

    #[test]
    fn test_create_from_definition() {
        let config = SkillCreatorConfig::default();
        let gateway = Arc::new(
            UnifiedGateway::new(&crate::config::GatewaySettings {
                base_url: "http://localhost:3000".to_string(),
                api_key: "test".to_string(),
                default_model: "test".to_string(),
                timeout_seconds: 30,
                max_retries: 1,
                retry_base_ms: 500,
                use_responses_api: false,
                model_mapping: Default::default(),
            })
            .unwrap(),
        );
        let graph_store = Arc::new(SkillGraphStore::new());
        let registry = Arc::new(SkillRegistry::new());
        let creator = SkillCreator::new(gateway, graph_store, registry, config);

        let def = SkillDefinition {
            name: "test_skill".to_string(),
            description: "Test skill".to_string(),
            category: "system".to_string(),
            security_level: "low".to_string(),
            allowed_roles: vec!["DA".to_string()],
            input_schema: json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}),
            output_schema: json!({"type":"object","properties":{"result":{"type":"string"}}}),
            steps: vec![SkillStep::new("step_1", 1, "Execute test")],
            tags: vec!["test".to_string()],
            what: "Test".to_string(),
            why: "Verify".to_string(),
            approach: "Direct execution".to_string(),
            input_properties: vec![],
        };

        let created = creator.create_from_definition(def, None).unwrap();

        assert_eq!(created.skill_iri, "iri://skills/test_skill");
        assert_eq!(created.name, "test_skill");
        assert!(created.json_ld.get("@id").is_some());
        let security = created.graph_node.security_info.as_ref().unwrap();
        assert_eq!(security.source, SkillSource::UserDefined);
        assert_eq!(security.trust_level, TrustLevel::Untrusted);
        assert_eq!(security.risk_score, 0.1);
    }

    #[test]
    fn validation_rejects_unknown_security_level_before_registration() {
        let config = SkillCreatorConfig::default();
        let gateway = Arc::new(
            UnifiedGateway::new(&crate::config::GatewaySettings {
                base_url: "http://localhost:3000".to_string(),
                api_key: "test".to_string(),
                default_model: "test".to_string(),
                timeout_seconds: 30,
                max_retries: 1,
                retry_base_ms: 500,
                use_responses_api: false,
                model_mapping: Default::default(),
            })
            .unwrap(),
        );
        let graph_store = Arc::new(SkillGraphStore::new());
        let registry = Arc::new(SkillRegistry::new());
        let creator = SkillCreator::new(gateway, graph_store.clone(), registry.clone(), config);
        let def = SkillDefinition {
            name: "unsafe_skill".to_string(),
            description: "Should not be registered".to_string(),
            category: "system".to_string(),
            security_level: "unbounded".to_string(),
            allowed_roles: vec!["DA".to_string()],
            input_schema: json!({"type":"object"}),
            output_schema: json!({"type":"object"}),
            steps: vec![],
            tags: vec![],
            what: "Reject invalid security".to_string(),
            why: "Keep registration governed".to_string(),
            approach: "Validate first".to_string(),
            input_properties: vec![],
        };

        assert!(matches!(
            creator.create_from_definition(def, None),
            Err(CoreError::ValidationFailed { .. })
        ));
        assert!(graph_store.get_skill("iri://skills/unsafe_skill").is_none());
        assert!(registry.get_skill("iri://skills/unsafe_skill").is_none());
    }

    #[test]
    fn definition_creation_rejects_existing_iri_without_overwriting_it() {
        let output = tempfile::tempdir().unwrap();
        let mut config = SkillCreatorConfig::default();
        config.output_dir = output.path().join("skills");
        let graph = Arc::new(SkillGraphStore::new());
        let registry = Arc::new(SkillRegistry::new());
        let creator = SkillCreator::new(test_gateway(), graph.clone(), registry.clone(), config);
        creator
            .create_from_definition(valid_definition("unique_skill"), None)
            .unwrap();

        let mut replacement = valid_definition("unique_skill");
        replacement.description = "must not overwrite".to_string();
        assert!(matches!(
            creator.create_from_definition(replacement, None),
            Err(CoreError::ValidationFailed { .. })
        ));
        assert_eq!(
            graph
                .get_skill("iri://skills/unique_skill")
                .unwrap()
                .description,
            "Test skill"
        );
        assert_eq!(
            registry
                .get_skill("iri://skills/unique_skill")
                .unwrap()
                .description,
            "Test skill"
        );
    }

    #[test]
    fn vector_index_failure_compensates_graph_and_registry_registration() {
        let output = tempfile::tempdir().unwrap();
        let vector_dir = tempfile::tempdir().unwrap();
        let vector_store = Arc::new(
            HyperspaceStore::open(vector_dir.path(), Arc::new(FailingEmbeddingService)).unwrap(),
        );
        let mut config = SkillCreatorConfig::default();
        config.output_dir = output.path().join("skills");
        let graph = Arc::new(SkillGraphStore::new());
        let registry = Arc::new(SkillRegistry::new());
        let creator = SkillCreator::new(test_gateway(), graph.clone(), registry.clone(), config)
            .with_vector_store(vector_store);
        let created = creator
            .build_and_register(valid_definition("compensated_skill"), None)
            .unwrap();

        let runtime = tokio::runtime::Runtime::new().unwrap();
        assert!(runtime
            .block_on(creator.index_created_skill_or_compensate(&created))
            .is_err());
        assert!(graph.get_skill("iri://skills/compensated_skill").is_none());
        assert!(registry
            .get_skill("iri://skills/compensated_skill")
            .is_none());
        assert!(!output
            .path()
            .join("skills/compensated_skill/skill.jsonld")
            .exists());
    }
}
