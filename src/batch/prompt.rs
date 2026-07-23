use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::Value;
use tracing::warn;

use crate::batch::error::BatchError;
use crate::batch::types::{BatchAgentConfig, PromptContext, PromptSource, WindowEntry};
use crate::batch::vocabulary::{EntityTypeConfig, IntentTypeConfig, RelationTypeConfig};
use crate::core::system_prompt::{SystemPromptBuilder, SystemPromptRegion};
use crate::memory::l0_store::L0Store;
use crate::templates::TemplateEngine;

pub struct CachedPrompt {
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub source: PromptSource,
}

pub struct DynamicPromptEngine {
    template_engine: Arc<TemplateEngine>,
    #[allow(dead_code)]
    prompt_builder: SystemPromptBuilder,
    #[allow(dead_code)]
    l0_store: Option<Arc<L0Store>>,
    #[allow(dead_code)]
    prompt_cache: HashMap<String, CachedPrompt>,
    #[allow(dead_code)]
    cache_ttl: chrono::Duration,
}

impl DynamicPromptEngine {
    pub fn new(template_engine: Arc<TemplateEngine>, l0_store: Option<Arc<L0Store>>) -> Self {
        Self {
            template_engine,
            prompt_builder: SystemPromptBuilder::new(),
            l0_store,
            prompt_cache: HashMap::new(),
            cache_ttl: chrono::Duration::minutes(30),
        }
    }

    pub async fn build_system_prompt(
        &self,
        config: &BatchAgentConfig,
        window_content: &[WindowEntry],
        context: &PromptContext,
    ) -> Result<String, BatchError> {
        let source = self.decide_source(config, context).await;

        let _base_prompt = match &source {
            PromptSource::TemplateFile | PromptSource::TemplateEngine => {
                self.load_from_template(config, context)?
            }
            PromptSource::LlmGenerated | PromptSource::HybridWithTemplate => {
                self.generate_prompt(config, context).await?
            }
        };

        // Build the final prompt with regions
        let mut builder = SystemPromptBuilder::new();

        builder.set_region(
            SystemPromptRegion::RoleDefinition,
            format!(
                "You are a specialized knowledge extraction agent: {}.\n{}",
                config.name,
                if config.description.is_empty() {
                    "Extract structured knowledge from user conversations.".to_string()
                } else {
                    config.description.clone()
                }
            ),
        );

        // Use ExtractionPrompt region as task description
        builder.set_region(
            SystemPromptRegion::ExtractionPrompt,
            format!(
                "Analyze the following conversation window and extract:\n\
                 1. Business entities and concepts mentioned\n\
                 2. Relationships between entities\n\
                 3. User intent and key decisions\n\
                 4. Context summary\n\n\
                 Conversation window ({} messages):\n{}",
                window_content.len(),
                window_content
                    .iter()
                    .map(|e| format!(
                        "[{}] {}: {}",
                        e.timestamp.format("%H:%M:%S"),
                        e.role,
                        e.content
                    ))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        );

        // Controlled vocabulary as part of FiveW2HConstraints
        let vocab = Self::format_vocabulary_for_prompt(
            &config.entity_types,
            &config.relation_types,
            &config.intent_types,
        );
        builder.set_region(SystemPromptRegion::FiveW2HConstraints, vocab);

        builder.set_region(
            SystemPromptRegion::OutputFormat,
            r#"Output must be valid JSON with this exact structure:
{
  \"entities\": [
    {\"name\": \"...\", \"entity_type\": \"from vocabulary\", \"description\": \"...\", \"aliases\": [...], \"confidence\": 0.0-1.0}
  ],
  \"relations\": [
    {\"from\": \"entity_name\", \"relation\": \"from vocabulary\", \"to\": \"entity_name\", \"properties\": {}, \"confidence\": 0.0-1.0}
  ],
  \"intent\": {\"intent_type\": \"from vocabulary\", \"confidence\": 0.0-1.0, \"details\": {}},
  \"key_decisions\": [
    {\"decision\": \"...\", \"rationale\": \"...\", \"evidence\": [...], \"confidence\": \"high|medium|low\"}
  ],
  \"context_summary\": \"One-sentence summary of the conversation window\"
}"#
                .to_string(),
        );

        // Injected context as EmphasizedConstraints
        let injected = self.format_injected_context(context);
        if !injected.is_empty() {
            builder.set_region(SystemPromptRegion::EmphasizedConstraints, injected);
        }

        // Extraction prompt (task description)
        builder.set_region(
            SystemPromptRegion::ExtractionPrompt,
            format!(
                "Analyze the following conversation window and extract:\n\
                 1. Business entities and concepts mentioned\n\
                 2. Relationships between entities\n\
                 3. User intent and key decisions\n\
                 4. Context summary\n\n\
                 Conversation window ({} messages):\n{}",
                window_content.len(),
                window_content
                    .iter()
                    .map(|e| format!(
                        "[{}] {}: {}",
                        e.timestamp.format("%H:%M:%S"),
                        e.role,
                        e.content
                    ))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        );

        // Output format (replace existing)
        builder.set_region(
            SystemPromptRegion::OutputFormat,
            r#"Output must be valid JSON with this exact structure:
{
  \"entities\": [
    {\"name\": \"...\", \"entity_type\": \"from vocabulary\", \"description\": \"...\", \"aliases\": [...], \"confidence\": 0.0-1.0}
  ],
  \"relations\": [
    {\"from\": \"entity_name\", \"relation\": \"from vocabulary\", \"to\": \"entity_name\", \"properties\": {}, \"confidence\": 0.0-1.0}
  ],
  \"intent\": {\"intent_type\": \"from vocabulary\", \"confidence\": 0.0-1.0, \"details\": {}},
  \"key_decisions\": [
    {\"decision\": \"...\", \"rationale\": \"...\", \"evidence\": [...], \"confidence\": \"high|medium|low\"}
  ],
  \"context_summary\": \"One-sentence summary of the conversation window\"
}"#
                .to_string(),
        );

        Ok(builder.build())
    }

    pub fn load_from_template(
        &self,
        config: &BatchAgentConfig,
        context: &PromptContext,
    ) -> Result<String, BatchError> {
        let template_name = config.prompt_template_name.as_deref().unwrap_or("default");

        // Look up template using direct key matching template discovery
        let template_key = format!("prompts/batch/{}", template_name);
        let template = self
            .template_engine
            .get_template(&template_key)
            .or_else(|| {
                let alt_key = format!("batch/{}", template_name);
                self.template_engine.get_template(&alt_key)
            });

        let template_content = match template {
            Some(t) => t.content,
            None => {
                warn!(
                    template = %template_key,
                    "Template not found, using fallback"
                );
                return Ok(self.build_fallback_prompt(config));
            }
        };

        let mut vars = HashMap::new();
        vars.insert(
            "controlled_vocabulary".to_string(),
            Value::String(Self::format_vocabulary_for_prompt(
                &config.entity_types,
                &config.relation_types,
                &config.intent_types,
            )),
        );
        vars.insert(
            "injected_context".to_string(),
            Value::String(self.format_injected_context(context)),
        );
        vars.insert(
            "user_reminders".to_string(),
            Value::String(self.format_user_reminders(context)),
        );
        for (k, v) in &config.prompt_params {
            vars.insert(k.clone(), Value::String(v.clone()));
        }

        Ok(TemplateEngine::render_string(&template_content, &vars))
    }

    pub async fn generate_prompt(
        &self,
        config: &BatchAgentConfig,
        _context: &PromptContext,
    ) -> Result<String, BatchError> {
        // Static generation based on config — no LLM call needed for the prompt template itself.
        // In a full implementation, this would call an LLM to generate an optimised prompt
        // for the specific domain and context. For now, we use a structured template.
        Ok(self.build_fallback_prompt(config))
    }

    pub async fn decide_source(
        &self,
        config: &BatchAgentConfig,
        context: &PromptContext,
    ) -> PromptSource {
        match &config.prompt_source {
            PromptSource::TemplateFile | PromptSource::TemplateEngine => {
                config.prompt_source.clone()
            }
            PromptSource::LlmGenerated => PromptSource::LlmGenerated,
            PromptSource::HybridWithTemplate => {
                let has_template = config
                    .prompt_template_name
                    .as_ref()
                    .map(|n| {
                        self.template_engine
                            .get_template(&format!("batch/{}", n))
                            .is_some()
                    })
                    .unwrap_or(false);

                if !has_template {
                    return PromptSource::LlmGenerated;
                }

                // Evaluate factors for decision
                let domain_maturity = self.evaluate_domain_maturity(&config.business_domain);
                let context_complexity = context.total_tokens();
                let history_success_rate = 0.9; // Default optimistic value

                if domain_maturity > 0.7 && history_success_rate > 0.85 && context_complexity < 2000
                {
                    PromptSource::TemplateFile
                } else {
                    PromptSource::LlmGenerated
                }
            }
        }
    }

    pub fn evaluate_domain_maturity(&self, domain: &str) -> f64 {
        if domain.is_empty() || domain == "default" {
            return 0.3;
        }
        // Check if the domain has entity types defined in existing ontologies
        // Simple heuristic: known domains get higher maturity
        let known_domains = [
            "business_design",
            "software_engineering",
            "user_context",
            "ecommerce",
            "healthcare",
            "finance",
        ];
        if known_domains.contains(&domain) {
            0.8
        } else {
            0.4
        }
    }

    fn build_fallback_prompt(&self, config: &BatchAgentConfig) -> String {
        format!(
            r#"You are a knowledge extraction agent: {}
Domain: {}

Extract entities, relations, and intent from the user conversation window.

Entity types: {}
Relation types: {}
Intent types: {}

Output strict JSON only.
"#,
            config.name,
            config.business_domain,
            config.entity_types.effective_types().join(", "),
            config.relation_types.effective_types().join(", "),
            config.intent_types.effective_types().join(", "),
        )
    }

    fn format_vocabulary_for_prompt(
        entity_types: &EntityTypeConfig,
        relation_types: &RelationTypeConfig,
        intent_types: &IntentTypeConfig,
    ) -> String {
        let mut result = String::new();

        result.push_str("## Entity types (select from these only)\n");
        for et in entity_types.effective_types() {
            result.push_str(&format!("- {}\n", et));
        }

        result.push_str("\n## Relation types (select from these only)\n");
        for rt in relation_types.effective_types() {
            result.push_str(&format!("- {}\n", rt));
        }

        if !intent_types.effective_types().is_empty() {
            result.push_str("\n## Intent types (select from these only)\n");
            for it in intent_types.effective_types() {
                result.push_str(&format!("- {}\n", it));
            }
        }

        result
    }

    fn format_injected_context(&self, context: &PromptContext) -> String {
        let mut parts = Vec::new();

        if let Some(ref summary) = context.window_summary {
            parts.push(format!("## Window Summary\n{}", summary));
        }

        if let Some(ref summary) = context.context_summary {
            parts.push(format!("## Conversation Context\n{}", summary));
        }

        let reminders = self.format_user_reminders(context);
        if !reminders.is_empty() {
            parts.push(format!("## User Reminders\n{}", reminders));
        }

        if !context.related_entities.is_empty() {
            parts.push("## Related Knowledge".to_string());
            for entity in &context.related_entities {
                parts.push(format!("- {}", entity));
            }
        }

        parts.join("\n\n")
    }

    fn format_user_reminders(&self, context: &PromptContext) -> String {
        if context.user_reminders.is_empty() {
            return String::new();
        }

        let mut sorted = context.user_reminders.clone();
        sorted.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted
            .iter()
            .map(|r| {
                format!(
                    "- [importance={:.2}] {} ({})",
                    r.importance,
                    r.text,
                    r.created_at.format("%Y-%m-%d")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
