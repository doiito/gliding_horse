use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::batch::vocabulary::{EntityTypeConfig, IntentTypeConfig, RelationTypeConfig};

// ============================================================
// BatchAgentConfig — serialisable per-agent configuration
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAgentConfig {
    pub name: String,
    pub description: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    // Window
    #[serde(default)]
    pub window_type: WindowType,

    // Triggers
    #[serde(default)]
    pub triggers: Vec<TriggerConfig>,

    // Prompt
    #[serde(default)]
    pub prompt_source: PromptSource,
    pub prompt_template_path: Option<String>,
    pub prompt_template_name: Option<String>,
    #[serde(default)]
    pub prompt_params: HashMap<String, String>,

    // Business domain
    pub business_domain: String,
    #[serde(default)]
    pub entity_types: EntityTypeConfig,
    #[serde(default)]
    pub relation_types: RelationTypeConfig,
    #[serde(default)]
    pub intent_types: IntentTypeConfig,

    // Execution overrides
    pub model: Option<String>,
    pub temperature: Option<f32>,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    // Event config
    #[serde(default)]
    pub emit_on: Vec<EmitCondition>,

    // Context injection
    #[serde(default = "default_true")]
    pub inject_user_reminders: bool,
    #[serde(default = "default_true")]
    pub inject_context_summary: bool,
    #[serde(default = "default_true")]
    pub inject_related_entities: bool,
    /// Explicit opt-in: maintenance handlers can change SkillGraphStore only
    /// when deployment governance enables this flag.
    #[serde(default)]
    pub apply_graph_mutations: bool,
}

impl Default for BatchAgentConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            enabled: default_enabled(),
            window_type: WindowType::default(),
            triggers: Vec::new(),
            prompt_source: PromptSource::default(),
            prompt_template_path: None,
            prompt_template_name: None,
            prompt_params: HashMap::new(),
            business_domain: String::new(),
            entity_types: EntityTypeConfig::default(),
            relation_types: RelationTypeConfig::default(),
            intent_types: IntentTypeConfig::default(),
            model: None,
            temperature: None,
            max_retries: default_max_retries(),
            timeout_seconds: default_timeout(),
            emit_on: Vec::new(),
            inject_user_reminders: default_true(),
            inject_context_summary: default_true(),
            inject_related_entities: default_true(),
            apply_graph_mutations: false,
        }
    }
}

fn default_enabled() -> bool {
    true
}
fn default_max_retries() -> u32 {
    3
}
fn default_timeout() -> u64 {
    300
}
fn default_true() -> bool {
    true
}

// ============================================================
// Window types
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    MessageCount(usize),
    TimeWindow(u64),
    Hybrid {
        max_messages: usize,
        max_seconds: u64,
    },
    Manual,
}

impl Default for WindowType {
    fn default() -> Self {
        Self::Hybrid {
            max_messages: 5,
            max_seconds: 600,
        }
    }
}

// ============================================================
// Trigger config
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerConfig {
    pub trigger_type: TriggerType,
    #[serde(default)]
    pub params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerType {
    WindowFull,
    CronSchedule(String),
    IntentShift,
    MessageThreshold(usize),
    CustomEvent(String),
}

// ============================================================
// Prompt source
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PromptSource {
    TemplateFile,
    TemplateEngine,
    LlmGenerated,
    HybridWithTemplate,
}

impl Default for PromptSource {
    fn default() -> Self {
        Self::HybridWithTemplate
    }
}

// ============================================================
// Emit conditions
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmitCondition {
    NewEntity,
    NewRelation,
    IntentDetected(Vec<String>),
    ConfidenceAbove(f64),
    Always,
}

// ============================================================
// Window entry — single message in the sliding window
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowEntry {
    pub message_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub estimated_intent: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

// ============================================================
// Extraction results
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub batch_id: String,
    pub extracted_at: DateTime<Utc>,

    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
    pub intent: Option<DetectedIntent>,
    pub key_decisions: Vec<ExtractedDecision>,
    pub context_summary: String,

    pub llm_calls: u32,
    pub tokens_consumed: u32,
    #[serde(default)]
    pub confidence_scores: HashMap<String, f64>,
    pub raw_response: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub confidence: f64,
    #[serde(default)]
    pub source_messages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelation {
    pub from: String,
    pub relation: String,
    pub to: String,
    #[serde(default)]
    pub properties: HashMap<String, String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedIntent {
    pub intent_type: String,
    pub confidence: f64,
    #[serde(default)]
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedDecision {
    pub decision: String,
    pub rationale: Option<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub confidence: String,
}

// ============================================================
// Persist report
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistReport {
    pub entities_persisted: usize,
    pub relations_persisted: usize,
    pub new_entities: usize,
    pub updated_entities: usize,
    pub named_graph: String,
    pub task_iri: Option<String>,
}

// ============================================================
// Trigger reason — why the batch was triggered
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerReason {
    NotReady,
    WindowFull { count: usize, max: usize },
    TimeElapsed { elapsed_secs: u64, max_secs: u64 },
    IntentShift { from: String, to: String },
    CustomEvent { event_type: String, payload: String },
}

// ============================================================
// Window status
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowStatus {
    pub entry_count: usize,
    pub oldest: Option<DateTime<Utc>>,
    pub newest: Option<DateTime<Utc>>,
    pub has_summary: bool,
    pub last_trigger: Option<DateTime<Utc>>,
}

// ============================================================
// Prompt context — assembled before calling LLM
// ============================================================

#[derive(Debug, Clone, Default)]
pub struct PromptContext {
    pub user_reminders: Vec<EmphasisItem>,
    pub context_summary: Option<String>,
    pub related_entities: Vec<String>,
    pub window_summary: Option<String>,
    pub total_tokens_hint: usize,
}

impl PromptContext {
    pub fn total_tokens(&self) -> usize {
        let mut total = self.total_tokens_hint;
        for r in &self.user_reminders {
            total += r.text.len();
        }
        if let Some(ref s) = self.context_summary {
            total += s.len();
        }
        for e in &self.related_entities {
            total += e.len();
        }
        total
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmphasisItem {
    pub text: String,
    pub importance: f64,
    pub created_at: DateTime<Utc>,
}

// ============================================================
// RDF quad for knowledge graph writes
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfQuad {
    pub subject: String,
    pub predicate: String,
    pub object: RdfValue,
    pub graph: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RdfValue {
    Iri(String),
    Literal(String),
    TypedLiteral(String, String),
}

// ============================================================
// Agent status for lifecycle tracking
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BatchAgentStatus {
    Registered,
    Running,
    Paused,
    Stopped,
    Error(String),
}

// ============================================================
// Batch metrics
// ============================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchMetrics {
    pub total_extractions: u64,
    pub total_entities_found: u64,
    pub total_relations_found: u64,
    pub total_tokens_consumed: u64,
    pub success_count: u64,
    pub failure_count: u64,

    // Rolling window for success rate (last 100)
    pub last_outcomes: Vec<bool>,
}

impl BatchMetrics {
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 1.0;
        }
        self.success_count as f64 / total as f64
    }

    pub fn record_success(&mut self, tokens: u32, entities: usize, relations: usize) {
        self.total_extractions += 1;
        self.success_count += 1;
        self.total_tokens_consumed += tokens as u64;
        self.total_entities_found += entities as u64;
        self.total_relations_found += relations as u64;
        self.last_outcomes.push(true);
        if self.last_outcomes.len() > 100 {
            self.last_outcomes.remove(0);
        }
    }

    pub fn record_failure(&mut self) {
        self.total_extractions += 1;
        self.failure_count += 1;
        self.last_outcomes.push(false);
        if self.last_outcomes.len() > 100 {
            self.last_outcomes.remove(0);
        }
    }

    pub fn recent_success_rate(&self) -> f64 {
        if self.last_outcomes.is_empty() {
            return 1.0;
        }
        let successes = self.last_outcomes.iter().filter(|&&x| x).count();
        successes as f64 / self.last_outcomes.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::BatchAgentConfig;

    #[test]
    fn graph_mutations_are_opt_in_by_default() {
        assert!(!BatchAgentConfig::default().apply_graph_mutations);

        let config: BatchAgentConfig = serde_json::from_value(serde_json::json!({
            "name": "legacy-agent",
            "description": "legacy config without the new field",
            "business_domain": "test"
        }))
        .unwrap();
        assert!(!config.apply_graph_mutations);
    }
}
