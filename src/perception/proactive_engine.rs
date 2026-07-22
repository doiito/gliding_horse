use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::core::event_bus::EventBus;
use crate::core::perception_store::{PerceptionEntry, PerceptionSource, PerceptionStore};
use crate::memory::hyperspace_store::HyperspaceStore;
use crate::memory::l0_store::{L0Entry, L0Store, MesiState};
#[cfg(feature = "ontology")]
use crate::ontology_bridge::OntologyBridgeManager;
use crate::CoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionTrigger {
    TaskStart,
    PlanCompleted,
    ProgressAnomaly,
    CheckCompleted,
    TaskEnd,
    CycleTimeout,
    AgentBlocked,
    ResourceConflict,
    QualityDegradation,
    UserFeedback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAnalysis {
    pub summary: String,
    pub complexity: String,
    pub estimated_steps: u32,
    pub risks: Vec<String>,
    pub recommended_approach: String,
    pub agent_assignments: HashMap<String, String>,
    #[serde(default)]
    pub relevant_experience_hints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisoryNode {
    pub advisory_id: String,
    pub advisory_type: String,
    pub severity: String,
    pub content: Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionPlan {
    pub anomaly_id: String,
    pub diagnosis: String,
    pub actions: Vec<String>,
    pub priority: String,
    pub should_interrupt: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub experience_id: String,
    pub scenario: String,
    pub pattern: String,
    pub success_rating: f32,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PerceptionConfig {
    pub cache_ttl_seconds: i64,
    pub cache_max_entries: usize,
    pub anomaly_dedup_window_seconds: i64,
    pub simple_input_threshold: usize,
    pub medium_input_threshold: usize,
    pub simple_steps: u32,
    pub medium_steps: u32,
    pub complex_steps: u32,
    pub complex_subtask_threshold: usize,
    pub cycle_timeout_secs: i64,
    pub max_iterations_before_alert: usize,
    pub error_rate_threshold: f64,
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            cache_ttl_seconds: 300,
            cache_max_entries: 1000,
            anomaly_dedup_window_seconds: 60,
            simple_input_threshold: 50,
            medium_input_threshold: 200,
            simple_steps: 1,
            medium_steps: 3,
            complex_steps: 5,
            complex_subtask_threshold: 5,
            cycle_timeout_secs: 300,
            max_iterations_before_alert: 10,
            error_rate_threshold: 0.5,
        }
    }
}

impl PerceptionConfig {
    pub fn from_settings(settings: &crate::config::PerceptionSettings) -> Self {
        Self {
            cache_ttl_seconds: settings.cache_ttl_seconds as i64,
            cache_max_entries: settings.cache_max_entries,
            anomaly_dedup_window_seconds: settings.anomaly_dedup_window_seconds as i64,
            simple_input_threshold: settings.simple_input_threshold,
            medium_input_threshold: settings.medium_input_threshold,
            cycle_timeout_secs: settings.cycle_timeout_secs as i64,
            max_iterations_before_alert: settings.max_iterations_before_alert,
            error_rate_threshold: settings.error_rate_threshold,
            ..Default::default()
        }
    }
}

pub struct ProactiveEngine {
    cache: HashMap<String, (DateTime<Utc>, Value)>,
    pub(crate) config: PerceptionConfig,
    anomaly_history: Vec<(String, DateTime<Utc>)>,
    l0: Arc<L0Store>,
    #[allow(dead_code)]
    event_bus: Arc<EventBus>,
    pub(crate) perception_store: Option<Arc<PerceptionStore>>,
    pub(crate) hyperspace: Option<Arc<HyperspaceStore>>,
    /// OntologyBridge dual-space embedding store for cross-domain structural search.
    #[cfg(feature = "ontology")]
    pub(crate) ontology_bridge: Option<Arc<OntologyBridgeManager>>,
}

impl ProactiveEngine {
    pub fn new(l0: Arc<L0Store>, event_bus: Arc<EventBus>) -> Self {
        Self {
            cache: HashMap::new(),
            config: PerceptionConfig::default(),
            anomaly_history: Vec::new(),
            l0,
            event_bus,
            perception_store: None,
            hyperspace: None,
            #[cfg(feature = "ontology")]
            ontology_bridge: None,
        }
    }

    pub fn cycle_timeout_secs(&self) -> i64 {
        self.config.cycle_timeout_secs
    }

    pub fn with_config(config: PerceptionConfig, l0: Arc<L0Store>, event_bus: Arc<EventBus>) -> Self {
        Self {
            cache: HashMap::new(),
            config,
            anomaly_history: Vec::new(),
            l0,
            event_bus,
            perception_store: None,
            hyperspace: None,
            #[cfg(feature = "ontology")]
            ontology_bridge: None,
        }
    }

    /// Set proactive perception store, enabling ProactiveEngine to inject alert perception data into Agent
    pub fn with_perception_store(mut self, store: Arc<PerceptionStore>) -> Self {
        self.perception_store = Some(store);
        self
    }

    /// Set HyperspaceStore for semantic experience retrieval.
    /// When configured, experience queries use embedding-based semantic search
    /// instead of L0 tag-based substring matching.
    pub fn with_hyperspace_store(mut self, store: Arc<HyperspaceStore>) -> Self {
        self.hyperspace = Some(store);
        self
    }

    /// Attach OntologyBridge for dual-space (text + structural) embedding writes.
    #[cfg(feature = "ontology")]
    pub fn with_ontology_bridge(mut self, ob: Arc<OntologyBridgeManager>) -> Self {
        self.ontology_bridge = Some(ob);
        self
    }

    pub fn check_5w2h_constraints(&self, five_w2h_iri: &str) -> Option<String> {
        let entry = self.l0.retrieve(five_w2h_iri).ok()??;
        let node: serde_json::Value = serde_json::from_str(&entry.content).ok()?;

        let mut alerts = Vec::new();

        if let Some(when) = node.get("task:when") {
            if let Some(deadline_str) = when.get("task:deadline").and_then(|v| v.as_str()) {
                if let Ok(deadline) = deadline_str.parse::<DateTime<Utc>>() {
                    let now = Utc::now();
                    let reminder_duration = when.get("task:reminderBefore")
                        .and_then(|v| v.as_str())
                        .and_then(|s| parse_iso8601_duration(s))
                        .unwrap_or(chrono::Duration::hours(1));
                    let until = deadline.signed_duration_since(now);
                    if until.num_seconds() > 0 && until <= reminder_duration {
                        alerts.push("DEADLINE_APPROACHING".to_string());
                    } else if until.num_seconds() <= 0 {
                        alerts.push("DEADLINE_EXCEEDED".to_string());
                    }
                }
            }
        }

        if let Some(how_much) = node.get("task:howMuch") {
            if let Some(budget) = how_much.get("task:tokenBudget").and_then(|v| v.as_u64()) {
                if let Some(actual) = how_much.get("task:actualCost") {
                    if let Some(used) = actual.get("tokensUsed").and_then(|v| v.as_u64()) {
                        if budget > 0 && used * 100 > budget * 80 {
                            alerts.push("BUDGET_EXCEEDED".to_string());
                        }
                    }
                }
            }
        }

        if alerts.is_empty() {
            None
        } else {
            Some(alerts.join(","))
        }
    }

    async fn query_relevant_experiences(&self, query: &str) -> Vec<serde_json::Value> {
        // When HyperspaceStore is available, use semantic search with time decay
        if let Some(ref hs) = self.hyperspace {
            // Wrap query to give the embedding model more context
            let augmented_query = format!("agent experience: {}", query);
            // Use time-decay search: λ=0.5 means experiences are halved in score every ~1.4h
            // Filter to only "experience" tagged entries with "experience" JSON-LD type
            let filter = crate::memory::hyperspace_store::HybridSearchFilter::new()
                .with_must_tags(vec!["experience".to_string()])
                .with_jsonld_types(vec!["Experience".to_string()])
                .with_min_importance(0.05);
            match hs.search_with_time_decay(&augmented_query, &filter, 0.5, 5).await {
                Ok(entries) => {
                    return entries.iter()
                        .filter_map(|entry| {
                            // ScoredEntry has iri, text, score, tags, importance, jsonld_types, stored_at
                            // The full experience JSON was stored in L0, indexed by IRI
                            if !entry.iri.is_empty() {
                                if let Ok(Some(l0_entry)) = self.l0.retrieve(&entry.iri) {
                                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&l0_entry.content) {
                                        return Some(val);
                                    }
                                }
                            }
                            None
                        })
                        .take(5)
                        .collect();
                }
                Err(_) => {} // Fall through to L0 fallback
            }
        }

        // Fallback: L0 tag-based + substring matching (original behavior)
        let results = match self.l0.search_by_tags(&["experience".to_string()]) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let query_lower = query.to_lowercase();
        results.into_iter()
            .filter(|entry| {
                entry.content.to_lowercase().contains(&query_lower)
                    || entry.tags.iter().any(|t| query_lower.contains(&t.to_lowercase()))
            })
            .take(5)
            .filter_map(|entry| serde_json::from_str(&entry.content).ok())
            .collect()
    }

    fn cache_key(trigger: &str, context: &str) -> String {
        format!("{}:{}", trigger, context)
    }

    fn is_cached(&self, key: &str) -> bool {
        self.cache.get(key).map_or(false, |(ts, _)| {
            Utc::now().signed_duration_since(*ts).num_seconds() < self.config.cache_ttl_seconds
        })
    }

    fn evict_cache(&mut self) {
        if self.cache.len() > self.config.cache_max_entries
            && self.config.cache_max_entries > 0
        {
            let now = Utc::now();
            self.cache.retain(|_, (ts, _)| {
                now.signed_duration_since(*ts).num_seconds() < self.config.cache_ttl_seconds
            });
            while self.cache.len() > self.config.cache_max_entries {
                let oldest_key = self.cache.iter()
                    .min_by_key(|(_, (ts, _))| *ts)
                    .map(|(k, _)| k.clone());
                if let Some(key) = oldest_key {
                    self.cache.remove(&key);
                } else {
                    break;
                }
            }
        }
    }

    fn evict_anomaly_history(&mut self) {
        let now = Utc::now();
        self.anomaly_history.retain(|(_, ts)| {
            now.signed_duration_since(*ts).num_seconds() < self.config.anomaly_dedup_window_seconds * 2
        });
        let max_history = self.config.cache_max_entries.max(1000);
        if self.anomaly_history.len() > max_history {
            self.anomaly_history.sort_by_key(|(_, ts)| *ts);
            self.anomaly_history.truncate(max_history);
        }
    }

    pub async fn on_task_start(&mut self, user_input: &str, task_iri: &str) -> Result<TaskAnalysis, CoreError> {
        let key = Self::cache_key("task_start", task_iri);
        if self.is_cached(&key) {
            return Ok(serde_json::from_value(self.cache[&key].1.clone())
                .unwrap_or_else(|_| self.analyze_task(user_input)));
        }

        let mut analysis = self.analyze_task(user_input);

        // Query workspace perception data: check if files have changed externally
        if let Some(ref ps) = self.perception_store {
            let ws_events = ps.peek_unconsumed_for_source(PerceptionSource::WorkspaceMonitor);
            if !ws_events.is_empty() {
                // Merge workspace file events into the analysis as relevant context
                for event_text in &ws_events {
                    analysis.relevant_experience_hints.push(event_text.clone());
                }
                analysis.risks.push(format!("External file changes detected ({} event(s))", ws_events.len()));
            }
        }

        let experiences = self.query_relevant_experiences(user_input).await;
        analysis.relevant_experience_hints = experiences.iter()
            .filter_map(|e| e.get("scenario").and_then(|s| s.as_str()).map(|s| s.to_string()))
            .take(5)
            .collect();

        let val = serde_json::to_value(&analysis).unwrap_or_default();
        self.cache.insert(key, (Utc::now(), val));
        self.evict_cache();

        info!(task = %task_iri, complexity = %analysis.complexity, "Task analyzed");
        Ok(analysis)
    }

    pub fn on_plan_completed(&self, plan: &Value, task_iri: &str) -> Vec<AdvisoryNode> {
        let mut advisories = Vec::new();
        if let Some(sub_tasks) = plan.get("sub_tasks").and_then(|v| v.as_array()) {
            if sub_tasks.len() > self.config.complex_subtask_threshold {
                advisories.push(AdvisoryNode {
                    advisory_id: format!("adv_{}", uuid::Uuid::new_v4().hyphenated()),
                    advisory_type: "complexity_warning".to_string(),
                    severity: "medium".to_string(),
                    content: json!({"message": format!("Plan has {} sub-tasks (threshold: {}), consider parallelization", sub_tasks.len(), self.config.complex_subtask_threshold)}),
                    created_at: Utc::now(),
                });
            }
        }
        debug!(task = %task_iri, advisories = advisories.len(), "Plan assessed");
        advisories
    }

    pub fn on_progress_anomaly(&mut self, anomaly: &Value, task_iri: &str) -> InterventionPlan {
        self.evict_anomaly_history();

        let desc = anomaly.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let anomaly_id = format!("anomaly_{}", uuid::Uuid::new_v4().hyphenated());

        if self.anomaly_history.iter().any(|(d, t)| {
            d == desc && Utc::now().signed_duration_since(*t).num_seconds() < self.config.anomaly_dedup_window_seconds
        }) {
            return InterventionPlan {
                anomaly_id: String::new(),
                diagnosis: "already_handled".to_string(),
                actions: vec![],
                priority: "low".to_string(),
                should_interrupt: false,
            };
        }
        self.anomaly_history.push((desc.to_string(), Utc::now()));

        warn!(task = %task_iri, anomaly = %desc, "Progress anomaly detected");
        InterventionPlan {
            anomaly_id,
            diagnosis: format!("Progress anomaly: {}", desc),
            actions: vec!["Reassess current plan".to_string(), "Consider additional resources".to_string()],
            priority: "high".to_string(),
            should_interrupt: true,
        }
    }

    pub fn on_check_completed(&self, check_result: &Value, task_iri: &str) -> Option<AdvisoryNode> {
        let verdict = check_result.get("verdict").and_then(|v| v.as_str()).unwrap_or("pass");
        if verdict == "fail" {
            debug!(task = %task_iri, "Check failed, generating advisory");
            return Some(AdvisoryNode {
                advisory_id: format!("adv_{}", uuid::Uuid::new_v4().hyphenated()),
                advisory_type: "check_failure".to_string(),
                severity: "high".to_string(),
                content: json!({"message": "Check failed, review required", "details": check_result}),
                created_at: Utc::now(),
            });
        }
        None
    }

    pub async fn on_task_end(&self, task_result: &Value, task_iri: &str) -> Option<Experience> {
        let status = task_result.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
        if status == "success" || status == "failed" {
            debug!(task = %task_iri, status = %status, "Extracting experience");

            let cache_key = Self::cache_key("task_start", task_iri);
            let task_analysis = self.cache.get(&cache_key).and_then(|(_, v)| {
                serde_json::from_value::<TaskAnalysis>(v.clone()).ok()
            });

            let scenario = task_result.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let mut tags = vec![format!("task:{}", task_iri), format!("status:{}", status), "experience".to_string()];
            if let Some(ref analysis) = task_analysis {
                tags.push(format!("complexity:{}", analysis.complexity));
            }

            let experience = Experience {
                experience_id: format!("exp_{}", uuid::Uuid::new_v4().hyphenated()),
                scenario: scenario.clone(),
                pattern: format!("task_{}", status),
                success_rating: if status == "success" { 0.9 } else { 0.1 },
                tags,
                created_at: Utc::now(),
            };

            let iri = format!("iri://experience/{}", experience.experience_id);
            let mut experience_json = serde_json::json!({
                "@id": iri,
                "@type": "Experience",
                "scenario": &experience.scenario,
                "pattern": &experience.pattern,
                "success_rating": experience.success_rating,
                "tags": &experience.tags,
            });
            if let Some(ref analysis) = task_analysis {
                if let Some(obj) = experience_json.as_object_mut() {
                    obj.insert("complexity".to_string(), json!(analysis.complexity));
                    obj.insert("risks".to_string(), json!(analysis.risks));
                    obj.insert("recommended_approach".to_string(), json!(analysis.recommended_approach));
                    obj.insert("estimated_steps".to_string(), json!(analysis.estimated_steps));
                }
            }
            let content = serde_json::to_string(&experience_json).unwrap_or_default();
            let hs_content = content.clone();
            if !content.is_empty() {
                let entry = L0Entry {
                    iri: iri.clone(),
                    content,
                    importance: experience.success_rating,
                    access_count: 0,
                    created_at: Utc::now(),
                    last_accessed: Utc::now(),
                    tags: experience.tags.clone(),
                    metadata: serde_json::Map::new(),
                    mesi_state: MesiState::Shared,
                    content_hash: String::new(),
                    named_graph: None,
                    jsonld_context: None,
                    jsonld_types: Vec::new(),
                    hyperspace_point_id: None,
                };
                let _ = self.l0.store_entry(&entry);

                // Also store to HyperspaceStore for semantic retrieval when available
                if let Some(ref hs) = self.hyperspace {
                    let hs_tags = experience.tags.clone();
                    let _ = hs.upsert_with_metadata(
                        &iri,
                        &hs_content,
                        &hs_tags,
                        Some(experience.success_rating as f32),
                        Some(&["Experience".to_string()]),
                        None,
                    ).await;
                }

                // Store dual-space embedding to OntologyBridge (text Cosine + struct Poincaré)
                #[cfg(feature = "ontology")]
                if let Some(ref ob) = self.ontology_bridge {
                    let jsonld = serde_json::json!({
                        "@id": iri,
                        "@type": ["Experience"],
                        "scenario": experience.scenario,
                        "pattern": experience.pattern,
                        "success_rating": experience.success_rating,
                    });
                    let features = crate::ontology_bridge::StructuralFeatures {
                        tags: experience.tags.clone(),
                        jsonld_types: vec!["Experience".to_string()],
                        named_graph: None,
                    };
                    let _ = ob.store_dual_embedding(&iri, &hs_content, &features, &jsonld).await;
                }
            }

            return Some(experience);
        }
        None
    }

    pub fn on_cycle_timeout(&self, cycle_id: &str, task_iri: &str, elapsed_secs: f64) -> InterventionPlan {
        warn!(cycle = %cycle_id, task = %task_iri, elapsed = elapsed_secs, "Cycle timeout");
        InterventionPlan {
            anomaly_id: format!("timeout_{}", uuid::Uuid::new_v4().hyphenated()),
            diagnosis: format!("Cycle {} timeout after {:.0}s", cycle_id, elapsed_secs),
            actions: vec!["Extend timeout".to_string(), "Check agent health".to_string()],
            priority: "critical".to_string(),
            should_interrupt: true,
        }
    }

    pub fn on_agent_blocked(&self, agent_id: &str, task_iri: &str) -> InterventionPlan {
        warn!(agent = %agent_id, task = %task_iri, "Agent blocked");
        InterventionPlan {
            anomaly_id: format!("blocked_{}", uuid::Uuid::new_v4().hyphenated()),
            diagnosis: format!("Agent {} is blocked", agent_id),
            actions: vec!["Restart agent".to_string(), "Inject assistance message".to_string()],
            priority: "high".to_string(),
            should_interrupt: true,
        }
    }

    /// Inject conflict alerts into PerceptionStore
    fn inject_conflict_perception(&self, conflict_type: &str, path: &str, task_iri: &str) {
        if let Some(ref store) = self.perception_store {
            let type_label = match conflict_type {
                "stale_file_write" => "stale file write",
                "concurrent_write" => "concurrent write conflict",
                "unread_write" => "unread write",
                _ => "resource conflict",
            };
            let entry = PerceptionEntry::new(
                PerceptionSource::PerceptionEngine,
                format!("{}: {} (task: {})", type_label, path, task_iri),
            ).with_priority(8);
            store.store_global(entry);
        }
    }

    pub fn on_resource_conflict(&self, conflict: &Value, task_iri: &str) -> InterventionPlan {
        warn!(task = %task_iri, conflict = ?conflict, "Resource conflict");

        let conflict_type = conflict
            .get("conflict_type")
            .and_then(|v| v.as_str())
            .unwrap_or("generic");
        let resource_path = conflict
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // inject alerts into perception area
        self.inject_conflict_perception(conflict_type, resource_path, task_iri);

        let (diagnosis, actions, priority, interrupt) = match conflict_type {
            "stale_file_write" => (
                format!("Write to stale file without re-read: {}", resource_path),
                vec![
                    format!("Re-read file: {}", resource_path),
                    "Re-apply edit after reading latest version".to_string(),
                ],
                "high",
                true,
            ),
            "concurrent_write" => (
                format!("Concurrent write contention on: {}", resource_path),
                vec![
                    "Serialize writes to shared resource".to_string(),
                    format!("Invalidate cache for: {}", resource_path),
                    "Notify SA of contention".to_string(),
                ],
                "high",
                true,
            ),
            "unread_write" => (
                format!("Write to file without prior read: {}", resource_path),
                vec![
                    format!("Read file first: {}", resource_path),
                    "Verify write is intentional without prior context".to_string(),
                ],
                "medium",
                false,
            ),
            _ => (
                format!("Resource conflict detected on: {}", resource_path),
                vec![
                    "Queue conflicting requests".to_string(),
                    "Notify SA".to_string(),
                ],
                "medium",
                false,
            ),
        };

        InterventionPlan {
            anomaly_id: format!("conflict_{}", uuid::Uuid::new_v4().hyphenated()),
            diagnosis,
            actions,
            priority: priority.to_string(),
            should_interrupt: interrupt,
        }
    }

    pub fn on_quality_degradation(&self, degradation: &Value, task_iri: &str) -> InterventionPlan {
        warn!(task = %task_iri, degradation = ?degradation, "Quality degradation");
        InterventionPlan {
            anomaly_id: format!("quality_{}", uuid::Uuid::new_v4().hyphenated()),
            diagnosis: "Output quality degraded".to_string(),
            actions: vec!["Rollback to last checkpoint".to_string(), "Re-run with different approach".to_string()],
            priority: "high".to_string(),
            should_interrupt: true,
        }
    }

    pub fn on_user_feedback(&self, feedback: &Value, task_iri: &str) -> AdvisoryNode {
        let message = feedback.get("message").and_then(|v| v.as_str()).unwrap_or("");
        info!(task = %task_iri, feedback = %message, "User feedback received");
        AdvisoryNode {
            advisory_id: format!("fb_{}", uuid::Uuid::new_v4().hyphenated()),
            advisory_type: "user_feedback".to_string(),
            severity: "medium".to_string(),
            content: feedback.clone(),
            created_at: Utc::now(),
        }
    }

    /// Check for workspace file conflicts before a write operation.
    /// Returns Some(conflict_value) if a conflict is detected.
    pub fn check_file_conflict(
        &self,
        file_path: &str,
        current_state: &str,
        last_read_version: u64,
        current_version: u64,
        agent_id: &str,
    ) -> Option<Value> {
        if current_state == "ReadStale" {
            return Some(json!({
                "conflict_type": "stale_file_write",
                "path": file_path,
                "agent_id": agent_id,
                "current_version": current_version,
                "last_read_version": last_read_version,
                "severity": "high",
            }));
        }
        if current_state == "WrittenUnread" || current_state == "Discovered" {
            return Some(json!({
                "conflict_type": "unread_write",
                "path": file_path,
                "agent_id": agent_id,
                "current_state": current_state,
                "severity": "medium",
            }));
        }
        None
    }

    fn analyze_task(&self, user_input: &str) -> TaskAnalysis {
        let input_len = user_input.len();
        let (complexity, steps) = if input_len < self.config.simple_input_threshold {
            ("simple".to_string(), self.config.simple_steps)
        } else if input_len < self.config.medium_input_threshold {
            ("medium".to_string(), self.config.medium_steps)
        } else {
            ("complex".to_string(), self.config.complex_steps)
        };

        let risks = if complexity == "complex" {
            vec!["Large scope may require multiple iterations".to_string()]
        } else {
            vec![]
        };

        TaskAnalysis {
            summary: user_input.chars().take(100).collect(),
            complexity: complexity.clone(),
            estimated_steps: steps,
            risks: risks.clone(),
            recommended_approach: self.recommend_approach(&complexity),
            agent_assignments: HashMap::from([
                ("plan".to_string(), "PA".to_string()),
                ("execute".to_string(), "DA".to_string()),
                ("check".to_string(), "CA".to_string()),
                ("act".to_string(), "AA".to_string()),
            ]),
            relevant_experience_hints: Vec::new(),
        }
    }

    fn recommend_approach(&self, complexity: &str) -> String {
        match complexity {
            "complex" => "recursive_pdca".to_string(),
            "medium" => "standard_pdca".to_string(),
            _ => "direct_da".to_string(),
        }
    }
}

fn parse_iso8601_duration(s: &str) -> Option<chrono::Duration> {
    let s = s.strip_prefix("PT")?;
    let mut hours: i64 = 0;
    let mut minutes: i64 = 0;
    let mut secs: i64 = 0;
    let mut num_str = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            num_str.push(c);
        } else {
            let num: i64 = num_str.parse().ok()?;
            num_str.clear();
            match c {
                'H' => hours = num,
                'M' => minutes = num,
                'S' => secs = num,
                _ => {}
            }
        }
    }
    Some(chrono::Duration::hours(hours) + chrono::Duration::minutes(minutes) + chrono::Duration::seconds(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_l0<F, R>(f: F) -> R
    where
        F: FnOnce(Arc<L0Store>) -> R,
    {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        f(store)
    }

    fn test_event_bus() -> Arc<EventBus> {
        Arc::new(EventBus::new(100))
    }

    #[tokio::test]
    async fn test_task_start_analysis() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut engine = ProactiveEngine::new(l0, test_event_bus());
        let analysis = engine.on_task_start("Write a hello world program", "iri://task/1").await.unwrap();
        assert_eq!(analysis.complexity, "simple");
        assert_eq!(analysis.estimated_steps, 1);
    }

    #[test]
    fn test_check_completed_pass() {
        with_l0(|l0| {
            let engine = ProactiveEngine::new(l0, test_event_bus());
            let result = engine.on_check_completed(&json!({"verdict": "pass"}), "iri://task/1");
            assert!(result.is_none());
        });
    }

    #[test]
    fn test_check_completed_fail() {
        with_l0(|l0| {
            let engine = ProactiveEngine::new(l0, test_event_bus());
            let result = engine.on_check_completed(&json!({"verdict": "fail"}), "iri://task/1");
            assert!(result.is_some());
            assert_eq!(result.unwrap().severity, "high");
        });
    }

    #[tokio::test]
    async fn test_custom_config() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let config = PerceptionConfig {
            simple_input_threshold: 100,
            medium_input_threshold: 500,
            ..Default::default()
        };
        let mut engine = ProactiveEngine::with_config(config, l0, test_event_bus());
        let analysis = engine.on_task_start("A medium length task description", "iri://task/1").await.unwrap();
        assert_eq!(analysis.complexity, "simple");
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let config = PerceptionConfig {
            cache_max_entries: 2,
            cache_ttl_seconds: 300,
            ..Default::default()
        };
        let mut engine = ProactiveEngine::with_config(config, l0, test_event_bus());
        engine.on_task_start("task1", "iri://task/1").await.unwrap();
        engine.on_task_start("task2", "iri://task/2").await.unwrap();
        engine.on_task_start("task3", "iri://task/3").await.unwrap();
        assert!(engine.cache.len() <= 3);
    }

    #[test]
    fn test_cycle_timeout_returns_intervention() {
        with_l0(|l0| {
            let engine = ProactiveEngine::new(l0, test_event_bus());
            let plan = engine.on_cycle_timeout("cycle_1", "iri://task/1", 301.0);
            assert_eq!(plan.priority, "critical");
            assert!(plan.should_interrupt);
            assert!(plan.diagnosis.contains("timeout"));
            assert!(plan.anomaly_id.starts_with("timeout_"));
        });
    }

    #[test]
    fn test_agent_blocked_returns_intervention() {
        with_l0(|l0| {
            let engine = ProactiveEngine::new(l0, test_event_bus());
            let plan = engine.on_agent_blocked("agent_do_001", "iri://task/1");
            assert_eq!(plan.priority, "high");
            assert!(plan.should_interrupt);
            assert!(plan.diagnosis.contains("blocked"));
        });
    }

    #[test]
    fn test_resource_conflict_returns_non_interrupting() {
        with_l0(|l0| {
            let engine = ProactiveEngine::new(l0, test_event_bus());
            let conflict = serde_json::json!({"resource": "l0_store", "agents": ["a1", "a2"]});
            let plan = engine.on_resource_conflict(&conflict, "iri://task/1");
            assert_eq!(plan.priority, "medium");
            assert!(!plan.should_interrupt);
        });
    }

    #[test]
    fn test_quality_degradation_returns_intervention() {
        with_l0(|l0| {
            let engine = ProactiveEngine::new(l0, test_event_bus());
            let degradation = serde_json::json!({"metric": "accuracy", "dropped_by": 0.3});
            let plan = engine.on_quality_degradation(&degradation, "iri://task/1");
            assert_eq!(plan.priority, "high");
            assert!(plan.should_interrupt);
        });
    }

    #[test]
    fn test_user_feedback_returns_advisory() {
        with_l0(|l0| {
            let engine = ProactiveEngine::new(l0, test_event_bus());
            let feedback = serde_json::json!({"message": "Please retry with more details"});
            let advisory = engine.on_user_feedback(&feedback, "iri://task/1");
            assert_eq!(advisory.advisory_type, "user_feedback");
            assert_eq!(advisory.severity, "medium");
        });
    }

    #[test]
    fn test_progress_anomaly_dedup() {
        with_l0(|l0| {
            let mut engine = ProactiveEngine::new(l0, test_event_bus());
            let anomaly = serde_json::json!({"description": "stuck_at_step_3"});
            let first = engine.on_progress_anomaly(&anomaly, "iri://task/1");
            assert!(first.should_interrupt);
            let second = engine.on_progress_anomaly(&anomaly, "iri://task/1");
            assert!(!second.should_interrupt);
            assert_eq!(second.diagnosis, "already_handled");
        });
    }

    #[tokio::test]
    async fn test_on_task_end_success_creates_experience() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let engine = ProactiveEngine::new(l0.clone(), test_event_bus());
        let task_result = serde_json::json!({
            "status": "success",
            "summary": "Completed fibonacci function"
        });
        let experience = engine.on_task_end(&task_result, "iri://task/1").await;
        assert!(experience.is_some());
        let exp = experience.unwrap();
        assert_eq!(exp.success_rating, 0.9);
        assert!(exp.scenario.contains("fibonacci"));
        let retrieved = l0.search_by_tags(&["experience".to_string()]).ok();
        assert!(retrieved.is_some_and(|r| !r.is_empty()));
    }

    #[tokio::test]
    async fn test_on_task_end_failed_creates_experience() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let engine = ProactiveEngine::new(l0.clone(), test_event_bus());
        let task_result = serde_json::json!({
            "status": "failed",
            "summary": "Faulty implementation"
        });
        let experience = engine.on_task_end(&task_result, "iri://task/2").await;
        assert!(experience.is_some());
        assert_eq!(experience.unwrap().success_rating, 0.1);
    }

    #[tokio::test]
    async fn test_on_task_end_unknown_status_no_experience() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let engine = ProactiveEngine::new(l0, test_event_bus());
        let task_result = serde_json::json!({"status": "running"});
        assert!(engine.on_task_end(&task_result, "iri://task/3").await.is_none());
    }

    #[test]
    fn test_plan_completed_subtask_warning() {
        with_l0(|l0| {
            let engine = ProactiveEngine::with_config(
                PerceptionConfig { complex_subtask_threshold: 3, ..Default::default() },
                l0, test_event_bus(),
            );
            let plan = serde_json::json!({"sub_tasks": ["a", "b", "c", "d", "e"]});
            let advisories = engine.on_plan_completed(&plan, "iri://task/1");
            assert_eq!(advisories.len(), 1);
            assert_eq!(advisories[0].advisory_type, "complexity_warning");
        });
    }

    #[test]
    fn test_plan_completed_no_warning_below_threshold() {
        with_l0(|l0| {
            let engine = ProactiveEngine::with_config(
                PerceptionConfig { complex_subtask_threshold: 10, ..Default::default() },
                l0, test_event_bus(),
            );
            let plan = serde_json::json!({"sub_tasks": ["a", "b", "c"]});
            let advisories = engine.on_plan_completed(&plan, "iri://task/1");
            assert!(advisories.is_empty());
        });
    }

    #[test]
    fn test_check_completed_no_verdict() {
        with_l0(|l0| {
            let engine = ProactiveEngine::new(l0, test_event_bus());
            let result = engine.on_check_completed(&serde_json::json!({}), "iri://task/1");
            assert!(result.is_none());
        });
    }

    #[tokio::test]
    async fn test_evict_cache_on_insert() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut engine = ProactiveEngine::with_config(
            PerceptionConfig { cache_ttl_seconds: 300, cache_max_entries: 2, ..Default::default() },
            l0, test_event_bus(),
        );
        let _ = engine.on_task_start("task1", "iri://task/1").await;
        let _ = engine.on_task_start("task2", "iri://task/2").await;
        let _ = engine.on_task_start("task3", "iri://task/3").await;
        let _ = engine.on_task_start("task4", "iri://task/4").await;
        assert!(engine.cache.len() <= 2, "cache should evict to max_entries=2 got {}", engine.cache.len());
    }
}

#[cfg(test)]
mod tests_5w2h {
    use super::*;
    use crate::memory::l0_store::L0Store;
    use std::sync::Arc;

    fn test_event_bus() -> Arc<EventBus> {
        Arc::new(EventBus::new(100))
    }

    fn with_l0_5w2h<F, R>(f: F) -> R
    where
        F: FnOnce(Arc<L0Store>) -> R,
    {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        f(store)
    }

    #[test]
    fn test_check_5w2h_constraints_no_store() {
        with_l0_5w2h(|store| {
            let engine = ProactiveEngine::new(store, test_event_bus());
            assert!(engine.check_5w2h_constraints("iri://task/test/5w2h").is_none());
        })
    }

    #[test]
    fn test_check_5w2h_constraints_deadline_approaching() {
        with_l0_5w2h(|store| {
            let engine = ProactiveEngine::new(store.clone(), test_event_bus());
            let deadline = Utc::now() + chrono::Duration::minutes(30);
            let json_ld = serde_json::json!({
                "@type": "task:5W2H",
                "task:when": {
                    "task:deadline": deadline.to_rfc3339(),
                    "task:reminderBefore": "PT1H"
                }
            });
            store.store("iri://task/test/5w2h", &json_ld.to_string()).unwrap();
            let result = engine.check_5w2h_constraints("iri://task/test/5w2h");
            assert!(result.is_some());
            assert!(result.unwrap().contains("DEADLINE_APPROACHING"));
        })
    }

    #[test]
    fn test_check_5w2h_constraints_budget_exceeded() {
        with_l0_5w2h(|store| {
            let engine = ProactiveEngine::new(store.clone(), test_event_bus());
            let json_ld = serde_json::json!({
                "@type": "task:5W2H",
                "task:howMuch": {
                    "task:tokenBudget": 100000,
                    "task:actualCost": {
                        "tokensUsed": 85000,
                        "cyclesUsed": 2,
                        "durationSecs": 120.0
                    }
                }
            });
            store.store("iri://task/test/5w2h", &json_ld.to_string()).unwrap();
            let result = engine.check_5w2h_constraints("iri://task/test/5w2h");
            assert!(result.is_some());
            assert!(result.unwrap().contains("BUDGET_EXCEEDED"));
        })
    }

    #[test]
    fn test_check_5w2h_constraints_custom_reminder() {
        with_l0_5w2h(|store| {
            let engine = ProactiveEngine::new(store.clone(), test_event_bus());
            let deadline = Utc::now() + chrono::Duration::minutes(20);
            let json_ld = serde_json::json!({
                "@type": "task:5W2H",
                "task:when": {
                    "task:deadline": deadline.to_rfc3339(),
                    "task:reminderBefore": "PT30M"
                }
            });
            store.store("iri://task/test/custom/5w2h", &json_ld.to_string()).unwrap();
            let result = engine.check_5w2h_constraints("iri://task/test/custom/5w2h");
            assert!(result.is_some());
            assert!(result.unwrap().contains("DEADLINE_APPROACHING"));
        })
    }

    #[test]
    fn test_check_file_conflict_stale_write() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let engine = ProactiveEngine::new(store, test_event_bus());

        let result = engine.check_file_conflict(
            "/path/to/file.rs",
            "ReadStale",
            1,
            3,
            "agent_1",
        );
        assert!(result.is_some(), "Should detect stale file conflict");
        let conflict = result.unwrap();
        assert_eq!(conflict["conflict_type"], "stale_file_write");
        assert_eq!(conflict["path"], "/path/to/file.rs");
        assert_eq!(conflict["severity"], "high");
    }

    #[test]
    fn test_check_file_conflict_unread_write() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let engine = ProactiveEngine::new(store, test_event_bus());

        let result = engine.check_file_conflict(
            "/path/to/new.rs",
            "WrittenUnread",
            0,
            0,
            "agent_1",
        );
        assert!(result.is_some(), "Should detect unread file conflict");
        let conflict = result.unwrap();
        assert_eq!(conflict["conflict_type"], "unread_write");
        assert_eq!(conflict["severity"], "medium");
    }

    #[test]
    fn test_check_file_conflict_no_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let engine = ProactiveEngine::new(store, test_event_bus());

        let result = engine.check_file_conflict(
            "/path/to/fresh.rs",
            "ReadFresh",
            2,
            2,
            "agent_1",
        );
        assert!(result.is_none(), "Fresh file should have no conflict");
    }

    // ── PerceptionStore integration ──

    #[test]
    fn test_on_resource_conflict_injects_perception_store_stale_write() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let ps = Arc::new(PerceptionStore::new());
        let engine = ProactiveEngine::new(store, test_event_bus())
            .with_perception_store(ps.clone());

        let conflict = serde_json::json!({
            "conflict_type": "stale_file_write",
            "path": "/path/to/stale.rs",
            "agent_id": "agent_1",
        });
        let plan = engine.on_resource_conflict(&conflict, "iri://task/store_test");
        assert!(plan.should_interrupt);

        // PerceptionStore should have the conflict entry
        assert!(ps.has_new("iri://task/store_test"), "PerceptionStore should have entry after conflict");
        let text = ps.take_perception_text("iri://task/store_test");
        assert!(text.contains("Alert"), "Should contain alert prefix: {}", text);
        assert!(text.contains("stale"), "Should mention stale file: {}", text);
    }

    #[test]
    fn test_on_resource_conflict_injects_perception_store_concurrent() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let ps = Arc::new(PerceptionStore::new());
        let engine = ProactiveEngine::new(store, test_event_bus())
            .with_perception_store(ps.clone());

        let conflict = serde_json::json!({
            "conflict_type": "concurrent_write",
            "path": "/shared/file.rs",
            "agent_id": "agent_2",
        });
        let _plan = engine.on_resource_conflict(&conflict, "iri://task/concurrent");

        let text = ps.take_perception_text("iri://task/concurrent");
        assert!(text.contains("concurrent"), "Should mention concurrent write: {}", text);
    }

    #[test]
    fn test_on_resource_conflict_injects_perception_store_unread() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let ps = Arc::new(PerceptionStore::new());
        let engine = ProactiveEngine::new(store, test_event_bus())
            .with_perception_store(ps.clone());

        let conflict = serde_json::json!({
            "conflict_type": "unread_write",
            "path": "/path/to/unread.rs",
        });
        let _plan = engine.on_resource_conflict(&conflict, "iri://task/unread");

        let text = ps.take_perception_text("iri://task/unread");
        assert!(text.contains("unread"), "Should mention unread write: {}", text);
    }

    #[test]
    fn test_with_perception_store_noop_when_not_set() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        // No perception_store set — should not panic
        let engine = ProactiveEngine::new(store, test_event_bus());
        let conflict = serde_json::json!({
            "conflict_type": "stale_file_write",
            "path": "/path/to/f.rs",
        });
        let plan = engine.on_resource_conflict(&conflict, "iri://task/no_ps");
        assert!(plan.should_interrupt);
    }

    #[test]
    fn test_with_perception_store_after_construction() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let ps = Arc::new(PerceptionStore::new());
        let engine = ProactiveEngine::with_config(
            PerceptionConfig::default(),
            store,
            test_event_bus(),
        ).with_perception_store(ps.clone());

        let conflict = serde_json::json!({
            "conflict_type": "concurrent_write",
            "path": "/path/to/x.rs",
        });
        let _plan = engine.on_resource_conflict(&conflict, "iri://task/chain");

        let text = ps.take_perception_text("iri://task/chain");
        assert!(!text.is_empty(), "Chain-constructed engine should inject to PerceptionStore");
    }

    #[test]
    fn test_on_resource_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let engine = ProactiveEngine::new(store, test_event_bus());

        let conflict = serde_json::json!({
            "conflict_type": "stale_file_write",
            "path": "/path/to/stale.rs",
            "agent_id": "agent_1",
        });
        let plan = engine.on_resource_conflict(&conflict, "iri://task/test");
        assert!(plan.should_interrupt);
        assert_eq!(plan.priority, "high");
        assert!(plan.diagnosis.contains("stale"));
        assert!(plan.actions.len() >= 2);
    }

    #[test]
    fn test_on_resource_conflict_unread_write_plan() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let engine = ProactiveEngine::new(store, test_event_bus());

        let conflict = serde_json::json!({
            "conflict_type": "unread_write",
            "path": "/path/to/unread.rs",
        });
        let plan = engine.on_resource_conflict(&conflict, "iri://task/test");
        assert!(!plan.should_interrupt);
        assert_eq!(plan.priority, "medium");
    }
}
