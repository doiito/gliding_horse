use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::batch::emitter::BatchEventEmitter;
use crate::batch::error::BatchError;
use crate::batch::extractor::ExtractorPipeline;
use crate::batch::handlers;
use crate::batch::trigger::TriggerSystem;
use crate::batch::types::{BatchAgentConfig, BatchAgentStatus, BatchMetrics, ExtractionResult};
use crate::batch::window::SlidingWindow;
use crate::batch::window::WindowConfig;
use crate::config::settings::BatchAgentSettings;
use crate::core::event_bus::Event;
use crate::core::event_bus::EventBus;
use crate::skill_graph::graph_store::SkillGraphStore;

/// Holds the runtime state for a single Batch Agent instance.
pub struct BatchAgentInstance {
    pub name: String,
    pub config: BatchAgentConfig,
    pub status: BatchAgentStatus,
    pub window: Arc<RwLock<SlidingWindow>>,
    pub trigger_system: TriggerSystem,
    pub metrics: BatchMetrics,
}

pub struct BatchAgentManager {
    agents: HashMap<String, BatchAgentInstance>,
    event_bus: Option<Arc<EventBus>>,
    emitter: Option<BatchEventEmitter>,
    graph_store: Option<Arc<SkillGraphStore>>,
    running: bool,
}

impl BatchAgentManager {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            event_bus: None,
            emitter: None,
            graph_store: None,
            running: false,
        }
    }

    pub fn with_event_bus(mut self, event_bus: Arc<EventBus>) -> Self {
        let emitter = BatchEventEmitter::new(event_bus.clone());
        self.event_bus = Some(event_bus);
        self.emitter = Some(emitter);
        self
    }

    pub fn with_graph_store(mut self, graph_store: Arc<SkillGraphStore>) -> Self {
        self.graph_store = Some(graph_store);
        self
    }

    pub fn register(&mut self, config: BatchAgentConfig) -> Result<(), BatchError> {
        let name = config.name.clone();

        if self.agents.contains_key(&name) {
            return Err(BatchError::AgentAlreadyExists { name });
        }

        let window_config = match &config.window_type {
            crate::batch::types::WindowType::MessageCount(n) => WindowConfig {
                max_entries: *n,
                min_entries: 1,
                time_window_secs: u64::MAX,
                intent_shift_threshold: 0.6,
            },
            crate::batch::types::WindowType::TimeWindow(secs) => WindowConfig {
                max_entries: usize::MAX,
                min_entries: 1,
                time_window_secs: *secs,
                intent_shift_threshold: 0.6,
            },
            crate::batch::types::WindowType::Hybrid {
                max_messages,
                max_seconds,
            } => WindowConfig {
                max_entries: *max_messages,
                min_entries: 1,
                time_window_secs: *max_seconds,
                intent_shift_threshold: 0.6,
            },
            crate::batch::types::WindowType::Manual => WindowConfig {
                max_entries: usize::MAX,
                min_entries: 0,
                time_window_secs: u64::MAX,
                intent_shift_threshold: 0.6,
            },
        };

        let window = Arc::new(RwLock::new(SlidingWindow::new(window_config)));
        let mut trigger_system = TriggerSystem::new(config.triggers.clone(), window.clone());

        // Auto-register event types
        if let Some(ref mut emitter) = self.emitter {
            emitter.set_agent_config(&name, config.emit_on.clone());
        }
        if let Some(ref event_bus) = self.event_bus {
            trigger_system.listen_to(event_bus, vec![]);
        }

        let instance = BatchAgentInstance {
            name: name.clone(),
            config: config.clone(),
            status: BatchAgentStatus::Registered,
            window,
            trigger_system,
            metrics: BatchMetrics::default(),
        };

        self.agents.insert(name.clone(), instance);
        info!(agent = %name, domain = %config.business_domain, "Batch Agent registered");
        Ok(())
    }

    pub fn unregister(&mut self, name: &str) -> Result<(), BatchError> {
        if !self.agents.contains_key(name) {
            return Err(BatchError::AgentNotFound { name: name.into() });
        }
        self.agents.remove(name);
        debug!(agent = %name, "Batch Agent unregistered");
        Ok(())
    }

    pub async fn start(&mut self, name: Option<&str>) -> Result<(), BatchError> {
        if let Some(n) = name {
            let instance = self
                .agents
                .get_mut(n)
                .ok_or_else(|| BatchError::AgentNotFound { name: n.into() })?;
            if instance.status == BatchAgentStatus::Running {
                return Err(BatchError::AgentAlreadyRunning { name: n.into() });
            }
            instance.status = BatchAgentStatus::Running;
            if let Some(ref emitter) = self.emitter {
                emitter.emit_agent_started(n).await;
            }
            info!(agent = %n, "Batch Agent started");
        } else {
            let names: Vec<String> = self.agents.keys().cloned().collect();
            for n in names {
                if let Some(instance) = self.agents.get_mut(&n) {
                    if instance.status != BatchAgentStatus::Running {
                        instance.status = BatchAgentStatus::Running;
                        if let Some(ref emitter) = self.emitter {
                            emitter.emit_agent_started(&n).await;
                        }
                    }
                }
            }
            self.running = true;
            info!("All batch agents started");
        }
        Ok(())
    }

    pub async fn stop(&mut self, name: Option<&str>) -> Result<(), BatchError> {
        if let Some(n) = name {
            let instance = self
                .agents
                .get_mut(n)
                .ok_or_else(|| BatchError::AgentNotFound { name: n.into() })?;
            instance.status = BatchAgentStatus::Stopped;
            if let Some(ref emitter) = self.emitter {
                emitter.emit_agent_stopped(n, "manual_stop").await;
            }
        } else {
            let names: Vec<String> = self.agents.keys().cloned().collect();
            for n in names {
                if let Some(instance) = self.agents.get_mut(&n) {
                    instance.status = BatchAgentStatus::Stopped;
                    if let Some(ref emitter) = self.emitter {
                        emitter.emit_agent_stopped(&n, "manual_stop").await;
                    }
                }
            }
            self.running = false;
        }
        Ok(())
    }

    pub fn get_status(&self, name: &str) -> Option<BatchAgentStatus> {
        self.agents.get(name).map(|a| a.status.clone())
    }

    pub fn get_window_status(&self, name: &str) -> Option<crate::batch::types::WindowStatus> {
        self.agents.get(name).map(|a| a.window.read().status())
    }

    pub fn get_metrics(&self, name: &str) -> Option<BatchMetrics> {
        self.agents.get(name).map(|a| a.metrics.clone())
    }

    pub fn push_message(
        &mut self,
        agent_name: &str,
        entry: crate::batch::types::WindowEntry,
    ) -> Result<(), BatchError> {
        let instance =
            self.agents
                .get_mut(agent_name)
                .ok_or_else(|| BatchError::AgentNotFound {
                    name: agent_name.into(),
                })?;
        instance.window.write().push(entry)
    }

    /// Convert one bus event into window entries for the agents that explicitly
    /// declare it as a `CustomEvent` trigger.  This is deliberately opt-in:
    /// lifecycle and batch-output events must not be broadcast to every agent.
    /// Returns the affected agent names so a runtime adapter can evaluate and
    /// execute only those windows.
    pub fn enqueue_custom_event(&mut self, event: &Event) -> Vec<String> {
        let names: Vec<String> = self
            .agents
            .iter()
            .filter(|(_, instance)| {
                instance.status == BatchAgentStatus::Running
                    && instance.config.triggers.iter().any(|trigger| {
                        matches!(&trigger.trigger_type,
                            crate::batch::types::TriggerType::CustomEvent(event_type)
                            if event_type == &event.event_type)
                    })
            })
            .map(|(name, _)| name.clone())
            .collect();

        for name in &names {
            let entry = crate::batch::types::WindowEntry {
                message_id: event.event_id.clone(),
                role: "event".to_string(),
                content: event.payload.clone(),
                timestamp: event.timestamp,
                estimated_intent: None,
                metadata: std::collections::HashMap::from([
                    ("event_type".to_string(), event.event_type.clone()),
                    ("task_iri".to_string(), event.task_iri.clone()),
                    (
                        "source_agent_iri".to_string(),
                        event.source_agent_iri.clone(),
                    ),
                ]),
            };
            if let Err(error) = self.push_message(name, entry) {
                warn!(%error, agent = %name, "Failed to enqueue batch event");
            }
        }
        names
    }

    pub async fn evaluate_triggers(
        &self,
        agent_name: &str,
    ) -> Vec<crate::batch::types::TriggerReason> {
        match self.agents.get(agent_name) {
            Some(agent) => agent.trigger_system.evaluate().await,
            None => Vec::new(),
        }
    }

    /// Execute one ready batch using the configured extraction pipeline. This
    /// is the concrete consumption boundary used by event/cron adapters: it
    /// refuses stopped agents, drains exactly one window through the existing
    /// pipeline, runs the maintenance handler, and publishes outcomes.
    pub async fn execute_ready(
        &mut self,
        agent_name: &str,
        pipeline: &ExtractorPipeline,
        context: &crate::batch::types::PromptContext,
    ) -> Result<ExtractionResult, BatchError> {
        let (config, window_size) = {
            let instance =
                self.agents
                    .get(agent_name)
                    .ok_or_else(|| BatchError::AgentNotFound {
                        name: agent_name.into(),
                    })?;
            if instance.status != BatchAgentStatus::Running {
                return Err(BatchError::Internal {
                    message: format!("Batch agent '{}' is not running", agent_name),
                });
            }
            (
                instance.config.clone(),
                instance.window.read().status().entry_count,
            )
        };
        if window_size == 0 {
            return Err(BatchError::Internal {
                message: "Cannot execute an empty batch window".to_string(),
            });
        }
        if let Some(emitter) = &self.emitter {
            emitter
                .emit_extraction_started(agent_name, "pending", window_size)
                .await;
        }
        let result = {
            let instance = self
                .agents
                .get_mut(agent_name)
                .expect("agent validated above");
            // Drain under the lock, then release it before an async gateway
            // call. Holding a parking_lot guard across `.await` would block
            // producers and make the background consumer non-Send.
            instance.window.write().drain()
        };
        let result = pipeline.extract(&config, result, context).await?;
        if let Some(emitter) = &self.emitter {
            emitter.emit_extraction_complete(agent_name, &result).await;
        }
        let outcome = if config.apply_graph_mutations {
            self.run_maintenance_handler(agent_name, &result)
        } else {
            handlers::HandlerOutcome {
                handler_name: agent_name.to_string(),
                actions_taken: vec![
                    "Graph mutations skipped: apply_graph_mutations is disabled".to_string()
                ],
                ..Default::default()
            }
        };
        if let Some(emitter) = &self.emitter {
            for event in outcome.pending_events {
                let _ = emitter
                    .event_bus()
                    .emit(
                        &format!("batch://{}", agent_name),
                        &event.event_type,
                        &format!("batch:{}", agent_name),
                        &event.payload.to_string(),
                    )
                    .await;
            }
        }
        if let Some(instance) = self.agents.get_mut(agent_name) {
            instance
                .trigger_system
                .update_last_execution(chrono::Utc::now());
        }
        Ok(result)
    }

    pub fn drain_window(
        &mut self,
        agent_name: &str,
    ) -> Option<Vec<crate::batch::types::WindowEntry>> {
        self.agents
            .get_mut(agent_name)
            .map(|a| a.window.write().drain())
    }

    pub fn list_agents(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }

    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn emitter(&self) -> Option<&BatchEventEmitter> {
        self.emitter.as_ref()
    }

    /// Register the 8 standard maintenance agents from settings.
    pub fn register_maintenance_agents(
        &mut self,
        settings: &[BatchAgentSettings],
    ) -> Vec<Result<(), BatchError>> {
        let mut results = Vec::new();
        for s in settings {
            if !s.enabled {
                continue;
            }
            let window_type = s.window_type.as_deref().unwrap_or("manual");
            let config = BatchAgentConfig {
                name: s.name.clone(),
                description: s.description.clone(),
                enabled: true,
                window_type: match window_type.to_lowercase().as_str() {
                    "manual" => crate::batch::types::WindowType::Manual,
                    "hybrid" => crate::batch::types::WindowType::Hybrid {
                        max_messages: s.window_max_messages.unwrap_or(5),
                        max_seconds: s.window_max_seconds.unwrap_or(3600),
                    },
                    _ => crate::batch::types::WindowType::MessageCount(
                        s.window_max_messages.unwrap_or(5),
                    ),
                },
                triggers: s
                    .triggers
                    .iter()
                    .map(|t| {
                        let trigger_type = match t.trigger_type.to_lowercase().as_str() {
                            s if s.starts_with("cron") => {
                                crate::batch::types::TriggerType::CronSchedule(
                                    t.params.get("schedule").cloned().unwrap_or_default(),
                                )
                            }
                            "intent_shift" | "intent" => {
                                crate::batch::types::TriggerType::IntentShift
                            }
                            s if s.starts_with("message") => {
                                crate::batch::types::TriggerType::MessageThreshold(
                                    t.params
                                        .get("threshold")
                                        .and_then(|v| v.parse().ok())
                                        .unwrap_or(5),
                                )
                            }
                            s if s.starts_with("custom") => {
                                crate::batch::types::TriggerType::CustomEvent(
                                    t.params.get("event").cloned().unwrap_or_default(),
                                )
                            }
                            _ => crate::batch::types::TriggerType::WindowFull,
                        };
                        crate::batch::types::TriggerConfig {
                            trigger_type,
                            params: t.params.clone(),
                        }
                    })
                    .collect(),
                prompt_source: crate::batch::types::PromptSource::HybridWithTemplate,
                prompt_template_path: s.prompt_template_path.clone(),
                prompt_template_name: s
                    .prompt_template_name
                    .clone()
                    .or_else(|| Some(s.name.clone())),
                prompt_params: std::collections::HashMap::new(),
                business_domain: s.business_domain.clone(),
                entity_types: crate::batch::vocabulary::EntityTypeConfig {
                    vocabulary: s.entity_types.clone(),
                    custom: vec![],
                },
                relation_types: crate::batch::vocabulary::RelationTypeConfig {
                    vocabulary: s.relation_types.clone(),
                    custom: vec![],
                },
                intent_types: crate::batch::vocabulary::IntentTypeConfig {
                    vocabulary: s.intent_types.clone(),
                    custom: vec![],
                },
                model: s.model.clone(),
                temperature: s.temperature,
                max_retries: s.max_retries.unwrap_or(3),
                timeout_seconds: s.timeout_seconds.unwrap_or(120),
                emit_on: s
                    .emit_on
                    .iter()
                    .map(|e| match e.to_lowercase().as_str() {
                        "new_relation" => crate::batch::types::EmitCondition::NewRelation,
                        "intent_detected" => {
                            crate::batch::types::EmitCondition::IntentDetected(vec![])
                        }
                        "confidence_above" => {
                            crate::batch::types::EmitCondition::ConfidenceAbove(0.8)
                        }
                        _ => crate::batch::types::EmitCondition::Always,
                    })
                    .collect(),
                inject_user_reminders: s.inject_user_reminders,
                inject_context_summary: s.inject_context_summary,
                inject_related_entities: true,
                apply_graph_mutations: s.apply_graph_mutations,
            };
            results.push(self.register(config));
        }
        results
    }

    /// Run the post-extraction handler for a maintenance agent.
    /// This should be called after `ExtractorPipeline` produces a result for a maintenance agent.
    pub fn run_maintenance_handler(
        &self,
        agent_name: &str,
        result: &ExtractionResult,
    ) -> handlers::HandlerOutcome {
        match self.graph_store {
            Some(ref graph) => handlers::run_handler(agent_name, result, graph.as_ref()),
            None => {
                warn!(
                    "No SkillGraphStore available — maintenance handler skipped for {}",
                    agent_name
                );
                handlers::HandlerOutcome {
                    handler_name: agent_name.to_string(),
                    actions_taken: vec!["No graph store available".to_string()],
                    ..Default::default()
                }
            }
        }
    }
}

impl Default for BatchAgentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::types::{TriggerConfig, TriggerType, WindowType};
    use crate::core::event_bus::EventPriority;

    #[tokio::test]
    async fn custom_event_is_enqueued_only_for_matching_running_agents() {
        let mut manager = BatchAgentManager::new();
        let mut matching = BatchAgentConfig::default();
        matching.name = "matching".to_string();
        matching.window_type = WindowType::MessageCount(2);
        matching.triggers = vec![TriggerConfig {
            trigger_type: TriggerType::CustomEvent("WORKSPACE_FILE_MODIFIED".to_string()),
            params: HashMap::new(),
        }];
        manager.register(matching).unwrap();

        let mut other = BatchAgentConfig::default();
        other.name = "other".to_string();
        other.triggers = vec![TriggerConfig {
            trigger_type: TriggerType::CustomEvent("TASK_COMPLETED".to_string()),
            params: HashMap::new(),
        }];
        manager.register(other).unwrap();
        manager.start(None).await.unwrap();

        let event = Event {
            event_id: "evt-1".to_string(),
            task_iri: "task:1".to_string(),
            event_type: "WORKSPACE_FILE_MODIFIED".to_string(),
            source_agent_iri: "monitor".to_string(),
            payload: "src/lib.rs changed".to_string(),
            payload_json_ld: "src/lib.rs changed".to_string(),
            timestamp: chrono::Utc::now(),
            sequence: 1,
            type_mask: crate::core::event_bus::EventTypeMask::default(),
            priority: EventPriority::Normal,
        };

        assert_eq!(
            manager.enqueue_custom_event(&event),
            vec!["matching".to_string()]
        );
        assert_eq!(
            manager.get_window_status("matching").unwrap().entry_count,
            1
        );
        assert_eq!(manager.get_window_status("other").unwrap().entry_count, 0);
    }
}
