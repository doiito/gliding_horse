use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::broadcast;

use crate::core::agent_runner::AgentRunner;
use crate::core::event_bus::{Event, EventBus};
use crate::core::perception_store::PerceptionStore;
use crate::core::policy_learning::ConstrainedPolicy;
use crate::core::relevance_tracker::RelevanceTracker;
use crate::core::supplementary_store::SupplementaryInputStore;
use crate::jsonld::type_router::TypeRouter;
use crate::memory::hyperspace_store::HyperspaceStore;
use crate::memory::l2_blackboard::Blackboard;
use crate::memory::prefetch_engine::PrefetchEngine;
use crate::memory::scheduler::MemoryScheduler;
use crate::memory::EmbeddingService;
#[cfg(feature = "ontology")]
use crate::ontology_bridge::OntologyBridgeManager;
use crate::perception::proactive_engine::ProactiveEngine;
use crate::skill_graph::discovery::SkillDiscoveryEngine;
use crate::templates::template_engine::TemplateEngine;
use crate::tools::sharing::SharingProtocol;
use crate::tools::skill_registry::SkillRegistry;
use crate::tools::tool_executor::ToolExecutor;

use super::types::CycleState;

pub struct SupervisorAgent {
    pub(super) runner: Arc<AgentRunner>,
    /// Planned: used for SA prompt generation and skill discovery.
    #[allow(dead_code)]
    pub(super) template_engine: Arc<TemplateEngine>,
    #[allow(dead_code)]
    pub(super) skills: Arc<SkillRegistry>,
    pub(super) event_bus: Arc<EventBus>,
    pub(super) event_receiver: Option<broadcast::Receiver<Event>>,
    pub(super) active_cycles: HashMap<String, CycleState>,
    pub(super) max_iterations: u32,
    /// Maximum PDCA cycle re-entry count (for Recursive tasks, default 7)
    pub(super) max_pdca_cycles: u32,
    pub(super) perception: ProactiveEngine,
    pub(super) sharing: Arc<SharingProtocol>,
    pub(super) blackboard: Option<Arc<Blackboard>>,
    pub(super) prefetch_engine: Option<Arc<PrefetchEngine>>,
    pub(super) scheduler: Option<Arc<MemoryScheduler>>,
    pub(super) type_router: TypeRouter,
    pub(super) pending_approvals: Arc<tokio::sync::Mutex<HashMap<String, bool>>>,
    pub(super) supplementary_inputs: HashMap<String, Vec<(String, String)>>,
    /// Supplementary input shared store, written by SA and consumed by AgentRunner at CycleStart
    pub(super) supplement_store: SupplementaryInputStore,
    /// Embedding service (optional, for computing supplementary input embeddings and relevance scores)
    pub(super) embedder: Option<Arc<dyn EmbeddingService>>,
    /// Relevance tracker
    pub(super) relevance_tracker: RelevanceTracker,
    /// Timeout for intervention/LLM execution (seconds)
    pub(super) execution_timeout_secs: u64,
    /// Human approval wait window (seconds) before falling back to the default
    pub(super) approval_wait_secs: u64,
    /// Skill discovery engine for semantic skill search during planning
    pub(super) discovery_engine: Option<Arc<SkillDiscoveryEngine>>,
    /// Constrained contextual policy; it may rank hints but never overrides
    /// CA/AA, governance, security, or terminal-state rules.
    pub(super) policy_learning: ConstrainedPolicy,
    /// Whether accumulated experience is injected and allowed to update the
    /// online policy for this execution.
    pub(super) learning_mode: crate::core::policy_learning::LearningMode,
}

impl SupervisorAgent {
    pub fn new(
        runner: Arc<AgentRunner>,
        template_engine: Arc<TemplateEngine>,
        skills: Arc<SkillRegistry>,
        event_bus: Arc<EventBus>,
        max_iterations: u32,
    ) -> Self {
        Self::with_pdca_cycles(
            runner,
            template_engine,
            skills,
            event_bus,
            max_iterations,
            7,
        )
    }

    pub fn with_pdca_cycles(
        mut runner: Arc<AgentRunner>,
        template_engine: Arc<TemplateEngine>,
        skills: Arc<SkillRegistry>,
        event_bus: Arc<EventBus>,
        max_iterations: u32,
        max_pdca_cycles: u32,
    ) -> Self {
        // Wire up event bus on runner so it can emit detailed execution events
        // (TOOL_CALL, TOOL_RESULT, THOUGHT) during the ReAct loop.
        if let Some(r) = Arc::get_mut(&mut runner) {
            r.set_event_bus(event_bus.clone());
        }

        let event_bus_for_perception = event_bus.clone();
        // Create supplementary input shared store and inject into AgentRunner (ensure SA and Runner share the same instance)
        let supplement_store = SupplementaryInputStore::new();
        if let Some(r) = Arc::get_mut(&mut runner) {
            r.supplement_store = supplement_store.clone();
        }
        Self {
            runner: runner.clone(),
            template_engine,
            skills,
            event_receiver: Some(event_bus.subscribe()),
            event_bus,
            active_cycles: HashMap::new(),
            max_iterations,
            max_pdca_cycles,
            perception: ProactiveEngine::new(runner.l0_store.clone(), event_bus_for_perception),
            sharing: Arc::new(SharingProtocol::new()),
            blackboard: None,
            prefetch_engine: None,
            scheduler: None,
            type_router: TypeRouter::new(),
            pending_approvals: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            supplementary_inputs: HashMap::new(),
            supplement_store,
            embedder: None,
            relevance_tracker: RelevanceTracker::new(0.6),
            execution_timeout_secs: 30,
            approval_wait_secs: 5,
            discovery_engine: None,
            policy_learning: ConstrainedPolicy::default().with_persistence(runner.l0_store.clone()),
            learning_mode: crate::core::policy_learning::LearningMode::Active,
        }
    }

    pub fn with_execution_timeout(mut self, secs: u64) -> Self {
        self.execution_timeout_secs = secs;
        self
    }

    pub fn with_approval_wait_secs(mut self, secs: u64) -> Self {
        self.approval_wait_secs = secs;
        self
    }

    pub fn with_memory(
        mut self,
        blackboard: Option<Arc<Blackboard>>,
        prefetch_engine: Option<Arc<PrefetchEngine>>,
        scheduler: Option<Arc<MemoryScheduler>>,
    ) -> Self {
        self.blackboard = blackboard;
        self.prefetch_engine = prefetch_engine;
        self.scheduler = scheduler;
        self
    }

    /// Set embedding service (for computing supplementary input embeddings and relevance scores)
    /// Also propagates to AgentRunner so it can compute turn embeddings in the ReAct loop
    pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingService>) -> Self {
        self.embedder = Some(embedder.clone());
        // Propagate to AgentRunner
        if let Some(runner) = Arc::get_mut(&mut self.runner) {
            *runner = runner.clone().with_embedder(embedder);
        }
        self
    }

    /// Attach HyperspaceStore to the perception engine for semantic experience retrieval.
    pub fn with_perception_hyperspace(mut self, store: Arc<HyperspaceStore>) -> Self {
        self.perception.hyperspace = Some(store);
        self
    }

    /// Attach OntologyBridge to the perception engine for dual-space embedding writes.
    #[cfg(feature = "ontology")]
    pub fn with_perception_ontology_bridge(mut self, ob: Arc<OntologyBridgeManager>) -> Self {
        self.perception = self.perception.with_ontology_bridge(ob);
        self
    }

    /// Attach a PerceptionStore so the proactive engine can inject conflict alerts
    /// and anomaly perception data visible to the agent during execution.
    pub fn with_perception_store(mut self, store: Arc<PerceptionStore>) -> Self {
        self.perception.perception_store = Some(store);
        self
    }

    /// Attach SkillDiscoveryEngine for semantic skill search during task planning.
    /// When set, the SA will enrich the agent's context with relevant skill suggestions.
    pub fn with_discovery_engine(mut self, engine: Arc<SkillDiscoveryEngine>) -> Self {
        self.discovery_engine = Some(engine);
        self
    }

    pub fn with_learning_mode(mut self, mode: crate::core::policy_learning::LearningMode) -> Self {
        self.learning_mode = mode;
        self
    }

    pub fn with_policy_learning_settings(
        mut self,
        settings: &crate::config::settings::PolicyLearningSettings,
    ) -> Self {
        self.policy_learning = self
            .policy_learning
            .with_min_observations(settings.candidate_trial_min_baseline_samples)
            .with_gate(crate::core::policy_learning::PolicyGate {
                min_samples: settings.promotion_min_samples,
                min_improvement: settings.promotion_min_improvement,
            });
        self
    }

    /// Get supplementary input shared store (for AgentRunner injection)
    pub fn supplement_store(&self) -> SupplementaryInputStore {
        self.supplement_store.clone()
    }

    /// Update default model in gateway (uses RwLock interior mutability, no &mut self needed)
    /// This avoids redb file lock conflicts from rebuilding the entire Engine/L0Store
    pub fn set_model(&self, model: &str) {
        self.runner.gateway.set_default_model(model.to_string());
        for task_type in &["planning", "execution", "analysis", "default"] {
            self.runner
                .gateway
                .set_model_mapping(task_type.to_string(), model.to_string());
        }
    }

    pub fn set_api_key(&self, key: &str) {
        self.runner.gateway.set_api_key(key.to_string());
    }

    pub fn set_base_url(&self, url: &str) {
        self.runner.gateway.set_base_url(url.to_string());
    }

    pub fn blackboard(&self) -> Option<&Arc<Blackboard>> {
        self.blackboard.as_ref()
    }

    /// Access to the runner's ToolExecutor for runtime tool registration
    /// (e.g. MCP tools discovered after a lazy connection).
    pub fn tool_executor(&self) -> Arc<parking_lot::RwLock<ToolExecutor>> {
        self.runner.tool_executor.clone()
    }
}
