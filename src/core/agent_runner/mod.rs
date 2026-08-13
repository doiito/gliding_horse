use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use serde_json::Value;
use tracing::{info, warn};

use crate::causal::fused::FusedRootCauseEngine;
use crate::config::settings::AgentSettings;
use crate::core::constitution::ConstitutionRegistry;
use crate::core::context_compressor::{ContextWindowManager, ToolResultCompressor};
use crate::core::relevance_tracker::RelevanceTracker;
use crate::gateway::unified_gateway::{ChatMessage, UnifiedGateway};
use crate::memory::l0_store::L0Store;
use crate::memory::l2_blackboard::Blackboard;
use crate::memory::l3_projection::ProjectionEngine;
use crate::memory::memory_manager::MemoryManager;
use crate::memory::prefetch_engine::PrefetchEngine;
use crate::memory::scheduler::MemoryScheduler;
use crate::memory::EmbeddingService;
use crate::methodology::{
    evolution::{EvolutionEngine, EvolutionEngineHandle},
    gate::{MethodologyGate, MethodologyGateHandle},
    MethodologyRegistry,
};
use crate::root_cause::RootCauseEngine;
use crate::templates::template_engine::TemplateEngine;
use crate::tools::hooks::HookManager;
use crate::tools::sharing::SharingProtocol;
use crate::tools::skill_registry::SkillRegistry;
use crate::tools::tool_executor::ToolExecutor;
use crate::tools::tool_guard::ToolGuard;

mod execution;
mod prompt;
mod utils;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReActPhase {
    Thought,
    Action,
    Observation,
}

const LLM_RESPONSE_FORMAT_WITH_THOUGHT: &str = r#"
Return JSON: {"thought": "...", "content": "...", "summary": "...", "action": "tool_call|finish|continue", "emphasis": []}
- thought: Reasoning process
- summary: ≤50 char summary
- action: tool_call(invoke tool) / finish(task complete) / continue(continue reasoning)
- emphasis: Identified important constraints (array)

Example:
{"thought": "Need to create file", "content": "Create calculator.py", "summary": "Create main file", "action": "tool_call", "emphasis": []}
"#;

const LLM_RESPONSE_FORMAT_NO_THOUGHT: &str = r#"
Return JSON: {"content": "...", "summary": "...", "action": "tool_call|finish|continue", "emphasis": []}
- summary: ≤50 char summary
- action: tool_call(invoke tool) / finish(task complete) / continue(continue reasoning)
- emphasis: Identified important constraints (array)

Example:
{"content": "View file contents", "summary": "Read file", "action": "tool_call", "emphasis": []}
"#;

#[derive(Debug, Clone)]
pub struct TaskContext {
    pub task_iri: String,
    pub objective: String,
    pub parent_task_iri: Option<String>,
    pub input_data: HashMap<String, Value>,
    pub constraints: HashMap<String, String>,
    pub max_iterations: u32,
    pub prev_agent_summary: Option<String>,
    pub original_task: Option<String>,
    pub completed_steps: Vec<String>,
    pub pending_steps: Vec<String>,
    pub five_w2h_iri: String,
    pub five_w2h_snapshot: Option<crate::core::five_w2h::Task5W2H>,
    /// Historical messages restored from checkpoint, for resume mode
    pub resumed_messages: Option<Vec<ChatMessage>>,
    /// Turn count restored from checkpoint
    pub resumed_turn_count: u32,
    /// Tool call count restored from checkpoint
    pub resumed_tool_count: u32,
    /// JSON-LD workflow definition (optional, replaces LLM-generated plan)
    pub workflow_jsonld: Option<String>,
    /// Expected output (passed from PlanStep, for DA/CA reference)
    pub expected_output: String,
    /// Success criteria (passed from PlanStep, for DA/CA reference)
    pub success_criteria: String,
    /// PDCA cycle identifier for L2 blackboard filtered queries
    pub cycle_id: String,
    /// Summary of workspace file inventory (set by CodeCliEngine before passing to SA).
    /// Used by SA to decide verification-first routing when workspace has existing files.
    pub workspace_file_summary: Option<String>,
}

impl TaskContext {
    pub fn new(task_iri: &str, objective: &str, max_iterations: u32) -> Self {
        Self {
            task_iri: task_iri.to_string(),
            objective: objective.to_string(),
            parent_task_iri: None,
            input_data: HashMap::new(),
            constraints: HashMap::new(),
            max_iterations,
            prev_agent_summary: None,
            original_task: None,
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            five_w2h_iri: String::new(),
            five_w2h_snapshot: None,
            resumed_messages: None,
            resumed_turn_count: 0,
            resumed_tool_count: 0,
            workflow_jsonld: None,
            expected_output: String::new(),
            success_criteria: String::new(),
            cycle_id: String::new(),
            workspace_file_summary: None,
        }
    }

    pub fn with_cycle_id(mut self, cycle_id: &str) -> Self {
        self.cycle_id = cycle_id.to_string();
        self
    }

    pub fn with_step_info(mut self, expected_output: &str, success_criteria: &str) -> Self {
        self.expected_output = expected_output.to_string();
        self.success_criteria = success_criteria.to_string();
        self
    }

    /// Set JSON-LD workflow definition (replaces LLM-generated plan)
    pub fn with_workflow(mut self, jsonld: &str) -> Self {
        self.workflow_jsonld = Some(jsonld.to_string());
        self
    }

    pub fn with_prev_summary(mut self, summary: &str) -> Self {
        self.prev_agent_summary = Some(summary.to_string());
        self
    }

    pub fn with_original_task(mut self, task: &str) -> Self {
        self.original_task = Some(task.to_string());
        self
    }

    pub fn with_steps(mut self, completed: Vec<String>, pending: Vec<String>) -> Self {
        self.completed_steps = completed;
        self.pending_steps = pending;
        self
    }

    pub fn with_five_w2h(mut self, iri: &str, snapshot: crate::core::five_w2h::Task5W2H) -> Self {
        self.five_w2h_iri = iri.to_string();
        self.five_w2h_snapshot = Some(snapshot);
        if self.objective.is_empty() {
            self.objective = self
                .five_w2h_snapshot
                .as_ref()
                .map(|s| s.derive_objective())
                .unwrap_or_default();
        }
        self
    }

    /// Set historical messages restored from checkpoint (resume mode)
    pub fn with_resumed_messages(
        mut self,
        messages: Vec<ChatMessage>,
        turn_count: u32,
        tool_count: u32,
    ) -> Self {
        self.resumed_messages = Some(messages);
        self.resumed_turn_count = turn_count;
        self.resumed_tool_count = tool_count;
        self
    }

    pub fn add_completed_step(&mut self, step: &str) {
        self.completed_steps.push(step.to_string());
        if let Some(pos) = self.pending_steps.iter().position(|s| s == step) {
            self.pending_steps.remove(pos);
        }
    }

    /// Set workspace file inventory summary (from WorkspaceMonitor)
    pub fn with_workspace_summary(mut self, summary: &str) -> Self {
        self.workspace_file_summary = Some(summary.to_string());
        self
    }
}

impl Default for TaskContext {
    fn default() -> Self {
        Self {
            task_iri: String::new(),
            objective: String::new(),
            parent_task_iri: None,
            input_data: HashMap::new(),
            constraints: HashMap::new(),
            max_iterations: 20,
            prev_agent_summary: None,
            original_task: None,
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            five_w2h_iri: String::new(),
            five_w2h_snapshot: None,
            resumed_messages: None,
            resumed_turn_count: 0,
            resumed_tool_count: 0,
            workflow_jsonld: None,
            expected_output: String::new(),
            success_criteria: String::new(),
            cycle_id: String::new(),
            workspace_file_summary: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_iri: String,
    pub status: String,
    pub summary: String,
    pub output: Option<Value>,
    pub jsonld_output: Option<Value>,
    pub artifacts: Vec<Value>,
    pub errors: Vec<String>,
    pub turn_count: u32,
    pub tool_call_count: u32,
    pub five_w2h_updates: Option<serde_json::Value>,
    pub tracked_actions: Vec<crate::core::tracked_action::TrackedAction>,
    pub archive_iri: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlmParsedResponse {
    pub thought: Option<String>,
    pub content: String,
    pub summary: Option<String>,
    pub action: Option<String>,
    pub is_valid_json: bool,
    pub has_native_reasoning: bool,
    pub emphasis: Vec<String>,
}

#[derive(Clone)]
pub struct AgentRunner {
    pub gateway: Arc<UnifiedGateway>,
    pub skills: Arc<SkillRegistry>,
    pub blackboard: Arc<Blackboard>,
    pub l0_store: Arc<L0Store>,
    pub memory_manager: Arc<tokio::sync::Mutex<MemoryManager>>,
    pub templates: Arc<TemplateEngine>,
    pub tool_executor: Arc<parking_lot::RwLock<ToolExecutor>>,
    pub agent_settings: AgentSettings,
    pub hook_manager: Arc<HookManager>,
    pub projection: Arc<ProjectionEngine>,
    pub sharing: Arc<SharingProtocol>,
    pub emphasis_config: Option<crate::config::settings::EmphasisConfig>,
    pub event_bus: Option<Arc<crate::core::event_bus::EventBus>>,
    pub scheduler: Option<Arc<MemoryScheduler>>,
    pub prefetch_engine: Option<Arc<PrefetchEngine>>,
    pub unified_graph_store: Option<Arc<oxigraph::store::Store>>,
    pub tool_controller: Option<crate::core::tool_controller::ToolController>,
    pub total_prompt_tokens: Arc<AtomicU64>,
    pub total_completion_tokens: Arc<AtomicU64>,
    /// Prompt/completion token count from the last API call (non-cumulative, stores only the latest round)
    pub last_prompt_tokens: Arc<AtomicU64>,
    pub last_completion_tokens: Arc<AtomicU64>,
    pub tool_result_compressor: Option<Arc<std::sync::Mutex<ToolResultCompressor>>>,
    pub tool_result_aging: Option<crate::core::ToolResultAging>,
    pub context_window_manager: Option<Arc<std::sync::Mutex<ContextWindowManager>>>,
    pub prompt_loader: Option<Arc<crate::core::prompt_loader::PromptLoader>>,
    pub methodology_gate: Option<MethodologyGateHandle>,
    pub root_cause_engine: Option<Arc<RootCauseEngine>>,
    /// Supplementary input store (SA writes → AgentRunner consumes at CycleStart)
    pub supplement_store: crate::core::supplementary_store::SupplementaryInputStore,
    /// Perception content store (system components write → injected into messages header during exec() initial assembly)
    pub perception_store: crate::core::perception_store::PerceptionStore,
    /// Embedding service (for computing turn embedding and relevance_score)
    pub embedder: Option<Arc<dyn EmbeddingService>>,
    /// Relevance tracker (computes semantic relevance between each turn and the task)
    pub relevance_tracker: Option<Arc<std::sync::Mutex<RelevanceTracker>>>,
    /// Workspace root directory path (all Agent file operations are restricted to this scope)
    pub workspace_root: Option<PathBuf>,
    /// Causal engine for root-cause analysis of task failures and dimension audit observations
    pub causal_engine: Option<Arc<crate::causal::CausalEngine>>,
    /// Skill graph store — the cognitive network of registered skills
    pub skill_graph_store: Option<Arc<crate::skill_graph::graph_store::SkillGraphStore>>,
}

impl AgentRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gateway: Arc<UnifiedGateway>,
        skills: Arc<SkillRegistry>,
        blackboard: Arc<Blackboard>,
        l0_store: Arc<L0Store>,
        memory_manager: Arc<tokio::sync::Mutex<MemoryManager>>,
        templates: Arc<TemplateEngine>,
        agent_settings: AgentSettings,
    ) -> Self {
        let projection = Arc::new(ProjectionEngine::new(
            blackboard.clone(),
            agent_settings.max_projection_size,
        ));
        let sharing = Arc::new(SharingProtocol::new());
        let hook_manager = Arc::new(HookManager::new());
        ToolGuard::new().register_hooks(&hook_manager);

        // Initialize MethodologyGate with constitution bindings + EvolutionEngine
        let methodology_gate = {
            let mut registry = MethodologyRegistry::new();
            registry.load_bundled_nodes();
            let mut gate = MethodologyGate::new(registry, agent_settings.max_active);
            gate.register_constitution_bindings(&ConstitutionRegistry::new());
            let evolution = EvolutionEngineHandle::new(EvolutionEngine::new());
            let handle = MethodologyGateHandle::new(gate).with_evolution(evolution);
            handle.register_hooks(&hook_manager);
            Some(handle)
        };

        // Conditionally initialize RootCauseEngine (lightweight, always-on by default)
        let root_cause_engine = {
            let engine = Arc::new(RootCauseEngine::default());
            engine.register_hooks(&hook_manager, "agent");
            Some(engine)
        };

        let mut runner = Self {
            gateway,
            skills,
            blackboard,
            l0_store,
            memory_manager,
            templates,
            tool_executor: {
                let mut exe = ToolExecutor::new();
                exe.set_projection_engine(projection.clone());
                Arc::new(parking_lot::RwLock::new(exe))
            },
            agent_settings,
            hook_manager,
            projection,
            sharing,
            emphasis_config: None,
            event_bus: None,
            scheduler: None,
            prefetch_engine: None,
            unified_graph_store: None,
            tool_controller: None,
            total_prompt_tokens: Arc::new(AtomicU64::new(0)),
            total_completion_tokens: Arc::new(AtomicU64::new(0)),
            last_prompt_tokens: Arc::new(AtomicU64::new(0)),
            last_completion_tokens: Arc::new(AtomicU64::new(0)),
            tool_result_compressor: None,
            tool_result_aging: None,
            context_window_manager: None,
            prompt_loader: None,
            methodology_gate,
            root_cause_engine,
            supplement_store: crate::core::supplementary_store::SupplementaryInputStore::new(),
            perception_store: crate::core::perception_store::PerceptionStore::new(),
            embedder: None,
            relevance_tracker: None,
            workspace_root: None,
            causal_engine: None,
            skill_graph_store: None,
        };
        runner.init_context_compressors();
        runner
    }

    fn init_context_compressors(&mut self) {
        use crate::config::settings::{
            ContextWindowSettings, ToolResultAgingSettings, ToolResultCompressorSettings,
        };
        let trc_settings = ToolResultCompressorSettings::default();
        if trc_settings.enabled {
            self.tool_result_compressor = Some(Arc::new(std::sync::Mutex::new(
                ToolResultCompressor::new(&trc_settings),
            )));
        }
        let aging_settings = ToolResultAgingSettings::default();
        if aging_settings.enabled {
            self.tool_result_aging = Some(crate::core::ToolResultAging::new(&aging_settings));
        }
        let cwm_settings = ContextWindowSettings::default();
        if cwm_settings.max_messages > 0 {
            self.context_window_manager = Some(Arc::new(std::sync::Mutex::new(
                ContextWindowManager::new(&cwm_settings),
            )));
        }
    }

    pub fn with_scheduler(mut self, scheduler: Arc<MemoryScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    pub fn with_prefetch_engine(mut self, prefetch_engine: Arc<PrefetchEngine>) -> Self {
        self.prefetch_engine = Some(prefetch_engine);
        self
    }

    pub fn with_unified_graph_store(mut self, store: Arc<oxigraph::store::Store>) -> Self {
        if let Some(ref gate) = self.methodology_gate {
            let g = gate.inner();
            let guard = g.read();
            let kg = match crate::knowledge_graph::store::KnowledgeGraphStore::with_shared_store(
                store.clone(),
            ) {
                Err(e) => {
                    warn!("Failed to create KG for methodology seed: {}", e);
                    self.unified_graph_store = Some(store);
                    return self;
                }
                Ok(kg) => kg,
            };
            for m in guard.registry().all() {
                let quads = m.to_kg_quads();
                if let Err(e) = kg.write_quads(&quads, "graph:methodology") {
                    warn!("Failed to seed methodology {} into KG: {}", m.id, e);
                }
            }
            info!(
                "Seeded {} methodology definitions into knowledge graph",
                guard.registry().all().len()
            );
        }
        self.unified_graph_store = Some(store);
        self
    }

    pub fn with_tool_controller(
        mut self,
        tc: crate::core::tool_controller::ToolController,
    ) -> Self {
        self.tool_controller = Some(tc);
        self
    }

    pub fn with_emphasis_config(mut self, config: crate::config::settings::EmphasisConfig) -> Self {
        self.emphasis_config = Some(config);
        self
    }

    pub fn with_prompt_loader(mut self, loader: crate::core::prompt_loader::PromptLoader) -> Self {
        self.prompt_loader = Some(Arc::new(loader));
        self
    }

    /// Set the workspace root directory for all agents.
    /// When set, file operations (read/write/edit/search/exec) are restricted to this directory.
    /// The workspace path is also injected into the system prompt so agents know their boundary.
    pub fn with_workspace_root(mut self, root: PathBuf) -> Self {
        self.workspace_root = Some(root);
        self
    }

    pub fn with_hook_manager(mut self, hook_manager: HookManager) -> Self {
        self.hook_manager = Arc::new(hook_manager);
        self
    }

    /// Load ToolGuard rules from a JSON config file.
    /// The guard is registered into the hook_manager on the next `execute` call.
    /// Default rules are used for categories not specified in the file.
    pub fn with_tool_guard_config<P: AsRef<std::path::Path>>(self, path: P) -> Self {
        match ToolGuard::from_json(path) {
            Ok(guard) => {
                guard.register_hooks(&self.hook_manager);
            }
            Err(e) => {
                warn!("Failed to load ToolGuard config: {}, using defaults", e);
                ToolGuard::new().register_hooks(&self.hook_manager);
            }
        }
        self
    }

    pub fn set_event_bus(&mut self, event_bus: Arc<crate::core::event_bus::EventBus>) {
        self.event_bus = Some(event_bus);
    }

    /// Set supplementary input store (injected by SA during creation, ensures SA and AgentRunner share the same instance)
    pub fn with_supplement_store(
        mut self,
        store: crate::core::supplementary_store::SupplementaryInputStore,
    ) -> Self {
        self.supplement_store = store;
        self
    }

    /// Set up active perception store (system components like WorkspaceMonitor/BatchAgent write perception data)
    pub fn with_perception_store(
        mut self,
        store: crate::core::perception_store::PerceptionStore,
    ) -> Self {
        self.perception_store = store;
        self
    }

    /// Set up embedding service + relevance tracker
    pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingService>) -> Self {
        self.embedder = Some(embedder);
        self.relevance_tracker = Some(Arc::new(std::sync::Mutex::new(RelevanceTracker::new(0.6))));
        self
    }

    /// Upgrade RootCauseEngine with a three-dimensional fusion engine
    /// (structural dependency-graph BFS + semantic SPARQL neighbor traversal).
    /// Call this before finalize_setup() to ensure hooks are properly registered.
    pub fn with_fused_root_cause_engine(mut self, fused: FusedRootCauseEngine) -> Self {
        let mut engine = RootCauseEngine::default();
        engine = engine.with_fused_engine(fused);
        let engine = Arc::new(engine);
        engine.register_hooks(&self.hook_manager, "agent");
        self.root_cause_engine = Some(engine);
        self
    }

    /// Attach a CausalEngine for root-cause analysis of task failures and dimension audit observations.
    pub fn with_causal_engine(mut self, engine: Arc<crate::causal::CausalEngine>) -> Self {
        self.causal_engine = Some(engine);
        self
    }

    /// Attach a SkillGraphStore — the cognitive network — for skill-related operations.
    pub fn with_skill_graph_store(
        mut self,
        store: Arc<crate::skill_graph::graph_store::SkillGraphStore>,
    ) -> Self {
        self.skill_graph_store = Some(store);
        self
    }

    /// Complete initialization wiring: connect AgentRunner's perception_store to WorkspaceMonitor.
    /// Called once after AgentRunner construction and all sub-components are ready.
    pub fn finalize_setup(&self) {
        let executor = self.tool_executor.read();
        if let Some(wm) = executor.get_workspace_monitor() {
            wm.set_perception_store(Arc::new(self.perception_store.clone()));
        }
    }
}

#[cfg(test)]
mod tests;
