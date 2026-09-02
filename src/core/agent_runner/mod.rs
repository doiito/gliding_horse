use std::collections::{HashMap, HashSet};
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
pub(crate) use execution::{CA_DA_CORRECTION_MODE, SA_RECOVERY_MODE_CONSTRAINT};
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

/// Application-declared workspace context policy. The generic kernel does
/// not infer whether a task belongs to a mounted workspace.
pub const WORKSPACE_CONTEXT_SCOPE_CONSTRAINT: &str = "workspace_context_scope";
pub const WORKSPACE_CONTEXT_DISABLED: &str = "disabled";

/// Application-declared delivery boundary. The generic kernel enforces the
/// declared mode but does not infer a domain-specific deliverable from words
/// such as "report", "build", or "output".
pub const DELIVERY_MODE_CONSTRAINT: &str = "delivery_mode";
pub const DELIVERY_MODE_DIRECT_RESPONSE: &str = "direct_response";
/// A workspace-scoped artifact is the user-visible deliverable.  The path is
/// carried separately so applications can resolve it relative to their own
/// workspace root without teaching the kernel about application paths.
pub const DELIVERY_MODE_WORKSPACE_ARTIFACT: &str = "workspace_artifact";
pub const DELIVERY_TARGET_PATH_CONSTRAINT: &str = "delivery_target_path";

/// Application-declared external evidence capability. The kernel exposes and
/// enforces the generic capability; each application decides whether a task
/// actually requires current web evidence.
pub const REQUIRED_CAPABILITY_CONSTRAINT: &str = "required_capability";
pub const REQUIRED_CAPABILITY_WEB_RESEARCH: &str = "web_research";

pub(crate) fn direct_response_delivery_contract(
    constraints: &HashMap<String, String>,
) -> Option<&'static str> {
    constraints
        .get(DELIVERY_MODE_CONSTRAINT)
        .is_some_and(|mode| mode == DELIVERY_MODE_DIRECT_RESPONSE)
        .then_some(
            "Delivery mode is direct_response: the final deliverable must be returned in the agent response. A filesystem path, file artifact, or invented graph IRI is neither required nor valid acceptance evidence unless the original user request explicitly requires one.",
        )
}

pub(crate) fn workspace_artifact_delivery_contract(
    constraints: &HashMap<String, String>,
) -> Option<String> {
    (constraints
        .get(DELIVERY_MODE_CONSTRAINT)
        .is_some_and(|mode| mode == DELIVERY_MODE_WORKSPACE_ARTIFACT))
    .then(|| {
        let target = constraints
            .get(DELIVERY_TARGET_PATH_CONSTRAINT)
            .map(String::as_str)
            .unwrap_or("deliverable.md");
        format!(
            "Delivery mode is workspace_artifact: DA must create the complete final deliverable at workspace-relative path `{target}` with file_write. CA must read that exact file and verify its format and requested content. The final response must report the verified path; a chat-only answer is incomplete."
        )
    })
}

pub(crate) fn required_capability_contract(
    constraints: &HashMap<String, String>,
) -> Option<&'static str> {
    constraints
        .get(REQUIRED_CAPABILITY_CONSTRAINT)
        .is_some_and(|capability| capability == REQUIRED_CAPABILITY_WEB_RESEARCH)
        .then_some(
            "Current external evidence is required: use web_search for source discovery and web_fetch/http_request for targeted source reading before relying on RAG, KG, or model memory. If live retrieval is unavailable, state that limitation explicitly and do not present remembered or synthesized claims as newly verified facts.",
        )
}

/// AgentTurn identities must be unique inside a task. Each BizAgent owns a
/// distinct L1 session, so adding that session to the path prevents PA/DA/CA/
/// AA and later PDCA cycles from overwriting one another while preserving the
/// task prefix and the familiar turn suffix.
pub(crate) fn agent_turn_iri(task_iri: &str, session_id: &str, turn: u32) -> String {
    let task_id = task_iri
        .strip_prefix("iri://task/")
        .unwrap_or_else(|| task_iri.strip_prefix("iri://").unwrap_or(task_iri));
    format!("iri://task/{task_id}/session/{session_id}/turn_{turn}")
}

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
    /// Validated, versioned checkpoint state.  This carries phase and
    /// orchestration facts that cannot be reconstructed from chat messages.
    pub resumed_state: Option<crate::core::checkpoint::TaskResumeState>,
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
    /// Current-task paths eligible as independent verification evidence. This
    /// is deliberately distinct from the process-wide workspace inventory.
    pub workspace_evidence_paths: Vec<String>,
    /// Tool allowlist for this execution (None = all tools allowed)
    pub allowed_tools: Option<Vec<String>>,
    /// Generic execution-effect contract. Domain applications classify their
    /// tasks into this protocol; the kernel never infers domain semantics.
    pub effect_policy: crate::core::effect::EffectPolicy,
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
            resumed_state: None,
            workflow_jsonld: None,
            expected_output: String::new(),
            success_criteria: String::new(),
            cycle_id: String::new(),
            workspace_file_summary: None,
            workspace_evidence_paths: Vec::new(),
            allowed_tools: None,
            effect_policy: crate::core::effect::EffectPolicy::None,
        }
    }

    pub fn with_cycle_id(mut self, cycle_id: &str) -> Self {
        self.cycle_id = cycle_id.to_string();
        self
    }

    /// Build a bounded recall query from semantic task inputs. `task_iri`
    /// remains correlation metadata and is never used as default query text.
    pub fn context_recall_query(&self) -> crate::memory::ContextRecallQuery {
        let what = self
            .five_w2h_snapshot
            .as_ref()
            .map(|snapshot| snapshot.what.as_str())
            .unwrap_or("");
        let why = self
            .five_w2h_snapshot
            .as_ref()
            .map(|snapshot| snapshot.why.description.as_str())
            .unwrap_or("");
        crate::memory::ContextRecallQuery::from_fields(
            &self.task_iri,
            [
                ("objective", self.objective.as_str()),
                ("original_task", self.original_task.as_deref().unwrap_or("")),
                ("five_w2h_what", what),
                ("five_w2h_why", why),
                ("expected_output", self.expected_output.as_str()),
                ("success_criteria", self.success_criteria.as_str()),
            ],
        )
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

    /// Carry application-declared execution constraints through SA into each
    /// BizAgent context. The kernel interprets only generic constraint keys;
    /// domain applications decide when those constraints apply.
    pub fn with_constraints(mut self, constraints: HashMap<String, String>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn with_constraint(mut self, key: &str, value: &str) -> Self {
        self.constraints.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_effect_policy(mut self, policy: crate::core::effect::EffectPolicy) -> Self {
        self.effect_policy = policy;
        self
    }

    pub fn effective_effect_policy(&self) -> crate::core::effect::EffectPolicy {
        if self.effect_policy != crate::core::effect::EffectPolicy::None {
            self.effect_policy.clone()
        } else {
            crate::core::effect::EffectPolicy::from_legacy_constraints(&self.constraints)
        }
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

    /// Restore a task from the canonical checkpoint reader.  The message
    /// history remains available to the model while the structured state is
    /// forwarded to SA for phase and counter restoration.
    pub fn with_resumed_checkpoint(
        mut self,
        messages: Vec<ChatMessage>,
        state: crate::core::checkpoint::TaskResumeState,
    ) -> Self {
        self.resumed_turn_count = state.turn;
        self.resumed_tool_count = state.tool_call_count;
        self.resumed_messages = Some(messages);
        self.resumed_state = Some(state);
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

    pub fn with_workspace_evidence_paths(mut self, paths: Vec<String>) -> Self {
        self.workspace_evidence_paths = paths;
        self
    }

    pub fn workspace_context_enabled(&self) -> bool {
        self.constraints
            .get(WORKSPACE_CONTEXT_SCOPE_CONSTRAINT)
            .is_none_or(|scope| scope != WORKSPACE_CONTEXT_DISABLED)
    }

    pub fn requires_web_research(&self) -> bool {
        self.constraints
            .get(REQUIRED_CAPABILITY_CONSTRAINT)
            .is_some_and(|capability| capability == REQUIRED_CAPABILITY_WEB_RESEARCH)
    }

    pub fn with_allowed_tools(mut self, tools: Vec<String>) -> Self {
        // None means unrestricted by this layer; Some(empty) is an explicit
        // deny-all capability set (used by decision-only BizAgents such as AA).
        self.allowed_tools = Some(tools);
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
            resumed_state: None,
            workflow_jsonld: None,
            expected_output: String::new(),
            success_criteria: String::new(),
            cycle_id: String::new(),
            workspace_file_summary: None,
            workspace_evidence_paths: Vec::new(),
            allowed_tools: None,
            effect_policy: crate::core::effect::EffectPolicy::None,
        }
    }
}

/// Structured task outcome verdict — decoupled from the human-readable status string.
/// The `finish` action historically flattened a verdict into `status: "success"`,
/// losing blocked/failed intent; this enum preserves it so consumers (e.g. SA
/// verify-first logic) can react honestly instead of re-parsing summary text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskVerdict {
    Success,
    PartialSuccess,
    Failed,
    Timeout,
    Blocked,
}

impl TaskVerdict {
    /// Maps back to the legacy status string so existing consumers keep working.
    pub fn to_status_str(self) -> &'static str {
        match self {
            TaskVerdict::Success => "success",
            TaskVerdict::PartialSuccess => "partial_success",
            TaskVerdict::Failed => "failed",
            TaskVerdict::Timeout => "timeout",
            TaskVerdict::Blocked => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_iri: String,
    pub status: String,
    pub verdict: Option<TaskVerdict>,
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
    /// Token-optimization settings (compressor/aging/context-window tuning).
    /// Defaults match the historical hardcoded values when not provided.
    pub token_optimization: crate::config::settings::TokenOptimizationSettings,
    pub tool_result_router_settings: crate::config::settings::ToolResultRouterSettings,
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
    /// Optional application-level contract layered below kernel policy and
    /// above role/task context.
    pub application_prompt: Option<crate::core::prompt_contract::ApplicationPromptProfile>,
    /// Prompt experiment arm. Defaults to the optimized contract; set
    /// GLIDING_PROMPT_VARIANT=baseline for a controlled A/B comparison.
    pub prompt_variant: crate::core::prompt_contract::PromptVariant,
    pub methodology_gate: Option<MethodologyGateHandle>,
    pub root_cause_engine: Option<Arc<RootCauseEngine>>,
    /// Supplementary input store (SA writes → AgentRunner consumes at CycleStart)
    pub supplement_store: crate::core::supplementary_store::SupplementaryInputStore,
    /// At most one bounded archival write may be in flight. A stalled embedded
    /// database writer must degrade archival rather than freeze every agent
    /// turn that follows it.
    pub l0_archive_gate: Arc<tokio::sync::Semaphore>,
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
    /// Continuous-learning experiment arm. This is execution-wide so every
    /// BizAgent in one SA task observes the same causal treatment.
    pub learning_mode: crate::core::policy_learning::LearningMode,
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
            l0_store: l0_store.clone(),
            memory_manager,
            templates,
            tool_executor: {
                let mut exe = ToolExecutor::new();
                exe.set_projection_engine(projection.clone());
                exe.set_archived_result_store(l0_store.clone());
                Arc::new(parking_lot::RwLock::new(exe))
            },
            agent_settings,
            token_optimization: crate::config::settings::TokenOptimizationSettings::default(),
            tool_result_router_settings: crate::config::settings::ToolResultRouterSettings::default(
            ),
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
            application_prompt: None,
            prompt_variant: crate::core::prompt_contract::PromptVariant::from_env(),
            learning_mode: crate::core::policy_learning::LearningMode::Active,
            methodology_gate,
            root_cause_engine,
            supplement_store: crate::core::supplementary_store::SupplementaryInputStore::new(),
            l0_archive_gate: Arc::new(tokio::sync::Semaphore::new(1)),
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
        self.tool_result_compressor = None;
        self.tool_result_aging = None;
        self.context_window_manager = None;
        let trc_settings = &self.token_optimization.tool_result_compressor;
        if trc_settings.enabled {
            self.tool_result_compressor = Some(Arc::new(std::sync::Mutex::new(
                ToolResultCompressor::new(trc_settings),
            )));
        }
        let aging_settings = &self.token_optimization.tool_result_aging;
        if aging_settings.enabled {
            self.tool_result_aging = Some(crate::core::ToolResultAging::new(aging_settings));
        }
        let cwm_settings = &self.token_optimization.context_window;
        if cwm_settings.max_messages > 0 {
            self.context_window_manager = Some(Arc::new(std::sync::Mutex::new(
                ContextWindowManager::new(cwm_settings),
            )));
        }
    }

    pub fn with_scheduler(mut self, scheduler: Arc<MemoryScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Attach token-optimization settings; re-initializes the three compressors.
    pub fn with_token_optimization(
        mut self,
        token_optimization: crate::config::settings::TokenOptimizationSettings,
    ) -> Self {
        self.token_optimization = token_optimization;
        {
            let mut executor = self.tool_executor.write();
            if self.token_optimization.enabled && self.token_optimization.tool_groups.enabled {
                let roles = self
                    .token_optimization
                    .tool_groups
                    .roles
                    .iter()
                    .map(|(role, config)| {
                        (
                            role.clone(),
                            crate::tools::tool_groups::RoleToolConfig {
                                default: config.default.clone(),
                                on_demand: config.on_demand.clone(),
                            },
                        )
                    })
                    .collect();
                executor.set_tool_group_manager(crate::tools::tool_groups::ToolGroupManager::new(
                    Some(crate::tools::tool_groups::ToolGroupSettings {
                        enabled: true,
                        roles,
                    }),
                ));
            } else {
                executor.clear_tool_group_manager();
            }
        }
        self.init_context_compressors();
        self
    }

    pub fn with_tool_result_router_settings(
        mut self,
        settings: crate::config::settings::ToolResultRouterSettings,
    ) -> Self {
        self.tool_executor.write().set_micro_tool_limits(
            settings.max_micro_tools,
            settings.micro_tool_page_size,
            settings.micro_tool_max_page_size,
        );
        self.tool_result_router_settings = settings;
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

    /// Attach a domain application contract without replacing kernel policy.
    pub fn with_application_prompt(
        mut self,
        profile: crate::core::prompt_contract::ApplicationPromptProfile,
    ) -> Self {
        self.application_prompt = Some(profile);
        self
    }

    pub fn with_prompt_variant(
        mut self,
        variant: crate::core::prompt_contract::PromptVariant,
    ) -> Self {
        self.prompt_variant = variant;
        self
    }

    pub fn with_learning_mode(mut self, mode: crate::core::policy_learning::LearningMode) -> Self {
        self.learning_mode = mode;
        self
    }

    pub(super) fn tool_definitions_for_agent(&self, role: &str) -> Vec<Value> {
        let executor = self.tool_executor.read();
        match self.prompt_variant {
            crate::core::prompt_contract::PromptVariant::Baseline => {
                executor.tool_definitions_for_role(role)
            }
            crate::core::prompt_contract::PromptVariant::Optimized => {
                executor.visible_tool_definitions_for_role(role)
            }
        }
    }

    #[cfg(test)]
    pub(super) fn tool_definitions_for_context(
        &self,
        role: &str,
        allowed_tools: Option<&[String]>,
    ) -> Vec<Value> {
        self.tool_definitions_for_context_with_microtools(role, allowed_tools, &HashSet::new())
    }

    /// Build the tool window for one BizAgent execution. Dynamic result-reader
    /// tools are process-global handlers because their archived data can
    /// outlive a turn, but they must not become process-global prompt state.
    /// Only readers created by this execution are advertised to its model.
    pub(super) fn tool_definitions_for_context_with_microtools(
        &self,
        role: &str,
        allowed_tools: Option<&[String]>,
        session_micro_tools: &HashSet<String>,
    ) -> Vec<Value> {
        let definitions = match self.prompt_variant {
            crate::core::prompt_contract::PromptVariant::Baseline => {
                self.tool_definitions_for_agent(role)
            }
            crate::core::prompt_contract::PromptVariant::Optimized => {
                let executor = self.tool_executor.read();
                let mut visible = executor.visible_tool_definitions_for_role(role);
                let mut names: HashSet<String> = visible
                    .iter()
                    .filter_map(|definition| {
                        definition["function"]["name"].as_str().map(str::to_string)
                    })
                    .collect();
                for definition in executor.tool_definitions_for_role(role) {
                    let Some(name) = definition["function"]["name"].as_str() else {
                        continue;
                    };
                    if session_micro_tools.contains(name) && names.insert(name.to_string()) {
                        visible.push(definition);
                    }
                }
                // `max_micro_tools` bounds the process-wide catalog, not the
                // validity of references already present in this BizAgent's
                // current conversation. Rebuild evicted schemas only for the
                // owning session; never expose another agent's archived tools.
                for name in session_micro_tools {
                    if ToolExecutor::is_micro_tool_name(name) && names.insert(name.clone()) {
                        if let Some(definition) = executor.micro_tool_definition(name) {
                            visible.push(definition);
                        }
                    }
                }
                visible
            }
        };
        definitions
            .into_iter()
            .filter(|definition| {
                let Some(name) = definition["function"]["name"].as_str() else {
                    return false;
                };
                if ToolExecutor::is_micro_tool_name(name) && !session_micro_tools.contains(name) {
                    return false;
                }
                allowed_tools
                    .map(|allowed| ToolExecutor::explicit_allowlist_permits(name, allowed))
                    .unwrap_or(true)
            })
            .collect()
    }

    fn is_workspace_bound_tool(name: &str) -> bool {
        matches!(
            name,
            "glob_search"
                | "grep_search"
                | "file_read"
                | "file_write"
                | "file_edit"
                | "file_list"
                | "workspace_status"
                | "bash"
                | "powershell"
                | "code_execute"
                | "knowledge_import_file"
                | "knowledge_import_directory"
                | "knowledge_extract_code"
        )
    }

    fn apply_task_tool_scope(&self, definitions: Vec<Value>, ctx: &TaskContext) -> Vec<Value> {
        definitions
            .into_iter()
            .filter(|definition| {
                let Some(name) = definition["function"]["name"].as_str() else {
                    return false;
                };
                let allowlisted = ctx
                    .allowed_tools
                    .as_deref()
                    .map(|allowed| ToolExecutor::explicit_allowlist_permits(name, allowed))
                    .unwrap_or(true);
                let workspace_scoped =
                    ctx.workspace_context_enabled() || !Self::is_workspace_bound_tool(name);
                allowlisted && workspace_scoped
            })
            .collect()
    }

    /// Apply the application-declared workspace scope after normal role and
    /// allowlist filtering. This keeps external/research tasks from seeing or
    /// invoking tools against unrelated projects in a shared workspace.
    pub(super) fn tool_definitions_for_task_context_with_microtools(
        &self,
        role: &str,
        ctx: &TaskContext,
        session_micro_tools: &HashSet<String>,
    ) -> Vec<Value> {
        let mut definitions = self.tool_definitions_for_context_with_microtools(
            role,
            ctx.allowed_tools.as_deref(),
            session_micro_tools,
        );
        if ctx.requires_web_research() {
            let mut names = definitions
                .iter()
                .filter_map(|definition| definition["function"]["name"].as_str())
                .map(str::to_string)
                .collect::<HashSet<_>>();
            for definition in self.tool_executor.read().tool_definitions_for_role(role) {
                let Some(name) = definition["function"]["name"].as_str() else {
                    continue;
                };
                if !matches!(name, "web_search" | "web_fetch" | "http_request")
                    || !names.insert(name.to_string())
                    || ctx.allowed_tools.as_deref().is_some_and(|allowed| {
                        !ToolExecutor::explicit_allowlist_permits(name, allowed)
                    })
                {
                    continue;
                }
                definitions.push(definition);
            }
        }
        self.apply_task_tool_scope(definitions, ctx)
    }

    /// Complete role-authorized catalog after application task scoping. This
    /// is intentionally broader than the default prompt window: `tool_search`
    /// may discover on-demand tools, but must never reveal tools excluded by
    /// the task's allowlist or workspace boundary.
    pub(super) fn discoverable_tool_definitions_for_task_context(
        &self,
        role: &str,
        ctx: &TaskContext,
    ) -> Vec<Value> {
        let definitions = self.tool_executor.read().tool_definitions_for_role(role);
        self.apply_task_tool_scope(definitions, ctx)
    }

    pub(super) fn tool_definitions_for_task_context(
        &self,
        role: &str,
        ctx: &TaskContext,
    ) -> Vec<Value> {
        self.tool_definitions_for_task_context_with_microtools(role, ctx, &HashSet::new())
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
