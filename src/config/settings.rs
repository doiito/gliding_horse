use anyhow::Result;
use config::{Config, ConfigError, Environment};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub gateway: GatewaySettings,
    pub memory: MemorySettings,
    pub perception: PerceptionSettings,
    pub agents: AgentSettings,
    pub api: ApiSettings,
    pub output: OutputSettings,
    pub emphasis: EmphasisConfig,
    pub logging: LoggingSettings,
    pub tool_result_router: ToolResultRouterSettings,
    #[serde(default)]
    pub embedding: EmbeddingSettings,
    #[serde(default)]
    pub token_optimization: TokenOptimizationSettings,
    #[serde(default)]
    pub batch_agents: BatchSettings,
    #[serde(default)]
    pub workspace: WorkspaceSettings,
    /// Guardrails for the generic continuous-learning policy. Applications
    /// may supply evidence, but cannot bypass these promotion thresholds.
    #[serde(default)]
    pub policy_learning: PolicyLearningSettings,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PolicyLearningSettings {
    /// Same-family baseline outcomes required before a bounded candidate arm
    /// may be sampled.
    #[serde(default = "default_candidate_trial_min_baseline_samples")]
    pub candidate_trial_min_baseline_samples: u32,
    /// Independent baseline/candidate outcomes required on both arms before
    /// the executable model can be promoted.
    #[serde(default = "default_policy_promotion_min_samples")]
    pub promotion_min_samples: u32,
    /// Minimum candidate mean reward improvement over the baseline mean.
    #[serde(default = "default_policy_promotion_min_improvement")]
    pub promotion_min_improvement: f32,
}

fn default_candidate_trial_min_baseline_samples() -> u32 {
    1
}

fn default_policy_promotion_min_samples() -> u32 {
    5
}

fn default_policy_promotion_min_improvement() -> f32 {
    0.01
}

impl Default for PolicyLearningSettings {
    fn default() -> Self {
        Self {
            candidate_trial_min_baseline_samples: default_candidate_trial_min_baseline_samples(),
            promotion_min_samples: default_policy_promotion_min_samples(),
            promotion_min_improvement: default_policy_promotion_min_improvement(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct WorkspaceSettings {
    /// Workspace root directory path, uses process CWD if empty
    pub root: Option<String>,
    /// File scan exclusion patterns
    pub exclude_patterns: Vec<String>,
    /// Whether to enable filesystem watching
    pub watch_enabled: bool,
    /// Content cache maximum bytes
    pub content_store_max_bytes: usize,
    /// LRU content cache capacity (number of files).
    #[serde(default = "default_content_cache_capacity")]
    pub content_cache_capacity: usize,
    /// Polling interval in ms (fallback when native watching unavailable).
    #[serde(default = "default_poll_interval_ms")]
    pub poll_interval_ms: u64,
    /// Debounce window in ms for file events.
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
    /// Maximum debounce wait in ms.
    #[serde(default = "default_max_debounce_wait_ms")]
    pub max_debounce_wait_ms: u64,
    /// Maximum time a task waits for the deferred metadata scan. The TUI is
    /// already visible; after this bound tools fall back to targeted disk IO.
    #[serde(default = "default_initial_scan_wait_ms")]
    pub initial_scan_wait_ms: u64,
    #[serde(default = "default_workspace_change_history_capacity")]
    pub change_history_capacity: usize,
    /// Maximum file count hashed for controlled learning replay snapshots.
    #[serde(default = "default_learning_snapshot_max_files")]
    pub learning_snapshot_max_files: usize,
    /// Maximum aggregate bytes hashed for controlled learning replay snapshots.
    #[serde(default = "default_learning_snapshot_max_bytes")]
    pub learning_snapshot_max_bytes: u64,
    /// Maximum file count hashed when confirming that a shell-like tool made
    /// a semantic workspace change.  If exceeded, confirmation falls back to
    /// the monitor's generation/delta evidence.
    #[serde(default = "default_effect_snapshot_max_files")]
    pub effect_snapshot_max_files: usize,
    /// Maximum aggregate bytes hashed for one semantic effect snapshot.
    #[serde(default = "default_effect_snapshot_max_bytes")]
    pub effect_snapshot_max_bytes: u64,
}

fn default_content_cache_capacity() -> usize {
    1000
}
fn default_poll_interval_ms() -> u64 {
    5000
}
fn default_debounce_ms() -> u64 {
    500
}
fn default_max_debounce_wait_ms() -> u64 {
    5000
}
fn default_initial_scan_wait_ms() -> u64 {
    250
}
fn default_workspace_change_history_capacity() -> usize {
    2048
}
fn default_learning_snapshot_max_files() -> usize {
    100_000
}
fn default_learning_snapshot_max_bytes() -> u64 {
    2 * 1024 * 1024 * 1024
}
fn default_effect_snapshot_max_files() -> usize {
    10_000
}
fn default_effect_snapshot_max_bytes() -> u64 {
    64 * 1024 * 1024
}

impl Default for WorkspaceSettings {
    fn default() -> Self {
        Self {
            root: None,
            exclude_patterns: vec![
                "node_modules/".into(),
                "target/".into(),
                ".git/".into(),
                "dist/".into(),
                "build/".into(),
                "__pycache__/".into(),
                ".venv/".into(),
                "venv/".into(),
                ".next/".into(),
                "data/".into(),
                ".gliding_horse/".into(),
            ],
            watch_enabled: true,
            content_store_max_bytes: 64 * 1024 * 1024,
            content_cache_capacity: default_content_cache_capacity(),
            poll_interval_ms: default_poll_interval_ms(),
            debounce_ms: default_debounce_ms(),
            max_debounce_wait_ms: default_max_debounce_wait_ms(),
            initial_scan_wait_ms: default_initial_scan_wait_ms(),
            change_history_capacity: default_workspace_change_history_capacity(),
            learning_snapshot_max_files: default_learning_snapshot_max_files(),
            learning_snapshot_max_bytes: default_learning_snapshot_max_bytes(),
            effect_snapshot_max_files: default_effect_snapshot_max_files(),
            effect_snapshot_max_bytes: default_effect_snapshot_max_bytes(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct GatewaySettings {
    pub base_url: String,
    pub api_key: String,
    pub default_model: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    #[serde(default = "default_retry_base_ms")]
    pub retry_base_ms: u64,
    /// Route deepseek-v4-flash requests through the Responses API (`/v1/responses`)
    /// instead of chat completions. Other models keep using chat completions.
    #[serde(default)]
    pub use_responses_api: bool,
    pub model_mapping: std::collections::HashMap<String, String>,
}

fn default_retry_base_ms() -> u64 {
    500
}

#[derive(Debug, Deserialize, Clone)]
pub struct MemorySettings {
    pub l0: L0Settings,
    pub l1: L1Settings,
    pub l2: L2Settings,
    pub l3: L3Settings,
}

#[derive(Debug, Deserialize, Clone)]
pub struct L0Settings {
    pub path: String,
    pub max_entries: u64,
    pub compression: bool,
    #[serde(default = "default_l0_blob_inline_threshold")]
    pub blob_inline_threshold: usize,
    /// redb page-cache size. redb's upstream default is 1 GiB, which is too
    /// large for an interactive application with a ~200 MiB idle target.
    #[serde(default = "default_l0_cache_size_bytes")]
    pub cache_size_bytes: usize,
    /// Persist allocator state on every commit so an unclean exit does not
    /// require a multi-pass scan of the complete L0 database at next startup.
    #[serde(default = "default_true")]
    pub quick_repair: bool,
}

fn default_l0_blob_inline_threshold() -> usize {
    4_096
}
fn default_l0_cache_size_bytes() -> usize {
    128 * 1024 * 1024
}

#[derive(Debug, Deserialize, Clone)]
pub struct L1Settings {
    pub max_messages: usize,
    pub compression_threshold: usize,
    pub max_tokens: usize,
    #[serde(default)]
    pub max_memory_mb: u64,
    /// Override default L1 eviction recency weight (None = role-specific default).
    #[serde(default)]
    pub eviction_recency_weight: Option<f64>,
    /// Override default L1 eviction relevance weight.
    #[serde(default)]
    pub eviction_relevance_weight: Option<f64>,
    /// Override default L1 eviction cost weight.
    #[serde(default)]
    pub eviction_cost_weight: Option<f64>,
    /// Override default L1 eviction relevance threshold.
    #[serde(default)]
    pub eviction_relevance_threshold: Option<f64>,
    /// Override default L1 eviction safe window in seconds.
    #[serde(default)]
    pub eviction_safe_window_seconds: Option<i64>,
    /// Override default L1 eviction beta fusion weight.
    #[serde(default)]
    pub eviction_beta: Option<f64>,
    #[serde(default = "default_l1_max_low_relevance_refs")]
    pub max_low_relevance_refs: usize,
    #[serde(default = "default_l1_reload_preview_chars")]
    pub reload_preview_chars: usize,
}

fn default_l1_max_low_relevance_refs() -> usize {
    3
}
fn default_l1_reload_preview_chars() -> usize {
    400
}

#[derive(Debug, Deserialize, Clone)]
pub struct L2Settings {
    pub max_node_size: usize,
    pub max_projection_size: usize,
    #[serde(default)]
    pub max_memory_mb: u64,
    #[serde(default = "default_l2_sync_queue_capacity")]
    pub sync_queue_capacity: usize,
}

fn default_l2_sync_queue_capacity() -> usize {
    1_024
}

#[derive(Debug, Deserialize, Clone)]
pub struct L3Settings {
    pub default_frame: String,
    pub max_size: usize,
    #[serde(default)]
    pub max_memory_mb: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PerceptionSettings {
    pub enabled: bool,
    pub triggers: Vec<String>,
    pub cache_ttl_seconds: u64,
    pub cache_max_entries: usize,
    pub anomaly_dedup_window_seconds: u64,
    #[serde(default = "default_simple_threshold")]
    pub simple_input_threshold: usize,
    #[serde(default = "default_medium_threshold")]
    pub medium_input_threshold: usize,
    #[serde(default = "default_cycle_timeout_secs")]
    pub cycle_timeout_secs: u64,
    #[serde(default = "default_max_iterations_before_alert")]
    pub max_iterations_before_alert: usize,
    #[serde(default = "default_error_rate_threshold")]
    pub error_rate_threshold: f64,
}

fn default_simple_threshold() -> usize {
    50
}
fn default_medium_threshold() -> usize {
    200
}
fn default_cycle_timeout_secs() -> u64 {
    300
}
fn default_max_iterations_before_alert() -> usize {
    10
}
fn default_error_rate_threshold() -> f64 {
    0.5
}

#[derive(Debug, Deserialize, Clone)]
pub struct AgentSettings {
    pub max_iterations: u32,
    pub parallel_execution: bool,
    pub max_parallel_agents: usize,
    pub timeout_seconds: u64,
    pub api_timeout_seconds: u64,
    pub event_bus_capacity: usize,
    pub template_path: Option<String>,
    #[serde(default = "default_max_pdca_cycles")]
    pub max_pdca_cycles: u32,
    /// Maximum number of concurrently active methodologies (MethodologyGate).
    #[serde(default = "default_max_active")]
    pub max_active: usize,
    /// TimelineStore: take a full snapshot every N mutations.
    #[serde(default = "default_snapshot_frequency")]
    pub snapshot_frequency: u64,
    /// TimelineStore: maximum full snapshots to retain.
    #[serde(default = "default_max_full_snapshots")]
    pub max_full_snapshots: usize,
    /// L3 ProjectionEngine maximum projection size.
    #[serde(default = "default_max_projection_size")]
    pub max_projection_size: usize,
    /// SA intervention/LLM execution timeout in seconds (default 30).
    #[serde(default = "default_sa_execution_timeout_secs")]
    pub sa_execution_timeout_secs: u64,
    /// Tool executor HTTP call timeout in seconds (default 60).
    #[serde(default = "default_tool_timeout_secs")]
    pub tool_timeout_secs: u64,
    /// MCP client call timeout in seconds (default 30).
    #[serde(default = "default_mcp_timeout_secs")]
    pub mcp_timeout_secs: u64,
    /// Embedding service call timeout in seconds (default 30).
    #[serde(default = "default_embedding_timeout_secs")]
    pub embedding_timeout_secs: u64,
    /// Per-BizAgent execution budgets and progress guards. Role limits default
    /// to `None`, which inherits the task's configured max_iterations.
    #[serde(default)]
    pub execution_budget: AgentExecutionBudgetSettings,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct RoleTurnLimitSettings {
    #[serde(default)]
    pub plan: Option<u32>,
    #[serde(default)]
    pub do_agent: Option<u32>,
    #[serde(default)]
    pub check: Option<u32>,
    #[serde(default)]
    pub act: Option<u32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AgentExecutionBudgetSettings {
    #[serde(default)]
    pub role_max_turns: RoleTurnLimitSettings,
    #[serde(default = "default_turn_early_warning_remaining")]
    pub early_warning_remaining: u32,
    #[serde(default = "default_turn_final_warning_remaining")]
    pub final_warning_remaining: u32,
    #[serde(default = "default_effect_progress_warning_turns")]
    pub effect_progress_warning_turns: u32,
    /// Zero disables blocking; otherwise this must be no smaller than the
    /// progress-warning threshold.
    #[serde(default = "default_effect_progress_block_turns")]
    pub effect_progress_block_turns: u32,
    /// A CA→DA repair already carries criterion-linked evidence, so it should
    /// not consume the same cold-inspection window as a fresh execution. Zero
    /// inherits `effect_progress_block_turns`.
    #[serde(default = "default_da_repair_effect_block_turns")]
    pub da_repair_effect_block_turns: u32,
    /// After this many CA tool turns, advertise only focused verification
    /// tools and require CA to name a still-unverified criterion before doing
    /// more work. Zero disables the evidence-convergence prompt.
    #[serde(default = "default_ca_evidence_focus_turns")]
    pub ca_evidence_focus_turns: u32,
    /// After this many CA tool turns, remove tools and require the final
    /// criterion-linked PASS/FAIL verdict. Zero disables the close gate.
    #[serde(default = "default_ca_evidence_close_turns")]
    pub ca_evidence_close_turns: u32,
    /// After this many PA tool turns, runtime removes inspection tools and
    /// asks PA to emit the executable plan from evidence already collected.
    /// Zero disables the convergence gate.
    #[serde(default = "default_pa_planning_focus_turns")]
    pub pa_planning_focus_turns: u32,
    /// Evidence-only DA work must eventually synthesize the collected
    /// evidence instead of searching indefinitely. At this many tool turns,
    /// broad discovery is withdrawn while targeted source reads remain.
    #[serde(default = "default_da_evidence_focus_turns")]
    pub da_evidence_focus_turns: u32,
    /// At this many evidence-only DA tool turns, remove tools and require the
    /// final evidence-backed deliverable. Zero disables the close gate.
    #[serde(default = "default_da_evidence_close_turns")]
    pub da_evidence_close_turns: u32,
    #[serde(default = "default_biz_agent_max_sub_agents")]
    pub max_sub_agents: usize,
    #[serde(default = "default_ca_handoff_max_chars")]
    pub ca_handoff_max_chars: usize,
    #[serde(default = "default_recursive_handoff_max_chars")]
    pub recursive_handoff_max_chars: usize,
    #[serde(default = "default_sa_stream_emit_min_chars")]
    pub sa_stream_emit_min_chars: usize,
    #[serde(default = "default_sa_stream_emit_interval_ms")]
    pub sa_stream_emit_interval_ms: u64,
    #[serde(default = "default_max_plan_steps")]
    pub max_plan_steps: usize,
    #[serde(default = "default_max_recursive_sub_tasks")]
    pub max_recursive_sub_tasks: usize,
    /// Task-wide cap across the whole recursive residual tree.  This is
    /// separate from `max_recursive_sub_tasks`, which limits one
    /// decomposition result, so depth cannot multiply work without bound.
    #[serde(default = "default_max_recursive_task_executions")]
    pub max_recursive_task_executions: usize,
    /// Task-wide turn budget shared by every recursive residual BizAgent.
    /// Operators can raise it for unusually large tasks without weakening the
    /// normal DA/CA budgets.
    #[serde(default = "default_max_recursive_total_turns")]
    pub max_recursive_total_turns: u32,
    #[serde(default = "default_max_ca_da_corrections")]
    pub max_ca_da_corrections: usize,
    /// Independent cap for task-scope PA replans. Local DA/CA corrections and
    /// scoped DAG retries do not consume this budget.
    #[serde(default = "default_max_plan_revisions")]
    pub max_plan_revisions: u32,
    #[serde(default = "default_ca_correction_handoff_max_chars")]
    pub ca_correction_handoff_max_chars: usize,
    #[serde(default = "default_force_finish_max_tool_entries")]
    pub force_finish_max_tool_entries: usize,
    #[serde(default = "default_force_finish_tool_result_max_chars")]
    pub force_finish_tool_result_max_chars: usize,
}

fn default_turn_early_warning_remaining() -> u32 {
    8
}
fn default_turn_final_warning_remaining() -> u32 {
    3
}
fn default_effect_progress_warning_turns() -> u32 {
    5
}
fn default_effect_progress_block_turns() -> u32 {
    8
}
fn default_da_repair_effect_block_turns() -> u32 {
    4
}
fn default_ca_evidence_focus_turns() -> u32 {
    5
}
fn default_ca_evidence_close_turns() -> u32 {
    10
}
fn default_pa_planning_focus_turns() -> u32 {
    4
}
fn default_da_evidence_focus_turns() -> u32 {
    5
}
fn default_da_evidence_close_turns() -> u32 {
    8
}
fn default_biz_agent_max_sub_agents() -> usize {
    5
}
fn default_ca_handoff_max_chars() -> usize {
    6_000
}
fn default_recursive_handoff_max_chars() -> usize {
    4_000
}
fn default_sa_stream_emit_min_chars() -> usize {
    128
}
fn default_sa_stream_emit_interval_ms() -> u64 {
    50
}
fn default_max_plan_steps() -> usize {
    12
}
fn default_max_recursive_sub_tasks() -> usize {
    5
}
fn default_max_recursive_task_executions() -> usize {
    8
}
fn default_max_recursive_total_turns() -> u32 {
    60
}
fn default_max_ca_da_corrections() -> usize {
    3
}
fn default_max_plan_revisions() -> u32 {
    2
}
fn default_ca_correction_handoff_max_chars() -> usize {
    6_000
}
fn default_force_finish_max_tool_entries() -> usize {
    20
}
fn default_force_finish_tool_result_max_chars() -> usize {
    2_000
}

impl Default for AgentExecutionBudgetSettings {
    fn default() -> Self {
        Self {
            role_max_turns: RoleTurnLimitSettings::default(),
            early_warning_remaining: default_turn_early_warning_remaining(),
            final_warning_remaining: default_turn_final_warning_remaining(),
            effect_progress_warning_turns: default_effect_progress_warning_turns(),
            effect_progress_block_turns: default_effect_progress_block_turns(),
            da_repair_effect_block_turns: default_da_repair_effect_block_turns(),
            ca_evidence_focus_turns: default_ca_evidence_focus_turns(),
            ca_evidence_close_turns: default_ca_evidence_close_turns(),
            pa_planning_focus_turns: default_pa_planning_focus_turns(),
            da_evidence_focus_turns: default_da_evidence_focus_turns(),
            da_evidence_close_turns: default_da_evidence_close_turns(),
            max_sub_agents: default_biz_agent_max_sub_agents(),
            ca_handoff_max_chars: default_ca_handoff_max_chars(),
            recursive_handoff_max_chars: default_recursive_handoff_max_chars(),
            sa_stream_emit_min_chars: default_sa_stream_emit_min_chars(),
            sa_stream_emit_interval_ms: default_sa_stream_emit_interval_ms(),
            max_plan_steps: default_max_plan_steps(),
            max_recursive_sub_tasks: default_max_recursive_sub_tasks(),
            max_recursive_task_executions: default_max_recursive_task_executions(),
            max_recursive_total_turns: default_max_recursive_total_turns(),
            max_ca_da_corrections: default_max_ca_da_corrections(),
            max_plan_revisions: default_max_plan_revisions(),
            ca_correction_handoff_max_chars: default_ca_correction_handoff_max_chars(),
            force_finish_max_tool_entries: default_force_finish_max_tool_entries(),
            force_finish_tool_result_max_chars: default_force_finish_tool_result_max_chars(),
        }
    }
}

fn default_max_pdca_cycles() -> u32 {
    7
}
fn default_max_active() -> usize {
    20
}
fn default_snapshot_frequency() -> u64 {
    1000
}
fn default_max_full_snapshots() -> usize {
    10
}
fn default_max_projection_size() -> usize {
    500
}
fn default_sa_execution_timeout_secs() -> u64 {
    30
}
fn default_tool_timeout_secs() -> u64 {
    60
}
fn default_mcp_timeout_secs() -> u64 {
    30
}
fn default_embedding_timeout_secs() -> u64 {
    30
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            parallel_execution: true,
            max_parallel_agents: 10,
            timeout_seconds: 300,
            api_timeout_seconds: 120,
            event_bus_capacity: 100,
            template_path: None,
            max_pdca_cycles: 7,
            max_active: 20,
            snapshot_frequency: 1000,
            max_full_snapshots: 10,
            max_projection_size: 500,
            sa_execution_timeout_secs: 30,
            tool_timeout_secs: 60,
            mcp_timeout_secs: 30,
            embedding_timeout_secs: 30,
            execution_budget: AgentExecutionBudgetSettings::default(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ApiSettings {
    pub grpc_addr: String,
    pub http_addr: String,
    pub enable_metrics: bool,
    pub metrics_port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OutputSettings {
    pub directory: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmphasisConfig {
    pub enabled: bool,
    pub extraction_prompt: String,
    pub max_items: usize,
    pub dedup_threshold: f64,
}

impl Default for EmphasisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            extraction_prompt: r#"## Emphasis Content Extraction
If the user input contains emphatic content (such as "must", "important", "don't forget", "critical", etc.),
please extract these and place them in the "emphasis" field of the JSON (a string array).

Example:
{
  "thought": "The user emphasized that async must be used...",
  "content": "Okay, I will...",
  "summary": "Confirmed async implementation",
  "emphasis": ["must use async implementation"]
}"#.to_string(),
            max_items: 50,
            dedup_threshold: 0.85,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct LoggingSettings {
    pub level: String,
    pub format: String,
    pub console_output: bool,
    pub file_output: FileOutputSettings,
    pub filters: Vec<LogFilter>,
    pub sensitive_fields: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FileOutputSettings {
    pub enabled: bool,
    pub path: String,
    pub prefix: String,
    pub rotation: String,
    pub max_files: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LogFilter {
    pub module: String,
    pub level: String,
}

impl LoggingSettings {
    pub fn test_default(prefix: &str) -> Self {
        Self {
            level: "debug".to_string(),
            format: "text".to_string(),
            console_output: true,
            file_output: FileOutputSettings {
                enabled: true,
                path: "./logs".to_string(),
                prefix: prefix.to_string(),
                rotation: "daily".to_string(),
                max_files: 10,
            },
            filters: vec![
                LogFilter {
                    module: "glidinghorse::core".to_string(),
                    level: "debug".to_string(),
                },
                LogFilter {
                    module: "glidinghorse::gateway".to_string(),
                    level: "debug".to_string(),
                },
                LogFilter {
                    module: "glidinghorse::memory".to_string(),
                    level: "info".to_string(),
                },
                LogFilter {
                    module: "glidinghorse::tools".to_string(),
                    level: "info".to_string(),
                },
                LogFilter {
                    module: "redb".to_string(),
                    level: "warn".to_string(),
                },
            ],
            sensitive_fields: vec!["api_key".to_string(), "password".to_string()],
        }
    }
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "text".to_string(),
            console_output: true,
            file_output: FileOutputSettings {
                enabled: true,
                path: "./logs".to_string(),
                prefix: "agent_os".to_string(),
                rotation: "daily".to_string(),
                max_files: 30,
            },
            filters: vec![
                LogFilter {
                    module: "glidinghorse::gateway".to_string(),
                    level: "debug".to_string(),
                },
                LogFilter {
                    module: "glidinghorse::core".to_string(),
                    level: "debug".to_string(),
                },
            ],
            sensitive_fields: vec![
                "api_key".to_string(),
                "password".to_string(),
                "token".to_string(),
                "secret".to_string(),
            ],
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolResultRouterSettings {
    pub enabled: bool,
    pub threshold_small: usize,
    pub threshold_large: usize,
    pub micro_tool_threshold: usize,
    pub preview_size: usize,
    pub max_graph_entities: usize,
    pub max_micro_tools: usize,
    pub sparql_query_timeout_ms: u64,
    pub auto_cleanup: bool,
    /// Persist and register micro-tool when PassThrough result exceeds this byte size,
    /// preparing for reference-based reclamation under context pressure.
    #[serde(default = "default_prepare_threshold")]
    pub prepare_threshold: usize,
    #[serde(default = "default_micro_tool_page_size")]
    pub micro_tool_page_size: usize,
    #[serde(default = "default_micro_tool_max_page_size")]
    pub micro_tool_max_page_size: usize,
}

fn default_prepare_threshold() -> usize {
    3072
}
fn default_micro_tool_page_size() -> usize {
    100
}
fn default_micro_tool_max_page_size() -> usize {
    200
}

impl Default for ToolResultRouterSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_small: 16384,
            threshold_large: 32768,
            micro_tool_threshold: 16384,
            preview_size: 2000,
            max_graph_entities: 500,
            max_micro_tools: 5,
            sparql_query_timeout_ms: 100,
            auto_cleanup: true,
            prepare_threshold: default_prepare_threshold(),
            micro_tool_page_size: default_micro_tool_page_size(),
            micro_tool_max_page_size: default_micro_tool_max_page_size(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default)]
    pub ollama: OllamaEmbeddingConfig,
    #[serde(default)]
    pub oneapi: OneApiEmbeddingConfig,
    #[serde(default)]
    pub fallback: FallbackEmbeddingConfig,
}

fn default_true() -> bool {
    true
}
fn default_provider() -> String {
    "ollama".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct OllamaEmbeddingConfig {
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
    #[serde(default = "default_ollama_dim")]
    pub dimension: usize,
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}
fn default_ollama_model() -> String {
    "nomic-embed-text".to_string()
}
fn default_ollama_dim() -> usize {
    768
}

impl Default for OllamaEmbeddingConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_url(),
            model: default_ollama_model(),
            dimension: default_ollama_dim(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct OneApiEmbeddingConfig {
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_oneapi_model")]
    pub model: String,
    #[serde(default = "default_oneapi_dim")]
    pub dimension: usize,
}

fn default_oneapi_model() -> String {
    "text-embedding-3-small".to_string()
}
fn default_oneapi_dim() -> usize {
    1536
}

impl Default for OneApiEmbeddingConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: String::new(),
            model: default_oneapi_model(),
            dimension: default_oneapi_dim(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct FallbackEmbeddingConfig {
    #[serde(default = "default_fallback_dim")]
    pub dimension: usize,
}

fn default_fallback_dim() -> usize {
    128
}

impl Default for FallbackEmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: default_fallback_dim(),
        }
    }
}

impl Default for EmbeddingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: default_provider(),
            ollama: OllamaEmbeddingConfig::default(),
            oneapi: OneApiEmbeddingConfig::default(),
            fallback: FallbackEmbeddingConfig::default(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct TokenOptimizationSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub tool_groups: ToolGroupSettings,
    #[serde(default)]
    pub tool_result_compressor: ToolResultCompressorSettings,
    #[serde(default)]
    pub context_window: ContextWindowSettings,
    #[serde(default)]
    pub tool_result_aging: ToolResultAgingSettings,
    #[serde(default)]
    pub prompt_optimization: PromptOptimizationSettings,
}

impl Default for TokenOptimizationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            tool_groups: ToolGroupSettings::default(),
            tool_result_compressor: ToolResultCompressorSettings::default(),
            context_window: ContextWindowSettings::default(),
            tool_result_aging: ToolResultAgingSettings::default(),
            prompt_optimization: PromptOptimizationSettings::default(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolGroupSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub roles: std::collections::HashMap<String, RoleToolConfig>,
}

impl Default for ToolGroupSettings {
    fn default() -> Self {
        let mut roles = std::collections::HashMap::new();
        roles.insert(
            "Plan".to_string(),
            RoleToolConfig {
                default: vec![
                    "Core".to_string(),
                    "Search".to_string(),
                    "System".to_string(),
                ],
                on_demand: vec![
                    "Web".to_string(),
                    "Knowledge".to_string(),
                    "Code".to_string(),
                    "Skill".to_string(),
                ],
            },
        );
        roles.insert(
            "Do".to_string(),
            RoleToolConfig {
                default: vec![
                    "Core".to_string(),
                    "Write".to_string(),
                    "Search".to_string(),
                    "System".to_string(),
                ],
                on_demand: vec![
                    "Web".to_string(),
                    "Knowledge".to_string(),
                    "Code".to_string(),
                    "Skill".to_string(),
                ],
            },
        );
        roles.insert(
            "Check".to_string(),
            RoleToolConfig {
                default: vec![
                    "Core".to_string(),
                    "Search".to_string(),
                    "Verify".to_string(),
                    "System".to_string(),
                ],
                on_demand: vec![
                    "Web".to_string(),
                    "Knowledge".to_string(),
                    "Code".to_string(),
                ],
            },
        );
        roles.insert(
            "Act".to_string(),
            RoleToolConfig {
                default: vec!["Core".to_string(), "System".to_string()],
                on_demand: vec!["Search".to_string(), "Knowledge".to_string()],
            },
        );
        Self {
            enabled: true,
            roles,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct RoleToolConfig {
    #[serde(default)]
    pub default: Vec<String>,
    #[serde(default)]
    pub on_demand: Vec<String>,
}

impl Default for RoleToolConfig {
    fn default() -> Self {
        Self {
            default: vec![],
            on_demand: vec![],
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolResultCompressorSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_max_full_results")]
    pub max_full_results: usize,
    #[serde(default = "default_max_summary_length")]
    pub max_summary_length: usize,
    #[serde(default = "default_compression_trigger")]
    pub compression_trigger: usize,
    /// Replace tool message with reference compression if micro-tool exists and content exceeds this byte size.
    #[serde(default = "default_compress_tool_result_threshold")]
    pub compress_tool_result_threshold: usize,
}

fn default_compress_tool_result_threshold() -> usize {
    500
}

fn default_max_full_results() -> usize {
    2
}
fn default_max_summary_length() -> usize {
    200
}
fn default_compression_trigger() -> usize {
    10
}

impl Default for ToolResultCompressorSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_full_results: default_max_full_results(),
            max_summary_length: default_max_summary_length(),
            compression_trigger: default_compression_trigger(),
            compress_tool_result_threshold: default_compress_tool_result_threshold(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ContextWindowSettings {
    #[serde(default = "default_max_messages")]
    pub max_messages: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_compression_ratio")]
    pub compression_ratio: f32,
    #[serde(default = "default_preserve_recent")]
    pub preserve_recent: usize,
    /// When true, the active model's safe proportional budget is also applied.
    /// The configured `max_tokens` remains an operator cost ceiling.
    #[serde(default = "default_false")]
    pub model_aware: bool,
}

fn default_false() -> bool {
    false
}

fn default_max_messages() -> usize {
    30
}
fn default_max_tokens() -> usize {
    16000
}
fn default_compression_ratio() -> f32 {
    0.3
}
fn default_preserve_recent() -> usize {
    4
}

impl Default for ContextWindowSettings {
    fn default() -> Self {
        Self {
            max_messages: default_max_messages(),
            max_tokens: default_max_tokens(),
            compression_ratio: default_compression_ratio(),
            preserve_recent: default_preserve_recent(),
            model_aware: false,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolResultAgingSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Number of full results to keep (newest N tool results remain intact)
    #[serde(default = "default_aging_keep_full")]
    pub keep_full: usize,
    /// Number of old results to attempt micro-tool references (after keep_full)
    #[serde(default = "default_aging_try_microtool")]
    pub try_microtool: usize,
    /// Compression threshold: only process tool messages exceeding this byte size
    #[serde(default = "default_aging_compress_threshold")]
    pub compress_threshold: usize,
}

fn default_aging_keep_full() -> usize {
    5
}
fn default_aging_try_microtool() -> usize {
    5
}
fn default_aging_compress_threshold() -> usize {
    500
}

impl Default for ToolResultAgingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            keep_full: default_aging_keep_full(),
            try_microtool: default_aging_try_microtool(),
            compress_threshold: default_aging_compress_threshold(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct PromptOptimizationSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub use_layered_prompts: bool,
    #[serde(default = "default_true")]
    pub store_specs_in_kg: bool,
    #[serde(default = "default_max_injected_skills")]
    pub max_injected_skills: usize,
    #[serde(default = "default_max_workspace_manifest_files")]
    pub max_workspace_manifest_files: usize,
    #[serde(default = "default_max_workspace_manifest_chars")]
    pub max_workspace_manifest_chars: usize,
    #[serde(default = "default_max_kg_context_entities")]
    pub max_kg_context_entities: usize,
    #[serde(default = "default_max_kg_context_bytes")]
    pub max_kg_context_bytes: usize,
    #[serde(default = "default_max_learning_hints")]
    pub max_learning_hints: usize,
    #[serde(default = "default_max_learning_hint_chars")]
    pub max_learning_hint_chars: usize,
    #[serde(default = "default_max_learning_hint_total_chars")]
    pub max_learning_hint_total_chars: usize,
    #[serde(default = "default_max_discovered_skill_hints")]
    pub max_discovered_skill_hints: usize,
    #[serde(default = "default_max_knowledge_fragments")]
    pub max_knowledge_fragments: usize,
}

fn default_max_injected_skills() -> usize {
    10
}
fn default_max_workspace_manifest_files() -> usize {
    160
}
fn default_max_workspace_manifest_chars() -> usize {
    12_000
}
fn default_max_kg_context_entities() -> usize {
    12
}
fn default_max_kg_context_bytes() -> usize {
    4_096
}
fn default_max_learning_hints() -> usize {
    20
}
fn default_max_learning_hint_chars() -> usize {
    700
}
fn default_max_learning_hint_total_chars() -> usize {
    6_000
}
fn default_max_discovered_skill_hints() -> usize {
    10
}
fn default_max_knowledge_fragments() -> usize {
    12
}

impl Default for PromptOptimizationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            use_layered_prompts: true,
            store_specs_in_kg: true,
            max_injected_skills: default_max_injected_skills(),
            max_workspace_manifest_files: default_max_workspace_manifest_files(),
            max_workspace_manifest_chars: default_max_workspace_manifest_chars(),
            max_kg_context_entities: default_max_kg_context_entities(),
            max_kg_context_bytes: default_max_kg_context_bytes(),
            max_learning_hints: default_max_learning_hints(),
            max_learning_hint_chars: default_max_learning_hint_chars(),
            max_learning_hint_total_chars: default_max_learning_hint_total_chars(),
            max_discovered_skill_hints: default_max_discovered_skill_hints(),
            max_knowledge_fragments: default_max_knowledge_fragments(),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct BatchSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_batch_default_model")]
    pub default_model: String,
    #[serde(default = "default_batch_temperature")]
    pub default_temperature: f32,
    #[serde(default = "default_batch_max_retries")]
    pub default_max_retries: u32,
    #[serde(default = "default_true")]
    pub inject_user_reminders: bool,
    #[serde(default = "default_true")]
    pub inject_context_summary: bool,
    #[serde(default = "default_true")]
    pub inject_related_entities: bool,
    #[serde(default)]
    pub agents: Vec<BatchAgentSettings>,
}

fn default_batch_default_model() -> String {
    "deepseek-v4-flash".to_string()
}
fn default_batch_temperature() -> f32 {
    0.1
}
fn default_batch_max_retries() -> u32 {
    3
}

#[derive(Debug, Deserialize, Clone)]
pub struct BatchAgentSettings {
    pub name: String,
    pub description: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub window_type: Option<String>,
    pub window_max_messages: Option<usize>,
    pub window_max_seconds: Option<u64>,
    #[serde(default)]
    pub triggers: Vec<BatchTriggerSettings>,
    #[serde(default)]
    pub prompt_source: String,
    pub prompt_template_name: Option<String>,
    pub prompt_template_path: Option<String>,
    pub business_domain: String,
    #[serde(default)]
    pub entity_types: Vec<String>,
    #[serde(default)]
    pub relation_types: Vec<String>,
    #[serde(default)]
    pub intent_types: Vec<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_retries: Option<u32>,
    pub timeout_seconds: Option<u64>,
    #[serde(default)]
    pub emit_on: Vec<String>,
    #[serde(default = "default_true")]
    pub inject_user_reminders: bool,
    #[serde(default = "default_true")]
    pub inject_context_summary: bool,
    /// Explicit opt-in: graph-changing maintenance handlers are disabled by
    /// default until a deployment provides governance and idempotency policy.
    #[serde(default)]
    pub apply_graph_mutations: bool,

    // Maintenance Agent specific options
    #[serde(default)]
    pub min_confidence_auto_apply: Option<f64>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub max_candidates: Option<usize>,
    #[serde(default)]
    pub lookback_hours: Option<u64>,
    #[serde(default)]
    pub llm_analysis_threshold: Option<f64>,
    #[serde(default)]
    pub max_items_per_run: Option<usize>,
    #[serde(default)]
    pub max_suggestions_per_run: Option<usize>,
}

impl Default for BatchAgentSettings {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            enabled: true,
            window_type: None,
            window_max_messages: Some(5),
            window_max_seconds: Some(600),
            triggers: vec![],
            prompt_source: "HybridWithTemplate".to_string(),
            prompt_template_name: None,
            prompt_template_path: None,
            business_domain: "default".to_string(),
            entity_types: vec![],
            relation_types: vec![],
            intent_types: vec![],
            model: None,
            temperature: None,
            max_retries: None,
            timeout_seconds: None,
            emit_on: vec![],
            inject_user_reminders: true,
            inject_context_summary: true,
            apply_graph_mutations: false,
            min_confidence_auto_apply: None,
            batch_size: None,
            max_candidates: None,
            lookback_hours: None,
            llm_analysis_threshold: None,
            max_items_per_run: None,
            max_suggestions_per_run: None,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct BatchTriggerSettings {
    pub trigger_type: String,
    #[serde(default)]
    pub params: std::collections::HashMap<String, String>,
}

impl Default for BatchTriggerSettings {
    fn default() -> Self {
        Self {
            trigger_type: "WindowFull".to_string(),
            params: std::collections::HashMap::new(),
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            gateway: GatewaySettings {
                base_url: "http://localhost:3000".to_string(),
                api_key: String::new(),
                default_model: "deepseek-v4-flash".to_string(),
                timeout_seconds: 30,
                max_retries: 3,
                retry_base_ms: 500,
                use_responses_api: false,
                model_mapping: std::collections::HashMap::from([
                    ("planning".to_string(), "deepseek-v4-pro".to_string()),
                    ("execution".to_string(), "deepseek-v4-pro".to_string()),
                    ("analysis".to_string(), "deepseek-v4-flash".to_string()),
                    ("default".to_string(), "deepseek-v4-flash".to_string()),
                ]),
            },
            memory: MemorySettings {
                l0: L0Settings {
                    path: "./data/l0".to_string(),
                    max_entries: 1_000_000,
                    compression: true,
                    blob_inline_threshold: default_l0_blob_inline_threshold(),
                    cache_size_bytes: default_l0_cache_size_bytes(),
                    quick_repair: true,
                },
                l1: L1Settings {
                    max_messages: 100,
                    compression_threshold: 50,
                    max_tokens: 4096,
                    max_memory_mb: 0,
                    eviction_recency_weight: None,
                    eviction_relevance_weight: None,
                    eviction_cost_weight: None,
                    eviction_relevance_threshold: None,
                    eviction_safe_window_seconds: None,
                    eviction_beta: None,
                    max_low_relevance_refs: default_l1_max_low_relevance_refs(),
                    reload_preview_chars: default_l1_reload_preview_chars(),
                },
                l2: L2Settings {
                    max_node_size: 5_242_880,
                    max_projection_size: 500,
                    max_memory_mb: 0,
                    sync_queue_capacity: default_l2_sync_queue_capacity(),
                },
                l3: L3Settings {
                    default_frame: "summary_only".to_string(),
                    max_size: 500,
                    max_memory_mb: 0,
                },
            },
            perception: PerceptionSettings {
                enabled: true,
                triggers: vec![
                    "TaskStart".to_string(),
                    "PlanCompleted".to_string(),
                    "ProgressAnomaly".to_string(),
                    "CheckCompleted".to_string(),
                    "TaskEnd".to_string(),
                    "CycleTimeout".to_string(),
                    "AgentBlocked".to_string(),
                    "ResourceConflict".to_string(),
                    "QualityDegradation".to_string(),
                    "UserFeedback".to_string(),
                ],
                cache_ttl_seconds: 300,
                cache_max_entries: 1000,
                anomaly_dedup_window_seconds: 60,
                simple_input_threshold: 50,
                medium_input_threshold: 200,
                cycle_timeout_secs: 300,
                max_iterations_before_alert: 10,
                error_rate_threshold: 0.5,
            },
            agents: AgentSettings::default(),
            api: ApiSettings {
                grpc_addr: "0.0.0.0:50051".to_string(),
                http_addr: "0.0.0.0:8080".to_string(),
                enable_metrics: true,
                metrics_port: 9090,
            },
            output: OutputSettings {
                directory: "./data/output".to_string(),
            },
            emphasis: EmphasisConfig::default(),
            logging: LoggingSettings::default(),
            tool_result_router: ToolResultRouterSettings::default(),
            embedding: EmbeddingSettings::default(),
            token_optimization: TokenOptimizationSettings::default(),
            batch_agents: BatchSettings::default(),
            workspace: WorkspaceSettings::default(),
            policy_learning: PolicyLearningSettings::default(),
        }
    }
}

impl Settings {
    pub fn load() -> Result<Self, ConfigError> {
        let config = Config::builder()
            .add_source(config::File::with_name("config").required(false))
            .add_source(
                Environment::with_prefix("AGENT_OS")
                    .separator("_")
                    .try_parsing(true),
            )
            .build()?;

        config.try_deserialize()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.gateway.base_url.is_empty() {
            return Err("gateway.base_url must be set".to_string());
        }
        if self.gateway.api_key.is_empty() {
            return Err(
                "gateway.api_key must be set (via config.yaml or AGENT_OS_GATEWAY_API_KEY)"
                    .to_string(),
            );
        }
        if self.gateway.default_model.is_empty() {
            return Err("gateway.default_model must be set".to_string());
        }
        if self.agents.max_iterations == 0 {
            return Err("agents.max_iterations must be > 0".to_string());
        }
        let budget = &self.agents.execution_budget;
        for (role, limit) in [
            ("plan", budget.role_max_turns.plan),
            ("do_agent", budget.role_max_turns.do_agent),
            ("check", budget.role_max_turns.check),
            ("act", budget.role_max_turns.act),
        ] {
            if limit == Some(0) {
                return Err(format!(
                    "agents.execution_budget.role_max_turns.{role} must be > 0 when set"
                ));
            }
        }
        if budget.early_warning_remaining > 0
            && budget.final_warning_remaining > 0
            && budget.early_warning_remaining <= budget.final_warning_remaining
        {
            return Err(
                "agents.execution_budget.early_warning_remaining must be greater than final_warning_remaining"
                    .to_string(),
            );
        }
        if budget.effect_progress_block_turns != 0
            && budget.effect_progress_block_turns < budget.effect_progress_warning_turns
        {
            return Err(
                "agents.execution_budget.effect_progress_block_turns must be zero or >= effect_progress_warning_turns"
                    .to_string(),
            );
        }
        if budget.ca_evidence_close_turns != 0
            && budget.ca_evidence_focus_turns != 0
            && budget.ca_evidence_close_turns <= budget.ca_evidence_focus_turns
        {
            return Err(
                "agents.execution_budget.ca_evidence_close_turns must be zero or greater than ca_evidence_focus_turns"
                    .to_string(),
            );
        }
        if budget.da_evidence_close_turns != 0
            && budget.da_evidence_focus_turns != 0
            && budget.da_evidence_close_turns <= budget.da_evidence_focus_turns
        {
            return Err(
                "agents.execution_budget.da_evidence_close_turns must be zero or greater than da_evidence_focus_turns"
                    .to_string(),
            );
        }
        if budget.max_sub_agents == 0 {
            return Err("agents.execution_budget.max_sub_agents must be > 0".to_string());
        }
        if budget.ca_handoff_max_chars == 0 {
            return Err("agents.execution_budget.ca_handoff_max_chars must be > 0".to_string());
        }
        if budget.recursive_handoff_max_chars == 0 {
            return Err(
                "agents.execution_budget.recursive_handoff_max_chars must be > 0".to_string(),
            );
        }
        if budget.sa_stream_emit_min_chars == 0 || budget.sa_stream_emit_interval_ms == 0 {
            return Err(
                "agents.execution_budget SA stream emit thresholds must be > 0".to_string(),
            );
        }
        for (name, value) in [
            ("max_plan_steps", budget.max_plan_steps),
            ("max_recursive_sub_tasks", budget.max_recursive_sub_tasks),
            (
                "max_recursive_task_executions",
                budget.max_recursive_task_executions,
            ),
            ("max_ca_da_corrections", budget.max_ca_da_corrections),
            (
                "ca_correction_handoff_max_chars",
                budget.ca_correction_handoff_max_chars,
            ),
            (
                "force_finish_max_tool_entries",
                budget.force_finish_max_tool_entries,
            ),
            (
                "force_finish_tool_result_max_chars",
                budget.force_finish_tool_result_max_chars,
            ),
        ] {
            if value == 0 {
                return Err(format!("agents.execution_budget.{name} must be > 0"));
            }
        }
        if budget.max_recursive_total_turns == 0 {
            return Err(
                "agents.execution_budget.max_recursive_total_turns must be > 0".to_string(),
            );
        }
        if self.workspace.learning_snapshot_max_files == 0
            || self.workspace.learning_snapshot_max_bytes == 0
        {
            return Err("workspace learning snapshot limits must be > 0".to_string());
        }
        if self.workspace.effect_snapshot_max_files == 0
            || self.workspace.effect_snapshot_max_bytes == 0
        {
            return Err("workspace effect snapshot limits must be > 0".to_string());
        }
        if self.policy_learning.candidate_trial_min_baseline_samples == 0
            || self.policy_learning.promotion_min_samples == 0
        {
            return Err("policy learning sample thresholds must be > 0".to_string());
        }
        if !self.policy_learning.promotion_min_improvement.is_finite()
            || !(-2.0..=2.0).contains(&self.policy_learning.promotion_min_improvement)
        {
            return Err(
                "policy_learning.promotion_min_improvement must be finite and within [-2, 2]"
                    .to_string(),
            );
        }
        if self.tool_result_router.max_micro_tools == 0
            || self.tool_result_router.micro_tool_page_size == 0
            || self.tool_result_router.micro_tool_max_page_size
                < self.tool_result_router.micro_tool_page_size
        {
            return Err(
                "tool_result_router micro-tool limits must be positive and max_page_size >= page_size"
                    .to_string(),
            );
        }
        if self.memory.l0.blob_inline_threshold == 0 {
            return Err("memory.l0.blob_inline_threshold must be > 0".to_string());
        }
        if self.memory.l0.cache_size_bytes == 0 {
            return Err("memory.l0.cache_size_bytes must be > 0".to_string());
        }
        if self.memory.l2.sync_queue_capacity == 0 {
            return Err("memory.l2.sync_queue_capacity must be > 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_settings_test_default() {
        let settings = LoggingSettings::test_default("test_prefix");
        assert_eq!(settings.level, "debug");
        assert_eq!(settings.format, "text");
        assert!(settings.console_output);
        assert!(settings.file_output.enabled);
        assert_eq!(settings.file_output.prefix, "test_prefix");
        assert!(settings
            .filters
            .iter()
            .any(|f| f.module == "redb" && f.level == "warn"));
        assert!(settings
            .filters
            .iter()
            .any(|f| f.module == "glidinghorse::core" && f.level == "debug"));
        assert!(settings
            .filters
            .iter()
            .any(|f| f.module == "glidinghorse::memory" && f.level == "info"));
    }

    #[test]
    fn test_logging_settings_default_has_redb_in_init() {
        let settings = LoggingSettings::default();
        assert_eq!(settings.level, "info");
    }

    #[test]
    fn test_gateway_settings_deserializes_retry_base_ms() {
        let yaml = r#"
            base_url: "https://api.deepseek.com"
            api_key: "sk-test"
            default_model: "deepseek-v4-flash"
            timeout_seconds: 300
            max_retries: 3
            retry_base_ms: 750
            model_mapping: {}
        "#;
        let cfg = Config::builder()
            .add_source(config::File::from_str(yaml, config::FileFormat::Yaml))
            .build()
            .unwrap();
        let settings: GatewaySettings = cfg.try_deserialize().unwrap();
        assert_eq!(settings.retry_base_ms, 750);
    }

    #[test]
    fn test_gateway_settings_retry_base_ms_default() {
        let yaml = r#"
            base_url: "https://api.deepseek.com"
            api_key: "sk-test"
            default_model: "deepseek-v4-flash"
            timeout_seconds: 300
            max_retries: 3
            model_mapping: {}
        "#;
        let cfg = Config::builder()
            .add_source(config::File::from_str(yaml, config::FileFormat::Yaml))
            .build()
            .unwrap();
        let settings: GatewaySettings = cfg.try_deserialize().unwrap();
        // retry_base_ms omitted -> serde default 500
        assert_eq!(settings.retry_base_ms, 500);
    }

    #[test]
    fn test_agent_settings_deserializes_tunables() {
        let yaml = r#"
            max_iterations: 10
            parallel_execution: true
            max_parallel_agents: 10
            timeout_seconds: 300
            api_timeout_seconds: 120
            event_bus_capacity: 100
            max_pdca_cycles: 7
            max_active: 42
            snapshot_frequency: 2000
            max_full_snapshots: 5
            max_projection_size: 1024
            execution_budget:
              role_max_turns:
                check: 24
              early_warning_remaining: 6
              final_warning_remaining: 2
              effect_progress_warning_turns: 7
              effect_progress_block_turns: 15
              da_repair_effect_block_turns: 4
              ca_evidence_focus_turns: 9
              ca_evidence_close_turns: 14
              pa_planning_focus_turns: 6
              da_evidence_focus_turns: 7
              da_evidence_close_turns: 12
              max_sub_agents: 8
              ca_handoff_max_chars: 9000
              recursive_handoff_max_chars: 5000
              sa_stream_emit_min_chars: 64
              sa_stream_emit_interval_ms: 25
              max_plan_steps: 18
              max_recursive_sub_tasks: 7
              max_recursive_task_executions: 11
              max_recursive_total_turns: 90
              max_ca_da_corrections: 4
              ca_correction_handoff_max_chars: 8000
              force_finish_max_tool_entries: 30
              force_finish_tool_result_max_chars: 3000
        "#;
        let cfg = Config::builder()
            .add_source(config::File::from_str(yaml, config::FileFormat::Yaml))
            .build()
            .unwrap();
        let settings: AgentSettings = cfg.try_deserialize().unwrap();
        assert_eq!(settings.max_active, 42);
        assert_eq!(settings.snapshot_frequency, 2000);
        assert_eq!(settings.max_full_snapshots, 5);
        assert_eq!(settings.max_projection_size, 1024);
        assert_eq!(settings.execution_budget.role_max_turns.check, Some(24));
        assert_eq!(settings.execution_budget.early_warning_remaining, 6);
        assert_eq!(settings.execution_budget.ca_evidence_focus_turns, 9);
        assert_eq!(settings.execution_budget.ca_evidence_close_turns, 14);
        assert_eq!(settings.execution_budget.da_repair_effect_block_turns, 4);
        assert_eq!(settings.execution_budget.pa_planning_focus_turns, 6);
        assert_eq!(settings.execution_budget.da_evidence_focus_turns, 7);
        assert_eq!(settings.execution_budget.da_evidence_close_turns, 12);
        assert_eq!(settings.execution_budget.max_sub_agents, 8);
        assert_eq!(settings.execution_budget.ca_handoff_max_chars, 9_000);
        assert_eq!(settings.execution_budget.max_plan_steps, 18);
        assert_eq!(settings.execution_budget.max_recursive_sub_tasks, 7);
        assert_eq!(settings.execution_budget.max_recursive_task_executions, 11);
        assert_eq!(settings.execution_budget.max_recursive_total_turns, 90);
        assert_eq!(settings.execution_budget.max_ca_da_corrections, 4);
    }

    #[test]
    fn test_agent_settings_tunables_default() {
        let yaml = r#"
            max_iterations: 10
            parallel_execution: true
            max_parallel_agents: 10
            timeout_seconds: 300
            api_timeout_seconds: 120
            event_bus_capacity: 100
        "#;
        let cfg = Config::builder()
            .add_source(config::File::from_str(yaml, config::FileFormat::Yaml))
            .build()
            .unwrap();
        let settings: AgentSettings = cfg.try_deserialize().unwrap();
        assert_eq!(settings.max_active, 20);
        assert_eq!(settings.snapshot_frequency, 1000);
        assert_eq!(settings.max_full_snapshots, 10);
        assert_eq!(settings.max_projection_size, 500);
        assert_eq!(settings.execution_budget.role_max_turns.check, None);
        assert_eq!(settings.execution_budget.early_warning_remaining, 8);
        assert_eq!(settings.execution_budget.final_warning_remaining, 3);
        assert_eq!(settings.execution_budget.max_plan_steps, 12);
        assert_eq!(settings.execution_budget.max_recursive_sub_tasks, 5);
        assert_eq!(settings.execution_budget.max_recursive_task_executions, 8);
        assert_eq!(settings.execution_budget.max_recursive_total_turns, 60);
    }

    #[test]
    fn repository_config_deserializes_runtime_limits() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.yaml");
        let config = Config::builder()
            .add_source(config::File::from(path))
            .build()
            .unwrap();
        let settings: Settings = config.try_deserialize().unwrap();
        assert_eq!(settings.agents.execution_budget.role_max_turns.check, None);
        assert_eq!(settings.agents.execution_budget.max_plan_steps, 12);
        assert_eq!(settings.agents.execution_budget.max_recursive_sub_tasks, 5);
        assert_eq!(
            settings
                .agents
                .execution_budget
                .max_recursive_task_executions,
            8
        );
        assert_eq!(
            settings.agents.execution_budget.max_recursive_total_turns,
            60
        );
        assert_eq!(settings.tool_result_router.micro_tool_page_size, 100);
        assert_eq!(settings.memory.l1.reload_preview_chars, 400);
        assert_eq!(settings.workspace.effect_snapshot_max_files, 10_000);
        assert_eq!(settings.workspace.effect_snapshot_max_bytes, 67_108_864);
        settings.validate().unwrap();
    }
}
