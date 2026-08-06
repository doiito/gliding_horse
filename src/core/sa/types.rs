use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::core::agent_instance::AgentRole;
use crate::CoreError;

/// 5 categories, 16 predefined intervention actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionAction {
    // === 1. Normal Continuation ===
    Continue,
    ContinueWithMonitor,

    // === 2. Parameter Tuning ===
    IncreaseRetry {
        additional_retries: u32,
    },
    IncreaseTimeout {
        additional_seconds: u64,
    },
    ReduceComplexity,
    RestrictTools {
        allowed_tools: Vec<String>,
    },

    // === 3. Execution Flow Adjustment ===
    SkipStep {
        step_id: String,
    },
    RetryStep {
        step_id: String,
    },
    Parallelize,
    SplitStep {
        step_id: String,
        sub_steps: Vec<String>,
    },
    InsertExtraStep {
        description: String,
    },

    // === 4. Resource & Mode Switch ===
    FallbackToShallow,
    EmergencyMode,
    IncreaseBudget {
        additional_tokens: u64,
        additional_time_secs: u64,
    },
    FreezeAndReport,

    // === 5. Termination & Escalation ===
    AbortTask {
        reason: String,
    },
    NotifyHuman {
        message: String,
    },
}

impl InterventionAction {
    pub fn from_name(name: &str, params: ActionParams) -> Result<Self, CoreError> {
        match name {
            "Continue" => Ok(InterventionAction::Continue),
            "ContinueWithMonitor" => Ok(InterventionAction::ContinueWithMonitor),
            "IncreaseRetry" => Ok(InterventionAction::IncreaseRetry {
                additional_retries: params.additional_retries.unwrap_or(3),
            }),
            "IncreaseTimeout" => Ok(InterventionAction::IncreaseTimeout {
                additional_seconds: params.additional_seconds.unwrap_or(60),
            }),
            "ReduceComplexity" => Ok(InterventionAction::ReduceComplexity),
            "RestrictTools" => Ok(InterventionAction::RestrictTools {
                allowed_tools: params.allowed_tools.unwrap_or_default(),
            }),
            "SkipStep" => Ok(InterventionAction::SkipStep {
                step_id: params.step_id.clone().unwrap_or_default(),
            }),
            "RetryStep" => Ok(InterventionAction::RetryStep {
                step_id: params.step_id.clone().unwrap_or_default(),
            }),
            "Parallelize" => Ok(InterventionAction::Parallelize),
            "SplitStep" => Ok(InterventionAction::SplitStep {
                step_id: params.step_id.clone().unwrap_or_default(),
                sub_steps: params.sub_steps.unwrap_or_default(),
            }),
            "InsertExtraStep" => Ok(InterventionAction::InsertExtraStep {
                description: params.description.clone().unwrap_or_default(),
            }),
            "FallbackToShallow" => Ok(InterventionAction::FallbackToShallow),
            "EmergencyMode" => Ok(InterventionAction::EmergencyMode),
            "IncreaseBudget" => Ok(InterventionAction::IncreaseBudget {
                additional_tokens: params.additional_tokens.unwrap_or(1000),
                additional_time_secs: params.additional_time_secs.unwrap_or(120),
            }),
            "FreezeAndReport" => Ok(InterventionAction::FreezeAndReport),
            "AbortTask" => Ok(InterventionAction::AbortTask {
                reason: params.reason.clone().unwrap_or_default(),
            }),
            "NotifyHuman" => Ok(InterventionAction::NotifyHuman {
                message: params.message.clone().unwrap_or_default(),
            }),
            _ => Err(CoreError::Internal {
                message: format!("Unknown intervention action: {}", name),
            }),
        }
    }
}

/// Action parameters (structured parameters from LLM output)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionParams {
    pub additional_retries: Option<u32>,
    pub additional_seconds: Option<u64>,
    pub additional_tokens: Option<u64>,
    pub additional_time_secs: Option<u64>,
    pub step_id: Option<String>,
    pub sub_steps: Option<Vec<String>>,
    pub description: Option<String>,
    pub allowed_tools: Option<Vec<String>>,
    pub reason: Option<String>,
    pub message: Option<String>,
}

/// Intermediate structure for LLM classification decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct LlmActionDecision {
    pub(super) action: String,
    #[serde(default)]
    pub(super) params: ActionParams,
    pub(super) reasoning: Option<String>,
}

/// 4 categories, 12 predefined supplementary input actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupplementaryInputAction {
    // === 1. Information Supplement ===
    AddContext,
    RefineObjective,
    ProvideConstraint,

    // === 2. Direction Guidance ===
    GuideDirection,
    PrioritizeStep,
    SuggestApproach,

    // === 3. Execution Control ===
    PauseExecution,
    ResumeExecution,
    SkipCurrentStep,

    // === 4. Feedback & Correction ===
    ConfirmDirection,
    CorrectApproach,
    AbortCurrentStep,
}

impl SupplementaryInputAction {
    pub fn from_name(name: &str) -> Result<Self, CoreError> {
        match name {
            "AddContext" => Ok(SupplementaryInputAction::AddContext),
            "RefineObjective" => Ok(SupplementaryInputAction::RefineObjective),
            "ProvideConstraint" => Ok(SupplementaryInputAction::ProvideConstraint),
            "GuideDirection" => Ok(SupplementaryInputAction::GuideDirection),
            "PrioritizeStep" => Ok(SupplementaryInputAction::PrioritizeStep),
            "SuggestApproach" => Ok(SupplementaryInputAction::SuggestApproach),
            "PauseExecution" => Ok(SupplementaryInputAction::PauseExecution),
            "ResumeExecution" => Ok(SupplementaryInputAction::ResumeExecution),
            "SkipCurrentStep" => Ok(SupplementaryInputAction::SkipCurrentStep),
            "ConfirmDirection" => Ok(SupplementaryInputAction::ConfirmDirection),
            "CorrectApproach" => Ok(SupplementaryInputAction::CorrectApproach),
            "AbortCurrentStep" => Ok(SupplementaryInputAction::AbortCurrentStep),
            _ => Err(CoreError::Internal {
                message: format!("Unknown supplementary input action: {}", name),
            }),
        }
    }
}

/// Intermediate structure for LLM classification decisions (supplementary input)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SupplementaryLlmDecision {
    pub(super) action: String,
    #[serde(default)]
    pub(super) params: ActionParams,
    pub(super) reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskComplexity {
    Instant,
    Simple,
    Standard,
    Complex,
    Exploratory,
    Emergency,
    Recursive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    pub sub_task_id: String,
    pub objective: String,
    pub parent_step_id: String,
    pub depth: u32,
    pub status: String,
}

impl SubTask {
    pub fn new(objective: &str, parent_step_id: &str, depth: u32) -> Self {
        Self {
            sub_task_id: format!("sub_{}", uuid::Uuid::new_v4().hyphenated()),
            objective: objective.to_string(),
            parent_step_id: parent_step_id.to_string(),
            depth,
            status: "pending".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub step_id: String,
    pub role: AgentRole,
    pub objective: String,
    pub expected_output: String,
    pub dependencies: Vec<String>,
    pub tools_allowed: Vec<String>,
    pub success_criteria: String,
}

/// Execution result of a human approval node
#[derive(Debug, Clone)]
pub struct HumanApprovalNodeResult {
    pub node_id: String,
    pub approved: bool,
    pub comment: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub plan_id: String,
    pub agent_sequence: Vec<AgentRole>,
    pub parallel_groups: Vec<Vec<AgentRole>>,
    pub task_complexity: TaskComplexity,
    pub description: String,
    pub steps: Vec<PlanStep>,
    pub context_requirements: HashMap<String, String>,
    pub success_metrics: Vec<String>,
    pub max_recursion_depth: u32,
    pub sub_tasks: Vec<SubTask>,
    /// Original JSON-LD DAG definition (set when loading from --workflow file)
    /// Used in execute_plan() to preserve DAG features (conditional branching, retry, parallelism)
    pub dag_jsonld: Option<String>,
    /// When true, execute_plan starts with CA→AA (verify-first).
    /// If CA verification fails, falls back to fallback_steps (PA→DA→CA→AA).
    pub verify_first: bool,
    /// Steps to execute as fallback when verify-first CA fails.
    pub fallback_steps: Vec<PlanStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CyclePhase {
    Idle,
    Analyzing,
    Dispatching,
    Executing,
    Monitoring,
    Completed,
}

#[derive(Debug, Clone)]
pub struct CycleState {
    pub cycle_id: String,
    pub task_iri: String,
    pub phase: CyclePhase,
    pub iteration: u32,
    pub max_iterations: u32,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub phase_history: Vec<String>,
    pub task_completed: bool,
    pub experience_hints: Vec<String>,
}

/// Decide whether a verify-first AA verdict requires full execution.
///
/// The agent_runner's `finish` action hardcodes `status: "success"` regardless
/// of what the verify-AA actually concluded, so the final status is meaningless
/// for verify-first plans. This recovers the intent from the AA's summary text.
/// Conservative by design: only an explicit completion confirmation avoids
/// re-execution; any missing/ambiguous/negative verdict means full PDCA is needed.
pub fn verify_aa_needs_execution(summary: &str) -> bool {
    let s = summary.to_lowercase();
    const COMPLETION_MARKERS: [&str; 8] = [
        "verified-pass",
        "verified pass",
        "task already done",
        "already done",
        "already satisfies",
        "already complete",
        "no execution needed",
        "no need to execute",
    ];
    !COMPLETION_MARKERS.iter().any(|m| s.contains(m))
}
