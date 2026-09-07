use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::core::agent_instance::AgentRole;
use crate::core::agent_runner::{TaskResult, TaskVerdict};
use crate::core::context_model::ExecutionPlanProvenance;
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
pub struct PlanWorkPackage {
    /// Stable identifier generated by SA for one business work package.  It
    /// remains independent of the child Agent id selected later by BizAgent.
    pub id: String,
    pub objective: String,
    pub expected_output: String,
    pub success_criteria: String,
    /// Kernel-checkable evidence required to complete this package. Every
    /// entry is mandatory (logical AND); child prose and an unrelated receipt
    /// can never substitute for a missing evidence class.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_requirements: Vec<WorkPackageEvidenceRequirement>,
    /// Canonical work-package ids that must complete before this package may
    /// start.  This is the task-level prerequisite contract, not an advisory
    /// ordering hint.
    #[serde(default)]
    pub dependencies: Vec<String>,
}

/// Typed evidence admitted at a canonical work-package boundary.
///
/// The enum is intentionally compact: artifact delivery and workspace
/// mutation are ownership/effect evidence, while all deterministic verifier
/// classes retain their kernel-assessed `VerificationKind`. In particular, a
/// successful build cannot satisfy a test-execution requirement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkPackageEvidenceRequirement {
    ArtifactDelivery {
        /// Exact canonical workspace-relative file paths promised by this
        /// package. Every declared path is mandatory; `min_paths` is retained
        /// in the wire format as an explicit cardinality guard and must equal
        /// `paths.len()`. Alternatives need a separate typed contract instead
        /// of weakening this production-delivery AND contract.
        paths: Vec<String>,
        min_paths: u32,
    },
    WorkspaceMutation {
        min_actions: u32,
    },
    Verification {
        kind: crate::core::tracked_action::VerificationKind,
        min_count: u64,
    },
    /// Exact newly-authored test artifacts that a TestExecution receipt must
    /// explicitly name. This is a kernel-compiled scope, not artifact
    /// ownership: the upstream writer remains the sole owner of these paths.
    TestArtifactExecutionScope {
        paths: Vec<String>,
    },
}

const MAX_WORK_PACKAGE_EVIDENCE_REQUIREMENTS: usize = 9;
const MAX_REQUIRED_ARTIFACT_PATHS: u32 = 10_000;
const MAX_REQUIRED_MUTATION_ACTIONS: u32 = 1_000;
const MAX_REQUIRED_VERIFICATION_COUNT: u64 = 1_000_000;

pub(crate) fn normalize_work_package_artifact_path(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty()
        || trimmed.chars().count() > 512
        || trimmed.chars().any(char::is_control)
        || trimmed.starts_with('/')
        || trimmed.ends_with('/')
        || trimmed.as_bytes().get(1) == Some(&b':')
    {
        return None;
    }
    let portable = trimmed.replace('\\', "/");
    let segments = portable.split('/').collect::<Vec<_>>();
    if segments
        .iter()
        .any(|segment| segment.is_empty() || matches!(*segment, "." | ".."))
    {
        return None;
    }
    let normalized = segments.join("/");
    (normalized == trimmed).then_some(normalized)
}

/// Validate the shape of one package's typed evidence contract. `require`
/// distinguishes a freshly admitted/executed contract from legacy JSON: old
/// snapshots deserialize through the serde default, then fail closed at the
/// execution boundary instead of being silently upgraded.
pub(crate) fn validate_work_package_evidence_requirements(
    package: &PlanWorkPackage,
    require: bool,
) -> Result<(), String> {
    if package.evidence_requirements.is_empty() {
        return if require {
            Err(format!(
                "work package '{}' has no typed evidence requirements",
                package.id
            ))
        } else {
            Ok(())
        };
    }
    if package.evidence_requirements.len() > MAX_WORK_PACKAGE_EVIDENCE_REQUIREMENTS {
        return Err(format!(
            "work package '{}' has too many evidence requirements (maximum {})",
            package.id, MAX_WORK_PACKAGE_EVIDENCE_REQUIREMENTS
        ));
    }

    let mut artifact_delivery_seen = false;
    let mut workspace_mutation_seen = false;
    let mut test_execution_scope_seen = false;
    let mut verification_kinds = Vec::new();
    for requirement in &package.evidence_requirements {
        match requirement {
            WorkPackageEvidenceRequirement::ArtifactDelivery { paths, min_paths } => {
                if artifact_delivery_seen {
                    return Err(format!(
                        "work package '{}' repeats artifact_delivery evidence",
                        package.id
                    ));
                }
                artifact_delivery_seen = true;
                if !(1..=MAX_REQUIRED_ARTIFACT_PATHS).contains(min_paths)
                    || paths.is_empty()
                    || paths.len() > MAX_REQUIRED_ARTIFACT_PATHS as usize
                    || *min_paths as usize != paths.len()
                {
                    return Err(format!(
                        "work package '{}' artifact_delivery requires 1..={} exact paths and min_paths equal to the complete path inventory",
                        package.id, MAX_REQUIRED_ARTIFACT_PATHS
                    ));
                }
                let mut unique_paths = HashSet::with_capacity(paths.len());
                for path in paths {
                    let normalized = normalize_work_package_artifact_path(path).ok_or_else(|| {
                        format!(
                            "work package '{}' artifact_delivery path '{}' must be a canonical workspace-relative file path",
                            package.id, path
                        )
                    })?;
                    if !unique_paths.insert(normalized) {
                        return Err(format!(
                            "work package '{}' repeats artifact_delivery path '{}'",
                            package.id, path
                        ));
                    }
                    if require && !package.expected_output.contains(path) {
                        return Err(format!(
                            "work package '{}' artifact_delivery path '{}' must appear literally in expected_output",
                            package.id, path
                        ));
                    }
                }
            }
            WorkPackageEvidenceRequirement::WorkspaceMutation { min_actions } => {
                if workspace_mutation_seen {
                    return Err(format!(
                        "work package '{}' repeats workspace_mutation evidence",
                        package.id
                    ));
                }
                workspace_mutation_seen = true;
                if !(1..=MAX_REQUIRED_MUTATION_ACTIONS).contains(min_actions) {
                    return Err(format!(
                        "work package '{}' workspace_mutation min_actions must be in 1..={}",
                        package.id, MAX_REQUIRED_MUTATION_ACTIONS
                    ));
                }
            }
            WorkPackageEvidenceRequirement::Verification { kind, min_count } => {
                if verification_kinds.contains(kind) {
                    return Err(format!(
                        "work package '{}' repeats verification kind {:?}",
                        package.id, kind
                    ));
                }
                verification_kinds.push(*kind);
                if !(1..=MAX_REQUIRED_VERIFICATION_COUNT).contains(min_count) {
                    return Err(format!(
                        "work package '{}' verification {:?} min_count must be in 1..={}",
                        package.id, kind, MAX_REQUIRED_VERIFICATION_COUNT
                    ));
                }
            }
            WorkPackageEvidenceRequirement::TestArtifactExecutionScope { paths } => {
                if test_execution_scope_seen {
                    return Err(format!(
                        "work package '{}' repeats test_artifact_execution_scope",
                        package.id
                    ));
                }
                test_execution_scope_seen = true;
                if paths.is_empty() || paths.len() > MAX_REQUIRED_ARTIFACT_PATHS as usize {
                    return Err(format!(
                        "work package '{}' test_artifact_execution_scope requires 1..={} exact paths",
                        package.id, MAX_REQUIRED_ARTIFACT_PATHS
                    ));
                }
                let mut unique_targets = HashSet::with_capacity(paths.len());
                for target in paths {
                    let normalized = normalize_work_package_artifact_path(target).ok_or_else(|| {
                        format!(
                            "work package '{}' verification target '{}' must be a canonical workspace-relative file path",
                            package.id, target
                        )
                    })?;
                    if normalized != *target || !unique_targets.insert(normalized) {
                        return Err(format!(
                            "work package '{}' verification target_paths must be canonical and unique",
                            package.id
                        ));
                    }
                }
            }
        }
    }
    if test_execution_scope_seen
        && !verification_kinds
            .contains(&crate::core::tracked_action::VerificationKind::TestExecution)
    {
        return Err(format!(
            "work package '{}' test_artifact_execution_scope requires verification kind test_execution",
            package.id
        ));
    }
    Ok(())
}

pub(crate) fn validate_plan_work_package_dag(
    work_packages: &[PlanWorkPackage],
) -> Result<(), String> {
    use std::collections::{HashMap, HashSet};

    let mut ids = HashSet::with_capacity(work_packages.len());
    for package in work_packages {
        if package.id.trim().is_empty() {
            return Err("work-package id must not be empty".to_string());
        }
        if package.id.chars().count() > 80
            || !package
                .id
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "_-".contains(character))
        {
            return Err(format!(
                "work-package id '{}' must use at most 80 ASCII letters, digits, '_' or '-'",
                package.id
            ));
        }
        if !ids.insert(package.id.as_str()) {
            return Err(format!("duplicate work-package id '{}'", package.id));
        }
        if package.objective.trim().is_empty()
            || package.expected_output.trim().is_empty()
            || package.success_criteria.trim().is_empty()
        {
            return Err(format!(
                "work package '{}' requires objective, expected_output and success_criteria",
                package.id
            ));
        }
        validate_work_package_evidence_requirements(package, false)?;
    }

    // ArtifactDelivery grants exact package ownership. Shared paths and
    // ancestor/descendant claims are ambiguous even when packages are
    // ordered; workflows that intentionally hand off the same file need a
    // different explicit typed contract.
    let artifact_paths = work_packages
        .iter()
        .flat_map(|package| {
            package
                .evidence_requirements
                .iter()
                .filter_map(|requirement| match requirement {
                    WorkPackageEvidenceRequirement::ArtifactDelivery { paths, .. } => Some(paths),
                    _ => None,
                })
                .flatten()
                .filter_map(|path| {
                    normalize_work_package_artifact_path(path)
                        .map(|normalized| (package.id.as_str(), normalized))
                })
        })
        .collect::<Vec<_>>();
    for (index, (left_owner, left)) in artifact_paths.iter().enumerate() {
        for (right_owner, right) in artifact_paths.iter().skip(index + 1) {
            let overlap = left == right
                || left
                    .strip_prefix(right.as_str())
                    .is_some_and(|suffix| suffix.starts_with('/'))
                || right
                    .strip_prefix(left.as_str())
                    .is_some_and(|suffix| suffix.starts_with('/'));
            if overlap {
                return Err(format!(
                    "artifact_delivery ownership overlaps between package '{}' path '{}' and package '{}' path '{}'",
                    left_owner, left, right_owner, right
                ));
            }
        }
    }

    for package in work_packages {
        let mut seen_dependencies = HashSet::new();
        for dependency in &package.dependencies {
            if dependency == &package.id {
                return Err(format!("work package '{}' depends on itself", package.id));
            }
            if !ids.contains(dependency.as_str()) {
                return Err(format!(
                    "work package '{}' depends on unknown package '{}'",
                    package.id, dependency
                ));
            }
            if !seen_dependencies.insert(dependency) {
                return Err(format!(
                    "work package '{}' repeats dependency '{}'",
                    package.id, dependency
                ));
            }
        }
    }

    let mut remaining = work_packages
        .iter()
        .map(|package| (package.id.as_str(), package.dependencies.len()))
        .collect::<HashMap<_, _>>();
    let mut ready = remaining
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    let mut visited = 0usize;
    while let Some(id) = ready.pop() {
        if remaining.remove(id).is_none() {
            continue;
        }
        visited += 1;
        for dependent in work_packages
            .iter()
            .filter(|package| package.dependencies.iter().any(|value| value == id))
        {
            if let Some(count) = remaining.get_mut(dependent.id.as_str()) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    ready.push(dependent.id.as_str());
                }
            }
        }
    }
    if visited == work_packages.len() {
        Ok(())
    } else {
        Err("work-package dependencies contain a cycle".to_string())
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
    /// Same-role business work that was collapsed into this one BizAgent
    /// parent.  Keeping its DAG here prevents role normalization from erasing
    /// explicit user prerequisites (for example A must be produced before B).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub work_packages: Vec<PlanWorkPackage>,
    #[serde(default)]
    pub branch_on_failure: bool,
    #[serde(default)]
    pub branch_fallback: Option<String>,
    #[serde(default)]
    pub retry_count: u32,
    #[serde(default)]
    pub retry_delay_secs: u64,
    #[serde(default)]
    pub effect_policy: crate::core::effect::EffectPolicy,
}

/// Execution result of a human approval node
#[derive(Debug, Clone)]
pub struct HumanApprovalNodeResult {
    pub node_id: String,
    pub approved: bool,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub plan_id: String,
    pub agent_sequence: Vec<AgentRole>,
    pub parallel_groups: Vec<Vec<AgentRole>>,
    pub task_complexity: TaskComplexity,
    pub description: String,
    pub steps: Vec<PlanStep>,
    /// Typed provenance for the plan and any steps with a different origin.
    /// `None` is accepted only as a construction-time state; execution rejects
    /// it unless an explicit external workflow source can be established.
    pub agent_spec_provenance: Option<ExecutionPlanProvenance>,
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

/// Mutable intervention state carried on a cycle, written by SA intervention
/// handlers (actions.rs) and consumed at dispatch time (execution.rs).
#[derive(Debug, Clone, Default)]
pub struct InterventionState {
    /// IncreaseRetry/DecreaseRetry effect: added to `max_iterations` at dispatch.
    pub max_iterations_delta: i32,
    /// IncreaseTimeout/DecreaseTimeout effect: added to dispatch timeout seconds.
    pub timeout_delta_secs: i64,
    /// RestrictTools effect: only these tools are offered to the agent.
    pub tool_allowlist_override: Option<Vec<String>>,
    /// ContinueWithMonitor effect: monitor the cycle for anomalies.
    pub monitor: bool,
}

#[derive(Debug, Clone)]
pub struct CycleState {
    pub cycle_id: String,
    pub task_iri: String,
    pub phase: CyclePhase,
    pub iteration: u32,
    pub max_iterations: u32,
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Start/deadline of the current outer PDCA attempt. `started_at` remains
    /// the task lifetime clock for task-level SLO reporting.
    pub pdca_started_at: chrono::DateTime<chrono::Utc>,
    pub cycle_deadline_at: chrono::DateTime<chrono::Utc>,
    pub last_progress_at: chrono::DateTime<chrono::Utc>,
    pub last_timeout_alert_at: Option<chrono::DateTime<chrono::Utc>>,
    pub next_timeout_alert_at: Option<chrono::DateTime<chrono::Utc>>,
    pub timeout_alert_count: u32,
    pub outer_cycle_number: u32,
    pub phase_history: Vec<String>,
    pub task_completed: bool,
    /// Number retrieved before treatment is applied; used by shadow-mode
    /// evaluation without exposing the content to BizAgents.
    pub observed_experience_hint_count: usize,
    /// Stable, non-content fingerprints prove which experience records were
    /// observed across treatments without leaking their prompt text.
    pub observed_experience_hint_fingerprints: Vec<String>,
    pub experience_hints: Vec<String>,
    /// Accumulated SA intervention state, applied at dispatch time.
    pub intervention: InterventionState,
}

/// Decide whether a verify-first AA verdict requires full execution.
///
/// The agent_runner's `finish` action hardcodes `status: "success"` regardless
/// of what the verify-AA actually concluded, so the final status is meaningless
/// for verify-first plans. The structured `verdict` takes priority: blocked,
/// failed, and timeout verdicts always require execution; success and partial
/// success fall through to a summary double-check. Summary text matching is the
/// fallback for results without a structured verdict. Conservative by design:
/// only an explicit completion confirmation avoids re-execution; any
/// missing/ambiguous/negative verdict means full PDCA is needed.
pub fn verify_aa_needs_execution(result: &TaskResult) -> bool {
    if let Some(verdict) = result.verdict {
        match verdict {
            TaskVerdict::Blocked | TaskVerdict::Failed | TaskVerdict::Timeout => return true,
            TaskVerdict::Success | TaskVerdict::PartialSuccess => {}
        }
    }
    let s = result.summary.to_lowercase();
    const COMPLETION_MARKERS: [&str; 12] = [
        "success:",
        "aa success:",
        "pass:",
        "ca pass:",
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

#[cfg(test)]
mod work_package_evidence_tests {
    use super::*;

    fn package(evidence_requirements: Vec<WorkPackageEvidenceRequirement>) -> PlanWorkPackage {
        PlanWorkPackage {
            id: "package".to_string(),
            objective: "produce result".to_string(),
            expected_output: "result.txt".to_string(),
            success_criteria: "result exists".to_string(),
            evidence_requirements,
            dependencies: Vec::new(),
        }
    }

    #[test]
    fn legacy_missing_evidence_deserializes_but_fails_execution_admission() {
        let legacy: PlanWorkPackage = serde_json::from_value(serde_json::json!({
            "id": "legacy",
            "objective": "old objective",
            "expected_output": "old.txt",
            "success_criteria": "old complete",
            "dependencies": []
        }))
        .unwrap();
        assert!(legacy.evidence_requirements.is_empty());
        assert!(validate_work_package_evidence_requirements(&legacy, false).is_ok());
        assert!(validate_work_package_evidence_requirements(&legacy, true)
            .unwrap_err()
            .contains("no typed evidence"));
    }

    #[test]
    fn duplicate_or_zero_evidence_thresholds_are_rejected() {
        let duplicate = package(vec![
            WorkPackageEvidenceRequirement::Verification {
                kind: crate::core::tracked_action::VerificationKind::TestExecution,
                min_count: 1,
            },
            WorkPackageEvidenceRequirement::Verification {
                kind: crate::core::tracked_action::VerificationKind::TestExecution,
                min_count: 2,
            },
        ]);
        assert!(
            validate_work_package_evidence_requirements(&duplicate, true)
                .unwrap_err()
                .contains("repeats verification")
        );

        let zero = package(vec![WorkPackageEvidenceRequirement::ArtifactDelivery {
            paths: vec!["result.txt".to_string()],
            min_paths: 0,
        }]);
        assert!(validate_work_package_evidence_requirements(&zero, true)
            .unwrap_err()
            .contains("1..="));

        let partial_inventory = package(vec![WorkPackageEvidenceRequirement::ArtifactDelivery {
            paths: vec!["result.txt".to_string(), "README.md".to_string()],
            min_paths: 1,
        }]);
        assert!(
            validate_work_package_evidence_requirements(&partial_inventory, true)
                .unwrap_err()
                .contains("complete path inventory")
        );

        let mismatched_output = package(vec![WorkPackageEvidenceRequirement::ArtifactDelivery {
            paths: vec!["junk.tmp".to_string()],
            min_paths: 1,
        }]);
        assert!(
            validate_work_package_evidence_requirements(&mismatched_output, true)
                .unwrap_err()
                .contains("must appear literally in expected_output")
        );

        for path in ["/absolute/result.txt", "../escape.txt", "a/../result.txt"] {
            let unsafe_path = package(vec![WorkPackageEvidenceRequirement::ArtifactDelivery {
                paths: vec![path.to_string()],
                min_paths: 1,
            }]);
            assert!(
                validate_work_package_evidence_requirements(&unsafe_path, true)
                    .unwrap_err()
                    .contains("workspace-relative")
            );
        }
        let duplicate_path = package(vec![WorkPackageEvidenceRequirement::ArtifactDelivery {
            paths: vec!["result.txt".to_string(), "result.txt".to_string()],
            min_paths: 2,
        }]);
        assert!(
            validate_work_package_evidence_requirements(&duplicate_path, true)
                .unwrap_err()
                .contains("repeats artifact_delivery path")
        );
    }

    #[test]
    fn artifact_ownership_rejects_cross_package_equal_or_nested_paths() {
        let artifact = |id: &str, path: &str| PlanWorkPackage {
            id: id.to_string(),
            objective: format!("deliver {path}"),
            expected_output: path.to_string(),
            success_criteria: format!("{path} exists"),
            evidence_requirements: vec![WorkPackageEvidenceRequirement::ArtifactDelivery {
                paths: vec![path.to_string()],
                min_paths: 1,
            }],
            dependencies: Vec::new(),
        };
        for right in ["project", "project/app.py", "project/app.py/generated"] {
            let packages = vec![artifact("left", "project/app.py"), artifact("right", right)];
            assert!(validate_plan_work_package_dag(&packages)
                .unwrap_err()
                .contains("ownership overlaps"));
        }
        validate_plan_work_package_dag(&[
            artifact("left", "project/app.py"),
            artifact("right", "project/tests/test_app.py"),
        ])
        .unwrap();
    }
}
