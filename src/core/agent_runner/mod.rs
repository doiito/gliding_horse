use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use serde_json::Value;
use tracing::{info, warn};

use crate::causal::fused::FusedRootCauseEngine;
use crate::config::settings::AgentSettings;
use crate::config::RuntimeHookConfig;
use crate::core::constitution::ConstitutionRegistry;
use crate::core::context_compressor::{ContextWindowManager, ToolResultCompressor};
use crate::core::relevance_tracker::RelevanceTracker;
use crate::gateway::unified_gateway::{ChatMessage, UnifiedGateway};
use crate::llm::LlmInteractionService;
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
use crate::tools::builtin::hooks::HookRunner;
use crate::tools::hooks::{HookManager, LoggingHook, MetricsHook, RateLimitHook, TimingHook};
use crate::tools::sharing::SharingProtocol;
use crate::tools::skill_registry::SkillRegistry;
use crate::tools::tool_executor::ToolExecutor;
use crate::tools::tool_guard::ToolGuard;

mod execution;
pub(crate) use execution::{
    explicit_test_execution_target_path, structured_ca_verdict, CA_DA_CORRECTION_MODE,
    SA_RECOVERY_MODE_CONSTRAINT,
};
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

/// Application-declared project layout requirement.  This is separate from
/// the workspace scope: the configured workspace is the security boundary,
/// while a newly created child directory can be part of the user's acceptance
/// criteria.  Keeping the value typed prevents model-authored plans from
/// weakening "new directory" into "use the existing workspace root".
pub const WORKSPACE_LAYOUT_CONSTRAINT: &str = "workspace_layout";
pub const WORKSPACE_LAYOUT_NEW_CHILD_DIRECTORY: &str = "new_child_directory";

pub(crate) fn new_child_directory_contract(
    constraints: &HashMap<String, String>,
    role: crate::core::agent_instance::AgentRole,
) -> Option<&'static str> {
    if !constraints
        .get(WORKSPACE_LAYOUT_CONSTRAINT)
        .is_some_and(|value| value == WORKSPACE_LAYOUT_NEW_CHILD_DIRECTORY)
    {
        return None;
    }

    use crate::core::agent_instance::AgentRole;
    match role {
        AgentRole::Plan => Some(
            "The user requires a new project directory strictly below the configured workspace. PA must name one workspace-relative child-directory path and place every planned project artifact (design, source, tests, and documentation) below that path. `.` and the configured workspace root do not satisfy this requirement.",
        ),
        AgentRole::Do => Some(
            "The user requires a new project directory strictly below the configured workspace. DA must create that child directory before creating project artifacts, keep the design, source, tests, and documentation below it, and report the exact workspace-relative directory and artifact paths. Writing project artifacts directly in the configured workspace root does not satisfy this requirement.",
        ),
        AgentRole::Check => Some(
            "The user requires a new project directory strictly below the configured workspace. CA must independently verify that the reported project directory exists, is a strict descendant rather than the configured workspace root itself, and contains the task's design, source, tests, and documentation. Root-level project artifacts or a missing child directory require a failing verdict.",
        ),
        AgentRole::Act => None,
    }
}

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

/// Kernel-owned contract established from the original user order and a
/// validated prerequisite DAG.  The serialized value is checkpoint data, not
/// free-form prompt text: every consumer must deserialize and validate it
/// before granting it authority.
pub const CONFORMANCE_CONTRACT_CONSTRAINT: &str = "conformance_contract";
pub const CONFORMANCE_CONTRACT_SCHEMA_VERSION: &str = "glidinghorse.conformance-contract/v2";
const MAX_CONFORMANCE_CONTRACT_BYTES: usize = 48 * 1024;
const MAX_CONFORMANCE_RELATIONS: usize = 32;
const MAX_CONFORMANCE_SUCCESSORS_PER_RELATION: usize = 64;
const MAX_CONFORMANCE_PATHS_PER_PACKAGE: usize = 128;
const MAX_CONFORMANCE_VERIFICATION_RECEIPTS_PER_PACKAGE: usize = 16;
const MAX_CONFORMANCE_TOTAL_PATHS: usize = 512;
const MAX_CONFORMANCE_PATH_CHARS: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConformanceContract {
    pub schema_version: String,
    /// The plan revision whose validated work-package DAG created the
    /// relations. A CA-only retry deliberately retains this source identity.
    pub source_plan_id: String,
    pub relations: Vec<NormativeDesignRelation>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NormativeDesignRelation {
    pub do_step_id: String,
    pub design_predecessor_id: String,
    /// Every transitive successor, including tests and documentation rather
    /// than only the first implementation package.
    pub transitive_successor_ids: Vec<String>,
    pub evidence: ConformanceRelationEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ConformanceRelationEvidence {
    Planned,
    Verified {
        order_receipt_sha256: String,
        design_paths: Vec<String>,
        successor_deliveries: Vec<WorkPackageDeliveryEvidence>,
    },
    Unavailable {
        reason: ConformanceUnavailableReason,
        missing_work_package_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ConformanceUnavailableReason {
    MissingOrderReceipt,
    MultipleOrderReceipts,
    InvalidOrderReceipt,
    InvalidArtifactPath,
    WorkspaceRootUnavailable,
    MissingDesignPaths,
    MissingSuccessorPaths,
    ArtifactOwnershipConflict,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "evidence_kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum WorkPackageDeliveryEvidence {
    ArtifactDelivery {
        work_package_id: String,
        paths: Vec<String>,
    },
    VerificationExecution {
        work_package_id: String,
        verification_receipt_sha256s: Vec<String>,
    },
}

impl WorkPackageDeliveryEvidence {
    pub(crate) fn work_package_id(&self) -> &str {
        match self {
            Self::ArtifactDelivery {
                work_package_id, ..
            }
            | Self::VerificationExecution {
                work_package_id, ..
            } => work_package_id,
        }
    }
}

/// Pair-preserving view for validators that must prove a design artifact was
/// compared with the deliveries of its own successor work packages rather
/// than merely matching a task-wide flattened union.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedConformanceRelationPaths {
    pub do_step_id: String,
    pub design_predecessor_id: String,
    pub design_paths: std::collections::BTreeSet<String>,
    pub successor_deliveries:
        std::collections::BTreeMap<String, VerifiedConformanceSuccessorEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VerifiedConformanceSuccessorEvidence {
    ArtifactPaths(std::collections::BTreeSet<String>),
    VerificationReceipts(std::collections::BTreeSet<String>),
}

fn valid_contract_id(value: &str) -> bool {
    !value.is_empty()
        && value.chars().count() <= 80
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_-".contains(character))
}

pub(crate) fn valid_conformance_path(value: &str) -> bool {
    let path = Path::new(value);
    if value.is_empty()
        || value.chars().count() > MAX_CONFORMANCE_PATH_CHARS
        || path.is_absolute()
        || value.chars().any(char::is_control)
    {
        return false;
    }
    let components = path.components().collect::<Vec<_>>();
    !components.is_empty()
        && components
            .iter()
            .all(|component| matches!(component, Component::Normal(_)))
        && components
            .iter()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
            == value
}

fn strictly_sorted_unique(values: &[String]) -> bool {
    values
        .windows(2)
        .all(|window| window[0].as_str() < window[1].as_str())
}

fn valid_prefixed_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

impl ConformanceContract {
    pub(crate) fn planned(
        source_plan_id: impl Into<String>,
        mut relations: Vec<NormativeDesignRelation>,
    ) -> Result<Self, String> {
        for relation in &mut relations {
            relation.transitive_successor_ids.sort();
            relation.transitive_successor_ids.dedup();
            relation.evidence = ConformanceRelationEvidence::Planned;
        }
        relations.sort_by(|left, right| {
            (&left.do_step_id, &left.design_predecessor_id)
                .cmp(&(&right.do_step_id, &right.design_predecessor_id))
        });
        let contract = Self {
            schema_version: CONFORMANCE_CONTRACT_SCHEMA_VERSION.to_string(),
            source_plan_id: source_plan_id.into(),
            relations,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.schema_version != CONFORMANCE_CONTRACT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported conformance contract schema '{}'",
                self.schema_version
            ));
        }
        if self.source_plan_id.trim().is_empty()
            || self.source_plan_id.chars().count() > 256
            || self.source_plan_id.chars().any(char::is_control)
        {
            return Err("conformance contract has an invalid source plan id".to_string());
        }
        if self.relations.is_empty() {
            return Err("conformance contract has no design relations".to_string());
        }
        if self.relations.len() > MAX_CONFORMANCE_RELATIONS {
            return Err("conformance contract contains too many design relations".to_string());
        }

        let mut prior_relation: Option<(&str, &str)> = None;
        let mut total_paths = 0usize;
        for relation in &self.relations {
            if !valid_contract_id(&relation.do_step_id)
                || !valid_contract_id(&relation.design_predecessor_id)
            {
                return Err("conformance relation has an invalid step or package id".to_string());
            }
            let identity = (
                relation.do_step_id.as_str(),
                relation.design_predecessor_id.as_str(),
            );
            if prior_relation.is_some_and(|prior| prior >= identity) {
                return Err("conformance relations must be strictly sorted and unique".to_string());
            }
            prior_relation = Some(identity);
            if relation.transitive_successor_ids.is_empty()
                || relation.transitive_successor_ids.len() > MAX_CONFORMANCE_SUCCESSORS_PER_RELATION
                || !strictly_sorted_unique(&relation.transitive_successor_ids)
                || relation
                    .transitive_successor_ids
                    .iter()
                    .any(|id| !valid_contract_id(id) || id == &relation.design_predecessor_id)
            {
                return Err(
                    "conformance successors must be non-empty, valid, sorted and unique"
                        .to_string(),
                );
            }

            let relevant_ids = std::iter::once(&relation.design_predecessor_id)
                .chain(relation.transitive_successor_ids.iter())
                .collect::<HashSet<_>>();
            match &relation.evidence {
                ConformanceRelationEvidence::Planned => {}
                ConformanceRelationEvidence::Verified {
                    order_receipt_sha256,
                    design_paths,
                    successor_deliveries,
                } => {
                    if !valid_prefixed_sha256(order_receipt_sha256)
                        || design_paths.is_empty()
                        || design_paths.len() > MAX_CONFORMANCE_PATHS_PER_PACKAGE
                        || !strictly_sorted_unique(design_paths)
                        || design_paths
                            .iter()
                            .any(|path| !valid_conformance_path(path))
                    {
                        return Err(
                            "verified conformance has an invalid receipt digest or design paths"
                                .to_string(),
                        );
                    }
                    total_paths = total_paths.saturating_add(design_paths.len());
                    let delivery_ids = successor_deliveries
                        .iter()
                        .map(WorkPackageDeliveryEvidence::work_package_id)
                        .collect::<Vec<_>>();
                    if delivery_ids.len() != relation.transitive_successor_ids.len()
                        || !delivery_ids
                            .iter()
                            .zip(relation.transitive_successor_ids.iter())
                            .all(|(actual, expected)| *actual == expected)
                    {
                        return Err(
                            "verified conformance does not bind every successor in canonical order"
                                .to_string(),
                        );
                    }
                    let mut artifact_delivery_observed = false;
                    for delivery in successor_deliveries {
                        match delivery {
                            WorkPackageDeliveryEvidence::ArtifactDelivery { paths, .. } => {
                                artifact_delivery_observed = true;
                                if paths.is_empty()
                                    || paths.len() > MAX_CONFORMANCE_PATHS_PER_PACKAGE
                                    || !strictly_sorted_unique(paths)
                                    || paths.iter().any(|path| !valid_conformance_path(path))
                                {
                                    return Err(
                                        "artifact delivery evidence requires safe, non-empty, sorted unique paths"
                                            .to_string(),
                                    );
                                }
                                total_paths = total_paths.saturating_add(paths.len());
                            }
                            WorkPackageDeliveryEvidence::VerificationExecution {
                                verification_receipt_sha256s,
                                ..
                            } => {
                                if verification_receipt_sha256s.is_empty()
                                    || verification_receipt_sha256s.len()
                                        > MAX_CONFORMANCE_VERIFICATION_RECEIPTS_PER_PACKAGE
                                    || !strictly_sorted_unique(verification_receipt_sha256s)
                                    || verification_receipt_sha256s
                                        .iter()
                                        .any(|digest| !valid_prefixed_sha256(digest))
                                {
                                    return Err(
                                        "verification execution evidence requires valid, non-empty, sorted unique receipts"
                                            .to_string(),
                                    );
                                }
                            }
                        }
                    }
                    if !artifact_delivery_observed {
                        return Err(
                            "a design-to-implementation relation requires at least one artifact delivery successor"
                                .to_string(),
                        );
                    }
                }
                ConformanceRelationEvidence::Unavailable {
                    missing_work_package_ids,
                    ..
                } => {
                    if missing_work_package_ids.is_empty()
                        || !strictly_sorted_unique(missing_work_package_ids)
                        || missing_work_package_ids
                            .iter()
                            .any(|id| !relevant_ids.contains(id))
                    {
                        return Err(
                            "unavailable conformance must identify valid missing work packages"
                                .to_string(),
                        );
                    }
                }
            }
        }
        if total_paths > MAX_CONFORMANCE_TOTAL_PATHS {
            return Err("conformance contract contains too many artifact paths".to_string());
        }
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("failed to size conformance contract: {error}"))?;
        if encoded.len() > MAX_CONFORMANCE_CONTRACT_BYTES {
            return Err(
                "conformance contract exceeds its authoritative context budget".to_string(),
            );
        }
        Ok(())
    }

    pub(crate) fn to_constraint_value(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("failed to serialize conformance contract: {error}"))
    }

    pub(crate) fn from_constraint_value(value: &str) -> Result<Self, String> {
        let contract = serde_json::from_str::<Self>(value)
            .map_err(|error| format!("invalid conformance contract JSON: {error}"))?;
        contract.validate()?;
        Ok(contract)
    }

    pub(crate) fn is_fully_verified(&self) -> bool {
        self.relations.iter().all(|relation| {
            matches!(
                relation.evidence,
                ConformanceRelationEvidence::Verified { .. }
            )
        })
    }

    pub(crate) fn verified_relation_path_sets(
        &self,
    ) -> Option<Vec<VerifiedConformanceRelationPaths>> {
        if !self.is_fully_verified() {
            return None;
        }
        self.relations
            .iter()
            .map(|relation| {
                let ConformanceRelationEvidence::Verified {
                    design_paths,
                    successor_deliveries,
                    ..
                } = &relation.evidence
                else {
                    return None;
                };
                Some(VerifiedConformanceRelationPaths {
                    do_step_id: relation.do_step_id.clone(),
                    design_predecessor_id: relation.design_predecessor_id.clone(),
                    design_paths: design_paths.iter().cloned().collect(),
                    successor_deliveries: successor_deliveries
                        .iter()
                        .map(|delivery| match delivery {
                            WorkPackageDeliveryEvidence::ArtifactDelivery {
                                work_package_id,
                                paths,
                            } => (
                                work_package_id.clone(),
                                VerifiedConformanceSuccessorEvidence::ArtifactPaths(
                                    paths.iter().cloned().collect(),
                                ),
                            ),
                            WorkPackageDeliveryEvidence::VerificationExecution {
                                work_package_id,
                                verification_receipt_sha256s,
                            } => (
                                work_package_id.clone(),
                                VerifiedConformanceSuccessorEvidence::VerificationReceipts(
                                    verification_receipt_sha256s.iter().cloned().collect(),
                                ),
                            ),
                        })
                        .collect(),
                })
            })
            .collect()
    }
}

/// Valid serialized fixture retained for existing internal prompt tests. New
/// production code must construct a `ConformanceContract` from a validated
/// plan and must not use this value as a marker.
pub const CONFORMANCE_CONTRACT_NORMATIVE_DESIGN: &str = r#"{"schema_version":"glidinghorse.conformance-contract/v2","source_plan_id":"test-plan","relations":[{"do_step_id":"do","design_predecessor_id":"design","transitive_successor_ids":["implementation"],"evidence":{"status":"planned"}}]}"#;

pub(crate) fn normative_design_conformance_required(constraints: &HashMap<String, String>) -> bool {
    constraints
        .get(CONFORMANCE_CONTRACT_CONSTRAINT)
        .is_some_and(|value| ConformanceContract::from_constraint_value(value).is_ok())
}

/// Select the most actionable classification when independent CA evidence
/// contains more than one legitimate non-pass cause.  Keep this precedence in
/// the kernel so child aggregation and terminal-contract validation cannot
/// disagree about an otherwise valid audit.
pub(crate) fn preferred_ca_failure_class<'a>(
    classes: impl Iterator<Item = &'a str>,
) -> Option<&'static str> {
    let mut saw_verification_gap = false;
    let mut saw_external_blocker = false;
    for class in classes {
        match class.trim() {
            "observed_defect" => return Some("observed_defect"),
            "verification_gap" => saw_verification_gap = true,
            "external_blocker" => saw_external_blocker = true,
            _ => {}
        }
    }
    if saw_verification_gap {
        Some("verification_gap")
    } else if saw_external_blocker {
        Some("external_blocker")
    } else {
        None
    }
}

pub(crate) fn parsed_conformance_contract(
    constraints: &HashMap<String, String>,
) -> Option<ConformanceContract> {
    constraints
        .get(CONFORMANCE_CONTRACT_CONSTRAINT)
        .and_then(|value| ConformanceContract::from_constraint_value(value).ok())
}

pub(crate) fn normative_design_conformance_contract(
    constraints: &HashMap<String, String>,
) -> Option<String> {
    let contract = parsed_conformance_contract(constraints)?;
    let serialized = serde_json::to_string_pretty(&contract).ok()?;
    let state = if contract.is_fully_verified() {
        "Every relation is receipt-verified. CA may inspect only the exact design and successor delivery paths listed below for this conformance decision."
    } else if contract.relations.iter().any(|relation| {
        matches!(
            relation.evidence,
            ConformanceRelationEvidence::Unavailable { .. }
        )
    }) {
        "At least one relation is unavailable. CA must return a non-pass design_conformance verdict with failure_class `verification_gap`; model prose, summaries, or guessed paths cannot repair this kernel evidence gap."
    } else {
        "The relation is planned but has no Do receipt yet. DA must execute the exact predecessor/successor work-package order; this state cannot support a positive CA verdict."
    };
    Some(format!(
        "A validated prerequisite DAG establishes the following normative design relations. Success requires explicit cross-artifact conformance for file/module layout, public interfaces, behavior/data flow, architecture/algorithms, and user documentation. Equivalent implementation choices are allowed only when the design labels them non-normative or the design is updated before completion to describe the delivered project truthfully.\n\n{state}\n\n## Kernel ConformanceContract (exact JSON)\n{serialized}"
    ))
}

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

/// Canonical delivery boundary included in the typed task contract. Absence
/// of an application override is explicit so prompt construction never has to
/// infer a delivery mode from words such as "report" or "output".
pub(crate) fn normalized_delivery_contract(constraints: &HashMap<String, String>) -> String {
    if let Some(contract) = direct_response_delivery_contract(constraints) {
        contract.to_string()
    } else if let Some(contract) = workspace_artifact_delivery_contract(constraints) {
        contract
    } else {
        "No application delivery override is declared. Follow the original user request exactly and do not invent an additional file, path, graph node, or external side effect."
            .to_string()
    }
}

/// Canonical text for the runtime-enforced effect policy. This representation
/// is shared by typed context messages and policy diagnostics.
pub(crate) fn normalized_effect_contract(policy: &crate::core::effect::EffectPolicy) -> String {
    use crate::core::effect::EffectPolicy;
    match policy {
        EffectPolicy::None => "Effect policy is none: no additional effect is required beyond the original task and delivery contract.".to_string(),
        EffectPolicy::Required { effect } => format!(
            "Effect policy is required: completion requires concrete {effect:?} evidence. A narrative claim alone is insufficient."
        ),
        EffectPolicy::Conditional { effect, condition } => format!(
            "Effect policy is conditional: verify `{condition}`; when it holds, produce {effect:?}, otherwise report concrete evidence that it does not hold."
        ),
        EffectPolicy::EvidenceOnly => "Effect policy is evidence_only: inspect and report evidence without creating external effects or mutating workspace state.".to_string(),
        EffectPolicy::DecisionOnly => "Effect policy is decision_only: decide only from admitted evidence and do not invoke execution or mutation tools.".to_string(),
    }
}

/// AgentTurn identities must be unique inside a task. Each BizAgent owns a
/// distinct L1 session, so adding that session to the path prevents PA/DA/CA/
/// AA and later PDCA cycles from overwriting one another while preserving the
/// task prefix and the familiar turn suffix.
pub(crate) fn agent_turn_iri_prefix(task_iri: &str, session_id: &str) -> String {
    let task_id = task_iri
        .strip_prefix("iri://task/")
        .unwrap_or_else(|| task_iri.strip_prefix("iri://").unwrap_or(task_iri));
    format!("iri://task/{task_id}/session/{session_id}/turn_")
}

pub(crate) fn agent_turn_iri(task_iri: &str, session_id: &str, turn: u32) -> String {
    format!("{}{turn}", agent_turn_iri_prefix(task_iri, session_id))
}

/// A role-to-role handoff whose trust is determined by the typed field that
/// carries it, never by words such as `PASS` inside the payload. Construction
/// is crate-private so applications cannot label arbitrary text as a verified
/// CA result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentContextHandoff {
    pub content: String,
    pub source_ref: String,
    pub producer: String,
}

impl AgentContextHandoff {
    fn new(
        content: impl Into<String>,
        source_ref: impl Into<String>,
        producer: impl Into<String>,
    ) -> Self {
        let content = content.into();
        let source_ref = source_ref.into();
        Self {
            content: crate::tools::tool_executor::sanitize_session_tool_references(&content).0,
            source_ref: crate::tools::tool_executor::sanitize_session_tool_references(&source_ref)
                .0,
            producer: producer.into(),
        }
    }
}

fn sanitize_cross_agent_message(mut message: ChatMessage) -> ChatMessage {
    let sanitize_text =
        |text: &str| crate::tools::tool_executor::sanitize_session_tool_references(text).0;
    let is_ephemeral_tool = |name: &str| {
        name.starts_with("read_full_result_")
            || crate::tools::result_router::is_session_scoped_micro_tool_name(name)
    };

    message.content = sanitize_text(&message.content);
    message.reasoning_content = message.reasoning_content.as_deref().map(sanitize_text);
    if message.name.as_deref().is_some_and(is_ephemeral_tool) {
        message.name = Some("session_result_unavailable".to_string());
    }
    if let Some(calls) = message.tool_calls.as_mut() {
        for call in calls {
            if is_ephemeral_tool(&call.function.name) {
                call.function.name = "session_result_unavailable".to_string();
                call.function.arguments = "{}".to_string();
            } else {
                call.function.arguments = sanitize_text(&call.function.arguments);
            }
        }
    }
    message
}

#[derive(Debug, Clone)]
pub struct TaskContext {
    pub task_iri: String,
    pub objective: String,
    pub parent_task_iri: Option<String>,
    /// Telemetry-only correlation to the model interaction that created this
    /// task (for example, a BizAgent decomposition call). It is deliberately
    /// not rendered into the model prompt as an instruction.
    pub parent_interaction_id: Option<String>,
    pub input_data: HashMap<String, Value>,
    /// Application execution metadata. Unknown keys are never promoted to
    /// authoritative model instructions; only kernel-recognized contracts
    /// (delivery, evidence capability and effect policy) are normalized into
    /// authoritative typed fragments.
    pub(crate) constraints: HashMap<String, String>,
    pub max_iterations: u32,
    pub prev_agent_summary: Option<String>,
    /// PA output explicitly handed to DA. It remains unverified model history,
    /// but its stable archive source is a kernel-issued, exact read capability.
    /// Generic previous-agent summaries never populate this field.
    pub(crate) plan_handoff: Option<AgentContextHandoff>,
    /// DA deliverable under review by CA. This is an unverified execution
    /// subject (`ModelHistory`), not a success assertion.
    pub(crate) execution_handoff: Option<AgentContextHandoff>,
    /// CA evidence admitted to AA only after SA's verification/audit path has
    /// completed. Generic previous-agent text can never populate this field.
    pub(crate) verified_check_handoff: Option<AgentContextHandoff>,
    /// Bounded prior-cycle observations available only to planning/execution;
    /// they remain model history and never enter CA/AA.
    pub(crate) historical_experience: Vec<String>,
    /// Bounded DA repair input (prior deliverable and CA findings), kept out of
    /// the objective so it cannot bypass role-context admission.
    pub(crate) correction_handoff: Option<AgentContextHandoff>,
    /// Runtime-only source for a kernel-validated BizAgent delta recovery.
    ///
    /// This result is never rendered into an LLM prompt and never restores a
    /// producing Agent's transcript/L1 session.  A fresh corrective BizAgent
    /// may use it only to cross-check its deterministic child manifest,
    /// canonical order receipt and tracked-action ledger, then retain already
    /// successful prerequisite packages while re-dispatching failed/blocked
    /// packages as fresh child Agents.
    pub(crate) prior_biz_agent_result: Option<Arc<TaskResult>>,
    /// Canonical package ids invalidated by the structured CA audit. This is
    /// runtime-only parent-orchestrator metadata paired with the prior result;
    /// it is never rendered to a model and is cleared at every child boundary.
    pub(crate) biz_agent_recovery_invalidated_packages: Arc<HashSet<String>>,
    /// Kernel-authenticated evidence contract for one concrete BizAgent
    /// child. The human-readable copy in `input_data` is prompt context only;
    /// convergence gates must use this runtime-only value so an embedder or
    /// model-authored payload cannot forge completion authority.
    pub(crate) biz_agent_child_evidence_contract: Option<Arc<crate::core::sa::PlanWorkPackage>>,
    pub original_task: Option<String>,
    pub completed_steps: Vec<String>,
    pub pending_steps: Vec<String>,
    pub five_w2h_iri: String,
    pub five_w2h_snapshot: Option<crate::core::five_w2h::Task5W2H>,
    /// Bounded prior completed turns for ordinary multi-turn continuity.
    /// Unlike `resumed_messages`, this never restores counters, skips phases,
    /// or activates execution-journal replay semantics.
    pub conversation_history: Option<Vec<ChatMessage>>,
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
    /// Kernel-enforced exact workspace mutation lease for a parallel
    /// BizAgent child. This is execution metadata and is never rendered into
    /// the model context.
    pub workspace_resource_lease: Option<crate::core::effect::WorkspaceResourceLease>,
    /// Monotonic outer dispatch deadline. This is runtime-only kernel
    /// metadata used by optional tail work (for example BizAgent prose
    /// aggregation) so it can yield before the authoritative phase timeout.
    pub(crate) dispatch_deadline: Option<std::time::Instant>,
    /// Exact SA/DAG step and dispatch identities used only to bind runtime
    /// checkpoints to the Agent instance which owns their transcript.
    pub(crate) checkpoint_step_id: Option<String>,
    pub(crate) checkpoint_dispatch_id: Option<String>,
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
            parent_interaction_id: None,
            input_data: HashMap::new(),
            constraints: HashMap::new(),
            max_iterations,
            prev_agent_summary: None,
            plan_handoff: None,
            execution_handoff: None,
            verified_check_handoff: None,
            historical_experience: Vec::new(),
            correction_handoff: None,
            prior_biz_agent_result: None,
            biz_agent_recovery_invalidated_packages: Arc::new(HashSet::new()),
            biz_agent_child_evidence_contract: None,
            original_task: None,
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            five_w2h_iri: String::new(),
            five_w2h_snapshot: None,
            conversation_history: None,
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
            workspace_resource_lease: None,
            dispatch_deadline: None,
            checkpoint_step_id: None,
            checkpoint_dispatch_id: None,
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
        self.prev_agent_summary =
            Some(crate::tools::tool_executor::sanitize_session_tool_references(summary).0);
        self
    }

    pub(crate) fn with_plan_handoff(
        mut self,
        content: impl Into<String>,
        source_ref: impl Into<String>,
    ) -> Self {
        self.plan_handoff = Some(AgentContextHandoff::new(content, source_ref, "PA"));
        self
    }

    pub(crate) fn with_execution_handoff(
        mut self,
        content: impl Into<String>,
        source_ref: impl Into<String>,
    ) -> Self {
        self.execution_handoff = Some(AgentContextHandoff::new(content, source_ref, "DA"));
        self
    }

    pub(crate) fn with_verified_check_handoff(
        mut self,
        content: impl Into<String>,
        source_ref: impl Into<String>,
    ) -> Self {
        self.verified_check_handoff = Some(AgentContextHandoff::new(content, source_ref, "CA"));
        self
    }

    pub(crate) fn with_historical_experience(mut self, items: Vec<String>) -> Self {
        self.historical_experience = items
            .into_iter()
            .map(|item| crate::tools::tool_executor::sanitize_session_tool_references(&item).0)
            .collect();
        self
    }

    pub(crate) fn with_correction_handoff(
        mut self,
        content: impl Into<String>,
        source_ref: impl Into<String>,
        producer: impl Into<String>,
    ) -> Self {
        self.correction_handoff = Some(AgentContextHandoff::new(content, source_ref, producer));
        self
    }

    /// Attach a prior deterministic BizAgent result to one explicit recovery
    /// dispatch.  `Arc` keeps TaskContext cloning bounded; consumers still
    /// have to authenticate every reusable child receipt against the current
    /// canonical work-package contract.
    pub(crate) fn with_prior_biz_agent_result(
        mut self,
        result: TaskResult,
        invalidated_packages: HashSet<String>,
    ) -> Self {
        self.prior_biz_agent_result = Some(Arc::new(result));
        self.biz_agent_recovery_invalidated_packages = Arc::new(invalidated_packages);
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

    /// Read-only execution metadata for embedders and diagnostics. Mutation
    /// goes through builders so future validation cannot be bypassed by a
    /// public map field.
    pub fn constraints(&self) -> &HashMap<String, String> {
        &self.constraints
    }

    pub fn constraint(&self, key: &str) -> Option<&str> {
        self.constraints.get(key).map(String::as_str)
    }

    pub fn with_effect_policy(mut self, policy: crate::core::effect::EffectPolicy) -> Self {
        self.effect_policy = policy;
        self
    }

    pub(crate) fn correlate_llm_scope(
        &self,
        scope: crate::llm::LlmInteractionScope,
    ) -> crate::llm::LlmInteractionScope {
        let scope = scope.with_usage_scope(
            self.parent_task_iri
                .as_deref()
                .unwrap_or(self.task_iri.as_str()),
        );
        match &self.parent_interaction_id {
            Some(parent) => scope.with_parent(parent.clone()),
            None => scope,
        }
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

    /// Restore a task from the canonical checkpoint reader.  The message
    /// history remains available to the model while the structured state is
    /// forwarded to SA for phase and counter restoration.
    pub fn with_resumed_checkpoint(
        mut self,
        messages: Vec<ChatMessage>,
        state: crate::core::checkpoint::TaskResumeState,
    ) -> Self {
        // A restored task normally creates a fresh isolated Agent. Task-wide
        // counters belong to SA's accumulator and must not seed this new
        // AgentRunner (doing so caused the parent to add them a second time).
        self.resumed_turn_count = 0;
        self.resumed_tool_count = 0;
        self.resumed_messages = Some(
            messages
                .into_iter()
                .map(sanitize_cross_agent_message)
                .collect(),
        );
        self.resumed_state = Some(state);
        self
    }

    /// Attach ordinary multi-turn conversation context without claiming a
    /// durable checkpoint boundary.
    pub fn with_conversation_history(mut self, messages: Vec<ChatMessage>) -> Self {
        self.conversation_history = Some(
            messages
                .into_iter()
                .map(sanitize_cross_agent_message)
                .collect(),
        );
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

    pub fn with_workspace_resource_lease(
        mut self,
        lease: crate::core::effect::WorkspaceResourceLease,
    ) -> Self {
        self.workspace_resource_lease = Some(lease);
        self
    }

    /// Build the tool security boundary from kernel-owned execution state.
    /// Stable AgentTurn grants come only from the current L1 session and typed
    /// SA handoffs; objective/input/tool JSON text is deliberately ignored.
    pub(crate) fn tool_security_context(
        &self,
        agent_id: &str,
        agent_role: &str,
        l1_session_id: &str,
    ) -> crate::skill_graph::security::SecurityContext {
        let handoff_source_refs = [
            self.plan_handoff.as_ref(),
            self.execution_handoff.as_ref(),
            self.verified_check_handoff.as_ref(),
            self.correction_handoff.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(|handoff| handoff.source_ref.clone());

        crate::skill_graph::security::SecurityContext::new(agent_id, agent_role)
            .with_task(&self.task_iri)
            .with_agent_turn_read_scope(
                agent_turn_iri_prefix(&self.task_iri, l1_session_id),
                handoff_source_refs,
            )
    }
}

impl Default for TaskContext {
    fn default() -> Self {
        Self {
            task_iri: String::new(),
            objective: String::new(),
            parent_task_iri: None,
            parent_interaction_id: None,
            input_data: HashMap::new(),
            constraints: HashMap::new(),
            max_iterations: 20,
            prev_agent_summary: None,
            plan_handoff: None,
            execution_handoff: None,
            verified_check_handoff: None,
            historical_experience: Vec::new(),
            correction_handoff: None,
            prior_biz_agent_result: None,
            biz_agent_recovery_invalidated_packages: Arc::new(HashSet::new()),
            biz_agent_child_evidence_contract: None,
            original_task: None,
            completed_steps: Vec::new(),
            pending_steps: Vec::new(),
            five_w2h_iri: String::new(),
            five_w2h_snapshot: None,
            conversation_history: None,
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
            workspace_resource_lease: None,
            dispatch_deadline: None,
            checkpoint_step_id: None,
            checkpoint_dispatch_id: None,
            effect_policy: crate::core::effect::EffectPolicy::None,
        }
    }
}

/// Structured task outcome verdict — decoupled from the human-readable status string.
/// The `finish` action historically flattened a verdict into `status: "success"`,
/// losing blocked/failed intent; this enum preserves it so consumers (e.g. SA
/// verify-first logic) can react honestly instead of re-parsing summary text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

impl TaskResult {
    /// Detached result safe for an explicit cross-Agent boundary. Kernel
    /// status/accounting and stable AgentTurn archive IRIs are preserved;
    /// model-originated payloads are recursively stripped of ephemeral result
    /// tools belonging to the producing L1 session.
    pub(crate) fn sanitized_for_agent_boundary(&self) -> Self {
        let sanitize_value =
            |value: &Value| crate::tools::tool_executor::sanitized_session_handoff_value(value).0;
        let mut sanitized = self.clone();
        sanitized.summary =
            crate::tools::tool_executor::sanitize_session_tool_references(&self.summary).0;
        sanitized.output = self.output.as_ref().map(sanitize_value);
        sanitized.jsonld_output = self.jsonld_output.as_ref().map(sanitize_value);
        sanitized.artifacts = self.artifacts.iter().map(sanitize_value).collect();
        sanitized.errors = self
            .errors
            .iter()
            .map(|error| crate::tools::tool_executor::sanitize_session_tool_references(error).0)
            .collect();
        sanitized
    }
}

/// RAII cleanup for transcript-local compression state. AgentRunner can be
/// shared by sequential role instances, while child runners may execute in
/// parallel; removing only the completed L1 session avoids both stale-memory
/// growth and destructive global clears.
pub(super) struct ToolResultSessionGuard {
    compressor: Option<Arc<std::sync::Mutex<ToolResultCompressor>>>,
    tool_executor: Arc<parking_lot::RwLock<ToolExecutor>>,
    unified_graph_store: Option<Arc<oxigraph::store::Store>>,
    session_id: String,
}

impl ToolResultSessionGuard {
    pub(super) fn new(
        compressor: Option<Arc<std::sync::Mutex<ToolResultCompressor>>>,
        tool_executor: Arc<parking_lot::RwLock<ToolExecutor>>,
        unified_graph_store: Option<Arc<oxigraph::store::Store>>,
        session_id: &str,
    ) -> Self {
        Self {
            compressor,
            tool_executor,
            unified_graph_store,
            session_id: session_id.to_string(),
        }
    }
}

impl Drop for ToolResultSessionGuard {
    fn drop(&mut self) {
        if let Some(compressor) = self.compressor.as_ref() {
            if let Ok(mut compressor) = compressor.lock() {
                compressor.remove_session(&self.session_id);
            }
        }
        self.tool_executor
            .write()
            .remove_micro_tools_for_session(&self.session_id);
        if let Some(store) = self.unified_graph_store.as_ref() {
            let session_scope = crate::tools::result_router::ResultRoutingIdentity::new(
                &self.session_id,
                "cleanup",
            )
            .session_scope;
            let graph_prefix = format!("graph:tool-result:{session_scope}_c");
            let graph_names = store
                .named_graphs()
                .filter_map(Result::ok)
                .map(|graph| graph.to_string())
                .map(|graph| {
                    graph
                        .trim_start_matches('<')
                        .trim_end_matches('>')
                        .to_string()
                })
                .filter(|graph| graph.starts_with(&graph_prefix))
                .collect::<Vec<_>>();
            for graph in graph_names {
                if let Err(error) = store.update(&format!("DROP SILENT GRAPH <{graph}>")) {
                    warn!(%graph, %error, "Failed to retire session-scoped result graph");
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmParsedResponse {
    pub thought: Option<String>,
    pub content: String,
    /// True only when a null/empty response `content` was backfilled from
    /// provider-native reasoning. This may support continuity, but must never
    /// be promoted to CA audit evidence.
    pub content_from_reasoning: bool,
    pub summary: Option<String>,
    pub action: Option<String>,
    pub is_valid_json: bool,
    pub has_native_reasoning: bool,
    pub emphasis: Vec<String>,
}

#[derive(Clone)]
pub struct AgentRunner {
    pub(crate) gateway: Arc<UnifiedGateway>,
    /// The only business-level entry point for model calls. `gateway` remains
    /// exposed for provider configuration and health checks, while requests
    /// must flow through this lifecycle facade.
    pub llm_interactions: Arc<LlmInteractionService>,
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
        // Hook policy is runner-local. A gateway-scoped interaction-service
        // registry would make the most recently created AgentRunner overwrite
        // earlier runners' HookManager. Children clone this Arc from their
        // parent, so one business execution still has one common interaction
        // plane without cross-runner policy leakage.
        let llm_interactions = Arc::new(LlmInteractionService::with_event_capacity(
            gateway.clone(),
            agent_settings.event_bus_capacity,
        ));
        let (
            total_prompt_tokens,
            total_completion_tokens,
            last_prompt_tokens,
            last_completion_tokens,
        ) = llm_interactions.token_usage_arcs();
        let projection = Arc::new(ProjectionEngine::new(
            blackboard.clone(),
            agent_settings.max_projection_size,
        ));
        let sharing = Arc::new(SharingProtocol::new());
        let hook_manager = Arc::new(HookManager::new());
        if agent_settings.hooks.logging {
            hook_manager.register(LoggingHook::new());
        }
        if agent_settings.hooks.timing {
            hook_manager.register(TimingHook::new());
        }
        if agent_settings.hooks.metrics {
            hook_manager.register(MetricsHook::new());
        }
        let rate_limit = &agent_settings.hooks.llm_rate_limit;
        if rate_limit.enabled {
            if rate_limit.max_calls > 0 && rate_limit.window_seconds > 0 {
                hook_manager.register(RateLimitHook::new(
                    rate_limit.max_calls,
                    rate_limit.window_seconds,
                ));
            } else {
                warn!(
                    max_calls = rate_limit.max_calls,
                    window_seconds = rate_limit.window_seconds,
                    "invalid enabled LLM rate-limit Hook ignored; validate Settings before constructing AgentRunner"
                );
            }
        }
        ToolGuard::new().register_hooks(&hook_manager);
        llm_interactions.set_hook_manager(hook_manager.clone());

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

        let mut tool_executor = ToolExecutor::new();
        tool_executor.set_projection_engine(projection.clone());
        tool_executor.set_shared_skill_creator_interactions(llm_interactions.clone());
        // Retain the gateway fallback for standalone/direct handler consumers,
        // while canonical AgentRunner execution always prefers the exact
        // runner-local interaction service injected above.
        tool_executor.set_shared_skill_creator_gateway(gateway.clone());
        let external_hooks = &agent_settings.hooks.external_tools;
        if external_hooks.enabled {
            if external_hooks.timeout_ms > 0 {
                let runtime_config = RuntimeHookConfig::new(
                    external_hooks.pre_tool_use.clone(),
                    external_hooks.post_tool_use.clone(),
                    external_hooks.post_tool_use_failure.clone(),
                );
                let hook_runner = HookRunner::new(runtime_config).with_command_timeout(
                    std::time::Duration::from_millis(external_hooks.timeout_ms),
                );
                tool_executor.set_hook_runner_with_input_rewrite(
                    hook_runner,
                    external_hooks.allow_input_rewrite,
                );
            } else {
                warn!(
                    "invalid enabled external Tool Hook timeout ignored; validate Settings before constructing AgentRunner"
                );
            }
        }

        let mut runner = Self {
            gateway,
            llm_interactions,
            skills,
            blackboard,
            l0_store: l0_store.clone(),
            memory_manager,
            templates,
            tool_executor: Arc::new(parking_lot::RwLock::new(tool_executor)),
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
            total_prompt_tokens,
            total_completion_tokens,
            last_prompt_tokens,
            last_completion_tokens,
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

    /// Fork the immutable/shared runtime services for one concurrently
    /// executing child while rebuilding state that belongs to a single ReAct
    /// transcript. Kernel policy, hooks, tool registry, durable memory and
    /// centralized usage counters remain shared; compression queues,
    /// relevance history and consuming input stores do not. The parent
    /// BizAgent copies the relevant dependency/input evidence into the child's
    /// typed TaskContext before execution.
    pub(crate) fn fork_for_child_execution(&self) -> Self {
        let mut child = self.clone();
        child.init_context_compressors();
        child.relevance_tracker = self.relevance_tracker.as_ref().map(|tracker| {
            let time_decay_lambda = tracker
                .lock()
                .map(|tracker| tracker.time_decay_lambda())
                .unwrap_or_default();
            Arc::new(std::sync::Mutex::new(RelevanceTracker::with_time_decay(
                0.6,
                time_decay_lambda,
            )))
        });
        child.perception_store = crate::core::perception_store::PerceptionStore::new();
        child.supplement_store = crate::core::supplementary_store::SupplementaryInputStore::new();
        child
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
                    let required_ca_verifier = matches!(role, "CA" | "Check")
                        && execution::is_ca_verification_tool_name(name);
                    if (session_micro_tools.contains(name) || required_ca_verifier)
                        && names.insert(name.to_string())
                    {
                        visible.push(definition);
                    }
                }
                // `max_micro_tools` bounds the process-wide catalog, not the
                // validity of references already present in this BizAgent's
                // current conversation. Rebuild evicted schemas only for the
                // owning session; never expose another agent's archived tools.
                for name in session_micro_tools {
                    if ToolExecutor::is_micro_tool_name(name) && names.insert(name.clone()) {
                        if let Some(definition) = executor.micro_tool_definition_for_history(name) {
                            visible.push(definition);
                        }
                    }
                }
                visible
            }
        };
        let executor = self.tool_executor.read();
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
                    .map(|allowed| executor.allowlist_permits(name, allowed))
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
        let executor = self.tool_executor.read();
        definitions
            .into_iter()
            .filter(|definition| {
                let Some(name) = definition["function"]["name"].as_str() else {
                    return false;
                };
                let allowlisted = ctx
                    .allowed_tools
                    .as_deref()
                    .map(|allowed| executor.allowlist_permits(name, allowed))
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
        definitions = self.apply_canonical_child_tool_scope(definitions, role, ctx);
        self.apply_task_tool_scope(definitions, ctx)
    }

    /// Turn a canonical child evidence contract into its least-authority tool
    /// window. The SA step allowlist is an upper bound shared by all sibling
    /// packages; copying it verbatim into each child allowed an artifact
    /// writer to rediscover unrelated tools and made a pure final verifier
    /// spend turns searching for the command runner. Exact package semantics
    /// are stronger and can safely narrow that ceiling.
    fn apply_canonical_child_tool_scope(
        &self,
        mut definitions: Vec<Value>,
        role: &str,
        ctx: &TaskContext,
    ) -> Vec<Value> {
        if !matches!(role, "DA" | "Do") {
            return definitions;
        }
        let Some(package) = ctx.biz_agent_child_evidence_contract.as_deref() else {
            return definitions;
        };
        let has_artifact = package.evidence_requirements.iter().any(|requirement| {
            matches!(
                requirement,
                crate::core::sa::WorkPackageEvidenceRequirement::ArtifactDelivery { .. }
            )
        });
        let has_unscoped_mutation = package.evidence_requirements.iter().any(|requirement| {
            matches!(
                requirement,
                crate::core::sa::WorkPackageEvidenceRequirement::WorkspaceMutation { .. }
            )
        });
        let has_verification = package.evidence_requirements.iter().any(|requirement| {
            matches!(
                requirement,
                crate::core::sa::WorkPackageEvidenceRequirement::Verification { .. }
            )
        });
        if !has_artifact && !has_verification || has_unscoped_mutation {
            return definitions;
        }

        // These schemas are executable contract primitives rather than
        // optional discovery results. Add them only when both the parent
        // allowlist and role catalog already authorize them.
        let required = [
            has_artifact.then_some("file_write"),
            has_verification.then_some("bash"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let mut names = definitions
            .iter()
            .filter_map(|definition| definition["function"]["name"].as_str())
            .map(str::to_string)
            .collect::<HashSet<_>>();
        if required.iter().any(|name| !names.contains(*name)) {
            let catalog = self.tool_executor.read().tool_definitions_for_role(role);
            for required_name in &required {
                if names.contains(*required_name)
                    || ctx.allowed_tools.as_deref().is_some_and(|allowed| {
                        !ToolExecutor::explicit_allowlist_permits(required_name, allowed)
                    })
                {
                    continue;
                }
                if let Some(definition) = catalog.iter().find(|definition| {
                    definition["function"]["name"].as_str() == Some(*required_name)
                }) {
                    definitions.push(definition.clone());
                    names.insert((*required_name).to_string());
                }
            }
        }

        definitions.retain(|definition| {
            let Some(name) = definition["function"]["name"].as_str() else {
                return false;
            };
            ToolExecutor::is_micro_tool_name(name)
                || (has_artifact
                    && matches!(name, "file_write" | "file_read" | "read_agent_output"))
                || (has_verification && matches!(name, "bash" | "powershell" | "code_execute"))
        });
        definitions
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
        let definitions = self.apply_canonical_child_tool_scope(
            self.tool_executor.read().tool_definitions_for_role(role),
            role,
            ctx,
        );
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
        let hook_manager = Arc::new(hook_manager);
        self.llm_interactions.set_hook_manager(hook_manager.clone());
        self.hook_manager = hook_manager;
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
        self.llm_interactions.attach_event_bus(event_bus.clone());
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
mod conformance_contract_tests {
    use super::*;

    fn verified_contract() -> ConformanceContract {
        ConformanceContract {
            schema_version: CONFORMANCE_CONTRACT_SCHEMA_VERSION.to_string(),
            source_plan_id: "plan".to_string(),
            relations: vec![NormativeDesignRelation {
                do_step_id: "do".to_string(),
                design_predecessor_id: "design".to_string(),
                transitive_successor_ids: vec!["docs".to_string(), "implementation".to_string()],
                evidence: ConformanceRelationEvidence::Verified {
                    order_receipt_sha256: format!("sha256:{}", "a".repeat(64)),
                    design_paths: vec!["calculator/DESIGN.md".to_string()],
                    successor_deliveries: vec![
                        WorkPackageDeliveryEvidence::ArtifactDelivery {
                            work_package_id: "docs".to_string(),
                            paths: vec!["calculator/README.md".to_string()],
                        },
                        WorkPackageDeliveryEvidence::ArtifactDelivery {
                            work_package_id: "implementation".to_string(),
                            paths: vec!["calculator/calculator.py".to_string()],
                        },
                    ],
                },
            }],
        }
    }

    #[test]
    fn conformance_contract_round_trip_is_strict_and_pair_preserving() {
        let contract = verified_contract();
        let encoded = contract.to_constraint_value().unwrap();
        let decoded = ConformanceContract::from_constraint_value(&encoded).unwrap();
        assert_eq!(decoded, contract);

        let relations = decoded.verified_relation_path_sets().unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].design_predecessor_id, "design");
        assert_eq!(
            relations[0]
                .successor_deliveries
                .get("implementation")
                .unwrap(),
            &VerifiedConformanceSuccessorEvidence::ArtifactPaths(std::collections::BTreeSet::from(
                ["calculator/calculator.py".to_string()]
            ))
        );
    }

    #[test]
    fn conformance_contract_accepts_a_receipted_pure_verification_successor() {
        let mut contract = verified_contract();
        let ConformanceRelationEvidence::Verified {
            successor_deliveries,
            ..
        } = &mut contract.relations[0].evidence
        else {
            unreachable!()
        };
        successor_deliveries[0] = WorkPackageDeliveryEvidence::VerificationExecution {
            work_package_id: "docs".to_string(),
            verification_receipt_sha256s: vec![format!("sha256:{}", "b".repeat(64))],
        };

        contract.validate().unwrap();
        let relation = contract.verified_relation_path_sets().unwrap().remove(0);
        assert_eq!(
            relation.successor_deliveries.get("docs"),
            Some(&VerifiedConformanceSuccessorEvidence::VerificationReceipts(
                std::collections::BTreeSet::from([format!("sha256:{}", "b".repeat(64))])
            ))
        );
    }

    #[test]
    fn conformance_contract_rejects_unknown_fields_and_unsafe_or_unsorted_paths() {
        let contract = verified_contract();
        let mut value = serde_json::to_value(&contract).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(ConformanceContract::from_constraint_value(&value.to_string()).is_err());

        let mut unsafe_contract = contract.clone();
        let ConformanceRelationEvidence::Verified { design_paths, .. } =
            &mut unsafe_contract.relations[0].evidence
        else {
            unreachable!()
        };
        *design_paths = vec!["../DESIGN.md".to_string()];
        assert!(unsafe_contract.validate().is_err());

        let mut unsorted_contract = contract;
        let ConformanceRelationEvidence::Verified {
            successor_deliveries,
            ..
        } = &mut unsorted_contract.relations[0].evidence
        else {
            unreachable!()
        };
        successor_deliveries.swap(0, 1);
        assert!(unsorted_contract.validate().is_err());
    }

    #[test]
    fn conformance_contract_enforces_relation_path_and_serialized_size_limits() {
        let mut too_many_relations = verified_contract();
        too_many_relations.relations = (0..=MAX_CONFORMANCE_RELATIONS)
            .map(|index| NormativeDesignRelation {
                do_step_id: format!("do_{index:02}"),
                design_predecessor_id: format!("design_{index:02}"),
                transitive_successor_ids: vec![format!("impl_{index:02}")],
                evidence: ConformanceRelationEvidence::Planned,
            })
            .collect();
        assert!(too_many_relations
            .validate()
            .is_err_and(|reason| reason.contains("too many design relations")));

        let mut too_many_paths = verified_contract();
        let ConformanceRelationEvidence::Verified { design_paths, .. } =
            &mut too_many_paths.relations[0].evidence
        else {
            unreachable!()
        };
        *design_paths = (0..=MAX_CONFORMANCE_PATHS_PER_PACKAGE)
            .map(|index| format!("calculator/design/{index:03}.md"))
            .collect();
        assert!(too_many_paths
            .validate()
            .is_err_and(|reason| reason.contains("invalid receipt digest or design paths")));

        let mut oversized = verified_contract();
        oversized.source_plan_id = "p".repeat(257);
        assert!(oversized
            .validate()
            .is_err_and(|reason| reason.contains("invalid source plan id")));
    }
}

#[cfg(test)]
mod tests;
