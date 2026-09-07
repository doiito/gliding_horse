use sha2::{Digest, Sha256};
use tracing::{info, instrument, warn};

use crate::core::agent_instance::AgentRole;
use crate::core::agent_runner::{TaskContext, TaskResult, TaskVerdict};
use crate::core::context_model::{AgentSpecSourceKind, AgentSpecSourceRecord};
use crate::core::policy_learning::{
    learning_families_compatible, learning_task_context, LearningTaskContext,
};
use crate::CoreError;

use super::agent::SupervisorAgent;
use super::types::*;

#[derive(Debug, Clone, serde::Serialize)]
struct PolicyRewardBreakdown {
    status_reward: f32,
    failed_action_penalty: f32,
    excess_turn_penalty: f32,
    recovery_penalty: f32,
    error_penalty: f32,
    prompt_token_penalty: f32,
    latency_penalty: f32,
    late_substantive_action_penalty: f32,
    redundant_read_ratio: f32,
    redundant_read_penalty: f32,
    no_effect_tail: usize,
    no_effect_tail_penalty: f32,
    first_substantive_action_ordinal: Option<usize>,
    total: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct LearningTreatmentMetrics {
    perception_hints_observed: usize,
    experience_hint_fingerprints: Vec<String>,
    skills_observed: usize,
    skill_iris_observed: Vec<String>,
    skill_iris_injected: Vec<String>,
    knowledge_fragments_observed: usize,
    knowledge_fragment_iris_observed: Vec<String>,
    knowledge_fragment_iris_injected: Vec<String>,
    hints_injected: usize,
    hint_chars_injected: usize,
    task_family_raw_features: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experiment_pair_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experiment_seed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experiment_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experiment_config_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_fingerprint: Option<String>,
    objective_fingerprint: String,
    orchestration_mode: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LearningHintSource {
    Experience,
    Skill { iri: String },
    Knowledge { iri: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LearningHintCandidate {
    text: String,
    source: LearningHintSource,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct MaterializedLearningTreatment {
    hints: Vec<String>,
    skill_iris: Vec<String>,
    knowledge_fragment_iris: Vec<String>,
}

/// Deduplicate hints preserving first-seen order, then truncate to `cap`.
#[cfg(test)]
fn dedup_hints(
    hints: Vec<String>,
    cap: usize,
    max_hint_chars: usize,
    max_total_chars: usize,
) -> Vec<String> {
    let mut seen = std::collections::HashSet::with_capacity(cap.min(hints.len()));
    let mut out = Vec::with_capacity(cap.min(hints.len()));
    let mut total_chars = 0usize;
    for h in hints {
        if out.len() >= cap || total_chars >= max_total_chars {
            break;
        }
        let mut bounded = h.chars().take(max_hint_chars).collect::<String>();
        if h.chars().count() > max_hint_chars {
            bounded.push_str("…");
        }
        let remaining = max_total_chars.saturating_sub(total_chars);
        bounded = bounded.chars().take(remaining).collect();
        if !bounded.is_empty() && seen.insert(bounded.clone()) {
            total_chars += bounded.chars().count();
            out.push(bounded);
        }
    }
    out
}

fn rank_knowledge_fragments(
    fragments: Vec<crate::skill_graph::types::KnowledgeFragment>,
    task_context: &LearningTaskContext,
    max_fragments: usize,
) -> Vec<crate::skill_graph::types::KnowledgeFragment> {
    let current_features = task_context
        .raw_features
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let applicability = |fragment: &crate::skill_graph::types::KnowledgeFragment| {
        let Some(stored_family) = fragment.task_family.as_deref() else {
            return true;
        };
        if stored_family == task_context.family {
            return true;
        }
        if !learning_families_compatible(stored_family, &task_context.family) {
            return false;
        }
        // A coarse policy family is safe for choosing an evidence ordering,
        // but task-specific procedures need lexical applicability as a second
        // gate. Knowledge descriptions retain the original audited objective.
        let historical = learning_task_context(&fragment.description);
        let overlap = historical
            .raw_features
            .iter()
            .filter(|feature| current_features.contains(feature.as_str()))
            .count();
        let denominator = historical
            .raw_features
            .len()
            .min(task_context.raw_features.len())
            .max(1);
        overlap >= 2 && overlap * 5 >= denominator
    };
    let mut fragments = fragments
        .into_iter()
        // A structured fragment is safe only inside its declared family.
        // Legacy failure fragments have no family and remain available at a
        // lower rank for backward compatibility.
        .filter(applicability)
        .collect::<Vec<_>>();
    fragments.sort_by(|left, right| {
        let score = |fragment: &crate::skill_graph::types::KnowledgeFragment| {
            let exact_family =
                u32::from(fragment.task_family.as_deref() == Some(task_context.family.as_str()));
            let passed = u32::from(fragment.ca_verdict.as_deref() == Some("pass"));
            let evidence = fragment.evidence_count.min(100);
            (exact_family, passed, evidence, fragment.last_verified_at)
        };
        score(right).cmp(&score(left))
    });
    let mut seen = std::collections::HashSet::new();
    fragments.retain(|fragment| seen.insert(fragment.fragment_iri.clone()));
    fragments.truncate(max_fragments);
    fragments
}

fn render_knowledge_hint(fragment: &crate::skill_graph::types::KnowledgeFragment) -> String {
    if fragment.kind != "ca_validated_task_knowledge" {
        return format!(
            "[Knowledge source={} kind=legacy_failure] problem={} mitigation={}",
            fragment.fragment_iri, fragment.problem, fragment.recommendation
        );
    }
    format!(
        "[Knowledge source={} task={} family={} ca={} evidence={}/{}] applicability={}; procedure={}; checks={}; boundary={}",
        fragment.fragment_iri,
        fragment.source_task_iri.as_deref().unwrap_or("unknown"),
        fragment.task_family.as_deref().unwrap_or("unknown"),
        fragment.ca_verdict.as_deref().unwrap_or("unknown"),
        fragment.success_count,
        fragment.evidence_count,
        fragment.problem,
        if fragment.procedure.is_empty() { "not recorded".to_string() } else { fragment.procedure.join(" | ") },
        if fragment.successful_checks.is_empty() { "not recorded".to_string() } else { fragment.successful_checks.join(" | ") },
        if fragment.counterexamples.is_empty() { "none recorded".to_string() } else { fragment.counterexamples.join(" | ") },
    )
}

fn eligible_policy_candidates(
    skill_count: usize,
    knowledge_count: usize,
    experience_count: usize,
) -> Vec<String> {
    crate::core::retrieval_policy::RetrievalPolicyArm::candidate_names(
        skill_count,
        knowledge_count,
        experience_count,
    )
}

/// Materialize the treatment selected by the constrained policy. `baseline`
/// is a real ablation (no durable history), while each learned arm receives
/// the same bounded evidence in a different source-priority order.
#[cfg(test)]
fn policy_treatment_hints(
    action: &str,
    experience: &[String],
    skills: &[String],
    knowledge: &[String],
    max_hints: usize,
    max_hint_chars: usize,
    max_total_chars: usize,
) -> Vec<String> {
    let candidates = |hints: &[String], source: LearningHintSource| {
        hints
            .iter()
            .map(|text| LearningHintCandidate {
                text: text.clone(),
                source: source.clone(),
            })
            .collect::<Vec<_>>()
    };
    materialize_policy_treatment(
        action,
        &candidates(experience, LearningHintSource::Experience),
        &candidates(skills, LearningHintSource::Experience),
        &candidates(knowledge, LearningHintSource::Experience),
        max_hints,
        max_hint_chars,
        max_total_chars,
    )
    .hints
}

/// Apply policy ordering and prompt limits while retaining the provenance of
/// each hint that actually survives deduplication and truncation. Keeping the
/// source beside the string avoids reconstructing adoption from rendered
/// content, which can collide or change as templates evolve.
fn materialize_policy_treatment(
    action: &str,
    experience: &[LearningHintCandidate],
    skills: &[LearningHintCandidate],
    knowledge: &[LearningHintCandidate],
    max_hints: usize,
    max_hint_chars: usize,
    max_total_chars: usize,
) -> MaterializedLearningTreatment {
    use crate::core::retrieval_policy::RetrievalPolicyArm;

    let arm = RetrievalPolicyArm::parse(action).unwrap_or(RetrievalPolicyArm::Baseline);
    let groups: [&[LearningHintCandidate]; 3] = match arm {
        RetrievalPolicyArm::Baseline => [&[], &[], &[]],
        RetrievalPolicyArm::ExperienceFirst => [experience, knowledge, skills],
        RetrievalPolicyArm::KnowledgeFirst => [knowledge, experience, skills],
        RetrievalPolicyArm::SkillFirst => [skills, experience, knowledge],
    };
    let mut materialized = MaterializedLearningTreatment::default();
    let mut seen_hints = std::collections::HashSet::new();
    let mut seen_skills = std::collections::HashSet::new();
    let mut seen_knowledge = std::collections::HashSet::new();
    let mut total_chars = 0usize;
    for candidate in groups.into_iter().flatten() {
        if materialized.hints.len() >= max_hints || total_chars >= max_total_chars {
            break;
        }
        let mut bounded = candidate
            .text
            .chars()
            .take(max_hint_chars)
            .collect::<String>();
        if candidate.text.chars().count() > max_hint_chars {
            bounded.push('…');
        }
        let remaining = max_total_chars.saturating_sub(total_chars);
        bounded = bounded.chars().take(remaining).collect();
        if bounded.is_empty() || !seen_hints.insert(bounded.clone()) {
            continue;
        }
        total_chars = total_chars.saturating_add(bounded.chars().count());
        materialized.hints.push(bounded);
        match &candidate.source {
            LearningHintSource::Experience => {}
            LearningHintSource::Skill { iri } if seen_skills.insert(iri.clone()) => {
                materialized.skill_iris.push(iri.clone());
            }
            LearningHintSource::Knowledge { iri } if seen_knowledge.insert(iri.clone()) => {
                materialized.knowledge_fragment_iris.push(iri.clone());
            }
            LearningHintSource::Skill { .. } | LearningHintSource::Knowledge { .. } => {}
        }
    }
    materialized
}

/// Materialize verify-first from the CA/AA definitions in the exact LLM plan.
/// The kernel changes only execution topology. It must never substitute a
/// generic CA/AA business profile for the model-authored role definitions.
fn materialize_verify_first_from_llm_plan(
    plan: &mut ExecutionPlan,
    task_iri: &str,
) -> Result<(), CoreError> {
    let mut verify_ca = plan
        .steps
        .iter()
        .find(|step| step.role == AgentRole::Check)
        .cloned()
        .ok_or_else(|| CoreError::InteractionRejected {
            stage: "sa_verify_first_plan".to_string(),
            reason: "verify-first requires an LLM-authored CA definition".to_string(),
        })?;
    let mut verify_aa = plan
        .steps
        .iter()
        .find(|step| step.role == AgentRole::Act)
        .cloned()
        .ok_or_else(|| CoreError::InteractionRejected {
            stage: "sa_verify_first_plan".to_string(),
            reason: "verify-first requires an LLM-authored AA definition".to_string(),
        })?;

    let source_for = |step: &PlanStep| -> Result<AgentSpecSourceRecord, CoreError> {
        let source = plan
            .agent_spec_source_for_step(&step.step_id)
            .map_err(|error| CoreError::InteractionRejected {
                stage: "sa_verify_first_plan".to_string(),
                reason: format!(
                    "verify-first cannot resolve provenance for LLM step '{}': {error}",
                    step.step_id
                ),
            })?
            .ok_or_else(|| CoreError::InteractionRejected {
                stage: "sa_verify_first_plan".to_string(),
                reason: format!(
                    "verify-first step '{}' has no model interaction provenance",
                    step.step_id
                ),
            })?;
        if source.kind != AgentSpecSourceKind::LlmGeneratedPlan {
            return Err(CoreError::InteractionRejected {
                stage: "sa_verify_first_plan".to_string(),
                reason: format!(
                    "verify-first refuses non-LLM role definition for step '{}': {:?}",
                    step.step_id, source.kind
                ),
            });
        }
        Ok(source)
    };
    let mut ca_source = source_for(&verify_ca)?;
    let mut aa_source = source_for(&verify_aa)?;
    let original_ca_id = verify_ca.step_id.clone();
    let original_aa_id = verify_aa.step_id.clone();

    plan.fallback_steps = plan.steps.clone();
    plan.verify_first = true;
    verify_ca.step_id = "verify_first_ca".to_string();
    verify_ca.dependencies.clear();
    verify_aa.step_id = "verify_first_aa".to_string();
    verify_aa.dependencies = vec![verify_ca.step_id.clone()];

    let specialize_source =
        |source: &mut AgentSpecSourceRecord, original_step_id: &str, verify_step_id: &str| {
            let source_ref = source
                .source_ref
                .take()
                .unwrap_or_else(|| format!("{task_iri}#llm-plan"));
            source.source_ref = Some(format!(
                "{source_ref}/verify-first-clone/{original_step_id}/as/{verify_step_id}"
            ));
        };
    specialize_source(&mut ca_source, &original_ca_id, &verify_ca.step_id);
    specialize_source(&mut aa_source, &original_aa_id, &verify_aa.step_id);

    let original_description = plan.description.clone();
    plan.steps = vec![verify_ca, verify_aa];
    plan.agent_sequence = vec![AgentRole::Check, AgentRole::Act];
    plan.parallel_groups.clear();
    plan.description =
        format!("[Verify-first using LLM-authored CA/AA] Fallback plan: {original_description}");
    plan.set_agent_spec_step_source("verify_first_ca", ca_source)
        .map_err(|error| CoreError::InteractionRejected {
            stage: "sa_verify_first_plan".to_string(),
            reason: format!("failed to bind verify-first CA provenance: {error}"),
        })?;
    plan.set_agent_spec_step_source("verify_first_aa", aa_source)
        .map_err(|error| CoreError::InteractionRejected {
            stage: "sa_verify_first_plan".to_string(),
            reason: format!("failed to bind verify-first AA provenance: {error}"),
        })?;
    Ok(())
}

/// Build a PDCA delta plan beginning at the failed step. Completed upstream
/// nodes are retained as evidence in cycle feedback instead of being executed
/// again. External DAG workflows keep their own retry/branch topology and are
/// therefore never rewritten here.
fn recovery_descendant_step_ids(
    plan: &ExecutionPlan,
    failed_step_id: &str,
) -> Option<std::collections::HashSet<String>> {
    plan.steps
        .iter()
        .any(|step| step.step_id == failed_step_id)
        .then_some(())?;
    let mut selected = std::collections::HashSet::from([failed_step_id.to_string()]);
    loop {
        let before = selected.len();
        for step in &plan.steps {
            if !selected.contains(&step.step_id)
                && step
                    .dependencies
                    .iter()
                    .any(|dependency| selected.contains(dependency))
            {
                selected.insert(step.step_id.clone());
            }
        }
        if selected.len() == before {
            break;
        }
    }
    Some(selected)
}

pub(super) fn scoped_recovery_plan(
    plan: &ExecutionPlan,
    failed_step_id: &str,
    revision: u32,
) -> Option<ExecutionPlan> {
    if plan.dag_jsonld.is_some() {
        return None;
    }
    let retained = recovery_descendant_step_ids(plan, failed_step_id)?;
    let mut scoped = plan.clone();
    scoped.plan_id = format!("{}_delta_{}", plan.plan_id, revision);
    scoped.steps = plan
        .steps
        .iter()
        .filter(|step| retained.contains(&step.step_id))
        .cloned()
        .collect();
    for step in &mut scoped.steps {
        step.dependencies
            .retain(|dependency| retained.contains(dependency));
    }
    if let Some(provenance) = scoped.agent_spec_provenance.as_mut() {
        provenance
            .step_sources
            .retain(|step_id, _| retained.contains(step_id));
    }
    scoped.agent_sequence = scoped.steps.iter().map(|step| step.role).collect();
    scoped.parallel_groups.clear();
    scoped.verify_first = false;
    scoped.fallback_steps.clear();
    scoped.description = format!(
        "Scoped recovery from failed step {}; completed predecessors preserved",
        failed_step_id
    );
    Some(scoped)
}

/// Read only kernel-authored recovery provenance from the result error list.
/// The same words in model prose have no routing authority.
pub(super) fn kernel_recovery_route(
    result: &TaskResult,
) -> Option<(crate::core::recovery::RecoveryDirective, String)> {
    const PREFIX: &str = "SA kernel recovery route: directive=";
    result.errors.iter().rev().find_map(|error| {
        let encoded = error.strip_prefix(PREFIX)?;
        let (directive, failed_step) = encoded.split_once(";failed_step=")?;
        if failed_step.is_empty() || failed_step.chars().any(char::is_whitespace) {
            return None;
        }
        let directive = match directive {
            "RetryCa" => crate::core::recovery::RecoveryDirective::RetryCa,
            "RetryDa" => crate::core::recovery::RecoveryDirective::RetryDa,
            "ReplanPa" => crate::core::recovery::RecoveryDirective::ReplanPa,
            "Blocked" => crate::core::recovery::RecoveryDirective::Blocked,
            _ => return None,
        };
        Some((directive, failed_step.to_string()))
    })
}

pub(super) fn kernel_recovery_directive(
    result: &TaskResult,
) -> Option<crate::core::recovery::RecoveryDirective> {
    kernel_recovery_route(result).map(|(directive, _)| directive)
}

/// Resolve a structured role-owned retry into a PDCA delta plan.
///
/// `RetryDa` begins at the nearest implementation boundary. `RetryCa` begins
/// at the verifier and therefore cannot replay completed implementation or
/// acquire mutation authority merely because an evidence receipt was absent.
/// Recovery markers are emitted by SA; model-authored prose alone is never
/// treated as capability authority.
/// External DAG workflows retain their own retry/branch topology and are not
/// rewritten here.
pub(super) fn scoped_retry_plan_for_decision(
    plan: &ExecutionPlan,
    decision: &crate::core::recovery::DecisionReport,
    kernel_failed_step: Option<&str>,
) -> Option<ExecutionPlan> {
    if !matches!(
        decision.directive,
        crate::core::recovery::RecoveryDirective::RetryDa
            | crate::core::recovery::RecoveryDirective::RetryCa
    ) || plan.dag_jsonld.is_some()
    {
        return None;
    }

    let reported_index = kernel_failed_step.and_then(|failed_step| {
        plan.steps
            .iter()
            .position(|step| step.step_id == failed_step)
    });
    let start_index = if decision.directive == crate::core::recovery::RecoveryDirective::RetryCa {
        reported_index
            .filter(|index| plan.steps[*index].role == AgentRole::Check)
            .or_else(|| {
                plan.steps
                    .iter()
                    .position(|step| step.role == AgentRole::Check)
            })?
    } else {
        let nearest_reported_da = reported_index.and_then(|index| {
            (0..=index)
                .rev()
                .find(|candidate| plan.steps[*candidate].role == AgentRole::Do)
        });
        let first_da = plan
            .steps
            .iter()
            .position(|step| step.role == AgentRole::Do);
        let first_downstream_gate = plan
            .steps
            .iter()
            .position(|step| matches!(step.role, AgentRole::Check | AgentRole::Act));
        nearest_reported_da
            .or(first_da)
            .or_else(|| {
                reported_index.filter(|index| {
                    matches!(plan.steps[*index].role, AgentRole::Check | AgentRole::Act)
                })
            })
            .or(first_downstream_gate)?
    };

    if decision.directive == crate::core::recovery::RecoveryDirective::RetryCa {
        let retained = recovery_descendant_step_ids(plan, &plan.steps[start_index].step_id)?;
        // A verifier-only recovery cannot cross a downstream implementation
        // boundary. Reject malformed/mixed-role topology instead of silently
        // granting DA mutation or deleting dependencies by suffix position.
        if plan.steps.iter().any(|step| {
            retained.contains(&step.step_id)
                && !matches!(step.role, AgentRole::Check | AgentRole::Act)
        }) {
            return None;
        }
    }

    scoped_recovery_plan(
        plan,
        &plan.steps[start_index].step_id,
        decision.plan_revision.saturating_add(1),
    )
}

fn policy_reward_breakdown(
    result: &TaskResult,
    prompt_tokens: u64,
    elapsed_ms: u64,
    workspace_mutation_required: bool,
) -> PolicyRewardBreakdown {
    let status_reward = match result.status.as_str() {
        "success" | "completed" if !result.summary.contains("[Recovery] scope=Task") => 1.0,
        "partial_success" => 0.25,
        "failed" | "timeout" => -1.0,
        _ => -0.5,
    };
    let failed_actions = result
        .tracked_actions
        .iter()
        .filter(|action| {
            matches!(
                action.status,
                crate::core::tracked_action::ActionStatus::Failed
                    | crate::core::tracked_action::ActionStatus::Retried
            )
        })
        .count() as f32;
    let failed_action_penalty = if result.tracked_actions.is_empty() {
        0.0
    } else {
        (failed_actions / result.tracked_actions.len() as f32 * 0.35).min(0.35)
    };
    // A tool call normally needs one reasoning turn. Penalize only turns that
    // exceed that work plus a small PA/CA/AA coordination allowance.
    let efficient_turn_ceiling = result.tool_call_count.saturating_add(4);
    let excess_turns = result.turn_count.saturating_sub(efficient_turn_ceiling);
    let excess_turn_penalty = (excess_turns as f32 * 0.025).min(0.2);
    let recovery_penalty = if result.summary.contains("[Recovery] scope=Task") {
        0.15
    } else {
        0.0
    };
    let error_penalty = (result.errors.len() as f32 * 0.04).min(0.16);
    let prompt_token_penalty = if prompt_tokens > 120_000 {
        (((prompt_tokens - 120_000) as f32 / 40_000.0) * 0.03).min(0.15)
    } else {
        0.0
    };
    let latency_penalty = if elapsed_ms > 120_000 {
        (((elapsed_ms - 120_000) as f32 / 60_000.0) * 0.03).min(0.12)
    } else {
        0.0
    };
    let first_substantive_action_ordinal = result
        .tracked_actions
        .iter()
        .position(|action| action.substantive_effect)
        .map(|index| index + 1);
    let late_substantive_action_penalty = first_substantive_action_ordinal
        .map(|ordinal| (ordinal.saturating_sub(8) as f32 * 0.02).min(0.1))
        .unwrap_or(0.0);
    let mut evidence_keys = std::collections::HashSet::new();
    let mut evidence_count = 0usize;
    let mut duplicate_evidence_count = 0usize;
    for action in &result.tracked_actions {
        if matches!(
            action.tool_name.as_str(),
            "file_read" | "file_list" | "glob_search" | "grep_search" | "workspace_status"
        ) {
            evidence_count += 1;
            let key = format!(
                "{}:{}",
                action.tool_name,
                serde_json::to_string(&action.tool_args).unwrap_or_default()
            );
            if !evidence_keys.insert(key) {
                duplicate_evidence_count += 1;
            }
        }
    }
    let redundant_read_ratio = if evidence_count == 0 {
        0.0
    } else {
        duplicate_evidence_count as f32 / evidence_count as f32
    };
    let redundant_read_penalty = (redundant_read_ratio * 0.16).min(0.16);
    let no_effect_tail = result
        .tracked_actions
        .iter()
        .rev()
        .take_while(|action| !action.substantive_effect)
        .count();
    let no_effect_tail_penalty =
        if !workspace_mutation_required || result.tracked_actions.is_empty() {
            0.0
        } else {
            (no_effect_tail as f32 / result.tracked_actions.len() as f32 * 0.12).min(0.12)
        };
    let total = (status_reward
        - failed_action_penalty
        - excess_turn_penalty
        - recovery_penalty
        - error_penalty
        - prompt_token_penalty
        - latency_penalty
        - late_substantive_action_penalty
        - redundant_read_penalty
        - no_effect_tail_penalty)
        .clamp(-1.0, 1.0);
    PolicyRewardBreakdown {
        status_reward,
        failed_action_penalty,
        excess_turn_penalty,
        recovery_penalty,
        error_penalty,
        prompt_token_penalty,
        latency_penalty,
        late_substantive_action_penalty,
        first_substantive_action_ordinal,
        redundant_read_ratio,
        redundant_read_penalty,
        no_effect_tail,
        no_effect_tail_penalty,
        total,
    }
}

/// The CA/AA evidence writer uses this same deterministic task key. Keeping
/// the derivation here avoids putting execution payloads into learning records
/// merely to discover whether independent verification was completed.
fn task_audit_evidence_iri(task_iri: &str) -> String {
    format!(
        "{}{}",
        crate::core::policy_learning::AUDIT_EVIDENCE_PREFIX,
        hex::encode(Sha256::digest(task_iri.as_bytes()))
    )
}

impl SupervisorAgent {
    /// Terminate before role materialization when SA could not obtain a real,
    /// task-specific execution plan.  There is deliberately no synthetic
    /// `ExecutionPlan` or resume contract here: both would give a kernel
    /// fallback the appearance of LLM-authored PA/DA/CA/AA authority.
    /// Because no role or tool has run, submitting the same task again is a
    /// safe planning retry. Contract rejection and provider unavailability
    /// remain distinct so the operator is not sent toward the wrong remedy.
    async fn planning_blocked_result(
        &mut self,
        task_iri: &str,
        cycle_id: &str,
        error: &CoreError,
        previous_result: Option<TaskResult>,
    ) -> TaskResult {
        let diagnostic = error.to_string();
        let (block_reason, error_code, blocked_stage, retry_guidance, required_action) = match error {
            CoreError::InteractionRejected { stage, .. }
                if stage == "sa_plan_contract_rejected" => (
                "sa_plan_contract_rejected",
                "SA_PLAN_CONTRACT_REJECTED",
                stage.as_str(),
                "The configured LLM responded, but its plan and bounded correction were rejected by the planning contract. Inspect the diagnostic before resubmitting the task.",
                "inspect the planning diagnostic and retry with a contract-valid plan",
            ),
            _ => (
                "sa_planning_unavailable",
                "SA_PLANNING_UNAVAILABLE",
                "sa_plan_generation",
                "It is safe to submit the same task again after the configured LLM becomes available.",
                "retry planning after the configured LLM is available",
            ),
        };
        let summary = if previous_result.is_none() {
            format!(
                "Planning blocked before role dispatch: SA could not obtain a valid task-specific LLM plan. No generic PA/DA/CA/AA fallback was executed. {retry_guidance} Diagnostic: {diagnostic}"
            )
        } else {
            format!(
                "Planning blocked before fallback role dispatch: SA could not obtain a valid task-specific LLM plan. Earlier verify-first evidence is preserved, and no generic PA/DA/CA/AA fallback was executed. {retry_guidance} Diagnostic: {diagnostic}"
            )
        };
        warn!(
            task_iri = %task_iri,
            cycle_id = %cycle_id,
            error = %diagnostic,
            "SA planning failed closed before role dispatch"
        );

        if let Some(cycle) = self.active_cycles.get_mut(cycle_id) {
            cycle.phase = CyclePhase::Idle;
            cycle.task_completed = false;
            cycle.phase_history.push("PlanningBlocked".to_string());
        }
        self.emit_sa_thought(task_iri, &summary, "planning_blocked")
            .await;
        self.event_bus
            .emit(
                task_iri,
                "RECOVERY_BLOCKED",
                "SA",
                &serde_json::json!({
                    "reason": block_reason,
                    "stage": blocked_stage,
                    "diagnostic": diagnostic,
                    "automatic_role_dispatch": false,
                    "generic_plan_fallback": false,
                    "resume_safe": true,
                    "resume_mode": "resubmit_same_task",
                    "required_action": required_action,
                })
                .to_string(),
            )
            .await;
        let unprocessed_commands = self
            .event_bus
            .close_and_take_supplementary_commands(task_iri)
            .len();

        let mut result = previous_result.unwrap_or_else(|| TaskResult {
            task_iri: task_iri.to_string(),
            status: "blocked".to_string(),
            verdict: Some(TaskVerdict::Blocked),
            summary: String::new(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 0,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        });
        result.status = "blocked".to_string();
        result.verdict = Some(TaskVerdict::Blocked);
        result.summary = if result.summary.trim().is_empty() {
            summary
        } else {
            format!("{}\n\n{}", result.summary, summary)
        };
        result.errors.push(format!("{error_code}: {diagnostic}"));
        if unprocessed_commands > 0 {
            result.errors.push(format!(
                "{unprocessed_commands} supplementary command(s) arrived after the planning retry boundary"
            ));
        }
        result
    }

    #[instrument(skip(self, user_input), fields(task_iri = %task_iri))]
    pub async fn process_task(
        &mut self,
        user_input: &str,
        task_iri: &str,
    ) -> Result<TaskResult, CoreError> {
        self.process_task_with_context(
            user_input,
            task_iri,
            TaskContext::new(task_iri, user_input, self.max_iterations),
        )
        .await
    }

    /// Process task with custom TaskContext, supports resume mode
    #[instrument(skip(self, user_input, ctx), fields(task_iri = %task_iri))]
    pub async fn process_task_with_context(
        &mut self,
        user_input: &str,
        task_iri: &str,
        ctx: TaskContext,
    ) -> Result<TaskResult, CoreError> {
        let canonical_user_input = if let Some(state) = ctx.resumed_state.as_ref() {
            state.validate()?;
            state.contract.original_user_task.clone()
        } else {
            user_input.to_string()
        };
        let user_input = canonical_user_input.as_str();
        // The lease keeps a root scope resident while all parallel BizAgent
        // children contribute to it and retires counters on every return,
        // error, or cancellation path when this future is dropped.
        let _task_accounting_lease = self
            .runner
            .llm_interactions
            .begin_scope_accounting(task_iri);
        self.event_bus.open_supplementary_commands(task_iri);
        let cycle_id = self.start_cycle(user_input, task_iri).await?;
        let task_started_at = std::time::Instant::now();

        // This is the authoritative, mutable task contract for the complete
        // outer PDCA lifetime. A supplementary delivery update received in
        // one execute_plan call must survive retries/replans instead of
        // reverting to the immutable entry context on the next cycle.
        let mut effective_task_constraints = ctx
            .resumed_state
            .as_ref()
            .map(|state| state.contract.constraints_hash_map())
            .unwrap_or_else(|| ctx.constraints.clone());
        let mut effective_task_effect_policy = ctx
            .resumed_state
            .as_ref()
            .map(|state| state.contract.effect_policy.clone())
            .unwrap_or_else(|| ctx.effective_effect_policy());

        // UI counters remain process-wide, while budgets and learning costs
        // use the interaction plane's root-task ledger. This includes every
        // BizAgent child but excludes unrelated/background model traffic.
        let task_usage_start = self
            .runner
            .llm_interactions
            .usage_snapshot_for_scope(task_iri);
        let task_prompt_tokens_start = task_usage_start.prompt_tokens;
        let task_completion_tokens_start = task_usage_start.completion_tokens;
        let task_context_quality_start = self
            .runner
            .llm_interactions
            .context_quality_snapshot_for_scope(task_iri);

        let declared_effect_execution = effective_task_constraints
            .get("required_effect")
            .is_some_and(|value| value == "workspace_mutation");
        let extraction_started_at = std::time::Instant::now();
        let mut five_w2h = self
            .extract_5w2h_from_input(task_iri, user_input, declared_effect_execution)
            .await;
        tracing::info!(
            task_iri = %task_iri,
            elapsed_ms = extraction_started_at.elapsed().as_millis() as u64,
            "SA 5W2H extraction completed"
        );
        let task_id = task_iri
            .strip_prefix("iri://task/")
            .unwrap_or_else(|| task_iri.strip_prefix("iri://").unwrap_or(task_iri));
        let five_w2h_iri = format!("iri://task/{}/5w2h", task_id);

        // A3: Calculate task_embedding from 5W2H → set to relevance_tracker
        if let Some(ref embedder) = self.embedder {
            let task_text = format!("{}\n{}", five_w2h.what, five_w2h.why.description);
            if let Ok(task_emb) = embedder.embed(&task_text).await {
                self.relevance_tracker.set_task_context(task_emb);
            }
        }

        // Inject current working directory as execution environment, so LLM knows where to create files
        if five_w2h
            .where_
            .as_ref()
            .and_then(|w| w.execution_environment.as_ref())
            .is_none()
        {
            if let Ok(cwd) = std::env::current_dir() {
                let cwd_str = cwd.to_string_lossy().to_string();
                five_w2h = five_w2h.with_where(crate::core::five_w2h::WhereDetail {
                    data_sources: vec![],
                    execution_environment: Some(cwd_str),
                    target_repository: None,
                    target_branch: None,
                });
            }
        }

        // Fill missing 5W2H dimensions before PA dispatch (SA phase), not at CA stage.
        five_w2h.derive_defaults(self.max_iterations, self.max_pdca_cycles);

        if let Ok(json_ld) = five_w2h.to_json_ld(task_iri) {
            let _ = self
                .runner
                .l0_store
                .store(&five_w2h_iri, &json_ld.to_string());
            let cfg = crate::CoreConfig::default();
            if let Some(ref bb) = self.blackboard {
                if bb
                    .write_node(&five_w2h_iri, &json_ld.to_string(), &cfg)
                    .is_ok()
                {
                    tracing::debug!(five_w2h_iri = %five_w2h_iri, "5W2H written to blackboard");
                    let route = self.type_router.get_route("task:5W2H");
                    if let Some(route) = route {
                        for event in &route.events {
                            let _ = self
                                .event_bus
                                .emit(task_iri, event, "system:sa", &five_w2h_iri)
                                .await;
                        }
                    }
                }
            }
            let what_sha256 = format!(
                "sha256:{}",
                hex::encode(&Sha256::digest(five_w2h.what.as_bytes())[..12])
            );
            tracing::info!(
                task_iri = %task_iri,
                what_chars = five_w2h.what.chars().count(),
                %what_sha256,
                "5W2H initialization complete"
            );
        }

        // start_cycle already performed perception once. Reusing its result
        // avoids a duplicate semantic lookup and gives baseline/shadow/active
        // one unambiguous treatment assignment for the whole task.
        let perception_hints = self
            .active_cycles
            .get(&cycle_id)
            .map(|cycle| cycle.experience_hints.clone())
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, hint)| format!("[Experience source=perception:{}] {}", index + 1, hint))
            .collect::<Vec<_>>();
        let perceived_hint_count = self
            .active_cycles
            .get(&cycle_id)
            .map(|cycle| cycle.observed_experience_hint_count)
            .unwrap_or(perception_hints.len());
        // Preserve the exact experience treatment before skill and knowledge
        // enrichment so the constrained policy can choose a genuinely
        // different, measurable ordering arm.
        // Enrich with skill discovery results if a discovery engine is available.
        // Keep sources separate until after policy selection so every action
        // corresponds to the treatment recorded in the learning evaluation.
        let mut skill_hints: Vec<LearningHintCandidate> = Vec::new();
        let mut knowledge_hints: Vec<LearningHintCandidate> = Vec::new();
        // The original objective is deterministic for matched tasks. Using
        // model-generated 5W2H text here fragments identical observations
        // because the extraction paraphrases `what` on every call.
        let task_context = learning_task_context(user_input);
        let policy_context = task_context.family.clone();
        let mut discovered_skill_count = 0usize;
        let mut discovered_knowledge_count = 0usize;
        let mut observed_skill_iris = Vec::new();
        let mut observed_knowledge_fragment_iris = Vec::new();
        let prompt_settings = &self.runner.token_optimization.prompt_optimization;
        if self.learning_mode.retrieves_history() {
            if let Some(ref de) = self.discovery_engine {
                let mut discovery_constraints = five_w2h.why.success_criteria.clone();
                discovery_constraints.extend(
                    effective_task_constraints
                        .iter()
                        .map(|(key, value)| format!("{key}={value}")),
                );
                let disc_task = crate::skill_graph::discovery::Task5W2H {
                    what: user_input.to_string(),
                    why: five_w2h.why.description.clone(),
                    who: five_w2h.who.as_ref().and_then(|w| w.required_role.clone()),
                    when_phase: five_w2h.when.as_ref().map(|w| format!("{:?}", w)),
                    where_context: five_w2h.where_.as_ref().map(|w| format!("{:?}", w)),
                    how_approach: five_w2h.how.as_ref().and_then(|h| h.required_steps.clone()),
                    constraints: discovery_constraints,
                };
                let matches = de.discover_for_task(&disc_task).await;
                skill_hints = matches
                    .iter()
                    .filter_map(|m| {
                        let name = if !m.skill.name.is_empty() {
                            m.skill.name.clone()
                        } else {
                            m.skill.skill_iri.rsplit('/').next()?.to_string()
                        };
                        Some(LearningHintCandidate {
                            text: format!(
                                "[Skill source={} relevance={:.2}] {}",
                                m.skill.skill_iri, m.relevance_score, name
                            ),
                            source: LearningHintSource::Skill {
                                iri: m.skill.skill_iri.clone(),
                            },
                        })
                    })
                    .take(prompt_settings.max_discovered_skill_hints)
                    .collect();
                observed_skill_iris = skill_hints
                    .iter()
                    .filter_map(|hint| match &hint.source {
                        LearningHintSource::Skill { iri } => Some(iri.clone()),
                        _ => None,
                    })
                    .collect();
                let mut seen_skill_iris = std::collections::HashSet::new();
                observed_skill_iris.retain(|iri| seen_skill_iris.insert(iri.clone()));
                knowledge_hints = if let Some(graph) = self.runner.skill_graph_store.as_ref() {
                    let fragments = rank_knowledge_fragments(
                        matches
                            .iter()
                            .flat_map(|m| graph.get_fragments_for_skill(&m.skill.skill_iri))
                            .collect::<Vec<_>>(),
                        &task_context,
                        prompt_settings.max_knowledge_fragments,
                    );
                    observed_knowledge_fragment_iris = fragments
                        .iter()
                        .map(|fragment| fragment.fragment_iri.clone())
                        .collect();
                    fragments
                        .iter()
                        .map(|fragment| LearningHintCandidate {
                            text: render_knowledge_hint(fragment),
                            source: LearningHintSource::Knowledge {
                                iri: fragment.fragment_iri.clone(),
                            },
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                discovered_skill_count = skill_hints.len();
                discovered_knowledge_count = knowledge_hints.len();
            }
        }
        let policy_candidates = eligible_policy_candidates(
            discovered_skill_count,
            discovered_knowledge_count,
            perceived_hint_count,
        );
        let policy_choice = if self.learning_mode.injects_history() {
            self.policy_learning
                .choose(&policy_context, &policy_candidates, "baseline")
        } else {
            crate::core::policy_learning::PolicyChoice {
                context: policy_context.clone(),
                action: "baseline".to_string(),
                used_fallback: true,
                confidence: 0.0,
                explored: false,
                candidates: policy_candidates,
            }
        };
        let experience_hint_candidates = perception_hints
            .iter()
            .map(|text| LearningHintCandidate {
                text: text.clone(),
                source: LearningHintSource::Experience,
            })
            .collect::<Vec<_>>();
        let materialized_treatment = if self.learning_mode.injects_history() {
            materialize_policy_treatment(
                &policy_choice.action,
                &experience_hint_candidates,
                &skill_hints,
                &knowledge_hints,
                prompt_settings.max_learning_hints,
                prompt_settings.max_learning_hint_chars,
                prompt_settings.max_learning_hint_total_chars,
            )
        } else {
            MaterializedLearningTreatment::default()
        };
        let all_hints = materialized_treatment.hints.clone();
        if !all_hints.is_empty() {
            tracing::info!(
                task_iri = %task_iri,
                policy_action = %policy_choice.action,
                hints = all_hints.len(),
                "Selected learning treatment enriched planning"
            );
        }
        // start_cycle captures observed history before the policy exists. From
        // this point onward BizAgents must see only the selected treatment.
        if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
            cycle.experience_hints = all_hints.clone();
        }
        let learning_treatment = LearningTreatmentMetrics {
            perception_hints_observed: perceived_hint_count,
            experience_hint_fingerprints: self
                .active_cycles
                .get(&cycle_id)
                .map(|cycle| cycle.observed_experience_hint_fingerprints.clone())
                .unwrap_or_default(),
            skills_observed: discovered_skill_count,
            skill_iris_observed: observed_skill_iris,
            skill_iris_injected: materialized_treatment.skill_iris,
            knowledge_fragments_observed: discovered_knowledge_count,
            knowledge_fragment_iris_observed: observed_knowledge_fragment_iris,
            knowledge_fragment_iris_injected: materialized_treatment.knowledge_fragment_iris,
            hints_injected: all_hints.len(),
            hint_chars_injected: all_hints.iter().map(|hint| hint.chars().count()).sum(),
            task_family_raw_features: task_context.raw_features,
            experiment_pair_id: ctx.constraints.get("learning_pair_id").cloned(),
            experiment_seed: ctx.constraints.get("learning_seed").cloned(),
            experiment_model: ctx.constraints.get("learning_model").cloned(),
            experiment_config_fingerprint: ctx
                .constraints
                .get("learning_experiment_config_fingerprint")
                .cloned(),
            workspace_fingerprint: ctx
                .constraints
                .get("learning_workspace_fingerprint")
                .cloned(),
            objective_fingerprint: {
                use sha2::{Digest, Sha256};
                format!(
                    "sha256:{}",
                    hex::encode(&Sha256::digest(user_input.as_bytes())[..12])
                )
            },
            orchestration_mode: if ctx.workflow_jsonld.is_some() {
                "dag".to_string()
            } else {
                "pdca".to_string()
            },
        };
        self.event_bus
            .emit(
                task_iri,
                "LEARNING_TREATMENT",
                "SA",
                &serde_json::json!({
                    "mode": self.learning_mode,
                    "policy_context": policy_context,
                    "policy_action": policy_choice.action,
                    "treatment": learning_treatment,
                    "model_version": self.policy_learning.model_version(),
                })
                .to_string(),
            )
            .await;

        // Unified execution path: build ExecutionPlan from JSON-LD workflow or LLM.
        // Normal TUI tasks always obtain the detailed LLM plan before any
        // role dispatch. Verify-first may reorder its LLM-authored CA/AA, but
        // may not defer planning through a generic kernel role profile.
        let mut plan = if let Some(ref wf_jsonld) = ctx.workflow_jsonld {
            info!(task_iri = %task_iri, "Using JSON-LD workflow mode — converting through adapter to ExecutionPlan");
            let def =
                crate::core::workflow::loader::load_workflow_jsonld(wf_jsonld).map_err(|e| {
                    CoreError::Internal {
                        message: format!("Workflow parsing failed: {}", e),
                    }
                })?;
            let dag = crate::core::workflow::loader::build_dag(&def).map_err(|e| {
                CoreError::Internal {
                    message: format!("DAG build failed: {}", e),
                }
            })?;
            let mut plan =
                crate::core::workflow::adapter::dag_to_execution_plan(&dag, &def, task_iri);
            plan.dag_jsonld = Some(wf_jsonld.clone());
            plan
        } else if let Some(state) = ctx.resumed_state.as_ref() {
            state.contract.execution_plan.clone()
        } else {
            // SA's planner supplies the task-specific definitions from which
            // every isolated PA/DA/CA/AA BizAgent materializes its own
            // agent.md. Workspace-effect tasks follow the same path: a generic
            // kernel PlanStep would preserve control flow but silently replace
            // the model-generated role specification that provides business
            // isolation and specialization.
            match self
                .analyze_task_with_llm(
                    task_iri,
                    user_input,
                    &five_w2h,
                    &all_hints,
                    &effective_task_constraints,
                )
                .await
            {
                Ok(plan) => plan,
                Err(error) => {
                    return Ok(self
                        .planning_blocked_result(task_iri, &cycle_id, &error, None)
                        .await);
                }
            }
        };
        tracing::info!(
            task_iri = %task_iri,
            elapsed_ms = task_started_at.elapsed().as_millis() as u64,
            steps = plan.steps.len(),
            "SA planning completed"
        );

        // ── Verify-first optimization ──
        // Existing work may be accepted before DA, but both preflight roles
        // must remain clones of the exact LLM-authored plan definitions.
        if !plan.verify_first
            && ctx.workspace_file_summary.is_some()
            && ctx.resumed_state.is_none()
            && ctx.workflow_jsonld.is_none()
            && !effective_task_constraints
                .get("required_effect")
                .is_some_and(|value| value == "workspace_mutation")
            && plan.steps.len() >= 2
            && plan
                .steps
                .iter()
                .any(|s| matches!(s.role, AgentRole::Plan | AgentRole::Do))
        {
            let ws_summary = ctx
                .workspace_file_summary
                .as_deref()
                .unwrap_or("workspace has files");
            if let Err(error) = materialize_verify_first_from_llm_plan(&mut plan, task_iri) {
                return Ok(self
                    .planning_blocked_result(task_iri, &cycle_id, &error, None)
                    .await);
            }

            info!(task_iri = %task_iri, ws = %ws_summary, "Verify-first: LLM-authored CA→AA cloned, fallback_steps={}", plan.fallback_steps.len());
        }

        // Adapt relevance tracker decay λ to task complexity
        self.relevance_tracker
            .adapt_to_complexity(&plan.task_complexity);

        let step_roles: Vec<String> = plan.steps.iter().map(|s| format!("{:?}", s.role)).collect();
        self.emit_sa_thought(
            task_iri,
            &format!(
                "Task classified. Plan: {} ({} steps: {})",
                plan.description,
                plan.steps.len(),
                step_roles.join(" → ")
            ),
            "plan_created",
        )
        .await;

        if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
            cycle.phase = CyclePhase::Executing;
            cycle
                .phase_history
                .push(format!("Plan: {}", plan.description));
        }

        // Route planning-time events through the same owner used between
        // execution steps. This prevents one drain site from consuming a user
        // command or approval intended for another stage.
        self.drain_and_route_runtime_events(task_iri).await;

        // ── Outer SA-level PDCA retry loop ──
        // Ensure at least 2 cycles when verify_first is active
        // (cycle 0 = verify-first CA→AA, cycle 1+ = fallback PDCA)
        let max_cycles = if plan.verify_first {
            (self.max_pdca_cycles.max(1)).max(2)
        } else {
            self.max_pdca_cycles.max(1)
        };
        let mut cycle_feedback: Option<String> = None;
        let mut final_result: Option<TaskResult> = None;
        // execute_plan aggregates every BizAgent inside one orchestration
        // cycle.  This second-level accumulator preserves facts across outer
        // SA PDCA retries so the terminal TaskResult describes the whole user
        // task rather than only its last cycle.
        let mut task_execution_facts = super::execution::TaskExecutionFacts::default();
        let mut plan_revision = 1u32;
        let mut next_scoped_plan: Option<ExecutionPlan> = None;
        let mut task_scope_replans_used = 0u32;
        let mut recovery_state = super::execution::PdcaRecoveryState::default();
        let token_budget = five_w2h.how_much.as_ref().and_then(|h| h.token_budget);
        tracing::info!(
            task_iri = %task_iri,
            token_budget = ?token_budget,
            prompt_tokens_start = task_prompt_tokens_start,
            completion_tokens_start = task_completion_tokens_start,
            "SA task token budget initialized"
        );

        for cycle_num in 0..max_cycles {
            // A previous cycle may have atomically closed its terminal input
            // boundary before returning a retryable failure.
            self.event_bus.open_supplementary_commands(task_iri);
            // Reset only the current PDCA attempt clock. The task lifetime
            // `started_at` remains unchanged for end-to-end SLO metrics.
            let now = chrono::Utc::now();
            let cycle_timeout = self.perception.cycle_timeout_secs().max(1);
            if let Some(cycle) = self
                .active_cycles
                .values_mut()
                .find(|cycle| cycle.task_iri == task_iri)
            {
                cycle.pdca_started_at = now;
                cycle.cycle_deadline_at = now + chrono::Duration::seconds(cycle_timeout);
                cycle.last_progress_at = now;
                cycle.last_timeout_alert_at = None;
                cycle.next_timeout_alert_at = None;
                cycle.timeout_alert_count = 0;
                cycle.outer_cycle_number = cycle_num + 1;
            }
            if let Some(limit) = token_budget {
                let cumulative_used = self
                    .runner
                    .llm_interactions
                    .usage_snapshot_for_scope(task_iri)
                    .total_tokens();
                let baseline_used =
                    task_prompt_tokens_start.saturating_add(task_completion_tokens_start);
                let used = cumulative_used.saturating_sub(baseline_used);
                if used >= limit {
                    let summary = format!(
                        "Recovery blocked: token budget exhausted before PDCA cycle {} (used {}, limit {}).",
                        cycle_num + 1,
                        used,
                        limit
                    );
                    tracing::warn!(task_iri = %task_iri, used, limit, "Task token budget exhausted");
                    self.event_bus
                        .emit(task_iri, "RECOVERY_BLOCKED", "SA", &summary)
                        .await;
                    // Preserve the last completed cycle's evidence. Replacing
                    // it with a zeroed result made a budget stop look like a
                    // task that had executed no turns or tools at all.
                    let mut blocked_result = final_result.take().unwrap_or_else(|| TaskResult {
                        task_iri: task_iri.to_string(),
                        status: "failed".to_string(),
                        summary: String::new(),
                        output: None,
                        jsonld_output: None,
                        artifacts: Vec::new(),
                        errors: Vec::new(),
                        turn_count: 0,
                        tool_call_count: 0,
                        five_w2h_updates: None,
                        tracked_actions: Vec::new(),
                        verdict: None,
                        archive_iri: None,
                    });
                    blocked_result.status = "failed".to_string();
                    blocked_result.summary = if blocked_result.summary.is_empty() {
                        summary.clone()
                    } else {
                        format!("{}\n\n{}", blocked_result.summary, summary)
                    };
                    blocked_result
                        .errors
                        .push("token budget exhausted".to_string());
                    final_result = Some(blocked_result);
                    break;
                }
            }
            let resumed = if cycle_num == 0 {
                ctx.resumed_messages.clone()
            } else {
                None
            };
            let resumed_state = if cycle_num == 0 {
                ctx.resumed_state.clone()
            } else {
                None
            };
            let conversation_history = if cycle_num == 0 {
                ctx.conversation_history.clone()
            } else {
                None
            };

            // On retry after verify-first failed, switch to fallback_steps (full PDCA)
            let current_plan = if let Some(scoped) = next_scoped_plan.take() {
                plan_revision = plan_revision.saturating_add(1);
                scoped
            } else if cycle_num >= 1 && plan.verify_first && !plan.fallback_steps.is_empty() {
                let mut fb = plan.clone();
                plan_revision = cycle_num as u32 + 1;
                fb.plan_id = format!("{}_rev_{}", plan.plan_id, plan_revision);
                fb.steps = plan.fallback_steps.clone();
                let retained_step_ids = fb
                    .steps
                    .iter()
                    .map(|step| step.step_id.as_str())
                    .collect::<std::collections::HashSet<_>>();
                if let Some(provenance) = fb.agent_spec_provenance.as_mut() {
                    provenance
                        .step_sources
                        .retain(|step_id, _| retained_step_ids.contains(step_id.as_str()));
                }
                fb.verify_first = false;
                fb.agent_sequence = fb.steps.iter().map(|s| s.role).collect();
                fb.description = format!(
                    "Fallback PDCA (verify-first CA did not pass): {}",
                    plan.description.trim_start_matches(
                        "[Verify-first] Check existing workspace code before full PDCA. Fallback: "
                    )
                );
                fb
            } else {
                let mut current = plan.clone();
                if cycle_num > 0 {
                    plan_revision = cycle_num as u32 + 1;
                    current.plan_id = format!("{}_rev_{}", plan.plan_id, plan_revision);
                }
                current
            };

            info!(
                task_iri = %task_iri,
                cycle_num = cycle_num + 1,
                max_cycles = max_cycles,
                has_feedback = cycle_feedback.is_some(),
                "Starting SA-level PDCA cycle"
            );

            if let Some(ref _feedback) = cycle_feedback {
                self.emit_sa_thought(
                    task_iri,
                    &format!(
                        "⚠️ PDCA cycle #{} did not pass its latest quality gate — restarting with targeted feedback",
                        cycle_num + 1
                    ),
                    "pdca_retry_start",
                )
                .await;
            } else {
                self.emit_sa_thought(
                    task_iri,
                    &format!("Starting PDCA cycle {}/{}", cycle_num + 1, max_cycles),
                    "pdca_cycle_start",
                )
                .await;
            }

            let mode = if current_plan.dag_jsonld.is_some() {
                crate::core::recovery::OrchestrationMode::Dag
            } else {
                crate::core::recovery::OrchestrationMode::Pdca
            };
            let executed_plan = current_plan.clone();
            let mut result = self
                .execute_plan(
                    current_plan,
                    task_iri,
                    user_input,
                    five_w2h.clone(),
                    &five_w2h_iri,
                    resumed,
                    resumed_state,
                    conversation_history,
                    cycle_feedback.clone(),
                    ctx.workspace_file_summary.as_deref(),
                    &mut effective_task_effect_policy,
                    &mut effective_task_constraints,
                    &mut recovery_state,
                )
                .await?;
            task_execution_facts.record(&result);

            // A timeout/blocked phase has an indeterminate replay boundary:
            // its durable journal may already contain a committed tool
            // effect even though the cancelled BizAgent could not return its
            // in-memory action ledger. Do not classify it as RetryDa/ReplanPa
            // below. Stop the outer PDCA loop and require an explicit,
            // journal-aware resume decision instead.
            if super::execution::requires_safe_plan_stop(&result) {
                task_execution_facts.apply_to(&mut result);
                warn!(
                    task_iri = %task_iri,
                    cycle_num = cycle_num + 1,
                    status = %result.status,
                    "PDCA recovery stopped at an indeterminate side-effect boundary"
                );
                self.event_bus
                    .emit(
                        task_iri,
                        "RECOVERY_BLOCKED",
                        "SA",
                        &serde_json::json!({
                            "reason": "indeterminate_side_effect_boundary",
                            "status": &result.status,
                            "automatic_retry": false,
                            "required_action": "inspect_durable_journal_before_resume",
                        })
                        .to_string(),
                    )
                    .await;
                final_result = Some(result);
                break;
            }

            // A bounded CA-only retry that still cannot obtain admissible
            // evidence is terminal for automatic recovery. Replaying PA/DA
            // cannot create a verifier receipt and may duplicate or corrupt
            // an already-complete implementation. This marker is accepted
            // only from the kernel-authored error list above, never from LLM
            // prose.
            if kernel_recovery_directive(&result)
                == Some(crate::core::recovery::RecoveryDirective::Blocked)
            {
                task_execution_facts.apply_to(&mut result);
                self.event_bus
                    .emit(
                        task_iri,
                        "RECOVERY_BLOCKED",
                        "SA",
                        &serde_json::json!({
                            "reason": "ca_verification_evidence_missing",
                            "status": &result.status,
                            "automatic_retry": false,
                            "required_action": "inspect or explicitly rerun the named acceptance check",
                        })
                        .to_string(),
                    )
                    .await;
                final_result = Some(result);
                break;
            }

            // Verify-first: the AA's finish action hardcodes status "success" even when it
            // concluded full execution is needed, so that status is unusable here. Only treat
            // a verify-first cycle as complete when the AA verdict explicitly confirms the
            // task is already done; otherwise fall through to the retry loop which runs the
            // stored fallback_steps (full PDCA) on the next cycle.
            let needs_execution_after_verify =
                cycle_num == 0 && plan.verify_first && verify_aa_needs_execution(&result);

            let declared_recovery_route = kernel_recovery_route(&result);
            let declared_recovery = declared_recovery_route
                .as_ref()
                .map(|(directive, _)| *directive);
            let declared_local_recovery = matches!(
                declared_recovery,
                Some(crate::core::recovery::RecoveryDirective::RetryCa)
                    | Some(crate::core::recovery::RecoveryDirective::RetryDa)
            );
            let task_scope_failure = result.summary.contains("[Recovery] scope=Task")
                || result.summary.contains("scope=Task");
            // A kernel-owned local directive takes precedence over incidental
            // `scope=Task` text in accumulated model summaries. Conversely,
            // summary prose can never manufacture a local retry capability.
            let local_failure = result.status != "success"
                && !needs_execution_after_verify
                && (declared_local_recovery || !task_scope_failure);
            let decision_report = if result.status == "success" && !needs_execution_after_verify {
                crate::core::recovery::DecisionReport {
                    mode,
                    directive: crate::core::recovery::RecoveryDirective::Accept,
                    reason: crate::core::recovery::RecoveryReason::Accepted,
                    scope: crate::core::recovery::RepairScope::Task,
                    plan_revision,
                }
            } else if local_failure {
                let directive = declared_recovery
                    .filter(|directive| {
                        matches!(
                            directive,
                            crate::core::recovery::RecoveryDirective::RetryCa
                                | crate::core::recovery::RecoveryDirective::RetryDa
                        )
                    })
                    .unwrap_or(crate::core::recovery::RecoveryDirective::RetryDa);
                crate::core::recovery::DecisionReport {
                    mode,
                    directive,
                    reason: if directive == crate::core::recovery::RecoveryDirective::RetryCa {
                        crate::core::recovery::RecoveryReason::EvidenceMissing
                    } else {
                        crate::core::recovery::RecoveryReason::LocalExecutionGap
                    },
                    scope: if directive == crate::core::recovery::RecoveryDirective::RetryCa {
                        crate::core::recovery::RepairScope::Phase
                    } else {
                        crate::core::recovery::RepairScope::Step
                    },
                    plan_revision,
                }
            } else {
                crate::core::recovery::DecisionReport {
                    mode,
                    directive: crate::core::recovery::RecoveryDirective::ReplanPa,
                    reason: if result.summary.contains("Dimension Audit") {
                        crate::core::recovery::RecoveryReason::PlanInvalid
                    } else {
                        crate::core::recovery::RecoveryReason::LocalExecutionGap
                    },
                    scope: crate::core::recovery::RepairScope::Task,
                    plan_revision,
                }
            };
            self.event_bus
                .emit(
                    task_iri,
                    "AA_DECISION",
                    "SA",
                    &serde_json::to_string(&decision_report).unwrap_or_else(|_| "{}".to_string()),
                )
                .await;

            if result.status == "success" && !needs_execution_after_verify {
                let terminal_commands = self
                    .event_bus
                    .close_and_take_supplementary_commands(task_iri);
                if !terminal_commands.is_empty() {
                    // execute_plan normally closes and drains this boundary.
                    // Refuse a success if a future execution path ever returns
                    // without doing so.
                    result.status = "failed".to_string();
                    result.verdict = Some(TaskVerdict::Failed);
                    result.errors.push(format!(
                        "{} supplementary command(s) reached the terminal boundary without processing",
                        terminal_commands.len()
                    ));
                    for event in terminal_commands {
                        self.enqueue_supplementary_input(task_iri, &event.payload);
                    }
                    final_result = Some(result);
                    continue;
                }
                task_execution_facts.apply_to(&mut result);
                info!(task_iri = %task_iri, cycle_num = cycle_num + 1, "PDCA cycle passed");
                self.emit_sa_thought(
                    task_iri,
                    &format!("✅ PDCA cycle #{} passed — task complete", cycle_num + 1),
                    "pdca_cycle_passed",
                )
                .await;

                if let Some(scheduler) = &self.scheduler {
                    let _ = scheduler.on_task_complete(task_iri).await;
                }
                self.record_final_learning_outcome(
                    task_iri,
                    &result,
                    &policy_choice,
                    &learning_treatment,
                    task_started_at,
                    task_prompt_tokens_start,
                    task_completion_tokens_start,
                    task_context_quality_start,
                    effective_task_effect_policy.requires_workspace_mutation(),
                )
                .await;
                return Ok(result);
            }

            let last_cycle = cycle_num + 1 >= max_cycles;
            if last_cycle {
                info!(task_iri = %task_iri, cycle_num = cycle_num + 1, "All PDCA cycles exhausted");
                self.emit_sa_thought(
                    task_iri,
                    &format!(
                        "⚠️ All {} PDCA cycles completed without full pass — returning last result",
                        max_cycles
                    ),
                    "pdca_cycles_exhausted",
                )
                .await;
                final_result = Some(result);
                break;
            }

            if matches!(
                decision_report.directive,
                crate::core::recovery::RecoveryDirective::RetryDa
                    | crate::core::recovery::RecoveryDirective::RetryCa
            ) {
                if mode == crate::core::recovery::OrchestrationMode::Dag {
                    // DAG node-level retry/branch semantics are authoritative;
                    // rewriting and replaying the external graph would violate
                    // its topology and duplicate completed side effects.
                    final_result = Some(result);
                    break;
                }
                next_scoped_plan = scoped_retry_plan_for_decision(
                    &executed_plan,
                    &decision_report,
                    declared_recovery_route
                        .as_ref()
                        .map(|(_, failed_step)| failed_step.as_str()),
                );
                if next_scoped_plan.is_none() {
                    result.errors.push(format!(
                        "{:?} recovery could not identify its role-owned boundary",
                        decision_report.directive
                    ));
                    final_result = Some(result);
                    break;
                }
            }
            if decision_report.directive == crate::core::recovery::RecoveryDirective::ReplanPa
                && !needs_execution_after_verify
            {
                let max_plan_revisions = self
                    .runner
                    .agent_settings
                    .execution_budget
                    .max_plan_revisions;
                if task_scope_replans_used >= max_plan_revisions {
                    result.errors.push(format!(
                        "task-scope PA replan budget exhausted ({})",
                        max_plan_revisions
                    ));
                    final_result = Some(result);
                    break;
                }
                task_scope_replans_used = task_scope_replans_used.saturating_add(1);
            }

            // Build targeted feedback from the latest quality gate. The
            // failing gate can be CA (before AA is allowed to run) or AA; do
            // not misreport every retry as an AA rejection.
            // The CA→DA correction loop (in execute_plan) already handles in-cycle fixes;
            // this SA-level feedback addresses persistent failures requiring plan adjustment.
            // This summary becomes model input for a new PA/DA AgentInstance.
            // Keep stable AgentTurn references, but strip all result tools and
            // tool-result IRIs whose authority ended with the prior L1.
            let quality_gate_summary =
                crate::tools::tool_executor::sanitize_session_tool_references(&result.summary).0;
            let task_level_audit = quality_gate_summary.contains("[Recovery] scope=Task");
            let ca_recheck =
                decision_report.directive == crate::core::recovery::RecoveryDirective::RetryCa;
            let da_fixes = !ca_recheck
                && !task_level_audit
                && (quality_gate_summary.contains("Dimension Audit")
                    || quality_gate_summary.contains("execution failed")
                    || quality_gate_summary.contains("not completed"));

            let authoritative_contract = super::execution::authoritative_task_contract(
                user_input,
                &five_w2h,
                &effective_task_constraints,
            );
            cycle_feedback = Some(if ca_recheck {
                format!(
                    "{}\n\nPDCA Cycle #{} (plan revision {}): implementation is preserved; the latest CA lacked admissible verification evidence.\n\
                     Status: {}\n\n\
                     CA evidence gap:\n{}\n\n\
                     ---\n\
                     Re-run only the named missing acceptance checks in a fresh isolated CA. Do not route this gap to DA and do not modify workspace content.",
                    authoritative_contract,
                    cycle_num + 1,
                    plan_revision,
                    result.status,
                    quality_gate_summary
                )
            } else if da_fixes {
                format!(
                    "{}\n\nPDCA Cycle #{} (plan revision {}): the latest quality gate identified execution-level issues.\n\
                     Status: {}\n\n\
                     Quality-gate evidence:\n{}\n\n\
                     ---\n\
                     PRESERVE your previous plan structure. Focus ONLY on:\n\
                     1. Which specific execution steps failed or were incomplete\n\
                     2. What DA needs to do differently (more detail, different approach)\n\
                     3. Do NOT create a brand new plan — refine the existing one",
                    authoritative_contract,
                    cycle_num + 1,
                    plan_revision,
                    result.status,
                    quality_gate_summary
                )
            } else {
                format!(
                    "{}\n\nPDCA Cycle #{} (plan revision {}): result\nStatus: {}\nRecovery directive: replan_pa\n\n\
                     Quality-gate evidence:\n{}\n\n\
                     ---\n\
                     The previous plan is not sufficient. Re-plan the task at PA,\
                     preserve verified evidence where possible, and create an improved approach.",
                    authoritative_contract,
                    cycle_num + 1,
                    plan_revision,
                    result.status,
                    quality_gate_summary
                )
            });
            final_result = Some(result);
        }

        if let Some(scheduler) = &self.scheduler {
            let _ = scheduler.on_task_complete(task_iri).await;
        }
        let mut final_result = final_result.unwrap_or_else(|| TaskResult {
            task_iri: task_iri.to_string(),
            status: "failed".to_string(),
            summary: "All PDCA cycles exhausted without success".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 0,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            verdict: None,
            archive_iri: None,
        });
        let unprocessed_terminal_commands = self
            .event_bus
            .close_and_take_supplementary_commands(task_iri)
            .len();
        task_execution_facts.apply_to(&mut final_result);
        if unprocessed_terminal_commands > 0 {
            final_result.errors.push(format!(
                "{unprocessed_terminal_commands} supplementary command(s) arrived after the last executable recovery boundary"
            ));
        }
        self.record_final_learning_outcome(
            task_iri,
            &final_result,
            &policy_choice,
            &learning_treatment,
            task_started_at,
            task_prompt_tokens_start,
            task_completion_tokens_start,
            task_context_quality_start,
            effective_task_effect_policy.requires_workspace_mutation(),
        )
        .await;
        Ok(final_result)
    }

    async fn record_final_learning_outcome(
        &mut self,
        task_iri: &str,
        result: &TaskResult,
        policy_choice: &crate::core::policy_learning::PolicyChoice,
        treatment: &LearningTreatmentMetrics,
        task_started_at: std::time::Instant,
        prompt_tokens_start: u64,
        completion_tokens_start: u64,
        context_quality_start: crate::llm::interaction::LlmContextQualitySnapshot,
        workspace_mutation_required: bool,
    ) {
        let task_usage = self
            .runner
            .llm_interactions
            .usage_snapshot_for_scope(task_iri);
        let prompt_tokens = task_usage.prompt_tokens.saturating_sub(prompt_tokens_start);
        let completion_tokens = task_usage
            .completion_tokens
            .saturating_sub(completion_tokens_start);
        let context_quality = self
            .runner
            .llm_interactions
            .context_quality_snapshot_for_scope(task_iri)
            .saturating_sub(context_quality_start);
        let elapsed_ms = task_started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let reward = policy_reward_breakdown(
            result,
            prompt_tokens,
            elapsed_ms,
            workspace_mutation_required,
        );
        // A terminal status asserted by an LLM is not learning evidence on its
        // own.  The execution path stores this compact CA/AA record before it
        // returns here; if it is absent or not reusable, a claimed successful
        // task cannot promote a retrieval treatment.
        let audit_iri = task_audit_evidence_iri(task_iri);
        let independent_ca_aa_pass = self
            .runner
            .l0_store
            .retrieve(&audit_iri)
            .ok()
            .flatten()
            .and_then(|entry| {
                serde_json::from_str::<crate::core::policy_learning::TaskAuditKnowledgeEvidence>(
                    &entry.content,
                )
                .ok()
            })
            .is_some_and(|audit| audit.reusable_success());
        let terminal_success = matches!(result.status.as_str(), "success" | "completed");
        let successful_learning_evidence = !terminal_success || independent_ca_aa_pass;
        let policy_model_version_before = self.policy_learning.model_version();
        let mut evaluation = None;
        let evidence_iri = format!("iri://learning/evaluations/{}", uuid::Uuid::new_v4());
        let observation_evidence = crate::core::policy_learning::PolicyObservationEvidence {
            task_iri: Some(task_iri.to_string()),
            experiment_pair_id: treatment.experiment_pair_id.clone(),
            experiment_seed: treatment.experiment_seed.clone(),
            experiment_model: treatment.experiment_model.clone(),
            experiment_config_fingerprint: treatment.experiment_config_fingerprint.clone(),
            workspace_fingerprint: treatment.workspace_fingerprint.clone(),
            objective_fingerprint: Some(treatment.objective_fingerprint.clone()),
            orchestration_mode: Some(treatment.orchestration_mode.clone()),
        };
        let mut policy_observation_recorded = false;
        if self.learning_mode.updates_learning() && successful_learning_evidence {
            let task_result = serde_json::json!({
                "status": &result.status,
                "summary": &result.summary,
                "turn_count": result.turn_count,
                "tool_call_count": result.tool_call_count,
                "errors": &result.errors,
                "tracked_actions": &result.tracked_actions,
            });
            // A durable experience represents the terminal user task, not an
            // intermediate PA/DA/CA/AA BizAgent invocation.
            self.perception.on_task_end(&task_result, task_iri).await;
            match self.policy_learning.record_reward_gated_with_evidence(
                policy_choice,
                reward.total,
                self.policy_learning.gate(),
                observation_evidence,
            ) {
                Ok(report) => {
                    policy_observation_recorded = true;
                    evaluation = Some(report);
                }
                Err(error) => {
                    tracing::warn!(task_iri = %task_iri, %error, "Policy reward persistence failed")
                }
            }
        } else if matches!(
            self.learning_mode,
            crate::core::policy_learning::LearningMode::Baseline
        ) && successful_learning_evidence
        {
            // A baseline run remains a true behavioral ablation: no history
            // retrieval/injection, perception update, or model training. Its
            // immutable outcome is nevertheless required for a controlled
            // treatment-effect promotion gate.
            match self.policy_learning.record_baseline_evidence(
                policy_choice,
                reward.total,
                observation_evidence,
            ) {
                Ok(recorded) => policy_observation_recorded = recorded,
                Err(error) => tracing::warn!(
                    task_iri = %task_iri,
                    %error,
                    "Controlled baseline evidence persistence failed"
                ),
            }
        } else if terminal_success {
            tracing::info!(
                task_iri = %task_iri,
                "Skipped positive learning observation without independent CA/AA evidence"
            );
        }

        let mut trajectory_evidence_iris = vec![evidence_iri.clone()];
        if independent_ca_aa_pass {
            trajectory_evidence_iris.push(audit_iri.clone());
        }
        let trajectory = crate::core::learning_trajectory::LearningTrajectory {
            schema_version: crate::core::learning_trajectory::LEARNING_TRAJECTORY_SCHEMA_VERSION,
            task_iri: task_iri.to_string(),
            task_family: policy_choice.context.clone(),
            mode: self.learning_mode,
            policy_action: policy_choice.action.clone(),
            policy_candidates: policy_choice.candidates.clone(),
            policy_model_version: self.policy_learning.model_version(),
            policy_explored: policy_choice.explored,
            observed_skill_iris: treatment.skill_iris_observed.clone(),
            observed_knowledge_fragment_iris: treatment.knowledge_fragment_iris_observed.clone(),
            injected_skill_iris: treatment.skill_iris_injected.clone(),
            injected_knowledge_fragment_iris: treatment.knowledge_fragment_iris_injected.clone(),
            context_quality,
            evidence_iris: trajectory_evidence_iris,
            tool_steps: result
                .tracked_actions
                .iter()
                .map(crate::core::learning_trajectory::TrajectoryToolStep::from)
                .collect(),
            outcome: crate::core::learning_trajectory::LearningTrajectoryOutcome {
                terminal_status: result.status.clone(),
                reward: reward.total,
                prompt_tokens,
                completion_tokens,
                elapsed_ms,
                independent_ca_aa_pass,
            },
            created_at: chrono::Utc::now(),
        };
        let trajectory_iri = match self.learning_trajectories.persist(&trajectory) {
            Ok(crate::core::learning_trajectory::TrajectoryPersistResult::Stored { iri })
            | Ok(crate::core::learning_trajectory::TrajectoryPersistResult::AlreadyPresent {
                iri,
            }) => Some(iri),
            Err(error) => {
                tracing::warn!(task_iri = %task_iri, %error, "Learning trajectory persistence failed");
                None
            }
        };

        // The constrained policy remains the authoritative promotion gate.
        // This secondary lifecycle record makes every promoted version
        // inspectable and provides a one-way automatic freeze path.
        let mut evolution_delta_iri = None;
        if independent_ca_aa_pass {
            if let Some(policy_evaluation) = evaluation.as_ref().filter(|report| report.accepted) {
                if let (Some(action), Some(trajectory_iri)) = (
                    policy_evaluation.candidate_action.as_deref(),
                    trajectory_iri.as_deref(),
                ) {
                    if crate::core::retrieval_policy::RetrievalPolicyArm::parse(action).is_some_and(
                        |arm| arm != crate::core::retrieval_policy::RetrievalPolicyArm::Baseline,
                    ) {
                        let candidate_revision = self.policy_learning.model_version();
                        if candidate_revision > policy_model_version_before {
                            let base_revision = candidate_revision.saturating_sub(1);
                            match crate::core::evolution_delta_gate::EvolutionDelta::proposed_policy(
                                task_iri,
                                &policy_choice.context,
                                action,
                                base_revision,
                                candidate_revision,
                                vec![audit_iri.clone(), trajectory_iri.to_string()],
                            ) {
                                Ok(delta) => {
                                    let delta_id = delta.delta_id.clone();
                                    let delta_iri = delta.storage_iri();
                                    let lifecycle = self
                                    .evolution_gate
                                    .propose(&delta)
                                    .and_then(|_| {
                                        self.evolution_gate.transition(
                                            &delta_id,
                                            crate::core::evolution_delta_gate::EvolutionDeltaState::ShadowValidated,
                                            "paired baseline and candidate evidence accepted by policy gate",
                                            false,
                                        )
                                    })
                                    .and_then(|_| {
                                        self.evolution_gate.transition(
                                            &delta_id,
                                            crate::core::evolution_delta_gate::EvolutionDeltaState::Active,
                                            "constrained policy promotion applied the candidate revision",
                                            false,
                                        )
                                    });
                                    match lifecycle {
                                        Ok(_) => evolution_delta_iri = Some(delta_iri),
                                        Err(error) => tracing::warn!(
                                            task_iri = %task_iri,
                                            %error,
                                            "Evolution delta lifecycle persistence failed"
                                        ),
                                    }
                                }
                                Err(error) => tracing::warn!(
                                    task_iri = %task_iri,
                                    %error,
                                    "Refused invalid evolution delta"
                                ),
                            }
                        }
                    }
                }
            }
        }

        let mut health_report = None;
        let mut frozen_delta_ids = Vec::new();
        if self.learning_mode.updates_learning()
            && policy_observation_recorded
            && crate::core::retrieval_policy::RetrievalPolicyArm::parse(&policy_choice.action)
                .is_some_and(|arm| {
                    arm != crate::core::retrieval_policy::RetrievalPolicyArm::Baseline
                })
        {
            let failed_actions = result
                .tracked_actions
                .iter()
                .filter(|action| {
                    matches!(
                        action.status,
                        crate::core::tracked_action::ActionStatus::Failed
                    )
                })
                .count();
            let tool_failure_rate = if result.tracked_actions.is_empty() {
                0.0
            } else {
                failed_actions as f64 / result.tracked_actions.len() as f64
            };
            let observation = crate::core::learning_health::LearningHealthObservation {
                schema_version: crate::core::learning_health::LEARNING_HEALTH_SCHEMA_VERSION,
                task_iri: task_iri.to_string(),
                task_family: policy_choice.context.clone(),
                policy_action: policy_choice.action.clone(),
                policy_model_version: self.policy_learning.model_version(),
                metrics: vec![
                    crate::core::learning_health::HealthMetricValue {
                        name: "reward".into(),
                        value: reward.total as f64,
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "terminal_success".into(),
                        value: if terminal_success && independent_ca_aa_pass {
                            1.0
                        } else {
                            0.0
                        },
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "verified_evidence".into(),
                        value: if independent_ca_aa_pass { 1.0 } else { 0.0 },
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "tool_failure_rate".into(),
                        value: tool_failure_rate,
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "elapsed_ms".into(),
                        value: elapsed_ms as f64,
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "total_tokens".into(),
                        value: prompt_tokens.saturating_add(completion_tokens) as f64,
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "context_drop_rate".into(),
                        value: context_quality.drop_rate(),
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "context_truncate_rate".into(),
                        value: context_quality.truncate_rate(),
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "context_expired_rate".into(),
                        value: context_quality.expired_rate(),
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "context_required_budget_overflow_rate".into(),
                        value: context_quality.required_budget_overflow_rate(),
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "context_chars_per_dispatch".into(),
                        value: context_quality.chars_per_dispatch(),
                    },
                ],
                created_at: chrono::Utc::now(),
            };
            match self.learning_health.record_and_assess(&observation) {
                Ok((_, report)) => {
                    if let Some(reason) = report.freeze_reason() {
                        match self
                            .policy_learning
                            .freeze_context(&policy_choice.context, &reason)
                        {
                            Ok(_) => {}
                            Err(error) => tracing::warn!(
                                task_iri = %task_iri,
                                %error,
                                "Policy safety freeze persistence failed"
                            ),
                        }
                        match self
                            .evolution_gate
                            .freeze_active_retrieval_family(&policy_choice.context, &reason)
                        {
                            Ok(deltas) => {
                                frozen_delta_ids =
                                    deltas.into_iter().map(|delta| delta.delta_id).collect()
                            }
                            Err(error) => tracing::warn!(
                                task_iri = %task_iri,
                                %error,
                                "Evolution delta freeze persistence failed"
                            ),
                        }
                    }
                    health_report = Some(report);
                }
                Err(error) => tracing::warn!(
                    task_iri = %task_iri,
                    %error,
                    "Learning health observation persistence failed"
                ),
            }
        }

        let evidence = serde_json::json!({
            "@id": format!("{}#learning-evaluation", task_iri),
            "@type": "LearningEvaluation",
            "task_iri": task_iri,
            "mode": self.learning_mode,
            "policy_context": policy_choice.context,
            "policy_action": policy_choice.action,
            "policy_explored": policy_choice.explored,
            "policy_deployment": if policy_choice.explored {
                "candidate_exploration"
            } else if self.policy_learning.model_version() > 0 && !policy_choice.used_fallback {
                "promoted_model"
            } else {
                "rule_baseline"
            },
            "treatment": treatment,
            "status": result.status,
            "turn_count": result.turn_count,
            "tool_call_count": result.tool_call_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "context_quality": context_quality,
            "elapsed_ms": elapsed_ms,
            "reward": reward,
            "policy_evaluation": evaluation,
            "policy_observation_recorded": policy_observation_recorded,
            "independent_ca_aa_pass": independent_ca_aa_pass,
            "learning_trajectory_iri": trajectory_iri,
            "evolution_delta_iri": evolution_delta_iri,
            "learning_health": health_report,
            "frozen_evolution_delta_ids": frozen_delta_ids,
            "policy_gate": self.policy_learning.gate(),
            "candidate_trial_min_baseline_samples": self.policy_learning.min_observations(),
            "model_version": self.policy_learning.model_version(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        if let Err(error) = self
            .runner
            .l0_store
            .store(&evidence_iri, &evidence.to_string())
        {
            tracing::warn!(task_iri = %task_iri, %error, "Learning evaluation persistence failed");
        }
        self.event_bus
            .emit(task_iri, "LEARNING_OUTCOME", "SA", &evidence.to_string())
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_contract_preserves_exact_user_quantity_and_boundary() {
        let original = "Create at least 10 test cases; do not require 10 test files.";
        let mut five_w2h = crate::core::five_w2h::Task5W2H::new(original, "verify behavior");
        five_w2h.why.success_criteria = vec!["at least 10 test cases pass".to_string()];

        let contract = crate::core::sa::execution::authoritative_task_contract(
            original,
            &five_w2h,
            &std::collections::HashMap::new(),
        );
        assert!(contract.contains(original));
        assert!(contract.contains("at least 10 test cases pass"));
        assert!(contract.contains("must not add, remove, strengthen, weaken, or reinterpret"));
    }

    #[test]
    fn direct_response_contract_does_not_require_a_file_or_graph_node() {
        let original = "输出一份 Markdown 调研报告";
        let five_w2h = crate::core::five_w2h::Task5W2H::new(original, "供用户阅读");
        let constraints = std::collections::HashMap::from([(
            crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT.to_string(),
            crate::core::agent_runner::DELIVERY_MODE_DIRECT_RESPONSE.to_string(),
        )]);
        let contract = crate::core::sa::execution::authoritative_task_contract(
            original,
            &five_w2h,
            &constraints,
        );
        assert!(contract.contains("direct_response"));
        assert!(contract.contains("filesystem path"));
        assert!(contract.contains("invented graph IRI"));
    }

    #[test]
    fn test_dedup_hints_removes_duplicates_preserving_order() {
        // Given: hints with interleaved duplicates from perception + skill discovery
        let hints = vec![
            "skill:calculator".to_string(),
            "workspace_event:main.rs modified".to_string(),
            "skill:calculator".to_string(),
            "scenario:debug loop".to_string(),
        ];
        // When: deduplicated with generous cap
        let deduped = dedup_hints(hints, 10, 700, 6_000);
        // Then: first occurrences kept, order preserved, count reduced
        assert_eq!(
            deduped,
            vec![
                "skill:calculator".to_string(),
                "workspace_event:main.rs modified".to_string(),
                "scenario:debug loop".to_string(),
            ]
        );
    }

    #[test]
    fn test_dedup_hints_respects_cap() {
        // Given: 5 distinct hints and a cap of 3
        let hints = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        // When: deduplicated with cap 3
        let deduped = dedup_hints(hints, 3, 700, 6_000);
        // Then: only the first 3 distinct hints remain
        assert_eq!(
            deduped,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_dedup_hints_empty_input() {
        // Given: empty hints
        let hints: Vec<String> = Vec::new();
        // When: deduplicated
        let deduped = dedup_hints(hints, 5, 700, 6_000);
        // Then: empty output
        assert!(deduped.is_empty());
    }

    #[test]
    fn policy_arms_require_real_treatment_material() {
        assert_eq!(eligible_policy_candidates(0, 0, 0), vec!["baseline"]);
        assert_eq!(
            eligible_policy_candidates(0, 0, 2),
            vec!["baseline", "experience_first"]
        );
        assert_eq!(
            eligible_policy_candidates(2, 0, 0),
            vec!["baseline", "skill_first"]
        );
        assert_eq!(
            eligible_policy_candidates(0, 1, 0),
            vec!["baseline", "knowledge_first"]
        );
        assert_eq!(
            eligible_policy_candidates(1, 1, 1),
            vec![
                "baseline",
                "knowledge_first",
                "experience_first",
                "skill_first"
            ]
        );
    }

    #[test]
    fn policy_treatments_are_distinct_and_baseline_is_a_true_ablation() {
        let experience = vec!["experience".to_string()];
        let skills = vec!["skill".to_string()];
        let knowledge = vec!["knowledge".to_string()];
        assert!(policy_treatment_hints(
            "baseline",
            &experience,
            &skills,
            &knowledge,
            20,
            700,
            6_000
        )
        .is_empty());
        assert_eq!(
            policy_treatment_hints(
                "experience_first",
                &experience,
                &skills,
                &knowledge,
                20,
                700,
                6_000,
            ),
            vec!["experience", "knowledge", "skill"]
        );
        assert_eq!(
            policy_treatment_hints(
                "knowledge_first",
                &experience,
                &skills,
                &knowledge,
                20,
                700,
                6_000,
            ),
            vec!["knowledge", "experience", "skill"]
        );
        assert_eq!(
            policy_treatment_hints(
                "skill_first",
                &experience,
                &skills,
                &knowledge,
                20,
                700,
                6_000,
            ),
            vec!["skill", "experience", "knowledge"]
        );
    }

    #[test]
    fn injected_skill_and_knowledge_provenance_tracks_budgeted_records_not_text_matches() {
        let knowledge = vec![
            LearningHintCandidate {
                text: "same rendered hint".into(),
                source: LearningHintSource::Knowledge {
                    iri: "iri://knowledge/first".into(),
                },
            },
            LearningHintCandidate {
                text: "same rendered hint".into(),
                source: LearningHintSource::Knowledge {
                    iri: "iri://knowledge/deduplicated".into(),
                },
            },
        ];
        let skills = vec![LearningHintCandidate {
            text: "skill hint".into(),
            source: LearningHintSource::Skill {
                iri: "iri://skills/budgeted-out".into(),
            },
        }];
        let treatment = materialize_policy_treatment(
            "knowledge_first",
            &[],
            &skills,
            &knowledge,
            1,
            700,
            6_000,
        );
        assert_eq!(treatment.hints, vec!["same rendered hint"]);
        assert_eq!(
            treatment.knowledge_fragment_iris,
            vec!["iri://knowledge/first"]
        );
        assert!(treatment.skill_iris.is_empty());
        assert!(!treatment
            .knowledge_fragment_iris
            .contains(&"iri://knowledge/deduplicated".to_string()));
    }

    #[test]
    fn verify_first_clones_llm_ca_aa_and_preserves_exact_interaction_provenance() {
        let make_step = |step_id: &str, role: AgentRole, dependency: Option<&str>| PlanStep {
            step_id: step_id.to_string(),
            role,
            objective: format!("model objective for {role}"),
            expected_output: format!("model output for {role}"),
            dependencies: dependency.into_iter().map(str::to_string).collect(),
            tools_allowed: Vec::new(),
            success_criteria: format!("model criterion for {role}"),
            work_packages: Vec::new(),
            branch_on_failure: false,
            branch_fallback: None,
            retry_count: 0,
            retry_delay_secs: 0,
            effect_policy: match role {
                AgentRole::Plan | AgentRole::Check => {
                    crate::core::effect::EffectPolicy::EvidenceOnly
                }
                AgentRole::Act => crate::core::effect::EffectPolicy::DecisionOnly,
                AgentRole::Do => crate::core::effect::EffectPolicy::None,
            },
        };
        let mut plan = ExecutionPlan {
            plan_id: "llm-plan".to_string(),
            agent_sequence: vec![
                AgentRole::Plan,
                AgentRole::Do,
                AgentRole::Check,
                AgentRole::Act,
            ],
            parallel_groups: Vec::new(),
            task_complexity: TaskComplexity::Standard,
            description: "model plan".to_string(),
            steps: vec![
                make_step("pa", AgentRole::Plan, None),
                make_step("da", AgentRole::Do, Some("pa")),
                make_step("ca", AgentRole::Check, Some("da")),
                make_step("aa", AgentRole::Act, Some("ca")),
            ],
            agent_spec_provenance: None,
            context_requirements: std::collections::HashMap::new(),
            success_metrics: vec!["model metric".to_string()],
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: Vec::new(),
        };
        plan.set_agent_spec_provenance(crate::core::context_model::ExecutionPlanProvenance::new(
            AgentSpecSourceRecord::new(AgentSpecSourceKind::LlmGeneratedPlan)
                .with_source_ref("iri://task/verify-first#llm-plan")
                .with_producer("SupervisorAgent.plan_generation")
                .with_model("test-model")
                .with_interaction_id("llm-plan-interaction"),
        ))
        .unwrap();
        let original_ca = plan.steps[2].clone();
        let original_aa = plan.steps[3].clone();

        materialize_verify_first_from_llm_plan(&mut plan, "iri://task/verify-first").unwrap();

        assert!(plan.verify_first);
        assert_eq!(plan.fallback_steps.len(), 4);
        assert_eq!(
            plan.steps.iter().map(|step| step.role).collect::<Vec<_>>(),
            vec![AgentRole::Check, AgentRole::Act]
        );
        assert_eq!(plan.steps[0].objective, original_ca.objective);
        assert_eq!(plan.steps[0].expected_output, original_ca.expected_output);
        assert_eq!(plan.steps[0].success_criteria, original_ca.success_criteria);
        assert_eq!(plan.steps[1].objective, original_aa.objective);
        assert!(plan.steps[0].dependencies.is_empty());
        assert_eq!(plan.steps[1].dependencies, vec!["verify_first_ca"]);
        for step in &plan.steps {
            let source = plan
                .agent_spec_source_for_step(&step.step_id)
                .unwrap()
                .unwrap();
            assert_eq!(source.kind, AgentSpecSourceKind::LlmGeneratedPlan);
            assert_eq!(source.model.as_deref(), Some("test-model"));
            assert_eq!(
                source.interaction_id.as_deref(),
                Some("llm-plan-interaction")
            );
        }
    }

    #[test]
    fn policy_context_ignores_order_and_volatile_numbers() {
        assert_eq!(
            crate::core::policy_learning::learning_policy_context(
                "Create report 20260825 from JSON data"
            ),
            crate::core::policy_learning::learning_policy_context(
                "JSON data create report 99117 from"
            )
        );
    }

    #[test]
    fn policy_context_is_stable_for_numbered_matched_artifacts() {
        assert_eq!(
            crate::core::policy_learning::learning_policy_context(
                "Create probe/1.txt with exact bytes"
            ),
            crate::core::policy_learning::learning_policy_context(
                "Create probe/987.txt with exact bytes"
            )
        );
    }

    #[test]
    fn reward_distinguishes_clean_success_from_costly_rework() {
        let clean = TaskResult {
            task_iri: "iri://task/clean".into(),
            status: "success".into(),
            verdict: None,
            summary: "accepted".into(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 5,
            tool_call_count: 2,
            five_w2h_updates: None,
            tracked_actions: vec![],
            archive_iri: None,
        };
        let mut costly = clean.clone();
        costly.turn_count = 20;
        costly.errors = vec!["retry one".into(), "retry two".into()];
        costly.summary = "[Recovery] scope=Task; eventually accepted".into();

        let clean_reward = policy_reward_breakdown(&clean, 10_000, 10_000, true);
        let costly_reward = policy_reward_breakdown(&costly, 180_000, 240_000, true);
        assert_eq!(clean_reward.total, 1.0);
        assert!(costly_reward.total < clean_reward.total);
        assert!(costly_reward.excess_turn_penalty > 0.0);
        assert!(costly_reward.recovery_penalty > 0.0);
        assert!(costly_reward.prompt_token_penalty > 0.0);
        assert!(costly_reward.latency_penalty > 0.0);
    }

    #[test]
    fn evidence_only_success_is_not_penalized_for_having_no_mutation_tail() {
        let mut tracker =
            crate::core::tracked_action::ActionTracker::new("iri://task/read-only", "CA");
        tracker.record(
            "file_read",
            &serde_json::json!({"path": "fixture.txt"}),
            &serde_json::json!({"success": true, "content": "ANSWER=helios-731"}),
            0.01,
        );
        let result = TaskResult {
            task_iri: "iri://task/read-only".into(),
            status: "success".into(),
            verdict: None,
            summary: "ANSWER=helios-731".into(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 2,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: tracker.actions,
            archive_iri: None,
        };
        let evidence_only = policy_reward_breakdown(&result, 1_000, 1_000, false);
        let mutation_required = policy_reward_breakdown(&result, 1_000, 1_000, true);
        assert_eq!(evidence_only.no_effect_tail, 1);
        assert_eq!(evidence_only.no_effect_tail_penalty, 0.0);
        assert!(mutation_required.no_effect_tail_penalty > 0.0);
        assert!(evidence_only.total > mutation_required.total);
    }

    #[test]
    fn structured_knowledge_is_family_scoped_and_token_bounded() {
        let mut matched = crate::skill_graph::types::KnowledgeFragment::new(
            "iri://fragment/matched",
            "iri://skill/app",
            "same family",
            "reuse",
        );
        matched.kind = "ca_validated_task_knowledge".into();
        matched.task_family = Some("family:a".into());
        matched.ca_verdict = Some("pass".into());
        matched.evidence_count = 2;
        matched.success_count = 2;
        matched.successful_checks = vec!["test passed".into()];
        let mut unrelated = matched.clone();
        unrelated.fragment_iri = "iri://fragment/unrelated".into();
        unrelated.task_family = Some("family:b".into());

        let task_context = LearningTaskContext {
            family: "family:a".into(),
            operations: vec![],
            modalities: vec![],
            raw_features: vec!["same".into(), "family".into()],
        };
        let ranked = rank_knowledge_fragments(vec![unrelated, matched], &task_context, 8);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].fragment_iri, "iri://fragment/matched");

        let owner_context = learning_task_context(
            "Extend the existing Python task queue with owner filtering and run tests",
        );
        let mut compatible_legacy = crate::skill_graph::types::KnowledgeFragment::new(
            "iri://fragment/legacy-tags",
            "iri://skill/app",
            "Add normalized tags to the existing Python task queue and run tests",
            "reuse with verification",
        );
        compatible_legacy.kind = "ca_validated_task_knowledge".into();
        compatible_legacy.task_family =
            Some("planning:v2:ops=build+operate+test+write;kinds=code+data".into());
        let mut wrong_feature = compatible_legacy.clone();
        wrong_feature.fragment_iri = "iri://fragment/unrelated-software".into();
        wrong_feature.description =
            "Modify a Rust network proxy certificate loader and benchmark TLS".into();
        let ranked =
            rank_knowledge_fragments(vec![wrong_feature, compatible_legacy], &owner_context, 8);
        assert_eq!(
            ranked
                .iter()
                .map(|fragment| fragment.fragment_iri.as_str())
                .collect::<Vec<_>>(),
            vec!["iri://fragment/legacy-tags"]
        );

        let max_hint_chars = 700;
        let max_hints = 20;
        let max_total_chars = 6_000;
        let oversized = vec!["x".repeat(max_hint_chars * 2); max_hints * 2];
        let bounded = dedup_hints(oversized, max_hints, max_hint_chars, max_total_chars);
        assert!(bounded
            .iter()
            .all(|hint| hint.chars().count() <= max_hint_chars + 1));
        assert!(
            bounded
                .iter()
                .map(|hint| hint.chars().count())
                .sum::<usize>()
                <= max_total_chars
        );
    }
}
