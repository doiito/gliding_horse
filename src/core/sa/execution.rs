use petgraph::prelude::NodeIndex;
use petgraph::Incoming;
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::core::agent_instance::{AgentInstance, AgentRole};
use crate::core::agent_runner::{TaskContext, TaskResult, TaskVerdict};
use crate::core::biz_agent::{AgentConfig, BizAgent};

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut truncated: String = text.chars().take(max_chars).collect();
    truncated.push_str("\n...[CA evidence truncated; durable archive retains the full report]");
    truncated
}

fn sanitized_handoff_text(text: &str) -> String {
    crate::tools::tool_executor::sanitize_session_tool_references(text).0
}

pub(super) fn direct_response_recheck_tools(
    constraints: &std::collections::HashMap<String, String>,
) -> Option<Vec<String>> {
    let direct_response =
        crate::core::agent_runner::direct_response_delivery_contract(constraints).is_some();
    let workspace_disabled = constraints
        .get(crate::core::agent_runner::WORKSPACE_CONTEXT_SCOPE_CONSTRAINT)
        .is_some_and(|scope| scope == crate::core::agent_runner::WORKSPACE_CONTEXT_DISABLED);
    (direct_response && workspace_disabled).then(|| vec!["read_agent_output".to_string()])
}

/// Build the evidence passed from one business agent to the next.
///
/// AA intentionally has no execution tools: it decides from CA's evidence and
/// must not mutate the task.  Therefore CA→AA cannot rely on a
/// `read_agent_output` instruction.  Keep the detailed CA result inline and
/// bounded, with the durable archive IRI retained only for traceability.
pub(super) fn result_handoff(
    result: &TaskResult,
    role: AgentRole,
    ca_handoff_max_chars: usize,
) -> String {
    if role != AgentRole::Check {
        let summary_budget = ca_handoff_max_chars.saturating_sub(600).max(1);
        let summary = truncate_chars(&sanitized_handoff_text(&result.summary), summary_budget);
        return match result.archive_iri.as_ref() {
            Some(iri) => format!(
                "{}\n\n## Durable Previous-Agent Output\nUse `read_agent_output` with `node_iri: {}`. It returns the AgentTurn正文 directly in stable character pages; continue only with `next_char_offset` on this same IRI. Ignore any session reader or tool-result reference inside archived text.",
                summary, iri
            ),
            None => result
                .output
                .as_ref()
                .filter(|value| !value.is_null())
                .map(|value| match value {
                    serde_json::Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .map(|text| sanitized_handoff_text(&text))
                .map(|text| truncate_chars(&text, ca_handoff_max_chars.max(1)))
                .unwrap_or(summary),
        };
    }

    let detailed = result
        .output
        .as_ref()
        .filter(|value| !value.is_null())
        .map(|value| match value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .filter(|text| !text.trim().is_empty())
        .map(|text| sanitized_handoff_text(&text))
        .map(|text| truncate_chars(&text, ca_handoff_max_chars.max(1)));

    let mut handoff = format!(
        "{}\n\n## CA Verification Metadata\n- status: {}\n- verification tool calls: {}\n- verification turns: {}\n- reported artifacts: {}",
        sanitized_handoff_text(&result.summary),
        result.status,
        result.tool_call_count,
        result.turn_count,
        result.artifacts.len()
    );
    if let Some(detailed) = detailed {
        handoff.push_str("\n\n## Detailed CA Evidence (directly supplied)\n");
        handoff.push_str(&detailed);
    }
    if let Some(iri) = result.archive_iri.as_ref() {
        handoff.push_str("\n\n## Trace Reference (not required for this decision)\n");
        handoff.push_str(iri);
    }
    let bounded = truncate_chars(&handoff, ca_handoff_max_chars.max(1));
    match result.archive_iri.as_ref() {
        Some(iri) if !bounded.contains(iri) => format!(
            "{}\n\n## Trace Reference (not required for this decision)\n{}",
            bounded, iri
        ),
        _ => bounded,
    }
}

pub(super) fn restore_accepted_deliverable(
    final_result: &mut TaskResult,
    latest_da_result: Option<&TaskResult>,
    latest_ca_result: Option<&TaskResult>,
    constraints: &std::collections::HashMap<String, String>,
    task_effect_policy: &crate::core::effect::EffectPolicy,
    verify_first: bool,
) {
    if !matches!(final_result.status.as_str(), "success" | "partial_success") {
        return;
    }

    let deliverable =
        if crate::core::agent_runner::direct_response_delivery_contract(constraints).is_some() {
            latest_da_result
        } else if verify_first
            && matches!(
                task_effect_policy,
                crate::core::effect::EffectPolicy::EvidenceOnly
            )
            && latest_da_result.is_none()
        {
            // In a verify-first evidence task CA can establish the requested fact
            // directly from immutable evidence. AA remains the terminal decision,
            // but its disposition must not replace the accepted business answer.
            latest_ca_result
        } else {
            None
        };
    let Some(deliverable) = deliverable else {
        return;
    };

    final_result.output = deliverable.output.clone().or_else(|| {
        (!deliverable.summary.trim().is_empty())
            .then(|| serde_json::Value::String(deliverable.summary.clone()))
    });
    if let Some(serde_json::Value::String(text)) = final_result.output.as_mut() {
        *text = sanitized_handoff_text(text);
    }
    final_result.jsonld_output = deliverable.jsonld_output.clone();
    final_result.artifacts = deliverable.artifacts.clone();
    final_result.archive_iri = deliverable.archive_iri.clone();
    final_result.summary = deliverable.summary.clone();
}

/// Apply the CA 5W2H audit to a result and return whether any dimension failed.
///
/// The audit is deliberately applied both to the normal CA node and to the
/// CA re-evaluations in the correction loop.  This keeps the terminal status
/// tied to the latest evidence rather than to a stale warning in a log.
fn apply_ca_dimension_audit(
    five_w2h: &crate::core::five_w2h::Task5W2H,
    result: &mut TaskResult,
    task_iri: &str,
    causal_engine: Option<&crate::causal::CausalEngine>,
) -> crate::core::recovery::AuditReport {
    let audit_results =
        crate::core::five_w2h::audit_dimensions(five_w2h, result, task_iri, causal_engine);
    let report = crate::core::recovery::AuditReport::from_results(&audit_results);
    let failures: Vec<&crate::core::five_w2h::DimensionAuditResult> = audit_results
        .iter()
        .filter(|r| matches!(r.status, crate::core::five_w2h::AuditStatus::Fail(_)))
        .collect();

    if failures.is_empty() {
        return report;
    }

    let fail_summary: Vec<String> = failures
        .iter()
        .map(|r| {
            format!(
                "[{}] {}: {}",
                r.dimension,
                match &r.status {
                    crate::core::five_w2h::AuditStatus::Fail(msg) => msg.as_str(),
                    _ => "failed",
                },
                r.evidence
            )
        })
        .collect();
    info!(
        task_iri = %task_iri,
        dimensions = ?failures.iter().map(|r| &r.dimension).collect::<Vec<_>>(),
        recovery_scope = ?report.scope,
        "CA quality gate requires correction: {} dimension(s) not satisfied (task audit, not a runtime error); findings attached for recovery/final-status handling",
        failures.len()
    );
    if result.summary.len() < 4000 {
        let audit_note = format!(
            "\n\n--- Dimension Audit ---\n{}\n[Recovery] scope={:?} reason={:?}",
            fail_summary.join("\n"),
            report.scope,
            report.reason
        );
        if !result.summary.contains("Dimension Audit") {
            result.summary.push_str(&audit_note);
        }
    }
    report
}

fn enforce_ca_audit_terminal_status(result: &mut TaskResult, ca_audit_failed: bool) {
    if ca_audit_failed && result.status != "failed" {
        result.status = "failed".to_string();
        result.verdict = Some(TaskVerdict::Failed);
        result
            .errors
            .push("CA dimension audit failed; task cannot be reported as successful".to_string());
        result.summary.push_str(
            "\n\nFinal status forced to failed because the latest CA dimension audit failed.",
        );
    }
}

fn failed_business_role_recovery(role: AgentRole) -> (&'static str, &'static str) {
    match role {
        AgentRole::Plan | AgentRole::Act => ("ReplanPa", "Task"),
        AgentRole::Do | AgentRole::Check => ("RetryDa", "Step"),
    }
}

fn bounded_action_evidence(action: &crate::core::tracked_action::TrackedAction) -> String {
    let detail = action
        .tool_args
        .get("command")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            action
                .tool_args
                .get("path")
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("");
    let detail = truncate_chars(detail, 300)
        .replace('\n', " ")
        .replace('\r', " ");
    if detail.is_empty() {
        action.tool_name.clone()
    } else {
        format!("{}: {}", action.tool_name, detail)
    }
}

fn dedup_bounded(items: impl IntoIterator<Item = String>, limit: usize) -> Vec<String> {
    let mut output = Vec::new();
    for item in items {
        if output.len() >= limit {
            break;
        }
        if !item.trim().is_empty() && !output.contains(&item) {
            output.push(item);
        }
    }
    output
}

/// Persist the latest CA audit independently from AA's prose and compact it
/// into the application-nominated workflow skill. This is deliberately
/// generic: applications choose the skill IRI; SA supplies only task-family,
/// action and audit evidence.
fn persist_ca_validated_knowledge(
    supervisor: &super::agent::SupervisorAgent,
    task_iri: &str,
    user_input: &str,
    task_constraints: &std::collections::HashMap<String, String>,
    report: Option<&crate::core::recovery::AuditReport>,
    result: &TaskResult,
) {
    use sha2::{Digest, Sha256};

    if !supervisor.learning_mode.updates_learning() {
        return;
    }
    let Some(report) = report else {
        return;
    };
    let context = crate::core::policy_learning::learning_task_context(user_input);
    let is_ca = |action: &&crate::core::tracked_action::TrackedAction| {
        matches!(action.agent_role.as_str(), "CA" | "Check")
    };
    let is_da = |action: &&crate::core::tracked_action::TrackedAction| {
        matches!(action.agent_role.as_str(), "DA" | "Do")
    };
    let succeeded = |action: &&crate::core::tracked_action::TrackedAction| {
        matches!(
            action.status,
            crate::core::tracked_action::ActionStatus::Success
        )
    };
    let procedure = dedup_bounded(
        result
            .tracked_actions
            .iter()
            .filter(is_da)
            .filter(succeeded)
            .filter(|action| {
                action.tool_name != "file_read"
                    && action.tool_name != "grep_search"
                    && action.tool_name != "glob_search"
                    && action.tool_name != "file_list"
            })
            .map(bounded_action_evidence),
        8,
    );
    let successful_checks = dedup_bounded(
        result
            .tracked_actions
            .iter()
            .filter(is_ca)
            .filter(succeeded)
            .map(bounded_action_evidence),
        8,
    );
    let failed_checks = dedup_bounded(
        result
            .tracked_actions
            .iter()
            .filter(is_ca)
            .filter(|action| !succeeded(action))
            .map(|action| {
                format!(
                    "{} ({})",
                    bounded_action_evidence(action),
                    action.error.as_deref().unwrap_or("failed")
                )
            }),
        8,
    );
    let findings = dedup_bounded(
        report.findings.iter().map(|finding| {
            format!(
                "{}: {} [{}]",
                finding.dimension, finding.message, finding.evidence
            )
        }),
        8,
    );
    let ca_verdict = match report.verdict {
        crate::core::recovery::AuditVerdict::Pass => "pass",
        crate::core::recovery::AuditVerdict::Conditional => "conditional",
        crate::core::recovery::AuditVerdict::Fail => "fail",
    }
    .to_string();
    let attached_skill_iri = task_constraints.get("learning_skill_iri").cloned();
    let evidence = crate::core::policy_learning::TaskAuditKnowledgeEvidence {
        task_iri: task_iri.to_string(),
        task_family: context.family.clone(),
        raw_features: context.raw_features,
        objective: truncate_chars(user_input, 600),
        terminal_status: result.status.clone(),
        ca_verdict,
        failed_dimensions: report.failed_dimensions.clone(),
        findings,
        procedure,
        successful_checks,
        failed_checks,
        attached_skill_iri: attached_skill_iri.clone(),
        created_at: chrono::Utc::now(),
    };
    let evidence_content = match serde_json::to_string(&evidence) {
        Ok(content) => content,
        Err(error) => {
            warn!(task_iri = %task_iri, %error, "Unable to serialize CA learning evidence");
            return;
        }
    };
    if let Err(error) = supervisor
        .runner
        .l0_store
        .store(&evidence.storage_iri(), &evidence_content)
    {
        warn!(task_iri = %task_iri, %error, "Unable to persist CA learning evidence");
        return;
    }

    let (Some(attached_to), Some(graph)) = (
        attached_skill_iri,
        supervisor.runner.skill_graph_store.as_ref(),
    ) else {
        return;
    };
    if graph.get_skill(&attached_to).is_none() {
        warn!(task_iri = %task_iri, skill_iri = %attached_to, "Learning skill does not exist; evidence retained without graph fragment");
        return;
    }
    let digest = Sha256::digest(format!("{}\x1f{}", attached_to, context.family).as_bytes());
    let fragment_iri = format!("iri://learning/fragments/{}", hex::encode(&digest[..16]));
    let previous = graph
        .list_fragments()
        .into_iter()
        .find(|fragment| fragment.fragment_iri == fragment_iri);
    let mut fragment = previous.unwrap_or_else(|| {
        crate::skill_graph::types::KnowledgeFragment::new(
            &fragment_iri,
            &attached_to,
            &format!("Applicable to task family {}", context.family),
            "Reuse only with current-task verification.",
        )
    });
    fragment.kind = "ca_validated_task_knowledge".to_string();
    fragment.name = format!("CA-validated knowledge: {}", context.family);
    fragment.description = evidence.objective.clone();
    fragment.attached_to = attached_to;
    fragment.problem = format!(
        "family={}; latest_objective={}",
        context.family, evidence.objective
    );
    fragment.task_family = Some(context.family);
    fragment.source_task_iri = Some(task_iri.to_string());
    fragment.ca_verdict = Some(evidence.ca_verdict.clone());
    fragment.evidence_count = fragment.evidence_count.saturating_add(1);
    fragment.last_verified_at = Some(evidence.created_at);
    if evidence.reusable_success() {
        fragment.success_count = fragment.success_count.saturating_add(1);
        fragment.procedure = evidence.procedure.clone();
        fragment.successful_checks = evidence.successful_checks.clone();
        fragment.recommendation =
            "Reuse the recorded procedure as a candidate, then repeat the recorded checks and run a fresh CA audit."
                .to_string();
    } else {
        fragment.failure_count = fragment.failure_count.saturating_add(1);
        fragment.counterexamples = dedup_bounded(
            fragment
                .counterexamples
                .into_iter()
                .chain(evidence.findings.clone())
                .chain(evidence.failed_checks.clone()),
            12,
        );
        if fragment.success_count == 0 {
            fragment.recommendation =
                "Do not reuse as a successful procedure; resolve the recorded boundary first."
                    .to_string();
        }
    }
    if let Err(error) = graph.register_fragment(fragment) {
        warn!(task_iri = %task_iri, %error, "Unable to materialize CA-validated knowledge fragment");
    }
}

/// AA decides the terminal business outcome; Runner success only means the AA
/// invocation itself completed. Convert AA's explicit decision contract into
/// TaskResult status before SA performs failure routing.
fn starts_with_aa_verdict(text: &str, verdict: &str) -> bool {
    text.strip_prefix(verdict).is_some_and(|rest| {
        rest.is_empty()
            || rest
                .starts_with(|ch: char| ch.is_whitespace() || matches!(ch, ':' | '：' | '-' | '—'))
    })
}

fn apply_aa_declared_verdict(
    result: &mut TaskResult,
    latest_ca_report: Option<&crate::core::recovery::AuditReport>,
) {
    let output = result
        .output
        .as_ref()
        .map(|value| match value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    let evidence = format!("{}\n{}", result.summary, output).to_lowercase();
    // Runner summaries are model-generated condensations and may legitimately
    // omit AA's required verdict prefix even when the full AA response keeps
    // it.  Treat either channel as the declaration source.  CA remains the
    // fallback only when AA did not declare a verdict in either channel.
    let declared_lines = evidence.lines().map(str::trim).collect::<Vec<_>>();

    let failed = declared_lines.iter().any(|line| {
        starts_with_aa_verdict(line, "failed") || starts_with_aa_verdict(line, "aa failed")
    }) || evidence.lines().any(|line| {
        (line.contains("task_verdict") || line.contains("task status") || line.contains("任务状态"))
            && (line.contains("failed") || line.contains("失败"))
    }) || evidence.contains("判定 failed");
    let partial = declared_lines.iter().any(|line| {
        starts_with_aa_verdict(line, "partial_success")
            || starts_with_aa_verdict(line, "aa partial_success")
    }) || evidence.lines().any(|line| {
        (line.contains("task_verdict") || line.contains("task status") || line.contains("任务状态"))
            && (line.contains("partial_success") || line.contains("部分成功"))
    });
    let success = declared_lines.iter().any(|line| {
        starts_with_aa_verdict(line, "success") || starts_with_aa_verdict(line, "aa success")
    }) || evidence.lines().any(|line| {
        (line.contains("task_verdict") || line.contains("task status") || line.contains("任务状态"))
            && (line.contains("success") || line.contains("成功"))
    });

    if failed {
        result.status = "failed".to_string();
        result.verdict = Some(TaskVerdict::Failed);
    } else if partial {
        result.status = "partial_success".to_string();
        result.verdict = Some(TaskVerdict::PartialSuccess);
    } else if success {
        result.status = "success".to_string();
        result.verdict = Some(TaskVerdict::Success);
    } else {
        // Models occasionally omit the required AA prefix. Runner success is
        // only an invocation result, so converge from the latest structured
        // CA evidence instead of silently accepting that transport status.
        match latest_ca_report.map(|report| report.verdict) {
            Some(crate::core::recovery::AuditVerdict::Pass) => {
                result.status = "success".to_string();
                result.verdict = Some(TaskVerdict::Success);
            }
            Some(crate::core::recovery::AuditVerdict::Conditional) => {
                result.status = "partial_success".to_string();
                result.verdict = Some(TaskVerdict::PartialSuccess);
            }
            Some(crate::core::recovery::AuditVerdict::Fail) => {
                result.status = "failed".to_string();
                result.verdict = Some(TaskVerdict::Failed);
            }
            None => {
                result.status = "failed".to_string();
                result.verdict = Some(TaskVerdict::Failed);
                result.errors.push(
                    "AA omitted a structured verdict and no CA audit report was available"
                        .to_string(),
                );
            }
        }
    }
}

use crate::memory::l2_blackboard::QueryFilter;
use crate::CoreError;

use super::agent::SupervisorAgent;
use super::types::*;

/// Preserve the user's acceptance boundary across PA summaries and PDCA
/// retries. Plans may operationalize this contract, but may not silently
/// strengthen, weaken, or reinterpret it.
pub(super) fn authoritative_task_contract(
    user_input: &str,
    five_w2h: &crate::core::five_w2h::Task5W2H,
    constraints: &std::collections::HashMap<String, String>,
) -> String {
    let criteria = if five_w2h.why.success_criteria.is_empty() {
        "- Use the original request as the complete acceptance boundary.".to_string()
    } else {
        five_w2h
            .why
            .success_criteria
            .iter()
            .map(|criterion| format!("- {criterion}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let delivery = crate::core::agent_runner::direct_response_delivery_contract(constraints)
        .map(str::to_string)
        .or_else(|| crate::core::agent_runner::workspace_artifact_delivery_contract(constraints))
        .map(|contract| format!("\n\nDelivery contract:\n- {contract}"))
        .unwrap_or_default();
    let capability = crate::core::agent_runner::required_capability_contract(constraints)
        .map(|contract| format!("\n\nEvidence capability contract:\n- {contract}"))
        .unwrap_or_default();
    format!(
        "## Authoritative Task Contract\nOriginal user request (verbatim):\n{user_input}\n\nDeclared success criteria:\n{criteria}{delivery}{capability}\n\nContract rule: planning and recovery may clarify execution steps, but must not add, remove, strengthen, weaken, or reinterpret requirements. Preserve exact quantities and scope."
    )
}

/// Apply a structured, user-issued artifact delivery update between plan
/// steps.  This is deliberately separate from plan text: otherwise the
/// original direct-response constraint continues to override a later TUI
/// instruction to write a file.
pub(super) fn apply_workspace_delivery_contract(
    constraints: &mut std::collections::HashMap<String, String>,
    task_effect_policy: &mut crate::core::effect::EffectPolicy,
    target_path: &str,
) {
    constraints.remove(crate::core::agent_runner::WORKSPACE_CONTEXT_SCOPE_CONSTRAINT);
    constraints.insert(
        crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT.to_string(),
        crate::core::agent_runner::DELIVERY_MODE_WORKSPACE_ARTIFACT.to_string(),
    );
    constraints.insert(
        crate::core::agent_runner::DELIVERY_TARGET_PATH_CONSTRAINT.to_string(),
        target_path.to_string(),
    );
    constraints.insert(
        "required_effect".to_string(),
        "workspace_mutation".to_string(),
    );
    constraints.insert(
        "effect_policy".to_string(),
        "required_workspace_mutation".to_string(),
    );
    *task_effect_policy = crate::core::effect::EffectPolicy::required_workspace_mutation();
}

#[derive(Default)]
struct RecursiveSubCycleOutcome {
    summary: String,
    failed_count: usize,
    partial_count: usize,
}

/// One budget is shared by the entire residual tree.  A per-node limit alone
/// permits exponential work as each child creates another bounded list.
#[derive(Debug)]
pub(super) struct RecursiveExecutionBudget {
    remaining_tasks: usize,
    remaining_turns: u32,
    seen_residuals: std::collections::HashSet<String>,
}

impl RecursiveExecutionBudget {
    pub(super) fn new(max_tasks: usize, max_turns: u32) -> Self {
        Self {
            remaining_tasks: max_tasks,
            remaining_turns: max_turns,
            seen_residuals: std::collections::HashSet::new(),
        }
    }

    fn reserve(&mut self, desired_turns: u32) -> Option<u32> {
        if self.remaining_tasks == 0 || self.remaining_turns == 0 {
            return None;
        }
        self.remaining_tasks = self.remaining_tasks.saturating_sub(1);
        Some(desired_turns.max(1).min(self.remaining_turns))
    }

    fn record_turns(&mut self, actual_turns: u32) {
        self.remaining_turns = self.remaining_turns.saturating_sub(actual_turns);
    }

    fn claim_residual(&mut self, task: &ResidualTaskDef) -> bool {
        let key = residual_task_key(task);
        key.is_empty() || self.seen_residuals.insert(key)
    }
}

#[derive(Deserialize)]
struct ResidualWorkPlan {
    has_sub_tasks: bool,
    sub_tasks: Vec<ResidualTaskDef>,
}

#[derive(Clone, Deserialize)]
struct ResidualTaskDef {
    objective: String,
    #[serde(default = "default_residual_role")]
    role: String,
    success_criteria: String,
    #[serde(default)]
    effect_policy: crate::core::effect::EffectPolicy,
}

fn default_residual_role() -> String {
    "Do".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum TimeoutDecision {
    None,
    ExtendedWithProgress,
    NeedsIntervention { elapsed_seconds: f64 },
}

pub(super) fn evaluate_cycle_timeout(
    cycle: &mut super::types::CycleState,
    now: chrono::DateTime<chrono::Utc>,
    cycle_timeout_seconds: i64,
    cooldown_seconds: i64,
) -> TimeoutDecision {
    let cooldown_seconds = cooldown_seconds.max(1);
    let alert_due = now > cycle.cycle_deadline_at
        && cycle
            .next_timeout_alert_at
            .is_none_or(|next_alert| now >= next_alert);
    if !alert_due {
        return TimeoutDecision::None;
    }
    cycle.timeout_alert_count = cycle.timeout_alert_count.saturating_add(1);
    cycle.last_timeout_alert_at = Some(now);
    cycle.next_timeout_alert_at = Some(now + chrono::Duration::seconds(cooldown_seconds));
    let progress_age = now
        .signed_duration_since(cycle.last_progress_at)
        .num_seconds();
    if progress_age < cooldown_seconds {
        cycle.intervention.monitor = true;
        cycle.cycle_deadline_at = now + chrono::Duration::seconds(cycle_timeout_seconds.max(1));
        TimeoutDecision::ExtendedWithProgress
    } else {
        TimeoutDecision::NeedsIntervention {
            elapsed_seconds: now
                .signed_duration_since(cycle.pdca_started_at)
                .num_seconds() as f64,
        }
    }
}

/// Recursive work needs a bounded share of the parent budget, but an
/// unconditional eight-turn ceiling is too small for implementation tasks:
/// the early warning fired at turn zero and force-finish arrived before a
/// sub-agent could inspect, modify, and verify. Deeper levels receive smaller
/// shares while retaining a useful execution window.
fn recursive_subtask_turn_budget(parent_max: u32, depth: u32) -> u32 {
    let parent_max = parent_max.max(1);
    let divisor = depth.saturating_add(1).max(2);
    (parent_max / divisor).max(12).min(parent_max)
}

/// Residual requirements are revalidated at execution time because an earlier
/// sibling may already have satisfied them.  Mutation remains permitted and
/// anti-stall tracking remains active, but a stale item may complete with
/// concrete evidence instead of manufacturing an unnecessary change.
fn recursive_effect_policy(
    residual: &crate::core::effect::EffectPolicy,
    task: &crate::core::effect::EffectPolicy,
) -> crate::core::effect::EffectPolicy {
    use crate::core::effect::EffectPolicy;
    let resolved = if *residual == EffectPolicy::None {
        task.clone()
    } else {
        residual.clone()
    };
    match resolved {
        EffectPolicy::Required { effect } => EffectPolicy::Conditional {
            effect,
            condition: "the residual effect is not already satisfied in current state".to_string(),
        },
        EffectPolicy::Conditional { effect, condition } if condition.trim().is_empty() => {
            EffectPolicy::Conditional {
                effect,
                condition: "the residual effect is not already satisfied in current state"
                    .to_string(),
            }
        }
        policy => policy,
    }
}

/// Resolve a plan step under the task-level effect contract supplied by the
/// application. Model-generated plans may narrow authority for an individual
/// step, but cannot upgrade an evidence-only/decision-only task into mutation
/// or strengthen a conditional task effect into an unconditional one.
fn effective_step_effect_policy(
    role: AgentRole,
    step: &crate::core::effect::EffectPolicy,
    task: &crate::core::effect::EffectPolicy,
) -> crate::core::effect::EffectPolicy {
    use crate::core::effect::EffectPolicy;
    match role {
        AgentRole::Plan | AgentRole::Check => EffectPolicy::EvidenceOnly,
        AgentRole::Act => EffectPolicy::DecisionOnly,
        AgentRole::Do => match task {
            EffectPolicy::EvidenceOnly => EffectPolicy::EvidenceOnly,
            EffectPolicy::DecisionOnly => EffectPolicy::DecisionOnly,
            EffectPolicy::Conditional { .. } => match step {
                EffectPolicy::EvidenceOnly | EffectPolicy::DecisionOnly => step.clone(),
                _ => task.clone(),
            },
            EffectPolicy::Required { .. } => match step {
                EffectPolicy::EvidenceOnly
                | EffectPolicy::DecisionOnly
                | EffectPolicy::Conditional { .. } => step.clone(),
                _ => task.clone(),
            },
            EffectPolicy::None => step.clone(),
        },
    }
}

fn residual_task_key(task: &ResidualTaskDef) -> String {
    let source = if task.objective.trim().is_empty() {
        &task.success_criteria
    } else {
        &task.objective
    };
    source
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

/// Task-wide execution facts collected across PA/DA/CA/AA, parallel agents,
/// correction passes, and recursive sub-agents. `last_result` remains the
/// semantic terminal answer, while this accumulator prevents earlier tool
/// evidence from disappearing when a later CA/AA result becomes terminal.
#[derive(Default)]
pub(super) struct TaskExecutionFacts {
    turn_count: u32,
    tool_call_count: u32,
    artifacts: Vec<serde_json::Value>,
    errors: Vec<String>,
    tracked_actions: Vec<crate::core::tracked_action::TrackedAction>,
}

impl TaskExecutionFacts {
    pub(super) fn record(&mut self, result: &TaskResult) {
        self.turn_count = self.turn_count.saturating_add(result.turn_count);
        self.tool_call_count = self.tool_call_count.saturating_add(result.tool_call_count);
        self.artifacts.extend(result.artifacts.iter().cloned());
        for error in &result.errors {
            if !self.errors.contains(error) {
                self.errors.push(error.clone());
            }
        }
        for action in &result.tracked_actions {
            if !self
                .tracked_actions
                .iter()
                .any(|existing| existing.action_id == action.action_id)
            {
                self.tracked_actions.push(action.clone());
            }
        }
    }

    pub(super) fn apply_to(&self, result: &mut TaskResult) {
        result.turn_count = self.turn_count;
        result.tool_call_count = self.tool_call_count;
        result.artifacts = self.artifacts.clone();
        result.errors = self.errors.clone();
        result.tracked_actions = self.tracked_actions.clone();
    }

    fn workspace_evidence_paths(&self) -> Vec<String> {
        let mut paths = Vec::new();
        for action in self.tracked_actions.iter().filter(|action| {
            matches!(action.agent_role.as_str(), "DA" | "Do")
                && action.status == crate::core::tracked_action::ActionStatus::Success
        }) {
            for change in action.files_created.iter().chain(&action.files_modified) {
                if !paths.contains(&change.path) {
                    paths.push(change.path.clone());
                }
            }
        }
        for artifact in &self.artifacts {
            if let Some(path) = artifact.get("path").and_then(serde_json::Value::as_str) {
                if !paths.iter().any(|existing| existing == path) {
                    paths.push(path.to_string());
                }
            } else if let Some(path) = artifact.as_str() {
                if !paths.iter().any(|existing| existing == path) {
                    paths.push(path.to_string());
                }
            }
        }
        paths
    }

    /// A late TUI delivery update is valid even after the original DA has
    /// completed.  Evidence paths can be absolute (tool tracking) or
    /// workspace-relative (artifact envelopes), so accept an exact match or a
    /// path ending at the requested relative target.
    fn contains_workspace_artifact(&self, target_path: &str) -> bool {
        let target = std::path::Path::new(target_path);
        self.workspace_evidence_paths().iter().any(|path| {
            let candidate = std::path::Path::new(path);
            candidate == target || candidate.ends_with(target)
        })
    }
}

/// Re-dispatch on failure up to `retry_count` times, sleeping `retry_delay_secs`
/// between attempts. Returns the final result after retries are exhausted.
pub(super) async fn dispatch_with_retry<F, Fut>(
    retry_count: u32,
    retry_delay_secs: u64,
    dispatch: F,
) -> Result<TaskResult, CoreError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<TaskResult, CoreError>>,
{
    let mut remaining = retry_count;
    let mut dispatch = dispatch;
    let mut result = dispatch().await?;
    let mut execution_facts = TaskExecutionFacts::default();
    execution_facts.record(&result);
    while result.status == "failed" && remaining > 0 {
        remaining -= 1;
        if retry_delay_secs > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(retry_delay_secs)).await;
        }
        result = dispatch().await?;
        execution_facts.record(&result);
    }
    execution_facts.apply_to(&mut result);
    Ok(result)
}

/// Construct and run one real business agent.
///
/// SA owns dispatch and prompt selection, BizAgent owns the PA/DA/CA/AA
/// identity and execution mode, and AgentRunner is only BizAgent's low-level
/// execution engine.  Keeping this boundary here prevents SA from bypassing
/// BizAgent and prevents AgentRunner from constructing its business owner.
async fn run_biz_agent(
    runner: std::sync::Arc<crate::core::agent_runner::AgentRunner>,
    agent: AgentInstance,
    context: TaskContext,
    plan_step: Option<PlanStep>,
) -> TaskResult {
    // Only TaskContext carries an authoritative capability restriction. A
    // generated PlanStep's `tools_allowed` is model advice and must not hide
    // essential role tools (the observed failure was DA receiving tool_search
    // but no file_read for an explicitly named file). Explicit DAG workflows
    // copy their declared list into TaskContext before dispatch below.
    let requested_tools = context.allowed_tools.clone();
    let mut context = context;
    context.allowed_tools = enforce_business_role_tool_policy(agent.role, requested_tools);
    debug!(
        role = %agent.role,
        effective_allowed_tools = ?context.allowed_tools,
        "BizAgent authoritative task capability resolved"
    );
    let agent_md = runner
        .build_biz_agent_md(agent.role, &context, plan_step.as_ref())
        .await;
    let config = AgentConfig {
        orchestrator_mode: false,
        max_sub_agents: runner.agent_settings.execution_budget.max_sub_agents,
        max_iterations: context.max_iterations,
        parallel_sub_agents: true,
    };
    let mut biz_agent = BizAgent::new(agent.agent_id, agent.role, &agent_md, runner, config);
    biz_agent.execute(context).await
}

/// Kernel-enforced capability ceiling for each BizAgent role.  Plan-generated
/// tool lists are model output and therefore cannot grant broader authority.
fn enforce_business_role_tool_policy(
    role: AgentRole,
    requested: Option<Vec<String>>,
) -> Option<Vec<String>> {
    let ceiling = crate::core::tool_controller::business_role_tool_ceiling(role);
    let Some(ceiling) = ceiling else {
        return requested;
    };
    Some(
        ceiling
            .iter()
            .filter(|tool| {
                requested
                    .as_ref()
                    .map(|tools| tools.iter().any(|candidate| candidate == **tool))
                    .unwrap_or(true)
            })
            .map(|tool| (*tool).to_string())
            .collect(),
    )
}

impl SupervisorAgent {
    fn create_agent(&self, role: AgentRole, cycle_id: &str) -> AgentInstance {
        let agent_id = format!(
            "{}_{}_{}",
            cycle_id,
            role,
            uuid::Uuid::new_v4().hyphenated()
        );
        AgentInstance::new(agent_id, role)
    }

    /// Per-cycle max iterations after SA intervention deltas (floor ≥ 1).
    pub(super) fn effective_max_iterations(&self, cycle_id: &str) -> u32 {
        let delta = self
            .active_cycles
            .get(cycle_id)
            .map(|c| c.intervention.max_iterations_delta)
            .unwrap_or(0);
        (self.max_iterations as i64 + delta as i64).max(1) as u32
    }

    /// Per-cycle dispatch timeout seconds after SA intervention deltas.
    /// base == 0 keeps the "no timeout" semantics (dispatch_agent only applies
    /// timeout when timeout_secs > 0); a positive base is floored at 1.
    pub(super) fn effective_timeout_secs(&self, cycle_id: &str, base: u64) -> u64 {
        if base == 0 {
            return 0;
        }
        let delta = self
            .active_cycles
            .get(cycle_id)
            .map(|c| c.intervention.timeout_delta_secs)
            .unwrap_or(0);
        (base as i64 + delta).max(1) as u64
    }

    async fn dispatch_agent(
        &self,
        role: AgentRole,
        context: TaskContext,
        cycle_id: &str,
        plan_step: Option<PlanStep>,
        timeout_secs: u64,
    ) -> Result<TaskResult, CoreError> {
        let agent = self.create_agent(role, cycle_id);

        // Query context from L2 blackboard (replaces prev_summary)
        // Use query_nodes_filtered for role/cycle-aware context (AA uses prev_summary)
        let prev_agent_summary = context.prev_agent_summary.clone();
        // An explicit handoff is already bounded and points to its L0 archive.
        // Never replace it with an unbounded L2 body; L2 is only a fallback for
        // callers that did not supply a current-task handoff.
        let prev_summary = if prev_agent_summary.is_some() {
            prev_agent_summary.clone()
        } else if let Some(blackboard) = &self.blackboard {
            let prev_role = match role {
                AgentRole::Do => Some(AgentRole::Plan),
                AgentRole::Check => Some(AgentRole::Do),
                _ => None,
            };
            // Only apply cycle_id filter when we have a specific role filter
            // (PA sees all context nodes; DA/CA see only the previous agent's output from this cycle)
            let filter = QueryFilter {
                role: prev_role.clone(),
                cycle_id: prev_role.map(|_| cycle_id.to_string()),
                node_type: None,
            };
            let nodes = blackboard
                .query_nodes_filtered(&context.task_iri, &filter)
                .unwrap_or_default();
            if !nodes.is_empty() {
                let mut contents: Vec<String> = Vec::new();
                let mut summaries: Vec<String> = Vec::new();
                for n in nodes.iter() {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&n.json_ld) {
                        let role = parsed.get("role").and_then(|v| v.as_str()).unwrap_or("");
                        let prefix = if !role.is_empty() {
                            format!("[{}] ", role)
                        } else {
                            String::new()
                        };
                        // Prefer content field (full LLM output)
                        if let Some(content) = parsed.get("content").and_then(|s| s.as_str()) {
                            let trimmed = content.trim();
                            if !trimmed.is_empty() && trimmed.len() > 20 {
                                contents.push(format!("{}{}", prefix, trimmed));
                                continue;
                            }
                        }
                        // Fallback to summary field
                        if let Some(summary) = parsed.get("summary").and_then(|s| s.as_str()) {
                            let trimmed = summary.trim();
                            if !trimmed.is_empty() {
                                summaries.push(format!("{}{}", prefix, trimmed));
                            }
                        }
                    }
                }
                // Prefer content with substance
                let max_chars = self
                    .runner
                    .agent_settings
                    .execution_budget
                    .ca_handoff_max_chars
                    .max(1);
                if !contents.is_empty() {
                    Some(truncate_chars(&contents.join("\n\n---\n\n"), max_chars))
                } else if !summaries.is_empty() {
                    Some(truncate_chars(&summaries.join("\n"), max_chars))
                } else {
                    prev_agent_summary.clone()
                }
            } else {
                prev_agent_summary.clone()
            }
        } else {
            prev_agent_summary.clone()
        };

        // Activate the MemoryScheduler context path: when the blackboard
        // yields no role-filtered context, recall via the scheduler instead of
        // silently falling back to prev_agent_summary.
        let prev_summary = if prev_summary.is_none() {
            if let Some(ref sched) = self.scheduler {
                match sched
                    .context_request_with_decay_query(role, &context.context_recall_query(), 0.5)
                    .await
                {
                    Ok(recalled) if !recalled.trim().is_empty() => Some(recalled),
                    _ => prev_summary,
                }
            } else {
                prev_summary
            }
        } else {
            prev_summary
        };

        let context = if let Some(ref summary) = prev_summary {
            context.with_prev_summary(summary)
        } else {
            context
        };
        info!(agent_id = %agent.agent_id, role = ?role, task = %context.task_iri, "Dispatching agent with isolation");

        self.event_bus
            .emit(
                &context.task_iri,
                &format!("{:?}_STARTED", role),
                &agent.agent_id,
                &serde_json::json!({"cycle_id": cycle_id}).to_string(),
            )
            .await;

        // Every real PA/DA/CA/AA is a BizAgent. BizAgent owns the business
        // identity and role prompt; its Runner owns the mature ReAct/tool path.
        let iri = context.task_iri.clone();
        let agent_id = agent.agent_id.clone();
        let exec_fut = run_biz_agent(self.runner.clone(), agent, context, plan_step);
        let result = if timeout_secs > 0 {
            match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), exec_fut).await
            {
                Ok(r) => r,
                Err(_) => {
                    warn!(role = ?role, timeout = timeout_secs, "Agent dispatch timed out");
                    return Ok(TaskResult {
                        task_iri: iri,
                        status: "timeout".to_string(),
                        summary: format!(
                            "Agent {:?} timed out after {} seconds",
                            role, timeout_secs
                        ),
                        output: None,
                        jsonld_output: None,
                        artifacts: vec![],
                        errors: vec![],
                        turn_count: 0,
                        tool_call_count: 0,
                        five_w2h_updates: None,
                        tracked_actions: vec![],
                        verdict: Some(TaskVerdict::Timeout),
                        archive_iri: None,
                    });
                }
            }
        } else {
            exec_fut.await
        };

        self.event_bus
            .emit(
                &result.task_iri,
                &format!("{:?}_COMPLETED", role),
                &agent_id,
                &serde_json::json!({"status": &result.status, "summary": &result.summary})
                    .to_string(),
            )
            .await;

        Ok(result)
    }

    async fn dispatch_agents_parallel(
        &self,
        role: AgentRole,
        count: usize,
        base_objective: &str,
        task_iri: &str,
        cycle_id: &str,
        max_iterations: u32,
        timeout_secs: u64,
        tools_allowed: &[String],
        task_constraints: &std::collections::HashMap<String, String>,
        effect_policy: crate::core::effect::EffectPolicy,
    ) -> Result<Vec<TaskResult>, CoreError> {
        let _ = self
            .event_bus
            .emit(
                task_iri,
                "PARALLEL_START",
                "system:sa",
                &serde_json::json!({
                    "role": format!("{:?}", role),
                    "count": count,
                    "cycle_id": cycle_id,
                })
                .to_string(),
            )
            .await;

        let runner = self.runner.clone();
        let mut handles = Vec::new();

        for i in 0..count {
            let objective = format!("[{}-{}] {}", role, i + 1, base_objective);
            let ctx = if tools_allowed.is_empty() {
                TaskContext::new(task_iri, &objective, max_iterations)
                    .with_constraints(task_constraints.clone())
                    .with_effect_policy(effect_policy.clone())
            } else {
                TaskContext::new(task_iri, &objective, max_iterations)
                    .with_constraints(task_constraints.clone())
                    .with_effect_policy(effect_policy.clone())
                    .with_allowed_tools(tools_allowed.to_vec())
            };
            let tid = cycle_id.to_string();
            let runner_clone = runner.clone();

            handles.push(tokio::spawn(async move {
                let agent_id = format!("{}_{}_{}", tid, role, uuid::Uuid::new_v4().hyphenated());
                let agent = AgentInstance::new(agent_id, role);
                let iri = ctx.task_iri.clone();
                if timeout_secs > 0 {
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(timeout_secs),
                        run_biz_agent(runner_clone, agent, ctx, None),
                    )
                    .await
                    {
                        Ok(r) => r,
                        Err(_) => TaskResult {
                            task_iri: iri,
                            status: "timeout".to_string(),
                            summary: format!(
                                "Parallel agent {:?} timed out after {}s",
                                role, timeout_secs
                            ),
                            output: None,
                            jsonld_output: None,
                            artifacts: vec![],
                            errors: vec![],
                            turn_count: 0,
                            tool_call_count: 0,
                            five_w2h_updates: None,
                            tracked_actions: vec![],
                            verdict: Some(TaskVerdict::Timeout),
                            archive_iri: None,
                        },
                    }
                } else {
                    run_biz_agent(runner_clone, agent, ctx, None).await
                }
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            match h.await {
                Ok(res) => results.push(res),
                Err(e) => warn!("Parallel agent panicked: {}", e),
            }
        }

        let _ = self
            .event_bus
            .emit(
                task_iri,
                "PARALLEL_COMPLETE",
                "system:sa",
                &serde_json::json!({
                    "role": format!("{:?}", role),
                    "success_count": results.len(),
                    "total_count": count,
                })
                .to_string(),
            )
            .await;

        info!(count = results.len(), "Parallel agents completed");
        Ok(results)
    }

    pub async fn execute_plan(
        &mut self,
        plan: ExecutionPlan,
        task_iri: &str,
        user_input: &str,
        mut five_w2h: crate::core::five_w2h::Task5W2H,
        five_w2h_iri: &str,
        resumed_messages: Option<Vec<crate::gateway::unified_gateway::ChatMessage>>,
        resumed_state: Option<crate::core::checkpoint::TaskResumeState>,
        initial_prev_summary: Option<String>,
        mut task_effect_policy: crate::core::effect::EffectPolicy,
        mut task_constraints: std::collections::HashMap<String, String>,
    ) -> Result<TaskResult, CoreError> {
        let cycle_id = self
            .active_cycles
            .iter()
            .find(|(_, c)| c.task_iri == task_iri)
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| format!("cycle_{}", uuid::Uuid::new_v4().hyphenated()));

        let _task_id = task_iri
            .strip_prefix("iri://task/")
            .unwrap_or_else(|| task_iri.strip_prefix("iri://").unwrap_or(task_iri));

        if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
            cycle.phase = CyclePhase::Dispatching;
            cycle.phase_history.push("Dispatching".to_string());
        }

        info!(plan_id = %plan.plan_id, steps = plan.steps.len(), "Executing plan with detailed steps");

        if let Some(prefetch) = &self.prefetch_engine {
            let entities: Vec<String> = plan
                .steps
                .iter()
                .filter_map(|s| {
                    if s.expected_output.starts_with("iri://") {
                        Some(s.expected_output.clone())
                    } else {
                        None
                    }
                })
                .collect();
            prefetch
                .on_intent_change(&plan.description, &entities)
                .await;
        }

        let mut last_result: Option<TaskResult> = None;
        let mut execution_facts = TaskExecutionFacts::default();
        let mut recursive_budget = RecursiveExecutionBudget::new(
            self.runner
                .agent_settings
                .execution_budget
                .max_recursive_task_executions,
            self.runner
                .agent_settings
                .execution_budget
                .max_recursive_total_turns,
        );
        let mut prev_summary: Option<String> = initial_prev_summary;
        // Track the Do agent's output separately so AA can access it alongside CA's evaluation.
        let mut da_output: Option<String> = None;
        // Preserve the latest concrete DA deliverable separately from CA/AA
        // decisions. Direct-response tasks return this accepted deliverable
        // after the decision gates pass instead of replacing it with an AA
        // disposition summary.
        let mut latest_da_result: Option<TaskResult> = None;
        // Preserve CA's concrete verification separately from its structured
        // AuditReport. Verify-first evidence tasks have no DA deliverable, so
        // the accepted CA evidence is their user-facing business result.
        let mut latest_ca_result: Option<TaskResult> = None;
        // The latest CA audit is a terminal quality gate.  A later successful
        // re-audit clears this flag; an unresolved failure forces final status.
        let mut latest_ca_report: Option<crate::core::recovery::AuditReport> = None;
        let mut local_repairs_used = 0u32;
        let mut previous_ca_signature: Option<Vec<String>> = None;
        let mut repeated_ca_failures = 0u32;
        let mut authoritative_contract =
            authoritative_task_contract(user_input, &five_w2h, &task_constraints);

        // Resume mode: determine which phase to start from
        // Load latest checkpoint from L0 to resolve phase tags
        let resume_skip_phases: Vec<AgentRole> = if resumed_messages.is_some() {
            let skip_roles = if let Some(state) = &resumed_state {
                crate::core::checkpoint::compute_skip_roles_from_phase(
                    &state.checkpoint_name,
                    state.current_role.as_deref(),
                )
            } else {
                // Compatibility fallback for callers that supplied an
                // ad-hoc message history instead of a RestoredTask.
                let cm = crate::core::checkpoint::CheckpointManager::with_persistence(
                    self.runner.l0_store.clone(),
                );
                cm.restore_latest_with_skip_roles(task_iri)
                    .ok()
                    .flatten()
                    .map(|(_, roles)| roles)
                    .unwrap_or_else(|| vec!["Plan".to_string()])
            };
            skip_roles
                .iter()
                .filter_map(|r| match r.as_str() {
                    "Plan" => Some(AgentRole::Plan),
                    "Do" => Some(AgentRole::Do),
                    "Check" => Some(AgentRole::Check),
                    "Act" => Some(AgentRole::Act),
                    _ => None,
                })
                .collect()
        } else {
            vec![]
        };
        info!("[resume] skip phases: {:?}", resume_skip_phases);

        // Resume mode: prefer restoring from checkpoint's prev_summary field
        // If no prev_summary in checkpoint, extract PA output from history messages
        let resume_prev_summary: Option<String> = if resumed_messages.is_some() {
            let from_cp = resumed_state
                .as_ref()
                .and_then(|state| state.prev_summary.clone())
                .or_else(|| {
                    let cm = crate::core::checkpoint::CheckpointManager::with_persistence(
                        self.runner.l0_store.clone(),
                    );
                    cm.restore_latest(task_iri)
                        .ok()
                        .flatten()
                        .and_then(|cp| cp.prev_summary)
                });
            if from_cp.is_some() {
                from_cp
            } else {
                // Fallback: extract PA phase output from history messages as prev_summary
                resumed_messages.as_ref().and_then(|msgs| {
                    let mut found_first_user = false;
                    for msg in msgs.iter() {
                        if msg.role == "user" && !found_first_user {
                            found_first_user = true;
                            continue;
                        }
                        if msg.role == "assistant" && found_first_user {
                            return Some(msg.content.clone());
                        }
                    }
                    msgs.iter()
                        .rev()
                        .find(|m| m.role == "assistant")
                        .map(|m| m.content.clone())
                })
            }
        } else {
            None
        };

        let _task_level = match plan.task_complexity {
            TaskComplexity::Instant => "Instant",
            TaskComplexity::Simple => "Simple",
            TaskComplexity::Standard => "Standard",
            TaskComplexity::Complex => "Complex",
            TaskComplexity::Exploratory => "Complex",
            TaskComplexity::Emergency => "Standard",
            TaskComplexity::Recursive => "Recursive",
        };

        // --- Unified DAG execution path ---
        // Convert ExecutionPlan to DAG (LLM path adapter) or use external JSON-LD DAG directly (--workflow path)
        let dag = if let Some(ref dag_jsonld) = plan.dag_jsonld {
            let def =
                crate::core::workflow::loader::load_workflow_jsonld(dag_jsonld).map_err(|e| {
                    CoreError::Internal {
                        message: format!("Workflow parsing failed: {}", e),
                    }
                })?;
            crate::core::workflow::loader::build_dag(&def).map_err(|e| CoreError::Internal {
                message: format!("DAG build failed: {}", e),
            })?
        } else {
            let wf = crate::core::workflow::adapter::plan_to_workflow(&plan, task_iri);
            crate::core::workflow::loader::build_dag(&wf).map_err(|e| CoreError::Internal {
                message: format!("DAG build failed: {}", e),
            })?
        };
        let order = crate::core::workflow::loader::topological_order(&dag).map_err(|e| {
            CoreError::Internal {
                message: format!("Topological sort failed: {}", e),
            }
        })?;

        let mut completed_node_results: std::collections::HashMap<
            String,
            crate::core::workflow::NodeResult,
        > = std::collections::HashMap::new();
        let mut skip_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();

        // ── Compute topological depth for wave-based parallel dispatch ──
        // Depth = longest path from entry node (all predecessors must complete before this depth)
        let mut node_depth: std::collections::HashMap<NodeIndex, usize> =
            std::collections::HashMap::new();
        for &nidx in &order {
            let depth = dag
                .graph
                .neighbors_directed(nidx, Incoming)
                .filter_map(|p| node_depth.get(&p))
                .max()
                .map(|d| d + 1)
                .unwrap_or(0);
            node_depth.insert(nidx, depth);
        }
        // Group consecutive order indices with the same depth into waves
        let mut waves: Vec<Vec<usize>> = Vec::new();
        {
            let mut pos = 0;
            while pos < order.len() {
                let d = node_depth[&order[pos]];
                let mut wave = vec![pos];
                pos += 1;
                while pos < order.len() && node_depth[&order[pos]] == d {
                    wave.push(pos);
                    pos += 1;
                }
                waves.push(wave);
            }
        }

        // Execute DAG wave by wave — nodes at the same topological depth have all deps met
        // and can run concurrently via join_all
        for wave in &waves {
            // ═══════════════════════════════════════════════════════════
            // Phase 1: Pre-process each node in the wave
            // (skip checks, HumanApprovalNode, objective building, context)
            // ═══════════════════════════════════════════════════════════
            struct WaveTask {
                wi: usize,
                ni: NodeIndex,
                step: PlanStep,
                ctx: TaskContext,
                timeout_secs: u64,
            }
            let mut agent_tasks: Vec<WaveTask> = Vec::new();

            for &wi in wave {
                let ni = order[wi];
                let nd = &dag.graph[ni].def;
                let step = crate::core::workflow::adapter::node_to_planstep(nd);

                // Check skip set (branch jump from HumanApprovalNode)
                if skip_nodes.contains(&nd.id) {
                    info!(node_id = %nd.id, "HumanApprovalNode branch jump: skipping this node");
                    continue;
                }

                // AA must evaluate the latest CA evidence.  CA failures are
                // repaired after the first DAG pass, so do not let AA make a
                // terminal decision against the pre-repair result.
                if step.role == AgentRole::Act
                    && latest_ca_report
                        .as_ref()
                        .is_some_and(|report| report.failed())
                {
                    info!(
                        node_id = %nd.id,
                        "Deferring AA until CA evidence has converged"
                    );
                    continue;
                }

                // Resume mode: skip completed phases
                if resume_skip_phases.contains(&step.role) {
                    info!(role = ?step.role, "[resume] skipping completed phase");
                    if prev_summary.is_none() {
                        prev_summary = resume_prev_summary.clone().or_else(|| {
                            Some("Restored from checkpoint, preceding phase completed.".to_string())
                        });
                    }
                    continue;
                }

                // HumanApprovalNode: blocking, runs inline in the wave's pre-phase
                if nd.node_type == "HumanApprovalNode" {
                    let approval = self
                        .request_human_approval_general(&nd.approval_prompt, &nd.id, task_iri)
                        .await?;

                    let status = if approval.approved {
                        "approved"
                    } else {
                        "rejected"
                    };
                    let summary = format!(
                        "[HumanApproval] {}: {}",
                        if approval.approved {
                            "Approved"
                        } else {
                            "Rejected"
                        },
                        approval.comment.as_deref().unwrap_or("")
                    );

                    completed_node_results.insert(
                        nd.id.clone(),
                        crate::core::workflow::NodeResult {
                            node_id: nd.id.clone(),
                            status: status.to_string(),
                            summary: summary.clone(),
                            archive_iri: None,
                            turn_count: 0,
                            tool_call_count: 0,
                            error: if approval.approved {
                                None
                            } else {
                                Some("User rejected".to_string())
                            },
                            output: None,
                            artifacts: vec![],
                        },
                    );

                    let ha_result = TaskResult {
                        task_iri: task_iri.to_string(),
                        status: status.to_string(),
                        summary: summary.clone(),
                        output: None,
                        jsonld_output: None,
                        artifacts: vec![],
                        errors: vec![],
                        turn_count: 0,
                        tool_call_count: 0,
                        five_w2h_updates: None,
                        tracked_actions: vec![],
                        verdict: None,
                        archive_iri: None,
                    };
                    prev_summary = Some(format!("## Human Approval Result\n{}", summary));
                    last_result = Some(ha_result);

                    // Branch jump handling (rejected → skip to reject target)
                    if !approval.approved {
                        if let Some(ref reject_target) = nd.approval_next_on_reject {
                            let mut found = false;
                            for skip_idx in (wi + 1)..order.len() {
                                let sid = dag.graph[order[skip_idx]].def.id.clone();
                                if sid == *reject_target {
                                    found = true;
                                    break;
                                }
                                skip_nodes.insert(sid);
                            }
                            if !found {
                                for skip_idx in (wi + 1)..order.len() {
                                    skip_nodes.insert(dag.graph[order[skip_idx]].def.id.clone());
                                }
                            }
                        }
                    }
                    // Approved → skip to approve target
                    if approval.approved {
                        if let Some(ref approve_target) = nd.approval_next_on_approve {
                            let mut found = false;
                            for skip_idx in (wi + 1)..order.len() {
                                let sid = dag.graph[order[skip_idx]].def.id.clone();
                                if sid == *approve_target {
                                    found = true;
                                    break;
                                }
                                skip_nodes.insert(sid);
                            }
                            if !found {
                                for skip_idx in (wi + 1)..order.len() {
                                    skip_nodes.insert(dag.graph[order[skip_idx]].def.id.clone());
                                }
                            }
                        }
                    }

                    info!(node_id = %nd.id, status = %status, "HumanApprovalNode processing complete");
                    continue;
                }

                // ── Supplementary input processing & pause check ──
                let supplementary = self
                    .check_and_process_supplementary_inputs(task_iri, &step.role, &step.objective)
                    .await?;
                if let Some(target_path) = supplementary.workspace_delivery_target {
                    apply_workspace_delivery_contract(
                        &mut task_constraints,
                        &mut task_effect_policy,
                        &target_path,
                    );
                    authoritative_contract =
                        authoritative_task_contract(user_input, &five_w2h, &task_constraints);
                    info!(
                        task_iri = %task_iri,
                        target_path,
                        "Applied supplementary workspace delivery contract"
                    );
                }
                // Cycle timeout check
                {
                    let now = chrono::Utc::now();
                    let cooldown = self.perception.anomaly_dedup_window_seconds();
                    let cycle_timeout = self.perception.cycle_timeout_secs().max(1);
                    let mut ambiguous_timeout_elapsed = None;
                    if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
                        match evaluate_cycle_timeout(cycle, now, cycle_timeout, cooldown) {
                            TimeoutDecision::ExtendedWithProgress => {
                                info!(
                                    cycle_id = %cycle_id,
                                    alert_count = cycle.timeout_alert_count,
                                    "PDCA deadline crossed with recent progress; monitoring window extended deterministically"
                                );
                            }
                            TimeoutDecision::NeedsIntervention { elapsed_seconds } => {
                                ambiguous_timeout_elapsed = Some(elapsed_seconds);
                            }
                            TimeoutDecision::None => {}
                        }
                    }
                    if let Some(elapsed) = ambiguous_timeout_elapsed {
                        let intervention = self
                            .perception
                            .on_cycle_timeout(&cycle_id, task_iri, elapsed);
                        if intervention.should_interrupt {
                            let _ = tokio::time::timeout(
                                std::time::Duration::from_secs(self.execution_timeout_secs),
                                self.execute_intervention_for_cycle(intervention, task_iri),
                            )
                            .await;
                        }
                    }
                }
                // Pause check
                let paused = self
                    .active_cycles
                    .get(&cycle_id)
                    .map(|c| c.phase == CyclePhase::Idle)
                    .unwrap_or(false);
                if paused {
                    info!(step_id = %step.step_id, role = ?step.role, "Execution paused, waiting for resume");
                    loop {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        let mut payloads = Vec::new();
                        if let Some(ref mut receiver) = self.event_receiver {
                            while let Ok(event) = receiver.try_recv() {
                                if event.task_iri != task_iri {
                                    continue;
                                }
                                if event.event_type == "USER_SUPPLEMENTARY_INPUT" {
                                    payloads.push(event.payload.clone());
                                }
                            }
                        }
                        for payload in payloads {
                            self.enqueue_supplementary_input(task_iri, &payload);
                        }
                        let resumed = self
                            .active_cycles
                            .get(&cycle_id)
                            .map(|c| c.phase == CyclePhase::Executing)
                            .unwrap_or(false);
                        if resumed {
                            break;
                        }
                    }
                }

                // ── Check for parallel_groups (ExecutionPlan same-role parallelism) ──
                if plan
                    .parallel_groups
                    .iter()
                    .any(|g| g.len() > 1 && g.contains(&step.role))
                {
                    let matching_groups: Vec<_> = plan
                        .parallel_groups
                        .iter()
                        .filter(|g| g.contains(&step.role))
                        .collect();
                    let parallel_group = match matching_groups.first() {
                        Some(g) => (*g).clone(),
                        None => {
                            warn!(role = ?step.role, "No parallel group found despite any() check");
                            continue;
                        }
                    };
                    let count = parallel_group.len();
                    let explicit_workflow_tools = if plan.dag_jsonld.is_some() {
                        step.tools_allowed.as_slice()
                    } else {
                        &[]
                    };
                    let results = self
                        .dispatch_agents_parallel(
                            step.role,
                            count,
                            &step.objective,
                            task_iri,
                            &cycle_id,
                            self.effective_max_iterations(&cycle_id),
                            self.effective_timeout_secs(&cycle_id, nd.timeout_secs),
                            explicit_workflow_tools,
                            &task_constraints,
                            effective_step_effect_policy(
                                step.role,
                                &step.effect_policy,
                                &task_effect_policy,
                            ),
                        )
                        .await?;

                    for result in &results {
                        execution_facts.record(result);
                    }

                    let failed = results.iter().find(|r| r.status == "failed");
                    if let Some(f) = failed {
                        warn!(role = ?step.role, step_id = %step.step_id, "Parallel agent failed");
                        let mut failed_result = TaskResult {
                            task_iri: task_iri.to_string(),
                            status: "partial_failure".to_string(),
                            summary: format!("Some parallel {:?} agents failed", step.role),
                            output: None,
                            jsonld_output: None,
                            artifacts: Vec::new(),
                            errors: f.errors.clone(),
                            turn_count: results.iter().map(|r| r.turn_count).sum(),
                            tool_call_count: results.iter().map(|r| r.tool_call_count).sum(),
                            five_w2h_updates: None,
                            tracked_actions: Vec::new(),
                            verdict: Some(TaskVerdict::Failed),
                            archive_iri: None,
                        };
                        execution_facts.apply_to(&mut failed_result);
                        return Ok(failed_result);
                    }

                    let ca_handoff_max_chars = self
                        .runner
                        .agent_settings
                        .execution_budget
                        .ca_handoff_max_chars;
                    let combined_summary: String = results
                        .iter()
                        .map(|r| {
                            format!(
                                "[{}] {}",
                                r.task_iri,
                                result_handoff(r, step.role, ca_handoff_max_chars)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    prev_summary = Some(truncate_chars(
                        &combined_summary,
                        ca_handoff_max_chars.max(1),
                    ));
                    last_result = results.into_iter().last();
                    continue;
                }

                // ── Build objective (PDCA role-specific templates) ──
                let cycle_hints = self
                    .active_cycles
                    .values()
                    .find(|c| c.task_iri == task_iri)
                    .map(|c| c.experience_hints.clone())
                    .unwrap_or_default();
                // Historical outcomes may guide planning, and may guide DA
                // only when no PA handoff exists (Simple/Emergency plans).
                // CA and AA must remain independent evidence-based auditors;
                // injecting a prior success conclusion would bias both gates.
                let role_may_use_history = matches!(step.role, AgentRole::Plan)
                    || (matches!(step.role, AgentRole::Do) && prev_summary.is_none());
                let hints_block = if cycle_hints.is_empty() || !role_may_use_history {
                    String::new()
                } else {
                    format!(
                        "\n\n## Historical Experience\n{}",
                        cycle_hints
                            .iter()
                            .map(|h| format!("- {}", h))
                            .collect::<Vec<_>>()
                            .join("\n")
                    )
                };
                let objective = match (&prev_summary, step.role) {
                    (Some(summary), AgentRole::Plan) => {
                        // PA has no predecessor-content slot in its generic
                        // role template, so feedback is embedded here exactly
                        // once. Other roles consume their typed handoff slot.
                        format!("{}\n\n{}{}\n\n## Feedback from Previous PDCA Cycle\n{}\n\nPlease create a detailed execution plan for the task contract, addressing the feedback without changing the acceptance boundary.", step.objective, authoritative_contract, hints_block, summary)
                    }
                    (Some(_), AgentRole::Do) => {
                        format!("{}\n\n{}{}\n\nThe upper PA plan is supplied once in the bounded previous-agent handoff. Execute it only within the authoritative contract; if it conflicts, the contract wins.", step.objective, authoritative_contract, hints_block)
                    }
                    (Some(_), AgentRole::Check) => {
                        format!("{}\n\n{}{}\n\nExecution results are supplied once in the bounded previous-agent handoff. Please independently verify whether they are correct and complete. The authoritative contract overrides any plan-generated expected output that would add a file, path, graph node, or other undeclared deliverable.", step.objective, authoritative_contract, hints_block)
                    }
                    (Some(_), AgentRole::Act) => {
                        let da_context = da_output
                            .as_ref()
                            .map(|da| format!("\n\n## Execution Results\n{}", da))
                            .unwrap_or_default();
                        format!("{}\n\n{}\n\n## Original Task\n{}{}{}\n\nThe latest CA conclusions are supplied once in the bounded previous-agent handoff. Please make the final decision and summarize without adding acceptance requirements.", step.objective, authoritative_contract, user_input, da_context, hints_block)
                    }
                    (None, AgentRole::Plan) => {
                        format!("{}\n\n{}{}\n\nPlease create a detailed execution plan for the task contract.", step.objective, authoritative_contract, hints_block)
                    }
                    (None, AgentRole::Do) => {
                        format!("{}\n\n{}{}\n\nExecute every requirement in the authoritative contract and verify the declared criteria.", step.objective, authoritative_contract, hints_block)
                    }
                    _ => step.objective.clone(),
                };

                // ── Build context ──
                let mut context = TaskContext::new(
                    task_iri,
                    &objective,
                    self.effective_max_iterations(&cycle_id),
                )
                .with_original_task(user_input)
                .with_constraints(task_constraints.clone())
                .with_effect_policy(effective_step_effect_policy(
                    step.role,
                    &step.effect_policy,
                    &task_effect_policy,
                ))
                .with_step_info(&step.expected_output, &step.success_criteria)
                .with_cycle_id(&cycle_id)
                .with_workspace_evidence_paths(execution_facts.workspace_evidence_paths());
                if plan.dag_jsonld.is_some() && !step.tools_allowed.is_empty() {
                    context = context.with_allowed_tools(step.tools_allowed.clone());
                }
                context = context.with_five_w2h(five_w2h_iri, five_w2h.clone());

                // Resume mode: history messages on first executed step
                let is_first_executed_step = if resume_skip_phases.is_empty() {
                    wi == 0
                } else {
                    !resume_skip_phases.contains(&step.role)
                        && plan.steps[..wi]
                            .iter()
                            .all(|s| resume_skip_phases.contains(&s.role))
                };
                if is_first_executed_step {
                    if let Some(ref msgs) = resumed_messages {
                        context = if let Some(state) = resumed_state.clone() {
                            context.with_resumed_checkpoint(msgs.clone(), state)
                        } else {
                            let turn_count =
                                msgs.iter().filter(|m| m.role == "assistant").count() as u32;
                            let tool_count = msgs
                                .iter()
                                .filter(|m| m.role == "tool" || m.tool_call_id.is_some())
                                .count() as u32;
                            context.with_resumed_messages(msgs.clone(), turn_count, tool_count)
                        };
                    }
                }
                if let Some(ref pv) = prev_summary {
                    context = context.with_prev_summary(pv);
                }

                // ── Thought emission ──
                let role_name = format!("{:?}", step.role);
                self.emit_sa_thought(
                    task_iri,
                    &format!(
                        "Wave step {}/{}: dispatching {} — {}",
                        wi + 1,
                        plan.steps.len(),
                        role_name,
                        step.objective
                    ),
                    &format!("dispatch_{}", role_name.to_lowercase()),
                )
                .await;

                agent_tasks.push(WaveTask {
                    wi,
                    ni,
                    step,
                    ctx: context,
                    timeout_secs: self.effective_timeout_secs(&cycle_id, nd.timeout_secs),
                });
            }

            // ═══════════════════════════════════════════════════════════
            // Phase 2: Dispatch all agent nodes in this wave
            // ═══════════════════════════════════════════════════════════
            let num_tasks = agent_tasks.len();

            if num_tasks > 1 {
                // Multi-node wave: concurrent dispatch (all futures share the same type)
                let self_ref: &Self = &*self;
                let mut futs = Vec::new();
                for wt in &agent_tasks {
                    let role = wt.step.role;
                    let ctx = wt.ctx.clone();
                    let step = wt.step.clone();
                    let cid = cycle_id.to_string();
                    let wi = wt.wi;
                    let to = wt.timeout_secs;
                    let retry_count = wt.step.retry_count;
                    let retry_delay = wt.step.retry_delay_secs;
                    futs.push(async move {
                        (
                            wi,
                            dispatch_with_retry(retry_count, retry_delay, || {
                                self_ref.dispatch_agent(
                                    role,
                                    ctx.clone(),
                                    &cid,
                                    Some(step.clone()),
                                    to,
                                )
                            })
                            .await,
                        )
                    });
                }

                let dispatch_results = futures::future::join_all(futs).await;

                if let Some(error) = dispatch_results
                    .iter()
                    .find_map(|(_, result)| result.as_ref().err())
                {
                    // Other nodes in the same wave may already have completed
                    // successfully. Preserve their observable facts even when
                    // one sibling fails at the dispatch boundary.
                    for (_, result) in &dispatch_results {
                        if let Ok(result) = result {
                            execution_facts.record(result);
                        }
                    }
                    warn!(error = %error, "Wave node dispatch error");
                    let mut failed = TaskResult {
                        task_iri: task_iri.to_string(),
                        status: "failed".to_string(),
                        summary: format!("Wave node dispatch failed: {}", error),
                        output: None,
                        jsonld_output: None,
                        artifacts: vec![],
                        errors: vec![error.to_string()],
                        turn_count: 0,
                        tool_call_count: 0,
                        five_w2h_updates: None,
                        tracked_actions: vec![],
                        verdict: Some(TaskVerdict::Failed),
                        archive_iri: None,
                    };
                    execution_facts.record(&failed);
                    execution_facts.apply_to(&mut failed);
                    return Ok(failed);
                }

                for (result_wi, result_res) in dispatch_results {
                    let task_ni = order[result_wi];
                    let task_nd = &dag.graph[task_ni].def;
                    let task_step = crate::core::workflow::adapter::node_to_planstep(task_nd);
                    let result = result_res.expect("wave errors handled before result processing");
                    if let Some(failed_task) = self
                        .handle_step_result(
                            result,
                            task_step,
                            task_ni,
                            result_wi,
                            &mut prev_summary,
                            &mut da_output,
                            &mut latest_da_result,
                            &mut latest_ca_result,
                            &mut latest_ca_report,
                            &mut previous_ca_signature,
                            &mut repeated_ca_failures,
                            &mut last_result,
                            &mut execution_facts,
                            &mut completed_node_results,
                            &mut skip_nodes,
                            &mut five_w2h,
                            task_iri,
                            &cycle_id,
                            &plan,
                            &dag,
                            &order,
                            five_w2h_iri,
                            &task_effect_policy,
                            &task_constraints,
                            &mut recursive_budget,
                        )
                        .await?
                    {
                        return Ok(failed_task);
                    }
                }
            } else if num_tasks == 1 {
                let wt = agent_tasks.into_iter().next().unwrap();
                let result =
                    dispatch_with_retry(wt.step.retry_count, wt.step.retry_delay_secs, || {
                        self.dispatch_agent(
                            wt.step.role,
                            wt.ctx.clone(),
                            &cycle_id,
                            Some(wt.step.clone()),
                            wt.timeout_secs,
                        )
                    })
                    .await?;
                if let Some(failed_task) = self
                    .handle_step_result(
                        result,
                        wt.step,
                        wt.ni,
                        wt.wi,
                        &mut prev_summary,
                        &mut da_output,
                        &mut latest_da_result,
                        &mut latest_ca_result,
                        &mut latest_ca_report,
                        &mut previous_ca_signature,
                        &mut repeated_ca_failures,
                        &mut last_result,
                        &mut execution_facts,
                        &mut completed_node_results,
                        &mut skip_nodes,
                        &mut five_w2h,
                        task_iri,
                        &cycle_id,
                        &plan,
                        &dag,
                        &order,
                        five_w2h_iri,
                        &task_effect_policy,
                        &task_constraints,
                        &mut recursive_budget,
                    )
                    .await?
                {
                    return Ok(failed_task);
                }
            }
        }

        // A user can change the delivery target while CA or AA is already
        // running.  At that point the original DAG has no remaining DA node,
        // so merely changing the constraints would acknowledge the input but
        // never create the requested file.  Drain once more after the DAG and
        // synthesize a narrow DA→CA reconciliation only when exact artifact
        // evidence is still absent.
        let mut delivery_reconciled_after_dag = false;
        let supplementary = self
            .check_and_process_supplementary_inputs(
                task_iri,
                &AgentRole::Act,
                "Final delivery reconciliation",
            )
            .await?;
        if let Some(target_path) = supplementary.workspace_delivery_target {
            apply_workspace_delivery_contract(
                &mut task_constraints,
                &mut task_effect_policy,
                &target_path,
            );
            authoritative_contract =
                authoritative_task_contract(user_input, &five_w2h, &task_constraints);
            info!(
                task_iri = %task_iri,
                target_path,
                "Applied supplementary workspace delivery contract after DAG completion"
            );
        }

        let delivery_target = task_constraints
            .get(crate::core::agent_runner::DELIVERY_TARGET_PATH_CONSTRAINT)
            .cloned();
        if let Some(target_path) =
            delivery_target.filter(|target| !execution_facts.contains_workspace_artifact(target))
        {
            self.event_bus
                .emit(
                    task_iri,
                    "DELIVERY_RECONCILIATION_STARTED",
                    "SA",
                    &serde_json::json!({"target_path": target_path}).to_string(),
                )
                .await;
            self.emit_sa_thought(
                task_iri,
                &format!(
                    "Late delivery instruction requires a verified workspace artifact at {target_path}"
                ),
                "reconcile_workspace_delivery",
            )
            .await;

            let handoff = latest_da_result
                .as_ref()
                .map(|result| {
                    result_handoff(
                        result,
                        AgentRole::Do,
                        self.runner
                            .agent_settings
                            .execution_budget
                            .ca_handoff_max_chars,
                    )
                })
                .or_else(|| prev_summary.clone())
                .unwrap_or_else(|| {
                    "No prior DA handoff is available; construct the complete deliverable from the original task."
                        .to_string()
                });
            let da_objective = format!(
                "## Late Delivery Reconciliation\n\n{}\n\nThe user added a workspace delivery requirement after the original plan began. Create the complete final deliverable at the exact workspace-relative path `{target_path}` using `file_write`. Preserve valid prior work from the handoff below, but do not return a chat-only answer. After writing, state the exact path and what was verified.\n\n## Prior Work Handoff\n{handoff}",
                authoritative_contract,
            );
            let da_context = TaskContext::new(
                task_iri,
                &da_objective,
                self.effective_max_iterations(&cycle_id),
            )
            .with_original_task(user_input)
            .with_constraints(task_constraints.clone())
            .with_effect_policy(crate::core::effect::EffectPolicy::required_workspace_mutation())
            .with_cycle_id(&cycle_id)
            .with_prev_summary(&handoff)
            .with_workspace_evidence_paths(execution_facts.workspace_evidence_paths());

            let da_result = self
                .dispatch_agent(AgentRole::Do, da_context, &cycle_id, None, 0)
                .await?;
            execution_facts.record(&da_result);
            if da_result.status == "failed" {
                self.event_bus
                    .emit(
                        task_iri,
                        "DELIVERY_RECONCILIATION_FAILED",
                        "SA",
                        &serde_json::json!({"target_path": target_path, "reason": da_result.summary})
                            .to_string(),
                    )
                    .await;
                let mut failed_result = da_result;
                execution_facts.apply_to(&mut failed_result);
                return Ok(failed_result);
            }

            let da_handoff = result_handoff(
                &da_result,
                AgentRole::Do,
                self.runner
                    .agent_settings
                    .execution_budget
                    .ca_handoff_max_chars,
            );
            da_output = Some(da_handoff.clone());
            latest_da_result = Some(da_result.clone());
            let ca_objective = format!(
                "## Verify Late Workspace Delivery\n\n{}\n\nIndependently verify that `{target_path}` now exists in the current workspace. Read the exact file, confirm it is the requested complete Markdown deliverable, and report evidence. Do not accept a chat-only answer or a differently named file.\n\n## DA Handoff\n{da_handoff}",
                authoritative_contract,
            );
            let ca_context = TaskContext::new(
                task_iri,
                &ca_objective,
                self.effective_max_iterations(&cycle_id),
            )
            .with_original_task(user_input)
            .with_constraints(task_constraints.clone())
            .with_effect_policy(crate::core::effect::EffectPolicy::EvidenceOnly)
            .with_allowed_tools(vec!["file_read".to_string()])
            .with_cycle_id(&cycle_id)
            .with_prev_summary(&da_handoff)
            .with_workspace_evidence_paths(execution_facts.workspace_evidence_paths());
            let mut ca_result = self
                .dispatch_agent(AgentRole::Check, ca_context, &cycle_id, None, 0)
                .await?;
            let mut ca_report = apply_ca_dimension_audit(
                &five_w2h,
                &mut ca_result,
                task_iri,
                self.runner.causal_engine.as_ref().map(|ce| ce.as_ref()),
            );
            crate::core::recovery::track_non_convergence(
                &mut ca_report,
                &mut previous_ca_signature,
                &mut repeated_ca_failures,
            );
            execution_facts.record(&ca_result);
            latest_ca_report = Some(ca_report);
            latest_ca_result = Some(ca_result.clone());
            prev_summary = Some(result_handoff(
                &ca_result,
                AgentRole::Check,
                self.runner
                    .agent_settings
                    .execution_budget
                    .ca_handoff_max_chars,
            ));
            last_result = Some(ca_result);
            delivery_reconciled_after_dag = true;
            self.event_bus
                .emit(
                    task_iri,
                    "DELIVERY_RECONCILIATION_COMPLETED",
                    "SA",
                    &serde_json::json!({"target_path": target_path}).to_string(),
                )
                .await;
        }

        // ── CA→DA correction loop ──
        // When CA's dimension audit detects failures in DA's work, re-dispatch DA
        // with corrective feedback instead of immediately proceeding to AA.
        let execution_budget = &self.runner.agent_settings.execution_budget;
        let max_ca_da_corrections = execution_budget.max_ca_da_corrections;
        let correction_handoff_max_chars = execution_budget.ca_correction_handoff_max_chars;
        let mut correction_count = 0;
        loop {
            let ca_summary = prev_summary.clone();
            let directive = latest_ca_report.as_ref().map(|report| {
                crate::core::recovery::select_directive(
                    report,
                    local_repairs_used,
                    max_ca_da_corrections as u32,
                )
            });
            let has_local_ca_failures = matches!(
                directive,
                Some(crate::core::recovery::RecoveryDirective::RetryDa)
            ) && ca_summary.is_some();

            if !has_local_ca_failures || correction_count >= max_ca_da_corrections {
                break;
            }

            correction_count += 1;
            local_repairs_used += 1;
            let ca_text = ca_summary.unwrap_or_default();
            let original_execution_handoff = latest_da_result
                .as_ref()
                .map(|result| {
                    result_handoff(result, AgentRole::Do, correction_handoff_max_chars.max(1))
                })
                .or_else(|| da_output.clone())
                .unwrap_or_else(|| "No prior DA deliverable was retained.".to_string());

            info!(
                task_iri = %task_iri,
                correction = correction_count,
                "CA dimension audit found failures — re-dispatching DA with corrective context"
            );

            let da_corrective_objective = format!(
                "## Corrective Re-Execution (iteration {})\n\n\
                 ## Original DA Deliverable\n{}\n\n\
                 ## Previous CA Findings\n{}\n\n\
                 Fix ALL identified issues without discarding already-valid work. Return the complete corrected deliverable, followed by a concise change note; do not return only a repair receipt.",
                correction_count,
                original_execution_handoff,
                ca_text
                    .chars()
                    .take(correction_handoff_max_chars)
                    .collect::<String>()
            );

            let mut correction_constraints = task_constraints.clone();
            correction_constraints.insert(
                crate::core::agent_runner::SA_RECOVERY_MODE_CONSTRAINT.to_string(),
                crate::core::agent_runner::CA_DA_CORRECTION_MODE.to_string(),
            );

            let da_ctx = TaskContext::new(
                task_iri,
                &da_corrective_objective,
                self.effective_max_iterations(&cycle_id),
            )
            .with_original_task(user_input)
            .with_constraints(correction_constraints)
            .with_effect_policy(if task_effect_policy.may_require_workspace_mutation() {
                crate::core::effect::EffectPolicy::required_workspace_mutation()
            } else {
                task_effect_policy.clone()
            })
            .with_cycle_id(&cycle_id)
            .with_workspace_evidence_paths(execution_facts.workspace_evidence_paths());

            match self
                .dispatch_agent(AgentRole::Do, da_ctx, &cycle_id, None, 0)
                .await
            {
                Ok(da_result) => {
                    execution_facts.record(&da_result);
                    let corrected_handoff = result_handoff(
                        &da_result,
                        AgentRole::Do,
                        self.runner
                            .agent_settings
                            .execution_budget
                            .ca_handoff_max_chars,
                    );
                    da_output = Some(corrected_handoff.clone());
                    latest_da_result = Some(da_result.clone());
                    let ca_objective = format!(
                        "Re-evaluate corrected execution:\n\n\
                         The complete corrected output for iteration {} is supplied once in the bounded previous-agent handoff. Read that exact AgentTurn when more content is needed. Verify ALL previous audit issues are resolved; do not search KG/RAG for a repair receipt.",
                        correction_count
                    );

                    let mut ca_ctx = TaskContext::new(
                        task_iri,
                        &ca_objective,
                        self.effective_max_iterations(&cycle_id),
                    )
                    .with_original_task(user_input)
                    .with_constraints(task_constraints.clone())
                    .with_effect_policy(crate::core::effect::EffectPolicy::EvidenceOnly)
                    .with_cycle_id(&cycle_id)
                    .with_prev_summary(&corrected_handoff)
                    .with_workspace_evidence_paths(execution_facts.workspace_evidence_paths());
                    if let Some(tools) = direct_response_recheck_tools(&task_constraints) {
                        ca_ctx = ca_ctx.with_allowed_tools(tools);
                    }

                    match self
                        .dispatch_agent(AgentRole::Check, ca_ctx, &cycle_id, None, 0)
                        .await
                    {
                        Ok(ca_result) => {
                            let mut ca_result = ca_result;
                            let mut ca_report = apply_ca_dimension_audit(
                                &five_w2h,
                                &mut ca_result,
                                task_iri,
                                self.runner.causal_engine.as_ref().map(|ce| ce.as_ref()),
                            );
                            crate::core::recovery::track_non_convergence(
                                &mut ca_report,
                                &mut previous_ca_signature,
                                &mut repeated_ca_failures,
                            );
                            execution_facts.record(&ca_result);
                            latest_ca_report = Some(ca_report);
                            latest_ca_result = Some(ca_result.clone());
                            let ca_evidence = result_handoff(
                                &ca_result,
                                AgentRole::Check,
                                self.runner
                                    .agent_settings
                                    .execution_budget
                                    .ca_handoff_max_chars,
                            );
                            prev_summary = Some(truncate_chars(
                                &format!(
                                "## Corrected Execution (iter {})\n{}\n\n## CA Re-Evaluation\n{}",
                                correction_count, da_result.summary, ca_evidence
                            ),
                                self.runner
                                    .agent_settings
                                    .execution_budget
                                    .ca_handoff_max_chars
                                    .max(1),
                            ));
                            last_result = Some(ca_result);

                            self.emit_sa_thought(
                                task_iri,
                                &format!("CA→DA correction #{} completed", correction_count),
                                "ca_da_correction",
                            )
                            .await;
                        }
                        Err(e) => {
                            warn!(error = %e, "CA re-dispatch after DA correction failed");
                            break;
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "DA corrective re-dispatch failed");
                    break;
                }
            }
        }

        // AA is a decision role, not a repair role. Never ask it to reconsider
        // evidence while the latest CA audit is still failed. SA first routes
        // unresolved executable gaps back to DA; repeated/non-convergent gaps
        // are promoted to PA by the outer PDCA controller.
        if let Some(report) = latest_ca_report.as_ref().filter(|report| report.failed()) {
            let directive = crate::core::recovery::select_directive(
                report,
                local_repairs_used,
                max_ca_da_corrections as u32,
            );
            let (scope, failed_step) = match directive {
                crate::core::recovery::RecoveryDirective::RetryDa => (
                    crate::core::recovery::RepairScope::Step,
                    plan.steps
                        .iter()
                        .find(|step| step.role == AgentRole::Do)
                        .map(|step| step.step_id.as_str()),
                ),
                _ => (
                    crate::core::recovery::RepairScope::Task,
                    plan.steps
                        .iter()
                        .find(|step| step.role == AgentRole::Plan)
                        .map(|step| step.step_id.as_str()),
                ),
            };
            if let Some(result) = last_result.as_mut() {
                result.status = "failed".to_string();
                result.verdict = Some(TaskVerdict::Failed);
                result.summary.push_str(&format!(
                    "\n\n[Recovery] directive={:?} scope={:?} failed_step={}",
                    directive,
                    scope,
                    failed_step.unwrap_or("unresolved_ca_audit")
                ));
                result.errors.push(format!(
                    "latest CA audit remains failed; SA routed recovery to {:?}",
                    directive
                ));
            }
        }

        // A correction invalidates any earlier AA evidence. Once CA has
        // converged, make one fresh final decision from the latest audit.
        if (correction_count > 0 || delivery_reconciled_after_dag)
            && latest_ca_report
                .as_ref()
                .is_some_and(|report| !report.failed())
        {
            if let Some((aa_index, aa_step)) = plan
                .steps
                .iter()
                .enumerate()
                .rfind(|(_, step)| step.role == AgentRole::Act)
                .map(|(idx, step)| (idx, step.clone()))
            {
                let aa_objective = format!(
                    "{}\n\n## Original Task\n{}\n\n## Latest CA/DA Evidence\n{}\n\nMake the final acceptance decision using only the latest evidence.",
                    aa_step.objective,
                    user_input,
                    prev_summary.as_deref().unwrap_or("No execution summary available")
                );
                let aa_ctx = TaskContext::new(
                    task_iri,
                    &aa_objective,
                    self.effective_max_iterations(&cycle_id),
                )
                .with_original_task(user_input)
                .with_constraints(task_constraints.clone())
                .with_effect_policy(crate::core::effect::EffectPolicy::DecisionOnly)
                .with_step_info(&aa_step.expected_output, &aa_step.success_criteria)
                .with_cycle_id(&cycle_id)
                .with_five_w2h(five_w2h_iri, five_w2h.clone());
                let aa_result = self
                    .dispatch_agent(
                        AgentRole::Act,
                        aa_ctx,
                        &cycle_id,
                        Some(aa_step.clone()),
                        self.effective_timeout_secs(&cycle_id, 0),
                    )
                    .await?;
                if let Some(failed_task) = self
                    .handle_step_result(
                        aa_result,
                        aa_step,
                        order
                            .get(aa_index)
                            .copied()
                            .unwrap_or_else(|| order[order.len() - 1]),
                        order.len().saturating_sub(1),
                        &mut prev_summary,
                        &mut da_output,
                        &mut latest_da_result,
                        &mut latest_ca_result,
                        &mut latest_ca_report,
                        &mut previous_ca_signature,
                        &mut repeated_ca_failures,
                        &mut last_result,
                        &mut execution_facts,
                        &mut completed_node_results,
                        &mut skip_nodes,
                        &mut five_w2h,
                        task_iri,
                        &cycle_id,
                        &plan,
                        &dag,
                        &order,
                        five_w2h_iri,
                        &task_effect_policy,
                        &task_constraints,
                        &mut recursive_budget,
                    )
                    .await?
                {
                    return Ok(failed_task);
                }
            }
        }

        if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
            cycle.phase = CyclePhase::Completed;
            cycle.task_completed = true;
            cycle.phase_history.push("Completed".to_string());
        }

        self.event_bus
            .emit(
                task_iri,
                "CYCLE_COMPLETED",
                "SA",
                &serde_json::json!({"cycle_id": &cycle_id}).to_string(),
            )
            .await;

        let mut final_result = last_result.unwrap_or(TaskResult {
            task_iri: task_iri.to_string(),
            status: "completed".to_string(),
            summary: "No agents executed".to_string(),
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
        restore_accepted_deliverable(
            &mut final_result,
            latest_da_result.as_ref(),
            latest_ca_result.as_ref(),
            &task_constraints,
            &task_effect_policy,
            plan.verify_first,
        );
        execution_facts.apply_to(&mut final_result);
        enforce_ca_audit_terminal_status(
            &mut final_result,
            latest_ca_report
                .as_ref()
                .is_some_and(|report| report.failed()),
        );
        persist_ca_validated_knowledge(
            self,
            task_iri,
            user_input,
            &task_constraints,
            latest_ca_report.as_ref(),
            &final_result,
        );
        Ok(final_result)
    }

    fn build_failed_step_result(
        &self,
        task_iri: &str,
        step: &PlanStep,
        result: &TaskResult,
    ) -> TaskResult {
        let error_detail = result
            .errors
            .first()
            .map(|e| format!("\n\n**Error details**: {}", e))
            .unwrap_or_default();
        let (directive, scope) = failed_business_role_recovery(step.role);
        TaskResult {
            task_iri: task_iri.to_string(),
            status: "failed".to_string(),
            summary: format!(
                "Agent {:?} failed at step {}{}\n\n[Recovery] directive={} scope={} failed_step={}",
                step.role, step.step_id, error_detail, directive, scope, step.step_id,
            ),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: result.errors.clone(),
            turn_count: result.turn_count,
            tool_call_count: result.tool_call_count,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            verdict: None,
            archive_iri: None,
        }
    }

    /// Process a single DAG node's execution result — handles failure, 5W2H, perception,
    /// AA early exit, recursive sub-cycles, prev_summary tracking, and checkpoint.
    /// Returns `Ok(Some(TaskResult))` if the caller should terminate (node failure),
    /// `Ok(None)` to continue normally.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_step_result(
        &mut self,
        mut result: TaskResult,
        step: PlanStep,
        _node_idx: NodeIndex,
        i: usize,
        prev_summary: &mut Option<String>,
        da_output: &mut Option<String>,
        latest_da_result: &mut Option<TaskResult>,
        latest_ca_result: &mut Option<TaskResult>,
        latest_ca_report: &mut Option<crate::core::recovery::AuditReport>,
        previous_ca_signature: &mut Option<Vec<String>>,
        repeated_ca_failures: &mut u32,
        last_result: &mut Option<TaskResult>,
        execution_facts: &mut TaskExecutionFacts,
        completed_node_results: &mut std::collections::HashMap<
            String,
            crate::core::workflow::NodeResult,
        >,
        skip_nodes: &mut std::collections::HashSet<String>,
        five_w2h: &mut crate::core::five_w2h::Task5W2H,
        task_iri: &str,
        cycle_id: &str,
        plan: &ExecutionPlan,
        dag: &crate::core::workflow::loader::WorkflowDag,
        order: &[NodeIndex],
        five_w2h_iri: &str,
        task_effect_policy: &crate::core::effect::EffectPolicy,
        task_constraints: &std::collections::HashMap<String, String>,
        recursive_budget: &mut RecursiveExecutionBudget,
    ) -> Result<Option<TaskResult>, CoreError> {
        if step.role == AgentRole::Act {
            apply_aa_declared_verdict(&mut result, latest_ca_report.as_ref());
        }
        if result.turn_count > 0 || result.tool_call_count > 0 || result.status == "success" {
            if let Some(cycle) = self.active_cycles.get_mut(cycle_id) {
                cycle.last_progress_at = chrono::Utc::now();
            }
        }
        execution_facts.record(&result);
        let task_id = task_iri
            .strip_prefix("iri://task/")
            .unwrap_or_else(|| task_iri.strip_prefix("iri://").unwrap_or(task_iri));

        // Early return on agent failure
        if result.status == "failed" {
            // branch_on_failure: skip intermediates up to branch_fallback, then continue
            if step.branch_on_failure {
                if let Some(ref target) = step.branch_fallback {
                    let mut found = false;
                    for node_idx in order.iter().skip(i + 1) {
                        let sid = dag.graph[*node_idx].def.id.clone();
                        if sid == *target {
                            found = true;
                            break;
                        }
                        skip_nodes.insert(sid);
                    }
                    if found {
                        warn!(role = ?step.role, step_id = %step.step_id, target = %target, "Agent failed, branching to fallback step");
                    } else {
                        warn!(role = ?step.role, step_id = %step.step_id, target = %target, "Agent failed, branch target not in remaining order, aborting plan");
                        let mut failed = self.build_failed_step_result(task_iri, &step, &result);
                        execution_facts.apply_to(&mut failed);
                        return Ok(Some(failed));
                    }
                } else {
                    warn!(role = ?step.role, step_id = %step.step_id, "Agent failed, no branch fallback target, aborting plan");
                    let mut failed = self.build_failed_step_result(task_iri, &step, &result);
                    execution_facts.apply_to(&mut failed);
                    return Ok(Some(failed));
                }
            } else {
                warn!(role = ?step.role, step_id = %step.step_id, "Agent failed, aborting plan");
                let mut failed = self.build_failed_step_result(task_iri, &step, &result);
                execution_facts.apply_to(&mut failed);
                return Ok(Some(failed));
            }
        }

        // Propagate 5W2H updates
        if let Some(ref updates) = result.five_w2h_updates {
            five_w2h.merge_updates(updates);
            if let Ok(updated_json_ld) = five_w2h.to_json_ld(task_iri) {
                let _ = self
                    .runner
                    .l0_store
                    .store(&five_w2h_iri, &updated_json_ld.to_string());
                let cfg = crate::CoreConfig::default();
                if let Some(ref bb) = self.blackboard {
                    if bb
                        .write_node(&five_w2h_iri, &updated_json_ld.to_string(), &cfg)
                        .is_ok()
                    {
                        tracing::debug!(five_w2h_iri = %five_w2h_iri, "5W2H update synced to blackboard");
                    }
                }
            }
        }

        // AA freeze
        if step.role == AgentRole::Act && result.status == "success" {
            let is_last_aa = plan
                .steps
                .iter()
                .rposition(|s| s.role == AgentRole::Act)
                .map(|last_act| {
                    plan.steps
                        .iter()
                        .position(|s| s.step_id == step.step_id)
                        .map(|idx| idx >= last_act)
                        .unwrap_or(true)
                })
                .unwrap_or(true);
            if is_last_aa {
                five_w2h.freeze();
                if let Ok(frozen_json_ld) = five_w2h.to_json_ld(task_iri) {
                    let snapshot_iri = format!("iri://task/{}/snapshot", task_id);
                    let _ = self
                        .runner
                        .l0_store
                        .store(&snapshot_iri, &frozen_json_ld.to_string());
                    let _ = self
                        .runner
                        .l0_store
                        .store(&five_w2h_iri, &frozen_json_ld.to_string());
                    let cfg = crate::CoreConfig::default();
                    if let Some(ref bb) = self.blackboard {
                        let _ = bb.write_node(&snapshot_iri, &frozen_json_ld.to_string(), &cfg);
                        let _ = bb.write_node(&five_w2h_iri, &frozen_json_ld.to_string(), &cfg);
                    }
                    info!(task_iri = %task_iri, "5W2H frozen and archived");
                }
            } else {
                info!(task_iri = %task_iri, step_id = %step.step_id, "Intermediate AA step: 5W2H not frozen yet");
            }
        }

        // Sharing
        self.sharing.create_share(
            &format!("iri://agent/{}", step.role),
            "iri://agent/next",
            &[format!("iri://task/{}/result", task_iri)],
            crate::tools::sharing::ShareType::Projection,
            crate::tools::sharing::Permission::Read,
            Some(3600),
            None,
        );

        // PA perception
        if step.role == AgentRole::Plan && result.status == "success" {
            let plan_data = serde_json::json!({
                "summary": &result.summary,
                "objective": &step.objective,
            });
            let advisories = self.perception.on_plan_completed(&plan_data, task_iri);
            if !advisories.is_empty() {
                info!(
                    count = advisories.len(),
                    "PA perception advisories generated"
                );
            }
        }

        // CA perception + dimension audit
        if step.role == AgentRole::Check {
            let check_data = serde_json::json!({
                "summary": &result.summary,
                "objective": &step.objective,
            });
            if let Some(advisory) = self.perception.on_check_completed(&check_data, task_iri) {
                info!(advisory = ?advisory, "CA perception advisories generated");
            }

            // Run dimension-level audit against the 5W2H specification.  The
            // latest CA result controls the terminal quality gate.
            let mut ca_report = apply_ca_dimension_audit(
                five_w2h,
                &mut result,
                task_iri,
                self.runner.causal_engine.as_ref().map(|ce| ce.as_ref()),
            );
            // The same failed dimensions twice in a row indicate that DA
            // cannot converge locally; SA must escalate to a PA re-plan.
            // The counters live in execute_plan and are intentionally shared
            // by the normal CA path and the CA→DA correction path.
            // (The report is still attached to the result for observability.)
            crate::core::recovery::track_non_convergence(
                &mut ca_report,
                previous_ca_signature,
                repeated_ca_failures,
            );
            *latest_ca_report = Some(ca_report);
            *latest_ca_result = Some(result.clone());
        }

        // AA early exit — skip remaining PDCA cycles after AA evaluates
        if step.role == AgentRole::Act {
            let has_remaining = (i + 1) < order.len();
            if has_remaining {
                let reason = match result.status.as_str() {
                    "success" => "AA passed, task completed",
                    "failed" | "partial_success" => "AA did not pass",
                    _ => "AA evaluated",
                };
                info!(step_id = %step.step_id, status = %result.status, "{}, skipping remaining PDCA cycles", reason);
                for skip_idx in (i + 1)..order.len() {
                    skip_nodes.insert(dag.graph[order[skip_idx]].def.id.clone());
                }
            }
        }

        // Recursive sub-cycle for Do agents. A successful DA now proceeds
        // directly to the independent CA unless its completion envelope says
        // executable work remains. This removes the former unconditional LLM
        // decomposition after every successful DA and every successful child.
        let completion_envelope = crate::core::effect::CompletionEnvelope::from_result(
            &result.status,
            result.output.as_ref(),
            &result.summary,
        );
        if step.role == AgentRole::Do
            && (result.status == "success" || result.status == "partial_success")
            && completion_envelope.needs_follow_up_execution()
            && plan.max_recursion_depth > 0
            && (plan.task_complexity == crate::core::sa::types::TaskComplexity::Recursive
                || plan.task_complexity == crate::core::sa::types::TaskComplexity::Complex)
        {
            let sub_results = self
                .execute_recursive_sub_cycle(
                    &result.summary,
                    &completion_envelope,
                    task_iri,
                    cycle_id,
                    &step.step_id,
                    plan.max_recursion_depth,
                    1,
                    five_w2h,
                    five_w2h_iri,
                    execution_facts,
                    task_effect_policy,
                    task_constraints,
                    recursive_budget,
                )
                .await;

            match sub_results {
                Ok(sub_outcome) => {
                    *prev_summary = Some(truncate_chars(
                        &format!(
                            "{}\n\n## Sub-task Execution Results\n{}",
                            result.summary, sub_outcome.summary
                        ),
                        self.runner
                            .agent_settings
                            .execution_budget
                            .ca_handoff_max_chars
                            .max(1),
                    ));
                    if sub_outcome.failed_count > 0 {
                        result.status = "failed".to_string();
                        result.verdict = Some(TaskVerdict::Failed);
                        result.errors.push(format!(
                            "{} recursive sub-task(s) failed",
                            sub_outcome.failed_count
                        ));
                        let mut failed = self.build_failed_step_result(task_iri, &step, &result);
                        failed.summary = prev_summary.clone().unwrap_or(failed.summary);
                        execution_facts.apply_to(&mut failed);
                        return Ok(Some(failed));
                    }
                    if sub_outcome.partial_count > 0 && result.status == "success" {
                        result.status = "partial_success".to_string();
                        result.verdict = Some(TaskVerdict::PartialSuccess);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Required recursive sub-cycle execution failed");
                    result.status = "failed".to_string();
                    result.verdict = Some(TaskVerdict::Failed);
                    result
                        .errors
                        .push(format!("Recursive sub-cycle failed: {e}"));
                    *prev_summary = Some(format!(
                        "{}\n\nRecursive sub-cycle failed: {}",
                        result.summary, e
                    ));
                    let mut failed = self.build_failed_step_result(task_iri, &step, &result);
                    failed.summary = prev_summary.clone().unwrap_or(failed.summary);
                    execution_facts.apply_to(&mut failed);
                    return Ok(Some(failed));
                }
            }
        } else {
            *prev_summary = Some(result_handoff(
                &result,
                step.role,
                self.runner
                    .agent_settings
                    .execution_budget
                    .ca_handoff_max_chars,
            ));
        }

        // Track Do agent output separately
        if step.role == AgentRole::Do {
            if let Some(ref s) = *prev_summary {
                *da_output = Some(s.clone());
            }
            *latest_da_result = Some(result.clone());
        }

        *last_result = Some(result);

        // 5W2H constraint check
        if let Some(alert) = self.perception.check_5w2h_constraints(five_w2h_iri) {
            tracing::warn!(alert = %alert, "5W2H constraint alert");
            self.event_bus
                .emit(
                    task_iri,
                    &alert,
                    "SA",
                    &serde_json::json!({"task_iri": task_iri}).to_string(),
                )
                .await;
        }

        info!(step_id = %step.step_id, role = ?step.role, status = ?last_result.as_ref().map(|r| &r.status), "Step completed");

        // ── Checkpoint ──
        {
            let cm = crate::core::checkpoint::CheckpointManager::with_persistence(
                self.runner.l0_store.clone(),
            );
            let role_name = format!("{:?}", step.role);
            let state_json = serde_json::json!({
                "turn": last_result.as_ref().map(|r| r.turn_count).unwrap_or(0),
                "tc": last_result.as_ref().map(|r| r.tool_call_count).unwrap_or(0),
                "prompt_tokens": self.runner.total_prompt_tokens.load(std::sync::atomic::Ordering::Relaxed),
                "completion_tokens": self.runner.total_completion_tokens.load(std::sync::atomic::Ordering::Relaxed),
            }).to_string();

            let cycle_state = self.active_cycles.get(cycle_id).map(|c| {
                serde_json::json!({
                    "phase": format!("{:?}", c.phase),
                    "iteration": c.iteration,
                    "phase_history": c.phase_history,
                    "task_completed": c.task_completed,
                    "experience_hints": c.experience_hints,
                })
                .to_string()
            });

            let completed_nodes = if completed_node_results.is_empty() {
                None
            } else {
                Some(serde_json::to_string(completed_node_results).unwrap_or_default())
            };

            let pending_approvals = {
                let map = self.pending_approvals.lock().await;
                if map.is_empty() {
                    None
                } else {
                    Some(serde_json::to_string(&*map).unwrap_or_default())
                }
            };

            let supplement_data = {
                // A checkpoint must never consume a user instruction.  The
                // next AgentRunner turn owns consumption via take_pending().
                let pending = self.supplement_store.snapshot_pending(task_iri);
                if pending.is_empty() {
                    None
                } else {
                    let entries: Vec<serde_json::Value> = pending
                        .iter()
                        .map(|e| {
                            serde_json::json!({
                                "content": e.content,
                                "relevance_score": e.relevance_score,
                                "timestamp": e.timestamp,
                            })
                        })
                        .collect();
                    Some(serde_json::to_string(&entries).unwrap_or_default())
                }
            };

            let cp_name = format!("step_complete_{}", role_name);
            let tags = vec![role_name.clone(), "step_complete".to_string()];

            let session_msgs_json: String = if let Some(ref bb) = self.blackboard {
                let filter = crate::memory::l2_blackboard::QueryFilter {
                    role: None,
                    cycle_id: Some(cycle_id.to_string()),
                    node_type: Some("AgentTurn".to_string()),
                };
                let nodes = bb
                    .query_nodes_filtered(task_iri, &filter)
                    .unwrap_or_default();
                let msgs: Vec<serde_json::Value> = nodes.iter().filter_map(|n| {
                    let parsed: serde_json::Value = serde_json::from_str(&n.json_ld).ok()?;
                    Some(serde_json::json!({
                        "role": parsed.get("role").and_then(|r| r.as_str()).unwrap_or("assistant"),
                        "content": parsed.get("content").and_then(|c| c.as_str()).unwrap_or(""),
                        "summary": parsed.get("summary").and_then(|s| s.as_str()),
                    }))
                }).collect();
                serde_json::to_string(&msgs).unwrap_or_else(|_| "[]".to_string())
            } else {
                "[]".to_string()
            };

            if let Err(e) = cm.create_ext(
                task_iri,
                &cp_name,
                "[]",
                &session_msgs_json,
                &state_json,
                &tags,
                Some(&role_name),
                None,
                prev_summary.as_deref(),
                cycle_state.as_deref(),
                completed_nodes.as_deref(),
                pending_approvals.as_deref(),
                supplement_data.as_deref(),
                None,
                None,
                None,
            ) {
                warn!("[checkpoint] step_complete save failed: {}", e);
            } else {
                info!("[checkpoint] step_complete_{} saved", role_name);
            }
        }

        Ok(None)
    }

    fn execute_recursive_sub_cycle<'a>(
        &'a self,
        da_summary: &'a str,
        completion_envelope: &'a crate::core::effect::CompletionEnvelope,
        task_iri: &'a str,
        cycle_id: &'a str,
        parent_step_id: &'a str,
        max_depth: u32,
        current_depth: u32,
        five_w2h: &'a crate::core::five_w2h::Task5W2H,
        five_w2h_iri: &'a str,
        execution_facts: &'a mut TaskExecutionFacts,
        task_effect_policy: &'a crate::core::effect::EffectPolicy,
        task_constraints: &'a std::collections::HashMap<String, String>,
        recursive_budget: &'a mut RecursiveExecutionBudget,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<RecursiveSubCycleOutcome, CoreError>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            if current_depth > max_depth {
                info!(
                    depth = current_depth,
                    max_depth, "Recursive depth limit reached, stopping sub-cycle"
                );
                return Ok(RecursiveSubCycleOutcome {
                    summary: "Recursive depth limit reached".to_string(),
                    ..Default::default()
                });
            }

            self.emit_sa_thought(
                task_iri,
                &format!(
                    "▶ Recursive sub-cycle (depth {}/{})",
                    current_depth, max_depth
                ),
                "recursive_sub_cycle_start",
            )
            .await;

            let sub_task = SubTask::new(
                &format!(
                    "Decomposing sub-tasks from DA result (depth={})",
                    current_depth
                ),
                parent_step_id,
                current_depth,
            );

            info!(
                sub_task_id = %sub_task.sub_task_id,
                depth = current_depth,
                max_depth,
                "Starting recursive sub-cycle"
            );

            self.emit_sa_thought(
                task_iri,
                &format!(
                    "Decomposing DA result, identifying sub-tasks... (depth {}/{})",
                    current_depth, max_depth
                ),
                "recursive_decompose",
            )
            .await;

            let parsed = if completion_envelope.structured {
                // Structured DA handoff is authoritative and avoids another
                // LLM call. This is the fast path for optimized prompts.
                ResidualWorkPlan {
                    has_sub_tasks: !completion_envelope.pending_effects.is_empty(),
                    sub_tasks: completion_envelope
                        .pending_effects
                        .iter()
                        .map(|pending| ResidualTaskDef {
                            objective: pending.objective.clone(),
                            role: "Do".to_string(),
                            success_criteria: if pending.reason.trim().is_empty() {
                                format!("Complete and verify: {}", pending.objective)
                            } else {
                                pending.reason.clone()
                            },
                            effect_policy: pending.effect_policy.clone(),
                        })
                        .collect(),
                }
            } else {
                let decompose_prompt = format!(
                    r#"You are a task decomposition expert. Below is an execution result summary of a DA (Do Agent). Analyze whether there are sub-tasks that need further execution.

## DA Execution Result
{}

## Task Context
- Original goal: {}
- Current recursion depth: {}/{}

## Output Requirements
Output the list of sub-tasks that need further execution in JSON format. If no further sub-tasks are needed, return an empty array.

```json
{{
  "has_sub_tasks": true/false,
  "sub_tasks": [
    {{
      "objective": "Sub-task objective description",
      "role": "Do",
      "success_criteria": "Success criteria",
      "effect_policy": {{"mode":"none"}}
    }}
  ]
}}
```

## Evaluation Criteria
1. If the DA result explicitly mentions "still needs...", "next step needs...", etc., there are sub-tasks
2. If the DA result has fully completed the goal with no remaining work, there are no sub-tasks
3. Sub-tasks must be concrete residual execution work, not review, testing-only, acceptance, or final-decision work; those belong to the normal Check/Act phases
4. Use role `Do` only. Use `none` to inherit the original task effect contract. If you emit `conditional`, include both its generic effect and a concrete `condition`
5. Maximum of {} sub-tasks

Output only JSON."#,
                    da_summary,
                    five_w2h.what,
                    current_depth,
                    max_depth,
                    self.runner
                        .agent_settings
                        .execution_budget
                        .max_recursive_sub_tasks,
                );

                let model = self.runner.gateway.get_model("default");
                let messages = vec![crate::gateway::unified_gateway::ChatMessage {
                    role: "user".to_string(),
                    content: decompose_prompt,
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                }];

                let response = self
                    .chat_sa_streaming(
                        task_iri,
                        "recursive_decomposition",
                        &model,
                        messages,
                        Some(0.3),
                        Some(8192),
                    )
                    .await
                    .map_err(|e| CoreError::Internal {
                        message: format!("Recursive decomposition LLM call failed: {}", e),
                    })?;

                let content = response
                    .choices
                    .first()
                    .and_then(|c| c.message.content.clone())
                    .unwrap_or_default();

                let json_str = if content.starts_with('{') {
                    content.clone()
                } else if let Some(start) = content.find('{') {
                    if let Some(end) = content.rfind('}') {
                        content[start..=end].to_string()
                    } else {
                        content.clone()
                    }
                } else {
                    return Ok(RecursiveSubCycleOutcome {
                        summary: "Recursive decomposition failed: LLM did not return valid JSON"
                            .to_string(),
                        failed_count: 1,
                        partial_count: 0,
                    });
                };
                serde_json::from_str::<ResidualWorkPlan>(&json_str).map_err(|e| {
                    CoreError::Internal {
                        message: format!("Recursive decomposition JSON parse failed: {}", e),
                    }
                })?
            };

            if !parsed.has_sub_tasks || parsed.sub_tasks.is_empty() {
                info!(
                    depth = current_depth,
                    "No further decomposition needed for DA result"
                );
                self.emit_sa_thought(task_iri,
                &format!("Sub-task decomposition complete: no further decomposition needed (depth {}/{})", current_depth, max_depth),
                "recursive_no_tasks").await;
                return Ok(RecursiveSubCycleOutcome {
                    summary: "No further decomposition needed".to_string(),
                    ..Default::default()
                });
            }

            self.emit_sa_thought(
                task_iri,
                &format!(
                    "Identified {} sub-tasks (depth {}/{})",
                    parsed.sub_tasks.len(),
                    current_depth,
                    max_depth
                ),
                "recursive_tasks_found",
            )
            .await;

            let mut outcome = RecursiveSubCycleOutcome::default();
            let mut sub_summaries = Vec::new();

            for (idx, sub_def) in parsed
                .sub_tasks
                .iter()
                .take(
                    self.runner
                        .agent_settings
                        .execution_budget
                        .max_recursive_sub_tasks,
                )
                .enumerate()
            {
                if !sub_def.role.eq_ignore_ascii_case("do")
                    || matches!(
                        sub_def.effect_policy,
                        crate::core::effect::EffectPolicy::EvidenceOnly
                            | crate::core::effect::EffectPolicy::DecisionOnly
                    )
                {
                    sub_summaries.push(format!(
                        "### Residual item {} deferred to normal Check/Act phase\n{}",
                        idx + 1,
                        sub_def.objective
                    ));
                    continue;
                }
                if !recursive_budget.claim_residual(sub_def) {
                    sub_summaries.push(format!(
                        "### Residual item {} skipped as task-wide duplicate\n{}",
                        idx + 1,
                        sub_def.objective
                    ));
                    continue;
                }
                let sub_effect_policy =
                    recursive_effect_policy(&sub_def.effect_policy, task_effect_policy);
                let sub_objective =
                    format!("[recursive depth={}] {}", current_depth, sub_def.objective);
                info!(sub_idx = idx, objective = %sub_def.objective, "Executing recursive sub-task");

                let desired_turn_budget = recursive_subtask_turn_budget(
                    self.effective_max_iterations(cycle_id),
                    current_depth,
                );
                let Some(sub_turn_budget) = recursive_budget.reserve(desired_turn_budget) else {
                    outcome.partial_count = outcome.partial_count.saturating_add(1);
                    sub_summaries.push(format!(
                        "### Residual execution budget exhausted\nDeferred to independent Check/recovery: {}",
                        sub_def.objective
                    ));
                    info!(
                        depth = current_depth,
                        remaining_tasks = recursive_budget.remaining_tasks,
                        remaining_turns = recursive_budget.remaining_turns,
                        "Task-wide recursive execution budget exhausted"
                    );
                    break;
                };

                let sibling_evidence = if sub_summaries.is_empty() {
                    String::new()
                } else {
                    format!(
                        "\n\nEarlier residual outcomes (revalidate against current state):\n{}",
                        truncate_chars(
                            &sub_summaries.join("\n"),
                            self.runner
                                .agent_settings
                                .execution_budget
                                .recursive_handoff_max_chars
                                .max(1),
                        )
                    )
                };

                let mut sub_ctx = TaskContext::new(task_iri, &sub_objective, sub_turn_budget)
                    .with_original_task(&format!(
                        "{}\n\nSub-task: {}\nSub-task success criteria: {}",
                        five_w2h.what, sub_def.objective, sub_def.success_criteria
                    ))
                    .with_constraints(task_constraints.clone())
                    .with_effect_policy(sub_effect_policy.clone())
                    .with_prev_summary(&format!(
                        "Parent DA evidence:\n{}{}",
                        da_summary
                            .chars()
                            .take(
                                self.runner
                                    .agent_settings
                                    .execution_budget
                                    .recursive_handoff_max_chars,
                            )
                            .collect::<String>(),
                        sibling_evidence
                    ));

                sub_ctx = sub_ctx.with_five_w2h(five_w2h_iri, five_w2h.clone());

                if let Some(ref bb) = self.blackboard {
                    let nodes = bb.query_nodes(task_iri).unwrap_or_default();
                    if !nodes.is_empty() {
                        let summaries: Vec<String> = nodes
                            .iter()
                            .filter_map(|n| {
                                let parsed: serde_json::Value =
                                    serde_json::from_str(&n.json_ld).ok()?;
                                parsed
                                    .get("summary")
                                    .and_then(|s| s.as_str())
                                    .map(String::from)
                            })
                            .collect();
                        if !summaries.is_empty() {
                            let max_chars = self
                                .runner
                                .agent_settings
                                .execution_budget
                                .recursive_handoff_max_chars
                                .max(1);
                            let parent = da_summary.chars().take(max_chars).collect::<String>();
                            let related = truncate_chars(&summaries.join("\n"), max_chars);
                            sub_ctx = sub_ctx.with_prev_summary(&format!(
                                "Parent DA evidence:\n{}{}\n\nRelated completed-step evidence:\n{}",
                                parent, sibling_evidence, related
                            ));
                        }
                    }
                }

                let sub_step = PlanStep {
                    step_id: format!("{}_sub_{}", parent_step_id, idx),
                    role: AgentRole::Do,
                    objective: sub_def.objective.clone(),
                    expected_output: sub_def.success_criteria.clone(),
                    dependencies: vec![parent_step_id.to_string()],
                    tools_allowed: vec![],
                    success_criteria: sub_def.success_criteria.clone(),
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: sub_effect_policy,
                };

                let total = parsed.sub_tasks.len();
                self.emit_sa_thought(
                    task_iri,
                    &format!(
                        "▶ Executing sub-task {}/{}: {} (depth {})",
                        idx + 1,
                        total,
                        sub_def.objective,
                        current_depth
                    ),
                    "recursive_sub_task_start",
                )
                .await;

                let sub_result = self
                    .dispatch_agent(AgentRole::Do, sub_ctx, cycle_id, Some(sub_step), 0)
                    .await?;
                recursive_budget.record_turns(sub_result.turn_count);
                execution_facts.record(&sub_result);

                self.emit_sa_thought(
                    task_iri,
                    &format!(
                        "{}/{} sub-task complete [{}]: {}",
                        idx + 1,
                        total,
                        sub_result.status,
                        sub_def.objective
                    ),
                    "recursive_sub_task_end",
                )
                .await;

                if sub_result.status == "success" || sub_result.status == "partial_success" {
                    let icon = if sub_result.status == "success" {
                        "✅"
                    } else {
                        outcome.partial_count += 1;
                        "⚠️"
                    };
                    sub_summaries.push(format!(
                        "### Sub-task {} {}\n{}",
                        idx + 1,
                        icon,
                        sub_result.summary
                    ));

                    let sub_completion = crate::core::effect::CompletionEnvelope::from_result(
                        &sub_result.status,
                        sub_result.output.as_ref(),
                        &sub_result.summary,
                    );
                    if current_depth < max_depth
                        && sub_result.status == "success"
                        && sub_completion.needs_follow_up_execution()
                    {
                        // Only fully successful sub-tasks continue deeper recursion; partial_success continues in upper recursion
                        self.emit_sa_thought(
                            task_iri,
                            &format!(
                                "Entering deeper recursion (depth {}/{})",
                                current_depth + 1,
                                max_depth
                            ),
                            "recursive_deeper",
                        )
                        .await;
                        match self
                            .execute_recursive_sub_cycle(
                                &sub_result.summary,
                                &sub_completion,
                                task_iri,
                                cycle_id,
                                &format!("{}_sub_{}", parent_step_id, idx),
                                max_depth,
                                current_depth + 1,
                                five_w2h,
                                five_w2h_iri,
                                execution_facts,
                                task_effect_policy,
                                task_constraints,
                                recursive_budget,
                            )
                            .await
                        {
                            Ok(deeper_outcome) => {
                                outcome.failed_count += deeper_outcome.failed_count;
                                outcome.partial_count += deeper_outcome.partial_count;
                                sub_summaries.push(format!(
                                    "#### Deep sub-task (depth={})\n{}",
                                    current_depth + 1,
                                    deeper_outcome.summary
                                ));
                            }
                            Err(e) => {
                                warn!(error = %e, "Deep recursive sub-cycle failed");
                                outcome.failed_count += 1;
                                sub_summaries.push(format!(
                                    "#### Deep sub-task (depth={}) ❌\nRecursive execution failed: {}",
                                    current_depth + 1,
                                    e
                                ));
                            }
                        }
                    }
                } else {
                    outcome.failed_count += 1;
                    sub_summaries.push(format!(
                        "### Sub-task {} ❌\nExecution failed: {}",
                        idx + 1,
                        sub_result.summary
                    ));
                }
            }

            self.emit_sa_thought(
                task_iri,
                &format!(
                    "Recursive sub-cycle complete (depth {}/{})",
                    current_depth, max_depth
                ),
                "recursive_sub_cycle_end",
            )
            .await;
            outcome.summary = sub_summaries.join("\n\n");
            Ok(outcome)
        })
    }
}

#[cfg(test)]
mod terminal_status_tests {
    use super::*;

    #[test]
    fn recursive_budget_is_shared_across_branches_and_turns() {
        let mut budget = RecursiveExecutionBudget::new(2, 10);
        assert_eq!(budget.reserve(8), Some(8));
        budget.record_turns(6);
        assert_eq!(budget.reserve(8), Some(4));
        budget.record_turns(4);
        assert_eq!(budget.reserve(1), None);
    }

    #[test]
    fn recursive_required_effect_is_revalidated_conditionally() {
        let required = crate::core::effect::EffectPolicy::Required {
            effect: crate::core::effect::EffectKind::WorkspaceMutation,
        };
        assert!(matches!(
            recursive_effect_policy(&required, &crate::core::effect::EffectPolicy::None),
            crate::core::effect::EffectPolicy::Conditional {
                effect: crate::core::effect::EffectKind::WorkspaceMutation,
                ..
            }
        ));
        let missing_condition = crate::core::effect::EffectPolicy::Conditional {
            effect: crate::core::effect::EffectKind::StateChange,
            condition: String::new(),
        };
        assert!(matches!(
            recursive_effect_policy(
                &missing_condition,
                &crate::core::effect::EffectPolicy::None
            ),
            crate::core::effect::EffectPolicy::Conditional { condition, .. }
                if !condition.is_empty()
        ));
    }

    #[test]
    fn task_effect_contract_cannot_be_escalated_by_model_generated_steps() {
        use crate::core::effect::{EffectKind, EffectPolicy};

        let model_requested_write = EffectPolicy::Required {
            effect: EffectKind::WorkspaceMutation,
        };
        assert_eq!(
            effective_step_effect_policy(
                AgentRole::Do,
                &model_requested_write,
                &EffectPolicy::EvidenceOnly,
            ),
            EffectPolicy::EvidenceOnly,
            "an application-scoped evidence task must not become a write task"
        );

        let conditional = EffectPolicy::Conditional {
            effect: EffectKind::WorkspaceMutation,
            condition: "only if current state is incomplete".to_string(),
        };
        assert_eq!(
            effective_step_effect_policy(AgentRole::Do, &model_requested_write, &conditional),
            conditional,
            "a model step cannot strengthen a conditional task effect"
        );
        assert_eq!(
            effective_step_effect_policy(
                AgentRole::Check,
                &model_requested_write,
                &EffectPolicy::required_workspace_mutation(),
            ),
            EffectPolicy::EvidenceOnly
        );
    }

    #[test]
    fn residual_task_key_deduplicates_case_and_punctuation() {
        let first = ResidualTaskDef {
            objective: "Verify API wiring!".to_string(),
            role: "Do".to_string(),
            success_criteria: "done".to_string(),
            effect_policy: crate::core::effect::EffectPolicy::None,
        };
        let mut second = first.clone();
        second.objective = " verify-api WIRING ".to_string();
        assert_eq!(residual_task_key(&first), residual_task_key(&second));
        let mut budget = RecursiveExecutionBudget::new(4, 20);
        assert!(budget.claim_residual(&first));
        assert!(!budget.claim_residual(&second));
    }

    #[test]
    fn aa_failure_reenters_pa_while_ca_and_da_failures_stay_executable() {
        assert_eq!(
            failed_business_role_recovery(AgentRole::Act),
            ("ReplanPa", "Task")
        );
        assert_eq!(
            failed_business_role_recovery(AgentRole::Plan),
            ("ReplanPa", "Task")
        );
        assert_eq!(
            failed_business_role_recovery(AgentRole::Check),
            ("RetryDa", "Step")
        );
        assert_eq!(
            failed_business_role_recovery(AgentRole::Do),
            ("RetryDa", "Step")
        );
    }

    #[test]
    fn recursive_subtasks_receive_an_executable_turn_budget() {
        assert_eq!(recursive_subtask_turn_budget(50, 1), 25);
        assert_eq!(recursive_subtask_turn_budget(50, 2), 16);
        assert_eq!(recursive_subtask_turn_budget(50, 3), 12);
        assert_eq!(recursive_subtask_turn_budget(8, 1), 8);
    }

    #[test]
    fn business_role_tool_policy_cannot_be_broadened_by_plan_output() {
        assert_eq!(
            enforce_business_role_tool_policy(AgentRole::Do, None),
            None,
            "a generated PDCA plan cannot implicitly narrow DA's task capability"
        );
        assert_eq!(
            enforce_business_role_tool_policy(
                AgentRole::Act,
                Some(vec!["file_read".into(), "bash".into()])
            ),
            Some(Vec::new()),
            "AA is decision-only even when an LLM plan requests tools"
        );
        assert_eq!(
            enforce_business_role_tool_policy(
                AgentRole::Check,
                Some(vec!["file_read".into(), "file_write".into()])
            ),
            Some(vec!["file_read".into()]),
            "CA may inspect evidence but may not mutate it"
        );
        assert_eq!(
            enforce_business_role_tool_policy(AgentRole::Do, Some(vec!["file_write".into()])),
            Some(vec!["file_write".into()]),
            "DA retains the plan's execution capability"
        );
    }

    #[test]
    fn aa_declared_failure_is_not_flattened_to_runner_success() {
        let mut result = TaskResult {
            task_iri: "iri://task/aa-verdict".into(),
            status: "success".into(),
            verdict: Some(TaskVerdict::Success),
            summary: "CA evidence PASS, but process audit failed; 判定 FAILED".into(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: vec![],
            archive_iri: None,
        };
        apply_aa_declared_verdict(&mut result, None);
        assert_eq!(result.status, "failed");
        assert_eq!(result.verdict, Some(TaskVerdict::Failed));
    }

    #[test]
    fn aa_fullwidth_success_prefix_is_terminal_success() {
        let mut result = TaskResult {
            task_iri: "iri://task/aa-fullwidth-verdict".into(),
            status: "failed".into(),
            verdict: Some(TaskVerdict::Failed),
            summary: "SUCCESS：CA 已验证全部原始要求".into(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: vec![],
            archive_iri: None,
        };
        apply_aa_declared_verdict(&mut result, None);
        assert_eq!(result.status, "success");
        assert_eq!(result.verdict, Some(TaskVerdict::Success));
    }

    #[test]
    fn aa_missing_prefix_converges_from_latest_ca_report() {
        let mut result = TaskResult {
            task_iri: "iri://task/aa-ca-fallback".into(),
            status: "success".into(),
            verdict: Some(TaskVerdict::Success),
            summary: "任务完成，闭环通过".into(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: vec![],
            archive_iri: None,
        };
        let report = crate::core::recovery::AuditReport {
            verdict: crate::core::recovery::AuditVerdict::Pass,
            failed_dimensions: vec![],
            findings: vec![],
            scope: crate::core::recovery::RepairScope::Step,
            reason: None,
        };

        apply_aa_declared_verdict(&mut result, Some(&report));

        assert_eq!(result.status, "success");
        assert_eq!(result.verdict, Some(TaskVerdict::Success));
    }

    #[test]
    fn aa_output_prefix_is_authoritative_when_summary_omits_it() {
        let mut result = TaskResult {
            task_iri: "iri://task/aa-output-verdict".into(),
            status: "success".into(),
            verdict: Some(TaskVerdict::Success),
            summary: "AA accepts after CA PASS".into(),
            output: Some(serde_json::Value::String(
                "SUCCESS: CA audit PASS confirmed the artifact byte-exact".into(),
            )),
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: vec![],
            archive_iri: None,
        };
        let conditional = crate::core::recovery::AuditReport {
            verdict: crate::core::recovery::AuditVerdict::Conditional,
            failed_dimensions: vec![],
            findings: vec![],
            scope: crate::core::recovery::RepairScope::Step,
            reason: None,
        };

        apply_aa_declared_verdict(&mut result, Some(&conditional));

        assert_eq!(result.status, "success");
        assert_eq!(result.verdict, Some(TaskVerdict::Success));
    }

    #[test]
    fn aa_failed_output_prefix_is_not_flattened_by_runner_summary() {
        let mut result = TaskResult {
            task_iri: "iri://task/aa-output-failure".into(),
            status: "success".into(),
            verdict: Some(TaskVerdict::Success),
            summary: "AA reviewed the evidence".into(),
            output: Some(serde_json::Value::String(
                "FAILED: required acceptance evidence is absent".into(),
            )),
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: vec![],
            archive_iri: None,
        };

        apply_aa_declared_verdict(&mut result, None);

        assert_eq!(result.status, "failed");
        assert_eq!(result.verdict, Some(TaskVerdict::Failed));
    }

    #[test]
    fn ca_dimension_failure_forces_terminal_failure() {
        let mut result = TaskResult {
            task_iri: "iri://task/ca-gate".to_string(),
            status: "success".to_string(),
            summary: "AA reported success".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            verdict: Some(TaskVerdict::Success),
            archive_iri: None,
        };

        enforce_ca_audit_terminal_status(&mut result, true);

        assert_eq!(result.status, "failed");
        assert_eq!(result.verdict, Some(TaskVerdict::Failed));
        assert!(result
            .errors
            .iter()
            .any(|error| error.contains("CA dimension audit failed")));
    }

    #[test]
    fn passing_ca_does_not_change_terminal_status() {
        let mut result = TaskResult {
            task_iri: "iri://task/ca-pass".to_string(),
            status: "success".to_string(),
            summary: "all checks passed".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            verdict: Some(TaskVerdict::Success),
            archive_iri: None,
        };

        enforce_ca_audit_terminal_status(&mut result, false);

        assert_eq!(result.status, "success");
        assert!(result.errors.is_empty());
    }

    #[test]
    fn task_execution_facts_preserve_earlier_agent_tools() {
        let action = crate::core::tracked_action::TrackedAction {
            action_id: "action-da-1".to_string(),
            tool_name: "file_write".to_string(),
            agent_role: "DA".to_string(),
            duration_secs: 0.1,
            status: crate::core::tracked_action::ActionStatus::Success,
            files_created: Vec::new(),
            files_modified: Vec::new(),
            files_read: Vec::new(),
            error: None,
            substantive_effect: false,
            tool_args: std::collections::HashMap::new(),
        };
        let da = TaskResult {
            task_iri: "iri://task/facts".to_string(),
            status: "success".to_string(),
            summary: "implemented".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: vec![serde_json::json!({"path": "out.txt"})],
            errors: Vec::new(),
            turn_count: 4,
            tool_call_count: 2,
            five_w2h_updates: None,
            tracked_actions: vec![action],
            verdict: Some(TaskVerdict::Success),
            archive_iri: None,
        };
        let mut aa = TaskResult {
            task_iri: "iri://task/facts".to_string(),
            status: "success".to_string(),
            summary: "accepted".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            verdict: Some(TaskVerdict::Success),
            archive_iri: None,
        };

        let mut facts = TaskExecutionFacts::default();
        facts.record(&da);
        facts.record(&aa);
        facts.apply_to(&mut aa);

        assert_eq!(aa.turn_count, 5);
        assert_eq!(aa.tool_call_count, 2);
        assert_eq!(aa.tracked_actions.len(), 1);
        assert_eq!(aa.tracked_actions[0].tool_name, "file_write");
        assert_eq!(aa.artifacts.len(), 1);
    }

    #[test]
    fn workspace_delivery_reconciliation_recognizes_absolute_artifact_path() {
        let mut facts = TaskExecutionFacts::default();
        facts
            .artifacts
            .push(serde_json::json!({"path": "/tmp/tui-workspace/AI_Agent_Research_Report.md"}));

        assert!(facts.contains_workspace_artifact("AI_Agent_Research_Report.md"));
        assert!(!facts.contains_workspace_artifact("other-report.md"));
    }
}
