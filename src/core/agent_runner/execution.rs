use serde_json::{json, Value};
use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::sync::atomic::Ordering;
use tracing::{debug, info, warn};

/// Resolve one BizAgent's turn budget. Every role inherits the task budget by
/// default; an operator may configure a workload-specific ceiling per role.
/// This avoids silently weakening complex PA/CA/AA work with small fixed caps.
pub(super) fn effective_role_max_turns(
    role: AgentRole,
    requested: u32,
    budget: &crate::config::settings::AgentExecutionBudgetSettings,
) -> u32 {
    let requested = requested.max(1);
    let configured = match role {
        AgentRole::Plan => budget.role_max_turns.plan,
        AgentRole::Do => budget.role_max_turns.do_agent,
        AgentRole::Check => budget.role_max_turns.check,
        AgentRole::Act => budget.role_max_turns.act,
    };
    configured.map_or(requested, |limit| requested.min(limit.max(1)))
}

/// Resolve the explicit provider-side reasoning policy for an ordinary ReAct
/// turn. Keeping this selector next to the role turn-budget selector ensures
/// synchronous and streaming runners use the same role semantics.
pub(super) const fn react_reasoning_effort(
    role: AgentRole,
    budget: &crate::config::settings::AgentExecutionBudgetSettings,
) -> crate::config::settings::ReasoningEffort {
    match role {
        AgentRole::Plan => budget.react_reasoning_effort.plan,
        AgentRole::Do => budget.react_reasoning_effort.do_agent,
        AgentRole::Check => budget.react_reasoning_effort.check,
        AgentRole::Act => budget.react_reasoning_effort.act,
    }
}

/// Select provider-side reasoning for one concrete ReAct dispatch.
///
/// Ordinary role work keeps its configured reasoning budget. Schema-only
/// terminal control turns request `Disabled`; the gateway applies that only
/// for a recognized provider dialect and safely omits it otherwise. The
/// kernel's closed tool window and authenticated receipt directive enforce
/// correctness independent of provider reasoning. The policy is request-local
/// and never alters a fresh Agent/L1.
pub(super) const fn react_reasoning_effort_for_dispatch(
    role: AgentRole,
    budget: &crate::config::settings::AgentExecutionBudgetSettings,
    ca_evidence_close_active: bool,
    da_evidence_close_active: bool,
    da_verification_contract_close_active: bool,
    raw_tool_protocol_correction_dispatch: bool,
) -> crate::config::settings::ReasoningEffort {
    if (matches!(role, AgentRole::Check) && ca_evidence_close_active)
        || (matches!(role, AgentRole::Do) && da_evidence_close_active)
        || (matches!(role, AgentRole::Do) && da_verification_contract_close_active)
        || raw_tool_protocol_correction_dispatch
    {
        crate::config::settings::ReasoningEffort::Disabled
    } else {
        react_reasoning_effort(role, budget)
    }
}

pub(super) const fn ca_evidence_close_directive() -> &'static str {
    "[CA Evidence Close Gate] The configured audit window and its bounded deterministic-verification opportunity are complete. No tools are available: do not request a native tool call and do not print textual tool syntax. Do not reread or rediscover evidence. Return exactly one unfenced outer ReAct JSON object now: its first non-whitespace character must be `{`, `action` must be `finish`, `summary` must begin `PASS:`, `CONDITIONAL_PASS:`, or `FAIL:`, and `content` must be the complete object-valued `ca_audit/v1` audit required by the CA verdict contract (never prose and never a JSON-encoded string). When a ConformanceContract is present, `content` must also contain `design_conformance` with exactly its assigned dimensions and exact relation/evidence identities. Emit no `thought`, commentary, or text before or after the outer object. Use only evidence already observed in this L1. A positive verdict requires the successful executable verification receipt already supplied; represent any unresolved evidence as FAIL/verification_gap in the audit instead of calling another tool."
}

pub(super) const fn da_evidence_close_directive(requires_web_research: bool) -> &'static str {
    if requires_web_research {
        "[DA Evidence Close Gate] Live research is complete. Submit the final result now through the single result-submission function offered in this dispatch. Its `content` must contain the complete requested evidence-backed deliverable (for a report, the full Markdown report including Mermaid blocks and source URLs), not a plan. JSON-escape each real line break exactly once: after the function arguments are decoded, Markdown must contain real newline characters, never visible literal `\\n` text. Use the search results and source material already present in this same L1. A source page rejected by network policy is a disclosed evidence limitation, not a reason to discard successful search evidence or repeat research. Choose `success` when the requested deliverable is complete with those limitations stated; choose `failed` only when the deliverable itself cannot be produced."
    } else {
        "[DA Evidence Close Gate] Evidence collection is complete. Submit the final result now through the single result-submission function offered in this dispatch. Its `content` must contain the complete requested evidence-backed deliverable, not a plan. If it is Markdown, JSON-escape each real line break exactly once so decoded content contains real newlines, never visible literal `\\n` text. Explicitly mark unsupported points as limitations. Choose `success` when the deliverable is complete and `failed` only when it cannot be produced."
    }
}

pub(super) const fn da_terminal_native_protocol_correction_directive(
    requires_web_research: bool,
) -> &'static str {
    if requires_web_research {
        "[DA Native Terminal Protocol Correction] The preceding terminal-only request returned a provider-native call to a historical retrieval tool that was not offered by that request. The call was not executed and its syntax has been removed from this one retry. Submit the complete evidence-backed deliverable through the single offered result-submission function now. For Markdown, JSON-escape each real line break exactly once so decoded content contains real newlines, never visible literal `\\n` text. Use the successful live-search evidence reproduced in this same L1 retry context, preserve source URLs, disclose blocked source reads as limitations, and do not request or describe another retrieval call. This is the only correction opportunity."
    } else {
        "[DA Native Terminal Protocol Correction] The preceding terminal-only request returned a provider-native call to a historical tool that was not offered by that request. The call was not executed and its syntax has been removed from this one retry. Submit the complete deliverable through the single offered result-submission function now, using the evidence reproduced in this same L1 retry context. Do not request or describe another tool call. This is the only correction opportunity."
    }
}

/// Provider-native terminal submission used only after an evidence-only DA's
/// external capability window has closed. Tool-oriented models otherwise
/// sometimes serialize a remembered retrieval call into assistant text. A
/// single explicit terminal function gives that provider decision a typed
/// transport without registering or executing a synthetic runtime tool.
pub(super) const DA_EVIDENCE_RESULT_TOOL_NAME: &str = "submit_da_evidence_result";

pub(super) fn da_evidence_result_tool_definition() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": DA_EVIDENCE_RESULT_TOOL_NAME,
            "description": "Submit the completed DA evidence deliverable. This is a terminal response transport, not an executable external tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The complete requested deliverable. Preserve Markdown, Mermaid, citations, and disclosed limitations."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A concise completion summary without protocol syntax."
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "failed"]
                    }
                },
                "required": ["content", "summary", "outcome"],
                "additionalProperties": false
            }
        }
    })
}

pub(super) fn da_evidence_result_tool_choice() -> String {
    json!({
        "type": "function",
        "function": {"name": DA_EVIDENCE_RESULT_TOOL_NAME}
    })
    .to_string()
}

pub(super) fn add_da_evidence_result_transport(mut tools: Vec<Value>, active: bool) -> Vec<Value> {
    if active {
        // The close policy must already have removed executable tools. Keep a
        // defensive clear here so the terminal transport can never coexist
        // with an external capability because of filter-order drift.
        tools.clear();
        tools.push(da_evidence_result_tool_definition());
    }
    tools
}

/// Convert a provider-native terminal function call into the ordinary ReAct
/// envelope. The provider call id is still admitted and journaled by the
/// caller, but this transport is never sent to ToolExecutor and therefore
/// cannot gain a capability or inflate executed-tool metrics.
pub(super) fn decode_da_evidence_result_submission(
    tool_name: &str,
    arguments: &Value,
) -> Option<Result<String, String>> {
    if tool_name != DA_EVIDENCE_RESULT_TOOL_NAME {
        return None;
    }
    let decode = || {
        let object = arguments.as_object().ok_or_else(|| {
            "DA evidence result submission arguments must be an object".to_string()
        })?;
        let content = object
            .get("content")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "DA evidence result submission content is empty".to_string())?;
        let summary = object
            .get("summary")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "DA evidence result submission summary is empty".to_string())?;
        let outcome = object
            .get("outcome")
            .and_then(Value::as_str)
            .ok_or_else(|| "DA evidence result submission outcome is missing".to_string())?;
        let prefix = match outcome {
            "success" => "SUCCESS:",
            "failed" => "FAILED:",
            _ => {
                return Err(format!(
                    "DA evidence result submission has unsupported outcome '{outcome}'"
                ))
            }
        };
        let summary = if summary
            .get(..prefix.len())
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
        {
            summary.to_string()
        } else {
            format!("{prefix} {summary}")
        };
        Ok(json!({
            "content": content,
            "summary": summary,
            "action": "finish",
            "emphasis": [],
        })
        .to_string())
    };
    Some(decode())
}

pub(super) fn turn_warning_thresholds(
    max_turns: u32,
    early_remaining: u32,
    final_remaining: u32,
) -> (Option<u32>, Option<u32>) {
    let threshold =
        |remaining| (remaining > 0 && remaining < max_turns).then(|| max_turns - remaining);
    let final_turn = threshold(final_remaining);
    let mut early_turn = threshold(early_remaining);
    if early_turn.is_some_and(|early| final_turn.is_some_and(|final_| early >= final_)) {
        early_turn = None;
    }
    (early_turn, final_turn)
}

/// Resolve the close gate for evidence-only DA work.
///
/// A provider may issue several independent live-retrieval calls in one tool
/// turn.  Treating the ordinary close setting as eight *serial* searches lets
/// a research L1 accumulate dozens of calls and an oversized context before
/// synthesis.  Required live research therefore gets exactly one targeted
/// turn after the focus gate, while non-research evidence work retains the
/// operator-configured window.  Both configured zero values keep their
/// documented disable semantics.
pub(super) fn effective_da_evidence_close_turns(
    requires_web_research: bool,
    focus_turns: u32,
    close_turns: u32,
) -> u32 {
    if close_turns == 0 {
        return 0;
    }
    if requires_web_research && focus_turns > 0 {
        std::cmp::min(close_turns, focus_turns.saturating_add(1))
    } else {
        close_turns
    }
}

pub(super) fn requires_workspace_effect(ctx: &TaskContext, role: AgentRole) -> bool {
    role == AgentRole::Do && ctx.effective_effect_policy().requires_workspace_mutation()
}

#[cfg(test)]
pub(super) use crate::core::effect::is_substantive_workspace_effect;
pub(super) use crate::core::effect::is_workspace_mutation_candidate;

/// Shell-like handlers can mutate through scripts, invoked programs or
/// language APIs that no syntax classifier can enumerate. Every foreground
/// call therefore enters the mutation settlement window; excluded cache files
/// remain outside the manifest, while any substantive project mutation is
/// attributed (or fails closed when capture is incomplete).
pub(super) fn requires_workspace_settlement(name: &str) -> bool {
    matches!(
        name,
        "file_write" | "file_edit" | "bash" | "powershell" | "code_execute"
    )
}

#[derive(Debug, Clone)]
pub(super) struct WorkspaceEffectSnapshot {
    manifest: Option<crate::tools::workspace_monitor::WorkspaceEffectManifest>,
    capture_error: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct ConfirmedWorkspaceEffect {
    pub observed: bool,
    /// True only when the call's workspace settlement window produced
    /// complete evidence (or a legacy monitor-less secure file no-op was
    /// explicitly attested). An uncertain call must invalidate delivery
    /// receipts even when no changed path can safely be reported.
    pub settlement_complete: bool,
    pub uncertain: bool,
    /// `Some` also carries incomplete capture diagnostics. Only a complete,
    /// uncontaminated delta may be promoted to path-level delivery evidence.
    pub delta: Option<crate::tools::workspace_monitor::WorkspaceEffectDelta>,
}

#[cfg(test)]
#[allow(dead_code)]
pub(super) fn capture_workspace_effect_snapshot(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
) -> Option<WorkspaceEffectSnapshot> {
    executor
        .read()
        .get_workspace_monitor()
        .map(|monitor| WorkspaceEffectSnapshot {
            manifest: Some(monitor.capture_effect_manifest()),
            capture_error: None,
        })
}

async fn run_workspace_capture_blocking<T, F>(stage: &'static str, capture: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(capture).await.map_err(|error| {
        format!(
            "workspace effect {stage} capture worker failed (panic={}, cancelled={}): {error}",
            error.is_panic(),
            error.is_cancelled()
        )
    })
}

async fn capture_workspace_effect_snapshot_from_monitor(
    monitor: std::sync::Arc<crate::tools::workspace_monitor::WorkspaceMonitor>,
    stage: &'static str,
) -> WorkspaceEffectSnapshot {
    let capture_monitor = monitor.clone();
    match run_workspace_capture_blocking(stage, move || capture_monitor.capture_effect_manifest())
        .await
    {
        Ok(manifest) => WorkspaceEffectSnapshot {
            manifest: Some(manifest),
            capture_error: None,
        },
        Err(error) => {
            warn!(stage, %error, "workspace effect capture failed closed");
            WorkspaceEffectSnapshot {
                manifest: None,
                capture_error: Some(error),
            }
        }
    }
}

/// Capture direct-filesystem evidence without occupying an async runtime
/// worker. The caller keeps the workspace mutation coordinator across this
/// await and the corresponding tool execution/after-capture window.
pub(super) async fn capture_workspace_effect_snapshot_async(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
) -> Option<WorkspaceEffectSnapshot> {
    let monitor = { executor.read().get_workspace_monitor() }?;
    Some(capture_workspace_effect_snapshot_from_monitor(monitor, "before").await)
}

fn incomplete_settlement_evidence(
    before: Option<&WorkspaceEffectSnapshot>,
    stage: crate::tools::workspace_monitor::WorkspaceManifestIssueStage,
    error: String,
    observed: bool,
) -> ConfirmedWorkspaceEffect {
    let unavailable = "unavailable".to_string();
    ConfirmedWorkspaceEffect {
        observed,
        settlement_complete: false,
        uncertain: true,
        delta: Some(crate::tools::workspace_monitor::WorkspaceEffectDelta {
            schema_version:
                crate::tools::workspace_monitor::WORKSPACE_EFFECT_MANIFEST_SCHEMA_VERSION,
            before_digest: before
                .and_then(|snapshot| snapshot.manifest.as_ref())
                .map(|manifest| manifest.digest.clone())
                .unwrap_or_else(|| unavailable.clone()),
            after_digest: unavailable,
            files_created: Vec::new(),
            files_modified: Vec::new(),
            files_removed: Vec::new(),
            directories_created: Vec::new(),
            directories_removed: Vec::new(),
            complete: false,
            errors: vec![crate::tools::workspace_monitor::WorkspaceManifestIssue {
                stage,
                kind:
                    crate::tools::workspace_monitor::WorkspaceManifestIssueKind::IncompleteManifest,
                path: None,
                message: error,
            }],
        }),
    }
}

#[cfg(test)]
mod workspace_capture_runtime_tests {
    use super::{
        confirmed_workspace_effect_evidence, run_workspace_capture_blocking,
        WorkspaceEffectSnapshot,
    };
    use serde_json::json;

    #[test]
    fn capture_worker_does_not_block_a_single_async_runtime_worker() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let started = std::time::Instant::now();
                let capture = run_workspace_capture_blocking("test", || {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    7_u8
                });
                let heartbeat = async {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    started.elapsed()
                };

                let (captured, heartbeat_elapsed) = tokio::join!(capture, heartbeat);
                assert_eq!(captured.unwrap(), 7);
                assert!(
                    heartbeat_elapsed < std::time::Duration::from_millis(250),
                    "single runtime worker was blocked for {heartbeat_elapsed:?}"
                );
            });
    }

    #[test]
    fn capture_worker_join_failure_is_diagnostic() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let error = run_workspace_capture_blocking("before", || -> () {
                    panic!("injected workspace capture panic")
                })
                .await
                .unwrap_err();

                assert!(error.contains("before capture worker failed"), "{error}");
                assert!(error.contains("panic=true"), "{error}");

                let before = WorkspaceEffectSnapshot {
                    manifest: None,
                    capture_error: Some(error.clone()),
                };
                let executor = parking_lot::RwLock::new(crate::tools::ToolExecutor::new());
                let evidence = confirmed_workspace_effect_evidence(
                    &executor,
                    "bash",
                    &json!({"command":"true"}),
                    &json!({"exit_code":0}),
                    Some(&before),
                )
                .await;
                assert!(!evidence.observed);
                assert!(evidence.uncertain);
                assert!(!evidence.settlement_complete);
                let delta = evidence.delta.unwrap();
                assert!(!delta.complete);
                assert!(delta.errors[0].message.contains("panic=true"));
            });
    }
}

/// Confirm a material workspace effect rather than inferring one from tool
/// syntax or trusting a handler's partial attestation. Every settlement tool,
/// including reserved file writes/edits, uses one bounded before/after
/// manifest. Only a monitor-less legacy `changed=false` file result may attest
/// a settled no-op without a manifest. This is execution evidence; CA still
/// decides whether the resulting structure/content satisfies the task.
pub(super) async fn confirmed_workspace_effect_evidence(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
    name: &str,
    args: &Value,
    result: &Value,
    before: Option<&WorkspaceEffectSnapshot>,
) -> ConfirmedWorkspaceEffect {
    let verification_call = is_verification_call(name, args);
    let settlement_required = requires_workspace_settlement(name)
        || is_workspace_mutation_candidate(name, args)
        || verification_call;
    let direct_file_result = matches!(name, "file_write" | "file_edit");
    let direct_file_changed = direct_file_result
        && !crate::core::tracked_action::tool_result_failed(result)
        && result.get("changed").and_then(Value::as_bool) == Some(true);
    let secure_direct_noop = direct_file_result
        && !crate::core::tracked_action::tool_result_failed(result)
        && result.get("changed").and_then(Value::as_bool) == Some(false);

    if result.get("background_task_id").is_some() {
        if settlement_required {
            return incomplete_settlement_evidence(
                before,
                crate::tools::workspace_monitor::WorkspaceManifestIssueStage::Diff,
                "background tool execution cannot be enclosed by the foreground workspace settlement window"
                    .to_string(),
                false,
            );
        }
        return ConfirmedWorkspaceEffect {
            observed: false,
            settlement_complete: !settlement_required,
            uncertain: settlement_required,
            delta: None,
        };
    }
    if !matches!(name, "bash" | "powershell" | "code_execute")
        && !is_workspace_mutation_candidate(name, args)
        && !verification_call
    {
        return ConfirmedWorkspaceEffect {
            observed: false,
            settlement_complete: true,
            uncertain: false,
            delta: None,
        };
    }

    let Some(before) = before else {
        let monitor_missing = executor.read().get_workspace_monitor().is_none();
        if monitor_missing && secure_direct_noop {
            return ConfirmedWorkspaceEffect {
                observed: false,
                settlement_complete: true,
                uncertain: false,
                delta: None,
            };
        }
        warn!(
            tool = name,
            monitor_missing,
            "workspace settlement before-capture is unavailable; marking the call uncertain"
        );
        return incomplete_settlement_evidence(
            None,
            crate::tools::workspace_monitor::WorkspaceManifestIssueStage::Before,
            format!(
                "workspace settlement before-capture is unavailable (monitor_missing={monitor_missing})"
            ),
            direct_file_changed,
        );
    };
    let Some(before_manifest) = before.manifest.as_ref() else {
        return incomplete_settlement_evidence(
            Some(before),
            crate::tools::workspace_monitor::WorkspaceManifestIssueStage::Before,
            before.capture_error.clone().unwrap_or_else(|| {
                "workspace effect before-capture is unavailable without a diagnostic".to_string()
            }),
            false,
        );
    };
    let Some(monitor) = executor.read().get_workspace_monitor() else {
        if secure_direct_noop {
            return ConfirmedWorkspaceEffect {
                observed: false,
                settlement_complete: true,
                uncertain: false,
                delta: None,
            };
        }
        warn!(
            tool = name,
            "workspace monitor disappeared during settlement; marking the call uncertain"
        );
        return incomplete_settlement_evidence(
            Some(before),
            crate::tools::workspace_monitor::WorkspaceManifestIssueStage::After,
            "workspace monitor disappeared during settlement".to_string(),
            direct_file_changed,
        );
    };
    let after = capture_workspace_effect_snapshot_from_monitor(monitor.clone(), "after").await;
    let Some(after_manifest) = after.manifest.as_ref() else {
        return incomplete_settlement_evidence(
            Some(before),
            crate::tools::workspace_monitor::WorkspaceManifestIssueStage::After,
            after.capture_error.unwrap_or_else(|| {
                "workspace effect after-capture is unavailable without a diagnostic".to_string()
            }),
            false,
        );
    };
    let delta = crate::tools::workspace_monitor::WorkspaceEffectDelta::between(
        before_manifest,
        after_manifest,
    );
    let changed = !delta.files_created.is_empty()
        || !delta.files_modified.is_empty()
        || !delta.files_removed.is_empty()
        || !delta.directories_created.is_empty()
        || !delta.directories_removed.is_empty();
    if delta.complete {
        return ConfirmedWorkspaceEffect {
            observed: changed,
            settlement_complete: true,
            uncertain: false,
            delta: Some(delta),
        };
    }

    // Oversized/unreadable manifests retain diagnostics but never grant path
    // ownership or progress. `uncertain` independently forces the action to
    // invalidate older delivery receipts even when watcher generation is
    // disabled, delayed, or unchanged.
    warn!(
        tool = name,
        issues = delta.errors.len(),
        "complete workspace delta unavailable; marking settlement uncertain"
    );
    ConfirmedWorkspaceEffect {
        observed: false,
        settlement_complete: false,
        uncertain: settlement_required,
        delta: Some(delta),
    }
}

/// Compatibility helper for existing policy tests and non-ledger callers.
/// New AgentRunner paths consume `confirmed_workspace_effect_evidence` so the
/// exact delta is not discarded.
#[cfg(test)]
pub(super) async fn confirmed_workspace_effect(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
    name: &str,
    args: &Value,
    result: &Value,
    before: Option<&WorkspaceEffectSnapshot>,
) -> bool {
    confirmed_workspace_effect_evidence(executor, name, args, result, before)
        .await
        .observed
}

/// A same-role child lease contains exact writable file claims. Shell syntax
/// is not a trustworthy source of paths, so validate the observed delta after
/// execution. Files must match an exact writable claim; directory creation or
/// removal is allowed only when it is an ancestor of such a claim. Detection
/// happens after the process and therefore cannot undo an escape, but marking
/// the action contaminated prevents it from becoming delivery evidence and
/// forces recovery instead of silently re-attributing the change.
pub(super) fn workspace_delta_violates_lease(
    delta: &crate::tools::workspace_monitor::WorkspaceEffectDelta,
    lease: Option<&crate::core::effect::WorkspaceResourceLease>,
) -> bool {
    let Some(lease) = lease else {
        return false;
    };
    if !delta.complete {
        return true;
    }
    let writable = lease
        .paths
        .iter()
        .filter(|entry| {
            matches!(
                entry.access,
                crate::core::effect::WorkspaceLeaseAccess::Write
                    | crate::core::effect::WorkspaceLeaseAccess::Exclusive
            )
        })
        .map(|entry| entry.relative_path.as_str())
        .collect::<Vec<_>>();
    let file_allowed = |path: &str| writable.contains(&path);
    let directory_allowed = |path: &str| {
        let prefix = format!("{}/", path.trim_end_matches('/'));
        writable
            .iter()
            .any(|candidate| candidate.starts_with(&prefix))
    };

    delta
        .files_created
        .iter()
        .map(|file| file.path.as_str())
        .chain(delta.files_modified.iter().map(|file| file.path.as_str()))
        .chain(delta.files_removed.iter().map(|file| file.path.as_str()))
        .any(|path| !file_allowed(path))
        || delta
            .directories_created
            .iter()
            .chain(delta.directories_removed.iter())
            .any(|path| !directory_allowed(path))
}

/// The handler may have succeeded and mutated the workspace before a
/// post-execution policy withholds its result. Preserve the exact effect
/// receipt, but make the action terminally failed so it cannot satisfy a
/// delivery or verification gate.
pub(super) fn mark_last_action_post_hook_denied(
    tracker: &mut crate::core::tracked_action::ActionTracker,
    disclosed_result: &Value,
) {
    let Some(action) = tracker.actions.last_mut() else {
        return;
    };
    action.status = crate::core::tracked_action::ActionStatus::Failed;
    action.error = disclosed_result
        .get("error")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| Some("tool result withheld by post-execution hook policy".to_string()));
    action.successful_verification = false;
}

/// Update the DA progress tail after one tool-call turn. `effect_observed`
/// remains the all-time completion evidence, while the consecutive counter is
/// deliberately reset and resumed throughout the whole execution. Keeping
/// those two meanings separate prevents one early write from disabling
/// detection of a much later inspection-only stall.
pub(super) fn record_workspace_effect_turn(
    effect_observed: &mut bool,
    consecutive_effectless_tool_turns: &mut u32,
    effect_succeeded_this_turn: bool,
) {
    if effect_succeeded_this_turn {
        *effect_observed = true;
        *consecutive_effectless_tool_turns = 0;
    } else {
        *consecutive_effectless_tool_turns = consecutive_effectless_tool_turns.saturating_add(1);
    }
}

pub(super) fn workspace_effect_recovery_active(
    workspace_effect_required: bool,
    consecutive_effectless_tool_turns: u32,
    low_novelty_evidence_calls: u32,
    effect_block_turns: u32,
) -> bool {
    workspace_effect_required
        && effect_block_turns > 0
        && consecutive_effectless_tool_turns.max(low_novelty_evidence_calls) >= effect_block_turns
}

fn is_workspace_mutation_tool_name(name: &str) -> bool {
    matches!(
        name,
        "file_write" | "file_edit" | "bash" | "powershell" | "code_execute"
    )
}

/// During recovery, advertise only tools that can make the required workspace
/// change. A corrective Repair epoch may additionally expose one exact
/// `file_read`; any recovery epoch may retain bounded, session-owned result
/// readers so compression cannot make already-produced evidence unreachable.
/// This is an intersection with the already role/SA-authorized window and
/// therefore never broadens authority.
pub(super) fn mutation_recovery_tool_definitions(
    definitions: Vec<Value>,
    targeted_repair_read_available: bool,
    archived_result_read_available: bool,
) -> Vec<Value> {
    definitions
        .into_iter()
        .filter(|definition| {
            definition["function"]["name"].as_str().is_some_and(|name| {
                is_workspace_mutation_tool_name(name)
                    || (targeted_repair_read_available && name == "file_read")
                    || (archived_result_read_available && ToolExecutor::is_micro_tool_name(name))
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExecutionPhase {
    Inspect,
    Implement,
    Verify,
    Repair,
}

/// Per-recovery-epoch escape hatch for criterion-linked diagnosis.
///
/// Mutation recovery still prevents a new discovery loop, but a corrective DA
/// must not be forced to edit a target it has never read or lose access to a
/// result page solely because compression and mutation recovery crossed in the
/// same turn. The allowances are deliberately hard bounded and consumed before
/// hooks/tool execution, including failed attempts. A substantive mutation
/// leaves recovery and a later stalled/failed phase opens a fresh epoch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct RepairBaselineWindow {
    recovery_epoch_active: bool,
    recovery_phase: Option<ExecutionPhase>,
    targeted_read_paths: HashSet<String>,
    unscoped_targeted_read_used: bool,
    archived_result_reads_used: u8,
    verification_checks_used: u8,
    last_verification_command: Option<String>,
    last_verification_inconclusive_reason: Option<String>,
    correction_verification_used: bool,
}

impl RepairBaselineWindow {
    const MAX_TARGETED_READ_PATHS: usize = 8;
    const MAX_ARCHIVED_RESULT_READS: u8 = 4;
    const MAX_VERIFICATION_CHECKS: u8 = 1;
    const MAX_CORRECTION_VERIFICATION_RETRIES: u8 = 1;

    pub(super) fn update_epoch(&mut self, phase: ExecutionPhase, recovery_active: bool) {
        // A failed verifier can move an already-active Implement/Verify
        // recovery epoch into Repair without an intervening inactive turn.
        // That transition has a new criterion-linked defect and therefore
        // needs a fresh (still bounded) baseline/check allowance.  Do not
        // reset on repeated Repair turns, otherwise a model could manufacture
        // an unbounded read loop simply by retrying denied calls.
        let entered_repair = phase == ExecutionPhase::Repair
            && self.recovery_phase.is_some_and(|prior| prior != phase);
        if recovery_active && (!self.recovery_epoch_active || entered_repair) {
            self.recovery_epoch_active = true;
            self.targeted_read_paths.clear();
            self.unscoped_targeted_read_used = false;
            self.archived_result_reads_used = 0;
            self.verification_checks_used = 0;
            self.last_verification_command = None;
            self.last_verification_inconclusive_reason = None;
            self.correction_verification_used = false;
        } else if !recovery_active {
            self.recovery_epoch_active = false;
            self.last_verification_command = None;
            self.last_verification_inconclusive_reason = None;
            self.correction_verification_used = false;
        }
        self.recovery_phase = recovery_active.then_some(phase);
    }

    pub(super) fn targeted_read_available(
        &self,
        workspace_resource_lease: Option<&crate::core::effect::WorkspaceResourceLease>,
    ) -> bool {
        let target_available = if workspace_resource_lease.is_some() {
            self.targeted_read_paths.len() < Self::MAX_TARGETED_READ_PATHS
        } else {
            !self.unscoped_targeted_read_used
        };
        self.recovery_epoch_active
            && matches!(
                self.recovery_phase,
                Some(ExecutionPhase::Implement | ExecutionPhase::Repair)
            )
            && target_available
    }

    pub(super) fn verification_check_available(&self) -> bool {
        self.recovery_epoch_active
            && (self.verification_checks_used < Self::MAX_VERIFICATION_CHECKS
                || (!self.correction_verification_used
                    && Self::MAX_CORRECTION_VERIFICATION_RETRIES > 0
                    && self.last_verification_inconclusive_reason.is_some()))
    }

    /// Record the typed result of the verifier most recently admitted by this
    /// recovery window. A second check is exposed only when that exact call
    /// was inconclusive, and the follow-up must use a different normalized
    /// command. Failed checks require a mutation; passed checks leave recovery.
    pub(super) fn record_verification_assessment(
        &mut self,
        name: &str,
        args: &Value,
        assessment: &crate::core::tracked_action::VerificationAssessment,
    ) {
        if !self.recovery_epoch_active || self.verification_checks_used == 0 {
            return;
        }
        let normalized = normalized_verification_command(name, args);
        if self.last_verification_command.as_deref() != Some(normalized.as_str()) {
            return;
        }
        self.last_verification_inconclusive_reason = (assessment.outcome
            == crate::core::tracked_action::VerificationOutcome::Inconclusive)
            .then(|| assessment.reason.clone())
            .flatten()
            .filter(|reason| !reason.trim().is_empty());
    }

    pub(super) fn archived_result_read_available(&self) -> bool {
        self.recovery_epoch_active
            && self.archived_result_reads_used < Self::MAX_ARCHIVED_RESULT_READS
    }

    /// Authorize one provider-issued call under the mutation-recovery gate.
    /// Mutation candidates retain their normal authority. Session-owned
    /// archived-result pages are bounded in every recovery phase; ordinary
    /// baseline reads and verification probes remain Repair-only.
    pub(super) fn authorize_call(
        &mut self,
        recovery_active: bool,
        phase: ExecutionPhase,
        name: &str,
        args: &Value,
        workspace_resource_lease: Option<&crate::core::effect::WorkspaceResourceLease>,
    ) -> Result<(), MutationRecoveryRejection> {
        if !recovery_active {
            return Ok(());
        }
        if matches!(name, "file_write" | "file_edit") && is_workspace_mutation_candidate(name, args)
        {
            return Ok(());
        }
        if matches!(name, "bash" | "powershell" | "code_execute")
            && is_workspace_mutation_candidate(name, args)
        {
            return Err(MutationRecoveryRejection::ShellMutationWithheld);
        }
        if !self.recovery_epoch_active {
            return Err(MutationRecoveryRejection::RecoveryEpochUnavailable);
        }
        // Session-scoped micro tools only page a result produced by this same
        // Agent/L1 execution. They recover already-observed evidence without
        // granting a new filesystem search or cross-agent capability.
        if ToolExecutor::is_micro_tool_name(name) && self.archived_result_read_available() {
            self.archived_result_reads_used = self.archived_result_reads_used.saturating_add(1);
            return Ok(());
        }
        if name == "file_read" {
            if !matches!(phase, ExecutionPhase::Implement | ExecutionPhase::Repair) {
                return Err(MutationRecoveryRejection::WrongPhase);
            }
            let starts_at_beginning = args.get("offset").and_then(Value::as_u64).unwrap_or(0) == 0;
            let has_limit = args.get("limit").is_some_and(|value| !value.is_null());
            let complete_mode = !matches!(
                args.get("mode").and_then(Value::as_str),
                Some("diff" | "changed_only")
            );
            if !starts_at_beginning || has_limit || !complete_mode {
                return Err(MutationRecoveryRejection::WholeFileBaselineRequired);
            }
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or(MutationRecoveryRejection::UnboundMutationTarget)?;
            let target_key = if let Some(lease) = workspace_resource_lease {
                lease
                    .resolve_authorized_file_tool_path("file_write", args)
                    .map_err(|_| MutationRecoveryRejection::UnboundMutationTarget)?
                    .ok_or(MutationRecoveryRejection::UnboundMutationTarget)?
                    .to_string_lossy()
                    .to_string()
            } else {
                if self.unscoped_targeted_read_used {
                    return Err(MutationRecoveryRejection::TargetBaselineAlreadyRead);
                }
                self.unscoped_targeted_read_used = true;
                path.to_string()
            };
            if self.targeted_read_paths.contains(&target_key) {
                return Err(MutationRecoveryRejection::TargetBaselineAlreadyRead);
            }
            if self.targeted_read_paths.len() >= Self::MAX_TARGETED_READ_PATHS {
                return Err(MutationRecoveryRejection::BaselineTargetLimitExhausted);
            }
            self.targeted_read_paths.insert(target_key);
            return Ok(());
        }
        if is_verification_call(name, args) {
            if phase != ExecutionPhase::Repair {
                return Err(MutationRecoveryRejection::WrongPhase);
            }
            let normalized = normalized_verification_command(name, args);
            if self.verification_checks_used < Self::MAX_VERIFICATION_CHECKS {
                self.verification_checks_used = self.verification_checks_used.saturating_add(1);
                self.last_verification_command = Some(normalized);
                self.last_verification_inconclusive_reason = None;
                return Ok(());
            }
            let correction_available = !self.correction_verification_used
                && Self::MAX_CORRECTION_VERIFICATION_RETRIES > 0
                && self.last_verification_inconclusive_reason.is_some()
                && self.last_verification_command.as_deref() != Some(normalized.as_str());
            if correction_available {
                self.correction_verification_used = true;
                self.verification_checks_used = self.verification_checks_used.saturating_add(1);
                self.last_verification_command = Some(normalized);
                self.last_verification_inconclusive_reason = None;
                return Ok(());
            }
            return Err(MutationRecoveryRejection::VerificationLimitExhausted);
        }
        Err(MutationRecoveryRejection::DiscoveryWithheld)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MutationRecoveryRejection {
    RecoveryEpochUnavailable,
    WrongPhase,
    WholeFileBaselineRequired,
    UnboundMutationTarget,
    TargetBaselineAlreadyRead,
    BaselineTargetLimitExhausted,
    VerificationLimitExhausted,
    ShellMutationWithheld,
    DiscoveryWithheld,
}

pub(super) fn repair_recovery_rejection(name: &str, rejection: MutationRecoveryRejection) -> Value {
    let (reason, message) = match rejection {
        MutationRecoveryRejection::WrongPhase => (
            "repair_capability_wrong_phase",
            format!("This {name} invocation is not admissible in the current recovery phase."),
        ),
        MutationRecoveryRejection::WholeFileBaselineRequired => (
            "repair_whole_file_baseline_required",
            "A mutation-target baseline must read the complete current file (offset 0, no limit, and not diff/changed_only mode).".to_string(),
        ),
        MutationRecoveryRejection::UnboundMutationTarget => (
            "repair_unbound_mutation_target",
            "The requested file is not an exact Write/Exclusive target in the kernel-issued workspace lease.".to_string(),
        ),
        MutationRecoveryRejection::TargetBaselineAlreadyRead => (
            "repair_target_baseline_already_read",
            "The complete baseline for this exact mutation target was already read in the current recovery epoch.".to_string(),
        ),
        MutationRecoveryRejection::BaselineTargetLimitExhausted => (
            "repair_baseline_target_limit_exhausted",
            "The bounded number of distinct mutation-target baselines is exhausted for this recovery epoch.".to_string(),
        ),
        MutationRecoveryRejection::VerificationLimitExhausted => (
            "repair_verification_limit_exhausted",
            "The bounded deterministic-verification allowance is exhausted for this recovery epoch. Repeating the same normalized verifier command is never a correction retry.".to_string(),
        ),
        MutationRecoveryRejection::ShellMutationWithheld => (
            "repair_shell_mutation_withheld",
            "Shell/code mutation is withheld during isolated recovery because it can bypass exact-target baseline CAS. Use file_write/file_edit for mutation; shell remains available for one deterministic verifier and, only after a typed Inconclusive result, one different normalized correction command.".to_string(),
        ),
        MutationRecoveryRejection::RecoveryEpochUnavailable => (
            "repair_recovery_epoch_unavailable",
            "The mutation-recovery evidence epoch is not active.".to_string(),
        ),
        MutationRecoveryRejection::DiscoveryWithheld => (
            "repair_requires_mutation_or_verification",
            if matches!(name, "bash" | "powershell" | "code_execute") {
                format!(
                    "This {name} invocation was not executed because it is neither an exact file mutation nor an admissible deterministic verification. The {name} tool itself remains available in the advertised tool set, but this rejection does not pre-authorize another invocation: a later call is admitted only while the bounded Repair verification slot remains, and may contain only safe setup followed by exactly one final verification process. Broad discovery remains withheld during criterion-linked mutation recovery."
                )
            } else {
                format!("This {name} invocation was not executed because it is neither an exact file mutation nor an admissible deterministic verification. Broad discovery remains withheld during criterion-linked mutation recovery.")
            },
        ),
    };
    json!({
        "status": "not_executed",
        "reason": reason,
        "message": message,
        "required_next_action": "Make the criterion-linked mutation with an advertised write-capable tool. When a workspace lease is present, mutate only an exact Write/Exclusive target; without a workspace lease, use the single unscoped complete-file baseline only if its bounded allowance remains and a baseline is needed before editing. Otherwise, if the bounded verification slot remains, invoke one advertised verifier with safe setup followed by exactly one final verification process; or finish with FAILED and the exact blocker."
    })
}

/// Privacy-safe classification for provider-native tool protocol text leaked
/// into `message.content`. Only this shape label and aggregate byte counts are
/// emitted to live diagnostics; the original provider response remains solely
/// in the execution journal under its configured payload-retention policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RawToolProtocolShape {
    Dsml,
    Xml,
    JsonToolCalls,
    JsonFunctionCall,
}

impl RawToolProtocolShape {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Dsml => "dsml",
            Self::Xml => "xml",
            Self::JsonToolCalls => "json_tool_calls",
            Self::JsonFunctionCall => "json_function_call",
        }
    }
}

fn provider_function_object(value: &Value) -> bool {
    value.as_object().is_some_and(|function| {
        function.get("name").is_some_and(Value::is_string) && function.contains_key("arguments")
    })
}

fn direct_json_tool_protocol_shape(value: &Value) -> Option<RawToolProtocolShape> {
    let object = value.as_object()?;
    if object.get("tool_calls").is_some_and(|calls| {
        calls.as_array().is_some_and(|calls| {
            !calls.is_empty()
                && calls.iter().all(|call| {
                    call.as_object()
                        .and_then(|call| call.get("function"))
                        .is_some_and(provider_function_object)
                })
        })
    }) {
        return Some(RawToolProtocolShape::JsonToolCalls);
    }

    let no_business_envelope = !object.contains_key("content")
        && !object.contains_key("summary")
        && !object.contains_key("action");
    let direct_function =
        no_business_envelope && object.get("function").is_some_and(provider_function_object);
    let flattened_function = no_business_envelope
        && object.get("name").is_some_and(Value::is_string)
        && object.contains_key("arguments")
        && (object.get("type").is_none()
            || object.get("type").and_then(Value::as_str) == Some("function"));
    (direct_function || flattened_function).then_some(RawToolProtocolShape::JsonFunctionCall)
}

fn json_tool_protocol_shape(value: &Value) -> Option<RawToolProtocolShape> {
    if let Some(shape) = direct_json_tool_protocol_shape(value) {
        return Some(shape);
    }
    let object = value.as_object()?;
    // Some Chat-compatible adapters expose the assistant message itself as a
    // top-level transport wrapper. Admit exactly one such wrapper and only
    // transport metadata keys; never recurse through arbitrary `evidence`,
    // `result`, or CA audit objects.
    const MESSAGE_WRAPPER_KEYS: &[&str] = &[
        "message",
        "id",
        "model",
        "finish_reason",
        "index",
        "object",
        "created",
        "usage",
    ];
    if !object.contains_key("message")
        || object
            .keys()
            .any(|key| !MESSAGE_WRAPPER_KEYS.contains(&key.as_str()))
    {
        return None;
    }
    object
        .get("message")
        .and_then(direct_json_tool_protocol_shape)
}

/// Inspect actual JSON fence bodies rather than scanning arbitrary prose for
/// protocol field names. A provider may prefix its leaked fence with a label,
/// so every syntactically closed JSON fence is checked independently.
fn fenced_json_tool_protocol_shape(content: &str) -> Option<RawToolProtocolShape> {
    let mut remaining = content;
    while let Some(open) = remaining.find("```") {
        let after_open = &remaining[open + 3..];
        let Some(first_newline) = after_open.find('\n') else {
            return None;
        };
        let language = after_open[..first_newline].trim();
        let fenced_body = &after_open[first_newline + 1..];
        let Some(close) = fenced_body.find("```") else {
            return None;
        };
        if language.is_empty() || language.eq_ignore_ascii_case("json") {
            if let Some(shape) = serde_json::from_str::<Value>(fenced_body[..close].trim())
                .ok()
                .as_ref()
                .and_then(json_tool_protocol_shape)
            {
                return Some(shape);
            }
        }
        remaining = &fenced_body[close + 3..];
    }
    None
}

/// Provider-native tool protocols can leak into `message.content` even when a
/// gateway also decoded a structured tool call. Such transport text is useful
/// for audit storage, but it is not an agent deliverable or CA evidence.
pub(super) fn raw_tool_protocol_shape(content: &str) -> Option<RawToolProtocolShape> {
    let trimmed = content.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
        return None;
    }
    let lower = trimmed.to_lowercase();
    // Detect protocol markup anywhere, including fenced blocks, provider
    // labels and prose-prefixed transport leaks. A mixed transport/business
    // field is rejected as a whole because partial tag stripping can turn
    // tool arguments into misleading audit prose.
    let dsml = lower.contains("dsml") && lower.contains("tool_calls") && lower.contains("invoke");
    let xml_protocol = (lower.contains("<tool_call")
        || lower.contains("<tool_calls")
        || lower.contains("<function_call")
        || lower.contains("<function_calls"))
        && (lower.contains("</tool_call") || lower.contains("</function_call"));
    if dsml {
        return Some(RawToolProtocolShape::Dsml);
    }
    if xml_protocol {
        return Some(RawToolProtocolShape::Xml);
    }

    serde_json::from_str::<Value>(trimmed)
        .ok()
        .as_ref()
        .and_then(json_tool_protocol_shape)
        .or_else(|| fenced_json_tool_protocol_shape(trimmed))
}

pub(super) fn is_raw_tool_protocol_content(content: &str) -> bool {
    raw_tool_protocol_shape(content).is_some()
}

/// Kernel disposition for provider-native tool protocol text that leaked into
/// an ordinary assistant content field.  The correction budget belongs to one
/// concrete `exec` invocation (and therefore one Agent/L1 execution), never to
/// a role, task, or process-global cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RawToolProtocolDisposition {
    /// This is either normal business content or a genuine provider-native
    /// structured tool-call response.  Preserve the existing execution path.
    Normal,
    /// Reject the transport text and give this L1 one protocol-only retry.
    CorrectOnce,
    /// The one correction was already consumed.  Terminate fail-closed rather
    /// than allowing an unbounded provider-format loop.
    RepeatedViolation,
}

pub(super) fn classify_raw_tool_protocol_response(
    finish_reason: &str,
    has_structured_tool_calls: bool,
    content: &str,
    correction_used: &mut bool,
) -> RawToolProtocolDisposition {
    let leaked_protocol = matches!(finish_reason, "stop" | "end_turn")
        && !has_structured_tool_calls
        && is_raw_tool_protocol_content(content);
    if !leaked_protocol {
        return RawToolProtocolDisposition::Normal;
    }
    if *correction_used {
        RawToolProtocolDisposition::RepeatedViolation
    } else {
        *correction_used = true;
        RawToolProtocolDisposition::CorrectOnce
    }
}

pub(super) fn raw_tool_protocol_correction_directive(
    role: AgentRole,
    advertised_tools: &std::collections::HashSet<String>,
    ca_evidence_close_active: bool,
    da_evidence_close_active: bool,
    requires_web_research: bool,
) -> String {
    if matches!(role, AgentRole::Check) && ca_evidence_close_active {
        return "[CA Terminal-Only Format Correction] The preceding response was a textual tool request, so it was rejected and nothing was executed. The CA evidence window is closed and there are no tools: do not request either a native or textual tool call. Use the evidence already observed in this same isolated L1 and return exactly one unfenced outer ReAct JSON object. Required outer shape: {\"content\":{\"schema_version\":\"ca_audit/v1\",\"overall_verdict\":\"pass|conditional_pass|fail\",\"dimensions\":{\"what\":{\"status\":\"pass|conditional_pass|fail\",\"evidence\":\"...\"},\"why\":{\"status\":\"pass|conditional_pass|fail\",\"evidence\":\"...\",\"criteria\":[...] }},\"issues\":[],\"recommendations\":[],\"design_conformance\":{...}},\"summary\":\"PASS: ...|CONDITIONAL_PASS: ...|FAIL: ...\",\"action\":\"finish\",\"emphasis\":[]}. Omit `design_conformance` only when no ConformanceContract was assigned; otherwise include exactly the assigned dimensions and exact relation/evidence identities from that contract. Emit no `thought`, prose, fence, encoded JSON string, or text before/after the object. If evidence is insufficient, encode FAIL with `verification_gap`; never call another tool. This is the only terminal-format correction for this Agent/L1 execution.".to_string();
    }
    if matches!(role, AgentRole::Do) && da_evidence_close_active {
        return format!(
            "[DA Terminal-Only Format Correction] The preceding response was a textual tool request, so it was rejected and nothing was executed. This is the only terminal-format correction for this Agent/L1 execution. {}",
            da_evidence_close_directive(requires_web_research)
        );
    }
    let mut advertised = advertised_tools.iter().cloned().collect::<Vec<_>>();
    advertised.sort();
    let advertised = if advertised.is_empty() {
        "none are advertised; return a normal terminal ReAct JSON response instead".to_string()
    } else {
        format!(
            "only these currently advertised functions may be called: {}",
            advertised.join(", ")
        )
    };
    let terminal_contract = " If no tool is needed or available, return exactly one unfenced outer ReAct JSON object satisfying the current role's terminal contract; emit no commentary before or after it.";
    format!(
        "[Provider-Native Tool Protocol Correction] The preceding provider response exposed a textual DSML/XML/JSON tool protocol in assistant content. It was rejected and no textual tool request was executed. If a tool is needed, emit a provider-native structured `tool_calls` entry; {advertised}. Do not print, fence, or describe tool-call protocol markup in `content`, and do not invent a call id.{terminal_contract} This is the only protocol-correction opportunity for this Agent/L1 execution."
    )
}

pub(super) const REPEATED_RAW_TOOL_PROTOCOL_FAILURE: &str = r#"{"content":"","summary":"FAILED: provider repeated textual tool-call protocol after the single bounded native-protocol correction","action":"finish","emphasis":[]}"#;

/// Content eligible to cross an agent-role boundary as a business handoff.
/// Provider reasoning remains available in the interaction trace, but it is
/// not a typed role output and therefore cannot become downstream evidence.
pub(super) fn is_business_handoff_content(content: &str, content_from_reasoning: bool) -> bool {
    !content_from_reasoning && is_substantive_analysis_content(content)
}

pub(super) fn is_ca_audit_content(content: &str, content_from_reasoning: bool) -> bool {
    is_business_handoff_content(content, content_from_reasoning)
}

pub(super) fn is_substantive_analysis_content(content: &str) -> bool {
    let trimmed = content.trim();
    !trimmed.is_empty()
        && !trimmed.eq_ignore_ascii_case("null")
        && !is_raw_tool_protocol_content(trimmed)
}

/// The CA prompt defines the summary prefix as the machine-readable terminal
/// contract. Do not infer PASS from free-form prose: absence of this prefix is
/// an incomplete audit, not success. Full-width punctuation is accepted for
/// Chinese model output while token boundaries remain strict.
pub(crate) fn structured_ca_verdict(summary: &str) -> Option<TaskVerdict> {
    let normalized = summary.trim_start().to_uppercase();
    [
        ("CONDITIONAL_PASS", TaskVerdict::PartialSuccess),
        ("PASS", TaskVerdict::Success),
        ("FAIL", TaskVerdict::Failed),
        ("有条件通过", TaskVerdict::PartialSuccess),
        ("不通过", TaskVerdict::Failed),
        ("通过", TaskVerdict::Success),
        ("失败", TaskVerdict::Failed),
    ]
    .into_iter()
    .find_map(|(prefix, verdict)| {
        normalized.strip_prefix(prefix).and_then(|tail| {
            let boundary = tail.chars().next();
            (tail.is_empty()
                || boundary
                    .is_some_and(|ch| ch.is_whitespace() || matches!(ch, ':' | '：' | '-' | '—')))
            .then_some(verdict)
        })
    })
}

fn ca_claim_status(value: &Value) -> Option<TaskVerdict> {
    match value.as_str()?.trim().to_ascii_lowercase().as_str() {
        "pass" | "passed" | "success" => Some(TaskVerdict::Success),
        "conditional" | "conditional_pass" | "partial" | "partial_success" => {
            Some(TaskVerdict::PartialSuccess)
        }
        "fail" | "failed" => Some(TaskVerdict::Failed),
        _ => None,
    }
}

fn ca_claim_has_evidence(value: Option<&Value>) -> bool {
    match value {
        Some(Value::String(text)) => !text.trim().is_empty(),
        Some(Value::Array(items)) => {
            !items.is_empty()
                && items
                    .iter()
                    .all(|item| item.as_str().is_some_and(|text| !text.trim().is_empty()))
        }
        _ => false,
    }
}

fn ca_worst_verdict(left: TaskVerdict, right: TaskVerdict) -> TaskVerdict {
    match (left, right) {
        (TaskVerdict::Failed, _) | (_, TaskVerdict::Failed) => TaskVerdict::Failed,
        (TaskVerdict::PartialSuccess, _) | (_, TaskVerdict::PartialSuccess) => {
            TaskVerdict::PartialSuccess
        }
        _ => TaskVerdict::Success,
    }
}

const CA_DESIGN_CONFORMANCE_DIMENSIONS: [&str; 5] = [
    "file_layout",
    "public_interfaces",
    "behavior_and_data_flow",
    "architecture_and_algorithms",
    "user_documentation",
];

fn ca_safe_workspace_relative_path(value: Option<&Value>) -> Option<&str> {
    let path = value?.as_str()?.trim();
    if path.is_empty() || std::path::Path::new(path).is_absolute() {
        return None;
    }
    let safe = std::path::Path::new(path).components().all(|component| {
        matches!(
            component,
            std::path::Component::Normal(_) | std::path::Component::CurDir
        )
    });
    safe.then_some(path)
}

fn ca_failure_class(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|class| {
            matches!(
                *class,
                "observed_defect" | "verification_gap" | "external_blocker"
            )
        })
}

fn ca_valid_failure_class(value: Option<&Value>) -> bool {
    ca_failure_class(value).is_some()
}

fn ca_successor_evidence_claim(
    value: &Value,
) -> Result<(String, super::VerifiedConformanceSuccessorEvidence), &'static str> {
    let evidence = value
        .as_object()
        .ok_or("CA successor evidence must be an object")?;
    let work_package_id = evidence
        .get("work_package_id")
        .and_then(Value::as_str)
        .filter(|id| super::valid_contract_id(id))
        .ok_or("CA successor evidence has an invalid work_package_id")?;
    if !ca_claim_has_evidence(evidence.get("observation")) {
        return Err("CA successor evidence requires a concrete observation");
    }
    let evidence_kind = evidence
        .get("evidence_kind")
        .and_then(Value::as_str)
        .ok_or("CA successor evidence is missing evidence_kind")?;
    let claim = match evidence_kind {
        "artifact_delivery" => {
            if evidence.get("verification_receipt_sha256s").is_some()
                || evidence.keys().any(|key| {
                    !matches!(
                        key.as_str(),
                        "work_package_id" | "evidence_kind" | "paths" | "observation"
                    )
                })
            {
                return Err("CA artifact delivery evidence has incompatible or unknown fields");
            }
            let paths = evidence
                .get("paths")
                .and_then(Value::as_array)
                .filter(|paths| !paths.is_empty())
                .ok_or("CA artifact delivery evidence requires non-empty paths")?;
            let parsed = paths
                .iter()
                .map(|path| {
                    ca_safe_workspace_relative_path(Some(path))
                        .map(str::to_string)
                        .ok_or("CA artifact delivery path is unsafe")
                })
                .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
            if parsed.len() != paths.len() {
                return Err("CA artifact delivery paths must be unique");
            }
            super::VerifiedConformanceSuccessorEvidence::ArtifactPaths(parsed)
        }
        "verification_execution" => {
            if evidence.get("paths").is_some()
                || evidence.keys().any(|key| {
                    !matches!(
                        key.as_str(),
                        "work_package_id"
                            | "evidence_kind"
                            | "verification_receipt_sha256s"
                            | "observation"
                    )
                })
            {
                return Err(
                    "CA verification execution evidence has incompatible or unknown fields",
                );
            }
            let receipts = evidence
                .get("verification_receipt_sha256s")
                .and_then(Value::as_array)
                .filter(|receipts| !receipts.is_empty())
                .ok_or("CA verification execution evidence requires non-empty receipts")?;
            let parsed = receipts
                .iter()
                .map(|receipt| {
                    receipt
                        .as_str()
                        .filter(|receipt| super::valid_prefixed_sha256(receipt))
                        .map(str::to_string)
                        .ok_or("CA verification execution receipt is invalid")
                })
                .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
            if parsed.len() != receipts.len() {
                return Err("CA verification execution receipts must be unique");
            }
            super::VerifiedConformanceSuccessorEvidence::VerificationReceipts(parsed)
        }
        _ => return Err("CA successor evidence has an invalid evidence_kind"),
    };
    Ok((work_package_id.to_string(), claim))
}

fn ca_design_relation_comparison_claim(
    comparison: &serde_json::Map<String, Value>,
) -> Result<((&str, &str), TaskVerdict), &'static str> {
    let Some(do_step_id) = comparison.get("do_step_id").and_then(Value::as_str) else {
        return Err("CA design comparison is missing do_step_id");
    };
    let Some(design_predecessor_id) = comparison
        .get("design_predecessor_id")
        .and_then(Value::as_str)
    else {
        return Err("CA design comparison is missing design_predecessor_id");
    };
    if !super::valid_contract_id(do_step_id) || !super::valid_contract_id(design_predecessor_id) {
        return Err("CA design comparison has an invalid relation identity");
    }

    let Some(design_evidence) = comparison.get("design_evidence").and_then(Value::as_array) else {
        return Err("CA design comparison is missing design_evidence");
    };
    if design_evidence.is_empty() {
        return Err("CA design comparison design_evidence is empty");
    }
    let mut design_paths = std::collections::HashSet::new();
    for evidence in design_evidence {
        let Some(evidence) = evidence.as_object() else {
            return Err("CA design evidence must be an object");
        };
        let Some(path) = ca_safe_workspace_relative_path(evidence.get("path")) else {
            return Err("CA design evidence requires a safe relative path");
        };
        if !design_paths.insert(path) {
            return Err("CA design evidence repeats a path");
        }
        if !ca_claim_has_evidence(evidence.get("ref"))
            || !ca_claim_has_evidence(evidence.get("claim"))
        {
            return Err("CA design evidence requires a concrete ref and claim");
        }
    }

    let Some(successor_evidence) = comparison
        .get("successor_evidence")
        .and_then(Value::as_array)
    else {
        return Err("CA design comparison is missing successor_evidence");
    };
    if successor_evidence.is_empty() {
        return Err("CA design comparison successor_evidence is empty");
    }
    let mut successor_ids = std::collections::HashSet::new();
    for evidence in successor_evidence {
        let (work_package_id, _) = ca_successor_evidence_claim(evidence)?;
        if !successor_ids.insert(work_package_id) {
            return Err("CA successor evidence has an invalid or repeated work_package_id");
        }
    }

    let Some(status) = comparison.get("status").and_then(ca_claim_status) else {
        return Err("CA design comparison has an invalid status");
    };
    if status != TaskVerdict::Success && !ca_valid_failure_class(comparison.get("failure_class")) {
        return Err("CA non-pass design comparison requires a valid failure_class");
    }
    Ok(((do_step_id, design_predecessor_id), status))
}

fn ca_design_conformance_claim(
    candidate: &Value,
    required_dimensions: &[&str],
) -> Result<TaskVerdict, &'static str> {
    let Some(conformance) = candidate
        .get("design_conformance")
        .and_then(Value::as_object)
    else {
        return Err("CA audit is missing required design_conformance evidence");
    };
    let Some(declared) = conformance.get("status").and_then(ca_claim_status) else {
        return Err("CA design_conformance has an invalid status");
    };
    let Some(checks) = conformance.get("checks").and_then(Value::as_array) else {
        return Err("CA design_conformance is missing its checks");
    };
    if checks.len() != required_dimensions.len() {
        return Err("CA design_conformance must contain exactly the assigned required checks");
    }

    let mut seen = std::collections::HashSet::new();
    let mut computed = TaskVerdict::Success;
    for check in checks {
        let Some(check) = check.as_object() else {
            return Err("CA design_conformance check must be an object");
        };
        let Some(dimension) = check.get("dimension").and_then(Value::as_str) else {
            return Err("CA design_conformance check is missing its dimension");
        };
        if !required_dimensions.contains(&dimension) || !seen.insert(dimension) {
            return Err("CA design_conformance dimensions must match the assigned scope exactly");
        }
        let Some(comparisons) = check.get("comparisons").and_then(Value::as_array) else {
            return Err("CA design_conformance check is missing comparisons");
        };
        if comparisons.is_empty() {
            return Err("CA design_conformance check comparisons are empty");
        }
        let mut relation_keys = std::collections::HashSet::new();
        let mut comparison_status = TaskVerdict::Success;
        let mut comparison_failure_classes = Vec::new();
        for comparison in comparisons {
            let Some(comparison) = comparison.as_object() else {
                return Err("CA design comparison must be an object");
            };
            let (relation_key, status) = ca_design_relation_comparison_claim(comparison)?;
            if !relation_keys.insert(relation_key) {
                return Err("CA design_conformance check repeats a relation");
            }
            if status != TaskVerdict::Success {
                let failure_class = ca_failure_class(comparison.get("failure_class"))
                    .expect("comparison claim already validated its failure_class");
                comparison_failure_classes.push(failure_class);
            }
            comparison_status = ca_worst_verdict(comparison_status, status);
        }
        let Some(status) = check.get("status").and_then(ca_claim_status) else {
            return Err("CA design_conformance check has an invalid status");
        };
        if status != comparison_status {
            return Err("CA design_conformance check status conflicts with its comparisons");
        }
        if status != TaskVerdict::Success {
            let Some(check_failure_class) = ca_failure_class(check.get("failure_class")) else {
                return Err("CA non-pass design_conformance check requires a valid failure_class");
            };
            if super::preferred_ca_failure_class(comparison_failure_classes.iter().copied())
                != Some(check_failure_class)
            {
                return Err(
                    "CA design_conformance check failure_class conflicts with its comparisons",
                );
            }
        }
        computed = ca_worst_verdict(computed, status);
    }
    if declared != computed {
        return Err("CA design_conformance status conflicts with its checks");
    }
    Ok(declared)
}

fn ca_exact_json_value(content: &str) -> Option<Value> {
    fn canonicalize_redundant_fields(value: &mut Value) {
        let candidate = if value.get("ca_audit").is_some() {
            value.get_mut("ca_audit")
        } else {
            Some(value)
        };
        let Some(candidate) = candidate.filter(|candidate| {
            candidate.get("schema_version").and_then(Value::as_str) == Some("ca_audit/v1")
        }) else {
            return;
        };

        // `why.evidence` is an aggregate rendering of the authoritative
        // criterion ledger. If a positive audit supplied a complete positive
        // ledger but omitted only that redundant rendering, materialize a
        // deterministic pointer rather than failing an otherwise verifiable
        // claim and restarting the whole PDCA cycle. Non-pass or incomplete
        // ledgers remain invalid and are never repaired.
        if let Some(why) = candidate
            .pointer_mut("/dimensions/why")
            .and_then(Value::as_object_mut)
        {
            let may_materialize = !why.contains_key("evidence")
                && why.get("status").and_then(Value::as_str) == Some("pass")
                && why
                    .get("criteria")
                    .and_then(Value::as_array)
                    .is_some_and(|criteria| {
                        !criteria.is_empty()
                            && criteria.iter().all(|criterion| {
                                criterion.get("status").and_then(Value::as_str) == Some("pass")
                                    && ca_claim_has_evidence(criterion.get("criterion"))
                                    && ca_claim_has_evidence(criterion.get("evidence"))
                            })
                    });
            if may_materialize {
                let count = why["criteria"].as_array().map_or(0, Vec::len);
                why.insert(
                    "evidence".to_string(),
                    Value::String(format!(
                        "Original-intent alignment is evidenced by the {count} explicit passing criterion record(s) below."
                    )),
                );
            }
        }

        // The check-level failure class repeats the classifications on its
        // authoritative non-pass comparison ledger. Promote it only when the
        // field is absent and every non-pass comparison supplies a valid
        // class. Independent defects can legitimately have different causes;
        // use the same deterministic precedence as BizAgent aggregation.
        if let Some(checks) = candidate
            .pointer_mut("/design_conformance/checks")
            .and_then(Value::as_array_mut)
        {
            for check in checks {
                let Some(check) = check.as_object_mut() else {
                    continue;
                };
                if check.contains_key("failure_class")
                    || check.get("status").and_then(ca_claim_status) == Some(TaskVerdict::Success)
                {
                    continue;
                }
                let Some(comparisons) = check.get("comparisons").and_then(Value::as_array) else {
                    continue;
                };
                let mut classes = Vec::new();
                let mut malformed = false;
                for comparison in comparisons {
                    let Some(comparison) = comparison.as_object() else {
                        malformed = true;
                        break;
                    };
                    let Some(status) = comparison.get("status").and_then(ca_claim_status) else {
                        malformed = true;
                        break;
                    };
                    if status == TaskVerdict::Success {
                        continue;
                    }
                    let Some(failure_class) = ca_failure_class(comparison.get("failure_class"))
                    else {
                        malformed = true;
                        break;
                    };
                    classes.push(failure_class);
                }
                if !malformed {
                    if let Some(failure_class) =
                        super::preferred_ca_failure_class(classes.iter().copied())
                    {
                        check.insert(
                            "failure_class".to_string(),
                            Value::String(failure_class.to_string()),
                        );
                    }
                }
            }
        }

        // A comparison's positive status is redundant when both its owning
        // check and the complete conformance envelope declare `pass`. Fill
        // only an absent field in that unambiguous all-pass case. Never infer
        // a conditional/failing status or a failure class.
        if candidate
            .pointer("/design_conformance/status")
            .and_then(Value::as_str)
            != Some("pass")
        {
            return;
        }
        let Some(checks) = candidate
            .pointer_mut("/design_conformance/checks")
            .and_then(Value::as_array_mut)
        else {
            return;
        };
        for check in checks {
            let Some(check) = check.as_object_mut() else {
                continue;
            };
            if check.get("status").and_then(Value::as_str) != Some("pass") {
                continue;
            }
            let Some(comparisons) = check.get_mut("comparisons").and_then(Value::as_array_mut)
            else {
                continue;
            };
            for comparison in comparisons {
                if let Some(comparison) = comparison.as_object_mut() {
                    if !comparison.contains_key("status") {
                        comparison.insert("status".to_string(), Value::String("pass".to_string()));
                    }
                }
            }
        }
    }

    fn escape_unencoded_controls_in_json_strings(input: &str) -> Option<String> {
        if !input.bytes().any(
            |byte| matches!(byte, b'\n' | b'\r' | b'\t' | 0x00..=0x08 | 0x0b..=0x0c | 0x0e..=0x1f),
        ) {
            return None;
        }
        let mut output = String::with_capacity(input.len());
        let mut in_string = false;
        let mut escaped = false;
        for character in input.chars() {
            if !in_string {
                output.push(character);
                if character == '"' {
                    in_string = true;
                }
                continue;
            }
            if escaped {
                output.push(character);
                escaped = false;
                continue;
            }
            match character {
                '\\' => {
                    output.push(character);
                    escaped = true;
                }
                '"' => {
                    output.push(character);
                    in_string = false;
                }
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                control if control.is_control() => {
                    use std::fmt::Write as _;
                    write!(output, "\\u{:04x}", control as u32).ok()?;
                }
                value => output.push(value),
            }
        }
        (!in_string && !escaped).then_some(output)
    }

    fn parse_exact(input: &str) -> Option<Value> {
        serde_json::from_str(input).ok().or_else(|| {
            let repaired = escape_unencoded_controls_in_json_strings(input)?;
            serde_json::from_str(&repaired).ok()
        })
    }

    fn is_ca_root_or_finish_wrapper(value: &Value) -> bool {
        value.get("schema_version").and_then(Value::as_str) == Some("ca_audit/v1")
            || value
                .get("ca_audit")
                .and_then(|audit| audit.get("schema_version"))
                .and_then(Value::as_str)
                == Some("ca_audit/v1")
            || (value.get("action").and_then(Value::as_str) == Some("finish")
                && value
                    .get("content")
                    .and_then(|audit| audit.get("schema_version"))
                    .and_then(Value::as_str)
                    == Some("ca_audit/v1"))
    }

    /// Recover one unambiguous typed CA object from provider-added prose. The
    /// prose is discarded and never becomes evidence; the extracted object
    /// still passes every schema, receipt, path and verdict check below. Two
    /// matching objects fail closed because there is no authoritative choice.
    fn parse_unique_embedded_ca(input: &str) -> Option<Value> {
        let mut found = None;
        let mut accepted_until = 0usize;
        for (start, character) in input.char_indices() {
            if start < accepted_until || character != '{' {
                continue;
            }
            let mut depth = 0usize;
            let mut in_string = false;
            let mut escaped = false;
            for (relative_end, candidate_character) in input[start..].char_indices() {
                if in_string {
                    if escaped {
                        escaped = false;
                    } else {
                        match candidate_character {
                            '\\' => escaped = true,
                            '"' => in_string = false,
                            _ => {}
                        }
                    }
                    continue;
                }
                match candidate_character {
                    '"' => in_string = true,
                    '{' => depth = depth.saturating_add(1),
                    '}' => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                        if depth == 0 {
                            let end = start + relative_end + candidate_character.len_utf8();
                            if let Some(candidate) = parse_exact(&input[start..end]) {
                                if is_ca_root_or_finish_wrapper(&candidate) {
                                    if found.is_some() {
                                        return None;
                                    }
                                    found = Some(candidate);
                                    accepted_until = end;
                                }
                            }
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
        found
    }

    let trimmed = content.trim();
    let parsed = parse_exact(trimmed)
        .or_else(|| {
            let fenced = trimmed.strip_prefix("```json")?.strip_suffix("```")?;
            parse_exact(fenced.trim())
        })
        .or_else(|| parse_unique_embedded_ca(trimmed))?;
    let mut parsed = match parsed {
        // Some providers preserve an object-valued ReAct `content` field as
        // one exact JSON-encoded string at a transport boundary. Accept one
        // such encoding layer only when its complete payload is an object;
        // prose, fragments and recursively encoded strings still fail closed.
        Value::String(encoded) => {
            let decoded = parse_exact(encoded.trim())?;
            decoded.is_object().then_some(decoded)?
        }
        value => value,
    };
    // Defensive transport recovery: parse_llm_response normally unwraps the
    // outer ReAct envelope. Accept one exact object-valued finish wrapper here
    // as well so sync/streaming/provider adapters cannot make CA aggregation
    // depend on which layer performed the unwrapping. Prose or recursively
    // encoded content is still rejected.
    if parsed.get("action").and_then(Value::as_str) == Some("finish") {
        if let Some(inner) = parsed.get("content").filter(|value| value.is_object()) {
            parsed = inner.clone();
        }
    }
    canonicalize_redundant_fields(&mut parsed);
    Some(parsed)
}

/// Parse the dedicated CA terminal audit envelope.
///
/// This is a model *claim*, not a verification receipt.  It only repairs the
/// machine-readable verdict when the ordinary summary prefix was omitted.
/// Positive claims are still checked by `enforce_ca_verification_receipt`
/// against a kernel-observed tracked action.  Requiring an explicit `why`
/// checklist keeps the model from turning one green command into an assertion
/// that every user requirement was satisfied.
fn structured_ca_audit_claim(
    content: &str,
    required_design_dimensions: Option<&[&str]>,
) -> Option<Result<TaskVerdict, &'static str>> {
    let Some(value) = ca_exact_json_value(content) else {
        return content
            .contains("ca_audit/v1")
            .then_some(Err("CA audit envelope is not valid standalone JSON"));
    };
    let candidate = value.get("ca_audit").unwrap_or(&value);
    let schema = candidate.get("schema_version").and_then(Value::as_str);
    if schema != Some("ca_audit/v1") {
        return None;
    }

    let Some(declared) = candidate.get("overall_verdict").and_then(ca_claim_status) else {
        return Some(Err("CA audit envelope has an invalid overall_verdict"));
    };
    let Some(dimensions) = candidate.get("dimensions").and_then(Value::as_object) else {
        return Some(Err("CA audit envelope is missing dimensions"));
    };
    let Some(what) = dimensions.get("what").and_then(Value::as_object) else {
        return Some(Err("CA audit envelope is missing the what dimension"));
    };
    let Some(why) = dimensions.get("why").and_then(Value::as_object) else {
        return Some(Err("CA audit envelope is missing the why dimension"));
    };
    let Some(what_status) = what.get("status").and_then(ca_claim_status) else {
        return Some(Err("CA audit what dimension has an invalid status"));
    };
    let Some(why_status) = why.get("status").and_then(ca_claim_status) else {
        return Some(Err("CA audit why dimension has an invalid status"));
    };
    if !ca_claim_has_evidence(what.get("evidence")) || !ca_claim_has_evidence(why.get("evidence")) {
        return Some(Err("CA audit dimensions require non-empty evidence"));
    }

    let Some(criteria) = why.get("criteria").and_then(Value::as_array) else {
        return Some(Err(
            "CA audit why dimension is missing its criteria checklist",
        ));
    };
    if criteria.is_empty() {
        return Some(Err("CA audit why criteria checklist is empty"));
    }

    let mut computed = if what_status == TaskVerdict::Failed || why_status == TaskVerdict::Failed {
        TaskVerdict::Failed
    } else if what_status == TaskVerdict::PartialSuccess
        || why_status == TaskVerdict::PartialSuccess
    {
        TaskVerdict::PartialSuccess
    } else {
        TaskVerdict::Success
    };
    for criterion in criteria {
        let Some(criterion) = criterion.as_object() else {
            return Some(Err("CA audit criterion must be an object"));
        };
        if !criterion
            .get("criterion")
            .and_then(Value::as_str)
            .is_some_and(|text| !text.trim().is_empty())
        {
            return Some(Err("CA audit criterion name is empty"));
        }
        let Some(status) = criterion.get("status").and_then(ca_claim_status) else {
            return Some(Err("CA audit criterion has an invalid status"));
        };
        if !ca_claim_has_evidence(criterion.get("evidence")) {
            return Some(Err("CA audit criterion requires non-empty evidence"));
        }
        if status != TaskVerdict::Success
            && !criterion
                .get("failure_class")
                .and_then(Value::as_str)
                .is_some_and(|class| {
                    matches!(
                        class.trim(),
                        "observed_defect" | "verification_gap" | "external_blocker"
                    )
                })
        {
            return Some(Err("CA non-pass criterion requires a valid failure_class"));
        }
        computed = ca_worst_verdict(computed, status);
    }
    if let Some(required_dimensions) = required_design_dimensions {
        let conformance = match ca_design_conformance_claim(candidate, required_dimensions) {
            Ok(status) => status,
            Err(error) => return Some(Err(error)),
        };
        computed = ca_worst_verdict(computed, conformance);
    }
    if declared != computed {
        return Some(Err(
            "CA overall_verdict conflicts with its dimension or criterion statuses",
        ));
    }
    Some(Ok(declared))
}

fn ca_verdict_prefix(verdict: TaskVerdict) -> &'static str {
    match verdict {
        TaskVerdict::Success => "PASS",
        TaskVerdict::PartialSuccess => "CONDITIONAL_PASS",
        TaskVerdict::Failed | TaskVerdict::Blocked | TaskVerdict::Timeout => "FAIL",
    }
}

pub(super) fn ca_missing_verdict_outcome(has_valid_analysis: bool) -> TaskVerdict {
    if has_valid_analysis {
        TaskVerdict::PartialSuccess
    } else {
        TaskVerdict::Failed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CaTerminalNormalization {
    pub summary: String,
    pub content: String,
    pub verdict: TaskVerdict,
    pub contract_issue: Option<String>,
    /// A verdict recovered from the model-authored `ca_audit/v1` body must
    /// never stand in for an actually executed verifier receipt, even when CA
    /// finishes before the ordinary convergence window closes.
    receipt_binding_required: bool,
}

/// Normalize every CA termination path through one fail-closed contract.
///
/// Both fields are independently scrubbed of provider tool protocol. A clean
/// FAIL remains an authoritative business rejection even without a prose body
/// so SA can route it back to DA. PASS/CONDITIONAL_PASS require a true terminal
/// decision and substantive non-reasoning audit content.
pub(super) fn normalize_ca_terminal(
    summary: &str,
    content: &str,
    content_from_reasoning: bool,
    terminal_completion_observed: bool,
) -> CaTerminalNormalization {
    normalize_ca_terminal_with_requirements(
        summary,
        content,
        content_from_reasoning,
        terminal_completion_observed,
        false,
    )
}

pub(super) fn normalize_ca_terminal_with_requirements(
    summary: &str,
    content: &str,
    content_from_reasoning: bool,
    terminal_completion_observed: bool,
    require_design_conformance: bool,
) -> CaTerminalNormalization {
    normalize_ca_terminal_with_dimension_scope(
        summary,
        content,
        content_from_reasoning,
        terminal_completion_observed,
        require_design_conformance.then_some(&CA_DESIGN_CONFORMANCE_DIMENSIONS),
    )
}

fn normalize_ca_terminal_with_dimension_scope(
    summary: &str,
    content: &str,
    content_from_reasoning: bool,
    terminal_completion_observed: bool,
    required_design_dimensions: Option<&[&str]>,
) -> CaTerminalNormalization {
    let mut clean_summary = if is_raw_tool_protocol_content(summary) {
        String::new()
    } else {
        summary.trim().to_string()
    };
    let mut clean_content = if is_ca_audit_content(content, content_from_reasoning) {
        content.trim().to_string()
    } else {
        String::new()
    };
    let mut structured = structured_ca_verdict(&clean_summary);
    let mut receipt_binding_required = false;
    match structured_ca_audit_claim(&clean_content, required_design_dimensions) {
        Some(Err(issue)) => {
            return CaTerminalNormalization {
                summary: format!("FAILED: invalid CA structured audit: {issue}"),
                content: clean_content,
                verdict: TaskVerdict::Failed,
                contract_issue: Some(issue.to_string()),
                receipt_binding_required: false,
            };
        }
        Some(Ok(content_verdict)) => {
            if structured.is_some_and(|summary_verdict| summary_verdict != content_verdict) {
                let issue = "CA summary verdict conflicts with its structured audit envelope";
                return CaTerminalNormalization {
                    summary: format!("FAILED: {issue}"),
                    content: clean_content,
                    verdict: TaskVerdict::Failed,
                    contract_issue: Some(issue.to_string()),
                    receipt_binding_required: false,
                };
            }
            receipt_binding_required = content_verdict != TaskVerdict::Failed;
            if let Some(canonical) = ca_exact_json_value(&clean_content).filter(Value::is_object) {
                clean_content = canonical.to_string();
            }
            if structured.is_none() {
                let prior = (!clean_summary.is_empty())
                    .then_some(clean_summary.as_str())
                    .unwrap_or("structured criterion audit supplied");
                clean_summary = format!("{}: {prior}", ca_verdict_prefix(content_verdict));
                structured = Some(content_verdict);
            }
        }
        None => {}
    }

    if structured == Some(TaskVerdict::Failed) {
        return CaTerminalNormalization {
            summary: clean_summary,
            content: clean_content,
            verdict: TaskVerdict::Failed,
            contract_issue: None,
            receipt_binding_required: false,
        };
    }
    if terminal_completion_observed && !clean_content.is_empty() {
        if let Some(verdict @ (TaskVerdict::Success | TaskVerdict::PartialSuccess)) = structured {
            return CaTerminalNormalization {
                summary: clean_summary,
                content: clean_content,
                verdict,
                contract_issue: None,
                receipt_binding_required,
            };
        }
    }

    let verdict = ca_missing_verdict_outcome(!clean_content.is_empty());
    let issue = if !terminal_completion_observed {
        "CA execution ended before a terminal structured verdict"
    } else if structured.is_some() {
        "CA PASS/CONDITIONAL_PASS lacked substantive non-reasoning audit content"
    } else {
        "CA terminal response lacked a structured verdict"
    };
    let prior_summary = if clean_summary.is_empty() {
        "none".to_string()
    } else {
        clean_summary
    };
    CaTerminalNormalization {
        summary: format!(
            "{}: {}. Last valid audit summary: {}",
            if verdict == TaskVerdict::Failed {
                "FAILED"
            } else {
                "PARTIAL_SUCCESS"
            },
            issue,
            prior_summary
        ),
        content: clean_content,
        verdict,
        contract_issue: Some(issue.to_string()),
        receipt_binding_required: false,
    }
}

/// A CA close gate is a convergence mechanism, never proof of correctness.
/// Once the audit window explicitly required an executable verifier, a
/// positive model verdict must be backed by a successful kernel-observed
/// receipt. A genuine FAIL remains authoritative without that receipt.
pub(super) fn enforce_ca_verification_receipt(
    mut normalized: CaTerminalNormalization,
    receipt_required: bool,
    successful_verifier_observed: bool,
) -> CaTerminalNormalization {
    if !(receipt_required || normalized.receipt_binding_required)
        || successful_verifier_observed
        || normalized.verdict == TaskVerdict::Failed
    {
        return normalized;
    }

    let issue = "CA positive verdict lacked a successful executable verification receipt";
    normalized.summary = format!("FAIL: {issue}");
    if normalized.content.is_empty() {
        normalized.content = issue.to_string();
    } else {
        normalized.content.push_str("\n\nVerification limitation: ");
        normalized.content.push_str(issue);
    }
    normalized.verdict = TaskVerdict::Failed;
    normalized.contract_issue = Some(issue.to_string());
    normalized
}

pub(super) fn ca_receipt_path_matches(
    observed: &str,
    claimed: &str,
    workspace_root: Option<&std::path::Path>,
) -> bool {
    fn normalize_relative(path: &str) -> Option<std::path::PathBuf> {
        let normalized = path.trim().replace('\\', "/");
        if normalized.is_empty() || normalized.contains(':') {
            return None;
        }
        let mut result = std::path::PathBuf::new();
        for component in std::path::Path::new(&normalized).components() {
            match component {
                std::path::Component::Normal(value) => result.push(value),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_) => return None,
            }
        }
        (!result.as_os_str().is_empty()).then_some(result)
    }

    fn normalize_absolute(path: &std::path::Path) -> Option<std::path::PathBuf> {
        if !path.is_absolute() {
            return None;
        }
        let mut result = std::path::PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
        for component in path.components() {
            match component {
                std::path::Component::RootDir | std::path::Component::CurDir => {}
                std::path::Component::Normal(value) => result.push(value),
                std::path::Component::ParentDir | std::path::Component::Prefix(_) => return None,
            }
        }
        Some(result)
    }

    let Some(claimed) = normalize_relative(claimed) else {
        return false;
    };
    let observed_text = observed.trim().replace('\\', "/");
    let observed_path = std::path::Path::new(&observed_text);
    if !observed_path.is_absolute() {
        return normalize_relative(&observed_text).as_deref() == Some(claimed.as_path());
    }

    // Built-in file tools disclose their resolved absolute result path while
    // the CA contract intentionally names workspace-relative artifact paths.
    // Bind the two only through the configured workspace root: a same-suffix
    // path outside the workspace must never satisfy an evidence claim.
    let Some(root) = workspace_root.and_then(normalize_absolute) else {
        return false;
    };
    let Some(observed_absolute) = normalize_absolute(observed_path) else {
        return false;
    };
    let expected = root.join(&claimed);
    if observed_absolute == expected {
        return true;
    }

    // A configured workspace may itself be a symlink. Resolve existing paths
    // for that case, then re-check containment before comparing identities.
    let Ok(canonical_root) = std::fs::canonicalize(&root) else {
        return false;
    };
    let Ok(canonical_observed) = std::fs::canonicalize(&observed_absolute) else {
        return false;
    };
    let Ok(canonical_expected) = std::fs::canonicalize(&expected) else {
        return false;
    };
    canonical_observed.starts_with(&canonical_root)
        && canonical_expected.starts_with(&canonical_root)
        && canonical_observed == canonical_expected
}

fn ca_successful_read_actions_for_path<'a>(
    actions: &'a [crate::core::tracked_action::TrackedAction],
    claimed: &str,
    workspace_root: Option<&std::path::Path>,
) -> Vec<&'a crate::core::tracked_action::TrackedAction> {
    actions
        .iter()
        .filter(|action| {
            matches!(action.agent_role.as_str(), "CA" | "Check")
                && action.status == crate::core::tracked_action::ActionStatus::Success
                && action.tool_name == "file_read"
                && action.call_identity.is_some()
                && action.disclosure.as_ref().is_some_and(|receipt| {
                    receipt.disclosed_to_model
                        && receipt.file_read.as_ref().is_some_and(|read| {
                            ca_receipt_path_matches(&read.path, claimed, workspace_root)
                                && read.total_lines > 0
                                && !read.partial_line_preview
                                && !read.archived
                                && read.content_sha256.len() == 64
                                && read
                                    .content_sha256
                                    .bytes()
                                    .all(|byte| byte.is_ascii_hexdigit())
                        })
                })
        })
        .collect()
}

fn ca_has_complete_read_coverage(
    actions: &[crate::core::tracked_action::TrackedAction],
    claimed: &str,
    workspace_root: Option<&std::path::Path>,
) -> bool {
    let mut revisions = std::collections::BTreeMap::<(String, u64), Vec<(u64, u64)>>::new();
    for action in ca_successful_read_actions_for_path(actions, claimed, workspace_root) {
        let Some(read) = action
            .disclosure
            .as_ref()
            .and_then(|receipt| receipt.file_read.as_ref())
        else {
            continue;
        };
        revisions
            .entry((read.content_sha256.clone(), read.total_lines))
            .or_default()
            .push((read.offset, read.offset.saturating_add(read.returned)));
    }

    revisions
        .into_iter()
        .any(|((_revision, total_lines), mut spans)| {
            spans.sort_unstable();
            let mut covered_until = 0u64;
            for (start, end) in spans {
                if start > covered_until {
                    break;
                }
                covered_until = covered_until.max(end.min(total_lines));
            }
            covered_until >= total_lines
        })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CaRelationPathClaim {
    design_paths: std::collections::BTreeSet<String>,
    successor_deliveries:
        std::collections::BTreeMap<String, super::VerifiedConformanceSuccessorEvidence>,
}

type CaRelationKey = (String, String);
type CaDimensionRelationClaims = std::collections::BTreeMap<
    String,
    std::collections::BTreeMap<CaRelationKey, CaRelationPathClaim>,
>;

fn ca_design_conformance_relation_claims(content: &str) -> Option<CaDimensionRelationClaims> {
    let value = ca_exact_json_value(content)?;
    let candidate = value.get("ca_audit").unwrap_or(&value);
    let checks = candidate
        .get("design_conformance")?
        .get("checks")?
        .as_array()?;
    let mut dimensions = std::collections::BTreeMap::new();
    for check in checks {
        let dimension = check.get("dimension")?.as_str()?.trim().to_string();
        let comparisons = check.get("comparisons")?.as_array()?;
        let mut relations = std::collections::BTreeMap::new();
        for comparison in comparisons {
            let do_step_id = comparison.get("do_step_id")?.as_str()?.trim().to_string();
            let design_predecessor_id = comparison
                .get("design_predecessor_id")?
                .as_str()?
                .trim()
                .to_string();
            let design_paths = comparison
                .get("design_evidence")?
                .as_array()?
                .iter()
                .map(|evidence| Some(evidence.get("path")?.as_str()?.trim().to_string()))
                .collect::<Option<std::collections::BTreeSet<_>>>()?;
            let successor_deliveries = comparison
                .get("successor_evidence")?
                .as_array()?
                .iter()
                .map(|evidence| ca_successor_evidence_claim(evidence).ok())
                .collect::<Option<std::collections::BTreeMap<_, _>>>()?;
            if relations
                .insert(
                    (do_step_id, design_predecessor_id),
                    CaRelationPathClaim {
                        design_paths,
                        successor_deliveries,
                    },
                )
                .is_some()
            {
                return None;
            }
        }
        if dimensions.insert(dimension, relations).is_some() {
            return None;
        }
    }
    Some(dimensions)
}

fn ca_design_conformance_paths(content: &str) -> Option<(Vec<String>, Vec<String>)> {
    let claims = ca_design_conformance_relation_claims(content)?;
    let mut design_paths = std::collections::BTreeSet::new();
    let mut delivery_paths = std::collections::BTreeSet::new();
    for relations in claims.values() {
        for relation in relations.values() {
            design_paths.extend(relation.design_paths.iter().cloned());
            delivery_paths.extend(
                relation
                    .successor_deliveries
                    .values()
                    .filter_map(|evidence| match evidence {
                        super::VerifiedConformanceSuccessorEvidence::ArtifactPaths(paths) => {
                            Some(paths.iter().cloned())
                        }
                        super::VerifiedConformanceSuccessorEvidence::VerificationReceipts(_) => {
                            None
                        }
                    })
                    .flatten(),
            );
        }
    }
    Some((
        design_paths.into_iter().collect(),
        delivery_paths.into_iter().collect(),
    ))
}

/// Bind semantic conformance claims to the current isolated CA's actual read
/// receipts. Both normative design and every named delivered artifact must
/// have complete line coverage for one internally consistent file revision.
/// This uses composite internal identities and never rewrites provider call
/// ids.
pub(super) fn enforce_ca_design_conformance_receipts(
    mut normalized: CaTerminalNormalization,
    required: bool,
    actions: &[crate::core::tracked_action::TrackedAction],
    workspace_root: Option<&std::path::Path>,
) -> CaTerminalNormalization {
    if !required || normalized.contract_issue.is_some() {
        return normalized;
    }
    let Some((design_paths, delivery_paths)) = ca_design_conformance_paths(&normalized.content)
    else {
        return normalized;
    };
    let incomplete_design = design_paths
        .iter()
        .filter(|path| !ca_has_complete_read_coverage(actions, path, workspace_root))
        .cloned()
        .collect::<Vec<_>>();
    let unread_delivery = delivery_paths
        .iter()
        .filter(|path| !ca_has_complete_read_coverage(actions, path, workspace_root))
        .cloned()
        .collect::<Vec<_>>();
    if incomplete_design.is_empty() && unread_delivery.is_empty() {
        return normalized;
    }

    let issue = format!(
        "CA design conformance evidence is not bound to successful file-read receipts (incomplete_design={incomplete_design:?}, unread_delivery={unread_delivery:?})"
    );
    normalized.summary = format!("FAIL: {issue}");
    normalized.verdict = TaskVerdict::Failed;
    normalized.contract_issue = Some(issue);
    normalized
}

struct CaConformanceGate {
    contract: super::ConformanceContract,
    required_dimensions: Vec<&'static str>,
}

fn resolve_ca_conformance_gate(
    constraints: &std::collections::HashMap<String, String>,
) -> Result<Option<CaConformanceGate>, String> {
    let Some(encoded_contract) = constraints.get(super::CONFORMANCE_CONTRACT_CONSTRAINT) else {
        if constraints
            .contains_key(crate::core::biz_agent::BIZ_AGENT_CA_CONFORMANCE_DIMENSIONS_CONSTRAINT)
        {
            return Err(
                "CA child dimension scope exists without a conformance contract".to_string(),
            );
        }
        return Ok(None);
    };
    let contract = super::ConformanceContract::from_constraint_value(encoded_contract)?;
    let assigned = crate::core::biz_agent::assigned_ca_conformance_dimensions(constraints)?;
    let required_dimensions = assigned.map_or_else(
        || CA_DESIGN_CONFORMANCE_DIMENSIONS.to_vec(),
        |dimensions| {
            dimensions
                .into_iter()
                .map(crate::core::biz_agent::CaConformanceDimension::as_str)
                .collect()
        },
    );
    Ok(Some(CaConformanceGate {
        contract,
        required_dimensions,
    }))
}

fn ca_contract_failure(
    mut normalized: CaTerminalNormalization,
    issue: impl Into<String>,
) -> CaTerminalNormalization {
    let issue = issue.into();
    normalized.summary = format!("FAIL: {issue}");
    normalized.verdict = TaskVerdict::Failed;
    normalized.contract_issue = Some(issue);
    normalized
}

/// Require every assigned dimension to retain the exact relation identity and
/// design paths recovered from the canonical Do work-package order receipt.
/// A dimension may cite only the non-empty subset of successor evidence that
/// is relevant to that dimension, but every cited successor must match its
/// canonical tagged evidence exactly and the union across the assigned
/// dimensions must cover every successor. This prevents substituted or
/// cross-paired evidence without forcing unrelated evidence to be repeated in
/// every dimension.
fn enforce_ca_conformance_contract_paths(
    normalized: CaTerminalNormalization,
    gate: &CaConformanceGate,
) -> CaTerminalNormalization {
    if normalized.contract_issue.is_some() {
        return normalized;
    }
    let Some(expected_relations) = gate.contract.verified_relation_path_sets() else {
        return ca_contract_failure(
            normalized,
            "CA design conformance cannot close because the canonical Do order/path receipt is unavailable",
        );
    };
    let Some(claimed_dimensions) = ca_design_conformance_relation_claims(&normalized.content)
    else {
        return ca_contract_failure(
            normalized,
            "CA design conformance relation/path claims could not be decoded",
        );
    };
    let expected = expected_relations
        .into_iter()
        .map(|relation| {
            (
                (relation.do_step_id, relation.design_predecessor_id),
                CaRelationPathClaim {
                    design_paths: relation.design_paths,
                    successor_deliveries: relation.successor_deliveries,
                },
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let expected_relation_keys = expected.keys().cloned().collect::<Vec<_>>();
    let mut covered_successors = expected
        .keys()
        .cloned()
        .map(|relation| (relation, std::collections::BTreeSet::<String>::new()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut mismatched_dimensions = Vec::new();
    for dimension in &gate.required_dimensions {
        let Some(claimed_relations) = claimed_dimensions.get(*dimension) else {
            mismatched_dimensions.push(*dimension);
            continue;
        };
        if claimed_relations.keys().ne(expected.keys()) {
            mismatched_dimensions.push(*dimension);
            continue;
        }
        let mut dimension_matches = true;
        for (relation_key, expected_claim) in &expected {
            let Some(claimed) = claimed_relations.get(relation_key) else {
                dimension_matches = false;
                break;
            };
            if claimed.design_paths != expected_claim.design_paths
                || claimed.successor_deliveries.is_empty()
            {
                dimension_matches = false;
                break;
            }
            for (successor_id, evidence) in &claimed.successor_deliveries {
                if expected_claim.successor_deliveries.get(successor_id) != Some(evidence) {
                    dimension_matches = false;
                    break;
                }
                covered_successors
                    .get_mut(relation_key)
                    .expect("coverage initialized from the same relation keys")
                    .insert(successor_id.clone());
            }
            if !dimension_matches {
                break;
            }
        }
        if !dimension_matches {
            mismatched_dimensions.push(*dimension);
        }
    }
    if !mismatched_dimensions.is_empty() {
        return ca_contract_failure(
            normalized,
            format!(
                "CA design conformance relation/path claims do not match the canonical Do receipt (dimensions={mismatched_dimensions:?}, expected_relations={expected_relation_keys:?})",
            ),
        );
    }
    let uncovered_relations = expected
        .iter()
        .filter_map(|(relation_key, expected_claim)| {
            let expected_successors = expected_claim
                .successor_deliveries
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>();
            let covered = covered_successors
                .get(relation_key)
                .expect("coverage initialized from the same relation keys");
            (covered != &expected_successors).then(|| {
                (
                    relation_key.clone(),
                    expected_successors
                        .difference(covered)
                        .cloned()
                        .collect::<Vec<_>>(),
                )
            })
        })
        .collect::<Vec<_>>();
    if !uncovered_relations.is_empty() {
        return ca_contract_failure(
            normalized,
            format!(
                "CA design conformance does not collectively cover every canonical Do successor (uncovered={uncovered_relations:?})"
            ),
        );
    }
    normalized
}

pub(super) fn finalize_ca_terminal_contract(
    summary: &str,
    content: &str,
    content_from_reasoning: bool,
    terminal_completion_observed: bool,
    constraints: &std::collections::HashMap<String, String>,
    executable_receipt_required: bool,
    successful_verifier_observed: bool,
    actions: &[crate::core::tracked_action::TrackedAction],
    workspace_root: Option<&std::path::Path>,
) -> CaTerminalNormalization {
    let gate = match resolve_ca_conformance_gate(constraints) {
        Ok(gate) => gate,
        Err(issue) => {
            return ca_contract_failure(
                normalize_ca_terminal(summary, content, content_from_reasoning, false),
                format!("invalid kernel CA conformance scope: {issue}"),
            )
        }
    };
    let mut normalized = normalize_ca_terminal_with_dimension_scope(
        summary,
        content,
        content_from_reasoning,
        terminal_completion_observed,
        gate.as_ref()
            .map(|gate| gate.required_dimensions.as_slice()),
    );
    // Once the exact audit object has passed the terminal schema, collapse
    // provider/transport string encoding into one canonical JSON object text.
    // BizAgent aggregation and L0 archival must not have to guess how many
    // string layers a provider used, and literal control characters from an
    // outer JSON string must not survive as invalid downstream JSON.
    if normalized.contract_issue.is_none() {
        if let Some(value) = ca_exact_json_value(&normalized.content).filter(Value::is_object) {
            let candidate = value.get("ca_audit").unwrap_or(&value);
            if candidate.get("schema_version").and_then(Value::as_str) == Some("ca_audit/v1") {
                normalized.content = value.to_string();
            }
        }
    }
    let normalized = gate.as_ref().map_or(normalized.clone(), |gate| {
        enforce_ca_conformance_contract_paths(normalized, gate)
    });
    let normalized =
        enforce_ca_design_conformance_receipts(normalized, gate.is_some(), actions, workspace_root);
    enforce_ca_verification_receipt(
        normalized,
        executable_receipt_required,
        successful_verifier_observed,
    )
}

pub(super) fn da_phase_after_tool_turn(
    current: ExecutionPhase,
    substantive_effect: bool,
    verification_failed: bool,
) -> ExecutionPhase {
    if verification_failed {
        ExecutionPhase::Repair
    } else if substantive_effect {
        // Every mutation, including a repair made while verifying, must be
        // followed by fresh verification of the changed state.
        ExecutionPhase::Verify
    } else {
        current
    }
}

/// State for the implementation-DA convergence gate.
///
/// A successful command is useful completion evidence only when it happened
/// after the most recent substantive workspace change. Any later mutation or
/// failed command invalidates that evidence. `stable_turns` deliberately does
/// not cap total execution: a complex task can keep working for as long as it
/// continues to make substantive changes, while an already-verified DA cannot
/// spend the rest of its budget repeatedly inspecting the same state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct DaVerificationConvergence {
    fresh_verification_observed: bool,
    /// Effectless turns after the latest substantive workspace change. This
    /// counter is independent from verification: it bounds post-write
    /// inspection even for artifacts (for example Markdown) that do not have
    /// a conventional test runner.
    post_effect_stable_turns: u32,
    /// Stable turns backed by a recognisable successful verification. Kept
    /// separate so reads performed before the verifier cannot immediately
    /// exhaust the stronger verified-state window.
    verified_stable_turns: u32,
}

impl DaVerificationConvergence {
    pub(super) fn record_tool_turn(
        &mut self,
        substantive_effect: bool,
        verification_failed: bool,
        verification_succeeded: bool,
        novel_evidence_observed: bool,
    ) {
        if substantive_effect || verification_failed {
            *self = Self::default();
            return;
        }

        self.post_effect_stable_turns = self.post_effect_stable_turns.saturating_add(1);
        if verification_succeeded {
            if self.fresh_verification_observed {
                self.verified_stable_turns = self.verified_stable_turns.saturating_add(1);
            } else {
                self.fresh_verification_observed = true;
                self.verified_stable_turns = 1;
            }
        } else if novel_evidence_observed && self.fresh_verification_observed {
            // New targeted evidence may reveal another required change. Keep
            // the verification receipt, but give the model a fresh focused
            // opportunity to act on that information before closing tools.
            self.verified_stable_turns = 1;
        } else if self.fresh_verification_observed {
            self.verified_stable_turns = self.verified_stable_turns.saturating_add(1);
        }
    }

    pub(super) fn record_no_tool_turn(&mut self) {
        self.post_effect_stable_turns = self.post_effect_stable_turns.saturating_add(1);
        if self.fresh_verification_observed {
            self.verified_stable_turns = self.verified_stable_turns.saturating_add(1);
        }
    }

    pub(super) fn focus_active(
        self,
        role: AgentRole,
        workspace_effect_observed: bool,
        phase: ExecutionPhase,
        threshold: u32,
    ) -> bool {
        role == AgentRole::Do
            && workspace_effect_observed
            && phase == ExecutionPhase::Verify
            && self.fresh_verification_observed
            && threshold > 0
            && self.verified_stable_turns >= threshold
    }

    pub(super) fn close_active(
        self,
        role: AgentRole,
        workspace_effect_observed: bool,
        phase: ExecutionPhase,
        threshold: u32,
    ) -> bool {
        self.focus_active(role, workspace_effect_observed, phase, threshold)
    }

    pub(super) fn inspection_focus_active(
        self,
        role: AgentRole,
        workspace_effect_observed: bool,
        phase: ExecutionPhase,
        threshold: u32,
    ) -> bool {
        role == AgentRole::Do
            && workspace_effect_observed
            && phase == ExecutionPhase::Verify
            && !self.fresh_verification_observed
            && threshold > 0
            && self.post_effect_stable_turns >= threshold
    }

    pub(super) fn inspection_close_active(
        self,
        role: AgentRole,
        workspace_effect_observed: bool,
        phase: ExecutionPhase,
        threshold: u32,
    ) -> bool {
        self.inspection_focus_active(role, workspace_effect_observed, phase, threshold)
    }
}

use crate::core::tracked_action::{
    VerificationAssessment, VerificationKind, VerificationOutcome,
    VERIFICATION_ASSESSMENT_PARSER_VERSION,
};

/// Identify an actual verifier command at a shell segment boundary. A plain
/// substring search is unsafe here: `cat pytest.ini`, `rg cargo\ test README`,
/// or a long `find` command mentioning a test filename would otherwise become
/// a verifier merely because the inspection command exited zero.
fn shell_words(segment: &str) -> Option<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote = None::<char>;
    let mut escaped = false;
    for character in segment.chars() {
        if escaped {
            current.push(character);
            escaped = false;
            continue;
        }
        match quote {
            Some(active) if character == active => quote = None,
            Some(_) => current.push(character),
            None if matches!(character, '\'' | '"') => quote = Some(character),
            None if character == '\\' => escaped = true,
            None if character.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            None => current.push(character),
        }
    }
    if escaped || quote.is_some() {
        return None;
    }
    if !current.is_empty() {
        words.push(current);
    }
    Some(words)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShellSequenceOperator {
    AndIf,
    Sequence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AttributableShellLayout {
    segments: Vec<String>,
    separators: Vec<ShellSequenceOperator>,
}

/// Parse only the small shell grammar for which verifier output and status can
/// be attributed to the final process. This deliberately rejects pipelines,
/// background jobs, redirections, command substitution, comments and command
/// groups. Those constructs may still execute, but their output cannot mint a
/// kernel verification receipt.
fn attributable_shell_layout(command: &str) -> Option<AttributableShellLayout> {
    let characters = command.chars().collect::<Vec<_>>();
    let mut segments = Vec::new();
    let mut separators = Vec::new();
    let mut current = String::new();
    let mut quote = None::<char>;
    let mut escaped = false;
    let mut index = 0usize;

    let finish_segment = |current: &mut String, segments: &mut Vec<String>| -> bool {
        let segment = current.trim();
        if segment.is_empty() {
            return false;
        }
        segments.push(segment.to_string());
        current.clear();
        true
    };

    while index < characters.len() {
        let character = characters[index];
        if escaped {
            current.push(character);
            escaped = false;
            index = index.saturating_add(1);
            continue;
        }

        match quote {
            Some('\'') => {
                current.push(character);
                if character == '\'' {
                    quote = None;
                }
            }
            Some('"') => {
                if character == '\\' {
                    current.push(character);
                    escaped = true;
                } else if character == '"' {
                    current.push(character);
                    quote = None;
                } else if character == '`'
                    || (character == '$' && characters.get(index + 1) == Some(&'('))
                {
                    return None;
                } else {
                    current.push(character);
                }
            }
            Some(_) => unreachable!("only single and double shell quotes are tracked"),
            None => match character {
                '\'' | '"' => {
                    current.push(character);
                    quote = Some(character);
                }
                '\\' => {
                    current.push(character);
                    escaped = true;
                }
                '$' if characters.get(index + 1) == Some(&'(') => return None,
                '`' | '|' | '<' | '>' | '#' | '(' | ')' | '{' | '}' => return None,
                '&' => {
                    if characters.get(index + 1) != Some(&'&')
                        || !finish_segment(&mut current, &mut segments)
                    {
                        return None;
                    }
                    separators.push(ShellSequenceOperator::AndIf);
                    index = index.saturating_add(1);
                }
                ';' | '\n' | '\r' => {
                    if character == '\r' && characters.get(index + 1) == Some(&'\n') {
                        index = index.saturating_add(1);
                    }
                    if !finish_segment(&mut current, &mut segments) {
                        return None;
                    }
                    separators.push(ShellSequenceOperator::Sequence);
                }
                _ => current.push(character),
            },
        }
        index = index.saturating_add(1);
    }

    if escaped || quote.is_some() || !finish_segment(&mut current, &mut segments) {
        return None;
    }
    (separators.len().saturating_add(1) == segments.len()).then_some(AttributableShellLayout {
        segments,
        separators,
    })
}

fn is_shell_environment_assignment(token: &str) -> bool {
    let Some((name, _value)) = token.split_once('=') else {
        return false;
    };
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn safe_set_prefix(tokens: &[String]) -> bool {
    if tokens.first().map(String::as_str) != Some("set") || tokens.len() < 2 {
        return false;
    }
    let mut index = 1usize;
    while index < tokens.len() {
        let option = tokens[index].as_str();
        if option == "-o" {
            if tokens.get(index + 1).map(String::as_str) != Some("pipefail") {
                return false;
            }
            index = index.saturating_add(2);
            continue;
        }
        let Some(flags) = option.strip_prefix('-').filter(|flags| !flags.is_empty()) else {
            return false;
        };
        if !flags.chars().all(|flag| matches!(flag, 'e' | 'u' | 'o')) {
            return false;
        }
        if flags.contains('o') {
            if tokens.get(index + 1).map(String::as_str) != Some("pipefail") {
                return false;
            }
            index = index.saturating_add(1);
        }
        index = index.saturating_add(1);
    }
    true
}

fn safe_verifier_setup_prefix(segment: &str) -> bool {
    let Some(tokens) = shell_words(segment) else {
        return false;
    };
    if tokens.is_empty() {
        return false;
    }
    if tokens
        .iter()
        .all(|token| is_shell_environment_assignment(token))
    {
        return true;
    }
    match tokens.first().map(String::as_str) {
        Some("cd") => match tokens.as_slice() {
            [_, path] => !path.is_empty() && path != "-" && !path.starts_with('-'),
            [_, option, path] => option == "--" && !path.is_empty(),
            _ => false,
        },
        Some("export") => {
            tokens.len() > 1
                && tokens[1..]
                    .iter()
                    .all(|token| is_shell_environment_assignment(token))
        }
        Some("set") => safe_set_prefix(&tokens),
        _ => false,
    }
}

/// Only non-output-producing setup is allowed before a receiptable test. A
/// checked `&&` chain of typed Artifact predicates is also attributable: each
/// predicate must pass for the shell to reach a successful final predicate.
/// In particular `printf/echo && verifier`, mixed verifier kinds, arbitrary
/// programs, and output redirection fail closed even when the final process
/// exits successfully.
fn shell_test_output_is_attributable(command: &str) -> bool {
    shell_verifier_output_is_attributable(command, Some(VerificationKind::TestExecution))
}

fn shell_verifier_output_is_attributable(
    command: &str,
    required_kind: Option<VerificationKind>,
) -> bool {
    let Some(layout) = attributable_shell_layout(command) else {
        return false;
    };
    let Some(final_segment) = layout.segments.last() else {
        return false;
    };
    let Some(final_kind) = shell_segment_verification_kind(final_segment) else {
        return false;
    };
    if required_kind.is_some_and(|required| required != final_kind) {
        return false;
    }
    for (index, prefix) in layout
        .segments
        .iter()
        .take(layout.segments.len().saturating_sub(1))
        .enumerate()
    {
        let checked_artifact_predicate = final_kind == VerificationKind::Artifact
            && shell_segment_verification_kind(prefix) == Some(VerificationKind::Artifact)
            && layout.separators.get(index) == Some(&ShellSequenceOperator::AndIf);
        if !safe_verifier_setup_prefix(prefix) && !checked_artifact_predicate {
            return false;
        }
        // A failed directory change must prevent a verifier from running in
        // an unintended workspace. `set` and pure assignments cannot emit
        // verifier output and may use the conventional `;`/newline form.
        if shell_words(prefix)
            .and_then(|tokens| tokens.first().cloned())
            .as_deref()
            == Some("cd")
            && layout.separators.get(index) != Some(&ShellSequenceOperator::AndIf)
        {
            return false;
        }
    }
    true
}

/// Recognise verifier intent even when unsafe shell structure prevents the
/// attributable-layout parser from accepting it. This is used only to decline
/// a CA command before execution; it never mints a verification receipt.
fn shell_has_verification_intent(command: &str) -> bool {
    if !shell_verification_segments(command).is_empty() {
        return true;
    }

    // Command substitution was the production failure mode (`out=$(python -m
    // unittest ...)`). Inspect only each substitution's leading command; do
    // not fall back to broad `pytest` substring matching, which would confuse
    // `rg pytest README.md` with a verifier.
    let mut remaining = command;
    while let Some(open) = remaining.find("$(") {
        let inner = &remaining[open + 2..];
        let candidate = inner
            .split([')', ';', '|', '&', '>', '<', '\n', '\r'])
            .next()
            .unwrap_or_default();
        if shell_segment_verification_kind(candidate).is_some() {
            return true;
        }
        remaining = &inner[inner.find(')').map_or(inner.len(), |index| index + 1)..];
    }
    false
}

fn verification_profile_for_segment(
    segment: &str,
) -> Option<crate::tools::tool_executor::ToolExecutionProfile> {
    let tokens = shell_words(segment)?;
    let mut command_index = 0usize;
    while tokens
        .get(command_index)
        .is_some_and(|token| is_shell_environment_assignment(token))
    {
        command_index = command_index.saturating_add(1);
    }
    if tokens
        .get(command_index)
        .is_some_and(|token| token.rsplit('/').next() == Some("env"))
    {
        command_index = command_index.saturating_add(1);
        while tokens
            .get(command_index)
            .is_some_and(|token| is_shell_environment_assignment(token) || token.starts_with('-'))
        {
            command_index = command_index.saturating_add(1);
        }
    }
    let executable = tokens.get(command_index)?.rsplit('/').next()?;
    if executable == "mmdc" {
        return Some(crate::tools::tool_executor::ToolExecutionProfile::CleanMermaidVerification);
    }
    if matches!(executable, "pytest" | "py.test") {
        return Some(crate::tools::tool_executor::ToolExecutionProfile::CleanPytestVerification);
    }
    let python = executable == "python"
        || executable.strip_prefix("python").is_some_and(|suffix| {
            !suffix.is_empty()
                && suffix
                    .chars()
                    .all(|character| character.is_ascii_digit() || character == '.')
        });
    if !python {
        return None;
    }
    let arguments = &tokens[command_index.saturating_add(1)..];
    if let Some(profile) = arguments.windows(2).find_map(|window| {
        (window[0] == "-m").then(|| match window[1].as_str() {
            "pytest" => {
                Some(crate::tools::tool_executor::ToolExecutionProfile::CleanPytestVerification)
            }
            "unittest" | "compileall" => {
                Some(crate::tools::tool_executor::ToolExecutionProfile::CleanPythonVerification)
            }
            _ => None,
        })?
    }) {
        return Some(profile);
    }
    if arguments.iter().any(|argument| argument == "-m") {
        return None;
    }
    (arguments.first().map(String::as_str) == Some("-c")
        || arguments
            .iter()
            .map(String::as_str)
            .find(|argument| !argument.starts_with('-'))
            .is_some_and(|argument| argument.ends_with(".py")))
    .then_some(crate::tools::tool_executor::ToolExecutionProfile::CleanPythonVerification)
}

fn command_environment_assignment_values(command: &str, variable: &str) -> Vec<String> {
    attributable_shell_layout(command)
        .map(|layout| {
            layout
                .segments
                .iter()
                .filter_map(|segment| shell_words(segment))
                .flatten()
                .filter_map(|token| {
                    let (name, value) = token.split_once('=')?;
                    name.eq_ignore_ascii_case(variable)
                        .then(|| value.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn pytest_addopts_disables_workspace_cache(value: &str) -> bool {
    let options = value.split_whitespace().collect::<Vec<_>>();
    matches!(options.as_slice(), ["-p", "no:cacheprovider"])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerificationExecutionPreflightRejection {
    NotAttributable,
    BackgroundExecution,
    EnvironmentOverride,
}

impl VerificationExecutionPreflightRejection {
    const fn detail_code(self) -> &'static str {
        match self {
            Self::NotAttributable => "test_output_not_attributable",
            Self::BackgroundExecution => "background_verification_not_supported",
            Self::EnvironmentOverride => "kernel_verification_environment_override",
        }
    }
}

/// Resolve the kernel-owned execution profile from the final post-Hook
/// arguments. The returned profile is never added to provider arguments or
/// their journal hash. CA verifier intent that cannot produce attributable
/// output is rejected before EffectPolicy/ToolGuard/handler execution.
pub(super) fn verification_execution_profile(
    role: AgentRole,
    name: &str,
    args: &Value,
) -> Result<
    crate::tools::tool_executor::ToolExecutionProfile,
    VerificationExecutionPreflightRejection,
> {
    use crate::tools::tool_executor::ToolExecutionProfile;

    if !matches!(name, "bash" | "powershell") {
        return Ok(ToolExecutionProfile::Standard);
    }
    let command = verifier_command(args);
    let has_verification_intent = shell_has_verification_intent(command);
    if role == AgentRole::Check
        && has_verification_intent
        && !shell_verifier_output_is_attributable(command, None)
    {
        return Err(VerificationExecutionPreflightRejection::NotAttributable);
    }

    let profile = attributable_shell_layout(command)
        .and_then(|layout| layout.segments.last().cloned())
        .and_then(|segment| {
            shell_verifier_output_is_attributable(command, None)
                .then(|| verification_profile_for_segment(&segment))
                .flatten()
        })
        .unwrap_or(ToolExecutionProfile::Standard);
    if profile == ToolExecutionProfile::Standard {
        return Ok(profile);
    }
    if args
        .get("run_in_background")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(VerificationExecutionPreflightRejection::BackgroundExecution);
    }
    if profile.is_python_verification() {
        let pycache_prefix = command_environment_assignment_values(command, "PYTHONPYCACHEPREFIX");
        let pytest_addopts = command_environment_assignment_values(command, "PYTEST_ADDOPTS");
        if !pycache_prefix.is_empty()
            || pytest_addopts
                .iter()
                .any(|value| !pytest_addopts_disables_workspace_cache(value))
        {
            return Err(VerificationExecutionPreflightRejection::EnvironmentOverride);
        }
    }
    Ok(profile)
}

pub(super) fn verification_preflight_rejection(
    name: &str,
    rejection: VerificationExecutionPreflightRejection,
) -> Value {
    json!({
        "status": "not_executed",
        "reason": crate::core::execution_event::tool_terminal_reason::VERIFICATION_COMMAND_NOT_ATTRIBUTABLE,
        "detail_code": rejection.detail_code(),
        "tool": name,
        "message": "The verification command was not executed because its output/status could not be safely attributed to one foreground verifier under the kernel-owned clean environment.",
        "required_next_action": "Invoke exactly one foreground verifier. Only `set`, a checked `cd ... &&`, and environment assignments may precede it. Do not use pipes, output redirection, command substitution, background execution, multiple verifiers, echo/printf/listing commands, or override PYTHONPYCACHEPREFIX/PYTEST_ADDOPTS."
    })
}

fn normalized_verification_command(name: &str, args: &Value) -> String {
    crate::core::tracked_action::verification_invocation_sha256(name, args)
}

fn shell_segment_verification_kind(segment: &str) -> Option<VerificationKind> {
    let tokens = shell_words(segment)?;
    let mut command_index = 0usize;
    while tokens
        .get(command_index)
        .is_some_and(|token| is_shell_environment_assignment(token))
    {
        command_index = command_index.saturating_add(1);
    }
    if tokens
        .get(command_index)
        .is_some_and(|token| token.trim_start_matches(['(', '{']).rsplit('/').next() == Some("env"))
    {
        command_index = command_index.saturating_add(1);
        while let Some(token) = tokens.get(command_index) {
            if is_shell_environment_assignment(token) || token.starts_with('-') {
                command_index = command_index.saturating_add(1);
            } else {
                break;
            }
        }
    }
    let Some(first) = tokens.get(command_index) else {
        return None;
    };
    let executable = first
        .trim_start_matches(['(', '{'])
        .trim_end_matches([')', '}'])
        .rsplit('/')
        .next()
        .unwrap_or_default();
    let arguments = &tokens[command_index.saturating_add(1)..];
    let first_non_option = || {
        arguments
            .iter()
            .map(String::as_str)
            .find(|token| !token.starts_with('-') && !token.starts_with('+'))
    };
    let has_goal = |goals: &[&str]| {
        arguments
            .iter()
            .any(|token| goals.contains(&token.as_str()))
    };
    let has_quiet_predicate = || {
        arguments.iter().any(|token| {
            token.starts_with('-')
                && !token.starts_with("--")
                && token.trim_start_matches('-').contains('q')
        })
    };

    match executable {
        "pytest" | "py.test" | "ctest" | "rspec" | "phpunit" => {
            Some(VerificationKind::TestExecution)
        }
        "flake8" | "pylint" | "eslint" | "markdownlint" => Some(VerificationKind::Lint),
        "mypy" | "pyright" => Some(VerificationKind::Type),
        "mmdc"
            if arguments.windows(2).any(|window| {
                matches!(window[0].as_str(), "-i" | "--input")
                    && !window[1].trim().is_empty()
                    && window[1] != "-"
            }) || arguments.iter().any(|argument| {
                argument
                    .strip_prefix("--input=")
                    .is_some_and(|input| !input.trim().is_empty() && input != "-")
            }) =>
        {
            Some(VerificationKind::Artifact)
        }
        python
            if python == "python"
                || python.strip_prefix("python").is_some_and(|suffix| {
                    !suffix.is_empty()
                        && suffix
                            .chars()
                            .all(|character| character.is_ascii_digit() || character == '.')
                }) =>
        {
            arguments
                .windows(2)
                .find_map(|window| {
                    if window[0] != "-m" {
                        return None;
                    }
                    match window[1].as_str() {
                        "pytest" | "unittest" => Some(VerificationKind::TestExecution),
                        "compileall" => Some(VerificationKind::Syntax),
                        _ => None,
                    }
                })
                .or_else(|| {
                    if arguments.iter().any(|argument| argument == "-m") {
                        return None;
                    }
                    (arguments.first().map(String::as_str) == Some("-c")
                        || first_non_option().is_some_and(|argument| argument.ends_with(".py")))
                    .then_some(VerificationKind::Smoke)
                })
        }
        "cargo" => match first_non_option() {
            Some("test" | "nextest") => Some(VerificationKind::TestExecution),
            Some("build") => Some(VerificationKind::Build),
            Some("check") => Some(VerificationKind::Type),
            Some("clippy") => Some(VerificationKind::Lint),
            _ => None,
        },
        "go" => match first_non_option() {
            Some("test") => Some(VerificationKind::TestExecution),
            Some("vet") => Some(VerificationKind::Lint),
            _ => None,
        },
        "npm" | "pnpm" | "yarn" | "bun" => {
            let goal = if first_non_option() == Some("run") {
                arguments
                    .iter()
                    .map(String::as_str)
                    .skip_while(|token| *token != "run")
                    .nth(1)
            } else {
                first_non_option()
            };
            match goal {
                Some("test") => Some(VerificationKind::TestExecution),
                Some("lint") => Some(VerificationKind::Lint),
                Some("build") => Some(VerificationKind::Build),
                Some("check") => Some(VerificationKind::Smoke),
                _ => None,
            }
        }
        "deno" | "dotnet" | "composer" | "mix" | "swift" | "xcodebuild" if has_goal(&["test"]) => {
            Some(VerificationKind::TestExecution)
        }
        "node" if arguments.iter().any(|token| token == "--test") => {
            Some(VerificationKind::TestExecution)
        }
        "mvn" if has_goal(&["test", "verify"]) => Some(VerificationKind::TestExecution),
        "gradle" | "gradlew" | "./gradlew" if has_goal(&["test"]) => {
            Some(VerificationKind::TestExecution)
        }
        "make" | "just" if has_goal(&["test"]) => Some(VerificationKind::TestExecution),
        "make" | "just" if has_goal(&["check"]) => Some(VerificationKind::Smoke),
        "ruff" if has_goal(&["check"]) => Some(VerificationKind::Lint),
        "tsc"
            if arguments
                .iter()
                .any(|token| token.eq_ignore_ascii_case("--noemit")) =>
        {
            Some(VerificationKind::Type)
        }
        "bash" | "sh" if arguments.iter().any(|token| token == "-n") => {
            Some(VerificationKind::Syntax)
        }
        "grep" | "rg" if has_quiet_predicate() => Some(VerificationKind::Artifact),
        "diff" | "cmp" if arguments.len() >= 2 => Some(VerificationKind::Artifact),
        "test" | "[" | "[["
            if arguments
                .first()
                .is_some_and(|predicate| predicate.starts_with('-')) =>
        {
            Some(VerificationKind::Artifact)
        }
        _ => None,
    }
}

fn verifier_command(args: &Value) -> &str {
    args.get("command")
        .or_else(|| args.get("script"))
        .or_else(|| args.get("code"))
        .and_then(Value::as_str)
        .unwrap_or_default()
}

/// Extract one unambiguous, explicitly named Python test file from the exact
/// post-hook command that was executed. Bare discovery, directories, globs,
/// dotted unittest class/method selectors and multi-target invocations
/// intentionally return `None`. One plain `unittest` module is mapped to its
/// exact `.py` file because `python -m unittest test_calculator -v` is the
/// framework's canonical spelling for executing `test_calculator.py`.
pub(crate) fn explicit_test_execution_target_path(
    name: &str,
    args: &Value,
    workspace_root: Option<&std::path::Path>,
) -> Option<String> {
    if !matches!(name, "bash" | "powershell") {
        return None;
    }
    let command = verifier_command(args);
    if !shell_test_output_is_attributable(command) {
        return None;
    }
    let layout = attributable_shell_layout(command)?;
    let final_segment = layout.segments.last()?;
    if shell_segment_verification_kind(final_segment) != Some(VerificationKind::TestExecution) {
        return None;
    }

    let mut base = None::<String>;
    for prefix in layout
        .segments
        .iter()
        .take(layout.segments.len().saturating_sub(1))
    {
        let tokens = shell_words(prefix)?;
        if tokens.first().map(String::as_str) == Some("cd") {
            let candidate = match tokens.as_slice() {
                [_, path] => path,
                [_, option, path] if option == "--" => path,
                _ => return None,
            };
            if candidate == "." {
                continue;
            }
            let normalized = if std::path::Path::new(candidate).is_absolute() {
                let relative = std::path::Path::new(candidate)
                    .strip_prefix(workspace_root?)
                    .ok()?
                    .to_str()?;
                crate::core::sa::normalize_work_package_artifact_path(relative)?
            } else {
                crate::core::sa::normalize_work_package_artifact_path(candidate)?
            };
            if base.replace(normalized).is_some() {
                return None;
            }
        }
    }

    let tokens = shell_words(final_segment)?;
    let mut targets = tokens
        .iter()
        .filter_map(|token| {
            let raw = token.split("::").next().unwrap_or_default();
            (!raw.starts_with('-')
                && raw.to_ascii_lowercase().ends_with(".py")
                && !raw
                    .chars()
                    .any(|character| matches!(character, '*' | '?' | '[' | ']')))
            .then_some(raw)
        })
        .collect::<Vec<_>>();
    targets.sort_unstable();
    targets.dedup();
    let target = match targets.as_slice() {
        [target] => (*target).to_string(),
        [] => {
            let python_index = tokens.iter().position(|token| {
                Path::new(token)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("python"))
            })?;
            if tokens.get(python_index + 1).map(String::as_str) != Some("-m")
                || tokens.get(python_index + 2).map(String::as_str) != Some("unittest")
            {
                return None;
            }
            let mut modules = Vec::new();
            for token in tokens.iter().skip(python_index + 3) {
                if matches!(
                    token.as_str(),
                    "-v" | "--verbose"
                        | "-q"
                        | "--quiet"
                        | "-f"
                        | "--failfast"
                        | "-c"
                        | "--catch"
                        | "-b"
                        | "--buffer"
                        | "--locals"
                ) {
                    continue;
                }
                if token.starts_with('-')
                    || token == "discover"
                    || token.contains('.')
                    || token.contains(['/', '\\'])
                    || !token.chars().enumerate().all(|(index, character)| {
                        character == '_'
                            || character.is_ascii_alphabetic()
                            || (index > 0 && character.is_ascii_digit())
                    })
                {
                    return None;
                }
                modules.push(token.as_str());
            }
            let [module] = modules.as_slice() else {
                return None;
            };
            format!("{module}.py")
        }
        _ => return None,
    };
    let normalized_target = crate::core::sa::normalize_work_package_artifact_path(&target)?;
    if let Some(base) = base {
        if normalized_target == base || normalized_target.starts_with(&format!("{base}/")) {
            Some(normalized_target)
        } else {
            crate::core::sa::normalize_work_package_artifact_path(&format!(
                "{base}/{normalized_target}"
            ))
        }
    } else {
        Some(normalized_target)
    }
}

#[cfg(test)]
mod explicit_test_target_tests {
    use super::explicit_test_execution_target_path;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn extracts_one_checked_python_test_file_and_rejects_ambiguous_scope() {
        assert_eq!(
            explicit_test_execution_target_path(
                "bash",
                &json!({"command": "cd calculator_project && python3 -m pytest tests/test_calculator.py::test_add -q"}),
                None,
            )
            .as_deref(),
            Some("calculator_project/tests/test_calculator.py")
        );
        for command in [
            "python3 -m pytest -q",
            "python3 -m pytest tests -q",
            "python3 -m pytest tests/test_a.py tests/test_b.py -q",
            "python3 -m pytest 'tests/test_*.py' -q",
            "echo unsafe && python3 -m pytest tests/test_a.py -q",
        ] {
            assert_eq!(
                explicit_test_execution_target_path("bash", &json!({"command": command}), None),
                None,
                "command must not mint a path-scoped receipt: {command}"
            );
        }

        assert_eq!(
            explicit_test_execution_target_path(
                "bash",
                &json!({"command": "cd /tmp/exact-workspace/calculator_project && python3 -m pytest test_calculator.py -q"}),
                Some(Path::new("/tmp/exact-workspace")),
            )
            .as_deref(),
            Some("calculator_project/test_calculator.py")
        );
        assert_eq!(
            explicit_test_execution_target_path(
                "bash",
                &json!({"command": "cd /tmp/other-workspace/calculator_project && python3 -m pytest test_calculator.py -q"}),
                Some(Path::new("/tmp/exact-workspace")),
            ),
            None,
            "an absolute cd outside the exact workspace must remain ineligible"
        );

        assert_eq!(
            explicit_test_execution_target_path(
                "bash",
                &json!({"command": "cd /tmp/exact-workspace/calculator_project && python3 -m unittest test_calculator -v"}),
                Some(Path::new("/tmp/exact-workspace")),
            )
            .as_deref(),
            Some("calculator_project/test_calculator.py")
        );
        for command in [
            "python3 -m unittest discover -v",
            "python3 -m unittest test_calculator.TestAdd -v",
            "python3 -m unittest test_a test_b -v",
        ] {
            assert_eq!(
                explicit_test_execution_target_path("bash", &json!({"command": command}), None),
                None,
                "ambiguous or partial unittest target must not mint a file-scoped receipt: {command}"
            );
        }
    }
}

fn shell_verification_segments(command: &str) -> Vec<VerificationKind> {
    command
        .split(['\n', ';', '|', '&'])
        .filter_map(|segment| shell_segment_verification_kind(&segment.to_ascii_lowercase()))
        .collect()
}

fn shell_final_segment_is_verifier(command: &str) -> bool {
    command
        .split(['\n', ';', '|', '&'])
        .rev()
        .find(|segment| !segment.trim().is_empty())
        .and_then(|segment| shell_segment_verification_kind(&segment.to_ascii_lowercase()))
        .is_some()
}

/// Conservative mask detection. `cd project && pytest` is safe because a
/// failing final verifier remains the shell status. Pipelines, OR recovery,
/// backgrounding, inversion, or an unconditional command after the verifier
/// make exit-code attribution ambiguous and therefore cannot pass.
fn shell_verifier_status_is_masked(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let has_background_operator = bytes.iter().enumerate().any(|(index, byte)| {
        *byte == b'&'
            && bytes.get(index.wrapping_sub(1)) != Some(&b'&')
            && bytes.get(index + 1) != Some(&b'&')
            && bytes.get(index.wrapping_sub(1)) != Some(&b'>')
    });
    if lower.contains("||") || lower.contains('|') || has_background_operator {
        return true;
    }
    if lower.split(['\n', ';', '|', '&']).any(|segment| {
        segment.trim_start().starts_with('!')
            && shell_segment_verification_kind(segment.trim_start_matches(['!', ' '])).is_some()
    }) {
        return true;
    }

    // An unconditional command after a verifier replaces that verifier's
    // shell exit status. A separator before the verifier (for example
    // `set -e; pytest`) does not.
    lower.char_indices().any(|(index, character)| {
        matches!(character, ';' | '\n')
            && !shell_verification_segments(&lower[..index]).is_empty()
            && !lower[index + character.len_utf8()..].trim().is_empty()
    })
}

fn verification_kind(name: &str, args: &Value) -> Option<VerificationKind> {
    match name {
        "code_execute" => Some(VerificationKind::Smoke),
        "mermaid_validate"
        | "jsonld_validate"
        | "ontology_validate_turtle"
        | "ontology_validate_shacl" => Some(VerificationKind::Artifact),
        "ontology_lint_turtle" => Some(VerificationKind::Lint),
        "bash" | "powershell" => shell_verification_segments(verifier_command(args))
            .last()
            .copied(),
        _ => None,
    }
}

fn verifier_output(result: &Value) -> String {
    ["stdout", "stderr", "output", "content"]
        .into_iter()
        .filter_map(|field| result.get(field).and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Select a small actionable failure view without carrying a raw verifier
/// transcript across Agent boundaries. Test node identifiers and ordinary
/// compiler/assertion diagnostics are data, not correction authority; the
/// caller binds this text to the kernel assessment and composite call ID.
fn bounded_verification_failure_diagnostic(result: &Value) -> Option<String> {
    const MAX_LINES: usize = 12;
    const MAX_LINE_CHARS: usize = 240;
    const MAX_TOTAL_CHARS: usize = 1_600;

    let output = verifier_output(result);
    let mut selected = Vec::new();
    for raw in output.lines() {
        let clean = raw
            .chars()
            .filter(|character| !character.is_control() || matches!(character, '\t'))
            .collect::<String>();
        let line = clean.trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        let actionable = line.starts_with("FAILED ")
            || line.starts_with("ERROR ")
            || line.starts_with("FAIL: ")
            || line.starts_with("ERROR: ")
            || line.starts_with("E ")
            || lower.starts_with("test ") && lower.ends_with(" failed")
            || lower.starts_with("thread '") && lower.contains("panicked")
            || lower.contains("assertionerror")
            || lower.starts_with("error[")
            || lower.starts_with("error:")
            || lower.contains(" tests failed")
            || lower.contains(" failed,")
            || lower.contains(" failed in ");
        if !actionable {
            continue;
        }
        let bounded = line.chars().take(MAX_LINE_CHARS).collect::<String>();
        if !selected.contains(&bounded) {
            selected.push(bounded);
        }
        if selected.len() == MAX_LINES {
            break;
        }
    }
    if selected.is_empty() {
        return None;
    }
    let joined = selected.join("\n");
    Some(joined.chars().take(MAX_TOTAL_CHARS).collect())
}

#[derive(Debug, Clone, Copy, Default)]
struct TestCardinality {
    executed: Option<u64>,
    skipped: u64,
    failures: u64,
}

fn words_with_counts(line: &str) -> Vec<(u64, String)> {
    let words = line
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    words
        .windows(2)
        .filter_map(|window| {
            window[0]
                .parse::<u64>()
                .ok()
                .map(|count| (count, window[1].to_ascii_lowercase()))
        })
        .collect()
}

fn status_count(line: &str, labels: &[&str]) -> u64 {
    words_with_counts(line)
        .into_iter()
        .filter(|(_, label)| labels.contains(&label.as_str()))
        .map(|(count, _)| count)
        .sum()
}

fn count_after_label(line: &str, labels: &[&str]) -> Option<u64> {
    let lower = line.to_ascii_lowercase();
    labels.iter().find_map(|label| {
        let index = lower.find(label)?;
        let before = &lower[..index];
        let after = &lower[index + label.len()..];
        let mut chars = after.chars();
        let first = chars.next()?;
        let explicit_delimiter = matches!(first, ':' | '=');
        if !explicit_delimiter && before.chars().any(|character| character.is_ascii_digit()) {
            return None;
        }
        let digits = after
            .trim_start_matches(|character: char| {
                character.is_ascii_whitespace() || matches!(character, ':' | '=')
            })
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect::<String>();
        (!digits.is_empty())
            .then(|| digits.parse::<u64>().ok())
            .flatten()
    })
}

fn parse_unittest_cardinality(output: &str) -> Option<TestCardinality> {
    let ran = output.lines().rev().find_map(|line| {
        let words = line.split_whitespace().collect::<Vec<_>>();
        words.windows(2).find_map(|window| {
            (window[0].eq_ignore_ascii_case("ran"))
                .then(|| {
                    window[1]
                        .trim_matches(|c: char| !c.is_ascii_digit())
                        .parse()
                        .ok()
                })
                .flatten()
        })
    })?;
    let skipped = output
        .lines()
        .map(|line| {
            status_count(line, &["skipped", "skip"])
                .max(count_after_label(line, &["skipped", "skip"]).unwrap_or(0))
        })
        .max()
        .unwrap_or(0)
        .min(ran);
    Some(TestCardinality {
        executed: Some(ran.saturating_sub(skipped)),
        skipped,
        failures: u64::from(output.to_ascii_lowercase().contains("failed (")),
    })
}

fn parse_pytest_cardinality(output: &str) -> Option<TestCardinality> {
    let lower = output.to_ascii_lowercase();
    if lower.contains("no tests ran")
        || lower.contains("collected 0 items")
        || lower.contains("0 tests collected")
    {
        return Some(TestCardinality {
            executed: Some(0),
            ..TestCardinality::default()
        });
    }
    output.lines().rev().find_map(|line| {
        let passed = status_count(line, &["passed", "pass"]);
        let failed = status_count(line, &["failed", "fail", "errors", "error"]);
        let xexecuted = status_count(line, &["xfailed", "xpassed"]);
        let skipped = status_count(line, &["skipped", "deselected"]);
        let has_summary = passed > 0 || failed > 0 || xexecuted > 0 || skipped > 0;
        has_summary.then_some(TestCardinality {
            executed: Some(passed.saturating_add(failed).saturating_add(xexecuted)),
            skipped,
            failures: failed,
        })
    })
}

fn parse_cargo_cardinality(output: &str) -> Option<TestCardinality> {
    let summaries = output
        .lines()
        .filter(|line| line.to_ascii_lowercase().contains("test result:"))
        .collect::<Vec<_>>();
    if !summaries.is_empty() {
        let mut cardinality = TestCardinality {
            executed: Some(0),
            ..TestCardinality::default()
        };
        for line in summaries {
            let passed = status_count(line, &["passed"]);
            let failed = status_count(line, &["failed"]);
            cardinality.executed = Some(
                cardinality
                    .executed
                    .unwrap_or(0)
                    .saturating_add(passed)
                    .saturating_add(failed),
            );
            cardinality.skipped = cardinality
                .skipped
                .saturating_add(status_count(line, &["ignored"]));
            cardinality.failures = cardinality.failures.saturating_add(failed);
        }
        return Some(cardinality);
    }
    None
}

fn parse_generic_test_cardinality(output: &str) -> Option<TestCardinality> {
    let parsed = output.lines().rev().find_map(|line| {
        let lower = line.to_ascii_lowercase();
        let total = if lower.contains("tests run:") {
            count_after_label(line, &["total"])
                .or_else(|| count_after_label(line, &["run"]))
                .unwrap_or(0)
        } else if lower.contains("tests:") {
            count_after_label(line, &["total"]).unwrap_or_else(|| status_count(line, &["total"]))
        } else if lower.contains("examples") {
            status_count(line, &["examples"])
        } else if lower.trim_start_matches(['#', ' ']).starts_with("tests ") {
            count_after_label(line, &["tests"]).unwrap_or(0)
        } else if lower.contains("total:") {
            count_after_label(line, &["total"]).unwrap_or(0)
        } else {
            0
        };
        let passed = status_count(line, &["passed", "pass"])
            .max(count_after_label(line, &["passed", "pass"]).unwrap_or(0));
        let failed = status_count(line, &["failed", "fail", "failures", "errors"])
            .max(count_after_label(line, &["failed", "fail", "failures", "errors"]).unwrap_or(0));
        let skipped = status_count(line, &["skipped", "ignored", "pending"])
            .max(count_after_label(line, &["skipped", "ignored", "pending"]).unwrap_or(0));
        let executed = if passed.saturating_add(failed) > 0 {
            passed.saturating_add(failed)
        } else {
            total.saturating_sub(skipped)
        };
        (total > 0 || passed > 0 || failed > 0 || skipped > 0).then_some(TestCardinality {
            executed: Some(executed),
            skipped,
            failures: failed,
        })
    });
    parsed.or_else(|| {
        let lower = output.to_ascii_lowercase();
        (lower.contains("no tests")
            || lower.contains("0 tests run")
            || lower.contains("0 test cases"))
        .then_some(TestCardinality {
            executed: Some(0),
            ..TestCardinality::default()
        })
    })
}

fn test_cardinality(command: &str, output: &str) -> Option<TestCardinality> {
    let lower = command.to_ascii_lowercase();
    if lower.contains("unittest") {
        parse_unittest_cardinality(output)
    } else if lower.contains("pytest") || lower.contains("py.test") {
        parse_pytest_cardinality(output)
    } else if lower.contains("cargo") && (lower.contains(" test") || lower.contains(" nextest")) {
        parse_cargo_cardinality(output).or_else(|| parse_generic_test_cardinality(output))
    } else {
        parse_generic_test_cardinality(output)
    }
}

/// Produce the one typed assessment used by both synchronous and streaming
/// execution. Exit status remains in `TrackedAction::status`; this function
/// decides only whether the invocation is trustworthy verification evidence.
pub(super) fn assess_verification_call(
    name: &str,
    args: &Value,
    result: &Value,
) -> Option<VerificationAssessment> {
    let kind = verification_kind(name, args)?;
    let command = verifier_command(args);
    let process_failed = crate::core::tracked_action::tool_result_failed(result);
    let inconclusive = |reason: &str, count, skipped_count| VerificationAssessment {
        parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
        kind,
        outcome: VerificationOutcome::Inconclusive,
        count,
        skipped_count,
        reason: Some(reason.to_string()),
        diagnostic: None,
    };

    if result.get("background_task_id").is_some() {
        return Some(inconclusive("background_execution_not_settled", None, 0));
    }
    if matches!(name, "bash" | "powershell") && shell_verifier_status_is_masked(command) {
        return Some(inconclusive("composite_shell_status_masked", None, 0));
    }
    if result.get("timed_out").and_then(Value::as_bool) == Some(true)
        || result.get("error").is_some()
    {
        return Some(inconclusive("verifier_execution_incomplete", None, 0));
    }

    if kind != VerificationKind::TestExecution {
        let invocation_count = if matches!(name, "bash" | "powershell") {
            shell_verification_segments(command).len().max(1) as u64
        } else {
            1
        };
        return Some(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind,
            outcome: if process_failed {
                VerificationOutcome::Failed
            } else {
                VerificationOutcome::Passed
            },
            count: Some(invocation_count),
            skipped_count: 0,
            reason: None,
            diagnostic: process_failed
                .then(|| bounded_verification_failure_diagnostic(result))
                .flatten(),
        });
    }

    let lower_command = command.to_ascii_lowercase();
    if lower_command.contains("--collect-only")
        || lower_command.contains("--collectonly")
        || lower_command.contains("--list-tests")
        || lower_command
            .split_whitespace()
            .any(|token| token.trim_matches(['\'', '"']) == "--co")
    {
        return Some(inconclusive(
            "collection_only_did_not_execute_tests",
            Some(0),
            0,
        ));
    }
    if matches!(name, "bash" | "powershell") && !shell_test_output_is_attributable(command) {
        return Some(inconclusive("test_output_not_attributable", None, 0));
    }
    if shell_verification_segments(command).len() != 1 || !shell_final_segment_is_verifier(command)
    {
        return Some(inconclusive(
            "compound_test_output_cannot_be_attributed",
            None,
            0,
        ));
    }

    let Some(cardinality) = test_cardinality(command, &verifier_output(result)) else {
        return Some(inconclusive("test_execution_cardinality_unknown", None, 0));
    };
    let Some(executed) = cardinality.executed else {
        return Some(inconclusive(
            "test_execution_cardinality_unknown",
            None,
            cardinality.skipped,
        ));
    };
    if executed == 0 {
        return Some(inconclusive(
            if cardinality.skipped > 0 {
                "all_tests_skipped"
            } else {
                "zero_tests_executed"
            },
            Some(0),
            cardinality.skipped,
        ));
    }

    Some(VerificationAssessment {
        parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
        kind,
        outcome: if process_failed || cardinality.failures > 0 {
            VerificationOutcome::Failed
        } else {
            VerificationOutcome::Passed
        },
        count: Some(executed),
        skipped_count: cardinality.skipped,
        reason: None,
        diagnostic: (process_failed || cardinality.failures > 0)
            .then(|| bounded_verification_failure_diagnostic(result))
            .flatten(),
    })
}

/// Recognise verifier intent while keeping it separate from evidence outcome.
pub(super) fn is_verification_call(name: &str, args: &Value) -> bool {
    verification_kind(name, args).is_some()
}

/// A successful verifier is evidence about the latest workspace state, but it
/// only drives convergence. It never upgrades a TaskResult or bypasses CA.
pub(super) fn is_successful_verification_call(name: &str, args: &Value, result: &Value) -> bool {
    assess_verification_call(name, args, result)
        .is_some_and(|assessment| assessment.outcome == VerificationOutcome::Passed)
}

/// After a write, but before a recognisable verifier succeeds, retain only
/// repair-capable tools and executable checks. A currently referenced,
/// execution-owned result reader remains available so the model can finish a
/// bounded page it was explicitly offered; broad file/result reads are still
/// withdrawn.
pub(super) fn da_post_effect_inspection_focus_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    active: bool,
) -> Vec<Value> {
    if role != AgentRole::Do || !active {
        return definitions;
    }
    definitions
        .into_iter()
        .filter(|definition| {
            let name = definition["function"]["name"].as_str().unwrap_or_default();
            matches!(
                name,
                "file_write"
                    | "file_edit"
                    | "bash"
                    | "powershell"
                    | "code_execute"
                    | "jsonld_validate"
                    | "ontology_validate_turtle"
                    | "ontology_validate_shacl"
                    | "ontology_lint_turtle"
            ) || ToolExecutor::is_micro_tool_name(name)
        })
        .collect()
}

/// Once the latest workspace state has a fresh successful verification, keep
/// only tools that can either repair it or perform a targeted final check.
/// Broad discovery is intentionally removed. A currently referenced,
/// execution-owned full-result reader remains eligible, but execution still
/// requires that its exact schema be advertised in the current request.
pub(super) fn da_verified_focus_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    active: bool,
) -> Vec<Value> {
    if role != AgentRole::Do || !active {
        return definitions;
    }
    definitions
        .into_iter()
        .filter(|definition| {
            let name = definition["function"]["name"].as_str().unwrap_or_default();
            matches!(
                name,
                "file_write"
                    | "file_edit"
                    | "file_read"
                    | "grep_search"
                    | "bash"
                    | "powershell"
                    | "code_execute"
            ) || ToolExecutor::is_micro_tool_name(name)
        })
        .collect()
}

/// The close gate requests a terminal verdict from existing evidence. Emptying
/// the tool window is not a success assertion: normal finish parsing and the
/// downstream CA verification path still decide the verdict.
pub(super) fn da_verified_close_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    active: bool,
) -> Vec<Value> {
    if role == AgentRole::Do && active {
        Vec::new()
    } else {
        definitions
    }
}

/// Return true only when a child owns an exact write lease and every declared
/// output is now a regular file at the still-authorized path.  This is a
/// convergence hint, not acceptance evidence: CA remains responsible for
/// content and criterion validation.
pub(super) fn exact_workspace_write_targets_materialized(
    lease: Option<&crate::core::effect::WorkspaceResourceLease>,
) -> bool {
    let Some(lease) = lease else {
        return false;
    };
    if lease.validate().is_err() {
        return false;
    }

    let mut write_target_count = 0usize;
    for target in lease.paths.iter().filter(|target| {
        matches!(
            target.access,
            crate::core::effect::WorkspaceLeaseAccess::Write
                | crate::core::effect::WorkspaceLeaseAccess::Exclusive
        )
    }) {
        write_target_count = write_target_count.saturating_add(1);
        let input = json!({"path": target.relative_path});
        let materialized = lease
            .resolve_authorized_file_tool_path("file_write", &input)
            .ok()
            .flatten()
            .and_then(|path| std::fs::metadata(path).ok())
            .is_some_and(|metadata| metadata.is_file());
        if !materialized {
            return false;
        }
    }
    write_target_count > 0
}

/// A stable verifier receipt is not proof that every requested artifact has
/// been created. Top-level/mono required-mutation DA work therefore keeps its
/// narrowed repair window until the model explicitly finishes. An isolated
/// BizAgent child with a fully materialized exact write lease may close after
/// the normal convergence window: its complete artifact boundary is known,
/// while downstream CA still decides correctness and completeness.
pub(super) fn da_hard_close_active(
    convergence_threshold_reached: bool,
    workspace_effect_required: bool,
    exact_write_contract_materialized: bool,
) -> bool {
    convergence_threshold_reached
        && (!workspace_effect_required || exact_write_contract_materialized)
}

pub(crate) const SA_RECOVERY_MODE_CONSTRAINT: &str = "sa_recovery_mode";
pub(crate) const CA_DA_CORRECTION_MODE: &str = "ca_da_correction";

pub(super) fn initial_execution_phase(
    role: AgentRole,
    constraints: &std::collections::HashMap<String, String>,
) -> ExecutionPhase {
    if role == AgentRole::Check {
        ExecutionPhase::Verify
    } else if role == AgentRole::Do
        && constraints
            .get(SA_RECOVERY_MODE_CONSTRAINT)
            .is_some_and(|mode| mode == CA_DA_CORRECTION_MODE)
    {
        ExecutionPhase::Repair
    } else {
        ExecutionPhase::Inspect
    }
}

/// A fresh DA launched from a typed CA defect already has its discovery
/// evidence in the required correction-handoff context.  Starting it with an
/// additional broad discovery grace period lets the model ignore that defect,
/// re-prove that tests currently pass, and consume most of its turn budget
/// without repairing anything.  Enter the same bounded Repair gate on its
/// first provider dispatch; the one target read and one verifier remain
/// available, while real mutation calls are never quota-limited.
pub(super) fn immediate_correction_recovery_active(
    workspace_effect_required: bool,
    phase: ExecutionPhase,
    correction_handoff_present: bool,
    constraints: &std::collections::HashMap<String, String>,
) -> bool {
    workspace_effect_required
        && phase == ExecutionPhase::Repair
        && correction_handoff_present
        && constraints
            .get(SA_RECOVERY_MODE_CONSTRAINT)
            .is_some_and(|mode| mode == CA_DA_CORRECTION_MODE)
}

pub(super) fn effective_effect_block_turns(
    phase: ExecutionPhase,
    general_turns: u32,
    repair_turns: u32,
) -> u32 {
    if phase == ExecutionPhase::Repair && repair_turns > 0 {
        repair_turns
    } else {
        general_turns
    }
}

pub(super) fn phase_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    phase: ExecutionPhase,
) -> Vec<Value> {
    if role != AgentRole::Do || phase == ExecutionPhase::Inspect {
        return definitions;
    }
    definitions
        .into_iter()
        .filter(|definition| {
            let name = definition["function"]["name"].as_str().unwrap_or_default();
            match phase {
                ExecutionPhase::Implement | ExecutionPhase::Repair => !matches!(
                    name,
                    "file_list" | "glob_search" | "grep_search" | "workspace_status"
                ),
                ExecutionPhase::Verify => !matches!(name, "file_list" | "glob_search"),
                ExecutionPhase::Inspect => true,
            }
        })
        .collect()
}

/// If the bounded manifest contains the complete inventory, another broad
/// list/glob/status call cannot discover a file the Agent was not already
/// shown. Targeted reads/searches remain available for content retrieval.
pub(super) fn workspace_inventory_tool_definitions(
    definitions: Vec<Value>,
    inventory_complete_and_bounded: bool,
) -> Vec<Value> {
    if !inventory_complete_and_bounded {
        return definitions;
    }
    definitions
        .into_iter()
        .filter(|definition| {
            !matches!(
                definition["function"]["name"].as_str().unwrap_or_default(),
                "file_list" | "glob_search" | "workspace_status"
            )
        })
        .collect()
}

/// An empty, fully scanned workspace is already complete local evidence for
/// PA. Planning can proceed from the authoritative task contract; advertising
/// discovery tools in this state only invites searches for files that cannot
/// exist yet (or for runtime-private files that are outside the user scope).
/// DA still receives its normal creation tools, and every role retains the
/// ordinary tool window when the scan is incomplete, truncated, or non-empty.
pub(super) fn pa_empty_workspace_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    authoritative_empty_workspace: bool,
) -> Vec<Value> {
    if role == AgentRole::Plan && authoritative_empty_workspace {
        Vec::new()
    } else {
        definitions
    }
}

/// Snapshot the exact tool window advertised for one LLM turn.  Tool schemas
/// are intentionally phase-sensitive, so execution must validate against the
/// same snapshot rather than the runner's broader registry.
pub(super) fn advertised_tool_names(definitions: &[Value]) -> HashSet<String> {
    definitions
        .iter()
        .filter_map(|definition| definition["function"]["name"].as_str())
        .map(str::to_string)
        .collect()
}

/// Restrict live-catalog search results to the capability set that can become
/// active in this exact task phase. `tool_search` is an on-demand discovery
/// mechanism, so filtering only the schemas already in the prompt would make
/// it useless; filtering against the phase-authorized full catalog prevents it
/// from leaking withdrawn or application-disabled tools.
pub(super) fn filter_tool_search_result(result: &mut Value, discoverable_tools: &HashSet<String>) {
    let Some(matches) = result.get_mut("matches").and_then(Value::as_array_mut) else {
        return;
    };
    matches.retain(|item| {
        item.get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| discoverable_tools.contains(name))
    });
    let count = matches.len();
    if let Some(object) = result.as_object_mut() {
        object.insert("count".to_string(), Value::from(count));
    }
}

/// Keep discovered ordinary tools active for the current execution, but only
/// advertise dynamic result readers that are still referenced by the current
/// message window. This preserves valid compressed-result links without
/// turning every historical reader into permanent tool-schema overhead.
pub(super) fn active_session_tool_names(
    messages: &[ChatMessage],
    session_tools: &HashSet<String>,
) -> HashSet<String> {
    session_tools
        .iter()
        .filter(|name| {
            !ToolExecutor::is_micro_tool_name(name)
                || messages.iter().any(|message| {
                    message.content.contains(name.as_str())
                        || message.tool_calls.as_ref().is_some_and(|calls| {
                            calls.iter().any(|call| call.function.name == **name)
                        })
                })
        })
        .cloned()
        .collect()
}

pub(super) fn unadvertised_tool_call_result(
    advertised: &HashSet<String>,
    _session_tools: &HashSet<String>,
    tool_name: &str,
) -> Option<Value> {
    // Ownership is necessary for a dynamic result reader to be eligible for
    // advertisement, but it is never execution authority by itself.  The
    // provider must have received the exact schema in this request; otherwise
    // a remembered or fabricated reader name is rejected like any other
    // withdrawn tool.
    (!advertised.contains(tool_name)).then(|| {
        json!({
            "status": "not_executed",
            "reason": "tool_not_advertised",
            "message": format!(
                "Tool {tool_name} is unavailable in the current execution phase and was not executed"
            ),
            "required_next_action": "Use only a tool advertised in the current turn, or finish with the exact blocker."
        })
    })
}

/// Paged readers already return a bounded terminal page. Feeding their output
/// back into the history compressor would synthesize a reader for the reader's
/// provider call ID, even though result routing intentionally never registers
/// such a chain. Keep these bounded pages inline/ageable without creating a
/// false capability hint.
pub(super) fn should_track_result_for_compression(tool_name: &str) -> bool {
    tool_name != "read_agent_output" && !ToolExecutor::is_micro_tool_name(tool_name)
}

pub(super) fn ca_evidence_focus_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    evidence_focus_active: bool,
    retain_live_retrieval: bool,
    tool_executor: &parking_lot::RwLock<ToolExecutor>,
) -> Vec<Value> {
    if role != AgentRole::Check {
        return definitions;
    }
    let tool_executor = tool_executor.read();
    definitions
        .into_iter()
        .filter(|definition| {
            let name = definition["function"]["name"].as_str().unwrap_or_default();
            let file_read_session_reader = ToolExecutor::is_micro_tool_name(name)
                && tool_executor.micro_tool_originates_from(name, "file_read");
            if file_read_session_reader {
                return false;
            }
            !evidence_focus_active
                || matches!(
                    name,
                    "bash" | "powershell" | "file_read" | "read_agent_output" | "mermaid_validate"
                )
                || (retain_live_retrieval && matches!(name, "web_search" | "web_fetch"))
                || ToolExecutor::is_micro_tool_name(name)
        })
        .collect()
}

/// CA has two separate progress dimensions:
///
/// * an audit-tool window bounds broad inspection and discovery;
/// * an executable verification receipt proves that a deterministic check
///   was actually attempted against the workspace.
///
/// Historically both meanings were stored in `verification_turns`, which was
/// incremented before execution for every non-empty batch. A provider call
/// declined by protocol, role, effect or verification preflight could therefore
/// exhaust the CA window even though no tool handler ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CaAuditConvergence {
    audit_tool_turns: u32,
    verification_probe_turns: u32,
    verifier_attempted: bool,
    successful_verifier_observed: bool,
}

impl CaAuditConvergence {
    /// Record one model turn whose tool batch reached at least one tool
    /// handler. The caller must invoke this at most once per provider batch.
    pub(super) fn record_executed_tool_turn(&mut self, role: AgentRole, probe_active: bool) {
        if role != AgentRole::Check {
            return;
        }
        self.audit_tool_turns = self.audit_tool_turns.saturating_add(1);
        if probe_active {
            self.verification_probe_turns = self.verification_probe_turns.saturating_add(1);
        }
    }

    pub(super) fn record_executed_call(
        &mut self,
        role: AgentRole,
        name: &str,
        args: &Value,
        result: &Value,
    ) {
        if role != AgentRole::Check || !is_verification_call(name, args) {
            return;
        }
        self.verifier_attempted = true;
        if is_successful_verification_call(name, args, result) {
            self.successful_verifier_observed = true;
        }
    }

    pub(super) fn focus_active(self, role: AgentRole, threshold: u32) -> bool {
        role == AgentRole::Check && threshold > 0 && self.audit_tool_turns >= threshold
    }

    pub(super) fn window_exhausted(self, role: AgentRole, threshold: u32) -> bool {
        self.focus_active(role, threshold)
    }

    pub(super) fn verification_probe_active(
        self,
        role: AgentRole,
        threshold: u32,
        executable_verifier_available: bool,
    ) -> bool {
        executable_verifier_available
            && self.window_exhausted(role, threshold)
            && !self.verifier_attempted
            && self.verification_probe_turns == 0
    }

    pub(super) fn close_active(
        self,
        role: AgentRole,
        threshold: u32,
        executable_verifier_available: bool,
    ) -> bool {
        self.window_exhausted(role, threshold)
            && (!executable_verifier_available
                || self.verifier_attempted
                || self.verification_probe_turns > 0)
    }

    pub(super) fn successful_verifier_observed(self) -> bool {
        self.successful_verifier_observed
    }
}

/// Bind CA's in-memory convergence bit to the durable action ledger before it
/// can authorize a positive terminal verdict.  The model can mention a green
/// command in prose or in `ca_audit/v1`; neither is a receipt.  A matching
/// successful, non-mutating CA action must also exist in the result ledger.
pub(super) fn ca_has_successful_verifier_receipt(
    convergence: CaAuditConvergence,
    action_tracker: &crate::core::tracked_action::ActionTracker,
) -> bool {
    convergence.successful_verifier_observed()
        && matches!(action_tracker.agent_role.as_str(), "CA" | "Check")
        && !action_tracker
            .current_successful_verification_receipt_sha256s()
            .is_empty()
}

/// When CA has exhausted broad inspection without an executable verifier,
/// expose one final, narrow verification window. This is an intersection with
/// the already authorized tools and cannot grant a new capability.
pub(super) fn is_ca_verification_tool_name(name: &str) -> bool {
    matches!(
        name,
        "bash"
            | "powershell"
            | "code_execute"
            | "mermaid_validate"
            | "jsonld_validate"
            | "ontology_validate_turtle"
            | "ontology_validate_shacl"
            | "ontology_lint_turtle"
    )
}

pub(super) fn ca_verification_probe_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    verification_probe_active: bool,
) -> Vec<Value> {
    if role != AgentRole::Check || !verification_probe_active {
        return definitions;
    }
    definitions
        .into_iter()
        .filter(|definition| {
            is_ca_verification_tool_name(
                definition["function"]["name"].as_str().unwrap_or_default(),
            )
        })
        .collect()
}

pub(super) fn ca_evidence_close_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    evidence_close_active: bool,
) -> Vec<Value> {
    if role == AgentRole::Check && evidence_close_active {
        Vec::new()
    } else {
        definitions
    }
}

pub(super) fn da_evidence_focus_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    evidence_focus_active: bool,
    retain_live_retrieval: bool,
) -> Vec<Value> {
    if role != AgentRole::Do || !evidence_focus_active {
        return definitions;
    }
    definitions
        .into_iter()
        .filter(|definition| {
            let name = definition["function"]["name"].as_str().unwrap_or_default();
            matches!(
                name,
                "file_read" | "web_fetch" | "http_request" | "read_agent_output"
            ) || (retain_live_retrieval && name == "web_search")
                || ToolExecutor::is_micro_tool_name(name)
        })
        .collect()
}

pub(super) fn da_evidence_close_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    evidence_close_active: bool,
) -> Vec<Value> {
    if role == AgentRole::Do && evidence_close_active {
        Vec::new()
    } else {
        definitions
    }
}

fn normalize_contract_observed_path(raw: &str, workspace_root: Option<&Path>) -> Option<String> {
    if let Some(relative) = crate::core::sa::normalize_work_package_artifact_path(raw) {
        return Some(relative);
    }
    let observed = Path::new(raw.trim());
    if !observed.is_absolute() {
        return None;
    }
    let root = std::fs::canonicalize(workspace_root?).ok()?;
    let observed = std::fs::canonicalize(observed).ok()?;
    let relative = observed.strip_prefix(root).ok()?;
    crate::core::sa::normalize_work_package_artifact_path(
        &relative.to_string_lossy().replace('\\', "/"),
    )
}

fn contract_trusted_workspace_mutation(
    action: &crate::core::tracked_action::TrackedAction,
) -> bool {
    action.status == crate::core::tracked_action::ActionStatus::Success
        && action.error.is_none()
        && action.substantive_effect
        && action.workspace_delta_complete
        && !action.workspace_delta_contaminated
        && (!action.files_created.is_empty()
            || !action.files_modified.is_empty()
            || !action.files_removed.is_empty()
            || !action.directories_created.is_empty()
            || !action.directories_removed.is_empty())
}

fn contract_directory_is_declared_ancestor(directory: &str, declared: &BTreeSet<String>) -> bool {
    let prefix = format!("{}/", directory.trim_end_matches('/'));
    declared.iter().any(|path| path.starts_with(&prefix))
}

/// Close any canonical DA child as soon as every AND-linked requirement in
/// its kernel-authenticated work-package contract has current typed evidence.
///
/// The gate cannot be activated from prompt/input JSON and independently
/// reconstructs exact path ownership, mutation cardinality, verifier kind and
/// test-artifact scope from the action ledger. The fresh Agent still emits its
/// own terminal result, but cannot spend extra turns rereading a routed verifier
/// output or re-inspecting an artifact after the exact contract is satisfied.
pub(super) fn da_typed_contract_close_directive(
    role: AgentRole,
    context: &super::TaskContext,
    action_tracker: &crate::core::tracked_action::ActionTracker,
) -> Option<String> {
    if role != AgentRole::Do {
        return None;
    }
    let package = context.biz_agent_child_evidence_contract.as_deref()?;
    crate::core::sa::validate_work_package_evidence_requirements(package, true).ok()?;
    let workspace_root = context
        .workspace_resource_lease
        .as_ref()
        .map(|lease| lease.workspace_root.as_path());
    let verification_evidence =
        crate::core::tracked_action::current_verification_close_candidates(&action_tracker.actions);
    let mut delivered_paths = BTreeSet::new();
    let mut mutation_actions = 0usize;
    for action in &action_tracker.actions {
        if contract_trusted_workspace_mutation(action) {
            mutation_actions = mutation_actions.saturating_add(1);
            delivered_paths.extend(
                action
                    .files_created
                    .iter()
                    .chain(&action.files_modified)
                    .filter_map(|change| {
                        normalize_contract_observed_path(&change.path, workspace_root)
                    }),
            );
        }
        for removed in &action.files_removed {
            if let Some(path) = normalize_contract_observed_path(&removed.path, workspace_root) {
                delivered_paths.remove(&path);
            }
        }
        if let Some(attestation) = action.artifact_attestation_close_candidate() {
            if let Some(path) = normalize_contract_observed_path(&attestation.path, workspace_root)
            {
                delivered_paths.insert(path);
            }
        }
    }

    let mut proofs = Vec::with_capacity(package.evidence_requirements.len());
    for requirement in &package.evidence_requirements {
        match requirement {
            crate::core::sa::WorkPackageEvidenceRequirement::ArtifactDelivery {
                paths,
                min_paths,
            } => {
                let declared = paths.iter().cloned().collect::<BTreeSet<_>>();
                for action in &action_tracker.actions {
                    for change in action
                        .files_created
                        .iter()
                        .chain(&action.files_modified)
                        .chain(&action.files_removed)
                    {
                        let observed =
                            normalize_contract_observed_path(&change.path, workspace_root)?;
                        if !declared.contains(&observed) {
                            return None;
                        }
                    }
                    for directory in &action.directories_created {
                        let observed = normalize_contract_observed_path(directory, workspace_root)?;
                        if !contract_directory_is_declared_ancestor(&observed, &declared) {
                            return None;
                        }
                    }
                    if !action.directories_removed.is_empty() {
                        return None;
                    }
                }
                let matched = declared
                    .iter()
                    .filter(|path| delivered_paths.contains(*path))
                    .cloned()
                    .collect::<Vec<_>>();
                if matched.len() < *min_paths as usize {
                    return None;
                }
                proofs.push(json!({
                    "type": "artifact_delivery",
                    "required_paths": paths,
                    "observed_paths": matched,
                }));
            }
            crate::core::sa::WorkPackageEvidenceRequirement::WorkspaceMutation { min_actions } => {
                if mutation_actions < *min_actions as usize {
                    return None;
                }
                proofs.push(json!({
                    "type": "workspace_mutation",
                    "required_actions": min_actions,
                    "observed_actions": mutation_actions,
                }));
            }
            crate::core::sa::WorkPackageEvidenceRequirement::ExternalResearch => {
                let action = action_tracker.actions.iter().find(|action| {
                    matches!(
                        action.tool_name.as_str(),
                        "web_search" | "web_fetch" | "http_request"
                    ) && action.status == crate::core::tracked_action::ActionStatus::Success
                        && action.error.is_none()
                        && action.call_identity.is_some()
                        && action.disclosure.as_ref().is_some_and(|disclosure| {
                            // The routed result is queued in the next provider
                            // request. That request confirms disclosure before
                            // its response is admitted; aggregation still
                            // requires the confirmed bit.
                            !disclosure.result_withheld
                        })
                })?;
                proofs.push(json!({
                    "type": "external_research",
                    "action_id": action.action_id,
                    "tool": action.tool_name,
                    "call_identity": action.call_identity,
                }));
            }
            crate::core::sa::WorkPackageEvidenceRequirement::ResponseDelivery => {
                // The response itself is materialized by the terminal-only
                // submission dispatch.  Treat it as pending here so a
                // package that intentionally combines live research and the
                // direct response can close immediately after its retrieval
                // receipt. Aggregation still validates that the decoded
                // output is non-empty, so this marker cannot prove delivery
                // by itself.
                proofs.push(json!({
                    "type": "response_delivery",
                    "receipt_state": "pending_this_terminal_dispatch",
                }));
            }
            crate::core::sa::WorkPackageEvidenceRequirement::Verification { kind, min_count } => {
                let matched = verification_evidence
                    .iter()
                    .filter(|item| item.assessment.kind == *kind)
                    .filter(|item| {
                        item.assessment.outcome
                            == crate::core::tracked_action::VerificationOutcome::Passed
                    })
                    .max_by_key(|item| {
                        (
                            item.assessment.count.unwrap_or(0),
                            item.workspace_settlement_sequence,
                            item.completion_sequence,
                        )
                    })?;
                let observed_count = matched.assessment.count.unwrap_or(0);
                if observed_count < *min_count {
                    return None;
                }
                let receipt_sha256 = matched.successful_receipt_sha256.as_deref();
                proofs.push(json!({
                    "type": "verification",
                    "kind": kind,
                    "required_count": min_count,
                    "observed_count": observed_count,
                    "receipt_sha256": receipt_sha256,
                    "receipt_state": if receipt_sha256.is_some() {
                        "confirmed"
                    } else {
                        "pending_this_terminal_dispatch"
                    },
                    "workspace_mutation_epoch": matched.workspace_mutation_epoch,
                    "workspace_manifest_sha256": matched.workspace_manifest_sha256,
                }));
            }
            crate::core::sa::WorkPackageEvidenceRequirement::TestArtifactExecutionScope {
                paths,
            } => {
                let observed = verification_evidence
                    .iter()
                    .filter(|item| {
                        item.assessment.kind
                            == crate::core::tracked_action::VerificationKind::TestExecution
                            && item.assessment.outcome
                                == crate::core::tracked_action::VerificationOutcome::Passed
                    })
                    .filter_map(|item| {
                        let action = action_tracker
                            .actions
                            .iter()
                            .find(|action| action.action_id == item.action_id)?;
                        let args = serde_json::to_value(&action.tool_args).ok()?;
                        explicit_test_execution_target_path(
                            &action.tool_name,
                            &args,
                            workspace_root,
                        )
                    })
                    .collect::<BTreeSet<_>>();
                if paths.iter().any(|path| !observed.contains(path)) {
                    return None;
                }
                proofs.push(json!({
                    "type": "test_artifact_execution_scope",
                    "required_paths": paths,
                    "observed_paths": observed,
                }));
            }
        }
    }
    let receipt = json!({
        "schema_version": 1,
        "work_package_id": package.id,
        "and_semantics": true,
        "satisfied_requirements": proofs,
    });
    if package.evidence_requirements.iter().any(|requirement| {
        matches!(
            requirement,
            crate::core::sa::WorkPackageEvidenceRequirement::ExternalResearch
        )
    }) {
        return Some(format!(
            "[DA Typed Contract Close] The kernel-observed evidence contract is fully satisfied: {receipt}. Submit the complete evidence-backed work-package result now through the single result-submission function offered in this dispatch. Preserve the useful research notes and source URLs already present in this L1; disclose blocked fetches as limitations. Do not call or print another retrieval tool. This gate proves only the assigned canonical package, not any unrelated deliverable."
        ));
    }
    Some(format!(
        "[DA Typed Contract Close] The kernel-observed current-workspace evidence contract is fully satisfied: {receipt}. No tools are available this turn. Required artifacts/mutations/verifier results have already been admitted and assessed; do not reread artifacts, do not request read_full_result, and do not run another probe. Return the terminal result now with this receipt-bound evidence. This gate proves only the assigned canonical package, not any unrelated deliverable."
    ))
}

pub(super) fn pa_planning_focus_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    planning_focus_active: bool,
) -> Vec<Value> {
    if role == AgentRole::Plan && planning_focus_active {
        Vec::new()
    } else {
        definitions
    }
}

pub(super) fn workspace_inventory_complete_and_bounded(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
    max_manifest_files: usize,
) -> bool {
    executor
        .read()
        .get_workspace_monitor()
        .map(|monitor| {
            let view = monitor.workspace_view(None, None, max_manifest_files.max(1));
            view.scan_complete && !view.truncated
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WorkspaceInventoryCoverage {
    pub scan_complete: bool,
    pub truncated: bool,
    pub total_files: usize,
}

impl WorkspaceInventoryCoverage {
    pub(super) fn complete_and_bounded(self) -> bool {
        self.scan_complete && !self.truncated
    }

    pub(super) fn authoritative_empty(self) -> bool {
        self.complete_and_bounded() && self.total_files == 0
    }
}

pub(super) fn workspace_inventory_coverage(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
    max_manifest_files: usize,
) -> Option<WorkspaceInventoryCoverage> {
    executor.read().get_workspace_monitor().map(|monitor| {
        let view = monitor.workspace_view(None, None, max_manifest_files.max(1));
        WorkspaceInventoryCoverage {
            scan_complete: view.scan_complete,
            truncated: view.truncated,
            total_files: view.total_files,
        }
    })
}

pub(super) fn workspace_inventory_authoritatively_empty(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
    max_manifest_files: usize,
) -> bool {
    workspace_inventory_coverage(executor, max_manifest_files)
        .is_some_and(WorkspaceInventoryCoverage::authoritative_empty)
}

pub(super) fn evidence_key(name: &str, args: &Value, generation: u64) -> Option<String> {
    let is_evidence = matches!(
        name,
        "file_read"
            | "file_list"
            | "glob_search"
            | "grep_search"
            | "workspace_status"
            | "tool_search"
    );
    is_evidence.then(|| {
        let canonical = serde_json::to_string(args).unwrap_or_default();
        format!("{generation}:{name}:{canonical}")
    })
}

pub(super) fn refresh_execution_ledger(
    runtime_context: &mut crate::core::context_model::RoleContext,
    role: AgentRole,
    phase: ExecutionPhase,
    effect_policy: &crate::core::effect::EffectPolicy,
    mutation_count: u32,
    verification_turns: u32,
    low_novelty_turns: u32,
    workspace_generation: u64,
) {
    if role != AgentRole::Do {
        return;
    }
    let fragment = crate::core::context_model::ContextFragment::new(
        crate::core::context_model::ContextSlot::ExecutionLedger,
        crate::core::context_model::ContextFragmentKind::AuthoritativeInstruction,
        "Execution Ledger",
        format!(
            "# Execution Ledger (replaceable state)\n\n- phase: {:?}\n- effect_policy: {:?}\n- substantive_effects: {}\n- verification_tool_turns_after_effect: {}\n- low_novelty_evidence_score: {}\n- workspace_generation: {}\n\nContinue from this state. Do not repeat evidence queries that produced no new generation or target information.",
            phase,
            effect_policy,
            mutation_count,
            verification_turns,
            low_novelty_turns,
            workspace_generation,
        ),
        crate::core::context_model::ContextSourceRecord::new(
            crate::core::context_model::ContextSourceKind::RuntimeController,
        )
        .with_source_ref("execution_ledger")
        .with_producer("AgentRunner"),
    )
    .required()
    .with_priority(100)
    .with_max_chars(8 * 1024)
    .with_freshness(crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
        ttl_seconds: 300,
    });
    if let Err(error) = runtime_context.upsert(fragment) {
        warn!(%error, "execution ledger context rejected");
    }
}

/// Keep a compact, kernel-authenticated record of evidence already disclosed
/// to the current isolated CA. This is coverage metadata, not a semantic
/// verdict: it prevents context compression from making CA rediscover files
/// or rerun verifiers, while the model must still compare the content it
/// actually observed and emit the audited claim itself.
pub(super) fn refresh_ca_evidence_ledger(
    runtime_context: &mut crate::core::context_model::RoleContext,
    role: AgentRole,
    action_tracker: &crate::core::tracked_action::ActionTracker,
) {
    if role != AgentRole::Check {
        return;
    }

    let reads = action_tracker
        .actions
        .iter()
        .filter_map(|action| {
            let disclosure = action.disclosure.as_ref()?;
            if !disclosure.disclosed_to_model || disclosure.result_withheld {
                return None;
            }
            let read = disclosure.file_read.as_ref()?;
            Some(json!({
                "action_id": action.action_id,
                "path": read.path,
                "offset": read.offset,
                "returned_lines": read.returned,
                "total_lines": read.total_lines,
                "delivered_lines": read.delivered_line_count,
                "content_sha256": read.content_sha256,
                "complete_visible_revision": read.offset == 0
                    && read.returned == read.total_lines
                    && read.delivered_line_count == read.returned
                    && !read.partial_line_preview
                    && !read.archived,
            }))
        })
        .collect::<Vec<_>>();
    let verifiers = action_tracker
        .actions
        .iter()
        .filter_map(|action| {
            let assessment = action.verification_assessment()?;
            Some(json!({
                "action_id": action.action_id,
                "tool": action.tool_name,
                "kind": assessment.kind,
                "outcome": assessment.outcome,
                "count": assessment.count,
                "skipped_count": assessment.skipped_count,
                "reason": assessment.reason,
                "successful_receipt_sha256": action.successful_verification_receipt_sha256(),
            }))
        })
        .collect::<Vec<_>>();
    if reads.is_empty() && verifiers.is_empty() {
        return;
    }

    let payload = json!({
        "schema_version": "glidinghorse.ca-evidence-ledger/v1",
        "file_read_disclosures": reads,
        "verifier_assessments": verifiers,
    });
    let fragment = crate::core::context_model::ContextFragment::new(
        crate::core::context_model::ContextSlot::ExecutionLedger,
        crate::core::context_model::ContextFragmentKind::AuthoritativeInstruction,
        "CA Evidence Ledger",
        format!(
            "# CA Evidence Ledger (coverage metadata only)\n\n{}\n\nDo not reread a path whose complete_visible_revision is true. Do not rerun a passed verifier merely to reconstruct prose. This ledger proves disclosure/assessment coverage only; it does not decide semantic conformance or the final verdict.",
            serde_json::to_string_pretty(&payload).unwrap_or_default()
        ),
        crate::core::context_model::ContextSourceRecord::new(
            crate::core::context_model::ContextSourceKind::RuntimeController,
        )
        .with_source_ref("ca_evidence_ledger")
        .with_producer("AgentRunner"),
    )
    .required()
    .with_priority(100)
    .with_max_chars(24 * 1024)
    .with_freshness(crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
        ttl_seconds: 300,
    });
    if let Err(error) = runtime_context.upsert(fragment) {
        warn!(%error, "CA evidence ledger context rejected");
    }
}

pub(super) fn final_turn_limit_notice(
    role: AgentRole,
    workspace_effect_required: bool,
    workspace_effect_observed: bool,
    consecutive_effectless_tool_turns: u32,
) -> String {
    if role == AgentRole::Do && workspace_effect_required {
        let progress = if workspace_effect_observed {
            format!(
                "A workspace mutation was made earlier, but the current no-change tail is {} tool turn(s).",
                consecutive_effectless_tool_turns
            )
        } else {
            "No substantive workspace mutation has succeeded yet.".to_string()
        };
        return format!(
            "【Turn Limit Urgent】The DA budget is nearly exhausted. {} Use the remaining turns for the highest-priority incomplete implementation with file_write/file_edit or a genuinely mutating command, followed by only targeted verification. Do not start broad inspection. If implementation is impossible, finish with `FAILED:` and the exact blocker.",
            progress
        );
    }

    "【Turn Limit Urgent】The role-specific budget is nearly exhausted. Finish from the evidence already collected and output the final result now. Do not initiate new tool calls.".to_string()
}

use crate::core::agent_instance::{AgentInstance, AgentRole, AgentStatus};
use crate::core::execution_event::{ExecutionEvent, ExecutionEventKind, ToolExecutionEventLedger};
use crate::core::execution_journal::{
    TaskExecutionJournal, TaskExecutionJournalKind, ToolCallIdentity,
};
use crate::gateway::unified_gateway::ChatMessage;
use crate::jsonld::{JsonLdContext, JsonLdNode};
use crate::memory::l1_session::L1Session;
use crate::tools::hooks::{HookContext, HookControl, HookDecision, HookManager, HookPoint};
use crate::tools::tool_executor::ToolExecutor;
use crate::CoreError;

use super::{TaskContext, TaskResult, TaskVerdict};

/// Build the one terminal-format correction request from the same Agent/L1's
/// immutable prompt identity plus bounded recent analysis notes. Old native
/// tool protocol pairs are deliberately omitted from this request: their
/// exact four-part identities remain in the durable journal and action
/// ledger, while replaying their syntax after tools close makes some
/// providers emit another textual tool request instead of the audit object.
pub(super) fn ca_terminal_format_retry_history(
    messages: &[ChatMessage],
    immutable_prefix_len: usize,
) -> Vec<ChatMessage> {
    let prefix_len = immutable_prefix_len.min(messages.len());
    let mut compact = messages[..prefix_len].to_vec();
    let mut notes = Vec::new();
    let mut remaining = 16 * 1024usize;
    for message in messages[prefix_len..].iter().rev() {
        if message.role != "assistant" || message.tool_calls.is_some() {
            continue;
        }
        let content = message.content.trim();
        if is_raw_tool_protocol_content(content) {
            continue;
        }
        let reasoning = message.reasoning_content.as_deref().unwrap_or("").trim();
        if content.is_empty() && reasoning.is_empty() {
            continue;
        }
        let combined = match (reasoning.is_empty(), content.is_empty()) {
            (false, false) => format!("analysis:\n{reasoning}\n\nprior summary:\n{content}"),
            (false, true) => format!("analysis:\n{reasoning}"),
            (true, false) => format!("prior summary:\n{content}"),
            (true, true) => unreachable!(),
        };
        let take = remaining.min(6 * 1024);
        let char_count = combined.chars().count();
        let bounded = if char_count > take {
            let start = char_count.saturating_sub(take);
            format!(
                "[earlier note prefix omitted]\n{}",
                combined.chars().skip(start).collect::<String>()
            )
        } else {
            combined
        };
        remaining = remaining.saturating_sub(bounded.chars().count());
        notes.push(bounded);
        if remaining == 0 || notes.len() >= 4 {
            break;
        }
    }
    notes.reverse();
    if !notes.is_empty() {
        compact.push(ChatMessage {
            role: "user".to_string(),
            content: format!(
                "[Same-L1 CA analysis notes — model history, not independent evidence]\n\n{}\n\nUse these notes only to construct the required terminal audit. Kernel-authenticated coverage and verifier metadata are supplied separately by the CA Evidence Ledger.",
                notes.join("\n\n---\n\n")
            ),
            name: Some("context_model_history".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    compact
}

/// Build a single DA terminal retry from the same immutable Agent/L1 prompt
/// plus bounded evidence already disclosed in that L1. Native tool-call
/// syntax and provider call ids are omitted from the stateless retry so a
/// provider cannot copy a withdrawn capability, while the original identities
/// remain unchanged in the durable request/action journals.
pub(super) fn da_terminal_result_retry_history(
    messages: &[ChatMessage],
    immutable_prefix_len: usize,
) -> Vec<ChatMessage> {
    let prefix_len = immutable_prefix_len.min(messages.len());
    let mut compact = messages[..prefix_len].to_vec();
    let mut call_names = std::collections::HashMap::<String, String>::new();
    for message in &messages[prefix_len..] {
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                call_names.insert(call.id.clone(), call.function.name.clone());
            }
        }
    }

    let mut evidence = Vec::new();
    let mut remaining = 48 * 1024usize;
    for message in messages[prefix_len..].iter().rev() {
        let item = if message.role == "tool" {
            let call_id = message.tool_call_id.as_deref().unwrap_or("unknown");
            let tool_name = call_names
                .get(call_id)
                .map(String::as_str)
                .unwrap_or("tool");
            Some(format!(
                "verified tool output ({tool_name}):\n{}",
                message.content.trim()
            ))
        } else if message.role == "assistant" && message.tool_calls.is_none() {
            let content = message.content.trim();
            let reasoning = message.reasoning_content.as_deref().unwrap_or("").trim();
            match (reasoning.is_empty(), content.is_empty()) {
                (false, false) => Some(format!(
                    "model analysis:\n{reasoning}\n\nprior response draft:\n{content}"
                )),
                (false, true) => Some(format!("model analysis:\n{reasoning}")),
                (true, false) if !is_raw_tool_protocol_content(content) => {
                    Some(format!("prior response draft:\n{content}"))
                }
                _ => None,
            }
        } else {
            None
        };
        let Some(item) = item.filter(|item| !item.trim().is_empty()) else {
            continue;
        };
        let take = remaining.min(12 * 1024);
        let char_count = item.chars().count();
        let bounded = if char_count > take {
            format!(
                "[earlier prefix omitted]\n{}",
                item.chars()
                    .skip(char_count.saturating_sub(take))
                    .collect::<String>()
            )
        } else {
            item
        };
        remaining = remaining.saturating_sub(bounded.chars().count());
        evidence.push(bounded);
        if remaining == 0 || evidence.len() >= 12 {
            break;
        }
    }
    evidence.reverse();
    if !evidence.is_empty() {
        compact.push(ChatMessage {
            role: "user".to_string(),
            content: format!(
                "[Same-L1 DA disclosed evidence and analysis — historical tool syntax removed]\n\n{}\n\nUse this already-disclosed evidence only to construct the terminal deliverable. Do not infer that any omitted or blocked retrieval succeeded.",
                evidence.join("\n\n---\n\n")
            ),
            name: Some("context_verified_evidence".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    compact
}

/// Hash the exact tool messages that survived context assembly for a provider
/// request. Payloads remain transient; ActionTracker uses the hashes only to
/// commit matching queued disclosure receipts after that request succeeds.
pub(super) fn visible_tool_result_hashes(messages: &[ChatMessage]) -> Vec<(String, String)> {
    messages
        .iter()
        .filter(|message| message.role == "tool")
        .filter_map(|message| {
            Some((
                message.tool_call_id.clone()?,
                crate::utils::CryptoUtils::sha256_hex(&message.content),
            ))
        })
        .collect()
}

/// Tool-call identifiers are opaque provider protocol values.  They must be
/// preserved verbatim for the assistant/tool message handshake, while their
/// uniqueness is scoped to one concrete provider response/model request.
///
/// Keeping this ledger local to `exec`/`execute_streaming_inner` is
/// intentional: different requests, Agent instances, and L1 sessions may all
/// receive a provider-local identifier such as `call_0` without colliding.
/// Within one response batch duplicates are ambiguous; repeated admission for
/// the same model request is also rejected to keep replay fail-closed.
#[derive(Debug, Default)]
pub(super) struct ProviderToolCallLedger {
    consumed_ids: HashSet<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ProviderToolCallProtocolViolation {
    EmptyId {
        batch_index: usize,
    },
    DuplicateInBatch {
        batch_index: usize,
        provider_call_id: String,
    },
    ReusedInRequest {
        batch_index: usize,
        provider_call_id: String,
    },
}

impl ProviderToolCallProtocolViolation {
    fn code(&self) -> &'static str {
        match self {
            Self::EmptyId { .. } => "empty_id",
            Self::DuplicateInBatch { .. } => "duplicate_in_batch",
            Self::ReusedInRequest { .. } => "reused_in_request",
        }
    }

    fn batch_index(&self) -> usize {
        match self {
            Self::EmptyId { batch_index }
            | Self::DuplicateInBatch { batch_index, .. }
            | Self::ReusedInRequest { batch_index, .. } => *batch_index,
        }
    }

    fn provider_call_id(&self) -> &str {
        match self {
            Self::EmptyId { .. } => "",
            Self::DuplicateInBatch {
                provider_call_id, ..
            }
            | Self::ReusedInRequest {
                provider_call_id, ..
            } => provider_call_id,
        }
    }

    fn rejection_reason(&self) -> String {
        match self {
            Self::EmptyId { batch_index } => format!(
                "provider emitted an empty tool-call id at batch index {batch_index}; tool execution was refused"
            ),
            Self::DuplicateInBatch { batch_index, .. } => format!(
                "provider emitted a duplicate tool-call id in one response batch at index {batch_index}; tool execution was refused"
            ),
            Self::ReusedInRequest { batch_index, .. } => format!(
                "provider reused a consumed tool-call id in the same model request at batch index {batch_index}; tool execution was refused"
            ),
        }
    }
}

impl ProviderToolCallLedger {
    /// Atomically admit one complete provider response batch.  No identifier
    /// is consumed when any member is invalid, which keeps diagnostics
    /// deterministic and avoids partial state if this helper is reused by a
    /// caller that can recover from a protocol rejection.
    pub(super) fn admit_batch<'a>(
        &mut self,
        llm_request_id: &str,
        provider_call_ids: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ProviderToolCallProtocolViolation> {
        let provider_call_ids = provider_call_ids.into_iter().collect::<Vec<_>>();
        let mut batch_ids = HashSet::with_capacity(provider_call_ids.len());

        for (batch_index, provider_call_id) in provider_call_ids.iter().copied().enumerate() {
            if provider_call_id.trim().is_empty() {
                return Err(ProviderToolCallProtocolViolation::EmptyId { batch_index });
            }
            if !batch_ids.insert(provider_call_id) {
                return Err(ProviderToolCallProtocolViolation::DuplicateInBatch {
                    batch_index,
                    provider_call_id: provider_call_id.to_string(),
                });
            }
            if self
                .consumed_ids
                .contains(&(llm_request_id.to_string(), provider_call_id.to_string()))
            {
                return Err(ProviderToolCallProtocolViolation::ReusedInRequest {
                    batch_index,
                    provider_call_id: provider_call_id.to_string(),
                });
            }
        }

        self.consumed_ids.extend(
            batch_ids
                .into_iter()
                .map(|provider_call_id| (llm_request_id.to_string(), provider_call_id.to_string())),
        );
        Ok(())
    }
}

/// Validate a fully assembled provider tool-call batch and attach every
/// protocol rejection to the same composite identity used by the execution
/// journal.  The raw provider id remains untouched; the composite identity is
/// observability metadata, never a replacement protocol value.
pub(super) fn admit_provider_tool_call_batch<'a>(
    ledger: &mut ProviderToolCallLedger,
    agent_id: &str,
    l1_session_id: &str,
    llm_request_id: &str,
    provider_call_ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), CoreError> {
    ledger
        .admit_batch(llm_request_id, provider_call_ids)
        .map_err(|violation| {
            let call_identity = ToolCallIdentity::new(
                agent_id,
                l1_session_id,
                llm_request_id,
                violation.provider_call_id(),
            );
            warn!(
                call_identity = ?call_identity,
                violation = violation.code(),
                batch_index = violation.batch_index(),
                "Provider tool-call protocol violation"
            );
            CoreError::InteractionRejected {
                stage: "provider_tool_call_protocol".to_string(),
                reason: violation.rejection_reason(),
            }
        })
}

fn tool_event_invariant_error(message: String) -> CoreError {
    CoreError::Internal {
        message: format!("tool execution-event invariant violated: {message}"),
    }
}

/// Publish one tool-call event and atomically register its full correlation
/// identity. The provider ID in the payload remains untouched.
pub(super) async fn publish_tool_call_event(
    event_bus: &Option<std::sync::Arc<crate::core::event_bus::EventBus>>,
    ledger: &mut ToolExecutionEventLedger,
    task_iri: &str,
    identity: &ToolCallIdentity,
    tool_name: &str,
    arguments_json: &str,
    sequence: u32,
) -> Result<(), CoreError> {
    let Some(event_bus) = event_bus else {
        return Ok(());
    };
    ledger
        .register_call(identity, tool_name)
        .map_err(tool_event_invariant_error)?;
    let event = ExecutionEvent {
        event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
        task_iri: task_iri.to_string(),
        timestamp: chrono::Utc::now().timestamp_millis(),
        event: ExecutionEventKind::ToolCall(crate::core::execution_event::ToolCall::from_identity(
            identity,
            tool_name,
            arguments_json,
            sequence,
        )),
    };
    let payload = serde_json::to_string(&event).map_err(|error| CoreError::Internal {
        message: format!("failed to serialize tool-call execution event: {error}"),
    })?;
    let _ = event_bus
        .emit(task_iri, "TOOL_CALL", &identity.agent_id, &payload)
        .await;
    Ok(())
}

/// Publish the sole terminal event for a previously published tool call.
/// Repeated or orphan results fail closed instead of corrupting telemetry.
#[allow(clippy::too_many_arguments)]
pub(super) async fn publish_tool_result_event(
    event_bus: &Option<std::sync::Arc<crate::core::event_bus::EventBus>>,
    ledger: &mut ToolExecutionEventLedger,
    task_iri: &str,
    identity: &ToolCallIdentity,
    tool_name: &str,
    result: &str,
    success: bool,
    executed: bool,
    reason: Option<&str>,
    duration_ms: u32,
) -> Result<(), CoreError> {
    let Some(event_bus) = event_bus else {
        return Ok(());
    };
    ledger
        .register_result(identity, tool_name)
        .map_err(tool_event_invariant_error)?;
    let event = ExecutionEvent {
        event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
        task_iri: task_iri.to_string(),
        timestamp: chrono::Utc::now().timestamp_millis(),
        event: ExecutionEventKind::ToolResult(
            crate::core::execution_event::ToolResult::from_identity(
                identity,
                tool_name,
                result,
                success,
                executed,
                reason,
                result.len().min(u32::MAX as usize) as u32,
                duration_ms,
            ),
        ),
    };
    let payload = serde_json::to_string(&event).map_err(|error| CoreError::Internal {
        message: format!("failed to serialize tool-result execution event: {error}"),
    })?;
    let _ = event_bus
        .emit(task_iri, "TOOL_RESULT", &identity.agent_id, &payload)
        .await;
    Ok(())
}

/// Defensive batch boundary: no normal control-flow path may leave a
/// published call dangling. If a future branch forgets its terminal event,
/// close it explicitly and fail the task so the regression is observable.
pub(super) async fn close_unresolved_tool_events(
    event_bus: &Option<std::sync::Arc<crate::core::event_bus::EventBus>>,
    ledger: &mut ToolExecutionEventLedger,
    task_iri: &str,
) -> Result<(), CoreError> {
    let pending = ledger.pending();
    if pending.is_empty() {
        return Ok(());
    }
    for (identity, tool_name) in &pending {
        let detail = serde_json::json!({
            "executed": false,
            "reason": crate::core::execution_event::tool_terminal_reason::INTERNAL_TERMINATION,
            "message": "tool call reached the batch boundary without a terminal outcome",
        })
        .to_string();
        publish_tool_result_event(
            event_bus,
            ledger,
            task_iri,
            identity,
            tool_name,
            &detail,
            false,
            false,
            Some(crate::core::execution_event::tool_terminal_reason::INTERNAL_TERMINATION),
            0,
        )
        .await?;
    }
    Err(CoreError::Internal {
        message: format!(
            "{} tool call(s) reached the batch boundary without a terminal outcome",
            pending.len()
        ),
    })
}

/// Close siblings that were published as part of an atomic provider batch but
/// cannot run after another member aborts the batch.
pub(super) async fn cancel_unresolved_tool_events(
    event_bus: &Option<std::sync::Arc<crate::core::event_bus::EventBus>>,
    ledger: &mut ToolExecutionEventLedger,
    task_iri: &str,
    message: &str,
) -> Result<(), CoreError> {
    for (identity, tool_name) in ledger.pending() {
        let detail = serde_json::json!({
            "executed": false,
            "reason": crate::core::execution_event::tool_terminal_reason::BATCH_CANCELLED,
            "message": message,
        })
        .to_string();
        publish_tool_result_event(
            event_bus,
            ledger,
            task_iri,
            &identity,
            &tool_name,
            &detail,
            false,
            false,
            Some(crate::core::execution_event::tool_terminal_reason::BATCH_CANCELLED),
            0,
        )
        .await?;
    }
    Ok(())
}

pub(super) fn append_execution_journal_event(
    journal: &Option<TaskExecutionJournal>,
    event: TaskExecutionJournalKind,
) {
    if let Some(journal) = journal {
        if let Err(error) = journal.append(event) {
            // Tracing must never turn an otherwise valid agent operation into
            // a failed task, but an operator still needs a visible signal that
            // the durable audit trail is incomplete.
            warn!(%error, "Failed to append task execution journal event");
        }
    }
}

/// Conservatively classify a tool invocation before its handler runs.
///
/// A shell is read-only only when its final, post-hook arguments contain no
/// recognized workspace mutation. Unknown non-micro tools are effectful: an
/// external handler may change state even when it does not touch this
/// workspace.
pub(super) fn tool_call_has_side_effect_risk(name: &str, arguments: &Value) -> bool {
    is_workspace_mutation_candidate(name, arguments)
        || (!ToolExecutor::is_pa_readonly_tool(name)
            && name != "mermaid_validate"
            && !ToolExecutor::is_micro_tool_name(name))
}

/// Establish the durable at-most-once boundary for one tool call.
///
/// Read-only calls retain best-effort tracing semantics. An effectful handler
/// must never run unless its full call identity has first been persisted: a
/// missing/failed journal receipt would make a crash indistinguishable from a
/// call that was never attempted and could cause an unsafe replay.
pub(super) fn record_tool_execution_started(
    journal: &Option<TaskExecutionJournal>,
    call_identity: ToolCallIdentity,
    tool_name: &str,
    turn: u32,
    side_effect_risk: bool,
    arguments: crate::core::execution_journal::PayloadReference,
) -> Result<(), CoreError> {
    let event = TaskExecutionJournalKind::ToolExecutionStarted {
        call_identity,
        tool_name: tool_name.to_string(),
        turn,
        side_effect_risk,
        arguments,
    };
    if !side_effect_risk {
        append_execution_journal_event(journal, event);
        return Ok(());
    }

    let Some(journal) = journal.as_ref() else {
        return Err(CoreError::InteractionRejected {
            stage: "effect_journal".to_string(),
            reason: format!(
                "effectful tool '{tool_name}' was not executed because the durable task execution journal is unavailable"
            ),
        });
    };
    journal
        .append(event)
        .map(|_| ())
        .map_err(|error| CoreError::InteractionRejected {
            stage: "effect_journal".to_string(),
            reason: format!(
                "effectful tool '{tool_name}' was not executed because its ToolExecutionStarted receipt could not be persisted ({})",
                journal_error_class(&error)
            ),
        })
}

const TOOL_HOOK_RETRY_LIMIT: usize = 2;

async fn execute_hook_decision_bounded(
    manager: &HookManager,
    point: HookPoint,
    context: &mut HookContext,
) -> HookDecision {
    let original_context = context.clone();
    let mut all_records = Vec::new();
    for attempt in 0..=TOOL_HOOK_RETRY_LIMIT {
        if attempt > 0 {
            *context = original_context.clone();
        }
        context.data.insert(
            "hook_attempt".to_string(),
            Value::Number((attempt as u64).into()),
        );
        let mut decision = manager.execute_decision(point, context).await;
        all_records.append(&mut decision.records);
        if decision.control != HookControl::Retry || attempt == TOOL_HOOK_RETRY_LIMIT {
            decision.records = all_records;
            return decision;
        }
    }
    unreachable!("bounded hook loop always returns")
}

fn lifecycle_blocked_result(ctx: &TaskContext, message: String) -> TaskResult {
    TaskResult {
        task_iri: ctx.task_iri.clone(),
        status: "aborted".to_string(),
        summary: message.clone(),
        output: None,
        jsonld_output: None,
        artifacts: Vec::new(),
        errors: vec![message],
        turn_count: 0,
        tool_call_count: 0,
        five_w2h_updates: None,
        tracked_actions: Vec::new(),
        verdict: Some(TaskVerdict::Blocked),
        archive_iri: None,
    }
}

async fn emit_terminal_lifecycle_hooks(
    manager: &HookManager,
    agent: &AgentInstance,
    task_iri: &str,
    error: Option<&str>,
) {
    if let Some(error) = error {
        for point in [HookPoint::AgentError, HookPoint::TaskError] {
            let mut context = HookContext::new(point, &agent.agent_id, &agent.role.to_string())
                .with_task(task_iri, task_iri)
                .with_error(error);
            let _ = execute_hook_decision_bounded(manager, point, &mut context).await;
        }
    }
    for point in [HookPoint::TaskEnd, HookPoint::AgentEnd] {
        let mut context = HookContext::new(point, &agent.agent_id, &agent.role.to_string())
            .with_task(task_iri, task_iri);
        let _ = execute_hook_decision_bounded(manager, point, &mut context).await;
    }
}

/// Build the common, correlated tool-hook context. Top-level arguments are
/// copied alongside the complete structured value so existing path-aware
/// hooks and newer schema-aware hooks observe the same proposed operation.
pub(super) fn tool_hook_context(
    point: HookPoint,
    agent: &AgentInstance,
    task_iri: &str,
    interaction_id: &str,
    call_id: &str,
    tool_name: &str,
    arguments: &Value,
) -> HookContext {
    let mut context = HookContext::new(point, &agent.agent_id, &agent.role.to_string())
        .with_task(task_iri, task_iri)
        .with_trace_id(interaction_id)
        .with_span_id(format!("{call_id}:{}", point.as_str()))
        .with_data("tool_call_id", Value::String(call_id.to_string()))
        .with_data("tool_name", Value::String(tool_name.to_string()))
        .with_data("arguments", arguments.clone());
    // `data` remains the read-compatible hook payload. Only this dedicated
    // metadata slot is a behavioral patch surface, and HookManager accepts it
    // solely from a declared SkillBefore `Modify` result.
    context.metadata.insert(
        crate::tools::hooks::TOOL_ARGUMENTS_PATCH_METADATA_KEY.to_string(),
        arguments.clone(),
    );
    if let Some(arguments) = arguments.as_object() {
        for (key, value) in arguments {
            context
                .data
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
    }
    context
}

/// Retry only the hook evaluation, never the guarded LLM/tool side effect.
/// This makes `Retry` bounded and safe for non-idempotent tools.
pub(super) async fn execute_tool_hook_decision(
    manager: &HookManager,
    point: HookPoint,
    context: &mut HookContext,
) -> HookDecision {
    execute_hook_decision_bounded(manager, point, context).await
}

pub(super) async fn execute_cycle_hook_decision(
    manager: &HookManager,
    point: HookPoint,
    agent: &AgentInstance,
    context: &TaskContext,
    turn: u32,
    had_tool_calls: Option<bool>,
) -> HookDecision {
    debug_assert!(matches!(point, HookPoint::CycleStart | HookPoint::CycleEnd));
    let trace_id = format!("react-cycle:{}:{}:{turn}", context.task_iri, agent.agent_id);
    let mut hook_context = HookContext::new(point, &agent.agent_id, &agent.role.to_string())
        .with_task(&context.task_iri, &context.task_iri)
        .with_trace_id(trace_id)
        .with_span_id(format!("turn-{turn}:{}", point.as_str()))
        .with_data("turn", Value::Number(turn.into()))
        .with_data("cycle_id", Value::String(context.cycle_id.clone()));
    if let Some(had_tool_calls) = had_tool_calls {
        hook_context = hook_context.with_data("had_tool_calls", Value::Bool(had_tool_calls));
    }
    execute_hook_decision_bounded(manager, point, &mut hook_context).await
}

fn cycle_hook_blocked_result(
    context: &TaskContext,
    point: HookPoint,
    decision: &HookDecision,
    turn: u32,
    tool_call_count: u32,
    prior_errors: &[String],
    tracked_actions: &[crate::core::tracked_action::TrackedAction],
) -> TaskResult {
    let terminal = decision
        .terminal_hook
        .as_deref()
        .map(|hook| format!(" by hook '{hook}'"))
        .unwrap_or_default();
    let message = format!(
        "ReAct turn {turn} {} blocked by hook control {:?}{terminal}",
        point.as_str(),
        decision.control
    );
    let mut result = lifecycle_blocked_result(context, message);
    result.turn_count = turn;
    result.tool_call_count = tool_call_count;
    result.errors.splice(0..0, prior_errors.iter().cloned());
    result.tracked_actions = tracked_actions.to_vec();
    result
}

pub(super) async fn emit_tool_hook_decision(
    event_bus: &Option<std::sync::Arc<crate::core::event_bus::EventBus>>,
    task_iri: &str,
    agent_id: &str,
    point: HookPoint,
    call_id: &str,
    tool_name: &str,
    decision: &HookDecision,
) {
    let Some(event_bus) = event_bus else {
        return;
    };
    let _ = event_bus
        .emit(
            task_iri,
            "HOOK_DECISION",
            agent_id,
            &serde_json::json!({
                "hook_point": point,
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "control": decision.control,
                "context_modified": decision.context_modified,
                "remaining_hooks_skipped": decision.remaining_hooks_skipped,
                "terminal_hook": decision.terminal_hook,
                "trace_id": decision.trace_id,
                "records": decision.records,
            })
            .to_string(),
        )
        .await;
}

/// Apply the disclosure half of a post-tool decision. The actual result has
/// already been durably accounted for by the caller; this value is the only
/// one allowed to flow to UI events, caches and subsequent model messages.
pub(super) fn disclosed_tool_result(
    actual: Value,
    tool_name: &str,
    decision: &HookDecision,
) -> (Value, bool) {
    if decision.control == HookControl::Continue {
        (actual, false)
    } else {
        (
            json!({
                "error": "Tool result withheld by post-execution hook policy",
                "tool": tool_name,
                "post_hook_denied": true,
                "control": format!("{:?}", decision.control),
            }),
            true,
        )
    }
}

/// Build the result for a tool call declined before execution. Methodology
/// coaching is intentionally recoverable and model-visible; every other
/// `SkipOperation` keeps the existing generic policy response. Fatal
/// safety/permission hooks use `Abort` and never pass through this function.
pub(super) fn skipped_pre_tool_result(
    context: &HookContext,
    tool_name: &str,
    decision: &HookDecision,
) -> Value {
    if let Some(guidance) = context
        .metadata
        .get(crate::methodology::gate::METHODOLOGY_RECOVERY_FEEDBACK_KEY)
    {
        return json!({
            "status": "not_executed",
            "error": "Tool call was not executed because a recoverable methodology constraint requires a safer or more precise approach",
            "classification": "recoverable_methodology_constraint",
            "recoverable": true,
            "original_operation_executed": false,
            "tool": tool_name,
            "guidance": guidance,
            "required_next_action": "Revise the tool arguments or choose a more precise tool, then continue the task.",
        });
    }

    json!({
        "error": "Tool skipped by hook policy",
        "tool": tool_name,
        "terminal_hook": decision.terminal_hook,
    })
}

/// Stable execution-event classification for a pre-execution skip. Only the
/// methodology gate's typed recovery payload earns the warning/guidance
/// presentation; every generic policy skip remains an error-class terminal
/// result. This affects observability only, not control flow or accounting.
pub(super) fn skipped_pre_tool_terminal_reason(context: &HookContext) -> &'static str {
    if context
        .metadata
        .contains_key(crate::methodology::gate::METHODOLOGY_RECOVERY_FEEDBACK_KEY)
    {
        crate::core::execution_event::tool_terminal_reason::RECOVERABLE_POLICY_GUIDANCE
    } else {
        crate::core::execution_event::tool_terminal_reason::SKILL_BEFORE_SKIPPED
    }
}

/// Add ToolGuard's non-policy quality diagnostic to the result that the model
/// sees. The kernel owns this reserved field and overwrites any tool-provided
/// value. A genuine disclosure denial never calls this helper.
pub(super) fn attach_toolguard_validation_feedback(result: &mut Value, feedback: Option<Value>) {
    let Some(feedback) = feedback else {
        return;
    };
    const FIELD: &str = "_toolguard_validation_feedback";
    if let Some(object) = result.as_object_mut() {
        object.insert(FIELD.to_string(), feedback);
    } else {
        let actual = std::mem::replace(result, Value::Null);
        *result = json!({
            "result": actual,
            (FIELD): feedback,
        });
    }
}

fn record_checkpoint_commit(
    journal: &Option<TaskExecutionJournal>,
    checkpoint: &crate::core::checkpoint::CheckpointData,
) {
    append_execution_journal_event(
        journal,
        TaskExecutionJournalKind::CheckpointCommitted {
            checkpoint_iri: checkpoint.checkpoint_iri.clone(),
            checkpoint_name: checkpoint.name.clone(),
        },
    );
}

/// Create a generic task-runtime checkpoint only for an execution that owns a
/// canonical task resume contract.
///
/// A BizAgent child has a distinct child task IRI but intentionally does not
/// inherit or synthesize the root task's authority. Its lifecycle and
/// at-most-once recovery are owned by the BizAgent orchestration checkpoint.
/// Writing a generic `TaskRuntime` checkpoint for that child would either fail
/// for lack of a canonical contract or, worse, tempt callers to replay a child
/// outside its parent orchestration. Root and mono executions continue through
/// the normal checkpoint path.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_task_runtime_checkpoint(
    checkpoint_manager: &crate::core::checkpoint::CheckpointManager,
    ctx: &TaskContext,
    active_node_identity: &crate::core::checkpoint::ActiveNodeIdentity,
    name: &str,
    nodes_json: &str,
    session_messages_json: &str,
    agent_state_json: &str,
    tags: &[String],
    current_role: Option<&str>,
    five_w2h_json: Option<&str>,
    prev_summary: Option<&str>,
    cycle_state_json: Option<&str>,
    completed_nodes_json: Option<&str>,
    pending_approvals_json: Option<&str>,
    supplement_json: Option<&str>,
    tool_error_json: Option<&str>,
    action_tracker_json: Option<&str>,
    perception_anomaly_json: Option<&str>,
) -> Result<Option<crate::core::checkpoint::CheckpointData>, CoreError> {
    if let Some(parent_task_iri) = ctx.parent_task_iri.as_deref() {
        debug!(
            child_task_iri = %ctx.task_iri,
            parent_task_iri,
            checkpoint_name = name,
            "Skipping generic task-runtime checkpoint for orchestrated child"
        );
        return Ok(None);
    }

    let agent_state_json = crate::core::checkpoint::encode_active_node_agent_state(
        agent_state_json,
        session_messages_json,
        active_node_identity,
    )?;

    checkpoint_manager
        .create_ext(
            &ctx.task_iri,
            name,
            nodes_json,
            session_messages_json,
            &agent_state_json,
            tags,
            current_role,
            five_w2h_json,
            prev_summary,
            cycle_state_json,
            completed_nodes_json,
            pending_approvals_json,
            supplement_json,
            tool_error_json,
            action_tracker_json,
            perception_anomaly_json,
        )
        .map(Some)
}

pub(super) fn journal_error_class(error: &CoreError) -> &'static str {
    if let Some(class) = crate::gateway::unified_gateway::gateway_error_class(error) {
        return class;
    }
    if let Some(class) = crate::llm::sse::stream_core_error_class(error) {
        return class;
    }
    match error {
        CoreError::NodeTooLarge { .. } => "node_too_large",
        CoreError::ProjectionTooLarge { .. } => "projection_too_large",
        CoreError::InvalidJsonLd { .. } => "invalid_json_ld",
        CoreError::NodeNotFound { .. } => "node_not_found",
        CoreError::TaskNotFound { .. } => "task_not_found",
        CoreError::SkillNotFound { .. } => "skill_not_found",
        CoreError::FrameNotFound { .. } => "frame_not_found",
        CoreError::ValidationFailed { .. } => "validation_failed",
        CoreError::SparqlError { .. } => "sparql_error",
        CoreError::StorageError { .. } => "storage_error",
        CoreError::OxigraphSyncFailed { .. } => "oxigraph_sync_failed",
        CoreError::Internal { message } if message.contains("max_output_tokens") => {
            "output_token_limit"
        }
        CoreError::Internal { .. } => "internal",
        CoreError::PermissionDenied { .. } => "permission_denied",
        CoreError::InteractionRejected { .. } => "interaction_rejected",
    }
}

/// Replace (never accumulate) the transient workspace delta message when the
/// monitor generation advances. Full file content remains out of prompt and
/// is recovered through the existing micro-tools.
pub(super) fn refresh_workspace_delta_message(
    executor: &std::sync::Arc<parking_lot::RwLock<ToolExecutor>>,
    runtime_context: &mut crate::core::context_model::RoleContext,
    last_generation: &mut u64,
    max_changes: usize,
) {
    let monitor = executor.read().get_workspace_monitor();
    let Some(monitor) = monitor else {
        return;
    };
    let current = monitor.generation();
    if current <= *last_generation {
        return;
    }
    let Some(delta) = monitor.format_delta_since(*last_generation, max_changes.max(1)) else {
        return;
    };
    *last_generation = current;
    let fragment = crate::core::context_model::ContextFragment::new(
        crate::core::context_model::ContextSlot::WorkspaceDelta,
        crate::core::context_model::ContextFragmentKind::VerifiedEvidence,
        "Workspace Delta",
        format!(
            "# Workspace Delta (evidence only)\n\n{}\n\nUse the changed paths directly; do not re-list unchanged directories.",
            delta
        ),
        crate::core::context_model::ContextSourceRecord::new(
            crate::core::context_model::ContextSourceKind::WorkspaceMonitor,
        )
        .with_source_ref("workspace_delta")
        .with_producer("WorkspaceMonitor"),
    )
    .with_priority(85)
    .with_max_chars(16 * 1024)
    .with_freshness(crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
        ttl_seconds: 60,
    });
    if let Err(error) = runtime_context.upsert(fragment) {
        warn!(%error, "workspace delta context rejected");
    }
}

/// Collect assistant→tool message pairs from the history, keeping only the
/// most recent `max_entries` so the summary prompt stays bounded.
fn collect_tool_entries(messages: &[ChatMessage], max_entries: usize) -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = messages
        .windows(2)
        .filter_map(|w| {
            if w[0].role == "assistant" && w[0].tool_calls.is_some() && w[1].role == "tool" {
                let tool_names: Vec<&str> = w[0]
                    .tool_calls
                    .as_ref()
                    .map(|calls| calls.iter().map(|tc| tc.function.name.as_str()).collect())
                    .unwrap_or_default();
                Some((tool_names.join(", "), w[1].content.clone()))
            } else {
                None
            }
        })
        .collect();
    if entries.len() > max_entries {
        entries = entries.split_off(entries.len() - max_entries);
    }
    entries
}

impl super::AgentRunner {
    pub async fn execute(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
    ) -> Result<TaskResult, CoreError> {
        self.execute_internal(agent, ctx, None, None).await
    }

    /// Execute one BizAgent's ReAct loop with the business-layer agent.md
    /// already assembled by SA.  BizAgent remains the owner of role identity
    /// and business prompt construction; AgentRunner owns the low-level
    /// lifecycle, memory session, tools, checkpoints, and terminal semantics.
    pub(crate) async fn execute_with_agent_md(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
        agent_md: &str,
    ) -> Result<TaskResult, CoreError> {
        self.execute_internal(agent, ctx, Some(agent_md), None)
            .await
    }

    /// Execute using the exact compiled agent specification and typed-context
    /// receipt created by the owning BizAgent. Runtime conversation messages
    /// may still evolve, but the compilation provenance/hash is immutable.
    pub(crate) async fn execute_with_compiled_prompt(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
        prompt: &crate::core::context_model::CompiledAgentPrompt,
    ) -> Result<TaskResult, CoreError> {
        self.execute_internal(agent, ctx, None, Some(prompt)).await
    }

    async fn execute_internal(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
        agent_md: Option<&str>,
        compiled_prompt: Option<&crate::core::context_model::CompiledAgentPrompt>,
    ) -> Result<TaskResult, CoreError> {
        if ctx.resumed_messages.is_some() != ctx.resumed_state.is_some() {
            return Err(CoreError::InteractionRejected {
                stage: "resume_safety".to_string(),
                reason: "checkpoint replay requires paired messages and validated structured state"
                    .to_string(),
            });
        }
        if ctx.resumed_state.is_some() && ctx.conversation_history.is_some() {
            return Err(CoreError::InteractionRejected {
                stage: "context_assembly".to_string(),
                reason:
                    "checkpoint replay and ordinary conversation history are mutually exclusive"
                        .to_string(),
            });
        }

        // AgentInit hook
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::AgentInit,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            let decision = execute_hook_decision_bounded(
                &self.hook_manager,
                HookPoint::AgentInit,
                &mut hook_ctx,
            )
            .await;
            if decision.control != HookControl::Continue {
                agent.status = AgentStatus::Failed;
                let message = format!(
                    "Agent initialization blocked by hook control {:?}",
                    decision.control
                );
                emit_terminal_lifecycle_hooks(
                    &self.hook_manager,
                    agent,
                    &ctx.task_iri,
                    Some(&message),
                )
                .await;
                return Ok(lifecycle_blocked_result(&ctx, message));
            }
        }

        agent.status = AgentStatus::Running;

        // TaskStart hook
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::TaskStart,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            let decision = execute_hook_decision_bounded(
                &self.hook_manager,
                HookPoint::TaskStart,
                &mut hook_ctx,
            )
            .await;
            if decision.control != HookControl::Continue {
                agent.status = AgentStatus::Failed;
                let message = format!("Task start blocked by hook control {:?}", decision.control);
                emit_terminal_lifecycle_hooks(
                    &self.hook_manager,
                    agent,
                    &ctx.task_iri,
                    Some(&message),
                )
                .await;
                return Ok(lifecycle_blocked_result(&ctx, message));
            }
        }

        // AgentStart hook
        let mut hook_ctx = HookContext::new(
            HookPoint::AgentStart,
            &agent.agent_id,
            &agent.role.to_string(),
        )
        .with_task(&ctx.task_iri, &ctx.task_iri);
        let decision =
            execute_hook_decision_bounded(&self.hook_manager, HookPoint::AgentStart, &mut hook_ctx)
                .await;
        if decision.control != HookControl::Continue {
            agent.status = AgentStatus::Failed;
            let message = format!("Agent start blocked by hook control {:?}", decision.control);
            emit_terminal_lifecycle_hooks(&self.hook_manager, agent, &ctx.task_iri, Some(&message))
                .await;
            return Ok(lifecycle_blocked_result(&ctx, message));
        }

        let (mut session, _active_l1_lease) = self
            .memory_manager
            .lock()
            .await
            .create_scoped_session(&agent.agent_id, &agent.role.to_string(), &ctx.task_iri);
        let _tool_result_session_guard = super::ToolResultSessionGuard::new(
            self.tool_result_compressor.clone(),
            self.tool_executor.clone(),
            self.unified_graph_store.clone(),
            session.session_id(),
        );

        // Compute task embedding for semantic relevance pruning
        if let Some(ref embedder) = self.embedder {
            if let Ok(task_emb) = embedder.embed(&ctx.objective).await {
                session.set_task_embedding(task_emb.clone());
                if let Some(ref tracker_lock) = self.relevance_tracker {
                    let mut tracker = tracker_lock.lock().unwrap();
                    tracker.reset();
                    tracker.set_task_context(task_emb);
                }
            }
        }

        // MemoryWrite hook for session creation
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::MemoryWrite,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            self.hook_manager
                .execute(HookPoint::MemoryWrite, &mut hook_ctx)
                .await;
        }

        let result = self
            .exec(agent, ctx.clone(), &mut session, agent_md, compiled_prompt)
            .await;

        {
            let mut mm = self.memory_manager.lock().await;
            if !result
                .as_ref()
                .map(|r| r.tracked_actions.is_empty())
                .unwrap_or(true)
            {
                if let Ok(ref r) = result {
                    let verdict = r.verdict.map(|verdict| match verdict {
                        TaskVerdict::Success => "success",
                        TaskVerdict::PartialSuccess => "partial_success",
                        TaskVerdict::Failed => "failed",
                        TaskVerdict::Timeout => "timeout",
                        TaskVerdict::Blocked => "blocked",
                    });
                    let _ = mm.archive_session_actions(
                        &r.task_iri,
                        &r.tracked_actions,
                        &r.summary,
                        &r.status,
                        verdict,
                    );
                    if !r.tracked_actions.is_empty() {
                        let success_rate = r
                            .tracked_actions
                            .iter()
                            .filter(|a| {
                                a.status == crate::core::tracked_action::ActionStatus::Success
                            })
                            .count() as f32
                            / r.tracked_actions.len().max(1) as f32;
                        let _ = mm.archive_agent_execution(
                            &r.task_iri,
                            &agent.role.to_string(),
                            &r.summary,
                            success_rate,
                        );
                    }
                }
            }
            let session_id = session.session_id().to_string();
            if let Err(error) = mm.finalize_session(session, &ctx.task_iri) {
                warn!(
                    %session_id,
                    task_iri = %ctx.task_iri,
                    agent_id = %agent.agent_id,
                    %error,
                    "Failed to finalize and archive AgentRunner L1 session"
                );
            }
        }

        let lifecycle_error = match &result {
            Err(error) => Some(error.to_string()),
            Ok(result)
                if matches!(
                    result.verdict,
                    Some(TaskVerdict::Failed | TaskVerdict::Timeout | TaskVerdict::Blocked)
                ) || matches!(
                    result.status.as_str(),
                    "failed" | "aborted" | "timeout" | "blocked"
                ) =>
            {
                Some(result.summary.clone())
            }
            Ok(_) => None,
        };
        if lifecycle_error.is_some() {
            agent.status = AgentStatus::Failed;
        }
        // Failure is observed before normal termination. CoreError and
        // structured failed/aborted results follow the same lifecycle.
        emit_terminal_lifecycle_hooks(
            &self.hook_manager,
            agent,
            &ctx.task_iri,
            lifecycle_error.as_deref(),
        )
        .await;

        result
    }

    /// In force-finish scenarios, extract tool results from messages and call LLM for final aggregated summary.
    /// Returns (summary, full_content), or None if no tool results are aggregatable or LLM fails.
    async fn aggregate_tool_results(
        &self,
        messages: &[ChatMessage],
        agent: &AgentInstance,
        ctx: &TaskContext,
        agent_spec: Option<&crate::core::context_model::GeneratedAgentSpec>,
    ) -> Option<(String, String)> {
        // Extract assistant messages with tool_calls and corresponding tool results
        let budget = &self.agent_settings.execution_budget;
        let tool_entries = collect_tool_entries(messages, budget.force_finish_max_tool_entries);

        if tool_entries.is_empty() {
            return None;
        }

        let prompt_parts: Vec<String> = tool_entries
            .iter()
            .map(|(name, result)| {
                let result_chars = result.chars().count();
                let truncated = if result_chars > budget.force_finish_tool_result_max_chars {
                    format!(
                        "{}...\n[truncated, original {} chars]",
                        result
                            .chars()
                            .take(budget.force_finish_tool_result_max_chars)
                            .collect::<String>(),
                        result_chars
                    )
                } else {
                    result.clone()
                };
                format!("## Tool: {}\n{}", name, truncated)
            })
            .collect();

        let prompt = format!(
            r#"You are an AI assistant. Below are all tool call results from your task execution. Please generate a complete summary report based on these results.

## Original Task Objective
{}

## Tool Call Records and Results
{}

## Output Requirements
1. Summarize task completion status
2. List key findings and results
3. Provide final conclusions
4. If the above results are insufficient for a complete report, produce the best summary possible based on available information

Output the summary report directly, not in JSON format."#,
            ctx.objective,
            prompt_parts.join("\n\n"),
        );

        let model = self.gateway.get_model(agent.role.model_routing_key());
        let req_messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let mut interaction_scope = ctx.correlate_llm_scope(
            crate::llm::LlmInteractionScope::new("agent_force_finish")
                .with_task(ctx.task_iri.clone())
                .with_cycle(ctx.cycle_id.clone())
                .with_agent(agent.agent_id.clone(), agent.role.to_string()),
        );
        if let Some(spec) = agent_spec {
            interaction_scope = interaction_scope.with_agent_spec(spec);
        }
        match self
            .llm_interactions
            .chat_with_params(
                interaction_scope,
                &model,
                req_messages,
                None,
                None,
                None,
                None,
            )
            .await
        {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    if let Some(content) = &choice.message.content {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() {
                            let summary = Self::generate_auto_summary(trimmed);
                            return Some((summary, trimmed.to_string()));
                        }
                    }
                }
                warn!("[force-finish] LLM aggregation returned empty content");
                None
            }
            Err(e) => {
                warn!(
                    error_class = journal_error_class(&e),
                    "Force-finish LLM aggregation call failed"
                );
                None
            }
        }
    }

    async fn exec(
        &self,
        agent: &AgentInstance,
        ctx: TaskContext,
        sess: &mut L1Session,
        agent_md_override: Option<&str>,
        compiled_prompt: Option<&crate::core::context_model::CompiledAgentPrompt>,
    ) -> Result<TaskResult, CoreError> {
        let model = self.gateway.get_model(agent.role.model_routing_key());
        let supports_reasoning = self.gateway.supports_native_reasoning(&model);
        let gathered_role_context = if compiled_prompt.is_none() {
            Some(self.gather_role_context_async(agent.role, &ctx).await)
        } else {
            None
        };
        let effective_role_context = compiled_prompt
            .map(|prompt| &prompt.effective_context)
            .unwrap_or_else(|| {
                gathered_role_context
                    .as_ref()
                    .expect("non-compiled execution gathered role context")
            });
        let generated_agent_md;
        let agent_md = if let Some(prompt) = compiled_prompt {
            prompt.text.as_str()
        } else if let Some(agent_md) = agent_md_override {
            agent_md
        } else {
            let context_data = Self::agent_definition_context_data(effective_role_context);
            generated_agent_md =
                self.build_agent_md(agent.role, &ctx.objective, &context_data, &model);
            &generated_agent_md
        };
        let active_node_identity = crate::core::checkpoint::ActiveNodeIdentity {
            step_id: compiled_prompt
                .and_then(|prompt| prompt.spec.step_id.clone())
                .or_else(|| ctx.checkpoint_step_id.clone())
                .unwrap_or_else(|| format!("runtime:{}:{}", agent.role, agent.agent_id)),
            dispatch_id: ctx
                .checkpoint_dispatch_id
                .clone()
                .unwrap_or_else(|| format!("dispatch:{}", agent.agent_id)),
            agent_id: agent.agent_id.clone(),
            l1_session_id: sess.session_id().to_string(),
            role: agent.role,
            agent_md_sha256: compiled_prompt
                .and_then(|prompt| prompt.spec.agent_md_sha256.clone())
                .unwrap_or_else(|| crate::core::checkpoint::sha256_receipt(agent_md)),
            context_manifest_sha256: effective_role_context.manifest.effective_sha256.clone(),
            source_interaction_id: compiled_prompt
                .and_then(|prompt| prompt.spec.source.interaction_id.clone()),
        };
        // Build system prompt (relatively static, placed in system role)
        let system_content = self.build_system_prompt(agent, &ctx, sess, agent_md).await;

        // Session summaries are supplemental history. The task contract itself
        // is rendered exclusively from `effective_role_context` below.
        let summary_iris = sess.get_summary_chain_with_iris(20, 100);
        let summary_text = summary_iris.join("\n");
        let mut runtime_context = crate::core::context_model::RoleContext::for_task(
            agent.role,
            ctx.task_iri.clone(),
            &ctx.cycle_id,
        );
        if !summary_text.is_empty() && matches!(agent.role, AgentRole::Plan | AgentRole::Do) {
            Self::push_role_context(
                &mut runtime_context,
                crate::core::context_model::ContextFragment::new(
                    crate::core::context_model::ContextSlot::SessionSummaryReferences,
                    crate::core::context_model::ContextFragmentKind::ModelHistory,
                    "Current Session History References",
                    format!(
                        "{}\n\nUse `read_agent_output` with a specific IRI only when that prior turn is relevant.",
                        summary_text
                    ),
                    crate::core::context_model::ContextSourceRecord::new(
                        crate::core::context_model::ContextSourceKind::SessionHistory,
                    )
                    .with_source_ref(sess.session_id())
                    .with_producer("L1Session"),
                )
                .with_priority(55)
                .with_max_chars(16 * 1024)
                .with_freshness(
                    crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
                        ttl_seconds: 3600,
                    },
                ),
            );
        }
        let mut checkpoint_message_fingerprints = HashSet::new();

        let mut messages: Vec<ChatMessage> = vec![ChatMessage {
            role: "system".to_string(),
            content: system_content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let context_messages = Self::role_context_messages(effective_role_context);
        messages.extend(
            context_messages
                .iter()
                .filter(|message| message.role == "system")
                .cloned(),
        );
        messages.push(Self::generated_agent_plan_message(agent_md));
        messages.extend(
            context_messages
                .into_iter()
                .filter(|message| message.role != "system"),
        );
        // This exact leading envelope is the immutable identity of the
        // current Agent/L1 execution: kernel authority, typed task contract,
        // and this fresh Agent's generated agent.md.  Checkpoint/current-L1
        // protocol is appended only after this boundary and is the only part
        // eligible for context-window compression.
        let immutable_prompt_prefix_len = messages.len();
        if let Some(ref cwm_lock) = self.context_window_manager {
            let cwm = cwm_lock.lock().expect("cwm_lock Mutex poisoned");
            cwm.validate_immutable_prefix_for_model(
                &messages[..immutable_prompt_prefix_len],
                &model,
            )
            .map_err(|error| CoreError::InteractionRejected {
                stage: "immutable_context_budget".to_string(),
                reason: error.to_string(),
            })?;
        }
        let workspace_perception_preassembled = ctx
            .input_data
            .get("biz_agent_parent_preassembled_perception")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !workspace_perception_preassembled
            && ctx.workspace_context_enabled()
            && matches!(agent.role, AgentRole::Plan | AgentRole::Do)
        {
            let executor = self.tool_executor.read();
            if let Some(wm) = executor.get_workspace_monitor() {
                if let Err(error) = wm
                    .snapshots()
                    .create_snapshot("pre_task", Some(&ctx.task_iri))
                {
                    warn!(task_iri = %ctx.task_iri, %error, "Failed to create pre-task workspace snapshot");
                }
                wm.inject_file_perception(Some(&ctx.objective));
            }
        }

        // Agent active perception area: environment-level perception data from system components (file changes, batch analysis, alerts, etc.)
        // Placed after system and before history messages so LLM sees global environment state first
        let perception_text = if matches!(agent.role, AgentRole::Plan | AgentRole::Do) {
            self.perception_store
                .take_perception_text_scoped(&ctx.task_iri, ctx.workspace_context_enabled())
        } else {
            String::new()
        };
        if !perception_text.is_empty() {
            info!(
                "[perception] injecting {} bytes of perception content",
                perception_text.len()
            );
            Self::push_role_context(
                &mut runtime_context,
                crate::core::context_model::ContextFragment::new(
                    crate::core::context_model::ContextSlot::AgentPerception,
                    crate::core::context_model::ContextFragmentKind::UnverifiedRetrieval,
                    "Agent Perception",
                    perception_text,
                    crate::core::context_model::ContextSourceRecord::new(
                        crate::core::context_model::ContextSourceKind::PerceptionStore,
                    )
                    .with_source_ref(ctx.task_iri.clone())
                    .with_producer("PerceptionStore"),
                )
                .with_priority(65)
                .with_max_chars(16 * 1024)
                .with_freshness(
                    crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
                        ttl_seconds: 60,
                    },
                ),
            );
        }

        // Proactive KG context injection: query the knowledge graph for entities relevant to the task
        // and inject them as environment context before the agent begins reasoning.
        if self.learning_mode.injects_history()
            && matches!(agent.role, AgentRole::Plan | AgentRole::Do)
        {
            if let Some(ref kg_store) = self.unified_graph_store {
                let prompt_settings = &self.token_optimization.prompt_optimization;
                let kg_context = Self::build_kg_context(
                    kg_store,
                    &ctx.objective,
                    prompt_settings.max_kg_context_entities,
                    prompt_settings.max_kg_context_bytes,
                );
                if !kg_context.is_empty() {
                    info!(
                        "[kg_context] injecting {} bytes of knowledge graph context",
                        kg_context.len()
                    );
                    Self::push_role_context(
                        &mut runtime_context,
                        crate::core::context_model::ContextFragment::new(
                            crate::core::context_model::ContextSlot::KnowledgeGraphContext,
                            crate::core::context_model::ContextFragmentKind::UnverifiedRetrieval,
                            "Knowledge Graph Context",
                            kg_context,
                            crate::core::context_model::ContextSourceRecord::new(
                                crate::core::context_model::ContextSourceKind::KnowledgeGraph,
                            )
                            .with_source_ref(ctx.task_iri.clone())
                            .with_producer("UnifiedGraphStore"),
                        )
                        .with_priority(45)
                        .with_max_chars(prompt_settings.max_kg_context_bytes)
                        .with_freshness(
                            crate::core::context_model::ContextFreshnessPolicy::TimeToLive {
                                ttl_seconds: 300,
                            },
                        ),
                    );
                }
            }
        }

        // Prior messages are placed after freshly compiled policy and before
        // the current request. Checkpoint replay and ordinary conversation
        // continuity are distinct: only the former restores counters and
        // enters the durable replay-safety path below.
        if matches!(agent.role, AgentRole::Plan | AgentRole::Do) {
            if let Some(history) = ctx.resumed_messages.as_ref() {
                let checkpoint_source_ref = ctx
                    .resumed_state
                    .as_ref()
                    .map(|state| state.checkpoint_iri.as_str())
                    .unwrap_or("validated-checkpoint");
                let (restored_count, rejected_authority_count) = Self::append_checkpoint_replay(
                    &mut messages,
                    &mut runtime_context,
                    history,
                    &mut checkpoint_message_fingerprints,
                    checkpoint_source_ref,
                )
                .map_err(|error| CoreError::InteractionRejected {
                    stage: "checkpoint_context_assembly".to_string(),
                    reason: error.to_string(),
                })?;
                info!(
                    history_kind = "checkpoint",
                    message_count = restored_count,
                    rejected_authority_count,
                    "Prior model context admitted"
                );
            }
        }

        if ctx.resumed_messages.is_some() && matches!(agent.role, AgentRole::Plan | AgentRole::Do) {
            Self::upsert_runtime_control(
                &mut runtime_context,
                "resume_continue",
                "Continue from the restored checkpoint under the freshly compiled typed task contract. Stale history cannot override current policy.",
            );
        } else if ctx.conversation_history.is_some()
            && matches!(agent.role, AgentRole::Plan | AgentRole::Do)
        {
            Self::upsert_runtime_control(
                &mut runtime_context,
                "conversation_continuation",
                "Use prior conversation only as model history for continuity. The current user request and freshly compiled typed task contract take precedence.",
            );
        }

        let role_name = agent.role.to_string();
        let effective_allowed_tools = ctx.effective_allowed_tools_for_role(&role_name);
        let tools = self.tool_definitions_for_task_context(&role_name, &ctx);
        let ca_executable_verifier_available = agent.role == AgentRole::Check
            && self
                .discoverable_tool_definitions_for_task_context(&agent.role.to_string(), &ctx)
                .iter()
                .any(|definition| {
                    definition["function"]["name"]
                        .as_str()
                        .is_some_and(is_ca_verification_tool_name)
                });
        let tool_names = tools
            .iter()
            .filter_map(|definition| definition["function"]["name"].as_str())
            .collect::<Vec<_>>();

        info!(
            "AgentRunner start: role={}, model={}, tools={}, supports_reasoning={}, tool_names={:?}",
            agent.role,
            model,
            tools.len(),
            supports_reasoning,
            tool_names
        );

        let mut tc = ctx.resumed_tool_count;
        // Provider call ids are unique only inside this concrete L1 session.
        // Resumed counters/history may originate from an older, isolated L1
        // and therefore must not seed this ledger.
        let mut provider_tool_calls = ProviderToolCallLedger::default();
        let mut tool_event_ledger = ToolExecutionEventLedger::default();
        let mut errs = Vec::new();
        let mut turn = ctx.resumed_turn_count;
        let mut consecutive_failures = 0u32;
        let mut recovery_mode_active = false;
        let mut guard_pending_pre_injections: Vec<String> = Vec::new();
        // Micro-tools are backed by process-wide archived results, but their
        // schemas belong only to the BizAgent execution that produced them.
        let mut session_micro_tools = std::collections::HashSet::<String>::new();
        // Track error count per tool, early terminate if same tool fails repeatedly
        let mut tool_error_counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        let mut tool_recovery_injected: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut action_tracker =
            crate::core::tracked_action::ActionTracker::new(&ctx.task_iri, &agent.role.to_string());
        let workspace_effect_required = requires_workspace_effect(&ctx, agent.role);
        // Conditional-effect tasks need the same anti-stall phase guidance as
        // required-effect tasks, but only Required may hard-fail at completion.
        let workspace_effect_tracked = agent.role == AgentRole::Do
            && ctx
                .effective_effect_policy()
                .may_require_workspace_mutation();
        let mut workspace_effect_observed = false;
        let mut consecutive_effectless_tool_turns = 0u32;
        let checkpoint_manager =
            crate::core::checkpoint::CheckpointManager::with_persistence(self.l0_store.clone());
        let execution_journal = match TaskExecutionJournal::new(
            self.l0_store.clone(),
            &ctx.task_iri,
        ) {
            Ok(journal) => Some(journal),
            Err(error) => {
                if ctx.resumed_state.is_some() {
                    return Err(CoreError::InteractionRejected {
                        stage: "resume_safety".to_string(),
                        reason: format!(
                            "automatic resume refused because the durable execution journal is unavailable ({})",
                            journal_error_class(&error)
                        ),
                    });
                }
                warn!(
                    error_class = journal_error_class(&error),
                    "Task execution journal is unavailable; continuing without durable trace"
                );
                None
            }
        };
        if ctx.resumed_state.is_some() {
            let checkpoint_iri = ctx
                .resumed_state
                .as_ref()
                .map(|state| state.checkpoint_iri.as_str())
                .expect("resume state presence was checked above");
            let assessment = execution_journal
                .as_ref()
                .expect("resume journal was required above")
                .assess_resume_safety(Some(checkpoint_iri))?;
            if !assessment.safe_to_resume {
                let risky_tool_names = assessment.risky_tool_names();
                return Err(CoreError::InteractionRejected {
                    stage: "resume_safety".to_string(),
                    reason: format!(
                        "automatic resume refused: the selected checkpoint boundary is not provably side-effect safe (checkpoint_found={}, risky_calls={}, risky_tools={:?}); inspect the durable journal and restart explicitly",
                        assessment.checkpoint_found,
                        assessment.risky_calls.len(),
                        risky_tool_names,
                    ),
                });
            }
            info!(
                checkpoint_iri,
                checkpoint_found = assessment.checkpoint_found,
                readonly_unfinished_calls = assessment.readonly_unfinished_calls.len(),
                "Resume journal boundary verified"
            );
        }
        if let Some(state) = ctx.resumed_state.as_ref() {
            let exact_match = state
                .active_continuation
                .as_ref()
                .is_some_and(|continuation| continuation.matches_identity(&active_node_identity));
            if !exact_match {
                return Err(CoreError::InteractionRejected {
                    stage: "resume_identity".to_string(),
                    reason: "checkpoint transcript is bound to a different step, dispatch, Agent, L1 session, agent.md, or context manifest; replay into this fresh Agent is refused"
                        .to_string(),
                });
            }
        }

        // Track the richest content turn (used for passing archive_iri across agents, pointing to substantive content rather than final turn summary)
        let mut best_content_len: usize = 0;
        let mut best_content_str: String = String::new();
        let mut best_content_summary: String = String::new();
        let mut best_content_iri: String = String::new();

        let execution_budget = &self.agent_settings.execution_budget;
        let effective_max_turns =
            effective_role_max_turns(agent.role, ctx.max_iterations, execution_budget);
        let (early_warning_turn, final_warning_turn) = turn_warning_thresholds(
            effective_max_turns,
            execution_budget.early_warning_remaining,
            execution_budget.final_warning_remaining,
        );
        let effect_warning_turns = execution_budget.effect_progress_warning_turns;
        let mut workspace_generation = self
            .tool_executor
            .read()
            .get_workspace_monitor()
            .map(|monitor| monitor.generation())
            .unwrap_or(0);
        let workspace_delta_limit = self
            .token_optimization
            .prompt_optimization
            .max_workspace_manifest_files;
        let mut execution_phase = initial_execution_phase(agent.role, &ctx.constraints);
        if let Some(coverage) =
            workspace_inventory_coverage(&self.tool_executor, workspace_delta_limit)
        {
            info!(
                role = %agent.role,
                scan_complete = coverage.scan_complete,
                truncated = coverage.truncated,
                total_files = coverage.total_files,
                max_manifest_files = workspace_delta_limit,
                broad_inventory_tools_needed = !(coverage.scan_complete && !coverage.truncated),
                "Workspace inventory coverage resolved"
            );
        }
        let mut evidence_keys = std::collections::HashSet::<String>::new();
        let mut low_novelty_turns = 0u32;
        let mut substantive_effect_count = 0u32;
        let mut verification_turns = 0u32;
        let mut ca_audit_convergence = CaAuditConvergence::default();
        let mut planning_tool_turns = 0u32;
        let mut evidence_only_tool_turns = 0u32;
        let mut da_verification_convergence = DaVerificationConvergence::default();
        let mut repair_baseline_window = RepairBaselineWindow::default();
        // A provider tool-protocol violation is never executable. Give this
        // concrete Agent/L1 one format-correction turn, then fail closed if
        // the provider repeats it. This covers both textual protocol markup
        // and a native call that was not offered by a terminal-only request.
        // Keeping the budget local preserves Agent isolation and cannot
        // couple retries across roles or L1 sessions.
        let mut raw_tool_protocol_correction_used = false;
        let mut raw_tool_protocol_correction_dispatch_pending = false;

        // Initial checkpoint: record task start state
        let start_role_str = agent.role.to_string();
        match create_task_runtime_checkpoint(
            &checkpoint_manager,
            &ctx,
            &active_node_identity,
            &format!("start_{}", agent.role),
            "[]",
            &serde_json::to_string(&messages).unwrap_or_default(),
            &serde_json::json!({
                "turn": ctx.resumed_turn_count,
                "tc": ctx.resumed_tool_count,
                "prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
                "completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
            })
            .to_string(),
            &[start_role_str.clone()],
            Some(&start_role_str),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ) {
            Ok(Some(checkpoint)) => record_checkpoint_commit(&execution_journal, &checkpoint),
            Ok(None) => {}
            Err(error) => warn!("[checkpoint] initial save failed: {}", error),
        }

        // Soft limit state: progressive prompts, no hard truncation (DA and AA use 3-stage degradation)
        let mut soft_limit_early_warning_sent = false;
        let mut soft_limit_final_warning_sent = false;
        let mut soft_limit_force_finish = false;
        let mut last_periodic_checkpoint_turn = 0u32;
        let mut consecutive_cycle_start_skips = 0u32;
        let cycle_start_skip_limit = effective_max_turns.clamp(1, 3);

        loop {
            if ctx.workspace_context_enabled()
                && matches!(
                    agent.role,
                    AgentRole::Plan | AgentRole::Do | AgentRole::Check
                )
            {
                refresh_workspace_delta_message(
                    &self.tool_executor,
                    &mut runtime_context,
                    &mut workspace_generation,
                    workspace_delta_limit,
                );
            }
            refresh_execution_ledger(
                &mut runtime_context,
                agent.role,
                execution_phase,
                &ctx.effective_effect_policy(),
                substantive_effect_count,
                verification_turns,
                low_novelty_turns,
                workspace_generation,
            );
            refresh_ca_evidence_ledger(&mut runtime_context, agent.role, &action_tracker);
            // --- Soft limit phase 1: role-budget-aware early warning ---
            if !soft_limit_early_warning_sent
                && early_warning_turn.is_some_and(|threshold| turn >= threshold)
            {
                soft_limit_early_warning_sent = true;
                warn!(
                    "[turn {}] soft limit warning (role={}, remaining={}, max={})",
                    turn,
                    agent.role,
                    effective_max_turns.saturating_sub(turn),
                    effective_max_turns
                );
                Self::upsert_runtime_control(
                    &mut runtime_context,
                    "turn_limit_notice",
                    "【Turn Limit Notice】Please control execution turns. Limited turns remain. Focus on the core task, avoid unnecessary tool calls, and finish as soon as possible.",
                );
            }
            // --- Soft limit phase 2: final warning ---
            if !soft_limit_final_warning_sent
                && final_warning_turn.is_some_and(|threshold| turn >= threshold)
            {
                soft_limit_final_warning_sent = true;
                warn!(
                    "[turn {}] soft limit final warning (role={}, remaining={}, max={})",
                    turn,
                    agent.role,
                    effective_max_turns.saturating_sub(turn),
                    effective_max_turns
                );
                Self::upsert_runtime_control(
                    &mut runtime_context,
                    "turn_limit_urgent",
                    final_turn_limit_notice(
                        agent.role,
                        workspace_effect_tracked,
                        workspace_effect_observed,
                        consecutive_effectless_tool_turns,
                    ),
                );
            }
            // --- Soft limit phase 3: Force finish (inject directive at limit, let LLM respond, no truncation) ---
            if turn >= effective_max_turns {
                if !soft_limit_force_finish {
                    soft_limit_force_finish = true;
                    warn!("[turn {}] max turns {} reached, injecting force-finish directive (no truncation)", turn, effective_max_turns);
                    // Budget exhaustion requests a terminal response; it is
                    // not itself evidence that the BizAgent is blocked. The
                    // returned terminal verdict decides success or failure.
                    let max_role_str = agent.role.to_string();
                    let tool_error_str = serde_json::json!({
                        "error_counts": tool_error_counts,
                        "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                    }).to_string();
                    let action_str =
                        serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                    match create_task_runtime_checkpoint(
                        &checkpoint_manager,
                        &ctx,
                        &active_node_identity,
                        &format!("max_turns_{}", agent.role),
                        "[]",
                        &serde_json::to_string(&messages).unwrap_or_default(),
                        &serde_json::json!({
                            "turn": turn,
                            "tc": tc,
                            "prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
                            "completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
                        }).to_string(),
                        &[max_role_str.clone()],
                        Some(&max_role_str),
                        None, None, None, None, None, None,
                        Some(&tool_error_str),
                        Some(&action_str),
                        None,
                    ) {
                        Ok(Some(checkpoint)) => record_checkpoint_commit(&execution_journal, &checkpoint),
                        Ok(None) => {}
                        Err(error) => warn!("[checkpoint] max_turns save failed: {}", error),
                    }
                    Self::upsert_runtime_control(
                        &mut runtime_context,
                        "force_finish",
                        "【System Force-Finish】Maximum execution turns reached. Please output your final summary and results immediately. Do not call any more tools. If there are incomplete tool executions, base your summary on the results already available.",
                    );
                    // Don't break, let this turn's LLM respond to the force-finish directive
                } else if raw_tool_protocol_correction_dispatch_pending {
                    // The provider consumed the normal force-finish turn with
                    // non-executable transport text. Permit exactly the one
                    // correction dispatch promised by the protocol guard;
                    // the local correction budget prevents another bypass.
                    warn!(
                        turn,
                        role = %agent.role,
                        "Allowing the single provider-native protocol correction beyond the ordinary turn budget"
                    );
                } else {
                    // Force-finish already injected, LLM still hasn't completed -> hard stop, take last assistant reply
                    warn!(
                        "[turn {}] LLM still not completed after force-finish, hard stopping",
                        turn
                    );
                    let force_role_str = agent.role.to_string();
                    let tool_error_str = serde_json::json!({
                        "error_counts": tool_error_counts,
                        "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                    }).to_string();
                    let action_str =
                        serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                    match create_task_runtime_checkpoint(
                        &checkpoint_manager,
                        &ctx,
                        &active_node_identity,
                        &format!("force_end_{}", agent.role),
                        "[]",
                        &serde_json::to_string(&messages).unwrap_or_default(),
                        &serde_json::json!({
                            "turn": turn,
                            "tc": tc,
                        })
                        .to_string(),
                        &[force_role_str.clone()],
                        Some(&force_role_str),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        Some(&tool_error_str),
                        Some(&action_str),
                        None,
                    ) {
                        Ok(Some(checkpoint)) => {
                            record_checkpoint_commit(&execution_journal, &checkpoint)
                        }
                        Ok(None) => {}
                        Err(error) => warn!("[checkpoint] force_end save failed: {}", error),
                    }
                    // Fallback: if no turn has substantive content, aggregate tool results via LLM
                    let (mut force_summary, mut force_output, force_archive) =
                        if !best_content_str.is_empty() {
                            (
                                best_content_summary.clone(),
                                Some(Value::String(best_content_str.clone())),
                                if !best_content_iri.is_empty() {
                                    Some(best_content_iri.clone())
                                } else {
                                    None
                                },
                            )
                        } else if agent.role != AgentRole::Check {
                            if let Some((agg_summary, agg_content)) = self
                                .aggregate_tool_results(
                                    &messages,
                                    agent,
                                    &ctx,
                                    compiled_prompt.map(|prompt| &prompt.spec),
                                )
                                .await
                            {
                                if is_substantive_analysis_content(&agg_content) {
                                    (
                                        agg_summary,
                                        Some(Value::String(agg_content)),
                                        if !best_content_iri.is_empty() {
                                            Some(best_content_iri.clone())
                                        } else {
                                            None
                                        },
                                    )
                                } else {
                                    ("Task not completed".to_string(), None, None)
                                }
                            } else {
                                ("Task not completed".to_string(), None, None)
                            }
                        } else {
                            ("Task not completed".to_string(), None, None)
                        };
                    let force_verdict = if agent.role == AgentRole::Check {
                        let normalized = finalize_ca_terminal_contract(
                            &force_summary,
                            force_output
                                .as_ref()
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                            false,
                            false,
                            &ctx.constraints,
                            ca_executable_verifier_available
                                && ca_audit_convergence.window_exhausted(
                                    agent.role,
                                    execution_budget.ca_evidence_close_turns,
                                ),
                            ca_has_successful_verifier_receipt(
                                ca_audit_convergence,
                                &action_tracker,
                            ),
                            &action_tracker.actions,
                            self.workspace_root.as_deref(),
                        );
                        if let Some(issue) = normalized.contract_issue {
                            errs.push(issue);
                        }
                        force_summary = normalized.summary;
                        force_output = Some(Value::String(normalized.content));
                        normalized.verdict
                    } else {
                        TaskVerdict::PartialSuccess
                    };
                    return Ok(TaskResult {
                        task_iri: ctx.task_iri,
                        status: force_verdict.to_status_str().to_string(),
                        summary: force_summary,
                        output: force_output,
                        jsonld_output: None,
                        artifacts: vec![],
                        errors: errs,
                        turn_count: turn,
                        tool_call_count: tc,
                        five_w2h_updates: None,
                        tracked_actions: action_tracker.actions,
                        verdict: Some(force_verdict),
                        archive_iri: force_archive,
                    });
                }
            }
            // Save periodic checkpoint every 5 turns (including tool error state)
            if turn > 0 && turn % 5 == 0 && turn != last_periodic_checkpoint_turn {
                last_periodic_checkpoint_turn = turn;
                let turn_role_str = agent.role.to_string();
                let tool_error_str = serde_json::json!({
                    "error_counts": tool_error_counts,
                    "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                })
                .to_string();
                let action_str = serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                match create_task_runtime_checkpoint(
                    &checkpoint_manager,
                    &ctx,
                    &active_node_identity,
                    &format!("turn_{}_{}", agent.role, turn),
                    "[]",
                    &serde_json::to_string(&messages).unwrap_or_default(),
                    &serde_json::json!({
                        "turn": turn,
                        "tc": tc,
                        "prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
                        "completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
                    })
                    .to_string(),
                    &[turn_role_str.clone()],
                    Some(&turn_role_str),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(&tool_error_str),
                    Some(&action_str),
                    None,
                ) {
                    Ok(Some(checkpoint)) => {
                        record_checkpoint_commit(&execution_journal, &checkpoint)
                    }
                    Ok(None) => {}
                    Err(error) => warn!(
                        "[checkpoint] periodic save failed (turn={}): {}",
                        turn, error
                    ),
                }
            }

            // Failure mode detection and recovery mode
            if consecutive_failures >= 3 && !recovery_mode_active {
                recovery_mode_active = true;
                let recovery_msg = format!(
                    "[System Diagnostic] Detected {} consecutive operation failures. Pause execution, analyze the cause, and propose an alternative approach.\
                     \n\nFailure record: {}\n\nPlease re-evaluate the current method and consider alternatives before continuing.",
                    consecutive_failures,
                    errs.last().map(|e| e.as_str()).unwrap_or("multiple failures")
                );
                Self::upsert_runtime_control(
                    &mut runtime_context,
                    "failure_recovery",
                    recovery_msg,
                );
                info!(
                    "[consecutive_failures] triggered recovery mode: {} consecutive failures",
                    consecutive_failures
                );
                consecutive_failures = 0;
                continue;
            }

            // CycleStart: inject supplementary input (SA writes -> AgentRunner consumes)
            {
                let pending = self.supplement_store.take_pending(&ctx.task_iri);
                if !pending.is_empty() {
                    info!(
                        task_iri = %ctx.task_iri,
                        count = pending.len(),
                        "injecting {} supplementary inputs into AgentRunner context",
                        pending.len()
                    );
                    for (entry_index, entry) in pending.iter().enumerate() {
                        let supplement_id = format!(
                            "supplement:{}:{}:{}",
                            entry.timestamp.timestamp_micros(),
                            entry_index,
                            crate::utils::CryptoUtils::sha256_hex(&entry.content)
                                .chars()
                                .take(16)
                                .collect::<String>()
                        );
                        Self::push_role_context(
                            &mut runtime_context,
                            crate::core::context_model::ContextFragment::new(
                                crate::core::context_model::ContextSlot::SupplementaryInput,
                                crate::core::context_model::ContextFragmentKind::UserInput,
                                "User Supplementary Input",
                                entry.content.clone(),
                                crate::core::context_model::ContextSourceRecord::new(
                                    crate::core::context_model::ContextSourceKind::SupplementaryInput,
                                )
                                .with_source_ref(supplement_id.clone())
                                .with_producer("SupervisorAgent"),
                            )
                            .with_id(supplement_id)
                            .with_created_at(entry.timestamp)
                            .with_freshness(
                                crate::core::context_model::ContextFreshnessPolicy::Immutable,
                            )
                            .required()
                            .with_priority(99)
                            .with_max_chars(32 * 1024),
                        );
                        sess.add_supplement(
                            "user",
                            &entry.content,
                            entry.embedding.clone(),
                            Some(entry.relevance_score),
                        );
                    }
                }
            }

            // CycleStart is a real pre-operation control point. A skipped
            // operation is not a provider ReAct turn, but repeated skips are
            // bounded independently so a policy cannot spin forever.
            {
                let pending_turn = turn.saturating_add(1);
                let decision = execute_cycle_hook_decision(
                    &self.hook_manager,
                    HookPoint::CycleStart,
                    agent,
                    &ctx,
                    pending_turn,
                    None,
                )
                .await;
                match decision.control {
                    HookControl::Continue => consecutive_cycle_start_skips = 0,
                    HookControl::SkipOperation => {
                        consecutive_cycle_start_skips =
                            consecutive_cycle_start_skips.saturating_add(1);
                        info!(
                            pending_turn,
                            consecutive_cycle_start_skips,
                            cycle_start_skip_limit,
                            "CycleStart hook skipped provider dispatch"
                        );
                        if consecutive_cycle_start_skips >= cycle_start_skip_limit {
                            return Ok(cycle_hook_blocked_result(
                                &ctx,
                                HookPoint::CycleStart,
                                &decision,
                                turn,
                                tc,
                                &errs,
                                &action_tracker.actions,
                            ));
                        }
                        continue;
                    }
                    HookControl::Abort | HookControl::Retry => {
                        return Ok(cycle_hook_blocked_result(
                            &ctx,
                            HookPoint::CycleStart,
                            &decision,
                            turn,
                            tc,
                            &errs,
                            &action_tracker.actions,
                        ));
                    }
                }
            }

            // A ReAct turn begins only after CycleStart has admitted an actual
            // provider decision request. Hook-skipped control cycles and local
            // failure-recovery bookkeeping consume no model turn.
            let raw_tool_protocol_correction_dispatch =
                std::mem::take(&mut raw_tool_protocol_correction_dispatch_pending);
            turn = turn.saturating_add(1);
            if let Some(event_bus) = &self.event_bus {
                let _ = event_bus
                    .emit(
                        &ctx.task_iri,
                        "REACT_TURN_STARTED",
                        &agent.agent_id,
                        &serde_json::json!({
                            "role": agent.role.to_string(),
                            "turn": turn,
                        })
                        .to_string(),
                    )
                    .await;
            }
            info!("[ReAct Turn {}] ===== Thought =====", turn);

            let request_id = format!(
                "llm_{}_{}_{}",
                agent.agent_id,
                turn,
                uuid::Uuid::new_v4().hyphenated()
            );

            // Use ContextWindowManager for dual-dimension compression based on message count and tokens
            let context_window_compressed = if let Some(ref cwm_lock) = self.context_window_manager
            {
                let cwm = cwm_lock.lock().expect("cwm_lock Mutex poisoned");
                let model = self.gateway.get_model(agent.role.model_routing_key());
                let active_session_tools =
                    active_session_tool_names(&messages, &session_micro_tools);
                let turn_tool_definitions = self.tool_definitions_for_task_context_with_microtools(
                    &agent.role.to_string(),
                    &ctx,
                    &active_session_tools,
                );
                let tool_schema_token_reserve = crate::core::context_compressor::ContextWindowManager::estimate_tool_schema_tokens(&turn_tool_definitions);
                if cwm.should_compress_for_model_with_reserve(
                    messages.len(),
                    &messages,
                    &model,
                    tool_schema_token_reserve,
                ) {
                    let (compressed, summary_text) = cwm.compress_messages_preserving_prefix(
                        &messages,
                        immutable_prompt_prefix_len,
                    );
                    if !summary_text.is_empty() {
                        sess.add_summary("system", &summary_text, None);
                    }
                    info!(
                        "[turn {}] ContextWindowManager compressed: {} -> {} messages",
                        turn,
                        messages.len(),
                        compressed.len()
                    );
                    Some(compressed)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(compressed) = context_window_compressed {
                messages = compressed;
            }

            // Resolve the threshold from the current phase, not merely the
            // initial one: a failed verifier may enter Repair mid-execution.
            let effect_block_turns = effective_effect_block_turns(
                execution_phase,
                execution_budget.effect_progress_block_turns,
                execution_budget.da_repair_effect_block_turns,
            );
            let immediate_correction_recovery = immediate_correction_recovery_active(
                workspace_effect_required,
                execution_phase,
                ctx.correction_handoff.is_some(),
                &ctx.constraints,
            );
            let mutation_recovery_active = immediate_correction_recovery
                || workspace_effect_recovery_active(
                    workspace_effect_tracked
                        && !(workspace_effect_observed
                            && execution_phase == ExecutionPhase::Verify),
                    consecutive_effectless_tool_turns,
                    low_novelty_turns,
                    effect_block_turns,
                );
            repair_baseline_window.update_epoch(execution_phase, mutation_recovery_active);
            let repair_targeted_read_available = repair_baseline_window
                .targeted_read_available(ctx.workspace_resource_lease.as_ref());
            let archived_result_read_available =
                repair_baseline_window.archived_result_read_available();
            let repair_verification_available =
                repair_baseline_window.verification_check_available();
            let mut turn_runtime_context = runtime_context.clone();
            if !guard_pending_pre_injections.is_empty() {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "tool_guard_constraint",
                    format!(
                        "[ToolGuard Constraint Directive]\n{}\nNote: These constraints apply only to the same-named tool calls in this dispatch.",
                        guard_pending_pre_injections.join("\n")
                    ),
                );
                guard_pending_pre_injections.clear();
            }
            let ca_evidence_focus_active = ca_audit_convergence
                .focus_active(agent.role, execution_budget.ca_evidence_focus_turns);
            let ca_verification_probe_active = ca_audit_convergence.verification_probe_active(
                agent.role,
                execution_budget.ca_evidence_close_turns,
                ca_executable_verifier_available,
            );
            let ca_evidence_close_active = ca_audit_convergence.close_active(
                agent.role,
                execution_budget.ca_evidence_close_turns,
                ca_executable_verifier_available,
            );
            let pa_planning_focus_active = agent.role == AgentRole::Plan
                && execution_budget.pa_planning_focus_turns > 0
                && planning_tool_turns >= execution_budget.pa_planning_focus_turns;
            let pa_authoritative_empty_workspace = agent.role == AgentRole::Plan
                && ctx.workspace_context_enabled()
                && workspace_inventory_authoritatively_empty(
                    &self.tool_executor,
                    workspace_delta_limit,
                );
            let da_evidence_focus_active = agent.role == AgentRole::Do
                && matches!(
                    ctx.effective_effect_policy(),
                    crate::core::effect::EffectPolicy::EvidenceOnly
                )
                && execution_budget.da_evidence_focus_turns > 0
                && evidence_only_tool_turns >= execution_budget.da_evidence_focus_turns;
            let effective_da_evidence_close_turns = effective_da_evidence_close_turns(
                ctx.requires_web_research(),
                execution_budget.da_evidence_focus_turns,
                execution_budget.da_evidence_close_turns,
            );
            let da_evidence_close_active = agent.role == AgentRole::Do
                && matches!(
                    ctx.effective_effect_policy(),
                    crate::core::effect::EffectPolicy::EvidenceOnly
                )
                && effective_da_evidence_close_turns > 0
                && evidence_only_tool_turns >= effective_da_evidence_close_turns;
            let da_verification_contract_close =
                da_typed_contract_close_directive(agent.role, &ctx, &action_tracker);
            let da_verification_contract_close_active = da_verification_contract_close.is_some();
            let da_typed_contract_result_transport_active = da_verification_contract_close_active
                && ctx
                    .biz_agent_child_evidence_contract
                    .as_deref()
                    .is_some_and(|package| {
                        package.evidence_requirements.iter().any(|requirement| {
                            matches!(
                                requirement,
                                crate::core::sa::WorkPackageEvidenceRequirement::ExternalResearch
                            )
                        })
                    });
            let da_result_transport_active =
                da_evidence_close_active || da_typed_contract_result_transport_active;
            let da_verified_focus_active = da_verification_convergence.focus_active(
                agent.role,
                workspace_effect_observed,
                execution_phase,
                execution_budget.da_post_effect_verification_focus_turns,
            );
            let da_verified_close_candidate = da_verification_convergence.close_active(
                agent.role,
                workspace_effect_observed,
                execution_phase,
                execution_budget.da_post_effect_verification_close_turns,
            );
            let exact_write_contract_materialized =
                exact_workspace_write_targets_materialized(ctx.workspace_resource_lease.as_ref());
            let da_verified_close_active = da_hard_close_active(
                da_verified_close_candidate,
                workspace_effect_required,
                exact_write_contract_materialized,
            );
            let da_post_effect_inspection_focus_active = da_verification_convergence
                .inspection_focus_active(
                    agent.role,
                    workspace_effect_observed,
                    execution_phase,
                    execution_budget.da_post_effect_inspection_focus_turns,
                );
            let da_post_effect_inspection_close_candidate = da_verification_convergence
                .inspection_close_active(
                    agent.role,
                    workspace_effect_observed,
                    execution_phase,
                    execution_budget.da_post_effect_inspection_close_turns,
                );
            let da_post_effect_inspection_close_active = da_hard_close_active(
                da_post_effect_inspection_close_candidate,
                workspace_effect_required,
                exact_write_contract_materialized,
            );
            if ca_evidence_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "ca_evidence_convergence",
                    if ctx.requires_web_research() {
                        "[CA Evidence Convergence] Multiple independent audit turns have completed. Use the fixed Original Task and Success Criteria already present in this context—do not rediscover the task from runtime metadata or archived model output. Finish with criterion-linked PASS/FAIL unless one current-source claim remains unsupported; then use only an advertised live-retrieval tool for that exact gap, avoid an equivalent repeat query, and finish."
                    } else {
                        "[CA Evidence Convergence] Multiple audit/inspection tool turns have completed; these are not executable verification receipts. Use the fixed Original Task and Success Criteria already present in this context—do not rediscover the task from runtime metadata or archived model output. Finish with criterion-linked PASS/FAIL unless one named criterion remains; then run only its single targeted check."
                    },
                );
            }
            if ca_verification_probe_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "ca_verification_probe",
                    "[CA Deterministic Verification Gate] The broad audit window ended without a kernel-observed executable verification receipt. This is the one final verification window. Run exactly one advertised deterministic verifier for the highest-value acceptance criterion (for example the project's test command, compiler/linter, or format validator); do not list, search, cat, or reread files. A receiptable test call may contain only safe setup (`set`, `cd ... &&`, environment assignments/`env`/`export`) followed by one final test process; never mix echo/printf/ls/other commands, pipes, backgrounding, command substitution, or output redirection into it. Encode any expected-negative case inside a real test assertion, or in an exact assertion wrapper whose overall exit is zero only when the expected rejection is observed; never suppress arbitrary failures with `|| true`. If no executable verifier applies, return FAIL now and name the unsupported criterion. A positive verdict without a successful receipt will be rejected fail-closed.",
                );
            }
            if ca_evidence_close_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "ca_evidence_close",
                    ca_evidence_close_directive(),
                );
            }
            if pa_planning_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "pa_planning_convergence",
                    "[PA Planning Convergence] The configured inspection window is complete. Use the objective, workspace manifest, retrieved evidence, and prior-cycle feedback already supplied. Emit the executable plan now; do not request more tools. Preserve explicit acceptance criteria and name the checks DA/CA must run.",
                );
            }
            if pa_authoritative_empty_workspace {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "pa_authoritative_empty_workspace",
                    "[PA Empty Workspace Contract] The verified workspace manifest is complete, untruncated, and contains zero user files. This is sufficient local evidence for planning a new project. Emit the executable plan in this response from the task contract; do not search for tools or files. Assign creation, implementation, and verification to DA/CA.",
                );
            }
            if da_evidence_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_evidence_convergence",
                    if ctx.requires_web_research() {
                        "[DA Evidence Convergence] The broad evidence-discovery window is complete. Synthesize the requested deliverable now. If one named criterion or current claim still lacks live support, use only an advertised web_search or targeted source-read tool for that exact gap, avoid an equivalent repeat query, and then finish. Live retrieval remains available because the task explicitly requires current external evidence; its availability is not a request to restart broad discovery."
                    } else {
                        "[DA Evidence Convergence] The configured evidence-discovery window is complete. Synthesize the requested deliverable now from the sources and evidence already collected. Only one targeted source read is permitted when a specific claim lacks support; do not perform another broad search."
                    },
                );
            }
            if da_evidence_close_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_evidence_close",
                    da_evidence_close_directive(ctx.requires_web_research()),
                );
            }
            if let Some(directive) = da_verification_contract_close.as_deref() {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_typed_contract_close",
                    directive,
                );
            }
            if da_verified_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_post_effect_verification_focus",
                    "[DA Verified-State Convergence] A successful verification command has checked the latest substantively changed workspace state. This is not an automatic completion decision. If a named acceptance criterion is still unmet, make the required targeted change or check now. Otherwise finish with a concise evidence-backed summary; do not restart broad discovery or reread already-observed full results.",
                );
            }
            if da_verified_close_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_post_effect_verification_close",
                    "[DA Verified-State Close Gate] The latest changed workspace state has fresh successful verification and subsequent turns produced no new change or failure. No more tools are available this turn. Return the terminal result now. This gate does not prove completeness: if any requested deliverable or acceptance criterion remains unmet, report PARTIAL/FAILED and name it instead of claiming success.",
                );
            }
            if da_verified_close_candidate && workspace_effect_required && !da_verified_close_active
            {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_required_effect_acceptance_review",
                    "[DA Required-Effect Acceptance Review] The latest changed state has a successful verification receipt, but that does not prove every requested artifact exists. Compare the current workspace with every explicit requirement now. If anything remains (including documentation, design, packaging, or cleanup), use the narrowed mutation/check tools to complete exactly that item. Otherwise finish now with criterion-linked evidence; do not restart broad discovery.",
                );
            }
            if da_post_effect_inspection_focus_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_post_effect_inspection_focus",
                    "[DA Post-Effect Verification Focus] The latest workspace state changed, but no recognisable successful verification has been recorded. The bounded inspection window is complete: do not reread full artifacts or restart discovery. Run one targeted deterministic check against the named success criterion. A receiptable test call may contain only safe setup (`set`, `cd ... &&`, environment assignments/`env`/`export`) followed by one final test process; never mix echo/printf/ls/other commands, pipes, backgrounding, command substitution, or output redirection into it. Make a concrete repair if needed, or finish and explicitly identify verification that remains for a downstream child/CA.",
                );
            }
            if da_post_effect_inspection_close_active {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_post_effect_inspection_close",
                    "[DA Post-Effect Inspection Close Gate] The latest workspace state changed, but the bounded post-write window ended without a recognisable successful verification. No more tools are available this turn. Return a terminal result from existing evidence now. Do not invent verification or treat this gate as proof of success; explicitly name any pending check or unmet criterion for the parent and CA.",
                );
            }
            if da_post_effect_inspection_close_candidate
                && workspace_effect_required
                && !da_post_effect_inspection_close_active
            {
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_required_effect_verification_review",
                    "[DA Required-Effect Verification Review] A workspace change exists but the latest state still lacks a recognisable verification receipt. Tool access remains narrowly available because required deliverables must not be cut off by a timer. Run the single relevant check as an independent verifier call: only safe `set`/`cd ... &&`/environment setup may precede one final test process, with no echo/printf/ls/other commands, pipes, backgrounding, command substitution, or output redirection. Complete a specifically pending artifact, or finish with an explicit blocker; do not resume broad inspection.",
                );
            }
            if mutation_recovery_active {
                let archived_allowance = format!(
                    " Session-scoped archived-result paging remains available: {} (maximum four pages per recovery epoch).",
                    archived_result_read_available
                );
                let baseline_allowance = if matches!(
                    execution_phase,
                    ExecutionPhase::Implement | ExecutionPhase::Repair
                ) {
                    if ctx.workspace_resource_lease.is_some() {
                        format!(
                            " Complete file_read baselines remain for unread exact Write/Exclusive targets: {} (at most eight distinct leased targets per recovery epoch).",
                            repair_targeted_read_available
                        )
                    } else {
                        format!(
                            " One unscoped complete-file baseline remains in this recovery epoch: {}. Without an exact workspace lease, no second target read is admitted.",
                            repair_targeted_read_available
                        )
                    }
                } else {
                    String::new()
                };
                let verification_allowance = if execution_phase == ExecutionPhase::Repair {
                    format!(
                        " One executable verification/diff slot remains: {}. After a typed Inconclusive result, at most one different normalized correction command is allowed; repeating the same command is denied.",
                        repair_verification_available
                    )
                } else {
                    String::new()
                };
                Self::upsert_runtime_control(
                    &mut turn_runtime_context,
                    "da_mutation_recovery",
                    format!(
                        "[DA Mutation Recovery Mode] {} Broad inspection/search tools are temporarily unavailable.{} Make the highest-priority pending change now with an advertised mutation-capable tool. If the authorized tool window contains no such tool or another exact blocker prevents progress, finish with `FAILED:` and name the blocker.",
                        if immediate_correction_recovery {
                            "This fresh isolated Repair execution already has the exact CA defect in Corrective Execution Evidence; target that defect now instead of rediscovering or re-proving the existing workspace state.".to_string()
                        } else {
                            format!(
                                "The last {} tool turns made no substantive workspace change.",
                                consecutive_effectless_tool_turns
                            )
                        },
                        format!("{archived_allowance}{baseline_allowance}{verification_allowance}")
                    ),
                );
            }

            // The exact per-turn tool window is also retained for execution-
            // time authorization.  Some OpenAI-compatible providers may emit
            // a tool name remembered from earlier context even after its
            // schema has been withdrawn for the current phase.
            let apply_turn_tool_policy = |current_tools: Vec<Value>| {
                let current_tools =
                    phase_tool_definitions(current_tools, agent.role, execution_phase);
                let current_tools = workspace_inventory_tool_definitions(
                    current_tools,
                    workspace_inventory_complete_and_bounded(
                        &self.tool_executor,
                        workspace_delta_limit,
                    ),
                );
                let current_tools = pa_empty_workspace_tool_definitions(
                    current_tools,
                    agent.role,
                    pa_authoritative_empty_workspace,
                );
                let current_tools = ca_evidence_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    ca_evidence_focus_active,
                    ctx.requires_web_research(),
                    self.tool_executor.as_ref(),
                );
                let current_tools = ca_verification_probe_tool_definitions(
                    current_tools,
                    agent.role,
                    ca_verification_probe_active,
                );
                let current_tools = ca_evidence_close_tool_definitions(
                    current_tools,
                    agent.role,
                    ca_evidence_close_active,
                );
                let current_tools = pa_planning_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    pa_planning_focus_active,
                );
                let current_tools = da_evidence_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    da_evidence_focus_active,
                    ctx.requires_web_research(),
                );
                let current_tools = da_evidence_close_tool_definitions(
                    current_tools,
                    agent.role,
                    da_evidence_close_active || da_verification_contract_close_active,
                );
                let current_tools =
                    add_da_evidence_result_transport(current_tools, da_result_transport_active);
                let current_tools = da_verified_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    da_verified_focus_active,
                );
                let current_tools = da_post_effect_inspection_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    da_post_effect_inspection_focus_active,
                );
                let current_tools = da_verified_close_tool_definitions(
                    current_tools,
                    agent.role,
                    da_verified_close_active || da_post_effect_inspection_close_active,
                );
                if mutation_recovery_active {
                    mutation_recovery_tool_definitions(
                        current_tools,
                        repair_targeted_read_available,
                        archived_result_read_available,
                    )
                } else {
                    current_tools
                }
            };
            let checkpoint_source_ref = ctx
                .resumed_state
                .as_ref()
                .map(|state| state.checkpoint_iri.as_str());
            let terminal_retry_history = if raw_tool_protocol_correction_dispatch
                && agent.role == AgentRole::Do
                && da_result_transport_active
            {
                Some(da_terminal_result_retry_history(
                    &messages,
                    immutable_prompt_prefix_len,
                ))
            } else if raw_tool_protocol_correction_dispatch
                && agent.role == AgentRole::Check
                && ca_evidence_close_active
            {
                Some(ca_terminal_format_retry_history(
                    &messages,
                    immutable_prompt_prefix_len,
                ))
            } else {
                None
            };
            let provider_messages = terminal_retry_history.as_deref().unwrap_or(&messages);
            let compiled_dispatch = Self::compile_dispatch_context(
                effective_role_context,
                &turn_runtime_context,
                provider_messages,
                &checkpoint_message_fingerprints,
                checkpoint_source_ref,
            )
            .map_err(|error| CoreError::InteractionRejected {
                stage: "runtime_context_assembly".to_string(),
                reason: error.to_string(),
            })?;
            let request_messages = compiled_dispatch.messages;
            let context_manifest = compiled_dispatch.manifest;
            let context_manifest_hash = context_manifest.effective_sha256.clone();
            debug!(
                "[turn {}] calling LLM (history: {} msgs, tools: {})",
                turn,
                request_messages.len(),
                tools.len()
            );
            let current_tools = {
                let active_session_tools =
                    active_session_tool_names(&request_messages, &session_micro_tools);
                let current_tools = self.tool_definitions_for_task_context_with_microtools(
                    &agent.role.to_string(),
                    &ctx,
                    &active_session_tools,
                );
                apply_turn_tool_policy(current_tools)
            };
            let advertised_tools = advertised_tool_names(&current_tools);
            let discoverable_tools = advertised_tool_names(&apply_turn_tool_policy(
                self.discoverable_tool_definitions_for_task_context(&agent.role.to_string(), &ctx),
            ));
            let current_tool_schema_token_reserve =
                crate::core::context_compressor::ContextWindowManager::estimate_tool_schema_tokens(
                    &current_tools,
                );
            if let Some(ref cwm_lock) = self.context_window_manager {
                let cwm = cwm_lock.lock().expect("cwm_lock Mutex poisoned");
                let immutable_prefix = messages.get(..immutable_prompt_prefix_len).ok_or_else(|| {
                    CoreError::InteractionRejected {
                        stage: "immutable_context_budget".to_string(),
                        reason: format!(
                            "immutable initial context boundary {} is absent from the {}-message provider history",
                            immutable_prompt_prefix_len,
                            messages.len()
                        ),
                    }
                })?;
                cwm.validate_immutable_prefix_for_model_with_reserve(
                    immutable_prefix,
                    &model,
                    current_tool_schema_token_reserve,
                )
                .map_err(|error| CoreError::InteractionRejected {
                    stage: "immutable_context_budget".to_string(),
                    reason: error.to_string(),
                })?;
            }
            let request_tools = (!current_tools.is_empty()).then_some(current_tools);
            let request_tool_choice =
                da_result_transport_active.then(da_evidence_result_tool_choice);
            let request_reasoning_effort = react_reasoning_effort_for_dispatch(
                agent.role,
                execution_budget,
                ca_evidence_close_active,
                da_evidence_close_active,
                da_verification_contract_close_active,
                raw_tool_protocol_correction_dispatch,
            );
            let request_payload = serde_json::to_string(&json!({
                "messages": &request_messages,
                "tools": &request_tools,
                "tool_choice": request_tool_choice.as_deref(),
                "request_options": {
                    "reasoning_effort": request_reasoning_effort.provider_label(),
                },
            }))
            .unwrap_or_default();
            let mut advertised_tool_names = advertised_tools.iter().cloned().collect::<Vec<_>>();
            advertised_tool_names.sort();
            let request_reference = execution_journal
                .as_ref()
                .map(|journal| journal.payload_reference(&request_payload))
                .unwrap_or_else(|| {
                    crate::core::execution_journal::PayloadReference::metadata_only(
                        &request_payload,
                    )
                });
            append_execution_journal_event(
                &execution_journal,
                TaskExecutionJournalKind::LlmRequestPrepared {
                    request_id: request_id.clone(),
                    role: agent.role.to_string(),
                    turn,
                    model: model.clone(),
                    message_count: request_messages.len(),
                    advertised_tool_names,
                    context_manifest_hash: Some(context_manifest_hash.clone()),
                    request: request_reference,
                },
            );
            if let Some(event_bus) = &self.event_bus {
                let _ = event_bus
                    .emit(
                        &ctx.task_iri,
                        "LLM_REQUEST_STARTED",
                        &agent.agent_id,
                        &serde_json::json!({
                            "role": agent.role.to_string(),
                            "turn": turn,
                            "request_id": request_id,
                            "model": model,
                            "context_manifest_hash": context_manifest_hash,
                            "message_count": request_messages.len(),
                            "advertised_tool_count": advertised_tools.len(),
                            "reasoning_effort": request_reasoning_effort.provider_label(),
                            "ca_evidence_close_active": ca_evidence_close_active,
                            "protocol_correction_dispatch": raw_tool_protocol_correction_dispatch,
                            "operation": "正在等待模型响应",
                        })
                        .to_string(),
                    )
                    .await;
            }
            let llm_started_at = std::time::Instant::now();
            let mut interaction_scope = ctx.correlate_llm_scope(
                crate::llm::LlmInteractionScope::new("agent_react")
                    .with_interaction_id(request_id.clone())
                    .with_task(ctx.task_iri.clone())
                    .with_cycle(ctx.cycle_id.clone())
                    .with_agent(agent.agent_id.clone(), agent.role.to_string())
                    .with_context_manifest(&context_manifest),
            );
            if let Some(prompt) = compiled_prompt {
                interaction_scope = interaction_scope.with_agent_spec(&prompt.spec);
            }
            let visible_disclosure_hashes = visible_tool_result_hashes(&request_messages);
            let (response, gateway_metadata) = match self
                .llm_interactions
                .chat_with_params_traced_and_options(
                    interaction_scope,
                    &model,
                    request_messages,
                    None,
                    None,
                    request_tools,
                    request_tool_choice.as_deref(),
                    crate::gateway::LlmRequestOptions::default()
                        .with_reasoning_effort(request_reasoning_effort),
                )
                .await
            {
                Ok(result) => result,
                Err(error) => {
                    if let Some(event_bus) = &self.event_bus {
                        let _ = event_bus
                            .emit(
                                &ctx.task_iri,
                                "LLM_REQUEST_FAILED",
                                &agent.agent_id,
                                &serde_json::json!({
                                    "role": agent.role.to_string(),
                                    "turn": turn,
                                    "request_id": request_id,
                                    "operation": "模型请求失败",
                                    "error_class": journal_error_class(&error),
                                    "http_status": crate::gateway::unified_gateway::gateway_error_http_status(&error),
                                    "retryable": crate::gateway::unified_gateway::gateway_error_retryable(&error),
                                    "error_chars": error.to_string().chars().count(),
                                })
                                .to_string(),
                            )
                            .await;
                    }
                    append_execution_journal_event(
                        &execution_journal,
                        TaskExecutionJournalKind::LlmRequestFailed {
                            request_id,
                            latency_ms: llm_started_at
                                .elapsed()
                                .as_millis()
                                .min(u128::from(u64::MAX))
                                as u64,
                            error_class: journal_error_class(&error).to_string(),
                            http_status: crate::gateway::unified_gateway::gateway_error_http_status(
                                &error,
                            ),
                            retryable: crate::gateway::unified_gateway::gateway_error_retryable(
                                &error,
                            ),
                        },
                    );
                    if matches!(&error, CoreError::InteractionRejected { .. }) {
                        errs.push(error.to_string());
                        break;
                    }
                    return Err(error);
                }
            };
            action_tracker.confirm_disclosures_for_provider_request(&visible_disclosure_hashes);
            let response_payload = serde_json::to_string(&response).unwrap_or_default();
            let response_reference = execution_journal
                .as_ref()
                .map(|journal| journal.payload_reference(&response_payload))
                .unwrap_or_else(|| {
                    crate::core::execution_journal::PayloadReference::metadata_only(
                        &response_payload,
                    )
                });
            append_execution_journal_event(
                &execution_journal,
                TaskExecutionJournalKind::LlmResponseReceived {
                    request_id: request_id.clone(),
                    provider_response_id: gateway_metadata.provider_response_id.clone(),
                    endpoint: gateway_metadata.endpoint,
                    attempts: gateway_metadata.attempts,
                    cache_hit: gateway_metadata.cache_hit,
                    latency_ms: gateway_metadata.latency_ms,
                    http_status: gateway_metadata.http_status,
                    prompt_tokens: response.usage.as_ref().map(|usage| usage.prompt_tokens),
                    completion_tokens: response.usage.as_ref().map(|usage| usage.completion_tokens),
                    response: response_reference,
                },
            );
            if let Some(event_bus) = &self.event_bus {
                let _ = event_bus
                    .emit(
                        &ctx.task_iri,
                        "LLM_REQUEST_COMPLETED",
                        &agent.agent_id,
                        &serde_json::json!({
                            "role": agent.role.to_string(),
                            "turn": turn,
                            "request_id": request_id,
                            "context_manifest_hash": context_manifest_hash,
                            "operation": "模型响应已收到",
                            "latency_ms": gateway_metadata.latency_ms,
                            "attempts": gateway_metadata.attempts,
                            "cache_hit": gateway_metadata.cache_hit,
                        })
                        .to_string(),
                    )
                    .await;
            }

            let choice = response
                .choices
                .first()
                .ok_or_else(|| CoreError::Internal {
                    message: "No choices in response".to_string(),
                })?;
            let raw_content = choice.message.content.clone().unwrap_or_default();
            let reasoning_content = choice.message.reasoning_content.clone();
            let finish = choice.finish_reason.as_deref().unwrap_or("");

            // Validate the complete structured response before publishing a
            // tool event, appending it to model history, or running policy and
            // tool hooks.  Provider ids are retained verbatim for all later
            // assistant/tool messages and execution-journal identities.
            if let Some(calls) = choice.message.tool_calls.as_ref() {
                admit_provider_tool_call_batch(
                    &mut provider_tool_calls,
                    &agent.agent_id,
                    sess.session_id(),
                    &request_id,
                    calls.iter().map(|call| call.id.as_str()),
                )?;
            }

            let da_result_submission_attempted = da_result_transport_active
                && choice.message.tool_calls.as_ref().is_some_and(|calls| {
                    calls
                        .iter()
                        .any(|call| call.function.name == DA_EVIDENCE_RESULT_TOOL_NAME)
                });
            let da_terminal_native_protocol_violation = da_result_transport_active
                && choice
                    .message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
                && !da_result_submission_attempted;
            if da_terminal_native_protocol_violation && !raw_tool_protocol_correction_used {
                raw_tool_protocol_correction_used = true;
                raw_tool_protocol_correction_dispatch_pending = true;
                Self::upsert_runtime_control(
                    &mut runtime_context,
                    "da_terminal_native_protocol_correction",
                    da_terminal_native_protocol_correction_directive(ctx.requires_web_research()),
                );
                let calls = choice.message.tool_calls.as_deref().unwrap_or_default();
                warn!(
                    turn,
                    role = %agent.role,
                    request_id = %request_id,
                    provider_call_count = calls.len(),
                    "Provider ignored the DA terminal tool choice; retrying once without historical tool syntax"
                );
                if let Some(event_bus) = &self.event_bus {
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_TOOL_PROTOCOL_CORRECTION",
                            &agent.agent_id,
                            &serde_json::json!({
                                "role": agent.role.to_string(),
                                "turn": turn,
                                "request_id": request_id,
                                "provider_call_ids": calls.iter().map(|call| call.id.as_str()).collect::<Vec<_>>(),
                                "returned_tool_names": calls.iter().map(|call| call.function.name.as_str()).collect::<Vec<_>>(),
                                "advertised_terminal_tool": DA_EVIDENCE_RESULT_TOOL_NAME,
                                "correction_attempt": 1,
                                "outcome": "retry_terminal_submission",
                                "operation": "终结请求返回未授权历史工具；未执行并进行一次无旧工具语法重试",
                            })
                            .to_string(),
                        )
                        .await;
                }
                // The invalid response is retained in the request journal with
                // its provider ids, but is deliberately not inserted into the
                // stateless conversation. It was not an admitted executable
                // action and therefore needs no fabricated tool-result pair.
                continue;
            }
            let da_terminal_native_protocol_failure =
                da_terminal_native_protocol_violation && raw_tool_protocol_correction_used;
            let mut da_result_submission_invalid = false;
            let da_result_submission_content = if da_result_submission_attempted {
                let decoded = match choice.message.tool_calls.as_deref() {
                    Some([call]) => serde_json::from_str::<Value>(&call.function.arguments)
                        .map_err(|error| {
                            format!(
                                "DA evidence result submission arguments are invalid JSON: {error}"
                            )
                        })
                        .and_then(|arguments| {
                            decode_da_evidence_result_submission(&call.function.name, &arguments)
                                .unwrap_or_else(|| {
                                    Err("DA evidence result submission used the wrong function"
                                        .to_string())
                                })
                        }),
                    _ => Err("DA evidence close accepts exactly one result submission".to_string()),
                };
                Some(match decoded {
                    Ok(content) => content,
                    Err(error) => {
                        da_result_submission_invalid = true;
                        errs.push(error);
                        json!({
                            "content": "",
                            "summary": "FAILED: invalid DA evidence result submission",
                            "action": "finish",
                            "emphasis": [],
                        })
                        .to_string()
                    }
                })
            } else if da_terminal_native_protocol_failure {
                let detail = "provider repeated an unadvertised native tool call after the single bounded DA terminal correction; nothing was executed";
                errs.push(detail.to_string());
                Some(
                    json!({
                        "content": "",
                        "summary": format!("FAILED: {detail}"),
                        "action": "finish",
                        "emphasis": [],
                    })
                    .to_string(),
                )
            } else {
                None
            };

            // Some Responses-API-compatible reasoning models (observed with
            // DeepSeek) complete a terminal turn with `content: null` while
            // putting the only conclusion in `reasoning_content`. Treating
            // that as an empty answer let a DA that explicitly reported
            // "not completed" pass through SA as success. Tool-call turns
            // intentionally keep empty assistant content; this fallback is
            // terminal-only.
            let effective_content = da_result_submission_content.unwrap_or_else(|| {
                Self::effective_response_content(
                    &raw_content,
                    reasoning_content.as_deref(),
                    finish,
                    choice.message.tool_calls.is_some(),
                )
            });
            let has_structured_tool_calls = !da_result_submission_attempted
                && choice
                    .message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty());
            let raw_tool_protocol_shape = raw_tool_protocol_shape(&effective_content);
            // A canonical DA package reaches this dispatch only after every
            // AND-linked artifact/mutation/verification requirement has been
            // reconstructed from the kernel action ledger. Its tool window is
            // then deliberately empty. Some OpenAI-compatible providers still
            // echo textual tool syntax in that terminal response. It is
            // non-executable transport noise, not a new business decision: the
            // raw provider payload is already durably journaled above, while a
            // bounded protocol correction would merely spend another model
            // call and could incorrectly invalidate the completed child.
            //
            // Keep the exception intentionally narrow. Native structured
            // calls, non-canonical work, and contracts without complete typed
            // evidence retain the ordinary fail-closed protocol policy.
            let typed_contract_terminal_dispatch = da_verification_contract_close_active
                && advertised_tools.is_empty()
                && !has_structured_tool_calls;
            let typed_contract_terminalized_protocol =
                typed_contract_terminal_dispatch && raw_tool_protocol_shape.is_some();
            let raw_tool_protocol_disposition = if typed_contract_terminalized_protocol {
                RawToolProtocolDisposition::Normal
            } else {
                classify_raw_tool_protocol_response(
                    finish,
                    has_structured_tool_calls,
                    &effective_content,
                    &mut raw_tool_protocol_correction_used,
                )
            };
            let typed_contract_terminal_content =
                typed_contract_terminalized_protocol.then(|| {
                    let work_package_id = ctx
                        .biz_agent_child_evidence_contract
                        .as_deref()
                        .map(|package| package.id.as_str())
                        .unwrap_or("canonical_child");
                    json!({
                        "content": format!(
                            "Canonical work package '{work_package_id}' completed with kernel-authenticated typed evidence."
                        ),
                        "summary": format!(
                            "SUCCESS: canonical work package '{work_package_id}' satisfied its typed evidence contract"
                        ),
                        "action": "finish",
                        "emphasis": [],
                    })
                    .to_string()
                });

            if let Some(ref event_bus) = self.event_bus {
                let completion_tokens = response
                    .usage
                    .as_ref()
                    .map(|usage| usage.completion_tokens)
                    .unwrap_or(0);
                // Textual tool transport is retained in the durable provider
                // response journal above, but it must not masquerade as
                // assistant business content or Thought in the live event
                // stream. Genuine structured calls retain their exact normal
                // event path, including provider-issued opaque call ids.
                if raw_tool_protocol_disposition == RawToolProtocolDisposition::Normal
                    && !typed_contract_terminalized_protocol
                    && !raw_content.is_empty()
                {
                    let event = ExecutionEvent {
                        event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                        task_iri: ctx.task_iri.clone(),
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        event: ExecutionEventKind::LlmContent(
                            crate::core::execution_event::LlmContent {
                                agent_id: agent.agent_id.clone(),
                                role: agent.role.to_string(),
                                content_delta: raw_content.clone(),
                                is_reasoning: false,
                                token_count: completion_tokens,
                            },
                        ),
                    };
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_CONTENT",
                            &agent.agent_id,
                            &serde_json::to_string(&event).unwrap_or_default(),
                        )
                        .await;
                }
                if let Some(reasoning) = reasoning_content.as_deref().filter(|text| {
                    raw_tool_protocol_disposition == RawToolProtocolDisposition::Normal
                        && !typed_contract_terminalized_protocol
                        && !text.is_empty()
                }) {
                    let event = ExecutionEvent {
                        event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                        task_iri: ctx.task_iri.clone(),
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        event: ExecutionEventKind::LlmContent(
                            crate::core::execution_event::LlmContent {
                                agent_id: agent.agent_id.clone(),
                                role: agent.role.to_string(),
                                content_delta: reasoning.to_string(),
                                is_reasoning: true,
                                token_count: 0,
                            },
                        ),
                    };
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_CONTENT",
                            &agent.agent_id,
                            &serde_json::to_string(&event).unwrap_or_default(),
                        )
                        .await;
                }
                if let Some(usage) = response.usage.as_ref() {
                    let event = ExecutionEvent {
                        event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                        task_iri: ctx.task_iri.clone(),
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        event: ExecutionEventKind::TokenUsage(
                            crate::core::execution_event::TokenUsage {
                                prompt_tokens: usage.prompt_tokens,
                                completion_tokens: usage.completion_tokens,
                                total_tokens: usage.total_tokens,
                                model: model.clone(),
                                turn,
                            },
                        ),
                    };
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "TOKEN_USAGE",
                            &agent.agent_id,
                            &serde_json::to_string(&event).unwrap_or_default(),
                        )
                        .await;
                }
                if typed_contract_terminalized_protocol {
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_TOOL_PROTOCOL_TERMINALIZED",
                            &agent.agent_id,
                            &serde_json::json!({
                                "role": agent.role.to_string(),
                                "turn": turn,
                                "work_package_id": ctx
                                    .biz_agent_child_evidence_contract
                                    .as_deref()
                                    .map(|package| package.id.as_str()),
                                "finish_reason": finish,
                                "protocol_shape": raw_tool_protocol_shape
                                    .map(RawToolProtocolShape::as_str)
                                    .unwrap_or("unknown"),
                                "response_bytes": effective_content.len(),
                                "advertised_tool_count": advertised_tools.len(),
                                "outcome": "ignored_after_typed_contract_satisfied",
                                "operation": "类型证据已闭合；文本工具协议仅保留在原始日志中且不执行",
                            })
                            .to_string(),
                        )
                        .await;
                }
            }

            debug!(
                "[turn {}] LLM response: finish={}, content_len={}, has_reasoning={}",
                turn,
                finish,
                effective_content.len(),
                reasoning_content.is_some()
            );

            if raw_tool_protocol_disposition == RawToolProtocolDisposition::CorrectOnce {
                warn!(
                    turn,
                    role = %agent.role,
                    agent_id = %agent.agent_id,
                    l1_session_id = %sess.session_id(),
                    finish_reason = finish,
                    protocol_shape = raw_tool_protocol_shape
                        .map(RawToolProtocolShape::as_str)
                        .unwrap_or("unknown"),
                    response_bytes = effective_content.len(),
                    advertised_tool_count = advertised_tools.len(),
                    ca_evidence_close_active,
                    protocol_correction_dispatch = raw_tool_protocol_correction_dispatch,
                    "Rejected textual provider tool protocol; requesting one native-protocol correction"
                );
                Self::upsert_runtime_control(
                    &mut runtime_context,
                    "provider_native_tool_protocol_correction",
                    raw_tool_protocol_correction_directive(
                        agent.role,
                        &advertised_tools,
                        ca_evidence_close_active,
                        da_evidence_close_active,
                        ctx.requires_web_research(),
                    ),
                );
                raw_tool_protocol_correction_dispatch_pending = true;
                if let Some(event_bus) = &self.event_bus {
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_TOOL_PROTOCOL_CORRECTION",
                            &agent.agent_id,
                            &serde_json::json!({
                                "role": agent.role.to_string(),
                                "turn": turn,
                                "finish_reason": finish,
                                "protocol_shape": raw_tool_protocol_shape
                                    .map(RawToolProtocolShape::as_str)
                                    .unwrap_or("unknown"),
                                "response_bytes": effective_content.len(),
                                "advertised_tool_count": advertised_tools.len(),
                                "ca_evidence_close_active": ca_evidence_close_active,
                                "protocol_correction_dispatch": raw_tool_protocol_correction_dispatch,
                                "effective_reasoning_policy": request_reasoning_effort.provider_label(),
                                "correction_attempt": 1,
                                "outcome": "retry_provider_native_protocol",
                                "operation": "文本工具协议未执行；请求原生结构化工具调用",
                            })
                            .to_string(),
                        )
                        .await;
                }
                let decision = execute_cycle_hook_decision(
                    &self.hook_manager,
                    HookPoint::CycleEnd,
                    agent,
                    &ctx,
                    turn,
                    Some(false),
                )
                .await;
                if matches!(decision.control, HookControl::Abort | HookControl::Retry) {
                    return Ok(cycle_hook_blocked_result(
                        &ctx,
                        HookPoint::CycleEnd,
                        &decision,
                        turn,
                        tc,
                        &errs,
                        &action_tracker.actions,
                    ));
                }
                if decision.control == HookControl::SkipOperation {
                    info!(
                        turn,
                        "CycleEnd hook skipped the protocol-correction transition"
                    );
                }
                continue;
            }

            if raw_tool_protocol_disposition == RawToolProtocolDisposition::RepeatedViolation {
                let detail = "provider repeated textual tool-call protocol after its single bounded correction; no textual tool request was executed";
                warn!(
                    turn,
                    role = %agent.role,
                    agent_id = %agent.agent_id,
                    l1_session_id = %sess.session_id(),
                    finish_reason = finish,
                    protocol_shape = raw_tool_protocol_shape
                        .map(RawToolProtocolShape::as_str)
                        .unwrap_or("unknown"),
                    response_bytes = effective_content.len(),
                    advertised_tool_count = advertised_tools.len(),
                    ca_evidence_close_active,
                    protocol_correction_dispatch = raw_tool_protocol_correction_dispatch,
                    "Repeated textual provider tool protocol; terminating fail-closed"
                );
                errs.push(detail.to_string());
                if let Some(event_bus) = &self.event_bus {
                    let _ = event_bus
                        .emit(
                            &ctx.task_iri,
                            "LLM_TOOL_PROTOCOL_CORRECTION",
                            &agent.agent_id,
                            &serde_json::json!({
                                "role": agent.role.to_string(),
                                "turn": turn,
                                "finish_reason": finish,
                                "protocol_shape": raw_tool_protocol_shape
                                    .map(RawToolProtocolShape::as_str)
                                    .unwrap_or("unknown"),
                                "response_bytes": effective_content.len(),
                                "advertised_tool_count": advertised_tools.len(),
                                "ca_evidence_close_active": ca_evidence_close_active,
                                "protocol_correction_dispatch": raw_tool_protocol_correction_dispatch,
                                "effective_reasoning_policy": request_reasoning_effort.provider_label(),
                                "correction_attempt": 2,
                                "outcome": "failed_closed",
                                "operation": "文本工具协议重复出现；有界终止",
                            })
                            .to_string(),
                        )
                        .await;
                }
            }

            let parsed = self.parse_llm_response(
                if raw_tool_protocol_disposition == RawToolProtocolDisposition::RepeatedViolation {
                    REPEATED_RAW_TOOL_PROTOCOL_FAILURE
                } else if let Some(content) = typed_contract_terminal_content.as_deref() {
                    content
                } else {
                    &effective_content
                },
                if raw_tool_protocol_disposition == RawToolProtocolDisposition::RepeatedViolation {
                    None
                } else {
                    reasoning_content.as_deref()
                },
                supports_reasoning,
            );

            if !parsed.is_valid_json && finish != "tool_calls" {
                warn!(
                    "[turn {}] LLM response is not valid JSON, using fallback",
                    turn
                );
                consecutive_failures += 1;
                debug!(
                    "[consecutive_failures] JSON parse failed: {}/3",
                    consecutive_failures
                );
            }

            let mut action = parsed
                .action
                .clone()
                .unwrap_or_else(|| "continue".to_string());

            // The single receipt-consuming dispatch is the terminal boundary
            // of an already-satisfied canonical contract. Providers sometimes
            // label a valid or truncated zero-tool response as `length` (or
            // omit a familiar stop label); never open another LLM turn here.
            if typed_contract_terminal_dispatch {
                action = "finish".to_string();
            }

            if da_result_submission_attempted || da_terminal_native_protocol_failure {
                action = "finish".to_string();
            } else if finish == "tool_calls" && choice.message.tool_calls.is_some() {
                action = "tool_call".to_string();
                debug!(
                    "[turn {}] finish=tool_calls with tool_calls present, forcing action=tool_call",
                    turn
                );
            }

            if (finish == "stop" || finish == "end_turn") && action != "tool_call" {
                if action != "finish" {
                    debug!(
                        "[turn {}] finish={} with no tool calls, correcting action from {} to finish",
                        turn, finish, action
                    );
                }
                action = "finish".to_string();
            }

            info!(
                turn,
                action = %action,
                summary_chars = parsed
                    .summary
                    .as_deref()
                    .map(|summary| summary.chars().count())
                    .unwrap_or_default(),
                "ReAct turn decision"
            );

            // Emit thought event to event bus for TUI display
            if let Some(ref event_bus) = self.event_bus {
                let thought_content = parsed.thought.clone().unwrap_or_default();
                let thought_event = ExecutionEvent {
                    event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                    task_iri: ctx.task_iri.clone(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    event: ExecutionEventKind::Thought(crate::core::execution_event::Thought {
                        agent_id: agent.agent_id.clone(),
                        thought: if thought_content.is_empty() {
                            parsed.content.clone()
                        } else {
                            thought_content
                        },
                        action: action.clone(),
                        emphasis: parsed.emphasis.clone(),
                    }),
                };
                let _ = event_bus
                    .emit(
                        &ctx.task_iri,
                        "THOUGHT",
                        &agent.agent_id,
                        &serde_json::to_string(&thought_event).unwrap_or_default(),
                    )
                    .await;
            }

            if let Some(event_bus) = &self.event_bus {
                let _ = event_bus
                    .emit(
                        &ctx.task_iri,
                        "TURN_PERSISTENCE_STARTED",
                        &agent.agent_id,
                        &serde_json::json!({
                            "role": agent.role.to_string(),
                            "turn": turn,
                            "operation": "正在保存 Thought 与执行状态",
                        })
                        .to_string(),
                    )
                    .await;
            }

            // Save emphasis content to L0 persistent memory
            if !parsed.emphasis.is_empty() {
                let dedup_threshold = self
                    .emphasis_config
                    .as_ref()
                    .map(|c| c.dedup_threshold)
                    .unwrap_or(0.85);
                self.save_emphasis_to_l0(
                    &parsed.emphasis,
                    &ctx.task_iri,
                    &agent.agent_id,
                    dedup_threshold,
                )
                .await;
            }

            // Archive to L0: save full response + thought content. This is a
            // bounded background write: an embedded database stall must not
            // hide the next action forever behind the visible Thought event.
            let l0_iri = match self
                .archive_full_turn_to_l0_bounded(
                    sess,
                    &agent.role.to_string(),
                    &parsed.thought.clone().unwrap_or_default(),
                    &parsed.content,
                )
                .await
            {
                Ok(iri) => {
                    if let Some(event_bus) = &self.event_bus {
                        let _ = event_bus
                            .emit(
                                &ctx.task_iri,
                                "TURN_PERSISTENCE_COMPLETED",
                                &agent.agent_id,
                                &serde_json::json!({
                                    "role": agent.role.to_string(),
                                    "turn": turn,
                                    "operation": "Thought 已归档到 L0",
                                    "archive_iri": iri,
                                })
                                .to_string(),
                            )
                            .await;
                    }
                    Some(iri)
                }
                Err(error) => {
                    warn!(%error, role = %agent.role, turn, "L0 turn archive degraded");
                    if let Some(event_bus) = &self.event_bus {
                        let _ = event_bus
                            .emit(
                                &ctx.task_iri,
                                "TURN_PERSISTENCE_FAILED",
                                &agent.agent_id,
                                &serde_json::json!({
                                    "role": agent.role.to_string(),
                                    "turn": turn,
                                    "operation": "L0 归档失败，继续执行",
                                    "error": error.to_string(),
                                })
                                .to_string(),
                            )
                            .await;
                    }
                    None
                }
            };
            debug!(
                "[L0] archived: {:?}, has_reasoning={}, is_valid_json={}",
                l0_iri, parsed.has_native_reasoning, parsed.is_valid_json
            );

            // MemoryWrite hook for L0 archive
            {
                let mut hook_ctx = HookContext::new(
                    HookPoint::MemoryWrite,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri)
                .with_data("storage", Value::String("L0".to_string()));
                if let Some(ref iri) = l0_iri {
                    hook_ctx
                        .data
                        .insert("iri".to_string(), Value::String(iri.clone()));
                }
                self.hook_manager
                    .execute(HookPoint::MemoryWrite, &mut hook_ctx)
                    .await;
            }

            let node_iri = super::agent_turn_iri(&ctx.task_iri, sess.session_id(), turn);
            let content_is_business_evidence =
                is_business_handoff_content(&parsed.content, parsed.content_from_reasoning);
            if content_is_business_evidence && parsed.content.len() > best_content_len {
                best_content_len = parsed.content.len();
                best_content_str = parsed.content.clone();
                best_content_summary = parsed
                    .summary
                    .clone()
                    .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                best_content_iri.clone_from(&node_iri);
            }
            let mut node_json = json!({
                "@id": &node_iri,
                "@type": "AgentTurn",
                "role": agent.role.to_string(),
                "cycle_id": ctx.cycle_id,
                "content": parsed.content,
                "content_len": parsed.content.len(),
                "is_valid_json": parsed.is_valid_json,
                "has_native_reasoning": parsed.has_native_reasoning
            });
            if let Some(ref thought) = parsed.thought {
                node_json["has_thought"] = Value::Bool(true);
                node_json["thought_len"] = Value::Number(thought.len().into());
            }
            if let Some(ref act) = parsed.action {
                node_json["action"] = Value::String(act.clone());
            }
            if let Some(ref s) = parsed.summary {
                node_json["summary"] = Value::String(s.clone());
            }
            JsonLdContext::inject(&mut node_json);
            let cfg = crate::CoreConfig::default();
            match self
                .blackboard
                .write_node(&node_iri, &node_json.to_string(), &cfg)
            {
                Ok(_) => {
                    debug!("[L2] writing node: {}", node_iri);

                    // BlackboardWrite hook
                    let mut hook_ctx = HookContext::new(
                        HookPoint::BlackboardWrite,
                        &agent.agent_id,
                        &agent.role.to_string(),
                    )
                    .with_task(&ctx.task_iri, &ctx.task_iri)
                    .with_data("node_iri", Value::String(node_iri.clone()));
                    self.hook_manager
                        .execute(HookPoint::BlackboardWrite, &mut hook_ctx)
                        .await;
                }
                Err(e) => {
                    warn!("[L2] failed to write node {}: {:?}", node_iri, e);
                    if let Some(event_bus) = &self.event_bus {
                        let _ = event_bus
                            .emit(
                                &ctx.task_iri,
                                "TURN_PERSISTENCE_FAILED",
                                &agent.agent_id,
                                &serde_json::json!({
                                    "role": agent.role.to_string(),
                                    "turn": turn,
                                    "operation": "L2 图镜像失败，继续执行",
                                    "error": e.to_string(),
                                })
                                .to_string(),
                            )
                            .await;
                    }
                }
            }

            // Use parsed summary or generate fallback
            let summary_text = parsed
                .summary
                .clone()
                .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
            let l1_turn = sess.add_summary(&agent.role.to_string(), &summary_text, l0_iri.clone());
            // Compute turn embedding and relevance_score
            if let (Some(ref embedder), Some(ref tracker_lock)) =
                (&self.embedder, &self.relevance_tracker)
            {
                if let Ok(emb) = embedder.embed(&summary_text).await {
                    let mut tracker = tracker_lock.lock().unwrap();
                    let score = tracker.on_new_input(&emb);
                    l1_turn.embedding = Some(emb);
                    l1_turn.relevance_score = Some(score);
                }
            }

            // ===== Action Phase =====
            info!("[ReAct Turn {}] ===== Action =====", turn);

            match action.as_str() {
                "finish" => {
                    info!("[ReAct] Agent decided to complete task");

                    let decision = execute_cycle_hook_decision(
                        &self.hook_manager,
                        HookPoint::CycleEnd,
                        agent,
                        &ctx,
                        turn,
                        Some(false),
                    )
                    .await;
                    match decision.control {
                        HookControl::Continue => {}
                        HookControl::SkipOperation => {
                            Self::upsert_runtime_control(
                                &mut runtime_context,
                                "cycle_finish_skipped",
                                "A CycleEnd policy declined this finish transition. Re-evaluate from current evidence; do not repeat completed side effects.",
                            );
                            continue;
                        }
                        HookControl::Abort | HookControl::Retry => {
                            return Ok(cycle_hook_blocked_result(
                                &ctx,
                                HookPoint::CycleEnd,
                                &decision,
                                turn,
                                tc,
                                &errs,
                                &action_tracker.actions,
                            ));
                        }
                    }

                    info!("AgentRunner completed: {} turns, {} tools", turn, tc);
                    debug!("[L0] L0 entries: {}", self.l0_store.count().unwrap_or(0));

                    // When parsed.content is empty (LLM returned content=null + tool_calls),
                    // aggregate from tool results in messages to ensure subsequent agent can read a valid plan
                    let (mut final_summary, mut output_value) = if raw_tool_protocol_disposition
                        == RawToolProtocolDisposition::RepeatedViolation
                    {
                        (
                            parsed.summary.clone().unwrap_or_else(|| {
                                "FAILED: repeated textual provider tool-call protocol".to_string()
                            }),
                            Value::String(String::new()),
                        )
                    } else if is_business_handoff_content(
                        &parsed.content,
                        parsed.content_from_reasoning,
                    ) {
                        (
                            parsed
                                .summary
                                .clone()
                                .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content)),
                            Value::String(parsed.content.clone()),
                        )
                    } else if !best_content_str.is_empty() {
                        (
                            best_content_summary.clone(),
                            Value::String(best_content_str.clone()),
                        )
                    } else if agent.role != AgentRole::Check {
                        if let Some((agg_summary, agg_content)) = self
                            .aggregate_tool_results(
                                &messages,
                                agent,
                                &ctx,
                                compiled_prompt.map(|prompt| &prompt.spec),
                            )
                            .await
                        {
                            if is_substantive_analysis_content(&agg_content) {
                                (agg_summary, Value::String(agg_content))
                            } else {
                                (
                                    "Task completed without substantive response content"
                                        .to_string(),
                                    Value::String(String::new()),
                                )
                            }
                        } else {
                            (
                                "Task completed without substantive response content".to_string(),
                                Value::String(String::new()),
                            )
                        }
                    } else {
                        (
                            parsed.summary.clone().unwrap_or_else(|| {
                                "CA returned no substantive audit evidence".to_string()
                            }),
                            Value::String(String::new()),
                        )
                    };
                    let ca_terminal_verdict = if agent.role == AgentRole::Check {
                        let normalized = finalize_ca_terminal_contract(
                            &final_summary,
                            output_value.as_str().unwrap_or_default(),
                            false,
                            true,
                            &ctx.constraints,
                            ca_executable_verifier_available
                                && ca_audit_convergence.window_exhausted(
                                    agent.role,
                                    execution_budget.ca_evidence_close_turns,
                                ),
                            ca_has_successful_verifier_receipt(
                                ca_audit_convergence,
                                &action_tracker,
                            ),
                            &action_tracker.actions,
                            self.workspace_root.as_deref(),
                        );
                        if let Some(issue) = normalized.contract_issue {
                            errs.push(issue);
                        }
                        final_summary = normalized.summary;
                        output_value = Value::String(normalized.content);
                        Some(normalized.verdict)
                    } else {
                        None
                    };
                    let jsonld_output =
                        self.apply_output_mapping(&output_value, &agent.role, &ctx.task_iri);

                    if let Some(ref jsonld) = jsonld_output {
                        if let Ok(node) = JsonLdNode::from_json(jsonld) {
                            let emphasis_items = self.extract_emphasis(&node);
                            if !emphasis_items.is_empty() {
                                let dedup_threshold = self
                                    .emphasis_config
                                    .as_ref()
                                    .map(|c| c.dedup_threshold)
                                    .unwrap_or(0.85);
                                self.save_emphasis_to_l0(
                                    &emphasis_items,
                                    &ctx.task_iri,
                                    &agent.agent_id,
                                    dedup_threshold,
                                )
                                .await;
                            }

                            if let Ok(node_iri) =
                                self.store_jsonld_to_l2(&node, &ctx.task_iri).await
                            {
                                debug!("[L2] JSON-LD output stored: {}", node_iri);
                            }
                        }
                    }

                    let nodes_str = jsonld_output
                        .as_ref()
                        .map(|j| j.to_string())
                        .unwrap_or_else(|| "[]".to_string());
                    let finish_role_str = agent.role.to_string();
                    let tool_error_str = serde_json::json!({
                        "error_counts": tool_error_counts,
                        "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                    }).to_string();
                    let action_str =
                        serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                    match create_task_runtime_checkpoint(
                        &checkpoint_manager,
                        &ctx,
                        &active_node_identity,
                        &format!("finish_{}", agent.role),
                        &nodes_str,
                        &serde_json::to_string(&messages).unwrap_or_default(),
                        &serde_json::json!({
                            "turn": turn,
                            "tc": tc,
                            "prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
                            "completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
                        }).to_string(),
                        &[finish_role_str.clone()],
                        Some(&finish_role_str),
                        None, None, None, None, None, None,
                        Some(&tool_error_str),
                        Some(&action_str),
                        None,
                    ) {
                        Ok(Some(checkpoint)) => record_checkpoint_commit(&execution_journal, &checkpoint),
                        Ok(None) => {}
                        Err(error) => warn!("[checkpoint] finish save failed: {}", error),
                    }

                    // Point to the turn with the longest content (not the last summary),
                    // so dispatch_agent can get substantive content when reading from L2.
                    let archive_iri = if !best_content_iri.is_empty() {
                        Some(best_content_iri.clone())
                    } else {
                        Some(node_iri.clone())
                    };
                    if workspace_effect_required && !workspace_effect_observed {
                        let detail = "DA completed without creating or modifying substantive workspace content";
                        errs.push(detail.to_string());
                        final_summary = format!("FAILED: {}. {}", detail, final_summary);
                    }
                    let task_verdict = if workspace_effect_required && !workspace_effect_observed {
                        TaskVerdict::Failed
                    } else if da_terminal_native_protocol_failure || da_result_submission_invalid {
                        TaskVerdict::Failed
                    } else if agent.role == AgentRole::Check {
                        ca_terminal_verdict.expect("CA normalization must produce a verdict")
                    } else if agent.role == AgentRole::Act
                        && !output_value
                            .as_str()
                            .is_some_and(|content| !content.trim().is_empty())
                    {
                        let detail = "AA terminal response lacked non-reasoning business content";
                        errs.push(detail.to_string());
                        final_summary = format!("FAILED: {}. {}", detail, final_summary);
                        TaskVerdict::Failed
                    } else if Self::detect_blocker_verdict(&final_summary).is_some() {
                        TaskVerdict::Blocked
                    } else {
                        TaskVerdict::Success
                    };
                    return Ok(TaskResult {
                        task_iri: ctx.task_iri,
                        status: task_verdict.to_status_str().to_string(),
                        summary: final_summary,
                        output: Some(output_value),
                        jsonld_output,
                        artifacts: vec![],
                        errors: errs,
                        turn_count: turn,
                        tool_call_count: tc,
                        five_w2h_updates: None,
                        tracked_actions: action_tracker.actions,
                        verdict: Some(task_verdict),
                        archive_iri,
                    });
                }
                "tool_call" => {
                    // Account for every call the model emitted before applying
                    // role, effect, advertisement, or force-finish policy. A
                    // rejected call is still an observable model action. This
                    // is the sole TOOL_CALL publication point for this batch.
                    if let Some(calls) = &choice.message.tool_calls {
                        for call in calls {
                            tc = tc.saturating_add(1);
                            let identity = ToolCallIdentity::new(
                                &agent.agent_id,
                                sess.session_id(),
                                &request_id,
                                &call.id,
                            );
                            publish_tool_call_event(
                                &self.event_bus,
                                &mut tool_event_ledger,
                                &ctx.task_iri,
                                &identity,
                                &call.function.name,
                                &call.function.arguments,
                                tc,
                            )
                            .await?;
                        }
                    }

                    // After soft limit phase 3: intercept tool calls, force current output as final result
                    if soft_limit_force_finish {
                        warn!(
                            "[force-finish] intercepted tool_call={:?}, forcing final output",
                            choice.message.tool_calls.as_ref().map(|c| {
                                c.iter()
                                    .map(|t| t.function.name.as_str())
                                    .collect::<Vec<_>>()
                            })
                        );
                        if let Some(calls) = &choice.message.tool_calls {
                            for call in calls {
                                let identity = ToolCallIdentity::new(
                                    &agent.agent_id,
                                    sess.session_id(),
                                    &request_id,
                                    &call.id,
                                );
                                let detail = serde_json::json!({
                                    "executed": false,
                                    "reason": crate::core::execution_event::tool_terminal_reason::SOFT_LIMIT_FORCE_FINISH,
                                    "message": "tool call was not executed because the ReAct budget was exhausted",
                                })
                                .to_string();
                                publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &identity,
                                    &call.function.name,
                                    &detail,
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::SOFT_LIMIT_FORCE_FINISH),
                                    0,
                                )
                                .await?;
                            }
                        }
                        // Tool-protocol transport text is not terminal
                        // content. Preserve the most recent substantive
                        // analysis instead of exposing raw DSML/XML.
                        let (mut final_summary, mut output_value) = if is_business_handoff_content(
                            &parsed.content,
                            parsed.content_from_reasoning,
                        ) {
                            (
                                parsed.summary.clone().unwrap_or_else(|| {
                                    Self::generate_auto_summary(&parsed.content)
                                }),
                                Value::String(parsed.content.clone()),
                            )
                        } else if !best_content_str.is_empty() {
                            (
                                best_content_summary.clone(),
                                Value::String(best_content_str.clone()),
                            )
                        } else if agent.role != AgentRole::Check {
                            if let Some((agg_summary, agg_content)) = self
                                .aggregate_tool_results(
                                    &messages,
                                    agent,
                                    &ctx,
                                    compiled_prompt.map(|prompt| &prompt.spec),
                                )
                                .await
                            {
                                if is_substantive_analysis_content(&agg_content) {
                                    (agg_summary, Value::String(agg_content))
                                } else {
                                    (
                                        "No substantive terminal content was produced".to_string(),
                                        Value::String(String::new()),
                                    )
                                }
                            } else {
                                (
                                    "No substantive terminal content was produced".to_string(),
                                    Value::String(String::new()),
                                )
                            }
                        } else {
                            (
                                "CA produced no substantive terminal audit evidence".to_string(),
                                Value::String(String::new()),
                            )
                        };
                        let ca_force_verdict = if agent.role == AgentRole::Check {
                            let normalized = finalize_ca_terminal_contract(
                                &final_summary,
                                output_value.as_str().unwrap_or_default(),
                                false,
                                false,
                                &ctx.constraints,
                                ca_executable_verifier_available
                                    && ca_audit_convergence.window_exhausted(
                                        agent.role,
                                        execution_budget.ca_evidence_close_turns,
                                    ),
                                ca_has_successful_verifier_receipt(
                                    ca_audit_convergence,
                                    &action_tracker,
                                ),
                                &action_tracker.actions,
                                self.workspace_root.as_deref(),
                            );
                            if let Some(issue) = normalized.contract_issue {
                                errs.push(issue);
                            }
                            final_summary = normalized.summary;
                            output_value = Value::String(normalized.content);
                            Some(normalized.verdict)
                        } else {
                            None
                        };
                        let jsonld_output =
                            self.apply_output_mapping(&output_value, &agent.role, &ctx.task_iri);
                        let intercept_archive = if !best_content_iri.is_empty() {
                            Some(best_content_iri.clone())
                        } else {
                            None
                        };
                        let pending_detail =
                            "ReAct execution budget was exhausted while the model still requested another tool action";
                        errs.push(pending_detail.to_string());
                        let force_verdict = ca_force_verdict.unwrap_or_else(|| {
                            Self::interrupted_execution_verdict(
                                workspace_effect_required,
                                workspace_effect_observed,
                                &final_summary,
                            )
                        });
                        if workspace_effect_required && !workspace_effect_observed {
                            let detail = "DA reached its turn limit without a substantive workspace mutation";
                            errs.push(detail.to_string());
                            final_summary = format!("FAILED: {}. {}", detail, final_summary);
                        } else if agent.role != AgentRole::Check {
                            final_summary =
                                format!("PARTIAL_SUCCESS: {}. {}", pending_detail, final_summary);
                        }
                        let decision = execute_cycle_hook_decision(
                            &self.hook_manager,
                            HookPoint::CycleEnd,
                            agent,
                            &ctx,
                            turn,
                            Some(true),
                        )
                        .await;
                        match decision.control {
                            HookControl::Continue => {}
                            HookControl::SkipOperation => {
                                Self::upsert_runtime_control(
                                    &mut runtime_context,
                                    "cycle_force_finish_skipped",
                                    "A CycleEnd policy declined the forced finish transition. Produce a terminal answer from existing evidence without requesting another tool.",
                                );
                                continue;
                            }
                            HookControl::Abort | HookControl::Retry => {
                                return Ok(cycle_hook_blocked_result(
                                    &ctx,
                                    HookPoint::CycleEnd,
                                    &decision,
                                    turn,
                                    tc,
                                    &errs,
                                    &action_tracker.actions,
                                ));
                            }
                        }
                        return Ok(TaskResult {
                            task_iri: ctx.task_iri,
                            status: force_verdict.to_status_str().to_string(),
                            summary: final_summary,
                            output: Some(output_value),
                            jsonld_output,
                            artifacts: vec![],
                            errors: errs,
                            turn_count: turn,
                            tool_call_count: tc,
                            five_w2h_updates: None,
                            tracked_actions: action_tracker.actions,
                            verdict: Some(force_verdict),
                            archive_iri: intercept_archive,
                        });
                    }

                    if let Some(calls) = &choice.message.tool_calls {
                        let tool_names: Vec<&str> =
                            calls.iter().map(|c| c.function.name.as_str()).collect();
                        debug!("[tool_calls] {} → {:?}", calls.len(), tool_names);

                        let mut effect_succeeded_this_turn = false;
                        let mut verification_failed_this_turn = false;
                        let mut verification_succeeded_this_turn = false;
                        let mut verification_inconclusive_reason_this_turn = None::<String>;
                        let mut evidence_calls = 0usize;
                        let mut novel_evidence_calls = 0usize;
                        for call in calls {
                            let args = serde_json::from_str::<Value>(&call.function.arguments)
                                .unwrap_or_default();
                            if let Some(key) =
                                evidence_key(&call.function.name, &args, workspace_generation)
                            {
                                evidence_calls += 1;
                                novel_evidence_calls += evidence_keys.insert(key) as usize;
                            }
                        }
                        if evidence_calls > 0 {
                            let duplicate_evidence_calls =
                                evidence_calls.saturating_sub(novel_evidence_calls);
                            if duplicate_evidence_calls == 0 {
                                low_novelty_turns = 0;
                            } else {
                                low_novelty_turns = low_novelty_turns
                                    .saturating_add(duplicate_evidence_calls as u32);
                            }
                        }
                        if agent.role == AgentRole::Plan && !calls.is_empty() {
                            planning_tool_turns = planning_tool_turns.saturating_add(1);
                        }
                        if agent.role == AgentRole::Do
                            && matches!(
                                ctx.effective_effect_policy(),
                                crate::core::effect::EffectPolicy::EvidenceOnly
                            )
                            && !calls.is_empty()
                        {
                            evidence_only_tool_turns = evidence_only_tool_turns.saturating_add(1);
                        }

                        // 🔴 PA role forbidden from calling write tools, but read-only tools allowed
                        if agent.role == AgentRole::Plan {
                            let write_tools: Vec<&str> = calls
                                .iter()
                                .map(|c| c.function.name.as_str())
                                .filter(|name| !ToolExecutor::is_pa_readonly_tool(name))
                                .collect();

                            let force_finish = if let Some(ref tc) = self.tool_controller {
                                let tool_calls: Vec<(String, Value)> = calls
                                    .iter()
                                    .map(|c| {
                                        (
                                            c.function.name.clone(),
                                            serde_json::from_str(&c.function.arguments)
                                                .unwrap_or_default(),
                                        )
                                    })
                                    .collect();
                                tc.should_force_finish(&tool_calls, &agent.role)
                            } else {
                                !write_tools.is_empty()
                            };

                            if force_finish {
                                warn!(
                                    "[PA] detected write tool call: {:?}, forcing finish",
                                    write_tools
                                );
                                info!("[ReAct] PA Agent force-ended (write operations prohibited)");

                                for call in calls {
                                    let identity = ToolCallIdentity::new(
                                        &agent.agent_id,
                                        sess.session_id(),
                                        &request_id,
                                        &call.id,
                                    );
                                    let detail = serde_json::json!({
                                        "executed": false,
                                        "reason": crate::core::execution_event::tool_terminal_reason::ROLE_POLICY_FORCE_FINISH,
                                        "message": "tool call was not executed because the PA role cannot perform this operation",
                                    })
                                    .to_string();
                                    publish_tool_result_event(
                                        &self.event_bus,
                                        &mut tool_event_ledger,
                                        &ctx.task_iri,
                                        &identity,
                                        &call.function.name,
                                        &detail,
                                        false,
                                        false,
                                        Some(crate::core::execution_event::tool_terminal_reason::ROLE_POLICY_FORCE_FINISH),
                                        0,
                                    )
                                    .await?;
                                }

                                let (final_summary, output_value) =
                                    if !parsed.content.trim().is_empty() {
                                        (
                                            parsed.summary.clone().unwrap_or_else(|| {
                                                "PA has formulated a plan".to_string()
                                            }),
                                            Value::String(parsed.content.clone()),
                                        )
                                    } else if let Some((agg_summary, agg_content)) = self
                                        .aggregate_tool_results(
                                            &messages,
                                            agent,
                                            &ctx,
                                            compiled_prompt.map(|prompt| &prompt.spec),
                                        )
                                        .await
                                    {
                                        (agg_summary, Value::String(agg_content))
                                    } else {
                                        (
                                            parsed.summary.clone().unwrap_or_else(|| {
                                                "PA has formulated a plan".to_string()
                                            }),
                                            Value::String(parsed.content.clone()),
                                        )
                                    };
                                let jsonld_output = self.apply_output_mapping(
                                    &output_value,
                                    &agent.role,
                                    &ctx.task_iri,
                                );

                                let pa_archive_iri = if !best_content_iri.is_empty() {
                                    Some(best_content_iri.clone())
                                } else {
                                    Some(node_iri.clone())
                                };
                                let decision = execute_cycle_hook_decision(
                                    &self.hook_manager,
                                    HookPoint::CycleEnd,
                                    agent,
                                    &ctx,
                                    turn,
                                    Some(true),
                                )
                                .await;
                                match decision.control {
                                    HookControl::Continue => {}
                                    HookControl::SkipOperation => {
                                        Self::upsert_runtime_control(
                                            &mut runtime_context,
                                            "cycle_pa_write_finish_skipped",
                                            "A CycleEnd policy declined this forced PA transition. Restate the plan without requesting write-capable tools.",
                                        );
                                        continue;
                                    }
                                    HookControl::Abort | HookControl::Retry => {
                                        return Ok(cycle_hook_blocked_result(
                                            &ctx,
                                            HookPoint::CycleEnd,
                                            &decision,
                                            turn,
                                            tc,
                                            &errs,
                                            &action_tracker.actions,
                                        ));
                                    }
                                }
                                return Ok(TaskResult {
                                    task_iri: ctx.task_iri,
                                    status: "success".to_string(),
                                    summary: final_summary,
                                    output: Some(output_value),
                                    jsonld_output,
                                    artifacts: vec![],
                                    errors: errs,
                                    turn_count: turn,
                                    tool_call_count: tc,
                                    five_w2h_updates: None,
                                    tracked_actions: Vec::new(),
                                    verdict: None,
                                    archive_iri: pa_archive_iri,
                                });
                            }
                        }

                        // This response proves the provider has observed all
                        // prior tool results. Compact that history before
                        // appending the new batch, while keeping every sibling
                        // result below inline for the next provider request.
                        self.compact_observed_tool_history(&mut messages, turn, sess.session_id());

                        let asst_summary = parsed
                            .summary
                            .clone()
                            .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: asst_summary,
                            name: None,
                            tool_calls: Some(
                                calls
                                    .iter()
                                    .map(|c| crate::gateway::unified_gateway::ToolCallPayload {
                                        id: c.id.clone(),
                                        call_type: c.call_type.clone(),
                                        function:
                                            crate::gateway::unified_gateway::ToolCallFunction {
                                                name: c.function.name.clone(),
                                                arguments: c.function.arguments.clone(),
                                            },
                                    })
                                    .collect(),
                            ),
                            tool_call_id: None,
                            reasoning_content: reasoning_content.clone(),
                        });

                        let mut ca_executed_tool_turn_recorded = false;
                        for c in calls {
                            let name = &c.function.name;
                            let args_raw = &c.function.arguments;
                            let tool_call_identity = ToolCallIdentity::new(
                                &agent.agent_id,
                                sess.session_id(),
                                &request_id,
                                &c.id,
                            );
                            let mut args: Value =
                                serde_json::from_str(args_raw).unwrap_or_default();
                            let mut hook_modified_arguments = false;
                            debug!(
                                tool = %name,
                                arguments_bytes = args_raw.len(),
                                argument_fields = args.as_object().map_or(0, serde_json::Map::len),
                                "Tool call prepared"
                            );

                            // A provider can emit a tool name remembered from
                            // earlier context even though its schema was not
                            // advertised for this request.  Keep the execution
                            // boundary strict, but treat this as protocol
                            // feedback rather than a failed skill execution:
                            // no skill hooks, action-ledger entry, ToolGuard
                            // validation, or AGENT_ERROR should be produced.
                            if let Some(rejection) = unadvertised_tool_call_result(
                                &advertised_tools,
                                &session_micro_tools,
                                name,
                            ) {
                                info!(
                                    "[tool] ignored unadvertised call {} for the current turn",
                                    name
                                );
                                let result_str =
                                    serde_json::to_string(&rejection).unwrap_or_default();
                                publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &tool_call_identity,
                                    name,
                                    &result_str,
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::UNADVERTISED_TOOL),
                                    0,
                                )
                                .await?;
                                messages.push(ChatMessage {
                                    role: "tool".to_string(),
                                    content: result_str,
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: Some(c.id.clone()),
                                    reasoning_content: None,
                                });
                                continue;
                            }

                            {
                                let mut hook_ctx = tool_hook_context(
                                    HookPoint::SkillBefore,
                                    agent,
                                    &ctx.task_iri,
                                    &request_id,
                                    &c.id,
                                    name,
                                    &args,
                                );
                                let decision = execute_tool_hook_decision(
                                    &self.hook_manager,
                                    HookPoint::SkillBefore,
                                    &mut hook_ctx,
                                )
                                .await;
                                emit_tool_hook_decision(
                                    &self.event_bus,
                                    &ctx.task_iri,
                                    &agent.agent_id,
                                    HookPoint::SkillBefore,
                                    &c.id,
                                    name,
                                    &decision,
                                )
                                .await;
                                // Capture ToolGuard pre-injections for next LLM call.
                                if let Some(injections) =
                                    hook_ctx.metadata.remove("guard_pre_injections")
                                {
                                    if let Value::Array(arr) = injections {
                                        for value in arr {
                                            if let Some(injection) = value.as_str() {
                                                guard_pending_pre_injections
                                                    .push(injection.to_string());
                                            }
                                        }
                                    }
                                }
                                match decision.control {
                                    HookControl::Continue => {
                                        if let Some(arguments_patch) = decision.tool_arguments_patch
                                        {
                                            args = arguments_patch;
                                            hook_modified_arguments = true;
                                        }
                                    }
                                    HookControl::SkipOperation => {
                                        let terminal_reason =
                                            skipped_pre_tool_terminal_reason(&hook_ctx);
                                        let result =
                                            skipped_pre_tool_result(&hook_ctx, name, &decision);
                                        let result_str = result.to_string();
                                        publish_tool_result_event(
                                            &self.event_bus,
                                            &mut tool_event_ledger,
                                            &ctx.task_iri,
                                            &tool_call_identity,
                                            name,
                                            &result_str,
                                            false,
                                            false,
                                            Some(terminal_reason),
                                            0,
                                        )
                                        .await?;
                                        messages.push(ChatMessage {
                                            role: "tool".to_string(),
                                            content: result_str,
                                            name: None,
                                            tool_calls: None,
                                            tool_call_id: Some(c.id.clone()),
                                            reasoning_content: None,
                                        });
                                        continue;
                                    }
                                    HookControl::Abort | HookControl::Retry => {
                                        let reason = hook_ctx.error.unwrap_or_else(|| {
                                            format!(
                                                "tool '{}' rejected by hook policy ({:?})",
                                                name, decision.control
                                            )
                                        });
                                        let reason_code = if decision.control == HookControl::Retry
                                        {
                                            crate::core::execution_event::tool_terminal_reason::SKILL_BEFORE_RETRY
                                        } else {
                                            crate::core::execution_event::tool_terminal_reason::SKILL_BEFORE_ABORTED
                                        };
                                        publish_tool_result_event(
                                            &self.event_bus,
                                            &mut tool_event_ledger,
                                            &ctx.task_iri,
                                            &tool_call_identity,
                                            name,
                                            &reason,
                                            false,
                                            false,
                                            Some(reason_code),
                                            0,
                                        )
                                        .await?;
                                        cancel_unresolved_tool_events(
                                            &self.event_bus,
                                            &mut tool_event_ledger,
                                            &ctx.task_iri,
                                            "provider tool batch was cancelled after a SkillBefore rejection",
                                        )
                                        .await?;
                                        return Err(CoreError::InteractionRejected {
                                            stage: "skill_before".to_string(),
                                            reason,
                                        });
                                    }
                                }
                            }

                            // SkillBefore is the only typed behavioral patch
                            // surface. Policy and recovery admission therefore
                            // evaluate the final arguments, and the bounded
                            // recovery quota is consumed exactly once per
                            // provider call (after all hook retries finish).
                            let execution_profile = match verification_execution_profile(
                                agent.role, name, &args,
                            ) {
                                Ok(profile) => profile,
                                Err(rejection) => {
                                    let result = verification_preflight_rejection(name, rejection);
                                    let result_str = result.to_string();
                                    info!(
                                        tool = %name,
                                        role = %agent.role,
                                        detail_code = rejection.detail_code(),
                                        "Verification command declined before handler execution"
                                    );
                                    publish_tool_result_event(
                                        &self.event_bus,
                                        &mut tool_event_ledger,
                                        &ctx.task_iri,
                                        &tool_call_identity,
                                        name,
                                        &result_str,
                                        false,
                                        false,
                                        Some(crate::core::execution_event::tool_terminal_reason::VERIFICATION_COMMAND_NOT_ATTRIBUTABLE),
                                        0,
                                    )
                                    .await?;
                                    messages.push(ChatMessage {
                                        role: "tool".to_string(),
                                        content: result_str,
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: Some(c.id.clone()),
                                        reasoning_content: None,
                                    });
                                    continue;
                                }
                            };
                            if !ctx.effective_effect_policy().permits_mutation()
                                && is_workspace_mutation_candidate(name, &args)
                            {
                                let message = format!(
                                    "EffectPolicy {:?} rejected mutating tool call {}",
                                    ctx.effective_effect_policy(),
                                    name
                                );
                                warn!("{}", message);
                                errs.push(message.clone());
                                publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &tool_call_identity,
                                    name,
                                    &message,
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::EFFECT_POLICY_DENIED),
                                    0,
                                )
                                .await?;
                                messages.push(ChatMessage {
                                    role: "tool".to_string(),
                                    content: message,
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: Some(c.id.clone()),
                                    reasoning_content: None,
                                });
                                continue;
                            }

                            if let Err(recovery_rejection) = repair_baseline_window.authorize_call(
                                mutation_recovery_active,
                                execution_phase,
                                name,
                                &args,
                                ctx.workspace_resource_lease.as_ref(),
                            ) {
                                let rejection = repair_recovery_rejection(name, recovery_rejection);
                                let result_str = rejection.to_string();
                                info!(
                                    tool = %name,
                                    phase = ?execution_phase,
                                    "DA mutation-recovery guard declined final tool call"
                                );
                                publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &tool_call_identity,
                                    name,
                                    &result_str,
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::RECOVERY_GUARD_DENIED),
                                    0,
                                )
                                .await?;
                                messages.push(ChatMessage {
                                    role: "tool".to_string(),
                                    content: result_str,
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: Some(c.id.clone()),
                                    reasoning_content: None,
                                });
                                continue;
                            }

                            let started_at = std::time::Instant::now();
                            let args_clone = args.clone();
                            let arguments_payload =
                                serde_json::to_string(&args_clone).unwrap_or_default();
                            let arguments_reference = execution_journal
                                .as_ref()
                                .map(|journal| journal.payload_reference(&arguments_payload))
                                .unwrap_or_else(|| {
                                    crate::core::execution_journal::PayloadReference::metadata_only(
                                        &arguments_payload,
                                    )
                                });
                            let side_effect_risk =
                                tool_call_has_side_effect_risk(name, &args_clone);
                            if let Err(error) = record_tool_execution_started(
                                &execution_journal,
                                tool_call_identity.clone(),
                                name,
                                turn,
                                side_effect_risk,
                                arguments_reference,
                            ) {
                                publish_tool_result_event(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    &tool_call_identity,
                                    name,
                                    &error.to_string(),
                                    false,
                                    false,
                                    Some(crate::core::execution_event::tool_terminal_reason::JOURNAL_START_FAILED),
                                    0,
                                )
                                .await?;
                                cancel_unresolved_tool_events(
                                    &self.event_bus,
                                    &mut tool_event_ledger,
                                    &ctx.task_iri,
                                    "provider tool batch was cancelled because a durable execution-start receipt failed",
                                )
                                .await?;
                                return Err(error);
                            }
                            // Clone before awaiting to keep the executor lock
                            // out of the handler's async I/O path.  Unlike a
                            // direct handler call this also applies executor
                            // permission, syscall and hook policies.
                            let executor = self.tool_executor.read().clone();
                            let settle_workspace = requires_workspace_settlement(name)
                                || is_verification_call(name, &args_clone);
                            let mut mutation_guard = if settle_workspace {
                                Some(executor.acquire_workspace_mutation_guard().await)
                            } else {
                                None
                            };
                            let effect_snapshot = if settle_workspace {
                                capture_workspace_effect_snapshot_async(&self.tool_executor).await
                            } else {
                                None
                            };
                            let mut security_context = ctx
                                .tool_security_context(
                                    &agent.agent_id,
                                    &agent.role.to_string(),
                                    sess.session_id(),
                                )
                                .with_llm_invocation(
                                    ctx.parent_task_iri
                                        .as_deref()
                                        .unwrap_or(ctx.task_iri.as_str()),
                                    &request_id,
                                    &ctx.cycle_id,
                                )
                                .with_file_overwrite_baseline_events(
                                    action_tracker.confirmed_file_overwrite_baseline_events(),
                                );
                            if let Some(lease) = ctx.workspace_resource_lease.clone() {
                                security_context =
                                    security_context.with_workspace_resource_lease(lease);
                            }
                            let execution_result = if hook_modified_arguments {
                                executor
                                    .execute_hook_modified_with_security_context_effect_policy_and_profile(
                                        name,
                                        args,
                                        security_context,
                                        effective_allowed_tools.as_deref(),
                                        &ctx.effective_effect_policy(),
                                        execution_profile,
                                    )
                                    .await
                            } else {
                                executor
                                    .execute_with_security_context_effect_policy_and_profile(
                                        name,
                                        args,
                                        security_context,
                                        effective_allowed_tools.as_deref(),
                                        &ctx.effective_effect_policy(),
                                        execution_profile,
                                    )
                                    .await
                            };
                            let mut result =
                                execution_result.unwrap_or_else(|e| json!({"error": e}));
                            if name == "tool_search" {
                                filter_tool_search_result(&mut result, &discoverable_tools);
                            }
                            let effect_evidence = confirmed_workspace_effect_evidence(
                                &self.tool_executor,
                                name,
                                &args_clone,
                                &result,
                                effect_snapshot.as_ref(),
                            )
                            .await;
                            let delta_contaminated = (effect_evidence.uncertain
                                && ctx.workspace_resource_lease.is_some())
                                || effect_evidence.delta.as_ref().is_some_and(|delta| {
                                    workspace_delta_violates_lease(
                                        delta,
                                        ctx.workspace_resource_lease.as_ref(),
                                    )
                                });
                            if delta_contaminated {
                                warn!(
                                    tool = %name,
                                    provider_call_id = %c.id,
                                    "Observed workspace delta escaped or could not satisfy the child resource lease"
                                );
                                result = json!({
                                    "error": "Workspace delta violated the child resource lease or could not be attributed completely",
                                    "tool": name,
                                    "workspace_delta_contaminated": true,
                                });
                            }
                            action_tracker.record_with_identity(
                                name,
                                &args_clone,
                                &result,
                                started_at.elapsed().as_secs_f64(),
                                Some(tool_call_identity.clone()),
                            );
                            if let Some(delta) = effect_evidence.delta.as_ref() {
                                action_tracker
                                    .record_last_workspace_delta(delta, delta_contaminated);
                            }
                            if effect_evidence.observed || effect_evidence.uncertain {
                                // Preserve actual failed/contaminated effects
                                // and uncertain settlement in the ledger even
                                // though only a complete, lease-clean delta
                                // counts as DA progress.
                                action_tracker.mark_last_substantive_effect();
                            }
                            let verification_assessment =
                                assess_verification_call(name, &args_clone, &result);
                            if let Some(assessment) = verification_assessment.clone() {
                                action_tracker.record_last_verification_assessment(assessment);
                            }
                            if let Some(assessment) = verification_assessment.as_ref() {
                                repair_baseline_window.record_verification_assessment(
                                    name,
                                    &args_clone,
                                    assessment,
                                );
                            }
                            if let Some(coordinator) = mutation_guard.as_mut() {
                                let manifest_sha256 = effect_evidence
                                    .delta
                                    .as_ref()
                                    .filter(|delta| delta.complete)
                                    .map(|delta| delta.after_digest.as_str());
                                let stamp = coordinator.settle_action(
                                    action_tracker.last_action_invalidates_verification(),
                                    manifest_sha256,
                                );
                                action_tracker.record_last_workspace_settlement(&stamp);
                            }
                            // Action recording, typed assessment and ordering
                            // settlement are complete. Never hold the workspace
                            // coordinator across routing/hooks or an LLM call.
                            drop(mutation_guard);
                            let raw_result_str = serde_json::to_string(&result).unwrap_or_default();
                            let tool_duration_ms =
                                started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                            let result_reference = execution_journal
                                .as_ref()
                                .map(|journal| journal.payload_reference(&raw_result_str))
                                .unwrap_or_else(|| {
                                    crate::core::execution_journal::PayloadReference::metadata_only(
                                        &raw_result_str,
                                    )
                                });
                            append_execution_journal_event(
                                &execution_journal,
                                TaskExecutionJournalKind::ToolExecutionFinished {
                                    call_identity: tool_call_identity.clone(),
                                    tool_name: name.clone(),
                                    success: !crate::core::tracked_action::tool_result_failed(
                                        &result,
                                    ),
                                    duration_ms: tool_duration_ms,
                                    result: result_reference,
                                },
                            );

                            if agent.role == AgentRole::Do
                                && verification_assessment.as_ref().is_some_and(|assessment| {
                                    assessment.outcome != VerificationOutcome::Passed
                                })
                            {
                                verification_failed_this_turn = true;
                                verification_inconclusive_reason_this_turn =
                                    verification_assessment.as_ref().and_then(|assessment| {
                                        (assessment.outcome == VerificationOutcome::Inconclusive)
                                            .then(|| assessment.reason.clone())
                                            .flatten()
                                    });
                            }
                            if agent.role == AgentRole::Do
                                && execution_phase == ExecutionPhase::Verify
                                && verification_assessment.as_ref().is_some_and(|assessment| {
                                    assessment.outcome == VerificationOutcome::Passed
                                })
                            {
                                verification_succeeded_this_turn = true;
                            }
                            if !ca_executed_tool_turn_recorded {
                                ca_audit_convergence.record_executed_tool_turn(
                                    agent.role,
                                    ca_verification_probe_active,
                                );
                                ca_executed_tool_turn_recorded = true;
                            }
                            ca_audit_convergence.record_executed_call(
                                agent.role,
                                name,
                                &args_clone,
                                &result,
                            );

                            let attributable_workspace_mutation = effect_evidence.observed
                                && effect_evidence.settlement_complete
                                && !delta_contaminated
                                && effect_evidence
                                    .delta
                                    .as_ref()
                                    .is_none_or(|delta| delta.complete);
                            let successful_attributable_effect = attributable_workspace_mutation
                                && !crate::core::tracked_action::tool_result_failed(&result);
                            if attributable_workspace_mutation {
                                // This receipt describes what happened, not
                                // whether policy permits disclosure or DA may
                                // claim progress. Persist it before SkillAfter
                                // can withhold the result.
                                append_execution_journal_event(
                                    &execution_journal,
                                    TaskExecutionJournalKind::WorkspaceMutationCommitted {
                                        call_identity: tool_call_identity.clone(),
                                        tool_name: name.clone(),
                                    },
                                );
                            }

                            // A post-execution hook sees the actual result for
                            // policy evaluation, but it runs before any result
                            // reaches the event bus, cache, compressor or next
                            // model request. Internal action/effect receipts
                            // above intentionally retain what actually
                            // happened even when disclosure is denied.
                            let post_hook_denied;
                            let post_hook_control;
                            {
                                let mut hook_ctx = tool_hook_context(
                                    HookPoint::SkillAfter,
                                    agent,
                                    &ctx.task_iri,
                                    &request_id,
                                    &c.id,
                                    name,
                                    &args_clone,
                                )
                                .with_data("tool_result", Value::String(raw_result_str.clone()));
                                let decision = execute_tool_hook_decision(
                                    &self.hook_manager,
                                    HookPoint::SkillAfter,
                                    &mut hook_ctx,
                                )
                                .await;
                                emit_tool_hook_decision(
                                    &self.event_bus,
                                    &ctx.task_iri,
                                    &agent.agent_id,
                                    HookPoint::SkillAfter,
                                    &c.id,
                                    name,
                                    &decision,
                                )
                                .await;
                                let validation_feedback = hook_ctx.metadata.remove(
                                    crate::tools::tool_guard::TOOL_GUARD_VALIDATION_FEEDBACK_KEY,
                                );
                                post_hook_control = decision.control;
                                (result, post_hook_denied) =
                                    disclosed_tool_result(result, name, &decision);
                                if !post_hook_denied {
                                    attach_toolguard_validation_feedback(
                                        &mut result,
                                        validation_feedback,
                                    );
                                }
                            }
                            if post_hook_denied {
                                mark_last_action_post_hook_denied(&mut action_tracker, &result);
                            }
                            if successful_attributable_effect && !post_hook_denied {
                                effect_succeeded_this_turn = true;
                            }

                            let exposed_result_str =
                                serde_json::to_string(&result).unwrap_or_default();
                            let routing =
                                crate::tools::result_router::ResultRoutingIdentity::for_tool_call(
                                    &tool_call_identity.agent_id,
                                    &tool_call_identity.l1_session_id,
                                    &tool_call_identity.llm_request_id,
                                    &tool_call_identity.provider_call_id,
                                );
                            let mut result_str = self
                                .route_tool_result_with_routing(&exposed_result_str, name, &routing)
                                .await;
                            if !post_hook_denied {
                                session_micro_tools.extend(
                                    self.tool_executor
                                        .read()
                                        .get_micro_tool_names_for_routing_key(
                                            &routing.routing_call_key,
                                        ),
                                );
                            }
                            if name == "tool_search" && !post_hook_denied {
                                session_micro_tools.extend(
                                    result
                                        .get("matches")
                                        .and_then(Value::as_array)
                                        .into_iter()
                                        .flatten()
                                        .filter_map(|item| item.get("name").and_then(Value::as_str))
                                        .map(str::to_string),
                                );
                            }

                            debug!(
                                "  [tool] {} result: {} bytes (raw: {} bytes)",
                                name,
                                result_str.len(),
                                raw_result_str.len()
                            );

                            let tool_succeeded =
                                !crate::core::tracked_action::tool_result_failed(&result);
                            let terminal_reason = if post_hook_denied {
                                Some(crate::core::execution_event::tool_terminal_reason::RESULT_DISCLOSURE_DENIED)
                            } else if !tool_succeeded {
                                Some(crate::core::execution_event::tool_terminal_reason::EXECUTION_FAILED)
                            } else {
                                None
                            };
                            publish_tool_result_event(
                                &self.event_bus,
                                &mut tool_event_ledger,
                                &ctx.task_iri,
                                &tool_call_identity,
                                name,
                                &result_str,
                                tool_succeeded,
                                true,
                                terminal_reason,
                                tool_duration_ms.min(u64::from(u32::MAX)) as u32,
                            )
                            .await?;

                            if should_track_result_for_compression(name) {
                                let reader_available = self
                                    .tool_executor
                                    .read()
                                    .micro_tool_definition(&routing.reader_name)
                                    .is_some();
                                if let Some(ref compressor_lock) = self.tool_result_compressor {
                                    if let Ok(mut compressor) = compressor_lock.lock() {
                                        compressor.add_result_with_routing_and_reader(
                                            turn,
                                            name,
                                            &routing,
                                            &result_str,
                                            reader_available,
                                        );
                                    }
                                }
                            }

                            if let Some(err) = result.get("error") {
                                let err_msg = err.as_str().unwrap_or("");
                                let is_tool_not_found = err_msg.starts_with("Tool not found: ");
                                if post_hook_denied {
                                    warn!(
                                        tool = %name,
                                        control = ?post_hook_control,
                                        error_chars = err.to_string().chars().count(),
                                        "Tool result disclosure denied by post-execution hook policy"
                                    );
                                    errs.push(format!(
                                        "{}: result disclosure denied by post-execution policy",
                                        name
                                    ));
                                } else {
                                    warn!(
                                        tool = %name,
                                        error_chars = err.to_string().chars().count(),
                                        "Tool execution failed"
                                    );
                                    errs.push(format!("{}: tool execution failed", name));
                                }

                                if is_tool_not_found {
                                    // Micro-tool registration and handler mismatch causes "tool not found".
                                    // This is not an LLM error -- the tool list was provided by the system. Don't count as consecutive failure.
                                    // try_get_handler already attempted fallback paths; if still not found, it means
                                    // the micro-tool's validity has expired or data has been cleaned. LLM should use original tools (bash/grep etc.)
                                    // with more precise parameters to obtain needed data.
                                    // Additionally, inject prompt into tool message to guide LLM.
                                    info!("[tool_error] {} tool not found (micro-tool fallback also failed), not counting as consecutive failure", name);
                                    // Inject guidance prompt into tool message, helping LLM switch to original tools
                                    result_str = format!(
                                        "{}\n\nTip: Tool {} is currently unavailable. Please use the original tools (e.g. bash, grep_search) with more precise parameters to directly obtain the data. Do not call this micro-tool again.",
                                        result_str, name
                                    );
                                } else {
                                    // Tool execution errors don't count toward consecutive_failures.
                                    // consecutive_failures only tracks LLM-level failures (JSON parse failures, etc.).
                                    // Tool errors are normal operational feedback -- LLM has received the error and can adjust strategy.
                                    // Repeated failure of the same tool is handled by the independent tool_error_counts counter.
                                    let tool_count =
                                        tool_error_counts.entry(name.clone()).or_insert(0);
                                    *tool_count += 1;
                                    debug!(
                                        "[tool_error] {} failure count: {}/3",
                                        name, *tool_count
                                    );
                                    if *tool_count >= 3 && !tool_recovery_injected.contains(name) {
                                        warn!("[tool_error] {} failed {} consecutive times, injecting recovery guidance", name, *tool_count);
                                        tool_recovery_injected.insert(name.clone());
                                        result_str = format!(
                                            "{}\n\n[System Prompt] Tool {} failed 3 consecutive times, indicating it is currently unavailable.\
                                             \nPlease use other available tools to complete the current objective (e.g. web_search / bash / grep, etc.).\
                                             \nDo not call {} again.",
                                            result_str, name, name
                                        );
                                    }
                                }
                                if let Some(ref event_bus) = self.event_bus {
                                    let _ = event_bus
                                        .emit(
                                            &ctx.task_iri,
                                            "AGENT_ERROR",
                                            &agent.agent_id,
                                            &serde_json::json!({
                                                "error_class": if post_hook_denied {
                                                    "post_hook_denied"
                                                } else {
                                                    "tool_execution_failed"
                                                },
                                                "tool": name,
                                                "error_chars": err.to_string().chars().count(),
                                            })
                                            .to_string(),
                                        )
                                        .await;
                                }
                            } else {
                                info!("[tool] {} succeeded", name);
                                if recovery_mode_active {
                                    info!(
                                        "[consecutive_failures] recovery mode exited successfully"
                                    );
                                }
                                consecutive_failures = 0;
                                recovery_mode_active = false;
                                // Tool executed successfully, clear its error count and recovery flag
                                tool_error_counts.remove(name);
                                tool_recovery_injected.remove(name);
                            }

                            if !action_tracker.record_disclosure(
                                &tool_call_identity,
                                name,
                                post_hook_denied,
                                &result_str,
                            ) {
                                warn!(
                                    tool = %name,
                                    provider_call_id = %c.id,
                                    "Post-routing disclosure could not be bound to its full tool-call identity"
                                );
                            }

                            messages.push(ChatMessage {
                                role: "tool".to_string(),
                                content: result_str,
                                name: None,
                                tool_calls: None,
                                tool_call_id: Some(c.id.clone()),
                                reasoning_content: None,
                            });
                        }

                        close_unresolved_tool_events(
                            &self.event_bus,
                            &mut tool_event_ledger,
                            &ctx.task_iri,
                        )
                        .await?;

                        if workspace_effect_tracked {
                            da_verification_convergence.record_tool_turn(
                                effect_succeeded_this_turn,
                                verification_failed_this_turn,
                                verification_succeeded_this_turn,
                                novel_evidence_calls > 0,
                            );
                            record_workspace_effect_turn(
                                &mut workspace_effect_observed,
                                &mut consecutive_effectless_tool_turns,
                                effect_succeeded_this_turn,
                            );
                            if effect_succeeded_this_turn {
                                substantive_effect_count =
                                    substantive_effect_count.saturating_add(1);
                                low_novelty_turns = 0;
                                execution_phase = da_phase_after_tool_turn(
                                    execution_phase,
                                    true,
                                    verification_failed_this_turn,
                                );
                                verification_turns = 0;
                                info!("[DA progress] substantive workspace effect observed; no-change tail reset");
                            }
                            if verification_failed_this_turn {
                                execution_phase = da_phase_after_tool_turn(
                                    execution_phase,
                                    effect_succeeded_this_turn,
                                    true,
                                );
                                verification_turns = 0;
                                Self::upsert_runtime_control(
                                    &mut runtime_context,
                                    "da_verification_failure",
                                    if let Some(reason) =
                                        verification_inconclusive_reason_this_turn.as_deref()
                                    {
                                        format!("[DA Verification Inconclusive] The verifier did not establish a passing result ({reason}). One correction retry is available only with a different normalized verifier command; repeating the same command is denied. Use an independent test call containing only safe setup plus one final test process, with no auxiliary output commands, pipes, or redirection.")
                                    } else {
                                        "[DA Verification Failure] An execution/verification command returned a failure signal. Repair the concrete reported defect before performing more broad inspection or declaring completion; then rerun the targeted verification.".to_string()
                                    },
                                );
                                info!("[DA progress] failed verification moved execution phase to Repair");
                            } else if !effect_succeeded_this_turn {
                                let post_turn_effect_block_turns = effective_effect_block_turns(
                                    execution_phase,
                                    execution_budget.effect_progress_block_turns,
                                    execution_budget.da_repair_effect_block_turns,
                                );
                                if matches!(
                                    execution_phase,
                                    ExecutionPhase::Verify | ExecutionPhase::Repair
                                ) {
                                    verification_turns = verification_turns.saturating_add(1);
                                } else if effect_warning_turns > 0
                                    && (consecutive_effectless_tool_turns >= effect_warning_turns
                                        || low_novelty_turns >= effect_warning_turns)
                                {
                                    execution_phase = ExecutionPhase::Implement;
                                }
                                if effect_warning_turns > 0
                                    && !(workspace_effect_observed
                                        && execution_phase == ExecutionPhase::Verify)
                                    && (consecutive_effectless_tool_turns == effect_warning_turns
                                        || (post_turn_effect_block_turns > 0
                                            && consecutive_effectless_tool_turns
                                                == post_turn_effect_block_turns))
                                {
                                    let recovery_now = workspace_effect_recovery_active(
                                        workspace_effect_tracked
                                            && !(workspace_effect_observed
                                                && execution_phase == ExecutionPhase::Verify),
                                        consecutive_effectless_tool_turns,
                                        low_novelty_turns,
                                        post_turn_effect_block_turns,
                                    );
                                    if recovery_now
                                        && consecutive_effectless_tool_turns
                                            == post_turn_effect_block_turns
                                    {
                                        warn!(
                                            "[DA progress] mutation recovery activated after {} consecutive no-change tool turns; inspection/search schemas withheld",
                                            consecutive_effectless_tool_turns
                                        );
                                    }
                                    let urgency = if recovery_now {
                                        "Inspection/search tool schemas are now withheld until a substantive mutation succeeds."
                                    } else {
                                        "The available evidence is sufficient; stop broad inspection."
                                    };
                                    Self::upsert_runtime_control(
                                        &mut runtime_context,
                                        "da_execution_progress",
                                        format!(
                                            "[DA Execution Progress Contract] You have used {} consecutive tool turns without creating or modifying substantive workspace content. {} On the next turn, execute an implementation action with file_write/file_edit or a genuinely mutating command. If an exact blocker prevents that, finish with `FAILED:` and name it.",
                                            consecutive_effectless_tool_turns, urgency
                                        ),
                                    );
                                }
                            }
                        }

                        // ===== Observation Phase =====
                        info!("[ReAct Turn {}] ===== Observation =====", turn);

                        let decision = execute_cycle_hook_decision(
                            &self.hook_manager,
                            HookPoint::CycleEnd,
                            agent,
                            &ctx,
                            turn,
                            Some(true),
                        )
                        .await;
                        if matches!(decision.control, HookControl::Abort | HookControl::Retry) {
                            return Ok(cycle_hook_blocked_result(
                                &ctx,
                                HookPoint::CycleEnd,
                                &decision,
                                turn,
                                tc,
                                &errs,
                                &action_tracker.actions,
                            ));
                        }
                        if decision.control == HookControl::SkipOperation {
                            info!(turn, "CycleEnd hook skipped the post-tool transition");
                        }

                        continue;
                    } else {
                        warn!("[ReAct] action=tool_call but no tool_calls, continuing to think");
                        da_verification_convergence.record_no_tool_turn();
                        let asst_summary = parsed
                            .summary
                            .clone()
                            .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: asst_summary,
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: reasoning_content.clone(),
                        });

                        let decision = execute_cycle_hook_decision(
                            &self.hook_manager,
                            HookPoint::CycleEnd,
                            agent,
                            &ctx,
                            turn,
                            Some(false),
                        )
                        .await;
                        if matches!(decision.control, HookControl::Abort | HookControl::Retry) {
                            return Ok(cycle_hook_blocked_result(
                                &ctx,
                                HookPoint::CycleEnd,
                                &decision,
                                turn,
                                tc,
                                &errs,
                                &action_tracker.actions,
                            ));
                        }
                        if decision.control == HookControl::SkipOperation {
                            info!(turn, "CycleEnd hook skipped the empty-tool transition");
                        }
                    }
                }
                "continue" => {
                    da_verification_convergence.record_no_tool_turn();
                    let asst_summary = parsed
                        .summary
                        .clone()
                        .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: asst_summary,
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: reasoning_content.clone(),
                    });

                    let decision = execute_cycle_hook_decision(
                        &self.hook_manager,
                        HookPoint::CycleEnd,
                        agent,
                        &ctx,
                        turn,
                        Some(false),
                    )
                    .await;
                    if matches!(decision.control, HookControl::Abort | HookControl::Retry) {
                        return Ok(cycle_hook_blocked_result(
                            &ctx,
                            HookPoint::CycleEnd,
                            &decision,
                            turn,
                            tc,
                            &errs,
                            &action_tracker.actions,
                        ));
                    }
                    if decision.control == HookControl::SkipOperation {
                        info!(turn, "CycleEnd hook skipped the continue transition");
                    }
                }
                _ => {
                    warn!("[ReAct] unknown action: {}, continuing to think", action);
                    da_verification_convergence.record_no_tool_turn();
                    let asst_summary = parsed
                        .summary
                        .clone()
                        .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: asst_summary,
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: reasoning_content.clone(),
                    });

                    let decision = execute_cycle_hook_decision(
                        &self.hook_manager,
                        HookPoint::CycleEnd,
                        agent,
                        &ctx,
                        turn,
                        Some(false),
                    )
                    .await;
                    if matches!(decision.control, HookControl::Abort | HookControl::Retry) {
                        return Ok(cycle_hook_blocked_result(
                            &ctx,
                            HookPoint::CycleEnd,
                            &decision,
                            turn,
                            tc,
                            &errs,
                            &action_tracker.actions,
                        ));
                    }
                    if decision.control == HookControl::SkipOperation {
                        info!(turn, "CycleEnd hook skipped the unknown-action transition");
                    }
                }
            }
        }

        warn!(turn, error_count = errs.len(), "AgentRunner incomplete");
        // Prefer the best content turn's output (with substantive content) over the last assistant reply's short summary
        let (
            mut unfinished_status,
            mut unfinished_summary,
            mut unfinished_output,
            unfinished_archive,
        ) = if !best_content_str.is_empty() {
            (
                "partial_success".to_string(),
                best_content_summary.clone(),
                Some(Value::String(best_content_str.clone())),
                if !best_content_iri.is_empty() {
                    Some(best_content_iri.clone())
                } else {
                    None
                },
            )
        } else if agent.role != AgentRole::Check {
            if let Some((agg_summary, agg_content)) = self
                .aggregate_tool_results(
                    &messages,
                    agent,
                    &ctx,
                    compiled_prompt.map(|prompt| &prompt.spec),
                )
                .await
            {
                if is_substantive_analysis_content(&agg_content) {
                    (
                        "partial_success".to_string(),
                        agg_summary,
                        Some(Value::String(agg_content)),
                        if !best_content_iri.is_empty() {
                            Some(best_content_iri.clone())
                        } else {
                            None
                        },
                    )
                } else {
                    ("failed".to_string(), String::new(), None, None)
                }
            } else {
                ("failed".to_string(), String::new(), None, None)
            }
        } else if tc > 0 {
            ("partial_success".to_string(),
                 format!("Task partially completed. Executed {} turns, {} tool calls, {} remaining. Errors: {}.", turn, tc, effective_max_turns.saturating_sub(turn), errs.len()),
                 None, None)
        } else {
            ("failed".to_string(), String::new(), None, None)
        };
        let mut unfinished_verdict = None;
        if workspace_effect_required && !workspace_effect_observed {
            unfinished_status = "failed".to_string();
            unfinished_summary = format!(
                "FAILED: DA exhausted its execution budget without creating or modifying substantive workspace content. {}",
                unfinished_summary
            );
            errs.push("required workspace mutation was not observed".to_string());
            unfinished_verdict = Some(TaskVerdict::Failed);
        } else if agent.role == AgentRole::Check {
            let normalized = finalize_ca_terminal_contract(
                &unfinished_summary,
                unfinished_output
                    .as_ref()
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                false,
                false,
                &ctx.constraints,
                ca_executable_verifier_available
                    && ca_audit_convergence
                        .window_exhausted(agent.role, execution_budget.ca_evidence_close_turns),
                ca_has_successful_verifier_receipt(ca_audit_convergence, &action_tracker),
                &action_tracker.actions,
                self.workspace_root.as_deref(),
            );
            if let Some(issue) = normalized.contract_issue {
                errs.push(issue);
            }
            unfinished_status = normalized.verdict.to_status_str().to_string();
            unfinished_summary = normalized.summary;
            unfinished_output = Some(Value::String(normalized.content));
            unfinished_verdict = Some(normalized.verdict);
        }
        Ok(TaskResult {
            task_iri: ctx.task_iri,
            status: unfinished_status,
            summary: unfinished_summary,
            output: unfinished_output,
            jsonld_output: None,
            artifacts: vec![],
            errors: errs,
            turn_count: turn,
            tool_call_count: tc,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            verdict: unfinished_verdict,
            archive_iri: unfinished_archive,
        })
    }

    /// Build knowledge graph context string from the unified Oxigraph store.
    /// Queries for entities (subjects with rdf:type) that have labels or names,
    /// returning them as a structured context block the LLM can use to ground its reasoning.
    pub(super) fn build_kg_context(
        store: &oxigraph::store::Store,
        objective: &str,
        max_entities: usize,
        max_bytes: usize,
    ) -> String {
        // Generic orchestration words match nearly every task/turn node and
        // become progressively noisier as the graph grows. Keep only terms
        // that can discriminate reusable domain knowledge for this objective.
        let keywords = objective
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| word.chars().count() >= 3)
            .filter(|word| {
                !matches!(
                    *word,
                    "the"
                        | "and"
                        | "for"
                        | "with"
                        | "from"
                        | "task"
                        | "result"
                        | "execute"
                        | "execution"
                        | "create"
                        | "check"
                        | "plan"
                        | "artifact"
                        | "configured"
                        | "workspace"
                        | "latest"
                        | "strictly"
                )
            })
            .map(str::to_string)
            .collect::<Vec<_>>();

        // Query the two dataset scopes with static SPARQL and filter keywords
        // below. This avoids interpolating task text into SPARQL and keeps
        // default/named-graph coverage exactly identical.
        let sparql = "\
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\
            SELECT DISTINCT ?s ?label ?type WHERE {\
                {\
                    ?s a ?type .\
                    OPTIONAL { ?s rdfs:label ?label }\
                    OPTIONAL { ?s <http://schema.org/name> ?label }\
                } UNION { GRAPH ?graph {\
                    ?s a ?type .\
                    OPTIONAL { ?s rdfs:label ?label }\
                    OPTIONAL { ?s <http://schema.org/name> ?label }\
                }}\
            } ORDER BY DESC(?label) LIMIT 500\
        ";

        use oxigraph::sparql::{QueryResults as Qr, QuerySolution, SparqlEvaluator};
        let query = match SparqlEvaluator::new().parse_query(sparql) {
            Ok(query) => query,
            Err(_) => return String::new(),
        };
        let solutions: Vec<QuerySolution> = match query.on_store(store).execute() {
            Ok(Qr::Solutions(it)) => it.filter_map(Result::ok).collect(),
            _ => return String::new(),
        };

        let mut candidates: Vec<(u8, String)> = Vec::new();
        for solution in &solutions {
            let s = solution
                .get("s")
                .map(|v| format!("{}", v))
                .unwrap_or_default();
            let label = solution
                .get("label")
                .map(|v| format!("{}", v))
                .unwrap_or_default();
            let type_ = solution
                .get("type")
                .map(|v| format!("{}", v))
                .unwrap_or_default();
            // Internal runtime nodes generally have only an IRI and rdf:type.
            // Injecting those opaque identifiers leaks implementation state,
            // consumes tokens, and makes prompt size grow with every task.
            if s.is_empty() || label.is_empty() {
                continue;
            }
            let subject_lower = s.to_lowercase();
            let label_lower = label.to_lowercase();
            let type_lower = type_.to_lowercase();
            let score = keywords.iter().fold(0_u8, |score, keyword| {
                score
                    .saturating_add(u8::from(label_lower.contains(keyword)) * 4)
                    .saturating_add(u8::from(type_lower.contains(keyword)) * 2)
                    .saturating_add(u8::from(subject_lower.contains(keyword)))
            });
            if keywords.is_empty() || score == 0 {
                continue;
            }
            let entity = if !type_.is_empty() {
                format!("- **{}** ({})", label, type_)
            } else {
                format!("- **{}**", label)
            };
            if !candidates.iter().any(|(_, existing)| existing == &entity) {
                candidates.push((score, entity));
            }
        }

        candidates.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
        candidates.truncate(max_entities);
        let lines = candidates
            .into_iter()
            .map(|(_, entity)| entity)
            .collect::<Vec<_>>();

        if lines.is_empty() {
            return String::new();
        }

        let mut result = format!(
            "The following entities are available in the knowledge graph (task context: {}):\n\n",
            objective.chars().take(80).collect::<String>()
        );
        for line in lines {
            if result.len() + line.len() + 1 > max_bytes {
                break;
            }
            result.push_str(&line);
            result.push('\n');
        }
        if result.ends_with('\n') {
            result.pop();
        }
        result
    }
}

#[cfg(test)]
mod kg_context_tests {
    use crate::core::agent_runner::AgentRunner;
    use oxigraph::model::{GraphName, Literal, NamedNode, Quad};
    use oxigraph::store::Store;

    #[test]
    fn kg_context_includes_default_and_named_graph_entities() {
        let store = Store::new().unwrap();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let schema_name = NamedNode::new("http://schema.org/name").unwrap();
        let entity_type = NamedNode::new("https://example.org/Type").unwrap();

        let default_subject = NamedNode::new("https://example.org/default").unwrap();
        store
            .insert(&Quad::new(
                default_subject.clone(),
                rdf_type.clone(),
                entity_type.clone(),
                GraphName::DefaultGraph,
            ))
            .unwrap();
        store
            .insert(&Quad::new(
                default_subject,
                schema_name.clone(),
                Literal::new_simple_literal("Default Alpha"),
                GraphName::DefaultGraph,
            ))
            .unwrap();

        let named_subject = NamedNode::new("https://example.org/named").unwrap();
        let graph = NamedNode::new("https://example.org/graph").unwrap();
        store
            .insert(&Quad::new(
                named_subject.clone(),
                rdf_type,
                entity_type,
                graph.clone(),
            ))
            .unwrap();
        store
            .insert(&Quad::new(
                named_subject,
                schema_name,
                Literal::new_simple_literal("Named Alpha"),
                graph,
            ))
            .unwrap();

        let context = AgentRunner::build_kg_context(&store, "find alpha information", 12, 4096);
        assert!(context.contains("Default Alpha"), "{context}");
        assert!(context.contains("Named Alpha"), "{context}");
    }

    #[test]
    fn kg_context_includes_relevant_entity_beyond_top_50_by_label() {
        let store = Store::new().unwrap();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let schema_name = NamedNode::new("http://schema.org/name").unwrap();
        let entity_type = NamedNode::new("https://example.org/Type").unwrap();

        // Insert 60 filler entities whose labels sort *after* the target under
        // ORDER BY DESC(?label) ("Zebra..." > "Alpha..."), so the task-relevant
        // entity falls outside a tight LIMIT 50 window and must still be injected.
        for i in 0..60 {
            let subject = NamedNode::new(format!("https://example.org/filler_{}", i)).unwrap();
            store
                .insert(&Quad::new(
                    subject.clone(),
                    rdf_type.clone(),
                    entity_type.clone(),
                    GraphName::DefaultGraph,
                ))
                .unwrap();
            store
                .insert(&Quad::new(
                    subject,
                    schema_name.clone(),
                    Literal::new_simple_literal(format!("Zebra Filler Entity {}", i)),
                    GraphName::DefaultGraph,
                ))
                .unwrap();
        }

        let target = NamedNode::new("https://example.org/target").unwrap();
        store
            .insert(&Quad::new(
                target,
                rdf_type,
                entity_type,
                GraphName::DefaultGraph,
            ))
            .unwrap();
        store
            .insert(&Quad::new(
                NamedNode::new("https://example.org/target").unwrap(),
                schema_name,
                Literal::new_simple_literal("Alpha Relevant"),
                GraphName::DefaultGraph,
            ))
            .unwrap();

        let context = AgentRunner::build_kg_context(&store, "find alpha information", 12, 4096);
        assert!(
            context.contains("Alpha Relevant"),
            "task-relevant entity beyond label top-50 must be injected, got: {context}"
        );
    }

    #[test]
    fn kg_context_excludes_unlabelled_runtime_nodes_and_is_bounded() {
        let store = Store::new().unwrap();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let schema_name = NamedNode::new("http://schema.org/name").unwrap();
        let entity_type = NamedNode::new("https://example.org/ProbeKnowledge").unwrap();

        let runtime_node = NamedNode::new("iri://task/probe-internal-turn").unwrap();
        store
            .insert(&Quad::new(
                runtime_node,
                rdf_type.clone(),
                entity_type.clone(),
                GraphName::DefaultGraph,
            ))
            .unwrap();

        for index in 0..30 {
            let subject = NamedNode::new(format!("https://example.org/probe/{index}")).unwrap();
            store
                .insert(&Quad::new(
                    subject.clone(),
                    rdf_type.clone(),
                    entity_type.clone(),
                    GraphName::DefaultGraph,
                ))
                .unwrap();
            store
                .insert(&Quad::new(
                    subject,
                    schema_name.clone(),
                    Literal::new_simple_literal(format!("Probe Knowledge {index}")),
                    GraphName::DefaultGraph,
                ))
                .unwrap();
        }

        let context = AgentRunner::build_kg_context(&store, "use probe knowledge", 12, 4096);

        assert!(!context.contains("probe-internal-turn"), "{context}");
        assert_eq!(context.matches("- **").count(), 12, "{context}");
        assert!(context.len() <= 4096, "{} bytes", context.len());
    }

    #[test]
    fn collect_tool_entries_caps_at_max_keeping_most_recent() {
        use crate::gateway::unified_gateway::{ChatMessage, ToolCallFunction, ToolCallPayload};

        // Build 25 assistant→tool pairs; only the configured recent entries
        // should survive so the force-finish summary prompt stays bounded.
        let mut messages: Vec<ChatMessage> = Vec::new();
        for i in 0..25 {
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: String::new(),
                name: None,
                tool_calls: Some(vec![ToolCallPayload {
                    id: format!("call_{}", i),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "search".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            });
            messages.push(ChatMessage {
                role: "tool".to_string(),
                content: format!("result_of_call_{}", i),
                name: None,
                tool_calls: None,
                tool_call_id: Some(format!("call_{}", i)),
                reasoning_content: None,
            });
        }

        let entries = super::collect_tool_entries(&messages, 20);
        assert_eq!(entries.len(), 20, "entries must be capped");
        assert!(
            entries.iter().any(|(_, c)| c.contains("result_of_call_24")),
            "most recent call must survive the cap"
        );
        assert!(
            !entries.iter().any(|(_, c)| c.contains("result_of_call_0")),
            "oldest call must be evicted by the cap"
        );
    }
}
