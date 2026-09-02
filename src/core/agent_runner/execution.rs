use serde_json::{json, Value};
use std::collections::HashSet;
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

pub(super) fn requires_workspace_effect(ctx: &TaskContext, role: AgentRole) -> bool {
    role == AgentRole::Do && ctx.effective_effect_policy().requires_workspace_mutation()
}

/// Conservative evidence that a tool call can create or modify substantive
/// workspace content. Directory creation alone intentionally does not count:
/// it was the observed failure mode where DA created a folder and then only
/// inspected files while claiming implementation progress.
pub(super) fn is_substantive_workspace_effect(name: &str, args: &Value) -> bool {
    match name {
        "file_write" | "file_edit" => true,
        "bash" | "powershell" | "code_execute" => {
            let command = args
                .get("command")
                .or_else(|| args.get("code"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_lowercase();
            if command.is_empty() {
                return false;
            }
            let mutating_patterns = [
                "sed -i",
                "perl -pi",
                "git apply",
                "patch ",
                "mv ",
                "install ",
                "curl -o",
                "curl --output",
                "wget -o",
                "unzip ",
                "tar -x",
                "npm create",
                "npx create-",
                "cargo new",
                "cargo add",
                "django-admin startproject",
                "rails new",
                "write_text(",
                "write_bytes(",
                "open(",
            ];
            mutating_patterns
                .iter()
                .any(|pattern| command.contains(pattern))
                || has_substantive_copy_effect(&command)
        }
        _ => false,
    }
}

#[derive(Debug, Clone)]
pub(super) struct WorkspaceEffectSnapshot {
    generation: u64,
    semantic_fingerprint: Option<String>,
    debounce_ms: u64,
}

pub(super) fn capture_workspace_effect_snapshot(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
) -> Option<WorkspaceEffectSnapshot> {
    executor
        .read()
        .get_workspace_monitor()
        .map(|monitor| WorkspaceEffectSnapshot {
            generation: monitor.generation(),
            semantic_fingerprint: monitor.semantic_effect_fingerprint().ok(),
            debounce_ms: monitor.config.debounce_ms,
        })
}

/// Confirm a successful semantic effect rather than inferring one from tool
/// syntax. Explicit file tools expose `changed`; shell-like tools use bounded
/// before/after content fingerprints and degrade to monitor generation only
/// when the configured snapshot limits cannot cover the workspace.
pub(super) async fn confirmed_workspace_effect(
    executor: &parking_lot::RwLock<crate::tools::ToolExecutor>,
    name: &str,
    args: &Value,
    result: &Value,
    before: Option<&WorkspaceEffectSnapshot>,
) -> bool {
    if crate::core::tracked_action::tool_result_failed(result)
        || result.get("background_task_id").is_some()
    {
        return false;
    }
    if matches!(name, "file_write" | "file_edit") {
        return result.get("changed").and_then(Value::as_bool) == Some(true);
    }
    if !is_substantive_workspace_effect(name, args) {
        return false;
    }

    let Some(before) = before else {
        return false;
    };
    let Some(monitor) = executor.read().get_workspace_monitor() else {
        return false;
    };
    let after_fingerprint = monitor.semantic_effect_fingerprint().ok();
    if let (Some(before_fingerprint), Some(after_fingerprint)) =
        (&before.semantic_fingerprint, after_fingerprint.as_ref())
    {
        return before_fingerprint != after_fingerprint;
    }

    // Oversized/unreadable snapshots use the existing monitor as a degraded
    // strategy. Wait only for its configured debounce window, and only on
    // this uncommon fallback path.
    warn!(
        tool = name,
        before_generation = before.generation,
        "semantic workspace effect snapshot unavailable; using generation fallback"
    );
    if monitor.generation() <= before.generation && before.debounce_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(before.debounce_ms)).await;
    }
    monitor.generation() > before.generation
}

/// Treat a normal copy (including restoring *from* a backup) as a workspace
/// effect, but do not let precautionary `cp source source.bak` calls reset the
/// DA progress guard. This is intentionally a narrow shell-token inspection,
/// not a shell parser: uncertain copy forms remain substantive so the guard
/// does not incorrectly block legitimate generators.
fn has_substantive_copy_effect(command: &str) -> bool {
    let tokens = command.split_whitespace().collect::<Vec<_>>();
    let mut index = 0usize;
    while index < tokens.len() {
        let token = tokens[index]
            .trim_matches(|ch: char| matches!(ch, '\'' | '"' | '(' | ')' | ';' | '&' | '|'));
        if token.rsplit('/').next() != Some("cp") {
            index += 1;
            continue;
        }

        let mut operands = Vec::new();
        index += 1;
        while index < tokens.len() {
            let raw = tokens[index];
            if matches!(raw, "&&" | "||" | "|" | ";") {
                break;
            }
            let operand =
                raw.trim_matches(|ch: char| matches!(ch, '\'' | '"' | '(' | ')' | ';' | '&' | '|'));
            if !operand.is_empty() && !operand.starts_with('-') {
                operands.push(operand);
            }
            index += 1;
        }

        let Some(destination) = operands.last() else {
            // An unrecognized/incomplete copy is not evidence of progress.
            continue;
        };
        let backup_only = destination.ends_with(".bak")
            || destination.ends_with(".backup")
            || destination.ends_with(".orig")
            || destination.ends_with('~');
        if !backup_only {
            return true;
        }
    }
    false
}

pub(super) fn is_workspace_mutation_candidate(name: &str, args: &Value) -> bool {
    if is_substantive_workspace_effect(name, args) {
        return true;
    }
    if !matches!(name, "bash" | "powershell" | "code_execute") {
        return false;
    }
    let command = args
        .get("command")
        .or_else(|| args.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    [
        "mkdir ",
        "touch ",
        "rm ",
        "rmdir ",
        "del ",
        "remove-item",
        "new-item",
        "set-content",
        "add-content",
    ]
    .iter()
    .any(|pattern| command.contains(pattern))
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
/// change. This is an intersection with the already role/SA-authorized window;
/// it never broadens authority. If a custom workflow allowed only reads, the
/// result is intentionally empty and DA must report the capability blocker.
pub(super) fn mutation_recovery_tool_definitions(definitions: Vec<Value>) -> Vec<Value> {
    definitions
        .into_iter()
        .filter(|definition| {
            definition["function"]["name"]
                .as_str()
                .is_some_and(is_workspace_mutation_tool_name)
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
    session_tools: &HashSet<String>,
    tool_name: &str,
) -> Option<Value> {
    // A result-reader may be created by an earlier call in the same provider
    // response or retained by this BizAgent after its reference was compressed
    // out of the current message window. It is still execution-local and safe
    // to honor; ordinary withdrawn tools remain strictly rejected.
    let owned_result_reader =
        ToolExecutor::is_micro_tool_name(tool_name) && session_tools.contains(tool_name);
    (!advertised.contains(tool_name) && !owned_result_reader).then(|| {
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

pub(super) fn ca_evidence_focus_tool_definitions(
    definitions: Vec<Value>,
    role: AgentRole,
    evidence_focus_active: bool,
) -> Vec<Value> {
    if role != AgentRole::Check || !evidence_focus_active {
        return definitions;
    }
    definitions
        .into_iter()
        .filter(|definition| {
            matches!(
                definition["function"]["name"].as_str().unwrap_or_default(),
                "bash" | "powershell" | "file_read" | "read_agent_output"
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
            ) || ToolExecutor::is_micro_tool_name(name)
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
    messages: &mut Vec<ChatMessage>,
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
    messages.retain(|message| message.name.as_deref() != Some("execution_ledger"));
    messages.push(ChatMessage {
        role: "system".to_string(),
        content: format!(
            "# Execution Ledger (replaceable state)\n\n- phase: {:?}\n- effect_policy: {:?}\n- substantive_effects: {}\n- verification_tool_turns_after_effect: {}\n- low_novelty_evidence_score: {}\n- workspace_generation: {}\n\nContinue from this state. Do not repeat evidence queries that produced no new generation or target information.",
            phase,
            effect_policy,
            mutation_count,
            verification_turns,
            low_novelty_turns,
            workspace_generation,
        ),
        name: Some("execution_ledger".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
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
use crate::core::execution_event::{ExecutionEvent, ExecutionEventKind};
use crate::core::execution_journal::{TaskExecutionJournal, TaskExecutionJournalKind};
use crate::gateway::unified_gateway::ChatMessage;
use crate::jsonld::{JsonLdContext, JsonLdNode};
use crate::memory::l1_session::L1Session;
use crate::tools::hooks::{HookContext, HookPoint, HookResult};
use crate::tools::tool_executor::ToolExecutor;
use crate::CoreError;

use super::{TaskContext, TaskResult, TaskVerdict};

fn append_execution_journal_event(
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

fn journal_error_class(error: &CoreError) -> &'static str {
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
        CoreError::Internal { .. } => "internal",
        CoreError::PermissionDenied { .. } => "permission_denied",
    }
}

/// Replace (never accumulate) the transient workspace delta message when the
/// monitor generation advances. Full file content remains out of prompt and
/// is recovered through the existing micro-tools.
pub(super) fn refresh_workspace_delta_message(
    executor: &std::sync::Arc<parking_lot::RwLock<ToolExecutor>>,
    messages: &mut Vec<ChatMessage>,
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
    messages.retain(|message| message.name.as_deref() != Some("workspace_delta"));
    messages.push(ChatMessage {
        role: "system".to_string(),
        content: format!(
            "# Workspace Delta (evidence only)\n\n{}\n\nUse the changed paths directly; do not re-list unchanged directories.",
            delta
        ),
        name: Some("workspace_delta".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
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
        self.execute_internal(agent, ctx, None).await
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
        self.execute_internal(agent, ctx, Some(agent_md)).await
    }

    async fn execute_internal(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
        agent_md: Option<&str>,
    ) -> Result<TaskResult, CoreError> {
        // AgentInit hook
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::AgentInit,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            self.hook_manager
                .execute(HookPoint::AgentInit, &mut hook_ctx)
                .await;
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
            let hook_result = self
                .hook_manager
                .execute(HookPoint::TaskStart, &mut hook_ctx)
                .await;
            if hook_result == HookResult::Abort {
                agent.status = AgentStatus::Failed;
                return Ok(TaskResult {
                    task_iri: ctx.task_iri,
                    status: "aborted".to_string(),
                    summary: "Task aborted by hook".to_string(),
                    output: None,
                    jsonld_output: None,
                    artifacts: Vec::new(),
                    errors: vec!["Task aborted by hook".to_string()],
                    turn_count: 0,
                    tool_call_count: 0,
                    five_w2h_updates: None,
                    tracked_actions: Vec::new(),
                    verdict: None,
                    archive_iri: None,
                });
            }
        }

        // AgentStart hook
        let mut hook_ctx = HookContext::new(
            HookPoint::AgentStart,
            &agent.agent_id,
            &agent.role.to_string(),
        )
        .with_task(&ctx.task_iri, &ctx.task_iri);
        let hook_result = self
            .hook_manager
            .execute(HookPoint::AgentStart, &mut hook_ctx)
            .await;
        if hook_result == HookResult::Abort {
            agent.status = AgentStatus::Failed;
            return Ok(TaskResult {
                task_iri: ctx.task_iri,
                status: "aborted".to_string(),
                summary: "Agent aborted by hook".to_string(),
                output: None,
                jsonld_output: None,
                artifacts: Vec::new(),
                errors: vec!["Agent aborted by hook".to_string()],
                turn_count: 0,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                verdict: None,
                archive_iri: None,
            });
        }

        let mut session = self.memory_manager.lock().await.create_session(
            &agent.agent_id,
            &agent.role.to_string(),
            &ctx.task_iri,
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

        let result = self.exec(agent, ctx.clone(), &mut session, agent_md).await;

        {
            let mut mm = self.memory_manager.lock().await;
            if !result
                .as_ref()
                .map(|r| r.tracked_actions.is_empty())
                .unwrap_or(true)
            {
                if let Ok(ref r) = result {
                    let _ = mm.archive_session_actions(&r.task_iri, &r.tracked_actions, &r.summary);
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
            let _ = mm.finalize_session(session, &ctx.task_iri);
        }

        // TaskEnd hook
        {
            let mut hook_ctx =
                HookContext::new(HookPoint::TaskEnd, &agent.agent_id, &agent.role.to_string())
                    .with_task(&ctx.task_iri, &ctx.task_iri);
            self.hook_manager
                .execute(HookPoint::TaskEnd, &mut hook_ctx)
                .await;
        }

        // AgentEnd hook
        let mut hook_ctx = HookContext::new(
            HookPoint::AgentEnd,
            &agent.agent_id,
            &agent.role.to_string(),
        );
        self.hook_manager
            .execute(HookPoint::AgentEnd, &mut hook_ctx)
            .await;

        // Handle errors
        if let Ok(ref r) = result {
            if r.status == "failed" {
                let mut hook_ctx = HookContext::new(
                    HookPoint::AgentError,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri);
                hook_ctx.error = Some(r.summary.clone());
                self.hook_manager
                    .execute(HookPoint::AgentError, &mut hook_ctx)
                    .await;

                let mut hook_ctx = HookContext::new(
                    HookPoint::TaskError,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri);
                hook_ctx.error = Some(r.summary.clone());
                self.hook_manager
                    .execute(HookPoint::TaskError, &mut hook_ctx)
                    .await;
            }
        }

        result
    }

    /// In force-finish scenarios, extract tool results from messages and call LLM for final aggregated summary.
    /// Returns (summary, full_content), or None if no tool results are aggregatable or LLM fails.
    async fn aggregate_tool_results(
        &self,
        messages: &[ChatMessage],
        agent: &AgentInstance,
        ctx: &TaskContext,
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

        let model = self
            .gateway
            .get_model(&agent.role.to_string().to_lowercase());
        let req_messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        match self
            .gateway
            .chat_with_params(&model, req_messages, None, None, None, None)
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
                warn!("[force-finish] LLM aggregation call failed: {}", e);
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
    ) -> Result<TaskResult, CoreError> {
        let model = self
            .gateway
            .get_model(&agent.role.to_string().to_lowercase());
        let supports_reasoning = self.gateway.supports_native_reasoning(&model);

        let generated_agent_md;
        let agent_md = if let Some(agent_md) = agent_md_override {
            agent_md
        } else {
            let context_data = self.gather_context_data_async(agent.role, &ctx).await;
            generated_agent_md =
                self.build_agent_md(agent.role, &ctx.objective, &context_data, &model);
            &generated_agent_md
        };

        // Build system prompt (relatively static, placed in system role)
        let system_content = self.build_system_prompt(agent, &ctx, sess, agent_md).await;

        // Build context message (dynamic, placed in the final user role)
        let summary_iris = sess.get_summary_chain_with_iris(20, 100);
        let summary_text = summary_iris.join("\n");

        let mut task_parts = vec![format!("## Current Task\n{}", ctx.objective)];
        if !ctx.expected_output.is_empty() {
            task_parts.push(format!("## Expected Output\n{}", ctx.expected_output));
        }
        if !ctx.success_criteria.is_empty() {
            task_parts.push(format!("## Success Criteria\n{}", ctx.success_criteria));
        }
        if let Some(contract) = super::direct_response_delivery_contract(&ctx.constraints) {
            task_parts.push(format!("## Authoritative Delivery Contract\n{contract}"));
        }
        if let Some(contract) = super::required_capability_contract(&ctx.constraints) {
            task_parts.push(format!("## Authoritative Evidence Capability\n{contract}"));
        }
        let task_section = task_parts.join("\n\n");

        let context_msg = if summary_text.is_empty() {
            format!(
                "{}\n\n## Available Tools\nUse tools as needed to complete the task.",
                task_section,
            )
        } else {
            format!(
                "{}\n\n## History Summary\n{}\n\nTo view the full report of a specific turn, use the read_agent_output tool with the corresponding IRI.\n\n## Available Tools\nUse tools as needed to complete the task.",
                task_section, summary_text
            )
            .to_string()
        };

        let mut messages: Vec<ChatMessage> = vec![ChatMessage {
            role: "system".to_string(),
            content: system_content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        if ctx.workspace_context_enabled() && agent.role != AgentRole::Check {
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
        let perception_text = self.perception_store.take_perception_text_scoped(
            &ctx.task_iri,
            ctx.workspace_context_enabled() && agent.role != AgentRole::Check,
        );
        if !perception_text.is_empty() {
            info!(
                "[perception] injecting {} bytes of perception content",
                perception_text.len()
            );
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: format!("# Agent Perception\n\n{}", perception_text),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
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
                    messages.push(ChatMessage {
                        role: "system".to_string(),
                        content: format!("# Knowledge Graph Context\n\n{}", kg_context),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                }
            }
        }

        // Resume mode: restore history messages from checkpoint, placed after system and before new user message
        // So LLM sees historical context first, then the continue instruction
        if let Some(ref resumed) = ctx.resumed_messages {
            // Skip original system message (replaced with new one), append remaining history
            for msg in resumed.iter().skip(1) {
                messages.push(msg.clone());
            }
            info!(
                "[resume] restored {} history messages from checkpoint",
                resumed.len().saturating_sub(1)
            );
        }

        // New user message placed after history as continue instruction
        let resume_task_parts = if ctx.expected_output.is_empty() && ctx.success_criteria.is_empty()
        {
            format!("Current Task: {}", ctx.objective)
        } else {
            let mut parts = vec![format!("Current Task: {}", ctx.objective)];
            if !ctx.expected_output.is_empty() {
                parts.push(format!("Expected Output: {}", ctx.expected_output));
            }
            if !ctx.success_criteria.is_empty() {
                parts.push(format!("Success Criteria: {}", ctx.success_criteria));
            }
            parts.join("\n")
        };
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: if ctx.resumed_messages.is_some() {
                format!(
                    "[Continue] Please continue the task from where you left off.\n\n{}",
                    resume_task_parts
                )
            } else {
                context_msg
            },
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });

        let tools = self.tool_definitions_for_task_context(&agent.role.to_string(), &ctx);
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
                warn!(%error, "Task execution journal is unavailable; continuing without durable trace");
                None
            }
        };

        // Track the richest content turn (used for passing archive_iri across agents, pointing to substantive content rather than final turn summary)
        let mut best_content_len: usize = 0;
        let mut best_content_str: String = String::new();
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
        let effect_block_turns = effective_effect_block_turns(
            execution_phase,
            execution_budget.effect_progress_block_turns,
            execution_budget.da_repair_effect_block_turns,
        );
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
        let mut planning_tool_turns = 0u32;
        let mut evidence_only_tool_turns = 0u32;

        // Initial checkpoint: record task start state
        let start_role_str = agent.role.to_string();
        match checkpoint_manager.create_ext(
            &ctx.task_iri,
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
            Ok(checkpoint) => record_checkpoint_commit(&execution_journal, &checkpoint),
            Err(error) => warn!("[checkpoint] initial save failed: {}", error),
        }

        // Soft limit state: progressive prompts, no hard truncation (DA and AA use 3-stage degradation)
        let mut soft_limit_early_warning_sent = false;
        let mut soft_limit_final_warning_sent = false;
        let mut soft_limit_force_finish = false;

        loop {
            if ctx.workspace_context_enabled() {
                refresh_workspace_delta_message(
                    &self.tool_executor,
                    &mut messages,
                    &mut workspace_generation,
                    workspace_delta_limit,
                );
            }
            refresh_execution_ledger(
                &mut messages,
                agent.role,
                execution_phase,
                &ctx.effective_effect_policy(),
                substantive_effect_count,
                verification_turns,
                low_novelty_turns,
                workspace_generation,
            );
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
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "【Turn Limit Notice】Please control execution turns. Limited turns remain. Focus on the core task, avoid unnecessary tool calls, and finish as soon as possible.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
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
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: final_turn_limit_notice(
                        agent.role,
                        workspace_effect_tracked,
                        workspace_effect_observed,
                        consecutive_effectless_tool_turns,
                    ),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
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
                    match checkpoint_manager.create_ext(
                        &ctx.task_iri,
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
                        Ok(checkpoint) => record_checkpoint_commit(&execution_journal, &checkpoint),
                        Err(error) => warn!("[checkpoint] max_turns save failed: {}", error),
                    }
                    messages.push(ChatMessage {
                        role: "user".to_string(),
                        content: "【System Force-Finish】Maximum execution turns reached. Please output your final summary and results immediately. Do not call any more tools. If there are incomplete tool executions, base your summary on the results already available.".to_string(),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                    // Don't break, let this turn's LLM respond to the force-finish directive
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
                    match checkpoint_manager.create_ext(
                        &ctx.task_iri,
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
                        Ok(checkpoint) => record_checkpoint_commit(&execution_journal, &checkpoint),
                        Err(error) => warn!("[checkpoint] force_end save failed: {}", error),
                    }
                    // Fallback: if no turn has substantive content, aggregate tool results via LLM
                    let (force_summary, force_output, force_archive) = if !best_content_str
                        .is_empty()
                    {
                        let s = Self::generate_auto_summary(&best_content_str);
                        (
                            s,
                            Some(Value::String(best_content_str.clone())),
                            if !best_content_iri.is_empty() {
                                Some(best_content_iri.clone())
                            } else {
                                None
                            },
                        )
                    } else if let Some((agg_summary, agg_content)) =
                        self.aggregate_tool_results(&messages, agent, &ctx).await
                    {
                        (
                            agg_summary,
                            Some(Value::String(agg_content)),
                            if !best_content_iri.is_empty() {
                                Some(best_content_iri.clone())
                            } else {
                                None
                            },
                        )
                    } else if let Some(last) = messages.iter().rev().find(|m| m.role == "assistant")
                    {
                        (
                            Self::generate_auto_summary(&last.content),
                            Some(Value::String(last.content.clone())),
                            None,
                        )
                    } else {
                        ("Task not completed".to_string(), None, None)
                    };
                    return Ok(TaskResult {
                        task_iri: ctx.task_iri,
                        status: "partial_success".to_string(),
                        summary: force_summary,
                        output: force_output,
                        jsonld_output: None,
                        artifacts: vec![],
                        errors: errs,
                        turn_count: turn,
                        tool_call_count: tc,
                        five_w2h_updates: None,
                        tracked_actions: action_tracker.actions,
                        verdict: None,
                        archive_iri: force_archive,
                    });
                }
            }
            turn += 1;

            // Save periodic checkpoint every 5 turns (including tool error state)
            if turn % 5 == 0 {
                let turn_role_str = agent.role.to_string();
                let tool_error_str = serde_json::json!({
                    "error_counts": tool_error_counts,
                    "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                })
                .to_string();
                let action_str = serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                match checkpoint_manager.create_ext(
                    &ctx.task_iri,
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
                    Ok(checkpoint) => record_checkpoint_commit(&execution_journal, &checkpoint),
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
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: recovery_msg,
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                info!(
                    "[consecutive_failures] triggered recovery mode: {} consecutive failures",
                    consecutive_failures
                );
                consecutive_failures = 0;
                continue;
            }

            // ===== Thought Phase =====
            info!("[ReAct Turn {}] ===== Thought =====", turn);

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
                    for entry in &pending {
                        messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: entry.content.clone(),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        });
                        sess.add_supplement(
                            "user",
                            &entry.content,
                            entry.embedding.clone(),
                            Some(entry.relevance_score),
                        );
                    }
                }
            }

            // CycleStart hook
            {
                let mut hook_ctx = HookContext::new(
                    HookPoint::CycleStart,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri)
                .with_data("turn", Value::Number(turn.into()));
                self.hook_manager
                    .execute(HookPoint::CycleStart, &mut hook_ctx)
                    .await;
            }

            {
                let mut hook_ctx = HookContext::new(
                    HookPoint::LlmRequest,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri);
                let hook_result = self
                    .hook_manager
                    .execute(HookPoint::LlmRequest, &mut hook_ctx)
                    .await;
                if hook_result == HookResult::Abort {
                    errs.push("LLM request aborted by hook".to_string());
                    break;
                }
            }

            // Use ContextWindowManager for dual-dimension compression based on message count and tokens
            let context_window_compressed = if let Some(ref cwm_lock) = self.context_window_manager
            {
                let cwm = cwm_lock.lock().expect("cwm_lock Mutex poisoned");
                let model = self
                    .gateway
                    .get_model(&agent.role.to_string().to_lowercase());
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
                    let (compressed, summary_text) = cwm.compress_messages(&messages);
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

            if !guard_pending_pre_injections.is_empty() {
                let prompt = format!(
                    "\n\n[ToolGuard Constraint Directive]\n{}\nNote: The above constraints only apply to the same-named tool calls you make next. Strictly comply.",
                    guard_pending_pre_injections.join("\n")
                );
                if let Some(sys_msg) = messages.first_mut() {
                    if sys_msg.role == "system" {
                        // Replace rather than append: remove old ToolGuard block to prevent cumulative bloat per turn
                        if let Some(pos) =
                            sys_msg.content.find("\n\n[ToolGuard Constraint Directive]")
                        {
                            sys_msg.content.truncate(pos);
                        }
                        sys_msg.content.push_str(&prompt);
                    }
                }
                guard_pending_pre_injections.clear();
            }

            debug!(
                "[turn {}] calling LLM (history: {} msgs, tools: {})",
                turn,
                messages.len(),
                tools.len()
            );

            let mutation_recovery_active = workspace_effect_recovery_active(
                workspace_effect_tracked,
                consecutive_effectless_tool_turns,
                low_novelty_turns,
                effect_block_turns,
            );
            let mut request_messages = messages.clone();
            let ca_evidence_focus_active = agent.role == AgentRole::Check
                && execution_budget.ca_evidence_focus_turns > 0
                && verification_turns >= execution_budget.ca_evidence_focus_turns;
            let ca_evidence_close_active = agent.role == AgentRole::Check
                && execution_budget.ca_evidence_close_turns > 0
                && verification_turns >= execution_budget.ca_evidence_close_turns;
            let pa_planning_focus_active = agent.role == AgentRole::Plan
                && execution_budget.pa_planning_focus_turns > 0
                && planning_tool_turns >= execution_budget.pa_planning_focus_turns;
            let da_evidence_focus_active = agent.role == AgentRole::Do
                && matches!(
                    ctx.effective_effect_policy(),
                    crate::core::effect::EffectPolicy::EvidenceOnly
                )
                && execution_budget.da_evidence_focus_turns > 0
                && evidence_only_tool_turns >= execution_budget.da_evidence_focus_turns;
            let da_evidence_close_active = agent.role == AgentRole::Do
                && matches!(
                    ctx.effective_effect_policy(),
                    crate::core::effect::EffectPolicy::EvidenceOnly
                )
                && execution_budget.da_evidence_close_turns > 0
                && evidence_only_tool_turns >= execution_budget.da_evidence_close_turns;
            if ca_evidence_focus_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[CA Evidence Convergence] Multiple verification tool turns have already completed. Finish now with PASS/FAIL and criterion-linked evidence unless one named acceptance criterion is still unverified. If one remains, perform only the single targeted check needed for that criterion; do not repeat broad discovery or already-passing checks.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if ca_evidence_close_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[CA Evidence Close Gate] The configured evidence window is exhausted. Do not call another tool. Return the final criterion-linked PASS/FAIL audit now. Any criterion lacking evidence must be marked FAIL; uncertainty is not a reason for more discovery.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if pa_planning_focus_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[PA Planning Convergence] The configured inspection window is complete. Use the objective, workspace manifest, retrieved evidence, and prior-cycle feedback already supplied. Emit the executable plan now; do not request more tools. Preserve explicit acceptance criteria and name the checks DA/CA must run.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if da_evidence_focus_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[DA Evidence Convergence] The configured evidence-discovery window is complete. Synthesize the requested deliverable now from the sources and evidence already collected. Only one targeted source read is permitted when a specific claim lacks support; do not perform another broad search.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if da_evidence_close_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "[DA Evidence Close Gate] The configured evidence window is exhausted. Do not call another tool. Return the complete evidence-backed deliverable now, explicitly marking any unsupported point as a limitation.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            if mutation_recovery_active {
                request_messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: format!(
                        "[DA Mutation Recovery Mode] The last {} tool turns made no substantive workspace change. Inspection/search tools are temporarily unavailable. Make the highest-priority pending change now with an advertised mutation-capable tool. If the authorized tool window contains no such tool or another exact blocker prevents progress, finish with `FAILED:` and name the blocker.",
                        consecutive_effectless_tool_turns
                    ),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
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
                let current_tools = ca_evidence_focus_tool_definitions(
                    current_tools,
                    agent.role,
                    ca_evidence_focus_active,
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
                );
                let current_tools = da_evidence_close_tool_definitions(
                    current_tools,
                    agent.role,
                    da_evidence_close_active,
                );
                if mutation_recovery_active {
                    mutation_recovery_tool_definitions(current_tools)
                } else {
                    current_tools
                }
            };
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
            let request_tools = (!current_tools.is_empty()).then_some(current_tools);
            let request_id = format!(
                "llm_{}_{}_{}",
                agent.agent_id,
                turn,
                uuid::Uuid::new_v4().hyphenated()
            );
            let request_payload = serde_json::to_string(&json!({
                "messages": &request_messages,
                "tools": &request_tools,
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
                            "operation": "正在等待模型响应",
                        })
                        .to_string(),
                    )
                    .await;
            }
            let llm_started_at = std::time::Instant::now();
            let (response, gateway_metadata) = match self
                .gateway
                .chat_with_params_traced(&model, request_messages, None, None, request_tools, None)
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
                                    "error": error.to_string(),
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
                        },
                    );
                    return Err(CoreError::Internal {
                        message: error.to_string(),
                    });
                }
            };
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
                    request_id,
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
                            "operation": "模型响应已收到",
                            "latency_ms": gateway_metadata.latency_ms,
                            "attempts": gateway_metadata.attempts,
                            "cache_hit": gateway_metadata.cache_hit,
                        })
                        .to_string(),
                    )
                    .await;
            }

            // Accumulate token usage
            if let Some(ref usage) = response.usage {
                self.total_prompt_tokens
                    .fetch_add(usage.prompt_tokens as u64, Ordering::Relaxed);
                self.total_completion_tokens
                    .fetch_add(usage.completion_tokens as u64, Ordering::Relaxed);
                self.last_prompt_tokens
                    .store(usage.prompt_tokens as u64, Ordering::Relaxed);
                self.last_completion_tokens
                    .store(usage.completion_tokens as u64, Ordering::Relaxed);
            }

            {
                let mut hook_ctx = HookContext::new(
                    HookPoint::LlmResponse,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri);
                self.hook_manager
                    .execute(HookPoint::LlmResponse, &mut hook_ctx)
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

            // Some Responses-API-compatible reasoning models (observed with
            // DeepSeek) complete a terminal turn with `content: null` while
            // putting the only conclusion in `reasoning_content`. Treating
            // that as an empty answer let a DA that explicitly reported
            // "not completed" pass through SA as success. Tool-call turns
            // intentionally keep empty assistant content; this fallback is
            // terminal-only.
            let effective_content = Self::effective_response_content(
                &raw_content,
                reasoning_content.as_deref(),
                finish,
                choice.message.tool_calls.is_some(),
            );

            if let Some(ref event_bus) = self.event_bus {
                let completion_tokens = response
                    .usage
                    .as_ref()
                    .map(|usage| usage.completion_tokens)
                    .unwrap_or(0);
                if !raw_content.is_empty() {
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
                if let Some(reasoning) =
                    reasoning_content.as_deref().filter(|text| !text.is_empty())
                {
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
            }

            debug!(
                "[turn {}] LLM response: finish={}, content_len={}, has_reasoning={}",
                turn,
                finish,
                effective_content.len(),
                reasoning_content.is_some()
            );

            let parsed = self.parse_llm_response(
                &effective_content,
                reasoning_content.as_deref(),
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

            if finish == "tool_calls" && choice.message.tool_calls.is_some() {
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
                "[ReAct Turn {}] Thought: action={}, summary={}",
                turn,
                action,
                parsed.summary.as_deref().unwrap_or("")
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
            if parsed.content.len() > best_content_len {
                best_content_len = parsed.content.len();
                best_content_str = parsed.content.clone();
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

                    // CycleEnd hook
                    {
                        let mut hook_ctx = HookContext::new(
                            HookPoint::CycleEnd,
                            &agent.agent_id,
                            &agent.role.to_string(),
                        )
                        .with_task(&ctx.task_iri, &ctx.task_iri)
                        .with_data("turn", Value::Number(turn.into()))
                        .with_data("had_tool_calls", Value::Bool(false));
                        self.hook_manager
                            .execute(HookPoint::CycleEnd, &mut hook_ctx)
                            .await;
                    }

                    info!("AgentRunner completed: {} turns, {} tools", turn, tc);
                    debug!("[L0] L0 entries: {}", self.l0_store.count().unwrap_or(0));

                    // When parsed.content is empty (LLM returned content=null + tool_calls),
                    // aggregate from tool results in messages to ensure subsequent agent can read a valid plan
                    let (mut final_summary, output_value) =
                        if !parsed.content.trim().is_empty() {
                            (
                                parsed.summary.clone().unwrap_or_else(|| {
                                    Self::generate_auto_summary(&parsed.content)
                                }),
                                Value::String(parsed.content.clone()),
                            )
                        } else if let Some((agg_summary, agg_content)) =
                            self.aggregate_tool_results(&messages, agent, &ctx).await
                        {
                            (agg_summary, Value::String(agg_content))
                        } else {
                            (
                                parsed.summary.clone().unwrap_or_else(|| {
                                    Self::generate_auto_summary(&parsed.content)
                                }),
                                Value::String(parsed.content.clone()),
                            )
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
                    match checkpoint_manager.create_ext(
                        &ctx.task_iri,
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
                        Ok(checkpoint) => record_checkpoint_commit(&execution_journal, &checkpoint),
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
                        // If parsed.content is empty (tool_calls-only response), try LLM aggregation of existing tool results
                        let (mut final_summary, output_value) = if !parsed.content.trim().is_empty()
                        {
                            (
                                parsed.summary.clone().unwrap_or_else(|| {
                                    Self::generate_auto_summary(&parsed.content)
                                }),
                                Value::String(parsed.content.clone()),
                            )
                        } else if let Some((agg_summary, agg_content)) =
                            self.aggregate_tool_results(&messages, agent, &ctx).await
                        {
                            (agg_summary, Value::String(agg_content))
                        } else {
                            (
                                parsed.summary.clone().unwrap_or_else(|| {
                                    Self::generate_auto_summary(&parsed.content)
                                }),
                                Value::String(parsed.content.clone()),
                            )
                        };
                        let jsonld_output =
                            self.apply_output_mapping(&output_value, &agent.role, &ctx.task_iri);
                        let intercept_archive = if !best_content_iri.is_empty() {
                            Some(best_content_iri.clone())
                        } else {
                            None
                        };
                        if workspace_effect_required && !workspace_effect_observed {
                            let detail = "DA reached its turn limit without a substantive workspace mutation";
                            errs.push(detail.to_string());
                            final_summary = format!("FAILED: {}. {}", detail, final_summary);
                        }
                        let force_verdict =
                            if workspace_effect_required && !workspace_effect_observed {
                                TaskVerdict::Failed
                            } else if soft_limit_force_finish
                                && errs
                                    .iter()
                                    .any(|e| e.contains("max turns") || e.contains("force-finish"))
                            {
                                TaskVerdict::PartialSuccess
                            } else {
                                TaskVerdict::Success
                            };
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

                        let has_effect_candidate = calls.iter().any(|call| {
                            let args = serde_json::from_str::<Value>(&call.function.arguments)
                                .unwrap_or_default();
                            is_substantive_workspace_effect(&call.function.name, &args)
                        });
                        let block_effectless_calls =
                            mutation_recovery_active && !has_effect_candidate;
                        let mut effect_succeeded_this_turn = false;
                        let mut verification_failed_this_turn = false;
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
                        if agent.role == AgentRole::Check && !calls.is_empty() {
                            verification_turns = verification_turns.saturating_add(1);
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

                                let (final_summary, output_value) =
                                    if !parsed.content.trim().is_empty() {
                                        (
                                            parsed.summary.clone().unwrap_or_else(|| {
                                                "PA has formulated a plan".to_string()
                                            }),
                                            Value::String(parsed.content.clone()),
                                        )
                                    } else if let Some((agg_summary, agg_content)) =
                                        self.aggregate_tool_results(&messages, agent, &ctx).await
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

                        for c in calls {
                            tc += 1;
                            let name = &c.function.name;
                            let args_raw = &c.function.arguments;
                            let args: Value = serde_json::from_str(args_raw).unwrap_or_default();
                            debug!(
                                "  [tool] {} args={}",
                                name,
                                &args_raw.chars().take(100).collect::<String>()
                            );

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

                            // Emit tool_call event for TUI display
                            if let Some(ref event_bus) = self.event_bus {
                                let tce = ExecutionEvent {
                                    event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                                    task_iri: ctx.task_iri.clone(),
                                    timestamp: chrono::Utc::now().timestamp_millis(),
                                    event: ExecutionEventKind::ToolCall(
                                        crate::core::execution_event::ToolCall {
                                            call_id: c.id.clone(),
                                            tool_name: name.clone(),
                                            arguments_json: args_raw.clone(),
                                            agent_id: agent.agent_id.clone(),
                                            sequence: tc,
                                        },
                                    ),
                                };
                                let _ = event_bus
                                    .emit(
                                        &ctx.task_iri,
                                        "TOOL_CALL",
                                        &agent.agent_id,
                                        &serde_json::to_string(&tce).unwrap_or_default(),
                                    )
                                    .await;
                            }

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
                                if let Some(ref event_bus) = self.event_bus {
                                    let tre = ExecutionEvent {
                                        event_id: format!(
                                            "evt_{}",
                                            uuid::Uuid::new_v4().hyphenated()
                                        ),
                                        task_iri: ctx.task_iri.clone(),
                                        timestamp: chrono::Utc::now().timestamp_millis(),
                                        event: ExecutionEventKind::ToolResult(
                                            crate::core::execution_event::ToolResult {
                                                call_id: c.id.clone(),
                                                tool_name: name.clone(),
                                                result: result_str.clone(),
                                                // The protocol feedback was
                                                // handled successfully; the
                                                // requested tool was not run.
                                                success: true,
                                                result_size_bytes: result_str.len() as u32,
                                                duration_ms: 0,
                                                agent_id: agent.agent_id.clone(),
                                            },
                                        ),
                                    };
                                    let _ = event_bus
                                        .emit(
                                            &ctx.task_iri,
                                            "TOOL_RESULT",
                                            &agent.agent_id,
                                            &serde_json::to_string(&tre).unwrap_or_default(),
                                        )
                                        .await;
                                }
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
                                let mut hook_ctx = HookContext::new(
                                    HookPoint::SkillBefore,
                                    &agent.agent_id,
                                    &agent.role.to_string(),
                                )
                                .with_task(&ctx.task_iri, &ctx.task_iri)
                                .with_data("tool_name", Value::String(name.clone()));
                                self.hook_manager
                                    .execute(HookPoint::SkillBefore, &mut hook_ctx)
                                    .await;
                                // Capture ToolGuard pre-injections for next LLM call
                                if let Some(injections) =
                                    hook_ctx.metadata.remove("guard_pre_injections")
                                {
                                    if let Value::Array(arr) = injections {
                                        for v in arr {
                                            if let Some(s) = v.as_str() {
                                                guard_pending_pre_injections.push(s.to_string());
                                            }
                                        }
                                    }
                                }
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
                            append_execution_journal_event(
                                &execution_journal,
                                TaskExecutionJournalKind::ToolExecutionStarted {
                                    call_id: c.id.clone(),
                                    tool_name: name.clone(),
                                    turn,
                                    arguments: arguments_reference,
                                },
                            );
                            let effect_snapshot =
                                (is_substantive_workspace_effect(name, &args_clone)
                                    && !matches!(name.as_str(), "file_write" | "file_edit"))
                                .then(|| capture_workspace_effect_snapshot(&self.tool_executor))
                                .flatten();
                            // Clone before awaiting to keep the executor lock
                            // out of the handler's async I/O path.  Unlike a
                            // direct handler call this also applies executor
                            // permission, syscall and hook policies.
                            let mut result = if block_effectless_calls {
                                json!({
                                    "error": "DA execution-progress guard blocked another inspection-only turn",
                                    "required_next_action": "Create or modify a substantive artifact with file_write/file_edit or a genuinely mutating command; otherwise finish with FAILED and the exact blocker."
                                })
                            } else {
                                let executor = self.tool_executor.read().clone();
                                executor
                                    .execute_with_security_context(
                                        name,
                                        args,
                                        crate::skill_graph::security::SecurityContext::new(
                                            &agent.agent_id,
                                            &agent.role.to_string(),
                                        )
                                        .with_task(&ctx.task_iri),
                                        ctx.allowed_tools.as_deref(),
                                    )
                                    .await
                                    .unwrap_or_else(|e| json!({"error": e}))
                            };
                            if name == "tool_search" {
                                filter_tool_search_result(&mut result, &discoverable_tools);
                            }
                            action_tracker.record(
                                name,
                                &args_clone,
                                &result,
                                started_at.elapsed().as_secs_f64(),
                            );
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
                                    call_id: c.id.clone(),
                                    tool_name: name.clone(),
                                    success: !crate::core::tracked_action::tool_result_failed(
                                        &result,
                                    ),
                                    duration_ms: tool_duration_ms,
                                    result: result_reference,
                                },
                            );

                            if agent.role == AgentRole::Do
                                && matches!(name.as_str(), "bash" | "powershell" | "code_execute")
                                && crate::core::tracked_action::tool_result_failed(&result)
                            {
                                verification_failed_this_turn = true;
                            }

                            if !block_effectless_calls
                                && confirmed_workspace_effect(
                                    &self.tool_executor,
                                    name,
                                    &args_clone,
                                    &result,
                                    effect_snapshot.as_ref(),
                                )
                                .await
                            {
                                effect_succeeded_this_turn = true;
                                action_tracker.mark_last_substantive_effect();
                                append_execution_journal_event(
                                    &execution_journal,
                                    TaskExecutionJournalKind::WorkspaceMutationCommitted {
                                        call_id: c.id.clone(),
                                        tool_name: name.clone(),
                                    },
                                );
                            }

                            let mut result_str =
                                self.route_tool_result(&raw_result_str, name, &c.id).await;
                            session_micro_tools.extend(
                                self.tool_executor
                                    .read()
                                    .get_micro_tool_names_for_call(&c.id),
                            );
                            if name == "tool_search" {
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

                            // Emit tool_result event for TUI display
                            if let Some(ref event_bus) = self.event_bus {
                                let tre = ExecutionEvent {
                                    event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                                    task_iri: ctx.task_iri.clone(),
                                    timestamp: chrono::Utc::now().timestamp_millis(),
                                    event: ExecutionEventKind::ToolResult(
                                        crate::core::execution_event::ToolResult {
                                            call_id: c.id.clone(),
                                            tool_name: name.clone(),
                                            result: result_str.clone(),
                                            success:
                                                !crate::core::tracked_action::tool_result_failed(
                                                    &result,
                                                ),
                                            result_size_bytes: result_str.len() as u32,
                                            duration_ms: tool_duration_ms.min(u64::from(u32::MAX))
                                                as u32,
                                            agent_id: agent.agent_id.clone(),
                                        },
                                    ),
                                };
                                let _ = event_bus
                                    .emit(
                                        &ctx.task_iri,
                                        "TOOL_RESULT",
                                        &agent.agent_id,
                                        &serde_json::to_string(&tre).unwrap_or_default(),
                                    )
                                    .await;
                            }

                            if let Some(ref compressor_lock) = self.tool_result_compressor {
                                if let Ok(mut compressor) = compressor_lock.lock() {
                                    compressor.add_result(turn, name, &c.id, &result_str);
                                    compressor.compress_tool_messages(&mut messages);
                                }
                            }
                            self.compress_tool_results_with_microtools(&mut messages);

                            // Cross-turn aging: compress old tool results by staleness
                            if let Some(ref aging) = self.tool_result_aging {
                                let (aged, freed) =
                                    aging.age_tool_results(&mut messages, &self.tool_executor);
                                if aged > 0 {
                                    info!(
                                        "[turn {}] ToolResultAging aged {} results, freed {} bytes",
                                        turn, aged, freed
                                    );
                                }
                            }

                            if let Some(err) = result.get("error") {
                                let err_msg = err.as_str().unwrap_or("");
                                let is_tool_not_found = err_msg.starts_with("Tool not found: ");
                                warn!("[tool] {} failed: {}", name, err);
                                errs.push(format!("{}: {}", name, err));

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
                                            &serde_json::json!({"error": err, "tool": name})
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

                            {
                                let mut hook_ctx = HookContext::new(
                                    HookPoint::SkillAfter,
                                    &agent.agent_id,
                                    &agent.role.to_string(),
                                )
                                .with_task(&ctx.task_iri, &ctx.task_iri)
                                .with_data("tool_name", Value::String(name.clone()))
                                .with_data("tool_result", Value::String(raw_result_str.clone()));
                                let hook_result = self
                                    .hook_manager
                                    .execute(HookPoint::SkillAfter, &mut hook_ctx)
                                    .await;

                                if hook_result == HookResult::Abort {
                                    let guard_msg = hook_ctx.error.unwrap_or_else(|| {
                                        "Tool result rejected by guard".to_string()
                                    });
                                    warn!("[tool] {} ToolGuard intercepted: {}", name, guard_msg);
                                    messages.push(ChatMessage {
                                        role: "tool".to_string(),
                                        content: format!("[ToolGuard Intercepted] Tool {} result rejected by security system. {}", name, guard_msg),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: Some(c.id.clone()),
                                        reasoning_content: None,
                                    });
                                } else {
                                    messages.push(ChatMessage {
                                        role: "tool".to_string(),
                                        content: result_str,
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: Some(c.id.clone()),
                                        reasoning_content: None,
                                    });
                                }
                            }
                        }

                        if workspace_effect_tracked {
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
                                messages.push(ChatMessage {
                                    role: "user".to_string(),
                                    content: "[DA Verification Failure] An execution/verification command returned a failure signal. Repair the concrete reported defect before performing more broad inspection or declaring completion; then rerun the targeted verification.".to_string(),
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: None,
                                    reasoning_content: None,
                                });
                                info!("[DA progress] failed verification moved execution phase to Repair");
                            } else if !effect_succeeded_this_turn {
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
                                    && (consecutive_effectless_tool_turns == effect_warning_turns
                                        || (effect_block_turns > 0
                                            && consecutive_effectless_tool_turns
                                                == effect_block_turns))
                                {
                                    let recovery_now = workspace_effect_recovery_active(
                                        workspace_effect_tracked,
                                        consecutive_effectless_tool_turns,
                                        low_novelty_turns,
                                        effect_block_turns,
                                    );
                                    if recovery_now
                                        && consecutive_effectless_tool_turns == effect_block_turns
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
                                    messages.push(ChatMessage {
                                        role: "user".to_string(),
                                        content: format!(
                                            "[DA Execution Progress Contract] You have used {} consecutive tool turns without creating or modifying substantive workspace content. {} On the next turn, execute an implementation action with file_write/file_edit or a genuinely mutating command. If an exact blocker prevents that, finish with `FAILED:` and name it.",
                                            consecutive_effectless_tool_turns, urgency
                                        ),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: None,
                                        reasoning_content: None,
                                    });
                                }
                            }
                        }

                        // ===== Observation Phase =====
                        info!("[ReAct Turn {}] ===== Observation =====", turn);

                        // CycleEnd hook (tool calls path)
                        {
                            let mut hook_ctx = HookContext::new(
                                HookPoint::CycleEnd,
                                &agent.agent_id,
                                &agent.role.to_string(),
                            )
                            .with_task(&ctx.task_iri, &ctx.task_iri)
                            .with_data("turn", Value::Number(turn.into()))
                            .with_data("had_tool_calls", Value::Bool(true));
                            self.hook_manager
                                .execute(HookPoint::CycleEnd, &mut hook_ctx)
                                .await;
                        }

                        continue;
                    } else {
                        warn!("[ReAct] action=tool_call but no tool_calls, continuing to think");
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

                        // CycleEnd hook
                        {
                            let mut hook_ctx = HookContext::new(
                                HookPoint::CycleEnd,
                                &agent.agent_id,
                                &agent.role.to_string(),
                            )
                            .with_task(&ctx.task_iri, &ctx.task_iri)
                            .with_data("turn", Value::Number(turn.into()))
                            .with_data("had_tool_calls", Value::Bool(false));
                            self.hook_manager
                                .execute(HookPoint::CycleEnd, &mut hook_ctx)
                                .await;
                        }
                    }
                }
                "continue" => {
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

                    // CycleEnd hook
                    {
                        let mut hook_ctx = HookContext::new(
                            HookPoint::CycleEnd,
                            &agent.agent_id,
                            &agent.role.to_string(),
                        )
                        .with_task(&ctx.task_iri, &ctx.task_iri)
                        .with_data("turn", Value::Number(turn.into()))
                        .with_data("had_tool_calls", Value::Bool(false));
                        self.hook_manager
                            .execute(HookPoint::CycleEnd, &mut hook_ctx)
                            .await;
                    }
                }
                _ => {
                    warn!("[ReAct] unknown action: {}, continuing to think", action);
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

                    // CycleEnd hook
                    {
                        let mut hook_ctx = HookContext::new(
                            HookPoint::CycleEnd,
                            &agent.agent_id,
                            &agent.role.to_string(),
                        )
                        .with_task(&ctx.task_iri, &ctx.task_iri)
                        .with_data("turn", Value::Number(turn.into()))
                        .with_data("had_tool_calls", Value::Bool(false));
                        self.hook_manager
                            .execute(HookPoint::CycleEnd, &mut hook_ctx)
                            .await;
                    }
                }
            }
        }

        warn!("AgentRunner incomplete: {} turns, errors: {:?}", turn, errs);
        // Prefer the best content turn's output (with substantive content) over the last assistant reply's short summary
        let (mut unfinished_status, mut unfinished_summary, unfinished_output, unfinished_archive) =
            if !best_content_str.is_empty() {
                (
                    "partial_success".to_string(),
                    Self::generate_auto_summary(&best_content_str),
                    Some(Value::String(best_content_str.clone())),
                    if !best_content_iri.is_empty() {
                        Some(best_content_iri.clone())
                    } else {
                        None
                    },
                )
            } else if let Some((agg_summary, agg_content)) =
                self.aggregate_tool_results(&messages, agent, &ctx).await
            {
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
            } else if let Some(last) = messages.iter().rev().find(|m| m.role == "assistant") {
                (
                    "partial_success".to_string(),
                    Self::generate_auto_summary(&last.content),
                    Some(Value::String(last.content.clone())),
                    None,
                )
            } else if tc > 0 {
                ("partial_success".to_string(),
                 format!("Task partially completed. Executed {} turns, {} tool calls, {} remaining. Errors: {}.", turn, tc, effective_max_turns.saturating_sub(turn), errs.len()),
                 None, None)
            } else {
                ("failed".to_string(), String::new(), None, None)
            };
        if workspace_effect_required && !workspace_effect_observed {
            unfinished_status = "failed".to_string();
            unfinished_summary = format!(
                "FAILED: DA exhausted its execution budget without creating or modifying substantive workspace content. {}",
                unfinished_summary
            );
            errs.push("required workspace mutation was not observed".to_string());
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
            verdict: None,
            archive_iri: unfinished_archive,
        })
    }

    /// Build knowledge graph context string from the unified Oxigraph store.
    /// Queries for entities (subjects with rdf:type) that have labels or names,
    /// returning them as a structured context block the LLM can use to ground its reasoning.
    fn build_kg_context(
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
