use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::gateway::unified_gateway::ChatMessage;
use crate::memory::l0_store::L0Store;
use crate::CoreError;

/// Current durable schema for task resumption.  The older checkpoint fields
/// remain for backwards compatibility, but new readers use this single
/// validated object instead of independently re-parsing loosely related JSON
/// fields at each application entry point.
pub const TASK_RESUME_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskResumeState {
    pub schema_version: u32,
    pub checkpoint_name: String,
    pub turn: u32,
    pub tool_call_count: u32,
    pub current_role: Option<String>,
    pub prev_summary: Option<String>,
    pub five_w2h_json: Option<String>,
    pub cycle_state_json: Option<String>,
    pub completed_nodes_json: Option<String>,
    pub pending_approvals_json: Option<String>,
    pub supplement_json: Option<String>,
    pub tool_error_json: Option<String>,
    pub action_tracker_json: Option<String>,
    pub perception_anomaly_json: Option<String>,
}

/// The sole supported checkpoint restoration input for callers.  Parsing and
/// schema validation happen before an application begins a resumed task.
#[derive(Debug, Clone)]
pub struct RestoredTask {
    pub checkpoint: CheckpointData,
    pub messages: Vec<ChatMessage>,
    pub state: TaskResumeState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    pub checkpoint_iri: String,
    pub task_iri: String,
    pub name: String,
    pub node_count: i32,
    pub total_size_bytes: i32,
    pub created_at: DateTime<Utc>,
    pub tags: Vec<String>,
    pub nodes_json: String,
    pub session_messages_json: String,
    pub agent_state_json: String,

    // ── Extension fields (Option ensures backward compatibility with old checkpoints) ──
    /// The currently executing agent role (PA/DA/CA/AA), used by resume to decide which phases to skip
    pub current_role: Option<String>,

    /// 5W2H snapshot (with fill stage tracking), used for full SA execution context restoration
    pub five_w2h_json: Option<String>,

    /// prev_summary chain value (summary passed through PA→DA→CA→AA)
    pub prev_summary: Option<String>,

    /// CycleState serialization (phase, iteration, phase_history, experience_hints)
    pub cycle_state_json: Option<String>,

    /// Completed DAG node results (key=node_id, value=NodeResult JSON)
    pub completed_nodes_json: Option<String>,

    /// Pending human approval requests
    pub pending_approvals_json: Option<String>,

    /// Pending supplementary input entries
    pub supplement_json: Option<String>,

    /// Accumulated tool error count + injected recovery tool set in the React loop
    pub tool_error_json: Option<String>,

    /// ActionTracker accumulated tracked actions
    pub action_tracker_json: Option<String>,

    /// Perception engine anomaly history (used for dedup)
    pub perception_anomaly_json: Option<String>,

    /// Versioned runtime state used by all new resume paths.  Missing values
    /// are migrated from legacy fields at read time.
    #[serde(default)]
    pub resume_state: Option<TaskResumeState>,
}

pub struct CheckpointManager {
    l0: Option<Arc<L0Store>>,
    task_checkpoints: RwLock<HashMap<String, Vec<String>>>,
    counter: AtomicU64,
}

/// Upper bound on checkpoints retained per task. Older entries beyond this
/// limit are evicted on the next create to keep L0 growth bounded for
/// long-running multi-turn tasks.
pub const MAX_CHECKPOINTS_PER_TASK: usize = 20;

impl CheckpointManager {
    pub fn new() -> Self {
        Self {
            l0: None,
            task_checkpoints: RwLock::new(HashMap::new()),
            counter: AtomicU64::new(0),
        }
    }

    pub fn with_persistence(l0: Arc<L0Store>) -> Self {
        Self {
            l0: Some(l0),
            task_checkpoints: RwLock::new(HashMap::new()),
            counter: AtomicU64::new(0),
        }
    }

    pub fn create(
        &self,
        task_iri: &str,
        name: &str,
        nodes_json: &str,
        session_messages_json: &str,
        agent_state_json: &str,
        tags: &[String],
    ) -> Result<CheckpointData, CoreError> {
        let seq = self.counter.fetch_add(1, Ordering::SeqCst);
        let checkpoint_iri = format!(
            "iri://checkpoint/{}/seq_{}_{}",
            task_iri.strip_prefix("iri://").unwrap_or(task_iri),
            seq,
            uuid::Uuid::new_v4().hyphenated(),
        );

        let nodes: Vec<serde_json::Value> = serde_json::from_str(nodes_json).unwrap_or_default();
        let node_count = nodes.len() as i32;
        let total_size_bytes = nodes_json.len() as i32
            + session_messages_json.len() as i32
            + agent_state_json.len() as i32;

        let checkpoint = CheckpointData {
            checkpoint_iri: checkpoint_iri.clone(),
            task_iri: task_iri.to_string(),
            name: name.to_string(),
            node_count,
            total_size_bytes,
            created_at: Utc::now(),
            tags: tags.to_vec(),
            nodes_json: nodes_json.to_string(),
            session_messages_json: session_messages_json.to_string(),
            agent_state_json: agent_state_json.to_string(),
            current_role: None,
            five_w2h_json: None,
            prev_summary: None,
            cycle_state_json: None,
            completed_nodes_json: None,
            pending_approvals_json: None,
            supplement_json: None,
            tool_error_json: None,
            action_tracker_json: None,
            perception_anomaly_json: None,
            resume_state: Some(TaskResumeState::from_fields(
                name,
                agent_state_json,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )),
        };

        let content = serde_json::to_string(&checkpoint).map_err(|e| CoreError::Internal {
            message: format!("Failed to serialize checkpoint: {}", e),
        })?;
        self.store_checkpoint(&checkpoint_iri, &content)?;

        {
            let mut task_cps = self.task_checkpoints.write();
            task_cps
                .entry(task_iri.to_string())
                .or_insert_with(Vec::new)
                .push(checkpoint_iri.clone());
        }
        self.prune_oldest(task_iri);

        Ok(checkpoint)
    }

    /// Extended creation method: supports all optional fields. None fields won't appear in serialization (saving L0 space).
    #[allow(clippy::too_many_arguments)]
    pub fn create_ext(
        &self,
        task_iri: &str,
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
    ) -> Result<CheckpointData, CoreError> {
        let seq = self.counter.fetch_add(1, Ordering::SeqCst);
        let checkpoint_iri = format!(
            "iri://checkpoint/{}/seq_{}_{}",
            task_iri.strip_prefix("iri://").unwrap_or(task_iri),
            seq,
            uuid::Uuid::new_v4().hyphenated(),
        );

        let nodes: Vec<serde_json::Value> = serde_json::from_str(nodes_json).unwrap_or_default();
        let node_count = nodes.len() as i32;

        let mut total = nodes_json.len() as i32
            + session_messages_json.len() as i32
            + agent_state_json.len() as i32;
        if let Some(v) = five_w2h_json {
            total += v.len() as i32;
        }
        if let Some(v) = cycle_state_json {
            total += v.len() as i32;
        }
        if let Some(v) = completed_nodes_json {
            total += v.len() as i32;
        }
        if let Some(v) = pending_approvals_json {
            total += v.len() as i32;
        }
        if let Some(v) = supplement_json {
            total += v.len() as i32;
        }
        if let Some(v) = tool_error_json {
            total += v.len() as i32;
        }
        if let Some(v) = action_tracker_json {
            total += v.len() as i32;
        }
        if let Some(v) = perception_anomaly_json {
            total += v.len() as i32;
        }

        let checkpoint = CheckpointData {
            checkpoint_iri: checkpoint_iri.clone(),
            task_iri: task_iri.to_string(),
            name: name.to_string(),
            node_count,
            total_size_bytes: total,
            created_at: Utc::now(),
            tags: tags.to_vec(),
            nodes_json: nodes_json.to_string(),
            session_messages_json: session_messages_json.to_string(),
            agent_state_json: agent_state_json.to_string(),
            current_role: current_role.map(|s| s.to_string()),
            five_w2h_json: five_w2h_json.map(|s| s.to_string()),
            prev_summary: prev_summary.map(|s| s.to_string()),
            cycle_state_json: cycle_state_json.map(|s| s.to_string()),
            completed_nodes_json: completed_nodes_json.map(|s| s.to_string()),
            pending_approvals_json: pending_approvals_json.map(|s| s.to_string()),
            supplement_json: supplement_json.map(|s| s.to_string()),
            tool_error_json: tool_error_json.map(|s| s.to_string()),
            action_tracker_json: action_tracker_json.map(|s| s.to_string()),
            perception_anomaly_json: perception_anomaly_json.map(|s| s.to_string()),
            resume_state: Some(TaskResumeState::from_fields(
                name,
                agent_state_json,
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
            )),
        };

        let content = serde_json::to_string(&checkpoint).map_err(|e| CoreError::Internal {
            message: format!("Failed to serialize checkpoint: {}", e),
        })?;
        self.store_checkpoint(&checkpoint_iri, &content)?;

        {
            let mut task_cps = self.task_checkpoints.write();
            task_cps
                .entry(task_iri.to_string())
                .or_insert_with(Vec::new)
                .push(checkpoint_iri.clone());
        }
        self.prune_oldest(task_iri);

        Ok(checkpoint)
    }

    fn store_checkpoint(&self, iri: &str, content: &str) -> Result<(), CoreError> {
        if let Some(ref l0) = self.l0 {
            l0.store(iri, content)?;
        }
        Ok(())
    }

    pub fn restore(&self, checkpoint_iri: &str) -> Result<CheckpointData, CoreError> {
        if let Some(ref l0) = self.l0 {
            if let Ok(Some(entry)) = l0.retrieve(checkpoint_iri) {
                return Self::deserialize_checkpoint(&entry.content);
            }
        }
        Err(CoreError::Internal {
            message: format!("Checkpoint not found: {}", checkpoint_iri),
        })
    }

    pub fn restore_latest(&self, task_iri: &str) -> Result<Option<CheckpointData>, CoreError> {
        let list = self.list(task_iri, 1);
        Ok(list.into_iter().next())
    }

    /// Restore the latest fully valid checkpoint and its parsed, versioned
    /// runtime state.  Invalid/corrupt newest records are skipped in favour of
    /// an older usable checkpoint rather than making recovery all-or-nothing.
    pub fn restore_task(&self, task_iri: &str) -> Result<Option<RestoredTask>, CoreError> {
        for checkpoint in self.list(task_iri, MAX_CHECKPOINTS_PER_TASK as i32) {
            match checkpoint.to_restored_task() {
                Ok(restored) => return Ok(Some(restored)),
                Err(error) => tracing::warn!(
                    checkpoint_iri = %checkpoint.checkpoint_iri,
                    %error,
                    "Skipping invalid checkpoint during task restore"
                ),
            }
        }
        Ok(None)
    }

    /// Restore the latest checkpoint for a given task, parsing its phase label.
    /// Returns (checkpoint, phase_label) where phase_label is one of:
    ///   "start_<Role>" / "turn_<Role>_N" / "finish_<Role>" / "max_turns_<Role>"
    ///   "force_end_<Role>" / "step_complete_<Role>" / "pre_dispatch_<Role>"
    ///   or "unknown"
    pub fn restore_latest_with_phase(
        &self,
        task_iri: &str,
    ) -> Result<Option<(CheckpointData, String)>, CoreError> {
        let cp = self.restore_latest(task_iri)?;
        Ok(cp.map(|c| {
            let phase = parse_checkpoint_phase(&c.name);
            (c, phase)
        }))
    }

    /// Restore the latest checkpoint for a given task and infer which phases are done based on the phase.
    /// Returns (checkpoint, skip_roles) — skip_roles is the list of AgentRoles to skip during resume.
    pub fn restore_latest_with_skip_roles(
        &self,
        task_iri: &str,
    ) -> Result<Option<(CheckpointData, Vec<String>)>, CoreError> {
        let restored = self.restore_task(task_iri)?;
        Ok(restored.map(|restored| {
            let skip_roles = compute_skip_roles_from_phase(
                &restored.state.checkpoint_name,
                restored.state.current_role.as_deref(),
            );
            (restored.checkpoint, skip_roles)
        }))
    }

    pub fn list(&self, task_iri: &str, limit: i32) -> Vec<CheckpointData> {
        // Try in-memory index first (valid within the same process)
        {
            let task_cps = self.task_checkpoints.read();
            if let Some(cp_iris) = task_cps.get(task_iri) {
                let mut results: Vec<CheckpointData> = cp_iris
                    .iter()
                    .rev()
                    .filter_map(|iri| {
                        if let Some(ref l0) = self.l0 {
                            l0.retrieve(iri)
                                .ok()
                                .flatten()
                                .and_then(|e| Self::deserialize_checkpoint(&e.content).ok())
                        } else {
                            None
                        }
                    })
                    .collect();
                results.truncate(limit as usize);
                return results;
            }
        }
        // In-memory index miss → scan from L0 by IRI prefix (for cross-process recovery)
        if let Some(ref l0) = self.l0 {
            let stripped = task_iri.strip_prefix("iri://").unwrap_or(task_iri);
            let prefix = format!("iri://checkpoint/{}/", stripped);
            if let Ok(entries) = l0.scan_iri_prefix(&prefix, 100_000) {
                let mut results: Vec<CheckpointData> = entries
                    .iter()
                    .filter_map(|e| Self::deserialize_checkpoint(&e.content).ok())
                    .collect();
                results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                results.truncate(limit as usize);
                return results;
            }
        }
        Vec::new()
    }

    pub fn delete(&self, checkpoint_iri: &str) -> Result<(), CoreError> {
        if let Some(ref l0) = self.l0 {
            if l0.retrieve(checkpoint_iri)?.is_none() {
                return Err(CoreError::Internal {
                    message: format!("Checkpoint not found: {}", checkpoint_iri),
                });
            }
            l0.delete(checkpoint_iri)?;
        }
        {
            let mut task_cps = self.task_checkpoints.write();
            for iris in task_cps.values_mut() {
                iris.retain(|iri| iri != checkpoint_iri);
            }
        }
        Ok(())
    }

    pub fn checkpoint_count(&self) -> u64 {
        self.task_checkpoints
            .read()
            .values()
            .map(|v| v.len() as u64)
            .sum()
    }

    fn prune_oldest(&self, task_iri: &str) {
        {
            let mut task_cps = self.task_checkpoints.write();
            if let Some(iris) = task_cps.get_mut(task_iri) {
                while iris.len() > MAX_CHECKPOINTS_PER_TASK {
                    iris.remove(0);
                }
            }
        }

        // CheckpointManager instances are intentionally short-lived in some
        // execution paths. Enforcing retention only through their in-memory
        // index both leaked old checkpoints and allowed a fresh manager's
        // `seq_0` to overwrite another BizAgent's evidence. UUID-backed IRIs
        // prevent collisions; this persisted scan enforces the task-wide cap
        // across roles and process lifetimes.
        let Some(ref l0) = self.l0 else {
            return;
        };
        let stripped = task_iri.strip_prefix("iri://").unwrap_or(task_iri);
        let prefix = format!("iri://checkpoint/{}/", stripped);
        let Ok(entries) = l0.scan_iri_prefix(&prefix, 100_000) else {
            return;
        };
        let mut checkpoints = entries
            .into_iter()
            .filter_map(|entry| {
                serde_json::from_str::<CheckpointData>(&entry.content)
                    .ok()
                    .map(|checkpoint| (entry.iri, checkpoint.created_at))
            })
            .collect::<Vec<_>>();
        checkpoints.sort_by_key(|(_, created_at)| *created_at);
        let remove_count = checkpoints.len().saturating_sub(MAX_CHECKPOINTS_PER_TASK);
        for (iri, _) in checkpoints.into_iter().take(remove_count) {
            let _ = l0.delete(&iri);
            let mut task_cps = self.task_checkpoints.write();
            for iris in task_cps.values_mut() {
                iris.retain(|candidate| candidate != &iri);
            }
        }
    }
}

impl TaskResumeState {
    #[allow(clippy::too_many_arguments)]
    fn from_fields(
        checkpoint_name: &str,
        agent_state_json: &str,
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
    ) -> Self {
        let state = serde_json::from_str::<serde_json::Value>(agent_state_json).ok();
        let turn = state
            .as_ref()
            .and_then(|value| value.get("turn"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            .min(u64::from(u32::MAX)) as u32;
        let tool_call_count = state
            .as_ref()
            .and_then(|value| value.get("tc"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            .min(u64::from(u32::MAX)) as u32;
        Self {
            schema_version: TASK_RESUME_STATE_SCHEMA_VERSION,
            checkpoint_name: checkpoint_name.to_string(),
            turn,
            tool_call_count,
            current_role: current_role.map(str::to_string),
            prev_summary: prev_summary.map(str::to_string),
            five_w2h_json: five_w2h_json.map(str::to_string),
            cycle_state_json: cycle_state_json.map(str::to_string),
            completed_nodes_json: completed_nodes_json.map(str::to_string),
            pending_approvals_json: pending_approvals_json.map(str::to_string),
            supplement_json: supplement_json.map(str::to_string),
            tool_error_json: tool_error_json.map(str::to_string),
            action_tracker_json: action_tracker_json.map(str::to_string),
            perception_anomaly_json: perception_anomaly_json.map(str::to_string),
        }
    }

    fn validate(&self) -> Result<(), CoreError> {
        if self.schema_version != TASK_RESUME_STATE_SCHEMA_VERSION {
            return Err(CoreError::Internal {
                message: format!(
                    "Unsupported task resume schema version {}",
                    self.schema_version
                ),
            });
        }
        if self.checkpoint_name.is_empty() {
            return Err(CoreError::Internal {
                message: "Task resume state has no checkpoint name".to_string(),
            });
        }
        Ok(())
    }
}

impl CheckpointData {
    fn legacy_resume_state(&self) -> TaskResumeState {
        TaskResumeState::from_fields(
            &self.name,
            &self.agent_state_json,
            self.current_role.as_deref(),
            self.five_w2h_json.as_deref(),
            self.prev_summary.as_deref(),
            self.cycle_state_json.as_deref(),
            self.completed_nodes_json.as_deref(),
            self.pending_approvals_json.as_deref(),
            self.supplement_json.as_deref(),
            self.tool_error_json.as_deref(),
            self.action_tracker_json.as_deref(),
            self.perception_anomaly_json.as_deref(),
        )
    }

    pub fn to_restored_task(&self) -> Result<RestoredTask, CoreError> {
        if self.task_iri.is_empty() || self.checkpoint_iri.is_empty() {
            return Err(CoreError::Internal {
                message: "Checkpoint is missing task or checkpoint IRI".to_string(),
            });
        }
        let messages = serde_json::from_str::<Vec<ChatMessage>>(&self.session_messages_json)
            .map_err(|error| CoreError::Internal {
                message: format!("Invalid checkpoint session messages: {error}"),
            })?;
        let state = self
            .resume_state
            .clone()
            .unwrap_or_else(|| self.legacy_resume_state());
        state.validate()?;
        if state.checkpoint_name != self.name {
            return Err(CoreError::Internal {
                message: "Checkpoint resume state does not match checkpoint name".to_string(),
            });
        }
        Ok(RestoredTask {
            checkpoint: self.clone(),
            messages,
            state,
        })
    }
}

impl CheckpointManager {
    fn deserialize_checkpoint(content: &str) -> Result<CheckpointData, CoreError> {
        let checkpoint = serde_json::from_str::<CheckpointData>(content).map_err(|error| {
            CoreError::Internal {
                message: format!("Invalid checkpoint data: {error}"),
            }
        })?;
        // Validate the structural invariant here while accepting legacy state
        // migration.  Session message validation happens in restore_task so a
        // caller can still inspect older metadata-only checkpoint records.
        if checkpoint.task_iri.is_empty() || checkpoint.checkpoint_iri.is_empty() {
            return Err(CoreError::Internal {
                message: "Checkpoint is missing task or checkpoint IRI".to_string(),
            });
        }
        if let Some(state) = &checkpoint.resume_state {
            state.validate()?;
        }
        Ok(checkpoint)
    }
}

/// Parse the phase label from a checkpoint name.
/// Examples: "start_DA" → "start_DA", "turn_CA_5" → "turn_CA_5", "finish_PA" → "finish_PA"
///       "step_complete_Do" → "step_complete_Do", "unknown_xxx" → "unknown"
pub fn parse_checkpoint_phase(name: &str) -> String {
    let known_prefixes = [
        "start_",
        "turn_",
        "finish_",
        "max_turns_",
        "force_end_",
        "step_complete_",
        "pre_dispatch_",
        "plan_created_",
    ];
    for prefix in &known_prefixes {
        if name.starts_with(prefix) {
            // Extract the role portion: "start_DA" → extract "DA" portion as phase
            // For turn_N_Role format: "turn_DA_5" → extract role between prefix and last _
            let rest = name.strip_prefix(prefix).unwrap_or("");
            if *prefix == "turn_" {
                // "turn_DA_5" → split by _, take first part
                if let Some(role) = rest.split('_').next() {
                    if matches!(
                        role,
                        "PA" | "DA" | "CA" | "AA" | "Plan" | "Do" | "Check" | "Act"
                    ) {
                        return format!("turn_{}", role);
                    }
                }
                return format!("turn_{}", rest);
            }
            return name.to_string();
        }
    }
    "unknown".to_string()
}

/// Infer which AgentRoles to skip during resume based on checkpoint name and current_role.
/// Returns a list of role strings, e.g. ["Plan", "Do"].
/// Rules:
///   - "start_<Role>" / "turn_<Role>_N" → all roles before this one are done, skip them
///   - "finish_<Role>" / "step_complete_<Role>" → this role is done, skip it
///   - If current_role is explicitly specified, it takes precedence
pub fn compute_skip_roles_from_phase(name: &str, current_role: Option<&str>) -> Vec<String> {
    // Role order
    let role_order = ["Plan", "Do", "Check", "Act"];
    let _alt_roles = ["PA", "DA", "CA", "AA"];
    let all_roles = ["Plan", "Do", "Check", "Act", "PA", "DA", "CA", "AA"];

    // Prefer current_role
    let active_role = current_role
        .and_then(|r| all_roles.iter().find(|ar| ar.eq_ignore_ascii_case(r)))
        .copied();

    // Extract role from name
    let name_role = {
        let mut found = None;
        for role in &all_roles {
            if name.contains(role) {
                found = Some(*role);
                break;
            }
        }
        found
    };

    let target_role = active_role.or(name_role);

    if let Some(role) = target_role {
        // Normalize role to canonical name
        let canonical = match role {
            "PA" => "Plan",
            "DA" => "Do",
            "CA" => "Check",
            "AA" => "Act",
            r => r,
        };

        let is_finish = name.starts_with("finish_") || name.starts_with("step_complete_");

        let mut skip = Vec::new();
        for r in &role_order {
            if *r == canonical {
                if is_finish {
                    skip.push(r.to_string());
                }
                break;
            }
            skip.push(r.to_string());
        }
        return skip;
    }

    // fallback: only skip Plan (backward compatible)
    vec!["Plan".to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_in_memory() {
        let manager = CheckpointManager::new();
        let checkpoint = manager
            .create(
                "iri://task/123",
                "test",
                r#"[{"@id":"iri://node/1"}]"#,
                r#"[{"role":"user"}]"#,
                r#"{"status":"running"}"#,
                &["important".to_string()],
            )
            .unwrap();
        assert!(checkpoint.checkpoint_iri.starts_with("iri://checkpoint/"));
        assert_eq!(checkpoint.task_iri, "iri://task/123");
    }

    #[test]
    fn test_list_empty() {
        let manager = CheckpointManager::new();
        let list = manager.list("iri://task/nonexistent", 10);
        assert!(list.is_empty());
    }

    #[test]
    fn test_list_via_l0_scan_cross_process() {
        use crate::memory::l0_store::L0Store;
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mgr = CheckpointManager::with_persistence(l0.clone());

        // Create checkpoint (simulating running in a previous process)
        mgr.create(
            "iri://task/abc-123",
            "finish_DA",
            "[]",
            r#"[{"role":"user","content":"hello"}]"#,
            r#"{"turn":3}"#,
            &["DA".to_string()],
        )
        .unwrap();

        // New CheckpointManager (simulating cross-process: new instance, empty memory index)
        let mgr2 = CheckpointManager::with_persistence(l0.clone());

        // restore_latest must find the checkpoint (fallback via scan_iri_prefix)
        let cp = mgr2.restore_latest("iri://task/abc-123").unwrap();
        assert!(cp.is_some(), "cross-process recovery must find checkpoint");
        assert_eq!(cp.unwrap().task_iri, "iri://task/abc-123");

        // list must also find it
        let list = mgr2.list("iri://task/abc-123", 10);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "finish_DA");
    }

    #[test]
    fn restore_task_uses_versioned_state_and_skips_a_corrupt_newer_checkpoint() {
        use crate::memory::l0_store::L0Store;
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let manager = CheckpointManager::with_persistence(l0.clone());
        let checkpoint = manager
            .create_ext(
                "iri://task/resume-fallback",
                "turn_Do_7",
                "[]",
                r#"[{"role":"system","content":"s"},{"role":"assistant","content":"done"}]"#,
                r#"{"turn":7,"tc":3}"#,
                &["Do".to_string()],
                Some("Do"),
                None,
                Some("plan summary"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

        let mut corrupt = checkpoint.clone();
        corrupt.checkpoint_iri = "iri://checkpoint/task/resume-fallback/corrupt".to_string();
        corrupt.created_at += chrono::Duration::seconds(1);
        corrupt.session_messages_json = "{not valid json".to_string();
        l0.store(
            &corrupt.checkpoint_iri,
            &serde_json::to_string(&corrupt).unwrap(),
        )
        .unwrap();

        let restored = CheckpointManager::with_persistence(l0)
            .restore_task("iri://task/resume-fallback")
            .unwrap()
            .unwrap();
        assert_eq!(
            restored.checkpoint.checkpoint_iri,
            checkpoint.checkpoint_iri
        );
        assert_eq!(
            restored.state.schema_version,
            TASK_RESUME_STATE_SCHEMA_VERSION
        );
        assert_eq!(restored.state.turn, 7);
        assert_eq!(restored.state.tool_call_count, 3);
        assert_eq!(restored.state.current_role.as_deref(), Some("Do"));
        assert_eq!(restored.state.prev_summary.as_deref(), Some("plan summary"));
    }

    #[test]
    fn test_checkpoint_retention_prunes_oldest() {
        use crate::memory::l0_store::L0Store;
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mgr = CheckpointManager::with_persistence(l0.clone());

        // Create enough checkpoints to exceed the per-task retention cap
        let total = MAX_CHECKPOINTS_PER_TASK + 5;
        for i in 0..total {
            mgr.create(
                "iri://task/retention",
                &format!("turn_Plan_{}", i),
                "[]",
                r#"[{"role":"user","content":"hello"}]"#,
                r#"{"turn":1}"#,
                &["Plan".to_string()],
            )
            .unwrap();
        }

        // In-memory index is bounded
        assert_eq!(mgr.checkpoint_count() as usize, MAX_CHECKPOINTS_PER_TASK);

        // Oldest entry physically evicted from L0
        let persisted = l0
            .scan_iri_prefix("iri://checkpoint/task/retention/", 100)
            .unwrap();
        assert_eq!(persisted.len(), MAX_CHECKPOINTS_PER_TASK);
        assert!(persisted.iter().all(|entry| !entry.iri.contains("/seq_0_")));

        // Latest entry still present and restorable.
        let cp = mgr.restore_latest("iri://task/retention").unwrap();
        assert_eq!(cp.unwrap().name, format!("turn_Plan_{}", total - 1));
    }

    #[test]
    fn separate_managers_do_not_overwrite_the_same_task_checkpoint() {
        use crate::memory::l0_store::L0Store;
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let first = CheckpointManager::with_persistence(l0.clone())
            .create("iri://task/shared", "finish_DA", "[]", "[]", "{}", &[])
            .unwrap();
        let second = CheckpointManager::with_persistence(l0.clone())
            .create("iri://task/shared", "finish_CA", "[]", "[]", "{}", &[])
            .unwrap();

        assert_ne!(first.checkpoint_iri, second.checkpoint_iri);
        assert_eq!(
            l0.scan_iri_prefix("iri://checkpoint/task/shared/", 10)
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn test_parse_checkpoint_phase() {
        assert_eq!(parse_checkpoint_phase("start_DA"), "start_DA");
        assert_eq!(parse_checkpoint_phase("turn_DA_5"), "turn_DA");
        assert_eq!(parse_checkpoint_phase("finish_CA"), "finish_CA");
        assert_eq!(
            parse_checkpoint_phase("step_complete_Do"),
            "step_complete_Do"
        );
        assert_eq!(parse_checkpoint_phase("max_turns_Plan"), "max_turns_Plan");
        assert_eq!(parse_checkpoint_phase("force_end_Act"), "force_end_Act");
        assert_eq!(parse_checkpoint_phase("unknown_xxx"), "unknown");
    }

    #[test]
    fn test_compute_skip_roles_no_current_role() {
        // finish_DA → Plan and Do are done, DA itself is done → skip Plan + Do + DA
        let roles = compute_skip_roles_from_phase("finish_DA", None);
        assert!(roles.contains(&"Plan".to_string()));
        assert!(roles.contains(&"Do".to_string()));
        assert!(!roles.contains(&"Check".to_string()));

        // start_DA → Plan is done (DA is starting), skip Plan only
        let roles = compute_skip_roles_from_phase("start_DA", None);
        assert!(roles.contains(&"Plan".to_string()));
        assert!(!roles.contains(&"Do".to_string()));

        // step_complete_CA → Plan, Do, Check all done
        let roles = compute_skip_roles_from_phase("step_complete_CA", None);
        assert!(roles.contains(&"Plan".to_string()));
        assert!(roles.contains(&"Do".to_string()));
        assert!(roles.contains(&"Check".to_string()));
        assert!(!roles.contains(&"Act".to_string()));

        // turn_DA_5 → Plan is done (DA in progress), skip Plan only
        let roles = compute_skip_roles_from_phase("turn_DA_5", None);
        assert!(roles.contains(&"Plan".to_string()));
        assert!(!roles.contains(&"Do".to_string()));
    }

    #[test]
    fn test_compute_skip_roles_with_current_role() {
        // current_role overrides name
        let roles = compute_skip_roles_from_phase("start_DA", Some("Check"));
        assert!(roles.contains(&"Plan".to_string()));
        assert!(roles.contains(&"Do".to_string()));
        assert!(!roles.contains(&"Check".to_string()));

        // finish_Check with current_role=Check → skip Plan, Do, Check
        let roles = compute_skip_roles_from_phase("finish_Check", Some("Check"));
        assert!(roles.contains(&"Plan".to_string()));
        assert!(roles.contains(&"Do".to_string()));
        assert!(roles.contains(&"Check".to_string()));
        assert!(!roles.contains(&"Act".to_string()));
    }

    #[test]
    fn test_create_ext_roundtrip() {
        let manager = CheckpointManager::new();
        let cp = manager
            .create_ext(
                "iri://task/roundtrip",
                "step_complete_DA",
                "[]",
                "[]",
                r#"{"turn":5}"#,
                &["DA".to_string(), "step_complete".to_string()],
                Some("DA"),
                Some(r#"{"what":"test"}"#),
                Some("prev summary here"),
                Some(r#"{"phase":"Executing"}"#),
                Some(r#"{"node1":{"status":"ok"}}"#),
                Some(r#"{"approval1":true}"#),
                None,
                Some(r#"{"bash":3}"#),
                Some(r#"[]"#),
                None,
            )
            .unwrap();

        assert_eq!(cp.name, "step_complete_DA");
        assert_eq!(cp.current_role.as_deref(), Some("DA"));
        assert_eq!(cp.prev_summary.as_deref(), Some("prev summary here"));
        assert_eq!(cp.tool_error_json.as_deref(), Some(r#"{"bash":3}"#));
    }
}
