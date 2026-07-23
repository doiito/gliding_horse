use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::memory::l0_store::L0Store;
use crate::CoreError;

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
}

pub struct CheckpointManager {
    l0: Option<Arc<L0Store>>,
    task_checkpoints: RwLock<HashMap<String, Vec<String>>>,
    counter: AtomicU64,
}

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
            "iri://checkpoint/{}/seq_{}",
            task_iri.strip_prefix("iri://").unwrap_or(task_iri),
            seq
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
            "iri://checkpoint/{}/seq_{}",
            task_iri.strip_prefix("iri://").unwrap_or(task_iri),
            seq
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
                return serde_json::from_str(&entry.content).map_err(|e| CoreError::Internal {
                    message: format!("Invalid checkpoint data: {}", e),
                });
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
        let cp = self.restore_latest(task_iri)?;
        Ok(cp.map(|c| {
            let skip_roles = compute_skip_roles_from_phase(&c.name, c.current_role.as_deref());
            (c, skip_roles)
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
                                .and_then(|e| serde_json::from_str(&e.content).ok())
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
            if let Ok(entries) = l0.scan_iri_prefix(&prefix, 100) {
                let mut results: Vec<CheckpointData> = entries
                    .iter()
                    .filter_map(|e| serde_json::from_str(&e.content).ok())
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
