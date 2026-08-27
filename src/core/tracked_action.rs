use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Normalize the heterogeneous failure signals returned by built-in and MCP
/// tools.  Command tools report a non-zero `exit_code` without an `error`
/// field, while other tools may use `success=false` or `timed_out=true`.
pub fn tool_result_failed(result: &Value) -> bool {
    result.get("error").is_some()
        || result
            .get("exit_code")
            .and_then(Value::as_i64)
            .is_some_and(|code| code != 0)
        || result.get("success").and_then(Value::as_bool) == Some(false)
        || result.get("timed_out").and_then(Value::as_bool) == Some(true)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub path: String,
    pub size_bytes: Option<u64>,
    pub hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActionStatus {
    Success,
    Failed,
    Retried,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedAction {
    pub action_id: String,
    pub tool_name: String,
    pub agent_role: String,
    pub duration_secs: f64,
    pub status: ActionStatus,
    pub files_created: Vec<FileChange>,
    pub files_modified: Vec<FileChange>,
    pub files_read: Vec<String>,
    pub error: Option<String>,
    /// Confirmed semantic effect, independent of command syntax. This is set
    /// directly for changed file tools and after before/after confirmation for
    /// shell-like tools.
    #[serde(default)]
    pub substantive_effect: bool,
    #[serde(default)]
    pub tool_args: HashMap<String, Value>,
}

pub struct ActionTracker {
    pub actions: Vec<TrackedAction>,
    pub task_iri: String,
    pub agent_role: String,
    pub started_at: DateTime<Utc>,
}

impl ActionTracker {
    pub fn new(task_iri: &str, agent_role: &str) -> Self {
        Self {
            actions: Vec::new(),
            task_iri: task_iri.to_string(),
            agent_role: agent_role.to_string(),
            started_at: Utc::now(),
        }
    }

    pub fn record(&mut self, tool_name: &str, args: &Value, result: &Value, duration_secs: f64) {
        let mut action = TrackedAction {
            action_id: format!("act_{}", uuid::Uuid::new_v4().hyphenated()),
            tool_name: tool_name.to_string(),
            agent_role: self.agent_role.clone(),
            duration_secs,
            status: if tool_result_failed(result) {
                ActionStatus::Failed
            } else {
                ActionStatus::Success
            },
            files_created: vec![],
            files_modified: vec![],
            files_read: vec![],
            error: result
                .get("error")
                .and_then(|e| e.as_str())
                .map(String::from),
            substantive_effect: false,
            tool_args: HashMap::new(),
        };

        match tool_name {
            "file_write" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    action
                        .tool_args
                        .insert("path".to_string(), Value::String(path.to_string()));
                    if action.status == ActionStatus::Success
                        && result.get("changed").and_then(Value::as_bool) == Some(true)
                    {
                        let change = FileChange {
                            path: path.to_string(),
                            size_bytes: args
                                .get("content")
                                .and_then(|v| v.as_str())
                                .map(|c| c.len() as u64),
                            hash: None,
                        };
                        if result.get("created").and_then(Value::as_bool) == Some(true) {
                            action.files_created.push(change);
                        } else {
                            action.files_modified.push(change);
                        }
                        action.substantive_effect = true;
                    }
                }
            }
            "file_edit" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    action
                        .tool_args
                        .insert("path".to_string(), Value::String(path.to_string()));
                    if action.status == ActionStatus::Success
                        && result.get("changed").and_then(Value::as_bool) == Some(true)
                    {
                        action.files_modified.push(FileChange {
                            path: path.to_string(),
                            size_bytes: None,
                            hash: None,
                        });
                        action.substantive_effect = true;
                    }
                }
            }
            "file_read" => {
                if action.status == ActionStatus::Success {
                    if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                        action.files_read.push(path.to_string());
                    }
                }
            }
            "bash" | "powershell" => {
                if !tool_result_failed(result) {
                    action.tool_args.insert(
                        "command".to_string(),
                        args.get("command").cloned().unwrap_or_default(),
                    );
                }
            }
            _ => {}
        }

        self.actions.push(action);
    }

    pub fn mark_last_substantive_effect(&mut self) {
        if let Some(action) = self.actions.last_mut() {
            action.substantive_effect = true;
        }
    }

    pub fn success_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| a.status == ActionStatus::Success)
            .count()
    }

    pub fn failure_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| a.status == ActionStatus::Failed)
            .count()
    }

    pub fn files_created_all(&self) -> Vec<&FileChange> {
        self.actions.iter().flat_map(|a| &a.files_created).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn file_edit_uses_the_builtin_path_argument() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "file_edit",
            &json!({"path": "/workspace/app.rs"}),
            &json!({"success": true, "changed": true}),
            0.01,
        );

        assert_eq!(tracker.actions[0].files_modified.len(), 1);
        assert_eq!(
            tracker.actions[0].files_modified[0].path,
            "/workspace/app.rs"
        );
        assert_eq!(
            tracker.actions[0].tool_args.get("path"),
            Some(&Value::String("/workspace/app.rs".to_string()))
        );
    }

    #[test]
    fn failed_writes_are_not_recorded_as_file_changes() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "file_write",
            &json!({"path": "/workspace/app.rs", "content": "invalid"}),
            &json!({"error": "permission denied"}),
            0.01,
        );

        assert_eq!(tracker.actions[0].status, ActionStatus::Failed);
        assert!(tracker.actions[0].files_created.is_empty());
    }

    #[test]
    fn nonzero_command_exit_is_a_failed_action() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "bash",
            &json!({"command": "run acceptance checks"}),
            &json!({"exit_code": 1, "stderr": "failed"}),
            0.01,
        );

        assert_eq!(tracker.actions[0].status, ActionStatus::Failed);
    }

    #[test]
    fn no_op_file_tool_is_not_a_substantive_action() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "file_write",
            &json!({"path": "/workspace/app.rs", "content": "same"}),
            &json!({"success": true, "changed": false, "created": false}),
            0.01,
        );

        assert!(!tracker.actions[0].substantive_effect);
        assert!(tracker.actions[0].files_created.is_empty());
        assert!(tracker.actions[0].files_modified.is_empty());
    }

    #[test]
    fn confirmed_shell_effect_can_be_recorded_after_execution() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "bash",
            &json!({"command": "generator"}),
            &json!({"exit_code": 0}),
            0.01,
        );
        tracker.mark_last_substantive_effect();
        assert!(tracker.actions[0].substantive_effect);
    }
}
