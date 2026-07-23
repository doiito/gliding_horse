use serde_json::Value;
use tracing::warn;

use crate::core::agent_instance::AgentRole;

#[derive(Clone)]
pub struct ToolController {
    readonly_tools: Vec<&'static str>,
    write_tools: Vec<&'static str>,
}

impl ToolController {
    pub fn new() -> Self {
        Self {
            readonly_tools: vec![
                "file_read",
                "file_list",
                "grep_search",
                "glob_search",
                "tool_search",
                "web_search",
                "web_fetch",
                "rag_search",
            ],
            write_tools: vec![
                "file_write",
                "bash",
                "code_execute",
                "http_request",
                "rag_index",
                "rag_chunk",
            ],
        }
    }

    pub fn is_readonly_tool(&self, tool_name: &str) -> bool {
        self.readonly_tools.contains(&tool_name)
    }

    pub fn is_write_tool(&self, tool_name: &str) -> bool {
        self.write_tools.contains(&tool_name)
    }

    pub fn filter_tools_for_role(
        &self,
        tool_calls: &[(String, Value)],
        role: &AgentRole,
    ) -> Vec<(String, Value)> {
        match role {
            AgentRole::Plan => {
                let write_calls: Vec<String> = tool_calls
                    .iter()
                    .filter(|(name, _)| self.is_write_tool(name))
                    .map(|(name, _)| name.clone())
                    .collect();
                if !write_calls.is_empty() {
                    warn!(
                        "[PA] Detected write tool calls: {:?}, filtered",
                        write_calls
                    );
                }
                tool_calls
                    .iter()
                    .filter(|(name, _)| self.is_readonly_tool(name))
                    .cloned()
                    .collect()
            }
            AgentRole::Do | AgentRole::Check | AgentRole::Act => tool_calls.to_vec(),
        }
    }

    pub fn should_force_finish(&self, tool_calls: &[(String, Value)], role: &AgentRole) -> bool {
        match role {
            AgentRole::Plan => tool_calls.iter().any(|(name, _)| self.is_write_tool(name)),
            AgentRole::Act => false,
            AgentRole::Do => false,
            AgentRole::Check => false,
        }
    }

    pub fn list_available_tools(&self, role: &AgentRole) -> Vec<String> {
        match role {
            AgentRole::Plan => self.readonly_tools.iter().map(|s| s.to_string()).collect(),
            AgentRole::Do | AgentRole::Check | AgentRole::Act => {
                let mut tools: Vec<String> = self
                    .readonly_tools
                    .iter()
                    .chain(self.write_tools.iter())
                    .map(|s| s.to_string())
                    .collect();
                tools.sort();
                tools.dedup();
                tools
            }
        }
    }
}

impl Default for ToolController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readonly_tools() {
        let tc = ToolController::new();
        assert!(tc.is_readonly_tool("file_read"));
        assert!(tc.is_readonly_tool("grep_search"));
        assert!(!tc.is_readonly_tool("file_write"));
        assert!(!tc.is_readonly_tool("bash"));
    }

    #[test]
    fn test_write_tools() {
        let tc = ToolController::new();
        assert!(tc.is_write_tool("file_write"));
        assert!(tc.is_write_tool("bash"));
        assert!(!tc.is_write_tool("file_read"));
    }

    #[test]
    fn test_filter_tools_for_plan() {
        let tc = ToolController::new();
        let calls = vec![
            ("file_read".to_string(), Value::String("test".to_string())),
            ("file_write".to_string(), Value::String("test".to_string())),
            ("bash".to_string(), Value::String("test".to_string())),
            ("grep_search".to_string(), Value::String("test".to_string())),
        ];
        let filtered = tc.filter_tools_for_role(&calls, &AgentRole::Plan);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, "file_read");
        assert_eq!(filtered[1].0, "grep_search");
    }

    #[test]
    fn test_filter_tools_for_do() {
        let tc = ToolController::new();
        let calls = vec![
            ("file_read".to_string(), Value::String("test".to_string())),
            ("file_write".to_string(), Value::String("test".to_string())),
        ];
        let filtered = tc.filter_tools_for_role(&calls, &AgentRole::Do);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_should_force_finish_plan() {
        let tc = ToolController::new();
        let calls = vec![("file_write".to_string(), Value::Null)];
        assert!(tc.should_force_finish(&calls, &AgentRole::Plan));
        let calls2 = vec![("file_read".to_string(), Value::Null)];
        assert!(!tc.should_force_finish(&calls2, &AgentRole::Plan));
    }

    #[test]
    fn test_list_available_tools() {
        let tc = ToolController::new();
        let plan_tools = tc.list_available_tools(&AgentRole::Plan);
        assert!(plan_tools.contains(&"file_read".to_string()));
        assert!(!plan_tools.contains(&"file_write".to_string()));
        let do_tools = tc.list_available_tools(&AgentRole::Do);
        assert!(do_tools.contains(&"file_write".to_string()));
    }
}
