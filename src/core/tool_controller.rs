use serde_json::Value;
use tracing::warn;

use crate::core::agent_instance::AgentRole;
use crate::tools::ToolExecutor;

/// Kernel capability ceilings for real business-agent roles.
///
/// `None` means that the role has no additional role-level ceiling (DA still
/// remains subject to task, skill-security, and sandbox policy). `Some([])` is
/// deliberately different: AA is a decision-only BizAgent and has no tools.
pub const PA_TOOL_CEILING: &[&str] = &[
    "file_read",
    "file_list",
    "grep_search",
    "glob_search",
    "tool_search",
    "web_search",
    "web_fetch",
    "rag_search",
    "kg_search",
    "codebase_search",
    "knowledge_list",
    "knowledge_search",
    "knowledge_query",
    "knowledge_neighbors",
    "read_agent_output",
];

pub const CA_TOOL_CEILING: &[&str] = &[
    "file_read",
    "file_list",
    "grep_search",
    "glob_search",
    "bash",
    "tool_search",
    "jsonld_validate",
    "rag_search",
    "kg_search",
    "codebase_search",
    "knowledge_list",
    "knowledge_search",
    "knowledge_query",
    "knowledge_neighbors",
    "read_agent_output",
    "ontology_validate_turtle",
    "ontology_validate_shacl",
    "ontology_lint_turtle",
    "ontology_diff_turtle",
    "ontology_reason",
];

pub fn business_role_tool_ceiling(role: AgentRole) -> Option<&'static [&'static str]> {
    match role {
        AgentRole::Plan => Some(PA_TOOL_CEILING),
        AgentRole::Do => None,
        AgentRole::Check => Some(CA_TOOL_CEILING),
        AgentRole::Act => Some(&[]),
    }
}

pub fn business_role_allows_tool(role: AgentRole, tool_name: &str) -> bool {
    business_role_tool_ceiling(role)
        .map(|ceiling| ceiling.contains(&tool_name))
        .unwrap_or(true)
}

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
            || tool_name == "read_agent_output"
            || ToolExecutor::is_micro_tool_name(tool_name)
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
            role => {
                let denied: Vec<String> = tool_calls
                    .iter()
                    .filter(|(name, _)| !business_role_allows_tool(*role, name))
                    .map(|(name, _)| name.clone())
                    .collect();
                if !denied.is_empty() {
                    warn!(role = %role, tools = ?denied, "Role capability ceiling filtered tool calls");
                }
                tool_calls
                    .iter()
                    .filter(|(name, _)| business_role_allows_tool(*role, name))
                    .cloned()
                    .collect()
            }
        }
    }

    pub fn should_force_finish(&self, tool_calls: &[(String, Value)], role: &AgentRole) -> bool {
        match role {
            AgentRole::Plan => tool_calls
                .iter()
                .any(|(name, _)| !self.is_readonly_tool(name)),
            role => tool_calls
                .iter()
                .any(|(name, _)| !business_role_allows_tool(*role, name)),
        }
    }

    pub fn list_available_tools(&self, role: &AgentRole) -> Vec<String> {
        match role {
            AgentRole::Plan => self.readonly_tools.iter().map(|s| s.to_string()).collect(),
            AgentRole::Do => {
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
            role => business_role_tool_ceiling(*role)
                .unwrap_or_default()
                .iter()
                .map(|tool| (*tool).to_string())
                .collect(),
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
        let agent_output = vec![("read_agent_output".to_string(), Value::Null)];
        assert!(!tc.should_force_finish(&agent_output, &AgentRole::Plan));
        let micro_tool = vec![("read_full_result_call_session_a".to_string(), Value::Null)];
        assert!(!tc.should_force_finish(&micro_tool, &AgentRole::Plan));
        let unknown = vec![("unregistered_dynamic_tool".to_string(), Value::Null)];
        assert!(tc.should_force_finish(&unknown, &AgentRole::Plan));
    }

    #[test]
    fn test_list_available_tools() {
        let tc = ToolController::new();
        let plan_tools = tc.list_available_tools(&AgentRole::Plan);
        assert!(plan_tools.contains(&"file_read".to_string()));
        assert!(!plan_tools.contains(&"file_write".to_string()));
        let do_tools = tc.list_available_tools(&AgentRole::Do);
        assert!(do_tools.contains(&"file_write".to_string()));
        let check_tools = tc.list_available_tools(&AgentRole::Check);
        assert!(check_tools.contains(&"bash".to_string()));
        assert!(!check_tools.contains(&"file_write".to_string()));
        assert!(tc.list_available_tools(&AgentRole::Act).is_empty());
    }

    #[test]
    fn test_act_is_decision_only() {
        let tc = ToolController::new();
        let calls = vec![("file_read".to_string(), Value::Null)];
        assert!(tc.filter_tools_for_role(&calls, &AgentRole::Act).is_empty());
        assert!(tc.should_force_finish(&calls, &AgentRole::Act));
    }
}
