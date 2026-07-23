use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolGroup {
    Core,
    Write,
    Search,
    Web,
    Knowledge,
    Code,
    Skill,
    System,
}

impl std::fmt::Display for ToolGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolGroup::Core => write!(f, "Core"),
            ToolGroup::Write => write!(f, "Write"),
            ToolGroup::Search => write!(f, "Search"),
            ToolGroup::Web => write!(f, "Web"),
            ToolGroup::Knowledge => write!(f, "Knowledge"),
            ToolGroup::Code => write!(f, "Code"),
            ToolGroup::Skill => write!(f, "Skill"),
            ToolGroup::System => write!(f, "System"),
        }
    }
}

impl std::str::FromStr for ToolGroup {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "core" => Ok(ToolGroup::Core),
            "write" => Ok(ToolGroup::Write),
            "search" => Ok(ToolGroup::Search),
            "web" => Ok(ToolGroup::Web),
            "knowledge" => Ok(ToolGroup::Knowledge),
            "code" => Ok(ToolGroup::Code),
            "skill" => Ok(ToolGroup::Skill),
            "system" => Ok(ToolGroup::System),
            _ => Err(format!("Unknown tool group: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleToolConfig {
    #[serde(default)]
    pub default: Vec<String>,
    #[serde(default)]
    pub on_demand: Vec<String>,
}

impl Default for RoleToolConfig {
    fn default() -> Self {
        Self {
            default: vec![],
            on_demand: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolGroupSettings {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub roles: HashMap<String, RoleToolConfig>,
}

fn default_true() -> bool {
    true
}

impl Default for ToolGroupSettings {
    fn default() -> Self {
        let mut roles = HashMap::new();

        roles.insert(
            "Plan".to_string(),
            RoleToolConfig {
                default: vec![
                    "Core".to_string(),
                    "Search".to_string(),
                    "Knowledge".to_string(),
                    "System".to_string(),
                ],
                on_demand: vec!["Web".to_string(), "Code".to_string(), "Skill".to_string()],
            },
        );

        roles.insert(
            "Do".to_string(),
            RoleToolConfig {
                default: vec![
                    "Core".to_string(),
                    "Write".to_string(),
                    "Search".to_string(),
                    "Web".to_string(),
                    "Code".to_string(),
                    "Skill".to_string(),
                    "System".to_string(),
                ],
                on_demand: vec!["Knowledge".to_string()],
            },
        );

        roles.insert(
            "Check".to_string(),
            RoleToolConfig {
                default: vec![
                    "Core".to_string(),
                    "Search".to_string(),
                    "Knowledge".to_string(),
                    "System".to_string(),
                ],
                on_demand: vec!["Web".to_string(), "Code".to_string()],
            },
        );

        roles.insert(
            "Act".to_string(),
            RoleToolConfig {
                default: vec!["Core".to_string(), "System".to_string()],
                on_demand: vec!["Search".to_string(), "Knowledge".to_string()],
            },
        );

        Self {
            enabled: true,
            roles,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolGroupManager {
    settings: ToolGroupSettings,
    group_tools: HashMap<ToolGroup, HashSet<String>>,
}

impl ToolGroupManager {
    pub fn new(settings: Option<ToolGroupSettings>) -> Self {
        let settings = settings.unwrap_or_default();
        let group_tools = Self::build_group_tools();

        Self {
            settings,
            group_tools,
        }
    }

    fn build_group_tools() -> HashMap<ToolGroup, HashSet<String>> {
        let mut map = HashMap::new();

        map.insert(
            ToolGroup::Core,
            HashSet::from(["file_read".to_string(), "file_list".to_string()]),
        );

        map.insert(
            ToolGroup::Write,
            HashSet::from([
                "file_write".to_string(),
                "file_delete".to_string(),
                "bash".to_string(),
                "powershell".to_string(),
            ]),
        );

        map.insert(
            ToolGroup::Web,
            HashSet::from(["web_search".to_string(), "web_fetch".to_string()]),
        );

        map.insert(
            ToolGroup::Search,
            HashSet::from([
                "grep_search".to_string(),
                "glob_search".to_string(),
                "rag_search".to_string(),
                "kg_search".to_string(),
                "codebase_search".to_string(),
            ]),
        );

        map.insert(
            ToolGroup::Knowledge,
            HashSet::from([
                "knowledge_query".to_string(),
                "knowledge_add".to_string(),
                "knowledge_update".to_string(),
                "knowledge_delete".to_string(),
                "kg_query".to_string(),
                "kg_add".to_string(),
                "kg_update".to_string(),
                "kg_delete".to_string(),
                "knowledge_list".to_string(),
                "knowledge_search".to_string(),
            ]),
        );

        map.insert(
            ToolGroup::Code,
            HashSet::from([
                "knowledge_extract_code".to_string(),
                "code_analyze".to_string(),
            ]),
        );

        map.insert(
            ToolGroup::Skill,
            HashSet::from([
                "create_skill".to_string(),
                "convert_skill".to_string(),
                "list_skills".to_string(),
                "get_skill".to_string(),
            ]),
        );

        map.insert(
            ToolGroup::System,
            HashSet::from(["tool_search".to_string()]),
        );

        map
    }

    pub fn get_groups_for_role(&self, role: &str) -> (Vec<ToolGroup>, Vec<ToolGroup>) {
        if !self.settings.enabled {
            return (
                vec![
                    ToolGroup::Core,
                    ToolGroup::Search,
                    ToolGroup::Web,
                    ToolGroup::Knowledge,
                    ToolGroup::Code,
                    ToolGroup::Skill,
                    ToolGroup::System,
                ],
                vec![],
            );
        }

        let role_config = self.settings.roles.get(role);

        match role_config {
            Some(config) => {
                let default: Vec<ToolGroup> = config
                    .default
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                let on_demand: Vec<ToolGroup> = config
                    .on_demand
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                (default, on_demand)
            }
            None => (vec![ToolGroup::Core, ToolGroup::System], vec![]),
        }
    }

    pub fn get_tools_for_groups(&self, groups: &[ToolGroup]) -> HashSet<String> {
        let mut tools = HashSet::new();
        for group in groups {
            if let Some(group_tools) = self.group_tools.get(group) {
                tools.extend(group_tools.clone());
            }
        }
        tools
    }

    pub fn get_tool_names_for_role(&self, role: &str) -> (HashSet<String>, HashSet<String>) {
        let (default_groups, on_demand_groups) = self.get_groups_for_role(role);
        let default_tools = self.get_tools_for_groups(&default_groups);
        let on_demand_tools = self.get_tools_for_groups(&on_demand_groups);
        (default_tools, on_demand_tools)
    }

    pub fn build_tool_summary(&self, role: &str, registered_tools: &[String]) -> String {
        let (default_tools, on_demand_tools) = self.get_tool_names_for_role(role);

        let available: Vec<&String> = registered_tools
            .iter()
            .filter(|t| default_tools.contains(*t))
            .collect();

        let on_demand_available: Vec<&String> = registered_tools
            .iter()
            .filter(|t| on_demand_tools.contains(*t))
            .collect();

        let mut summary = String::new();

        if !available.is_empty() {
            summary.push_str("## Default available tools\n");
            for tool in available {
                summary.push_str(&format!("- {}\n", tool));
            }
        }

        if !on_demand_available.is_empty() {
            summary.push_str("\n## On-demand tools (use tool_search to query)\n");
            for tool in on_demand_available {
                summary.push_str(&format!("- {}\n", tool));
            }
        }

        summary
    }

    pub fn is_tool_available_for_role(&self, role: &str, tool_name: &str) -> bool {
        let (default_tools, on_demand_tools) = self.get_tool_names_for_role(role);
        default_tools.contains(tool_name) || on_demand_tools.contains(tool_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_tool_group_from_str() {
        assert_eq!(ToolGroup::from_str("Core").unwrap(), ToolGroup::Core);
        assert_eq!(ToolGroup::from_str("core").unwrap(), ToolGroup::Core);
        assert_eq!(ToolGroup::from_str("SEARCH").unwrap(), ToolGroup::Search);
        assert!(ToolGroup::from_str("Unknown").is_err());
    }

    #[test]
    fn test_default_settings() {
        let settings = ToolGroupSettings::default();
        assert!(settings.enabled);
        assert!(settings.roles.contains_key("Plan"));
        assert!(settings.roles.contains_key("Do"));
        assert!(settings.roles.contains_key("Check"));
        assert!(settings.roles.contains_key("Act"));
    }

    #[test]
    fn test_get_groups_for_plan() {
        let manager = ToolGroupManager::new(None);
        let (default, on_demand) = manager.get_groups_for_role("Plan");

        assert!(default.contains(&ToolGroup::Core));
        assert!(default.contains(&ToolGroup::Search));
        assert!(default.contains(&ToolGroup::Knowledge));
        assert!(default.contains(&ToolGroup::System));
        assert!(!default.contains(&ToolGroup::Web));

        assert!(on_demand.contains(&ToolGroup::Web));
        assert!(on_demand.contains(&ToolGroup::Code));
    }

    #[test]
    fn test_get_tools_for_groups() {
        let manager = ToolGroupManager::new(None);
        let tools = manager.get_tools_for_groups(&[ToolGroup::Core, ToolGroup::Web]);

        assert!(tools.contains("file_read"));
        assert!(!tools.contains("file_write")); // file_write is in Write group
        assert!(tools.contains("web_search"));
        assert!(tools.contains("web_fetch"));
        assert!(!tools.contains("knowledge_query"));
    }

    #[test]
    fn test_write_group() {
        let manager = ToolGroupManager::new(None);
        let tools = manager.get_tools_for_groups(&[ToolGroup::Write]);

        assert!(tools.contains("file_write"));
        assert!(tools.contains("file_delete"));
        assert!(tools.contains("bash"));
        assert!(tools.contains("powershell"));
        assert!(!tools.contains("file_read"));
    }

    #[test]
    fn test_is_tool_available_for_role() {
        let manager = ToolGroupManager::new(None);

        assert!(manager.is_tool_available_for_role("Plan", "file_read"));
        assert!(manager.is_tool_available_for_role("Plan", "web_search"));
        assert!(!manager.is_tool_available_for_role("Plan", "bash"));

        assert!(manager.is_tool_available_for_role("Do", "bash"));
        assert!(manager.is_tool_available_for_role("Do", "web_search"));
    }
}
