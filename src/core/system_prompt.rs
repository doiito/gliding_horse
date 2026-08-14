use std::collections::HashMap;

use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SystemPromptRegion {
    RoleDefinition,
    TimeAwareness,
    EnvironmentInfo,
    BehavioralPolicy,
    FiveW2HConstraints,
    EmphasizedConstraints,
    OutputFormat,
    OutputManagement,
    Tools,
    ExtractionPrompt,
}

impl SystemPromptRegion {
    pub fn order(&self) -> usize {
        match self {
            Self::RoleDefinition => 1,
            Self::TimeAwareness => 2,
            Self::EnvironmentInfo => 3,
            Self::BehavioralPolicy => 4,
            Self::FiveW2HConstraints => 5,
            Self::EmphasizedConstraints => 6,
            Self::OutputFormat => 7,
            Self::OutputManagement => 8,
            Self::Tools => 9,
            Self::ExtractionPrompt => 10,
        }
    }

    pub fn header(&self) -> &'static str {
        match self {
            Self::RoleDefinition => "# Role",
            Self::TimeAwareness => "# Time Awareness",
            Self::EnvironmentInfo => "# Workspace Environment",
            Self::BehavioralPolicy => "# Behavioral Policy",
            Self::FiveW2HConstraints => "# Task Constraints",
            Self::EmphasizedConstraints => "# Critical Constraints",
            Self::OutputFormat => "# Output Format",
            Self::OutputManagement => "# Output Management",
            Self::Tools => "# Tools",
            Self::ExtractionPrompt => "# Emphasized Content",
        }
    }
}

/// Build the Time Awareness section text showing current time and session context
pub fn build_time_awareness_text(task_start_time: Option<&str>) -> String {
    let now = Utc::now();
    let now_str = now.format("%Y-%m-%d %H:%M:%S UTC").to_string();
    let mut parts = vec![
        format!("- Current time: {}", now_str),
        "- All timestamps in the system use UTC timezone".to_string(),
    ];
    if let Some(start) = task_start_time {
        parts.push(format!("- Task started at: {}", start));
        // Calculate elapsed time using DateTime::parse_from_rfc3339, convert to Utc for subtraction
        if let Ok(start_dt) = DateTime::parse_from_rfc3339(start) {
            let start_utc = start_dt.with_timezone(&Utc);
            let elapsed = now - start_utc;
            let hours = elapsed.num_hours();
            let mins = elapsed.num_minutes() % 60;
            let secs = elapsed.num_seconds() % 60;
            if hours > 0 {
                parts.push(format!("- Elapsed time: {}h {}m {}s", hours, mins, secs));
            } else if mins > 0 {
                parts.push(format!("- Elapsed time: {}m {}s", mins, secs));
            } else {
                parts.push(format!("- Elapsed time: {}s", secs));
            }
        }
    }
    parts.push("- Older information may be less relevant than recent information — prioritize recent context".to_string());
    parts.join("\n")
}

/// Instructions for output management — injected into system prompt between OutputFormat and Tools regions.
/// Tells the LLM how to handle large command output proactively.
pub const OUTPUT_MANAGEMENT: &str = r#"📋 Output Management — ALL tools (especially bash) MUST adhere to:

1. **Large output MUST be filtered**: Commands that may return >100 lines must pipe through | head -N or | grep keyword to limit output
2. **Precise search first**: grep / find etc must specify path scope, do NOT scan the entire workspace
3. **Acknowledge on demand**: When only confirming result existence, use | grep -c or | wc -l instead of viewing full content
   4. **Truncation awareness**: Output exceeding 16KB will be silently truncated; results exceeding 2KB will be summarized with an IRI archive reference
   - If you see an 'output truncated' marker or '[archived]' tag → output was too large, narrow scope and search again
   - To view full results, use read_full_result_* tools to read on demand"#;

pub fn build_five_w2h_section(snapshot: &crate::core::five_w2h::Task5W2H) -> String {
    let mut lines = Vec::new();

    lines.push(format!("- Objective: {}", snapshot.what));

    lines.push(format!("- Reason: {}", snapshot.why.description));
    if !snapshot.why.success_criteria.is_empty() {
        lines.push(format!(
            "- Success Criteria: {}",
            snapshot.why.success_criteria.join(", ")
        ));
    }

    if let Some(ref who) = snapshot.who {
        if let Some(ref requestor) = who.requestor {
            lines.push(format!("- Requestor: {}", requestor));
        }
        if let Some(ref access_level) = who.access_level {
            lines.push(format!("- Access Level: {:?}", access_level));
        }
        if !who.assignees.is_empty() {
            lines.push(format!("- Assignees: {}", who.assignees.join(", ")));
        }
    }

    if let Some(ref when) = snapshot.when {
        if let Some(ref deadline) = when.deadline {
            lines.push(format!("- Deadline: {}", deadline));
        }
    }

    if let Some(ref where_) = snapshot.where_ {
        if let Some(ref env) = where_.execution_environment {
            lines.push(format!("- Execution Env: {}", env));
        }
        if !where_.data_sources.is_empty() {
            lines.push(format!(
                "- Data Sources: {}",
                where_.data_sources.join(", ")
            ));
        }
    }

    if let Some(ref how) = snapshot.how {
        if !how.forbidden_tools.is_empty() {
            lines.push(format!(
                "- Forbidden Tools: {}",
                how.forbidden_tools.join(", ")
            ));
        }
        if let Some(ref steps) = how.required_steps {
            lines.push(format!("- Required Steps: {}", steps));
        }
        if !how.preferred_skills.is_empty() {
            lines.push(format!(
                "- Preferred Skills: {}",
                how.preferred_skills.join(", ")
            ));
        }
    }

    if let Some(ref how_much) = snapshot.how_much {
        if let Some(ref budget) = how_much.token_budget {
            lines.push(format!("- Token Budget: {}", budget));
        }
        if let Some(ref cycles) = how_much.max_pdca_cycles {
            lines.push(format!("- Max Cycles: {}", cycles));
        }
    }

    lines.join("\n")
}

pub struct ToolRegionContent {
    pub builtin_tools: String,
    pub dynamic_tools: String,
}

impl ToolRegionContent {
    pub fn new() -> Self {
        Self {
            builtin_tools: String::new(),
            dynamic_tools: String::new(),
        }
    }

    pub fn with_builtin(mut self, tools: &str) -> Self {
        self.builtin_tools = tools.to_string();
        self
    }

    pub fn with_dynamic(mut self, tools: &str) -> Self {
        self.dynamic_tools = tools.to_string();
        self
    }

    pub fn build(&self) -> String {
        let mut parts = Vec::new();

        if !self.builtin_tools.is_empty() {
            parts.push(format!("## Built-in Tools (Fixed)\n{}", self.builtin_tools));
        }

        if !self.dynamic_tools.is_empty() {
            parts.push(format!(
                "## Dynamic Tools (On-demand)\n{}",
                self.dynamic_tools
            ));
        }

        parts.join("\n\n")
    }
}

impl Default for ToolRegionContent {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SystemPromptBuilder {
    regions: HashMap<SystemPromptRegion, String>,
}

impl SystemPromptBuilder {
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
        }
    }

    /// Set EmphasizedConstraints region in-place, avoiding full builder clone.
    /// Preferred over build_with_emphasis() as it avoids HashMap cloning.
    pub fn set_emphasis(&mut self, emphasis_items: &[String]) {
        if emphasis_items.is_empty() {
            return;
        }
        let content = emphasis_items
            .iter()
            .map(|e| format!("- {}", e))
            .collect::<Vec<_>>()
            .join("\n");
        self.set_region(SystemPromptRegion::EmphasizedConstraints, content);
    }

    pub fn set_region(&mut self, region: SystemPromptRegion, content: String) {
        self.regions.insert(region, content);
    }

    pub fn get_region(&self, region: &SystemPromptRegion) -> Option<&String> {
        self.regions.get(region)
    }

    pub fn clear_region(&mut self, region: &SystemPromptRegion) {
        self.regions.remove(region);
    }

    pub fn build(&self) -> String {
        let mut ordered_regions: Vec<(&SystemPromptRegion, &String)> =
            self.regions.iter().collect();
        ordered_regions.sort_by_key(|(r, _)| r.order());

        let mut parts = Vec::new();
        for (region, content) in ordered_regions {
            if !content.is_empty() {
                parts.push(format!("{}\n\n{}", region.header(), content));
            }
        }
        parts.join("\n\n---\n\n")
    }

    pub fn build_with_emphasis(&self, emphasis_items: &[String]) -> String {
        if emphasis_items.is_empty() {
            return self.build();
        }
        let mut builder = self.clone();
        builder.set_emphasis(emphasis_items);
        builder.build()
    }
}

impl Default for SystemPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SystemPromptBuilder {
    fn clone(&self) -> Self {
        Self {
            regions: self.regions.clone(),
        }
    }
}

/// Build a constitution prompt text for a given agent role using the ConstitutionRegistry.
///
/// Driven by the structured registry (41 entries) for queryability; the only source
/// of behavioral policy text for agent system prompts.
pub fn build_constitution_prompt(role: crate::core::agent_instance::AgentRole) -> String {
    use crate::core::constitution::ConstitutionRole;
    let registry = crate::core::constitution::ConstitutionRegistry::new();
    let constitution_role = match role {
        crate::core::agent_instance::AgentRole::Plan => ConstitutionRole::Plan,
        crate::core::agent_instance::AgentRole::Do => ConstitutionRole::Do,
        crate::core::agent_instance::AgentRole::Check => ConstitutionRole::Check,
        crate::core::agent_instance::AgentRole::Act => ConstitutionRole::Act,
    };
    registry.build_prompt_for_role(constitution_role)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_order() {
        assert!(
            SystemPromptRegion::RoleDefinition.order()
                < SystemPromptRegion::BehavioralPolicy.order()
        );
        assert!(
            SystemPromptRegion::BehavioralPolicy.order()
                < SystemPromptRegion::FiveW2HConstraints.order()
        );
        assert!(
            SystemPromptRegion::FiveW2HConstraints.order()
                < SystemPromptRegion::EmphasizedConstraints.order()
        );
        assert!(
            SystemPromptRegion::EmphasizedConstraints.order()
                < SystemPromptRegion::OutputFormat.order()
        );
        assert!(
            SystemPromptRegion::OutputFormat.order() < SystemPromptRegion::OutputManagement.order()
        );
        assert!(SystemPromptRegion::OutputManagement.order() < SystemPromptRegion::Tools.order());
        assert!(SystemPromptRegion::Tools.order() < SystemPromptRegion::ExtractionPrompt.order());
    }

    #[test]
    fn test_build_system_prompt() {
        let mut builder = SystemPromptBuilder::new();
        builder.set_region(
            SystemPromptRegion::RoleDefinition,
            "You are a Plan Agent".to_string(),
        );
        builder.set_region(
            SystemPromptRegion::OutputFormat,
            "Output JSON format".to_string(),
        );

        let result = builder.build();
        assert!(result.contains("# Role"));
        assert!(result.contains("# Output Format"));
        assert!(result.contains("You are a Plan Agent"));
    }

    #[test]
    fn test_build_with_emphasis() {
        let mut builder = SystemPromptBuilder::new();
        builder.set_region(
            SystemPromptRegion::RoleDefinition,
            "You are a Plan Agent".to_string(),
        );

        let emphasis = vec![
            "Must use async mode".to_string(),
            "Handle errors carefully".to_string(),
        ];
        let result = builder.build_with_emphasis(&emphasis);

        assert!(result.contains("Critical Constraints"));
        assert!(result.contains("Must use async mode"));
        assert!(result.contains("Handle errors carefully"));
    }

    #[test]
    fn test_tool_region_content() {
        let tool_content = ToolRegionContent::new()
            .with_builtin("file_read: read files\nfile_write: write files")
            .with_dynamic("http_request: HTTP requests\ncode_execute: execute code");

        let result = tool_content.build();
        assert!(result.contains("Built-in Tools (Fixed)"));
        assert!(result.contains("Dynamic Tools (On-demand)"));
        assert!(result.contains("file_read"));
        assert!(result.contains("http_request"));
    }

    #[test]
    fn test_build_with_tools() {
        let mut builder = SystemPromptBuilder::new();
        builder.set_region(
            SystemPromptRegion::RoleDefinition,
            "You are an Execution Agent".to_string(),
        );

        let tool_content = ToolRegionContent::new()
            .with_builtin("file_read: read files")
            .with_dynamic("custom_tool: custom tool")
            .build();
        builder.set_region(SystemPromptRegion::Tools, tool_content);

        let result = builder.build();
        assert!(result.contains("# Tools"));
        assert!(result.contains("Built-in Tools (Fixed)"));
        assert!(result.contains("Dynamic Tools (On-demand)"));
    }
}
