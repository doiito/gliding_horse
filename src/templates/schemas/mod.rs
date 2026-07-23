mod agent_template;
mod prompt_segment;

pub use agent_template::{validate_template, AgentTemplate};
pub use prompt_segment::{PromptSegment, SegmentType};

use crate::CoreError;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

pub struct TemplateRegistry {
    templates: RwLock<HashMap<String, AgentTemplate>>,
    by_role: RwLock<HashMap<String, Vec<String>>>,
    by_type: RwLock<HashMap<String, Vec<String>>>,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self {
            templates: RwLock::new(HashMap::new()),
            by_role: RwLock::new(HashMap::new()),
            by_type: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, template: AgentTemplate) -> Result<(), CoreError> {
        validate_template(&template).map_err(|e| CoreError::ValidationFailed {
            message: format!("Template validation failed: {}", e),
        })?;

        let id = template.id.clone();
        let role = template.role.clone();
        let types = template.type_.clone();

        {
            let mut templates = self.templates.write();
            templates.insert(id.clone(), template);
        }

        {
            let mut by_role = self.by_role.write();
            by_role.entry(role).or_default().push(id.clone());
        }

        {
            let mut by_type = self.by_type.write();
            for type_iri in types {
                by_type.entry(type_iri).or_default().push(id.clone());
            }
        }

        debug!(template_id = %id, "Template registered");
        Ok(())
    }

    pub fn load_template(path: &Path) -> Result<AgentTemplate, CoreError> {
        let content = std::fs::read_to_string(path).map_err(|e| CoreError::Internal {
            message: format!("Failed to read template file {}: {}", path.display(), e),
        })?;

        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| CoreError::InvalidJsonLd {
                message: format!("Invalid JSON in {}: {}", path.display(), e),
            })?;

        AgentTemplate::from_json_ld(&json)
    }

    pub fn load_from_dir(&self, dir: &Path) -> Result<usize, CoreError> {
        let mut count = 0;

        if !dir.exists() {
            debug!(dir = %dir.display(), "Template directory does not exist");
            return Ok(0);
        }

        let entries = std::fs::read_dir(dir).map_err(|e| CoreError::Internal {
            message: format!("Failed to read directory {}: {}", dir.display(), e),
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                match Self::load_template(&path) {
                    Ok(template) => {
                        if let Err(e) = self.register(template) {
                            warn!(path = %path.display(), error = %e, "Failed to register template");
                        } else {
                            count += 1;
                        }
                    }
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "Failed to load template");
                    }
                }
            }
        }

        info!(count = count, dir = %dir.display(), "Templates loaded from directory");
        Ok(count)
    }

    pub fn get(&self, id: &str) -> Option<AgentTemplate> {
        self.templates.read().get(id).cloned()
    }

    pub fn find_template_by_role(&self, role: &str) -> Option<AgentTemplate> {
        let by_role = self.by_role.read();
        by_role.get(role).and_then(|ids| {
            ids.first()
                .and_then(|id| self.templates.read().get(id).cloned())
        })
    }

    pub fn find_templates_by_role(&self, role: &str) -> Vec<AgentTemplate> {
        let by_role = self.by_role.read();
        by_role
            .get(role)
            .map(|ids| {
                let templates = self.templates.read();
                ids.iter()
                    .filter_map(|id| templates.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn find_templates_by_type(&self, type_iri: &str) -> Vec<AgentTemplate> {
        let by_type = self.by_type.read();
        by_type
            .get(type_iri)
            .map(|ids| {
                let templates = self.templates.read();
                ids.iter()
                    .filter_map(|id| templates.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn list_all(&self) -> Vec<String> {
        self.templates.read().keys().cloned().collect()
    }

    pub fn count(&self) -> usize {
        self.templates.read().len()
    }

    pub fn clear(&self) {
        self.templates.write().clear();
        self.by_role.write().clear();
        self.by_type.write().clear();
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_pa_template() -> AgentTemplate {
    AgentTemplate::new("iri://template/pa", "PA")
        .with_type("agent:PlanTemplate")
        .add_prompt_segment(PromptSegment::fixed(
            "# Role\nYou are the Plan Agent (PA). Your job is to analyze tasks and create execution plans.\n\n"
        ))
        .add_prompt_segment(PromptSegment::fixed(
            "# Capabilities\nYou have access to the following tools:\n"
        ))
        .add_prompt_segment(PromptSegment::dynamic("tool_list", Some("join_comma".to_string())))
        .add_prompt_segment(PromptSegment::fixed("\n\n# Instructions\n"))
        .add_prompt_segment(PromptSegment::fixed(
            "1. Analyze the task requirements carefully\n\
             2. Gather information using available tools (read files, list directories)\n\
             3. Create a detailed execution plan\n\
             4. Break down into specific steps with file operations\n\
             5. Identify dependencies between steps\n\n"
        ))
        .add_prompt_segment(PromptSegment::fixed(
            "# Important Rules\n\
             - You MUST use tools to gather information before planning\n\
             - Read existing files to understand the codebase\n\
             - List directories to understand project structure\n\
             - Output a structured plan with clear steps\n\n"
        ))
        .add_prompt_segment(PromptSegment::fixed("# Output Format\n"))
        .add_prompt_segment(PromptSegment::variable("output_schema"))
        .add_skill("iri://skills/file_read")
        .add_skill("iri://skills/file_write")
        .add_output_mapping("plan", "execution_plan")
}

pub fn create_da_template() -> AgentTemplate {
    AgentTemplate::new("iri://template/da", "DA")
        .with_type("agent:DoTemplate")
        .add_prompt_segment(PromptSegment::fixed(
            "# Role\nYou are the Do Agent (DA). Your job is to execute plans and perform tasks.\n\n"
        ))
        .add_prompt_segment(PromptSegment::fixed("# Available Tools\n"))
        .add_prompt_segment(PromptSegment::dynamic("tool_list", Some("join_comma".to_string())))
        .add_prompt_segment(PromptSegment::fixed("\n\n# Task Context\n"))
        .add_prompt_segment(PromptSegment::variable("task_context"))
        .add_prompt_segment(PromptSegment::fixed("\n\n# Instructions\n"))
        .add_prompt_segment(PromptSegment::fixed(
            "1. Follow the plan step by step\n\
             2. Use tools to perform operations\n\
             3. Report progress and results\n\
             4. Handle errors gracefully\n\n"
        ))
        .add_skill("iri://skills/file_read")
        .add_skill("iri://skills/file_write")
        .add_skill("iri://skills/http_request")
        .add_skill("iri://skills/code_execute")
        .add_skill("iri://skills/llm_chat")
        .add_output_mapping("result", "execution_result")
}

pub fn create_ca_template() -> AgentTemplate {
    AgentTemplate::new("iri://template/ca", "CA")
        .with_type("agent:CheckTemplate")
        .add_prompt_segment(PromptSegment::fixed(
            "# Role\nYou are the Check Agent (CA). Your job is to review results and validate quality.\n\n"
        ))
        .add_prompt_segment(PromptSegment::fixed("# Context\n"))
        .add_prompt_segment(PromptSegment::variable("execution_result"))
        .add_prompt_segment(PromptSegment::fixed("\n\n# Instructions\n"))
        .add_prompt_segment(PromptSegment::fixed(
            "1. Review the execution results\n\
             2. Validate against requirements\n\
             3. Check for errors and issues\n\
             4. Provide feedback and recommendations\n\n"
        ))
        .add_prompt_segment(PromptSegment::fixed("# Output Format\n"))
        .add_prompt_segment(PromptSegment::variable("output_schema"))
        .add_skill("iri://skills/file_read")
        .add_skill("iri://skills/jsonld_validate")
        .add_output_mapping("review", "quality_review")
}

pub fn create_aa_template() -> AgentTemplate {
    AgentTemplate::new("iri://template/aa", "AA")
        .with_type("agent:ActTemplate")
        .add_prompt_segment(PromptSegment::fixed(
            "# Role\nYou are the Act Agent (AA). Your job is to make decisions and take final actions.\n\n"
        ))
        .add_prompt_segment(PromptSegment::fixed("# Review Summary\n"))
        .add_prompt_segment(PromptSegment::variable("quality_review"))
        .add_prompt_segment(PromptSegment::fixed("\n\n# Instructions\n"))
        .add_prompt_segment(PromptSegment::fixed(
            "1. Analyze the quality review\n\
             2. Make decisions based on findings\n\
             3. Determine next actions\n\
             4. Update task status\n\n"
        ))
        .add_prompt_segment(PromptSegment::fixed("# Output Format\n"))
        .add_prompt_segment(PromptSegment::variable("output_schema"))
        .add_skill("iri://skills/file_write")
        .add_output_mapping("decision", "final_decision")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_registry_register() {
        let registry = TemplateRegistry::new();
        let template = AgentTemplate::new("iri://template/test", "PA");

        assert!(registry.register(template).is_ok());
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_registry_find_by_role() {
        let registry = TemplateRegistry::new();
        let template = AgentTemplate::new("iri://template/pa", "PA");
        registry.register(template).unwrap();

        let found = registry.find_template_by_role("PA");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "iri://template/pa");
    }

    #[test]
    fn test_registry_find_by_type() {
        let registry = TemplateRegistry::new();
        let template =
            AgentTemplate::new("iri://template/pa", "PA").with_type("agent:PlanTemplate");
        registry.register(template).unwrap();

        let found = registry.find_templates_by_type("agent:PlanTemplate");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_create_pa_template() {
        let template = create_pa_template();
        assert_eq!(template.role, "PA");
        assert!(template.type_.contains(&"agent:PlanTemplate".to_string()));
        assert!(!template.system_prompt.is_empty());
        assert!(!template.skill_whitelist.is_empty());
    }

    #[test]
    fn test_create_da_template() {
        let template = create_da_template();
        assert_eq!(template.role, "DA");
        assert!(template.has_skill("iri://skills/code_execute"));
    }

    #[test]
    fn test_create_ca_template() {
        let template = create_ca_template();
        assert_eq!(template.role, "CA");
        assert!(template.has_skill("iri://skills/jsonld_validate"));
    }

    #[test]
    fn test_create_aa_template() {
        let template = create_aa_template();
        assert_eq!(template.role, "AA");
        assert!(template.output_mapping.contains_key("decision"));
    }

    #[test]
    fn test_render_pa_template() {
        let template = create_pa_template();
        let mut ctx = serde_json::Map::new();
        ctx.insert("tool_list".to_string(), json!(["file_read", "file_write"]));
        ctx.insert("output_schema".to_string(), json!({"type": "object"}));

        let result = template.render_system_prompt(&ctx);
        assert!(result.contains("Plan Agent"));
        assert!(result.contains("file_read, file_write"));
    }
}
