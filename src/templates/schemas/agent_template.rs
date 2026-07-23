use super::PromptSegment;
use crate::CoreError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTemplate {
    #[serde(rename = "@context")]
    pub context: String,

    #[serde(rename = "@id")]
    pub id: String,

    #[serde(rename = "@type")]
    pub type_: Vec<String>,

    pub role: String,

    #[serde(default)]
    pub system_prompt: Vec<PromptSegment>,

    #[serde(default)]
    pub output_mapping: HashMap<String, String>,

    #[serde(default)]
    pub skill_whitelist: Vec<String>,
}

impl AgentTemplate {
    pub fn new(id: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            context: "https://schema.org".to_string(),
            id: id.into(),
            type_: vec!["agent:RoleTemplate".to_string()],
            role: role.into(),
            system_prompt: Vec::new(),
            output_mapping: HashMap::new(),
            skill_whitelist: Vec::new(),
        }
    }

    pub fn with_type(mut self, type_iri: impl Into<String>) -> Self {
        self.type_.push(type_iri.into());
        self
    }

    pub fn add_prompt_segment(mut self, segment: PromptSegment) -> Self {
        self.system_prompt.push(segment);
        self
    }

    pub fn add_skill(mut self, skill: impl Into<String>) -> Self {
        self.skill_whitelist.push(skill.into());
        self
    }

    pub fn add_output_mapping(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.output_mapping.insert(key.into(), value.into());
        self
    }

    pub fn to_json_ld(&self) -> Result<serde_json::Value, CoreError> {
        serde_json::to_value(self).map_err(|e| CoreError::InvalidJsonLd {
            message: format!("Failed to serialize template: {}", e),
        })
    }

    pub fn from_json_ld(json: &serde_json::Value) -> Result<Self, CoreError> {
        serde_json::from_value(json.clone()).map_err(|e| CoreError::InvalidJsonLd {
            message: format!("Failed to deserialize template: {}", e),
        })
    }

    pub fn render_system_prompt(
        &self,
        context: &serde_json::Map<String, serde_json::Value>,
    ) -> String {
        self.system_prompt
            .iter()
            .map(|segment| segment.render(context))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn has_skill(&self, skill_iri: &str) -> bool {
        self.skill_whitelist.contains(&skill_iri.to_string())
    }
}

pub fn validate_template(template: &AgentTemplate) -> Result<(), String> {
    if template.id.is_empty() {
        return Err("Template @id is required".to_string());
    }

    if !template.id.starts_with("iri://") {
        return Err("Template @id must start with 'iri://'".to_string());
    }

    if template.role.is_empty() {
        return Err("Template role is required".to_string());
    }

    let valid_roles = ["PA", "DA", "CA", "AA"];
    if !valid_roles.contains(&template.role.as_str()) {
        return Err(format!(
            "Invalid role '{}', must be one of: {:?}",
            template.role, valid_roles
        ));
    }

    if template.type_.is_empty() {
        return Err("Template must have at least one @type".to_string());
    }

    for segment in &template.system_prompt {
        match segment.segment_type {
            super::SegmentType::Fixed => {
                if segment.content.is_none() {
                    return Err("Fixed segment must have content".to_string());
                }
            }
            super::SegmentType::Variable | super::SegmentType::Dynamic => {
                if segment.source.is_none() {
                    return Err("Variable/Dynamic segment must have source".to_string());
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_create_template() {
        let template = AgentTemplate::new("iri://template/pa", "PA")
            .with_type("agent:PlanTemplate")
            .add_prompt_segment(PromptSegment::fixed("You are a Plan Agent."))
            .add_skill("iri://skills/file_read")
            .add_output_mapping("plan", "execution_plan");

        assert_eq!(template.id, "iri://template/pa");
        assert_eq!(template.role, "PA");
        assert_eq!(template.type_.len(), 2);
        assert_eq!(template.system_prompt.len(), 1);
        assert!(template.has_skill("iri://skills/file_read"));
    }

    #[test]
    fn test_to_json_ld() {
        let template = AgentTemplate::new("iri://template/da", "DA")
            .add_prompt_segment(PromptSegment::fixed("Do Agent"));

        let json = template.to_json_ld().unwrap();
        assert_eq!(json["@id"], "iri://template/da");
        assert_eq!(json["role"], "DA");
    }

    #[test]
    fn test_from_json_ld() {
        let json = json!({
            "@context": "https://schema.org",
            "@id": "iri://template/ca",
            "@type": ["agent:RoleTemplate"],
            "role": "CA",
            "system_prompt": [],
            "output_mapping": {},
            "skill_whitelist": []
        });

        let template = AgentTemplate::from_json_ld(&json).unwrap();
        assert_eq!(template.id, "iri://template/ca");
        assert_eq!(template.role, "CA");
    }

    #[test]
    fn test_render_system_prompt() {
        let template = AgentTemplate::new("iri://template/test", "DA")
            .add_prompt_segment(PromptSegment::fixed("Hello "))
            .add_prompt_segment(PromptSegment::variable("name"))
            .add_prompt_segment(PromptSegment::fixed("!"));

        let mut ctx = serde_json::Map::new();
        ctx.insert("name".to_string(), json!("World"));

        let result = template.render_system_prompt(&ctx);
        assert_eq!(result, "Hello \nWorld\n!");
    }

    #[test]
    fn test_validate_template() {
        let valid = AgentTemplate::new("iri://template/pa", "PA");
        assert!(validate_template(&valid).is_ok());

        let invalid_role = AgentTemplate::new("iri://template/xx", "XX");
        assert!(validate_template(&invalid_role).is_err());

        let invalid_id = AgentTemplate::new("invalid-id", "PA");
        assert!(validate_template(&invalid_id).is_err());
    }
}
