use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentType {
    Fixed,
    Variable,
    Dynamic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSegment {
    #[serde(rename = "segment_type")]
    pub segment_type: SegmentType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<String>,
}

impl PromptSegment {
    pub fn fixed(content: impl Into<String>) -> Self {
        Self {
            segment_type: SegmentType::Fixed,
            content: Some(content.into()),
            source: None,
            transform: None,
        }
    }

    pub fn variable(source: impl Into<String>) -> Self {
        Self {
            segment_type: SegmentType::Variable,
            content: None,
            source: Some(source.into()),
            transform: None,
        }
    }

    pub fn dynamic(source: impl Into<String>, transform: Option<String>) -> Self {
        Self {
            segment_type: SegmentType::Dynamic,
            content: None,
            source: Some(source.into()),
            transform,
        }
    }

    pub fn render(&self, context: &serde_json::Map<String, serde_json::Value>) -> String {
        match self.segment_type {
            SegmentType::Fixed => self.content.clone().unwrap_or_default(),
            SegmentType::Variable => {
                if let Some(source) = &self.source {
                    context
                        .get(source)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                } else {
                    String::new()
                }
            }
            SegmentType::Dynamic => {
                if let Some(source) = &self.source {
                    let value = context
                        .get(source)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    if let Some(transform_fn) = &self.transform {
                        self.apply_transform(&value, transform_fn)
                    } else {
                        serde_json::to_string_pretty(&value).unwrap_or_default()
                    }
                } else {
                    String::new()
                }
            }
        }
    }

    fn apply_transform(&self, value: &serde_json::Value, transform: &str) -> String {
        match transform {
            "json_pretty" => serde_json::to_string_pretty(value).unwrap_or_default(),
            "json_compact" => serde_json::to_string(value).unwrap_or_default(),
            "join_comma" => {
                if let Some(arr) = value.as_array() {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                } else {
                    value.to_string()
                }
            }
            _ => value.to_string(),
        }
    }
}

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

    pub fn to_json_ld(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|e| format!("Failed to serialize: {}", e))
    }

    pub fn from_json_ld(json: &serde_json::Value) -> Result<Self, String> {
        serde_json::from_value(json.clone()).map_err(|e| format!("Failed to deserialize: {}", e))
    }

    pub fn render_system_prompt(
        &self,
        context: &serde_json::Map<String, serde_json::Value>,
    ) -> String {
        self.system_prompt
            .iter()
            .map(|s| s.render(context))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn has_skill(&self, skill_iri: &str) -> bool {
        self.skill_whitelist.contains(&skill_iri.to_string())
    }
}

fn main() {
    println!("=== 测试 PromptSegment ===");

    let fixed = PromptSegment::fixed("Hello, World!");
    println!("Fixed segment: {:?}", fixed);

    let mut ctx = serde_json::Map::new();
    ctx.insert("name".to_string(), serde_json::json!("Alice"));

    let variable = PromptSegment::variable("name");
    println!("Variable segment render: {}", variable.render(&ctx));

    ctx.insert("items".to_string(), serde_json::json!(["a", "b", "c"]));
    let dynamic = PromptSegment::dynamic("items", Some("join_comma".to_string()));
    println!("Dynamic segment render: {}", dynamic.render(&ctx));

    println!("\n=== 测试 AgentTemplate ===");

    let template = AgentTemplate::new("iri://template/pa", "PA")
        .with_type("agent:PlanTemplate")
        .add_prompt_segment(PromptSegment::fixed("You are a Plan Agent.\n"))
        .add_prompt_segment(PromptSegment::variable("task"))
        .add_skill("iri://skill/file_read")
        .add_skill("iri://skill/file_write");

    println!("Template ID: {}", template.id);
    println!("Template Role: {}", template.role);
    println!("Template Types: {:?}", template.type_);
    println!(
        "Has file_read skill: {}",
        template.has_skill("iri://skill/file_read")
    );

    println!("\n=== 测试 JSON-LD 转换 ===");
    let json_ld = template.to_json_ld().unwrap();
    println!(
        "JSON-LD:\n{}",
        serde_json::to_string_pretty(&json_ld).unwrap()
    );

    println!("\n=== 测试模板渲染 ===");
    let mut render_ctx = serde_json::Map::new();
    render_ctx.insert(
        "task".to_string(),
        serde_json::json!("Build a web application"),
    );

    let rendered = template.render_system_prompt(&render_ctx);
    println!("Rendered prompt:\n{}", rendered);

    println!("\n=== 测试从 JSON-LD 解析 ===");
    let json_str = r#"{
        "@context": "https://schema.org",
        "@id": "iri://template/da",
        "@type": ["agent:RoleTemplate", "agent:DoTemplate"],
        "role": "DA",
        "system_prompt": [
            {"segment_type": "fixed", "content": "Do Agent"}
        ],
        "output_mapping": {},
        "skill_whitelist": ["iri://skill/file_read"]
    }"#;

    let json_value: serde_json::Value = serde_json::from_str(json_str).unwrap();
    let parsed = AgentTemplate::from_json_ld(&json_value).unwrap();
    println!("Parsed template ID: {}", parsed.id);
    println!("Parsed template role: {}", parsed.role);

    println!("\n=== 所有测试通过! ===");
}
