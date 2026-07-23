use glidinghorse::templates::schemas::{
    create_aa_template, create_ca_template, create_da_template, create_pa_template,
    validate_template, AgentTemplate, PromptSegment, SegmentType, TemplateRegistry,
};
use serde_json::json;

#[test]
fn test_prompt_segment_fixed() {
    let segment = PromptSegment::fixed("Hello, World!");
    assert_eq!(segment.segment_type, SegmentType::Fixed);
    assert_eq!(segment.content, Some("Hello, World!".to_string()));

    let ctx = serde_json::Map::new();
    assert_eq!(segment.render(&ctx), "Hello, World!");
}

#[test]
fn test_prompt_segment_variable() {
    let segment = PromptSegment::variable("name");
    assert_eq!(segment.segment_type, SegmentType::Variable);
    assert_eq!(segment.source, Some("name".to_string()));

    let mut ctx = serde_json::Map::new();
    ctx.insert("name".to_string(), json!("Alice"));
    assert_eq!(segment.render(&ctx), "Alice");
}

#[test]
fn test_prompt_segment_dynamic_transform() {
    let segment = PromptSegment::dynamic("items", Some("join_comma".to_string()));

    let mut ctx = serde_json::Map::new();
    ctx.insert("items".to_string(), json!(["a", "b", "c"]));
    assert_eq!(segment.render(&ctx), "a, b, c");
}

#[test]
fn test_agent_template_creation() {
    let template = AgentTemplate::new("iri://template/test", "PA")
        .with_type("agent:PlanTemplate")
        .add_prompt_segment(PromptSegment::fixed("You are a Plan Agent."))
        .add_skill("iri://skills/file_read")
        .add_output_mapping("plan", "execution_plan");

    assert_eq!(template.id, "iri://template/test");
    assert_eq!(template.role, "PA");
    assert_eq!(template.type_.len(), 2);
    assert_eq!(template.system_prompt.len(), 1);
    assert!(template.has_skill("iri://skills/file_read"));
    assert!(template.output_mapping.contains_key("plan"));
}

#[test]
fn test_agent_template_json_ld() {
    let template = AgentTemplate::new("iri://template/da", "DA")
        .add_prompt_segment(PromptSegment::fixed("Do Agent"));

    let json = template.to_json_ld().unwrap();
    assert_eq!(json["@id"], "iri://template/da");
    assert_eq!(json["role"], "DA");

    let parsed = AgentTemplate::from_json_ld(&json).unwrap();
    assert_eq!(parsed.id, template.id);
    assert_eq!(parsed.role, template.role);
}

#[test]
fn test_agent_template_render() {
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
fn test_validate_template_valid() {
    let valid = AgentTemplate::new("iri://template/pa", "PA");
    assert!(validate_template(&valid).is_ok());
}

#[test]
fn test_validate_template_invalid_role() {
    let invalid = AgentTemplate::new("iri://template/xx", "XX");
    assert!(validate_template(&invalid).is_err());
}

#[test]
fn test_validate_template_invalid_id() {
    let invalid = AgentTemplate::new("invalid-id", "PA");
    assert!(validate_template(&invalid).is_err());
}

#[test]
fn test_template_registry_register() {
    let registry = TemplateRegistry::new();
    let template = AgentTemplate::new("iri://template/test", "PA");

    assert!(registry.register(template).is_ok());
    assert_eq!(registry.count(), 1);
}

#[test]
fn test_template_registry_find_by_role() {
    let registry = TemplateRegistry::new();
    let template = AgentTemplate::new("iri://template/pa", "PA");
    registry.register(template).unwrap();

    let found = registry.find_template_by_role("PA");
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, "iri://template/pa");
}

#[test]
fn test_template_registry_find_by_type() {
    let registry = TemplateRegistry::new();
    let template = AgentTemplate::new("iri://template/pa", "PA").with_type("agent:PlanTemplate");
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
    assert!(template.has_skill("iri://skills/file_read"));
}

#[test]
fn test_create_da_template() {
    let template = create_da_template();
    assert_eq!(template.role, "DA");
    assert!(template.has_skill("iri://skills/code_execute"));
    assert!(template.has_skill("iri://skills/llm_chat"));
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
fn test_render_pa_template_with_context() {
    let template = create_pa_template();
    let mut ctx = serde_json::Map::new();
    ctx.insert("tool_list".to_string(), json!(["file_read", "file_write"]));
    ctx.insert("output_schema".to_string(), json!({"type": "object"}));

    let result = template.render_system_prompt(&ctx);
    assert!(result.contains("Plan Agent"));
    assert!(result.contains("file_read, file_write"));
}

#[test]
fn test_template_serialization_roundtrip() {
    let original = create_da_template();
    let json = original.to_json_ld().unwrap();
    let restored = AgentTemplate::from_json_ld(&json).unwrap();

    assert_eq!(original.id, restored.id);
    assert_eq!(original.role, restored.role);
    assert_eq!(original.type_, restored.type_);
    assert_eq!(
        original.skill_whitelist.len(),
        restored.skill_whitelist.len()
    );
}

#[test]
fn test_multiple_templates_in_registry() {
    let registry = TemplateRegistry::new();

    registry.register(create_pa_template()).unwrap();
    registry.register(create_da_template()).unwrap();
    registry.register(create_ca_template()).unwrap();
    registry.register(create_aa_template()).unwrap();

    assert_eq!(registry.count(), 4);

    assert!(registry.find_template_by_role("PA").is_some());
    assert!(registry.find_template_by_role("DA").is_some());
    assert!(registry.find_template_by_role("CA").is_some());
    assert!(registry.find_template_by_role("AA").is_some());
}
