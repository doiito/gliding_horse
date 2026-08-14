use super::*;
use crate::core::agent_instance::AgentRole;
use crate::jsonld::JsonLdNode;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn create_test_runner() -> AgentRunner {
    use crate::config::settings::AgentSettings;
    use crate::config::settings::GatewaySettings;
    use crate::gateway::unified_gateway::UnifiedGateway;
    use crate::memory::l0_store::L0Store;
    use crate::memory::l2_blackboard::Blackboard;
    use crate::memory::memory_manager::MemoryManager;
    use crate::templates::template_engine::TemplateEngine;
    use crate::tools::skill_registry::SkillRegistry;
    use crate::CoreConfig;
    use std::path::Path;

    let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    // Tests may change the process working directory concurrently.  Use an
    // absolute, process-scoped directory so separate runners never resolve to
    // the same redb file and contend for its exclusive lock.
    let test_path = std::env::temp_dir()
        .join(format!(
            "glidinghorse-agent-runner-{}-{}",
            std::process::id(),
            test_id
        ))
        .to_string_lossy()
        .into_owned();
    let l0 = Arc::new(L0Store::new(&test_path).unwrap());
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let projection = Arc::new(ProjectionEngine::new(blackboard.clone(), 1024));
    let skills = Arc::new(SkillRegistry::new());
    let gateway_settings = GatewaySettings {
        base_url: "http://localhost:3000".to_string(),
        api_key: "test-key".to_string(),
        default_model: "deepseek-v4-pro".to_string(),
        timeout_seconds: 30,
        max_retries: 3,
        retry_base_ms: 500,
        use_responses_api: false,
        model_mapping: std::collections::HashMap::new(),
    };
    let gateway = Arc::new(UnifiedGateway::new(&gateway_settings).unwrap());
    let templates = Arc::new(TemplateEngine::new(Path::new("./templates")).unwrap());
    let config = CoreConfig::default();
    let memory_manager = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
        l0.clone(),
        blackboard.clone(),
        projection,
        config.clone(),
    )));
    let settings = AgentSettings::default();

    AgentRunner::new(
        gateway,
        skills,
        blackboard,
        l0,
        memory_manager,
        templates,
        settings,
    )
}

#[test]
fn test_token_optimization_settings_wired() {
    use crate::config::settings::{
        ContextWindowSettings, TokenOptimizationSettings, ToolResultAgingSettings,
        ToolResultCompressorSettings,
    };

    // Historical default: compressors are enabled by default (matching the
    // pre-config hardcoded behavior), so a default runner has them attached.
    let runner = create_test_runner();
    assert!(runner.tool_result_compressor.is_some());
    assert!(runner.tool_result_aging.is_some());
    assert!(runner.context_window_manager.is_some());

    // Disabled settings → with_token_optimization must detach all compressors.
    let disabled = TokenOptimizationSettings {
        enabled: false,
        tool_groups: Default::default(),
        tool_result_compressor: ToolResultCompressorSettings {
            enabled: false,
            max_full_results: 2,
            max_summary_length: 200,
            compression_trigger: 5,
            compress_tool_result_threshold: 500,
        },
        context_window: ContextWindowSettings {
            max_messages: 0,
            max_tokens: 16000,
            compression_ratio: 0.3,
            preserve_recent: 4,
            model_aware: false,
        },
        tool_result_aging: ToolResultAgingSettings {
            enabled: false,
            keep_full: 3,
            try_microtool: 5,
            compress_threshold: 500,
        },
        prompt_optimization: Default::default(),
    };
    let runner = runner.with_token_optimization(disabled);
    assert!(runner.tool_result_compressor.is_none());
    assert!(runner.tool_result_aging.is_none());
    assert!(runner.context_window_manager.is_none());

    // Enabled with distinct values → each compressor must exist and reflect the settings.
    let trc = ToolResultCompressorSettings {
        enabled: true,
        max_full_results: 3,
        max_summary_length: 99,
        compression_trigger: 5,
        compress_tool_result_threshold: 500,
    };
    let aging = ToolResultAgingSettings {
        enabled: true,
        keep_full: 4,
        try_microtool: 8,
        compress_threshold: 600,
    };
    let cwm = ContextWindowSettings {
        max_messages: 12,
        max_tokens: 8888,
        compression_ratio: 0.3,
        preserve_recent: 4,
        model_aware: false,
    };
    let to = TokenOptimizationSettings {
        enabled: true,
        tool_groups: Default::default(),
        tool_result_compressor: trc.clone(),
        context_window: cwm.clone(),
        tool_result_aging: aging.clone(),
        prompt_optimization: Default::default(),
    };
    let runner = runner.with_token_optimization(to);

    let compressor = runner
        .tool_result_compressor
        .as_ref()
        .expect("compressor should be created");
    let compressor = compressor.lock().unwrap();
    assert_eq!(compressor.max_full_results(), 3);
    assert_eq!(compressor.max_summary_length(), 99);

    let aging = runner.tool_result_aging.as_ref().expect("aging created");
    assert_eq!(aging.keep_full(), 4);
    assert_eq!(aging.try_microtool(), 8);

    let cwm = runner
        .context_window_manager
        .as_ref()
        .expect("cwm created");
    let cwm = cwm.lock().unwrap();
    assert_eq!(cwm.max_tokens(), 8888);
}

#[test]
fn test_parse_jsonld_response_valid() {
    let runner = create_test_runner();
    let response = json!({
        "@context": "https://pdca-agent.org/context/task",
        "@id": "iri://task/test123",
        "@type": "TaskNode",
        "summary": "Test task",
        "emphasis": ["important_constraint_1", "important_constraint_2"]
    })
    .to_string();

    let result = runner.parse_jsonld_response(&response);
    assert!(result.is_ok());

    let node = result.unwrap();
    assert_eq!(node.id, "iri://task/test123");
    assert_eq!(node.get_property("summary"), Some(&json!("Test task")));
}

#[test]
fn test_parse_jsonld_response_invalid() {
    let runner = create_test_runner();
    let response = json!({
        "summary": "Missing @id and @type"
    })
    .to_string();

    let result = runner.parse_jsonld_response(&response);
    assert!(result.is_err());
}

#[test]
fn test_extract_emphasis_from_array() {
    let runner = create_test_runner();
    let node = JsonLdNode::new("iri://task/test".to_string(), "TaskNode").with_property(
        "emphasis".to_string(),
        json!(["constraint_1", "constraint_2", "constraint_3"]),
    );

    let emphasis = runner.extract_emphasis(&node);
    assert_eq!(emphasis.len(), 3);
    assert_eq!(emphasis[0], "constraint_1");
}

#[test]
fn test_extract_emphasis_from_string() {
    let runner = create_test_runner();
    let node = JsonLdNode::new("iri://task/test".to_string(), "TaskNode")
        .with_property("emphasis".to_string(), json!("single_emphasis_content"));

    let emphasis = runner.extract_emphasis(&node);
    assert_eq!(emphasis.len(), 1);
    assert_eq!(emphasis[0], "single_emphasis_content");
}

#[test]
fn test_extract_emphasis_with_constraints() {
    let runner = create_test_runner();
    let node = JsonLdNode::new("iri://task/test".to_string(), "TaskNode")
        .with_property("emphasis".to_string(), json!(["emphasis_1"]))
        .with_property(
            "constraints".to_string(),
            json!(["constraint_A", "constraint_B"]),
        );

    let emphasis = runner.extract_emphasis(&node);
    assert_eq!(emphasis.len(), 3);
    assert!(emphasis.contains(&"emphasis_1".to_string()));
    assert!(emphasis.contains(&"[Constraint] constraint_A".to_string()));
}

#[test]
fn test_apply_output_mapping_plan() {
    let runner = create_test_runner();
    let output = json!({
        "plan": "execution_plan_content",
        "steps": ["step_1", "step_2"],
        "objective": "task_objective"
    });

    let result = runner.apply_output_mapping(&output, &AgentRole::Plan, "iri://task/123");
    assert!(result.is_some());

    let jsonld = result.unwrap();
    assert!(jsonld.get("@id").is_some());
    assert_eq!(
        jsonld.get("execution_plan"),
        Some(&json!("execution_plan_content"))
    );
    assert_eq!(jsonld.get("plan_steps"), Some(&json!(["step_1", "step_2"])));
    assert_eq!(jsonld.get("task_iri"), Some(&json!("iri://task/123")));
    assert_eq!(jsonld.get("agent_role"), Some(&json!("PA")));
}

#[test]
fn test_apply_output_mapping_do() {
    let runner = create_test_runner();
    let output = json!({
        "result": "execution_result",
        "artifacts": ["file_1.py", "file_2.rs"]
    });

    let result = runner.apply_output_mapping(&output, &AgentRole::Do, "iri://task/456");
    assert!(result.is_some());

    let jsonld = result.unwrap();
    assert_eq!(
        jsonld.get("execution_result"),
        Some(&json!("execution_result"))
    );
    assert_eq!(
        jsonld.get("created_artifacts"),
        Some(&json!(["file_1.py", "file_2.rs"]))
    );
}

#[test]
fn test_apply_output_mapping_check() {
    let runner = create_test_runner();
    let output = json!({
        "review": "check_result_ok",
        "passed": true
    });

    let result = runner.apply_output_mapping(&output, &AgentRole::Check, "iri://task/789");
    assert!(result.is_some());

    let jsonld = result.unwrap();
    assert_eq!(jsonld.get("check_review"), Some(&json!("check_result_ok")));
    assert_eq!(jsonld.get("check_passed"), Some(&json!(true)));
}

#[test]
fn test_apply_output_mapping_act() {
    let runner = create_test_runner();
    let output = json!({
        "decision": "final_decision",
        "action": "execute_next_step"
    });

    let result = runner.apply_output_mapping(&output, &AgentRole::Act, "iri://task/abc");
    assert!(result.is_some());

    let jsonld = result.unwrap();
    assert_eq!(jsonld.get("final_decision"), Some(&json!("final_decision")));
    assert_eq!(
        jsonld.get("recommended_action"),
        Some(&json!("execute_next_step"))
    );
}

#[test]
fn test_apply_output_mapping_string_output() {
    let runner = create_test_runner();
    let output = json!("simple_string_output");

    let result = runner.apply_output_mapping(&output, &AgentRole::Do, "iri://task/xyz");
    assert!(result.is_some());

    let jsonld = result.unwrap();
    assert_eq!(jsonld.get("content"), Some(&json!("simple_string_output")));
}

#[test]
fn test_task_result_jsonld_output() {
    let result = TaskResult {
        task_iri: "iri://task/test".to_string(),
        status: "success".to_string(),
        summary: "task_completed".to_string(),
        output: Some(json!("output_content")),
        jsonld_output: Some(json!({
            "@id": "iri://task/test_output",
            "@type": "DoOutput",
            "content": "output_content"
        })),
        artifacts: vec![],
        errors: vec![],
        turn_count: 5,
        tool_call_count: 3,
        five_w2h_updates: None,
        tracked_actions: Vec::new(),
        archive_iri: None,
    };

    assert!(result.jsonld_output.is_some());
    let jsonld = result.jsonld_output.unwrap();
    assert_eq!(jsonld.get("@id"), Some(&json!("iri://task/test_output")));
}

#[test]
fn test_try_extract_json_from_markdown_plain_json() {
    let input = r#"{"thought": "analyzing", "content": "testing", "action": "continue"}"#;
    let result = AgentRunner::try_extract_json_from_markdown(input);
    assert!(result.is_some());
    let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(parsed["action"], "continue");
}

#[test]
fn test_try_extract_json_from_markdown_json_code_block() {
    let input = "```json\n{\"thought\": \"thinking\", \"content\": \"content\", \"action\": \"tool_call\"}\n```";
    let result = AgentRunner::try_extract_json_from_markdown(input);
    assert!(result.is_some());
    let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(parsed["action"], "tool_call");
}

#[test]
fn test_try_extract_json_from_markdown_code_block_no_lang() {
    let input = "```\n{\"thought\": \"thinking\", \"content\": \"content\"}\n```";
    let result = AgentRunner::try_extract_json_from_markdown(input);
    assert!(result.is_some());
    let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(parsed["thought"], "thinking");
}

#[test]
fn test_try_extract_json_from_markdown_with_surrounding_text() {
    let input = "Okay_let_me_analyze.\n{\"thought\": \"analyze\", \"content\": \"result\", \"action\": \"finish\"}\nThat_is_my_analysis.";
    let result = AgentRunner::try_extract_json_from_markdown(input);
    assert!(result.is_some());
    let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(parsed["action"], "finish");
}

#[test]
fn test_try_extract_json_from_markdown_nested_braces() {
    let input = r#"{"thought": "nested", "content": {"sub": "value"}, "action": "continue"}"#;
    let result = AgentRunner::try_extract_json_from_markdown(input);
    assert!(result.is_some());
    let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(parsed["content"]["sub"], "value");
}

#[test]
fn test_try_extract_json_from_markdown_no_json() {
    let input = "This_is_plain_text_no_JSON.";
    let result = AgentRunner::try_extract_json_from_markdown(input);
    assert!(result.is_none());
}

#[test]
fn test_try_extract_json_from_markdown_incomplete_json() {
    let input = r#"{"thought": "incomplete", "content": "missing_closing_brace"#;
    let result = AgentRunner::try_extract_json_from_markdown(input);
    assert!(result.is_none());
}

#[test]
fn test_try_extract_json_from_markdown_multiple_json_objects() {
    let input =
        r#"prefix {"a": 1} suffix {"thought": "second", "content": "content", "action": "finish"}"#;
    let result = AgentRunner::try_extract_json_from_markdown(input);
    assert!(result.is_some());
    let parsed: Value = serde_json::from_str(&result.unwrap()).unwrap();
    assert_eq!(parsed["a"], 1);
}

#[test]
fn test_detect_blocker_verdict() {
    // Explicit blocker verdicts must downgrade from the hardcoded `success`.
    assert_eq!(
        AgentRunner::detect_blocker_verdict("Blocked: no task spec; archive loop stopped"),
        Some("failed")
    );
    assert_eq!(
        AgentRunner::detect_blocker_verdict(
            "Blocked: validated no-spec blocker; zero deliverables; terminate"
        ),
        Some("failed")
    );
    assert_eq!(
        AgentRunner::detect_blocker_verdict("Task status: success, all requirements met"),
        None
    );
    assert_eq!(AgentRunner::detect_blocker_verdict(""), None);
}

#[test]
fn test_task_result_partial_success_status() {
    let result = TaskResult {
        task_iri: "iri://task/test".to_string(),
        status: "partial_success".to_string(),
        summary: "task_partially_completed".to_string(),
        output: None,
        jsonld_output: None,
        artifacts: vec![],
        errors: vec!["bash: timeout".to_string()],
        turn_count: 15,
        tool_call_count: 5,
        five_w2h_updates: None,
        tracked_actions: Vec::new(),
        archive_iri: None,
    };
    assert_eq!(result.status, "partial_success");
    assert!(!result.errors.is_empty());
    assert!(result.summary.contains("partially_completed"));
}

#[test]
fn test_build_agent_md_da_renders_workspace_files_section() {
    let runner = create_test_runner();
    let mut context_data = std::collections::HashMap::new();
    context_data.insert(
        "workspace_files".to_string(),
        "3 files in workspace:\n- /tmp/a.js (200 bytes, 50 lines)\n- /tmp/b.js (1000 bytes, 120 lines)".to_string(),
    );
    let md = runner.build_agent_md(
        AgentRole::Do,
        "objective",
        &context_data,
        "deepseek-v4-pro",
    );
    assert!(md.contains("## Workspace Files"), "DA prompt renders file manifest");
    assert!(md.contains("/tmp/a.js"));
    assert!(
        md.contains("use file_read with offset/limit"),
        "DA prompt guides reading strategy"
    );
}

#[test]
fn test_build_agent_md_no_workspace_files_key_omits_section() {
    let runner = create_test_runner();
    let context_data = std::collections::HashMap::new();
    let md = runner.build_agent_md(
        AgentRole::Check,
        "objective",
        &context_data,
        "deepseek-v4-pro",
    );
    assert!(!md.contains("## Workspace Files"));
}

#[test]
fn test_available_skills_injects_role_skills() {
    use crate::tools::skill_registry::SkillMeta;

    let runner = create_test_runner();
    runner.skills.register_skill(SkillMeta {
        skill_iri: "iri://skills/da-analyze".to_string(),
        name: "analyze_output".to_string(),
        description: "Deep analysis of execution results".to_string(),
        version: "1.0.0".to_string(),
        category: "analysis".to_string(),
        security_level: "L0".to_string(),
        allowed_roles: vec!["DA".to_string()],
        input_schema: serde_json::json!({}),
        output_schema: serde_json::json!({}),
        compiled_template: "template".to_string(),
        signature: None,
        signature_algorithm: None,
        input_mapping: std::collections::HashMap::new(),
        output_mapping: std::collections::HashMap::new(),
        skill_types: vec![],
    });

    let tools = vec!["file_read".to_string(), "file_write".to_string()];
    let skills_text = AgentRunner::build_available_skills(&tools, &runner.skills, "DA");
    assert!(
        skills_text.contains("analyze_output: Deep analysis of execution results"),
        "role-visible skill should be injected, got: {}",
        skills_text
    );
    assert!(skills_text.starts_with("file_read, file_write"));
}

#[test]
fn test_agent_md_fallback_includes_injected_skills() {
    use crate::tools::skill_registry::SkillMeta;

    let runner = create_test_runner();
    runner.skills.register_skill(SkillMeta {
        skill_iri: "iri://skills/da-analyze".to_string(),
        name: "analyze_output".to_string(),
        description: "Deep analysis of execution results".to_string(),
        version: "1.0.0".to_string(),
        category: "analysis".to_string(),
        security_level: "L0".to_string(),
        allowed_roles: vec!["DA".to_string()],
        input_schema: serde_json::json!({}),
        output_schema: serde_json::json!({}),
        compiled_template: "template".to_string(),
        signature: None,
        signature_algorithm: None,
        input_mapping: std::collections::HashMap::new(),
        output_mapping: std::collections::HashMap::new(),
        skill_types: vec![],
    });

    let context_data = std::collections::HashMap::new();
    let md = runner.build_agent_md(AgentRole::Do, "objective", &context_data, "deepseek-v4-pro");
    assert!(
        md.contains("## Available Skills") && md.contains("analyze_output"),
        "fallback agent.md must include injected skills, got: {}",
        md
    );
}

#[test]
fn test_available_skills_dedupes_tool_names() {
    use crate::tools::skill_registry::SkillMeta;

    let runner = create_test_runner();
    // A skill whose name collides with an actual tool must not be injected twice.
    runner.skills.register_skill(SkillMeta {
        skill_iri: "iri://skills/dup".to_string(),
        name: "file_read".to_string(),
        description: "duplicate".to_string(),
        version: "1.0.0".to_string(),
        category: "misc".to_string(),
        security_level: "L0".to_string(),
        allowed_roles: vec!["DA".to_string()],
        input_schema: serde_json::json!({}),
        output_schema: serde_json::json!({}),
        compiled_template: "template".to_string(),
        signature: None,
        signature_algorithm: None,
        input_mapping: std::collections::HashMap::new(),
        output_mapping: std::collections::HashMap::new(),
        skill_types: vec![],
    });

    let tools = vec!["file_read".to_string()];
    let skills_text = AgentRunner::build_available_skills(&tools, &runner.skills, "DA");
    let occurrences = skills_text.matches("file_read").count();
    assert_eq!(
        occurrences, 1,
        "tool-named skill must be deduped against the tool list, found {} occurrences: {}",
        occurrences, skills_text
    );
}

#[test]
fn test_available_skills_skips_other_roles() {
    use crate::tools::skill_registry::SkillMeta;

    let runner = create_test_runner();
    runner.skills.register_skill(SkillMeta {
        skill_iri: "iri://skills/pa-only".to_string(),
        name: "plan_strategy".to_string(),
        description: "planning expertise".to_string(),
        version: "1.0.0".to_string(),
        category: "planning".to_string(),
        security_level: "L0".to_string(),
        allowed_roles: vec!["PA".to_string()],
        input_schema: serde_json::json!({}),
        output_schema: serde_json::json!({}),
        compiled_template: "template".to_string(),
        signature: None,
        signature_algorithm: None,
        input_mapping: std::collections::HashMap::new(),
        output_mapping: std::collections::HashMap::new(),
        skill_types: vec![],
    });

    let tools = vec!["file_read".to_string()];
    let skills_text = AgentRunner::build_available_skills(&tools, &runner.skills, "DA");
    assert!(
        !skills_text.contains("plan_strategy"),
        "PA-only skill must not appear for DA, got: {}",
        skills_text
    );
}
#[test]
fn test_agent_md_prompt_loader_fallback_injects_skills() {
    use crate::core::prompt_loader::{PromptConfig, PromptLoader};
    use crate::templates::template_engine::TemplateEngine;
    use crate::tools::skill_registry::SkillMeta;
    use std::path::Path;

    let runner = create_test_runner();
    // A loader backed by an empty template dir exercises the builtin-fallback
    // branch of PromptLoader::load, which ignores the `available_skills` var.
    let empty_dir = std::env::temp_dir().join(format!("gh-pl-empty-{}", std::process::id()));
    std::fs::create_dir_all(&empty_dir).unwrap();
    let tmpl = Arc::new(TemplateEngine::new(Path::new(&empty_dir)).unwrap());
    let runner = runner.with_prompt_loader(PromptLoader::new(PromptConfig::default(), tmpl));

    runner.skills.register_skill(SkillMeta {
        skill_iri: "iri://skills/da-analyze".to_string(),
        name: "analyze_output".to_string(),
        description: "Deep analysis of execution results".to_string(),
        version: "1.0.0".to_string(),
        category: "analysis".to_string(),
        security_level: "L0".to_string(),
        allowed_roles: vec!["DA".to_string()],
        input_schema: serde_json::json!({}),
        output_schema: serde_json::json!({}),
        compiled_template: "template".to_string(),
        signature: None,
        signature_algorithm: None,
        input_mapping: std::collections::HashMap::new(),
        output_mapping: std::collections::HashMap::new(),
        skill_types: vec![],
    });

    let context_data = std::collections::HashMap::new();
    let md = runner.build_agent_md(AgentRole::Do, "objective", &context_data, "deepseek-v4-pro");
    assert!(
        md.contains("## Available Skills") && md.contains("analyze_output"),
        "PromptLoader builtin-fallback agent.md must include injected skills, got: {}",
        md
    );
}
