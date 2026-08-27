use super::*;
use crate::core::agent_instance::AgentRole;
use crate::jsonld::JsonLdNode;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn workspace_effect_progress_detects_a_late_read_only_stall() {
    use super::execution::{record_workspace_effect_turn, workspace_effect_recovery_active};

    let mut observed = false;
    let mut effectless_tail = 0;

    // Early implementation progress must not permanently disable monitoring.
    record_workspace_effect_turn(&mut observed, &mut effectless_tail, true);
    assert!(observed);
    assert_eq!(effectless_tail, 0);

    for _ in 0..12 {
        record_workspace_effect_turn(&mut observed, &mut effectless_tail, false);
    }
    assert!(observed, "all-time mutation evidence must be retained");
    assert_eq!(effectless_tail, 12);
    assert!(workspace_effect_recovery_active(
        true,
        effectless_tail,
        0,
        12
    ));

    // A later successful edit restores the normal tool window.
    record_workspace_effect_turn(&mut observed, &mut effectless_tail, true);
    assert_eq!(effectless_tail, 0);
    assert!(!workspace_effect_recovery_active(
        true,
        effectless_tail,
        0,
        12
    ));
    assert!(
        workspace_effect_recovery_active(true, 2, 12, 12),
        "repeated evidence must activate recovery even when mixed with nominally new reads"
    );
}

#[test]
fn implementation_phase_withholds_broad_discovery_but_keeps_targeted_read_and_write() {
    let definition = |name: &str| serde_json::json!({"type":"function","function":{"name":name,"parameters":{}}});
    let filtered = super::execution::phase_tool_definitions(
        vec![
            definition("file_list"),
            definition("glob_search"),
            definition("file_read"),
            definition("file_write"),
            definition("bash"),
        ],
        crate::core::agent_instance::AgentRole::Do,
        super::execution::ExecutionPhase::Implement,
    );
    let names = filtered
        .iter()
        .filter_map(|value| value["function"]["name"].as_str())
        .collect::<Vec<_>>();
    assert!(!names.contains(&"file_list"));
    assert!(!names.contains(&"glob_search"));
    assert!(names.contains(&"file_read"));
    assert!(names.contains(&"file_write"));
    assert!(names.contains(&"bash"));
}

#[test]
fn complete_bounded_inventory_withholds_only_redundant_broad_discovery() {
    let definition = |name: &str| serde_json::json!({"type":"function","function":{"name":name,"parameters":{}}});
    let filtered = super::execution::workspace_inventory_tool_definitions(
        vec![
            definition("file_list"),
            definition("glob_search"),
            definition("workspace_status"),
            definition("grep_search"),
            definition("file_read"),
            definition("file_write"),
        ],
        true,
    );
    let names = filtered
        .iter()
        .filter_map(|value| value["function"]["name"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["grep_search", "file_read", "file_write"]);
}

#[test]
fn execution_rejects_a_tool_not_advertised_in_the_current_turn() {
    let definitions = vec![
        json!({"type":"function","function":{"name":"file_read","parameters":{}}}),
        json!({"type":"function","function":{"name":"bash","parameters":{}}}),
    ];
    let advertised = super::execution::advertised_tool_names(&definitions);

    assert!(super::execution::unadvertised_tool_call_result(&advertised, "file_read").is_none());
    let rejection = super::execution::unadvertised_tool_call_result(&advertised, "file_list")
        .expect("a withdrawn broad inventory tool must be rejected at execution time");
    assert_eq!(rejection["status"], "not_executed");
    assert_eq!(rejection["reason"], "tool_not_advertised");
    assert!(rejection.get("error").is_none());
    assert!(rejection["message"]
        .as_str()
        .is_some_and(|message| message.contains("was not executed")));
    assert!(
        !crate::core::tracked_action::tool_result_failed(&rejection),
        "protocol feedback must not be learned as a failed skill execution"
    );
}

#[test]
fn ca_evidence_focus_keeps_independent_checks_but_drops_new_discovery() {
    let definition = |name: &str| serde_json::json!({"type":"function","function":{"name":name,"parameters":{}}});
    let filtered = super::execution::ca_evidence_focus_tool_definitions(
        vec![
            definition("file_list"),
            definition("grep_search"),
            definition("file_read"),
            definition("bash"),
            definition("read_agent_output"),
        ],
        AgentRole::Check,
        true,
    );
    let names = filtered
        .iter()
        .filter_map(|value| value["function"]["name"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["file_read", "bash", "read_agent_output"]);
}

#[test]
fn ca_evidence_close_gate_removes_tools_only_for_ca() {
    let definition = |name: &str| serde_json::json!({"type":"function","function":{"name":name,"parameters":{}}});
    let ca_tools = super::execution::ca_evidence_close_tool_definitions(
        vec![definition("file_read"), definition("bash")],
        AgentRole::Check,
        true,
    );
    assert!(ca_tools.is_empty());

    let da_tools = super::execution::ca_evidence_close_tool_definitions(
        vec![definition("file_write")],
        AgentRole::Do,
        true,
    );
    assert_eq!(da_tools.len(), 1);
}

#[test]
fn ca_da_correction_starts_in_repair_with_its_configured_guard() {
    let mut constraints = std::collections::HashMap::new();
    constraints.insert(
        super::SA_RECOVERY_MODE_CONSTRAINT.to_string(),
        super::CA_DA_CORRECTION_MODE.to_string(),
    );
    let phase = super::execution::initial_execution_phase(AgentRole::Do, &constraints);
    assert_eq!(phase, super::execution::ExecutionPhase::Repair);
    assert_eq!(
        super::execution::effective_effect_block_turns(phase, 12, 4),
        4
    );
    assert_eq!(
        super::execution::effective_effect_block_turns(phase, 12, 0),
        12,
        "zero repair guard inherits the general configured threshold"
    );
}

#[test]
fn pa_planning_focus_closes_tools_after_configured_evidence_window() {
    let definition = |name: &str| serde_json::json!({"type":"function","function":{"name":name,"parameters":{}}});
    let filtered = super::execution::pa_planning_focus_tool_definitions(
        vec![definition("file_read"), definition("grep_search")],
        AgentRole::Plan,
        true,
    );
    assert!(filtered.is_empty());

    let da_tools = super::execution::pa_planning_focus_tool_definitions(
        vec![definition("file_read")],
        AgentRole::Do,
        true,
    );
    assert_eq!(da_tools.len(), 1);
}

#[test]
fn evidence_keys_change_with_workspace_generation() {
    let args = serde_json::json!({"path":"src/lib.rs"});
    let first = super::execution::evidence_key("file_read", &args, 1).unwrap();
    let same = super::execution::evidence_key("file_read", &args, 1).unwrap();
    let changed = super::execution::evidence_key("file_read", &args, 2).unwrap();
    assert_eq!(first, same);
    assert_ne!(first, changed);
}

#[test]
fn replaceable_execution_ledger_never_accumulates_prompt_state() {
    use crate::gateway::unified_gateway::ChatMessage;

    let mut messages = vec![ChatMessage {
        role: "system".to_string(),
        content: "stable application prompt".to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    for generation in 1..=25 {
        super::execution::refresh_execution_ledger(
            &mut messages,
            AgentRole::Do,
            super::execution::ExecutionPhase::Implement,
            &crate::core::effect::EffectPolicy::required_workspace_mutation(),
            generation,
            0,
            0,
            generation as u64,
        );
    }
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.name.as_deref() == Some("execution_ledger"))
            .count(),
        1
    );
    assert!(messages
        .iter()
        .any(|message| message.content == "stable application prompt"));
    assert!(messages
        .iter()
        .find(|message| message.name.as_deref() == Some("execution_ledger"))
        .unwrap()
        .content
        .contains("workspace_generation: 25"));
}

#[test]
fn mutation_recovery_window_keeps_only_effect_capable_authorized_tools() {
    use super::execution::mutation_recovery_tool_definitions;

    let definitions = vec![
        json!({"type":"function","function":{"name":"file_read"}}),
        json!({"type":"function","function":{"name":"grep_search"}}),
        json!({"type":"function","function":{"name":"file_write"}}),
        json!({"type":"function","function":{"name":"file_edit"}}),
        json!({"type":"function","function":{"name":"bash"}}),
    ];
    let names: Vec<String> = mutation_recovery_tool_definitions(definitions)
        .iter()
        .filter_map(|definition| definition["function"]["name"].as_str().map(str::to_string))
        .collect();

    assert_eq!(names, vec!["file_write", "file_edit", "bash"]);
}

#[test]
fn da_final_turn_notice_still_allows_required_implementation() {
    use super::execution::final_turn_limit_notice;

    let notice = final_turn_limit_notice(AgentRole::Do, true, true, 7);
    assert!(notice.contains("file_write/file_edit"));
    assert!(notice.contains("no-change tail is 7"));
    assert!(!notice.contains("Do not initiate new tool calls"));

    let ca_notice = final_turn_limit_notice(AgentRole::Check, false, false, 0);
    assert!(ca_notice.contains("Do not initiate new tool calls"));
}

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

    let cwm = runner.context_window_manager.as_ref().expect("cwm created");
    let cwm = cwm.lock().unwrap();
    assert_eq!(cwm.max_tokens(), 8888);
}

#[test]
fn test_parse_jsonld_response_valid() {
    let runner = create_test_runner();
    let response = json!({
        "@context": "https://agent-os.org/context/task",
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
        verdict: None,
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
    assert_eq!(
        AgentRunner::detect_blocker_verdict(
            "成功标准『浏览器端到端验证通过』未达成。不得转化为成功，status is partial/blocked."
        ),
        Some("failed")
    );
    assert_eq!(
        AgentRunner::detect_blocker_verdict("FAILED: implementation was not completed"),
        Some("failed")
    );
    assert_eq!(AgentRunner::detect_blocker_verdict(""), None);
}

#[test]
fn terminal_reasoning_is_not_lost_when_responses_content_is_null() {
    assert_eq!(
        AgentRunner::effective_response_content(
            "",
            Some("FAILED: acceptance criteria were not met"),
            "stop",
            false,
        ),
        "FAILED: acceptance criteria were not met"
    );
    assert_eq!(
        AgentRunner::effective_response_content("", Some("calling a tool"), "tool_calls", true),
        ""
    );
}

#[test]
fn nullish_react_content_uses_terminal_reasoning_as_evidence() {
    let runner = create_test_runner();
    let parsed = runner.parse_llm_response(
        r#"{"content":"null","summary":"Cleanup verified","action":"finish"}"#,
        Some("All success criteria independently verified; 19/19 tests pass."),
        true,
    );

    assert_eq!(parsed.summary.as_deref(), Some("Cleanup verified"));
    assert_eq!(parsed.action.as_deref(), Some("finish"));
    assert_eq!(
        parsed.content,
        "All success criteria independently verified; 19/19 tests pass."
    );
}

#[test]
fn decision_only_context_exposes_no_tools_to_aa() {
    let runner = create_test_runner();
    let definitions = runner.tool_definitions_for_context("AA", Some(&[]));
    assert!(
        definitions.is_empty(),
        "AA deny-all context must not advertise tools to the model"
    );
}

#[test]
fn dynamic_result_readers_are_visible_only_to_the_owning_execution() {
    let runner = create_test_runner();
    let tool_name = "read_full_result_call_session_a";
    let evicted_tool_name = "read_full_result_call_session_old";
    {
        let mut executor = runner.tool_executor.write();
        executor.set_micro_tool_limits(1, 100, 200);
        for (name, call_id) in [
            (evicted_tool_name, "call_session_old"),
            (tool_name, "call_session_a"),
        ] {
            executor.register_micro_tool(
                name,
                crate::tools::tool_executor::MicroToolContext {
                    call_id: call_id.to_string(),
                    storage_key: format!("iri://tool-result/{call_id}"),
                    tool_name: "file_read".to_string(),
                    entity_types: vec![],
                    preview_size: 100,
                },
            );
        }
    }

    let names = |definitions: Vec<Value>| {
        definitions
            .into_iter()
            .filter_map(|definition| definition["function"]["name"].as_str().map(str::to_string))
            .collect::<std::collections::HashSet<_>>()
    };
    assert!(!names(runner.tool_definitions_for_context("DA", None)).contains(tool_name));

    let session_tools =
        std::collections::HashSet::from([tool_name.to_string(), evicted_tool_name.to_string()]);
    let owning_names =
        names(runner.tool_definitions_for_context_with_microtools("DA", None, &session_tools));
    assert!(owning_names.contains(tool_name));
    assert!(
        owning_names.contains(evicted_tool_name),
        "an owning session must retain a reconstructable schema after global catalog eviction"
    );
}

#[test]
fn only_currently_referenced_dynamic_readers_stay_in_the_tool_window() {
    use super::execution::active_session_tool_names;
    use crate::gateway::unified_gateway::ChatMessage;

    let active = "read_full_result_call_active".to_string();
    let stale = "read_full_result_call_stale".to_string();
    let discovered = "knowledge_import_directory".to_string();
    let session = std::collections::HashSet::from([active.clone(), stale, discovered.clone()]);
    let messages = vec![ChatMessage {
        role: "tool".to_string(),
        content: format!("Full result available via `{active}`"),
        name: None,
        tool_calls: None,
        tool_call_id: Some("call_active".to_string()),
        reasoning_content: None,
    }];

    let names = active_session_tool_names(&messages, &session);
    assert!(names.contains(&active));
    assert!(names.contains(&discovered));
    assert_eq!(names.len(), 2);
}

#[test]
fn da_verification_failure_enters_repair_and_mutation_requires_reverification() {
    use super::execution::{da_phase_after_tool_turn, ExecutionPhase};

    assert_eq!(
        da_phase_after_tool_turn(ExecutionPhase::Verify, false, true),
        ExecutionPhase::Repair
    );
    assert_eq!(
        da_phase_after_tool_turn(ExecutionPhase::Repair, true, false),
        ExecutionPhase::Verify
    );
    assert_eq!(
        da_phase_after_tool_turn(ExecutionPhase::Verify, true, false),
        ExecutionPhase::Verify,
        "a mutation made during verification must be verified, not left in repair"
    );
}

#[tokio::test]
async fn read_agent_output_reads_archived_tool_result_iri_with_a_bound() {
    let runner = create_test_runner();
    let iri = "iri://tool-result/archived-test";
    runner
        .l0_store
        .store(
            iri,
            &serde_json::json!({"content": "one\ntwo\nthree", "tool_name": "file_read"})
                .to_string(),
        )
        .unwrap();

    let executor = runner.tool_executor.read().clone();
    let result = executor
        .execute(
            "read_agent_output",
            serde_json::json!({"node_iri": iri, "offset": 1, "limit": 1}),
        )
        .await
        .unwrap();
    assert_eq!(result["content"], "two");
    assert_eq!(result["returned"], 1);
    assert_eq!(result["total_lines"], 3);
}

#[tokio::test]
async fn pass_through_result_advertises_an_iri_only_when_it_is_resolvable() {
    let runner = create_test_runner();
    let small = runner
        .route_tool_result("small inline result", "file_read", "small-call")
        .await;
    assert!(!small.contains("iri://tool-result/"));

    let large_payload =
        "line\n".repeat(runner.tool_result_router_settings.prepare_threshold / "line\n".len() + 2);
    let large = runner
        .route_tool_result(&large_payload, "file_read", "large-call")
        .await;
    assert!(large.contains("iri://tool-result/large-call"));

    let executor = runner.tool_executor.read().clone();
    let archived = executor
        .execute(
            "read_agent_output",
            serde_json::json!({
                "node_iri": "iri://tool-result/large-call",
                "offset": 0,
                "limit": 1
            }),
        )
        .await
        .unwrap();
    assert_eq!(archived["content"], "line");
}

#[test]
fn optimized_tool_window_is_small_and_can_activate_discovered_tools() {
    let runner = create_test_runner()
        .with_token_optimization(crate::config::settings::TokenOptimizationSettings::default())
        .with_prompt_variant(crate::core::prompt_contract::PromptVariant::Optimized);
    let names = |definitions: Vec<Value>| {
        definitions
            .into_iter()
            .filter_map(|definition| definition["function"]["name"].as_str().map(str::to_string))
            .collect::<std::collections::HashSet<_>>()
    };

    let initial = names(runner.tool_definitions_for_context("DA", None));
    let full = names(runner.tool_executor.read().tool_definitions_for_role("DA"));
    assert!(initial.len() < full.len());
    assert!(initial.contains("tool_search"));
    assert!(!initial.contains("knowledge_import_directory"));

    let activated = std::collections::HashSet::from(["knowledge_import_directory".to_string()]);
    let after_search =
        names(runner.tool_definitions_for_context_with_microtools("DA", None, &activated));
    assert!(after_search.contains("knowledge_import_directory"));
}

#[test]
fn role_turn_budgets_inherit_task_budget_unless_configured() {
    use super::execution::effective_role_max_turns;

    let mut budget = crate::config::settings::AgentExecutionBudgetSettings::default();
    for role in [
        AgentRole::Plan,
        AgentRole::Do,
        AgentRole::Check,
        AgentRole::Act,
    ] {
        assert_eq!(effective_role_max_turns(role, 50, &budget), 50);
    }

    budget.role_max_turns.plan = Some(12);
    budget.role_max_turns.check = Some(24);
    budget.role_max_turns.act = Some(8);
    assert_eq!(effective_role_max_turns(AgentRole::Plan, 50, &budget), 12);
    assert_eq!(effective_role_max_turns(AgentRole::Do, 50, &budget), 50);
    assert_eq!(effective_role_max_turns(AgentRole::Check, 50, &budget), 24);
    assert_eq!(effective_role_max_turns(AgentRole::Check, 10, &budget), 10);
    assert_eq!(effective_role_max_turns(AgentRole::Act, 50, &budget), 8);
}

#[test]
fn short_role_budget_never_emits_colliding_warning_phases() {
    use super::execution::turn_warning_thresholds;

    assert_eq!(turn_warning_thresholds(50, 8, 3), (Some(42), Some(47)));
    assert_eq!(turn_warning_thresholds(5, 8, 3), (None, Some(2)));
    assert_eq!(turn_warning_thresholds(2, 8, 3), (None, None));
}

#[test]
fn da_effect_guard_distinguishes_inspection_from_substantive_changes() {
    use super::execution::is_substantive_workspace_effect;

    assert!(is_substantive_workspace_effect(
        "file_write",
        &json!({"path": "src/new.rs", "content": "fn main() {}"})
    ));
    assert!(!is_substantive_workspace_effect(
        "bash",
        &json!({"command": "printf 'x' > src/generated.txt"})
    ));
    assert!(!is_substantive_workspace_effect(
        "bash",
        &json!({"command": "python -m pytest -q > pytest_run_now.txt 2>&1"})
    ));
    assert!(!is_substantive_workspace_effect(
        "bash",
        &json!({"command": "python -m pytest -q | tee pytest_run_now.txt"})
    ));
    assert!(!is_substantive_workspace_effect(
        "bash",
        &json!({"command": "sed -n '1,200p' src/lib.rs"})
    ));
    assert!(!is_substantive_workspace_effect(
        "bash",
        &json!({"command": "mkdir -p empty_only"})
    ));
    assert!(!is_substantive_workspace_effect(
        "bash",
        &json!({"command": "cp taskqueue.py taskqueue.py.bak"})
    ));
    assert!(!is_substantive_workspace_effect(
        "bash",
        &json!({"command": "cp -a 'taskqueue.py' 'taskqueue.py.orig'"})
    ));
    assert!(is_substantive_workspace_effect(
        "bash",
        &json!({"command": "cp taskqueue.py generated/taskqueue.py"})
    ));
    assert!(is_substantive_workspace_effect(
        "bash",
        &json!({"command": "cp taskqueue.py.bak taskqueue.py"})
    ));
    assert!(is_substantive_workspace_effect(
        "bash",
        &json!({"command": "cp taskqueue.py taskqueue.py.bak && sed -i 's/old/new/' taskqueue.py"})
    ));
    assert!(!is_substantive_workspace_effect(
        "file_read",
        &json!({"path": "src/lib.rs"})
    ));
}

#[test]
fn shell_effect_confirmation_rejects_noop_and_accepts_content_change() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            use super::execution::{capture_workspace_effect_snapshot, confirmed_workspace_effect};
            use crate::tools::tool_executor::ToolExecutor;
            use crate::tools::workspace_monitor::{WorkspaceMonitor, WorkspaceMonitorConfig};
            use std::sync::Arc;

            let dir = tempfile::Builder::new()
                .prefix(".semantic-effect-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let path = dir.path().join("source.txt");
            std::fs::write(&path, "old\n").unwrap();
            let monitor = WorkspaceMonitor::initialize(
                WorkspaceMonitorConfig {
                    workspace_root: dir.path().to_path_buf(),
                    watch_enabled: false,
                    ..Default::default()
                },
                None,
                None,
            )
            .unwrap();
            let mut executor = ToolExecutor::new();
            executor.set_workspace_monitor(Arc::new(monitor));
            let executor = parking_lot::RwLock::new(executor);

            let noop_args = json!({
                "command": format!("sed -i 's/missing/replacement/' '{}'", path.display())
            });
            let before = capture_workspace_effect_snapshot(&executor).unwrap();
            let noop_result = executor
                .read()
                .clone()
                .execute("bash", noop_args.clone())
                .await
                .unwrap();
            assert!(
                !confirmed_workspace_effect(
                    &executor,
                    "bash",
                    &noop_args,
                    &noop_result,
                    Some(&before),
                )
                .await
            );

            let change_args = json!({
                "command": format!("sed -i 's/old/new/' '{}'", path.display())
            });
            let before = capture_workspace_effect_snapshot(&executor).unwrap();
            let change_result = executor
                .read()
                .clone()
                .execute("bash", change_args.clone())
                .await
                .unwrap();
            assert!(
                confirmed_workspace_effect(
                    &executor,
                    "bash",
                    &change_args,
                    &change_result,
                    Some(&before),
                )
                .await
            );
            assert_eq!(std::fs::read_to_string(path).unwrap(), "new\n");
        });
}

#[test]
fn workspace_effect_guard_is_enabled_only_by_generic_task_constraint() {
    use super::execution::requires_workspace_effect;

    let plain = TaskContext::new("iri://task/plain", "analyze", 10);
    assert!(!requires_workspace_effect(&plain, AgentRole::Do));

    let change = TaskContext::new("iri://task/change", "execute", 10)
        .with_constraint("required_effect", "workspace_mutation");
    assert!(requires_workspace_effect(&change, AgentRole::Do));
    assert!(!requires_workspace_effect(&change, AgentRole::Check));
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
        verdict: None,
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
    let md = runner.build_agent_md(AgentRole::Do, "objective", &context_data, "deepseek-v4-pro");
    assert!(
        md.contains("## Workspace Files"),
        "DA prompt renders file manifest"
    );
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
        discovery_5w2h: None,
    });

    let tools = vec!["file_read".to_string(), "file_write".to_string()];
    let skills_text = AgentRunner::build_available_skills(&tools, &runner.skills, "DA", 10);
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
        discovery_5w2h: None,
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
        discovery_5w2h: None,
    });

    let tools = vec!["file_read".to_string()];
    let skills_text = AgentRunner::build_available_skills(&tools, &runner.skills, "DA", 10);
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
        discovery_5w2h: None,
    });

    let tools = vec!["file_read".to_string()];
    let skills_text = AgentRunner::build_available_skills(&tools, &runner.skills, "DA", 10);
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
        discovery_5w2h: None,
    });

    let context_data = std::collections::HashMap::new();
    let md = runner.build_agent_md(AgentRole::Do, "objective", &context_data, "deepseek-v4-pro");
    assert!(
        md.contains("## Available Skills") && md.contains("analyze_output"),
        "PromptLoader builtin-fallback agent.md must include injected skills, got: {}",
        md
    );
}
