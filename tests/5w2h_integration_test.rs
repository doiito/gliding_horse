use std::collections::HashMap;
use std::sync::Arc;

use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::five_w2h::{
    FillStage, HowDetail, HowMuchDetail, Task5W2H, WhenDetail, WhereDetail,
};
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::perception::proactive_engine::ProactiveEngine;

fn setup_test_env() -> (Arc<L0Store>, Arc<Blackboard>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().join("l0").to_string_lossy().as_ref()).unwrap());
    let l2 = Arc::new(Blackboard::new().unwrap());
    (l0, l2, dir)
}

#[test]
fn test_5w2h_full_lifecycle() {
    let (l0, _l2, _dir) = setup_test_env();

    let mut w2h = Task5W2H::new("实现用户认证系统", "提供安全的用户登录功能");

    assert!(!w2h.frozen);
    assert!(w2h.dimension_meta.contains_key("what"));
    assert!(w2h.dimension_meta.contains_key("why"));
    assert_eq!(
        w2h.dimension_meta.get("what").unwrap().fill_stage,
        FillStage::Create
    );

    w2h.record_fill("who", FillStage::Plan, "PA");
    assert!(w2h.dimension_meta.contains_key("who"));
    assert_eq!(
        w2h.dimension_meta.get("who").unwrap().fill_stage,
        FillStage::Plan
    );
    assert_eq!(
        w2h.dimension_meta.get("who").unwrap().filled_by,
        Some("PA".to_string())
    );

    w2h.record_fill("when", FillStage::Plan, "PA");
    w2h.record_fill("where", FillStage::Do, "DA");
    w2h.record_fill("how", FillStage::Do, "DA");
    w2h.record_fill("how_much", FillStage::Check, "CA");

    let missing = w2h.check_completeness("Complex");
    assert!(
        missing.is_empty(),
        "Complex task should have all dimensions filled"
    );

    w2h.freeze();
    assert!(w2h.frozen, "5W2H should be frozen after freeze()");

    let json_ld = w2h.to_json_ld("test-task-001").unwrap();
    let iri = json_ld.get("@id").unwrap().as_str().unwrap();
    assert_eq!(iri, "iri://task/test-task-001/5w2h");

    l0.store(iri, &json_ld.to_string()).unwrap();

    let retrieved = l0.retrieve(iri).unwrap().unwrap();
    let restored = Task5W2H::from_json_ld(
        &serde_json::from_str::<serde_json::Value>(&retrieved.content).unwrap(),
    )
    .unwrap();

    assert_eq!(restored.what, w2h.what);
    assert!(restored.frozen);
    assert_eq!(restored.dimension_meta.len(), 7);
}

#[test]
fn test_5w2h_completeness_by_task_level() {
    let mut w2h = Task5W2H::new("简单查询", "获取信息");

    let missing_instant = w2h.check_completeness("Instant");
    assert!(missing_instant.is_empty(), "Instant task only needs what");

    let missing_simple = w2h.check_completeness("Simple");
    assert!(missing_simple.is_empty(), "Simple task needs what + why");

    let missing_standard = w2h.check_completeness("Standard");
    assert!(
        !missing_standard.is_empty(),
        "Standard task needs all dimensions"
    );
    assert!(missing_standard.contains(&"who".to_string()));
    assert!(missing_standard.contains(&"when".to_string()));

    w2h.record_fill("who", FillStage::Plan, "PA");
    w2h.record_fill("when", FillStage::Plan, "PA");
    w2h.record_fill("where", FillStage::Do, "DA");
    w2h.record_fill("how", FillStage::Do, "DA");
    w2h.record_fill("how_much", FillStage::Check, "CA");

    let missing_after_fill = w2h.check_completeness("Complex");
    assert!(missing_after_fill.is_empty(), "All dimensions filled");
}

#[test]
fn test_5w2h_reminder_before() {
    let (l0, _l2, _dir) = setup_test_env();
    let engine = ProactiveEngine::new(l0.clone(), Arc::new(EventBus::new(100)));

    let deadline = chrono::Utc::now() + chrono::Duration::minutes(20);
    let mut w2h = Task5W2H::new("紧急任务", "需要提醒");
    w2h = w2h.with_when(WhenDetail {
        deadline: Some(deadline),
        start_after: None,
        estimated_duration: None,
        timezone: None,
        reminder_before: Some("PT30M".to_string()),
    });

    let json_ld = w2h.to_json_ld("reminder-test").unwrap();
    let iri = "iri://task/reminder-test/5w2h";
    l0.store(iri, &json_ld.to_string()).unwrap();

    let result = engine.check_5w2h_constraints(iri);
    assert!(
        result.is_some(),
        "Should trigger reminder when within reminder_before window"
    );
    assert!(result.unwrap().contains("DEADLINE_APPROACHING"));
}

#[test]
fn test_5w2h_dimension_meta_tracking() {
    let mut w2h = Task5W2H::new("测试任务", "验证元数据追踪");

    let meta_before = w2h.dimension_meta.get("what").unwrap();
    assert_eq!(meta_before.fill_stage, FillStage::Create);
    assert_eq!(meta_before.filled_by, Some("SA".to_string()));

    w2h.record_fill("how", FillStage::Plan, "PA");
    let how_meta = w2h.dimension_meta.get("how").unwrap();
    assert_eq!(how_meta.fill_stage, FillStage::Plan);
    assert_eq!(how_meta.filled_by, Some("PA".to_string()));
    assert!(how_meta.filled_at.is_some());

    w2h.record_fill("how_much", FillStage::Check, "CA");
    let hm_meta = w2h.dimension_meta.get("how_much").unwrap();
    assert_eq!(hm_meta.fill_stage, FillStage::Check);
    assert_eq!(hm_meta.filled_by, Some("CA".to_string()));
}

#[test]
fn test_5w2h_frozen_prevents_modification() {
    let mut w2h = Task5W2H::new("冻结测试", "验证冻结功能");

    w2h.freeze();
    assert!(w2h.frozen);

    let json_ld = w2h.to_json_ld("frozen-test").unwrap();
    let restored = Task5W2H::from_json_ld(&json_ld).unwrap();
    assert!(restored.frozen);
}

#[test]
fn test_task_complexity_via_analyze_task() {
    let (l0, l2, _dir) = setup_test_env();

    use glidinghorse::config::settings::{AgentSettings, GatewaySettings};
    use glidinghorse::core::agent_runner::AgentRunner;
    use glidinghorse::core::event_bus::EventBus;
    use glidinghorse::gateway::unified_gateway::UnifiedGateway;
    use glidinghorse::memory::l3_projection::ProjectionEngine;
    use glidinghorse::memory::memory_manager::MemoryManager;
    use glidinghorse::templates::template_engine::TemplateEngine;
    use glidinghorse::tools::skill_registry::SkillRegistry;
    use glidinghorse::CoreConfig;

    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
        l0.clone(),
        l2.clone(),
        proj.clone(),
        CoreConfig::default(),
    )));
    let skills = Arc::new(SkillRegistry::new());
    let tmpl = Arc::new(TemplateEngine::new(std::path::Path::new("/nonexistent")).unwrap());
    let settings = GatewaySettings {
        base_url: "http://localhost:3000".to_string(),
        api_key: "sk-test".to_string(),
        default_model: "deepseek-v4-flash".to_string(),
        timeout_seconds: 30,
        max_retries: 3,
        retry_base_ms: 500,
        use_responses_api: false,
        model_mapping: HashMap::new(),
    };
    let gateway = Arc::new(UnifiedGateway::new(&settings).unwrap());
    let agent_settings = AgentSettings::default();
    let runner = Arc::new(AgentRunner::new(
        gateway,
        skills.clone(),
        l2.clone(),
        l0,
        mm,
        tmpl.clone(),
        agent_settings,
    ));
    let sa = SupervisorAgent::new(runner, tmpl, skills, Arc::new(EventBus::new(100)), 10);

    let plan_instant = sa.analyze_task("Hello");
    assert_eq!(plan_instant.agent_sequence.len(), 1);

    let plan_simple = sa.analyze_task("What is the weather?");
    assert_eq!(plan_simple.agent_sequence.len(), 1);

    let plan_emergency = sa.analyze_task("Fix this bug");
    assert_eq!(plan_emergency.agent_sequence.len(), 3);

    let plan_standard = sa
        .analyze_task("Build a web application with user authentication and database integration");
    assert_eq!(plan_standard.agent_sequence.len(), 4);

    // 调研类问题应该是 Standard
    let plan_research = sa.analyze_task("AI Agent在安防监控场景有哪些好的应用？");
    assert_eq!(plan_research.agent_sequence.len(), 4);
}

#[test]
fn test_5w2h_jsonld_roundtrip_with_all_features() {
    let mut w2h = Task5W2H::new("完整测试", "验证所有功能")
        .with_who(glidinghorse::core::five_w2h::WhoDetail {
            requestor: Some("user:test".to_string()),
            assignees: vec!["agent:pa".to_string()],
            stakeholders: vec![],
            required_role: Some("Do".to_string()),
            access_level: Some(glidinghorse::core::five_w2h::AccessLevel::Write),
        })
        .with_when(WhenDetail {
            deadline: Some(chrono::Utc::now() + chrono::Duration::hours(24)),
            start_after: None,
            estimated_duration: Some("2h".to_string()),
            timezone: Some("Asia/Shanghai".to_string()),
            reminder_before: Some("PT1H".to_string()),
        })
        .with_where(WhereDetail {
            data_sources: vec!["src/".to_string()],
            execution_environment: Some("sandbox".to_string()),
            target_repository: None,
            target_branch: None,
        })
        .with_how(HowDetail {
            plan_iri: Some("iri://plan/1".to_string()),
            preferred_skills: vec!["file_read".to_string()],
            forbidden_tools: vec![],
            required_steps: Some("1.分析 2.实现 3.测试".to_string()),
            dependencies: vec![],
        })
        .with_how_much(HowMuchDetail {
            token_budget: Some(50000),
            max_sub_agents: Some(3),
            max_pdca_cycles: Some(2),
            expected_quality: Some(0.9),
            actual_cost: None,
        });

    w2h.record_fill("who", FillStage::Plan, "PA");
    w2h.record_fill("when", FillStage::Plan, "PA");
    w2h.record_fill("where", FillStage::Do, "DA");
    w2h.record_fill("how", FillStage::Do, "DA");
    w2h.record_fill("how_much", FillStage::Check, "CA");

    w2h.freeze();

    let json_ld = w2h.to_json_ld("full-test").unwrap();
    let restored = Task5W2H::from_json_ld(&json_ld).unwrap();

    assert_eq!(restored.what, "完整测试");
    assert!(restored.frozen);
    assert_eq!(restored.dimension_meta.len(), 7);
    let when = restored.when.unwrap();
    assert!(when.reminder_before.is_some());
    assert_eq!(when.reminder_before.unwrap(), "PT1H");
}
