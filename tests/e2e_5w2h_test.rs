//! 5W2H 端到端测试 - 验证完整流程
//!
//! 运行: cargo run --example e2e_5w2h_test

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use glidinghorse::config::settings::{AgentSettings, GatewaySettings};
use glidinghorse::core::agent_runner::AgentRunner;
use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::five_w2h::{
    FillStage, HowDetail, HowMuchDetail, Task5W2H, WhenDetail, WhereDetail, WhoDetail,
};
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::gateway::unified_gateway::UnifiedGateway;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::perception::proactive_engine::ProactiveEngine;
use glidinghorse::templates::template_engine::TemplateEngine;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::CoreConfig;

const OUTPUT_DIR: &str = "/tmp/agent_os_e2e_5w2h";

fn print_section(title: &str) {
    println!("\n{}", "=".repeat(60));
    println!("  {}", title);
    println!("{}", "=".repeat(60));
}

fn print_result(test_name: &str, passed: bool) {
    let status = if passed { "✅ PASS" } else { "❌ FAIL" };
    println!("  {} - {}", status, test_name);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Agent OS - 5W2H 端到端测试");
    println!("   验证 5W2H 完整生命周期功能\n");

    std::fs::create_dir_all(OUTPUT_DIR)?;
    std::fs::create_dir_all(format!("{}/l0", OUTPUT_DIR))?;

    let mut all_passed = true;

    // ========== 测试 1: 5W2H 初始化 ==========
    print_section("测试 1: 5W2H 初始化");

    let mut w2h = Task5W2H::new("实现用户认证系统", "提供安全的用户登录功能");

    let test1_1 = w2h.what == "实现用户认证系统";
    let test1_2 = !w2h.frozen;
    let test1_3 = w2h.dimension_meta.contains_key("what");
    let test1_4 = w2h.dimension_meta.contains_key("why");
    let test1_5 = w2h.dimension_meta.get("what").unwrap().fill_stage == FillStage::Create;

    print_result("What 字段正确", test1_1);
    print_result("初始状态未冻结", test1_2);
    print_result("dimension_meta 包含 what", test1_3);
    print_result("dimension_meta 包含 why", test1_4);
    print_result("what 填充阶段为 Create", test1_5);

    all_passed = all_passed && test1_1 && test1_2 && test1_3 && test1_4 && test1_5;

    // ========== 测试 2: 维度渐进填充 ==========
    print_section("测试 2: 维度渐进填充");

    w2h.record_fill("who", FillStage::Plan, "PA");
    w2h.record_fill("when", FillStage::Plan, "PA");
    w2h.record_fill("where", FillStage::Do, "DA");
    w2h.record_fill("how", FillStage::Do, "DA");
    w2h.record_fill("how_much", FillStage::Check, "CA");

    let test2_1 = w2h.dimension_meta.get("who").unwrap().fill_stage == FillStage::Plan;
    let test2_2 = w2h.dimension_meta.get("who").unwrap().filled_by == Some("PA".to_string());
    let test2_3 = w2h.dimension_meta.get("how").unwrap().fill_stage == FillStage::Do;
    let test2_4 = w2h.dimension_meta.get("how_much").unwrap().fill_stage == FillStage::Check;

    print_result("who 填充阶段为 Plan", test2_1);
    print_result("who 填充者为 PA", test2_2);
    print_result("how 填充阶段为 Do", test2_3);
    print_result("how_much 填充阶段为 Check", test2_4);

    all_passed = all_passed && test2_1 && test2_2 && test2_3 && test2_4;

    // ========== 测试 3: 完形校验 ==========
    print_section("测试 3: 完形校验");

    let missing_instant = w2h.check_completeness("Instant");
    let missing_simple = w2h.check_completeness("Simple");
    let missing_complex = w2h.check_completeness("Complex");

    let test3_1 = missing_instant.is_empty();
    let test3_2 = missing_simple.is_empty();
    let test3_3 = missing_complex.is_empty();

    print_result("Instant 级别完形校验通过", test3_1);
    print_result("Simple 级别完形校验通过", test3_2);
    print_result("Complex 级别完形校验通过", test3_3);

    all_passed = all_passed && test3_1 && test3_2 && test3_3;

    // ========== 测试 4: 冻结功能 ==========
    print_section("测试 4: 冻结功能");

    w2h.freeze();

    let test4_1 = w2h.frozen;

    print_result("5W2H 已冻结", test4_1);

    all_passed = all_passed && test4_1;

    // ========== 测试 5: JSON-LD 序列化 ==========
    print_section("测试 5: JSON-LD 序列化");

    let json_ld = w2h.to_json_ld("test-task-001")?;
    let iri = json_ld.get("@id").unwrap().as_str().unwrap();

    let test5_1 = iri == "iri://task/test-task-001/5w2h";
    let test5_2 = json_ld.get("@type").unwrap().as_str().unwrap() == "task:5W2H";
    let test5_3 = json_ld.get("task:frozen").unwrap().as_bool().unwrap();

    print_result("IRI 格式正确", test5_1);
    print_result("@type 正确", test5_2);
    print_result("frozen 字段正确", test5_3);

    all_passed = all_passed && test5_1 && test5_2 && test5_3;

    // ========== 测试 6: L0 存储 ==========
    print_section("测试 6: L0 存储");

    let l0 = Arc::new(L0Store::new(&format!("{}/l0", OUTPUT_DIR))?);

    l0.store(iri, &json_ld.to_string())?;

    let retrieved = l0.retrieve(iri)?.unwrap();
    let restored = Task5W2H::from_json_ld(&serde_json::from_str::<serde_json::Value>(
        &retrieved.content,
    )?)?;

    let test6_1 = restored.what == w2h.what;
    let test6_2 = restored.frozen;
    let test6_3 = restored.dimension_meta.len() == 7;

    print_result("L0 存储后恢复 what 正确", test6_1);
    print_result("L0 存储后恢复 frozen 正确", test6_2);
    print_result("L0 存储后恢复 dimension_meta 数量正确", test6_3);

    all_passed = all_passed && test6_1 && test6_2 && test6_3;

    // ========== 测试 7: reminder_before 功能 ==========
    print_section("测试 7: reminder_before 功能");

    let deadline = chrono::Utc::now() + chrono::Duration::minutes(20);
    let mut w2h_reminder = Task5W2H::new("紧急任务", "需要提醒");
    w2h_reminder = w2h_reminder.with_when(WhenDetail {
        deadline: Some(deadline),
        start_after: None,
        estimated_duration: None,
        timezone: None,
        reminder_before: Some("PT30M".to_string()),
    });

    let json_ld_reminder = w2h_reminder.to_json_ld("reminder-test")?;
    let reminder_iri = "iri://task/reminder-test/5w2h";
    l0.store(reminder_iri, &json_ld_reminder.to_string())?;

    let engine = ProactiveEngine::new(l0.clone(), Arc::new(EventBus::new(100)));
    let result = engine.check_5w2h_constraints(reminder_iri);

    let test7_1 = result.is_some();
    let test7_2 = result.unwrap().contains("DEADLINE_APPROACHING");

    print_result("reminder_before 触发提醒", test7_1);
    print_result("提醒内容正确", test7_2);

    all_passed = all_passed && test7_1 && test7_2;

    // ========== 测试 8: 任务分级 ==========
    print_section("测试 8: 任务分级");

    let l2 = Arc::new(Blackboard::new()?);
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
        l0.clone(),
        l2.clone(),
        proj.clone(),
        CoreConfig::default(),
    )));
    let skills = Arc::new(SkillRegistry::new());
    let tmpl = Arc::new(TemplateEngine::new(Path::new("/nonexistent")).unwrap());
    let settings = GatewaySettings {
        base_url: "http://localhost:3000".to_string(),
        api_key: "sk-test".to_string(),
        default_model: "deepseek-v4-flash".to_string(),
        timeout_seconds: 30,
        max_retries: 3,
        retry_base_ms: 500,
        model_mapping: HashMap::new(),
    };
    let gateway = Arc::new(UnifiedGateway::new(&settings)?);
    let agent_settings = AgentSettings::default();
    let runner = Arc::new(AgentRunner::new(
        gateway,
        skills.clone(),
        l2.clone(),
        l0.clone(),
        mm,
        tmpl.clone(),
        agent_settings,
    ));
    let sa = SupervisorAgent::new(runner, tmpl, skills, Arc::new(EventBus::new(100)), 10);

    let plan_instant = sa.analyze_task("Hello");
    let plan_simple = sa.analyze_task("What is the weather?");
    let plan_emergency = sa.analyze_task("Fix this bug");
    let plan_standard =
        sa.analyze_task("Build a web application with user authentication and database");

    let test8_1 = plan_instant.agent_sequence.len() == 1;
    let test8_2 = plan_simple.agent_sequence.len() == 1;
    let test8_3 = plan_emergency.agent_sequence.len() == 3;
    let test8_4 = plan_standard.agent_sequence.len() == 4;

    print_result("Instant 任务: 1 个 Agent", test8_1);
    print_result("Simple 任务: 1 个 Agent", test8_2);
    print_result("Emergency 任务: 3 个 Agent", test8_3);
    print_result("Standard 任务: 4 个 Agent", test8_4);

    all_passed = all_passed && test8_1 && test8_2 && test8_3 && test8_4;

    // ========== 测试 9: 5W2H 写入黑板 ==========
    print_section("测试 9: 5W2H 写入黑板");

    let test_w2h = Task5W2H::new("黑板测试", "验证黑板写入");
    let test_json_ld = test_w2h.to_json_ld("blackboard-test")?;
    let test_iri = "iri://task/blackboard-test/5w2h";

    let cfg = CoreConfig::default();
    let write_result = l2.write_node(test_iri, &test_json_ld.to_string(), &cfg);

    let test9_1 = write_result.is_ok();

    if test9_1 {
        let nodes = l2.query_nodes("iri://task/blackboard-test")?;
        let test9_2 = !nodes.is_empty();
        print_result("黑板写入成功", test9_1);
        print_result("黑板查询返回节点", test9_2);
        all_passed = all_passed && test9_1 && test9_2;
    } else {
        print_result("黑板写入成功", false);
        all_passed = false;
    }

    // ========== 测试 10: 完整 JSON-LD 往返 ==========
    print_section("测试 10: 完整 JSON-LD 往返");

    let mut full_w2h = Task5W2H::new("完整测试", "验证所有功能")
        .with_who(WhoDetail {
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

    full_w2h.record_fill("who", FillStage::Plan, "PA");
    full_w2h.record_fill("when", FillStage::Plan, "PA");
    full_w2h.record_fill("where", FillStage::Do, "DA");
    full_w2h.record_fill("how", FillStage::Do, "DA");
    full_w2h.record_fill("how_much", FillStage::Check, "CA");
    full_w2h.freeze();

    let full_json_ld = full_w2h.to_json_ld("full-test")?;
    let full_restored = Task5W2H::from_json_ld(&full_json_ld)?;

    let test10_1 = full_restored.what == "完整测试";
    let test10_2 = full_restored.frozen;
    let test10_3 = full_restored.dimension_meta.len() == 7;
    let test10_4 = full_restored
        .when
        .as_ref()
        .unwrap()
        .reminder_before
        .is_some();
    let test10_5 = full_restored.how.is_some();
    let test10_6 = full_restored.how_much.is_some();

    print_result("完整往返 what 正确", test10_1);
    print_result("完整往返 frozen 正确", test10_2);
    print_result("完整往返 dimension_meta 数量正确", test10_3);
    print_result("完整往返 reminder_before 正确", test10_4);
    print_result("完整往返 how 正确", test10_5);
    print_result("完整往返 how_much 正确", test10_6);

    all_passed = all_passed && test10_1 && test10_2 && test10_3 && test10_4 && test10_5 && test10_6;

    // ========== 最终结果 ==========
    print_section("测试结果汇总");

    if all_passed {
        println!("\n  ✅ 所有测试通过！\n");
        println!("  5W2H 功能验证完成:");
        println!("  - 初始化与渐进填充 ✓");
        println!("  - 完形校验 ✓");
        println!("  - 冻结归档 ✓");
        println!("  - JSON-LD 序列化 ✓");
        println!("  - L0 持久化存储 ✓");
        println!("  - reminder_before 提醒 ✓");
        println!("  - 任务分级 ✓");
        println!("  - 黑板写入 ✓");
        println!("  - 完整往返 ✓");
    } else {
        println!("\n  ❌ 部分测试失败，请检查上述日志\n");
    }

    println!("{}", "=".repeat(60));

    Ok(())
}
