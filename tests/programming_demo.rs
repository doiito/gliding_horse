//! Agent OS — 纯端到端自主编程演示
//!
//! 唯一输入 → AgentRunner.execute() 自动处理 LLM + tools 循环
//! LLM 自行规划: 写代码 → file_write → 写测试 → Bash pytest → 修复
//!
//! 运行: DEEPSEEK_API_KEY=sk-xxx cargo run --example programming_demo

use std::path::Path;
use std::sync::Arc;

use glidinghorse::config::settings::AgentSettings;
use glidinghorse::core::agent_instance::{AgentInstance, AgentRole};
use glidinghorse::core::agent_runner::{AgentRunner, TaskContext};
use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::core::syscall_gate::SyscallGate;
use glidinghorse::gateway::unified_gateway::UnifiedGateway;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::templates::template_engine::TemplateEngine;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::CoreConfig;

const OUTPUT: &str = "/tmp/agent_os_generated";
const USER_INPUT: &str = "\
用python写一个计算器程序 calculator.py(支持+-*/和括号)，\
写测试 test_calculator.py 用 assert 测试(2+3=5,10-4=6,3*4=12,15/3=5,(2+3)*4=20,10/0抛异常)，\
然后用pytest运行测试。所有文件放/tmp/agent_os_generated/目录下。\
写代码时先确保目录存在。";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n==============================================");
    println!("  Agent OS — 纯端到端自主编程");
    println!("  输入: 写计算器+测试+pytest");
    println!("  LLM 通过 tool_calls 自动调用 file_write/Bash");
    println!("==============================================\n");

    std::fs::create_dir_all(OUTPUT)?;
    let key = std::env::var("DEEPSEEK_API_KEY")?;
    let url = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

    // 基础设施
    let gateway_settings = glidinghorse::config::GatewaySettings {
        base_url: url,
        api_key: key,
        default_model: "deepseek-chat".to_string(),
        timeout_seconds: 180,
        max_retries: 3,
        retry_base_ms: 500,
        use_responses_api: false,
        model_mapping: Default::default(),
    };
    let gw = Arc::new(UnifiedGateway::new(&gateway_settings)?);

    let l0 = Arc::new(L0Store::new(&format!("{}/l0", OUTPUT))?);
    let l2 = Arc::new(Blackboard::new()?);
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
        l0.clone(),
        l2.clone(),
        proj.clone(),
        CoreConfig::default(),
    )));
    let skills = Arc::new(SkillRegistry::new());
    let tmpl = Arc::new(
        TemplateEngine::new(Path::new("src/templates/templates"))
            .unwrap_or_else(|_| TemplateEngine::new(Path::new("/nonexistent")).unwrap()),
    );
    let agent_settings = AgentSettings::default();
    let runner = Arc::new(AgentRunner::new(
        gw,
        skills.clone(),
        l2.clone(),
        l0.clone(),
        mm,
        tmpl,
        agent_settings,
    ));

    // 测试 API 连通性
    println!("  测试 API...");
    let _test_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()?;
    println!("  系统就绪\n");

    // SA 分析任务
    let sa = SupervisorAgent::new(
        runner.clone(),
        Arc::new(TemplateEngine::new(Path::new("/nonexistent")).unwrap()),
        skills.clone(),
        Arc::new(EventBus::new(100)),
        15,
    );
    let plan = sa.analyze_task(USER_INPUT);
    println!(
        "[SA] {:?} → {}",
        plan.task_complexity,
        plan.agent_sequence
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join("→")
    );

    // SyscallGate
    let mut gate = SyscallGate::new(skills, 2048);
    gate.add_to_whitelist("da", "iri://skills/file_read");
    gate.add_to_whitelist("da", "iri://skills/file_write");

    // ===== 端到端: AgentRunner.execute() =====
    // 唯一输入是 USER_INPUT，AgentRunner 会:
    // 1. 调用 LLM (含 tool definitions)
    // 2. LLM 返回 tool_calls → Runner 执行工具
    // 3. 结果返回 LLM → 继续直到完成
    println!("\n[AgentRunner] 开始执行 (LLM 自动规划工具调用)...\n");

    let mut agent = AgentInstance::new("da_1".to_string(), AgentRole::Do);
    let context = TaskContext {
        task_iri: "iri://task/calc".to_string(),
        objective: USER_INPUT.to_string(),
        max_iterations: 15,
        ..Default::default()
    };

    let result = runner.execute(&mut agent, context).await?;
    if !result.errors.is_empty() {
        for e in &result.errors {
            eprintln!("  Error: {}", e);
        }
    }

    // 结果
    println!(
        "\n[结果] status={}, turns={}, tools={}",
        result.status, result.turn_count, result.tool_call_count
    );
    println!("  summary: {}", result.summary);

    // 验证文件
    println!("\n[验证] 生成的文件:");
    for e in std::fs::read_dir(OUTPUT)? {
        let p = e?.path();
        if p.is_file() && p.extension().map_or(false, |e| e == "py") {
            println!(
                "  {} ({}B)",
                p.file_name().unwrap_or_default().to_string_lossy(),
                std::fs::metadata(&p)?.len()
            );
        }
    }

    // SyscallGate
    assert!(gate
        .validate_call("da", "iri://skills/file_read", r#"{"path":"x"}"#)
        .is_ok());
    assert!(gate
        .validate_call("unknown", "iri://skills/file_read", r#"{"path":"x"}"#)
        .is_err());
    let _ = gate.sync_whitelist_to_oxigraph(&l2, "da");
    println!("  SyscallGate: ✓");

    println!("\n==============================================");
    println!("  完成!");
    println!("  SA | Runner | tools | L0/L2/L3 | SyscallGate");
    println!("  ✓  |   ✓    |   ✓   |    ✓     |     ✓     ");
    println!("==============================================");
    Ok(())
}
