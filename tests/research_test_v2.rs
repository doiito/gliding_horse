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

const OUTPUT: &str = "/tmp/agent_os_research2";
const USER_INPUT: &str = "AI Agent在安防监控场景有哪些好的应用？";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n==============================================");
    println!("  Agent OS — 调研类问题测试 v2");
    println!("  输入: {}", USER_INPUT);
    println!("==============================================\n");

    std::fs::create_dir_all(OUTPUT)?;
    let key = std::env::var("DEEPSEEK_API_KEY")?;
    let url = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

    let gateway_settings = glidinghorse::config::GatewaySettings {
        base_url: url,
        api_key: key,
        default_model: "deepseek-chat".to_string(),
        timeout_seconds: 180,
        max_retries: 3,
        retry_base_ms: 500,
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

    let mut gate = SyscallGate::new(skills, 2048);
    gate.add_to_whitelist("da", "iri://skills/file_read");
    gate.add_to_whitelist("da", "iri://skills/file_write");

    println!("\n[AgentRunner] 开始执行...\n");

    let mut agent = AgentInstance::new("da_1".to_string(), AgentRole::Do);
    let context = TaskContext {
        task_iri: "iri://task/research".to_string(),
        objective: USER_INPUT.to_string(),
        max_iterations: 10,
        ..Default::default()
    };

    let result = runner.execute(&mut agent, context).await?;
    if !result.errors.is_empty() {
        println!("[工具错误]");
        for e in &result.errors {
            println!("  - {}", e);
        }
    }

    println!(
        "\n[结果] status={}, turns={}, tools={}",
        result.status, result.turn_count, result.tool_call_count
    );

    println!("\n[完整回答]");
    println!("{}", "=".repeat(60));
    println!("{}", result.summary);
    println!("{}", "=".repeat(60));

    assert!(gate
        .validate_call("da", "iri://skills/file_read", r#"{"path":"x"}"#)
        .is_ok());
    assert!(gate
        .validate_call("unknown", "iri://skills/file_read", r#"{"path":"x"}"#)
        .is_err());
    let _ = gate.sync_whitelist_to_oxigraph(&l2, "da");
    println!("\n  SyscallGate: ✓");

    println!("\n==============================================");
    println!("  完成!");
    println!("==============================================");
    Ok(())
}
