use std::path::Path;
use std::sync::Arc;

use glidinghorse::config::AgentSettings;
use glidinghorse::config::GatewaySettings;
use glidinghorse::core::agent_instance::AgentRole;
use glidinghorse::core::agent_runner::{AgentRunner, TaskContext};
use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::gateway::UnifiedGateway;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::templates::template_engine::TemplateEngine;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::CoreConfig;
use tempfile::TempDir;

fn build_system(max_iterations: u32) -> (SupervisorAgent, TempDir) {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let base_url = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

    let settings = GatewaySettings {
        base_url,
        api_key,
        default_model: "deepseek-v4-flash".to_string(),
        timeout_seconds: 120,
        max_retries: 2,
        retry_base_ms: 500,
        model_mapping: Default::default(),
    };

    let gateway = Arc::new(UnifiedGateway::new(&settings).expect("Failed to create gateway"));
    let dir = TempDir::new().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().join("l0").to_string_lossy().as_ref()).unwrap());
    let l2 = Arc::new(Blackboard::new().unwrap());
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let core_config = CoreConfig::default();
    let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
        l0.clone(),
        l2.clone(),
        proj.clone(),
        core_config,
    )));
    let templates_dir = dir.path().join("templates");
    std::fs::create_dir_all(&templates_dir).unwrap();
    let tmpl = Arc::new(TemplateEngine::new(&templates_dir).unwrap());
    let skills = Arc::new(SkillRegistry::new());
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
    let sa = SupervisorAgent::new(
        runner,
        tmpl,
        skills,
        Arc::new(EventBus::new(100)),
        max_iterations,
    )
    .with_memory(Some(l2), None, None);
    (sa, dir)
}

fn build_runner() -> (Arc<AgentRunner>, TempDir) {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let base_url = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

    let settings = GatewaySettings {
        base_url,
        api_key,
        default_model: "deepseek-v4-flash".to_string(),
        timeout_seconds: 120,
        max_retries: 2,
        retry_base_ms: 500,
        model_mapping: Default::default(),
    };

    let gateway = Arc::new(UnifiedGateway::new(&settings).unwrap());
    let dir = TempDir::new().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().join("l0").to_string_lossy().as_ref()).unwrap());
    let l2 = Arc::new(Blackboard::new().unwrap());
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let core_config = CoreConfig::default();
    let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
        l0.clone(),
        l2.clone(),
        proj.clone(),
        core_config,
    )));
    let templates_dir = dir.path().join("templates");
    std::fs::create_dir_all(&templates_dir).unwrap();
    let tmpl = Arc::new(TemplateEngine::new(&templates_dir).unwrap());
    let skills = Arc::new(SkillRegistry::new());
    let agent_settings = AgentSettings::default();
    let runner = Arc::new(AgentRunner::new(
        gateway,
        skills.clone(),
        l2,
        l0,
        mm,
        tmpl.clone(),
        agent_settings,
    ));
    (runner, dir)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_e2e_da_single_agent_programming() {
    let (runner, _dir) = build_runner();

    let mut agent = glidinghorse::core::agent_instance::AgentInstance::new(
        "e2e_da_001".to_string(),
        AgentRole::Do,
    );

    let ctx = TaskContext::new(
        "iri://task/e2e_da",
        "Write a Python function that checks if a number is prime. Save it to /tmp/agent_os_e2e/prime.py and test it with the number 17.",
        8,
    );

    let result = runner.execute(&mut agent, ctx).await;

    assert!(
        result.is_ok(),
        "DA execution should succeed: {:?}",
        result.err()
    );
    let task_result = result.unwrap();
    eprintln!("=== DA Single Agent Result ===");
    eprintln!("Status: {}", task_result.status);
    eprintln!("Summary: {}", task_result.summary);
    eprintln!(
        "Turns: {}, Tool calls: {}",
        task_result.turn_count, task_result.tool_call_count
    );
    eprintln!("Errors: {:?}", task_result.errors);

    assert_eq!(task_result.status, "success", "DA task should succeed");

    let prime_path = Path::new("/tmp/agent_os_e2e/prime.py");
    if prime_path.exists() {
        let content = std::fs::read_to_string(prime_path).unwrap();
        eprintln!("=== Generated prime.py ===\n{}", content);
        assert!(
            content.contains("prime") || content.contains("Prime"),
            "File should contain prime function"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_e2e_sa_simple_task() {
    let (mut sa, _dir) = build_system(10);

    let user_input = "What is 2+2? Just answer the question.";

    let task_iri = "iri://task/e2e_simple";
    let result = sa.process_task(user_input, task_iri).await;

    assert!(
        result.is_ok(),
        "Task should complete without error: {:?}",
        result.err()
    );
    let task_result = result.unwrap();
    eprintln!("=== E2E Simple Task Result ===");
    eprintln!("Status: {}", task_result.status);
    eprintln!("Summary: {}", task_result.summary);
    eprintln!(
        "Turns: {}, Tool calls: {}",
        task_result.turn_count, task_result.tool_call_count
    );

    assert_eq!(task_result.status, "success", "Simple task should succeed");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_e2e_sa_programming_full_pipeline() {
    std::fs::create_dir_all("/tmp/agent_os_e2e").ok();
    let (mut sa, _dir) = build_system(10);

    let user_input = "Write a Python function called 'fibonacci' that takes an integer n and returns the nth Fibonacci number. Save it to /tmp/agent_os_e2e/fibonacci.py and then run it with n=10 to verify it works correctly.";

    let task_iri = "iri://task/e2e_full";
    let result = sa.process_task(user_input, task_iri).await;

    assert!(
        result.is_ok(),
        "Task should complete without error: {:?}",
        result.err()
    );
    let task_result = result.unwrap();
    eprintln!("=== E2E Full Pipeline Result ===");
    eprintln!("Status: {}", task_result.status);
    eprintln!("Summary: {}", task_result.summary);
    eprintln!(
        "Turns: {}, Tool calls: {}",
        task_result.turn_count, task_result.tool_call_count
    );
    eprintln!("Errors: {:?}", task_result.errors);

    assert_ne!(
        task_result.status, "failed",
        "Task should not fail completely"
    );

    let fib_path = Path::new("/tmp/agent_os_e2e/fibonacci.py");
    if fib_path.exists() {
        let content = std::fs::read_to_string(fib_path).unwrap();
        eprintln!("=== Generated fibonacci.py ===\n{}", content);
        assert!(
            content.contains("fibonacci") || content.contains("Fibonacci"),
            "File should contain fibonacci function"
        );
    } else {
        eprintln!("WARNING: fibonacci.py was not created at expected path");
    }
}
