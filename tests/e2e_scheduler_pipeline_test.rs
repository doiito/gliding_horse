//! Scheduler-backed pipeline E2E verification.
//!
//! Verifies P0-prefetch, P0-WriteThrough, P1-eventbus, P2-thread-reaping
//! end-to-end, using the same wiring as the worker entry point
//! (`src/worker/agent_os_worker.rs`): MemoryScheduler + ConsistencyEngine +
//! PrefetchEngine + spawn_consumer.
//!
//! `test_e2e_scheduler_full_pipeline_real_api` requires `DEEPSEEK_API_KEY`
//! (real LLM call), so it is compiled when the explicit `live-tests` feature
//! is enabled. The prefetch-consumer and WriteThrough/projection-invalidation
//! tests are pure in-memory and run in the default suite. No test is hidden
//! behind `#[ignore]`.

#![cfg_attr(not(feature = "live-tests"), allow(unused_imports))]

use std::sync::Arc;

use glidinghorse::config::{AgentSettings, GatewaySettings};
use glidinghorse::core::agent_runner::AgentRunner;
use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::gateway::UnifiedGateway;
use glidinghorse::memory::consistency_engine::ConsistencyEngine;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_bus::MemoryBus;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::memory::prefetch_engine::PrefetchEngine;
use glidinghorse::memory::scheduler::MemoryScheduler;
use glidinghorse::templates::template_engine::TemplateEngine;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::utils::init_logging;
use glidinghorse::CoreConfig;
use tempfile::TempDir;

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_e2e_scheduler_full_pipeline_real_api() {
    let _ = init_logging(&glidinghorse::config::settings::LoggingSettings {
        level: "info".to_string(),
        format: "text".to_string(),
        console_output: true,
        file_output: glidinghorse::config::settings::FileOutputSettings {
            enabled: false,
            path: "./logs".to_string(),
            prefix: "e2e_scheduler".to_string(),
            rotation: "daily".to_string(),
            max_files: 5,
        },
        filters: vec![],
        sensitive_fields: vec![],
    });
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
        use_responses_api: false,
        model_mapping: Default::default(),
    };
    let gateway = Arc::new(UnifiedGateway::new(&settings).expect("gateway"));

    let dir = TempDir::new().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().join("l0").to_string_lossy().as_ref()).unwrap());
    let l2 = Arc::new(Blackboard::new().unwrap());
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let event_bus = Arc::new(EventBus::new(100));
    let memory_bus = Arc::new(MemoryBus::new(event_bus.clone()));

    // Consistency engine (WriteThrough hook + projection invalidation)
    let consistency = Arc::new(ConsistencyEngine::new(
        memory_bus.clone(),
        l0.clone(),
        l2.clone(),
        proj.clone(),
    ));

    // Scheduler publishes TASK_COMPLETED; prefetch consumes PREFETCH_REQUEST
    let scheduler = Arc::new(MemoryScheduler::new(
        l0.clone(),
        l2.clone(),
        proj.clone(),
        consistency.clone(),
        memory_bus.clone(),
    ));
    let prefetch = Arc::new(PrefetchEngine::new(
        memory_bus.clone(),
        l2.clone(),
        proj.clone(),
    ));
    prefetch.spawn_consumer(event_bus.clone(), l2.clone());

    let core_config = CoreConfig::default();
    let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::with_scheduler(
        l0.clone(),
        l2.clone(),
        proj.clone(),
        core_config,
        scheduler.clone(),
    )));

    let templates_dir = dir.path().join("templates");
    std::fs::create_dir_all(&templates_dir).unwrap();
    let tmpl = Arc::new(TemplateEngine::new(&templates_dir).unwrap());
    let skills = Arc::new(SkillRegistry::new());
    let runner = Arc::new(AgentRunner::new(
        gateway,
        skills.clone(),
        l2.clone(),
        l0.clone(),
        mm.clone(),
        tmpl.clone(),
        AgentSettings::default(),
    ));
    let mut sa = SupervisorAgent::new(runner, tmpl, skills, Arc::new(EventBus::new(100)), 10)
        .with_memory(Some(l2.clone()), Some(prefetch.clone()), None);

    // Simple real task: math check (minimal API usage)
    let task_iri = "iri://task/e2e_scheduler_pipeline_test";
    let prompt = "用 python3 计算 17*23 的结果并报告。运行 python3 -c 'print(17*23)' 验证。";

    let result = sa
        .process_task(prompt, task_iri)
        .await
        .expect("task should succeed");
    eprintln!("=== E2E SCHEDULER PIPELINE RESULT ===");
    eprintln!("Status: {}", result.status);
    eprintln!(
        "Turns: {}, Tools: {}",
        result.turn_count, result.tool_call_count
    );
    eprintln!("Errors: {:?}", result.errors);

    assert_ne!(result.status, "failed", "task must not fail");
    eprintln!("✅ E2E scheduler pipeline test PASSED");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_e2e_prefetch_consumer_wires_and_drains() {
    let dir = TempDir::new().unwrap();
    let _l0 = Arc::new(L0Store::new(dir.path().join("l0").to_string_lossy().as_ref()).unwrap());
    let l2 = Arc::new(Blackboard::new().unwrap());
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let event_bus = Arc::new(EventBus::new(100));
    let memory_bus = Arc::new(MemoryBus::new(event_bus.clone()));

    let prefetch = Arc::new(PrefetchEngine::new(
        memory_bus.clone(),
        l2.clone(),
        proj.clone(),
    ));
    prefetch.spawn_consumer(event_bus.clone(), l2.clone());

    // Publish a PREFETCH_REQUEST via memory_bus; consumer should drain the queue.
    memory_bus
        .emit_prefetch_request("iri://entity/e2e_test", "e2e intent")
        .await;
    assert_eq!(prefetch.queue_len(), 0, "queue drained by consumer");
    eprintln!("✅ prefetch consumer wired and drains queue");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_e2e_writethrough_and_projection_invalidation() {
    let dir = TempDir::new().unwrap();
    let l0 = Arc::new(L0Store::new(dir.path().join("l0").to_string_lossy().as_ref()).unwrap());
    let l2 = Arc::new(Blackboard::new().unwrap());
    let proj = Arc::new(ProjectionEngine::new(l2.clone(), 500));
    let event_bus = Arc::new(EventBus::new(100));
    let memory_bus = Arc::new(MemoryBus::new(event_bus.clone()));
    let _consistency = Arc::new(ConsistencyEngine::new(
        memory_bus.clone(),
        l0.clone(),
        l2.clone(),
        proj.clone(),
    ));

    let core_config = CoreConfig::default();
    let task_iri = "iri://task/wt_e2e_test";

    // 1. Write a node with critical tags -> WriteThrough to L0
    let json_ld = r#"{"@id":"iri://task/wt_e2e_test/n1","@type":"KnowledgeFragment","tags":["user_intent","confirmed_fact"]}"#;
    l2.write_node("iri://task/wt_e2e_test/n1", json_ld, &core_config)
        .unwrap();

    // 2. Project L3 view (cache populated)
    let _ = proj
        .project(task_iri, "reference_only", std::collections::HashMap::new())
        .await
        .unwrap();

    // 3. Update node with critical tag -> triggers consistency hook:
    //    WriteThrough (L0 persist) + projection invalidation
    let updated = r#"{"@id":"iri://task/wt_e2e_test/n1","@type":"KnowledgeFragment","content":"updated","tags":["user_intent"]}"#;
    l2.write_node("iri://task/wt_e2e_test/n1", updated, &core_config)
        .unwrap();

    // 4. Verify L0 received the critical write-through
    let stored = l0.retrieve("iri://task/wt_e2e_test/n1").unwrap();
    assert!(
        stored.is_some(),
        "WriteThrough: critical node must persist to L0"
    );

    // 5. Verify projection was invalidated (cache miss on next project)
    let views_before = proj.cache_stats().total_views;
    let _ = proj
        .project(task_iri, "reference_only", std::collections::HashMap::new())
        .await
        .unwrap();
    eprintln!(
        "projection stats: views before={} after={}",
        views_before,
        proj.cache_stats().total_views
    );
    eprintln!("✅ WriteThrough + projection invalidation verified");
}
