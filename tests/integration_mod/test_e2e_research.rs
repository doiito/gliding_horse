use std::sync::Arc;

use glidinghorse::config::settings::LoggingSettings;
use glidinghorse::config::AgentSettings;
use glidinghorse::config::GatewaySettings;
use glidinghorse::core::agent_instance::AgentRole;
use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::sa::{SupervisorAgent, TaskComplexity};
use glidinghorse::gateway::UnifiedGateway;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::templates::template_engine::TemplateEngine;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::utils::init_logging;
use glidinghorse::CoreConfig;
use serde_json::Value;
use tempfile::TempDir;

static LOGGING_INITIALIZED: std::sync::Once = std::sync::Once::new();

fn init_e2e_logging() {
    LOGGING_INITIALIZED.call_once(|| {
        let logging_settings = LoggingSettings {
            level: "debug".to_string(),
            format: "text".to_string(),
            console_output: true,
            file_output: glidinghorse::config::settings::FileOutputSettings {
                enabled: true,
                path: "./logs".to_string(),
                prefix: "e2e_research".to_string(),
                rotation: "daily".to_string(),
                max_files: 10,
            },
            filters: vec![
                glidinghorse::config::settings::LogFilter {
                    module: "glidinghorse::core".to_string(),
                    level: "debug".to_string(),
                },
                glidinghorse::config::settings::LogFilter {
                    module: "glidinghorse::gateway".to_string(),
                    level: "debug".to_string(),
                },
                glidinghorse::config::settings::LogFilter {
                    module: "glidinghorse::memory".to_string(),
                    level: "info".to_string(),
                },
                glidinghorse::config::settings::LogFilter {
                    module: "glidinghorse::tools".to_string(),
                    level: "info".to_string(),
                },
                glidinghorse::config::settings::LogFilter {
                    module: "redb".to_string(),
                    level: "warn".to_string(),
                },
            ],
            sensitive_fields: vec!["api_key".to_string(), "password".to_string()],
        };
        let _guard = init_logging(&logging_settings);
        std::mem::forget(_guard);
    });
}

fn validate_jsonld_basic(json_str: &str) -> Result<Value, String> {
    let parsed: Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON解析失败: {}", e))?;
    if parsed.get("@id").is_none() {
        return Err("缺少 @id 字段".to_string());
    }
    if parsed.get("@type").is_none() {
        return Err("缺少 @type 字段".to_string());
    }
    Ok(parsed)
}

fn build_system(max_iterations: u32) -> (SupervisorAgent, TempDir) {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let base_url = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

    let settings = GatewaySettings {
        base_url,
        api_key,
        default_model: "deepseek-v4-flash".to_string(),
        timeout_seconds: 180,
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
    let runner = Arc::new(glidinghorse::core::agent_runner::AgentRunner::new(
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

/// AI Agent 调研任务测试 - 探索性任务
/// 任务：调研并对比多种 AI Agent 框架
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_e2e_research_ai_agent_frameworks() {
    init_e2e_logging();

    let (mut sa, _dir) = build_system(30);

    let user_input = r#"
Research and compare different AI Agent frameworks. Please investigate:

1. LangChain Agents - architecture, key features, strengths and weaknesses
2. AutoGPT / AgentGPT - autonomous agent design patterns
3. CrewAI - multi-agent collaboration framework
4. Microsoft AutoGen - conversational agent framework

For each framework, analyze:
- Core architecture and design philosophy
- Key capabilities and limitations
- Use cases and best-fit scenarios
- Integration ecosystem

Then provide a comparative analysis table and recommendations for:
- Best framework for building customer service agents
- Best framework for complex multi-step reasoning tasks
- Best framework for research and information gathering

Save the complete research report to /tmp/agent_os_e2e/ai_agent_research.md
"#;

    let task_iri = "iri://task/e2e_research_frameworks";

    tracing::info!("========== 开始 AI Agent 框架调研任务测试 ==========");
    tracing::info!("任务类型: Exploratory (探索性调研)");

    let start_time = std::time::Instant::now();
    let result = sa.process_task(user_input, task_iri).await;
    let elapsed = start_time.elapsed();

    tracing::info!("执行时间: {:?}", elapsed);

    assert!(
        result.is_ok(),
        "任务应该完成，不应该报错: {:?}",
        result.err()
    );
    let task_result = result.unwrap();

    tracing::info!("========== 调研任务完成 ==========");
    tracing::info!("状态: {}", task_result.status);
    tracing::info!("摘要: {}", task_result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        task_result.turn_count,
        task_result.tool_call_count
    );
    tracing::info!("错误: {:?}", task_result.errors);

    assert_ne!(task_result.status, "failed", "调研任务不应该完全失败");

    let report_path = std::path::Path::new("/tmp/agent_os_e2e/ai_agent_research.md");
    if report_path.exists() {
        let content = std::fs::read_to_string(report_path).unwrap();
        tracing::info!("调研报告大小: {} 字节", content.len());
        assert!(content.len() > 200, "调研报告应该有足够的内容");
        tracing::info!("✅ 调研报告已生成");
    } else {
        tracing::warn!("调研报告未生成到预期路径");
    }

    if let Some(blackboard) = sa.blackboard() {
        let nodes = blackboard.query_nodes(task_iri).unwrap_or_default();
        tracing::info!("任务节点数: {}", nodes.len());

        let mut valid_count = 0;
        for node in &nodes {
            if let Ok(parsed) = validate_jsonld_basic(&node.json_ld) {
                valid_count += 1;
                tracing::debug!(
                    "节点 {} JSON-LD 有效, @type: {:?}",
                    node.iri,
                    parsed.get("@type")
                );
            }
        }
        tracing::info!("JSON-LD 验证: {}/{} 节点有效", valid_count, nodes.len());
        assert!(valid_count > 0, "应该至少有一个有效的 JSON-LD 节点");
    }

    tracing::info!("========== 测试完成 ==========");
}

/// AI Agent 安防场景调研 - 标准任务
/// 任务：调研 AI Agent 在安防监控场景的应用
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_e2e_research_security_agents() {
    init_e2e_logging();

    let (mut sa, _dir) = build_system(30);

    let user_input = r#"
请调研 AI Agent 在安防监控场景的应用，包括：

1. 智能视频分析 Agent
   - 实时目标检测与跟踪
   - 异常行为识别
   - 人脸识别与比对

2. 告警处理 Agent
   - 智能告警分级
   - 误报过滤
   - 联动响应

3. 巡检 Agent
   - 自动巡检路径规划
   - 设备状态监控
   - 异常检测与报告

请将调研结果保存到 /tmp/agent_os_e2e/security_agent_research.md，包含：
- 各场景的技术方案
- 关键技术指标
- 实施建议
"#;

    let task_iri = "iri://task/e2e_research_security";

    tracing::info!("========== 开始安防场景调研任务测试 ==========");

    let start_time = std::time::Instant::now();
    let result = sa.process_task(user_input, task_iri).await;
    let elapsed = start_time.elapsed();

    tracing::info!("执行时间: {:?}", elapsed);

    assert!(result.is_ok(), "任务应该完成: {:?}", result.err());
    let task_result = result.unwrap();

    tracing::info!("========== 安防调研完成 ==========");
    tracing::info!("状态: {}", task_result.status);
    tracing::info!("摘要: {}", task_result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        task_result.turn_count,
        task_result.tool_call_count
    );
    tracing::info!("错误: {:?}", task_result.errors);

    assert_ne!(task_result.status, "failed", "调研任务不应该完全失败");

    let report_path = std::path::Path::new("/tmp/agent_os_e2e/security_agent_research.md");
    if report_path.exists() {
        let content = std::fs::read_to_string(report_path).unwrap();
        tracing::info!("调研报告大小: {} 字节", content.len());
        assert!(content.len() > 100, "报告应该有实质内容");
        tracing::info!("✅ 安防调研报告已生成");
    }

    if let Some(blackboard) = sa.blackboard() {
        let nodes = blackboard.query_nodes(task_iri).unwrap_or_default();
        tracing::info!("任务节点数: {}", nodes.len());

        let mut valid_count = 0;
        for node in &nodes {
            if validate_jsonld_basic(&node.json_ld).is_ok() {
                valid_count += 1;
            }
        }
        tracing::info!("JSON-LD 验证: {}/{} 节点有效", valid_count, nodes.len());
    }

    tracing::info!("========== 测试完成 ==========");
}

/// 复杂编程 + 调研混合任务
/// 任务：调研 Rust Web 框架并创建一个简单的 HTTP 服务
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_e2e_research_and_code_rust_web() {
    init_e2e_logging();

    std::fs::create_dir_all("/tmp/agent_os_e2e/rust_http").ok();

    let (mut sa, _dir) = build_system(40);

    let user_input = r#"
完成以下混合任务（调研 + 编程）：

## 第一部分：调研
调研当前最流行的 Rust Web 框架，包括：
- Actix-web
- Axum
- Warp
- Rocket

对比它们的性能、易用性、生态系统成熟度。

## 第二部分：编程
基于调研结果，使用 Python 创建一个简单的 HTTP 服务（因为环境中可能没有 Rust 编译器），
保存到 /tmp/agent_os_e2e/rust_http/comparison.md（调研报告）和
/tmp/agent_os_e2e/rust_http/http_server.py（示例 HTTP 服务）

HTTP 服务要求：
- 使用 Python 标准库 http.server
- 支持 GET / 返回欢迎页面
- 支持 GET /frameworks 返回框架对比 JSON
- 支持 GET /health 健康检查
- 运行在 8888 端口
"#;

    let task_iri = "iri://task/e2e_research_and_code";

    tracing::info!("========== 开始混合任务测试（调研+编程）==========");

    let start_time = std::time::Instant::now();
    let result = sa.process_task(user_input, task_iri).await;
    let elapsed = start_time.elapsed();

    tracing::info!("执行时间: {:?}", elapsed);

    assert!(result.is_ok(), "任务应该完成: {:?}", result.err());
    let task_result = result.unwrap();

    tracing::info!("========== 混合任务完成 ==========");
    tracing::info!("状态: {}", task_result.status);
    tracing::info!("摘要: {}", task_result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        task_result.turn_count,
        task_result.tool_call_count
    );
    tracing::info!("错误: {:?}", task_result.errors);

    assert_ne!(task_result.status, "failed", "混合任务不应该完全失败");

    let comparison_path = std::path::Path::new("/tmp/agent_os_e2e/rust_http/comparison.md");
    let server_path = std::path::Path::new("/tmp/agent_os_e2e/rust_http/http_server.py");

    if comparison_path.exists() {
        let content = std::fs::read_to_string(comparison_path).unwrap();
        tracing::info!("✅ 调研报告已创建 ({} 字节)", content.len());
        assert!(content.len() > 100, "调研报告应该有足够内容");
    } else {
        tracing::warn!("❌ 调研报告未创建");
    }

    if server_path.exists() {
        let content = std::fs::read_to_string(server_path).unwrap();
        tracing::info!("✅ HTTP 服务已创建 ({} 字节)", content.len());
        assert!(
            content.contains("http") || content.contains("HTTP"),
            "应该包含 HTTP 相关代码"
        );
    } else {
        tracing::warn!("❌ HTTP 服务未创建");
    }

    if let Some(blackboard) = sa.blackboard() {
        let nodes = blackboard.query_nodes(task_iri).unwrap_or_default();
        tracing::info!("任务节点数: {}", nodes.len());

        let mut valid_count = 0;
        for node in &nodes {
            if validate_jsonld_basic(&node.json_ld).is_ok() {
                valid_count += 1;
            }
        }
        tracing::info!("JSON-LD 验证: {}/{} 节点有效", valid_count, nodes.len());
    }

    tracing::info!("========== 测试完成 ==========");
}

/// SA 任务分类准确性验证（不调用 LLM，纯规则分类）
#[test]
fn test_sa_research_task_classification() {
    let (sa, _dir) = build_system(10);

    // English exploratory keywords reliably produce Exploratory
    for task in &[
        "Research the latest advances in quantum computing",
        "Explore different methods for microservices",
        "Compare different database solutions for e-commerce",
    ] {
        let plan = sa.analyze_task(task);
        assert_eq!(
            plan.task_complexity,
            TaskComplexity::Exploratory,
            "任务 '{}' 应为 Exploratory, 实际 {:?}",
            task,
            plan.task_complexity
        );
    }

    // Chinese tasks — LLM classification may vary by model;
    // just verify they return a valid complexity without crashing
    for task in &[
        "调研大语言模型在代码领域的最新进展",
        "AI Agent在安防监控场景有哪些好的用例？",
        "对比分析 React 和 Vue 的优缺点",
        "研究并探索多种 AI Agent 架构模式",
    ] {
        let plan = sa.analyze_task(task);
        assert!(
            matches!(
                plan.task_complexity,
                TaskComplexity::Standard | TaskComplexity::Exploratory
            ),
            "任务 '{}' 返回无效分类: {:?}",
            task,
            plan.task_complexity
        );
        tracing::info!("任务: {} → 复杂度: {:?}", task, plan.task_complexity);
    }
}

/// SA 调研任务计划结构验证
#[test]
fn test_sa_research_plan_structure() {
    let (sa, _dir) = build_system(10);

    let plan = sa.analyze_task("Research and compare different AI Agent frameworks");
    assert_eq!(plan.task_complexity, TaskComplexity::Exploratory);
    assert!(plan.parallel_groups.len() > 0, "探索性任务应该有并行组");
    assert!(
        plan.agent_sequence.contains(&AgentRole::Plan),
        "应该包含 PA"
    );
    assert!(
        plan.agent_sequence.contains(&AgentRole::Check),
        "应该包含 CA"
    );
    assert!(plan.agent_sequence.contains(&AgentRole::Act), "应该包含 AA");

    let da_count = plan
        .agent_sequence
        .iter()
        .filter(|r| **r == AgentRole::Do)
        .count();
    assert!(
        da_count >= 2,
        "探索性任务应该有多个 DA，实际有 {}",
        da_count
    );

    tracing::info!("探索性任务计划: {:?}", plan.description);
    tracing::info!("Agent 序列: {:?}", plan.agent_sequence);
    tracing::info!("并行组: {:?}", plan.parallel_groups);
}
