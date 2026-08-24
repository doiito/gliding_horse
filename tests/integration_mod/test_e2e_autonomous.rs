#![cfg_attr(not(feature = "live-tests"), allow(dead_code, unused_imports))]

use std::path::Path;
use std::sync::Arc;

use glidinghorse::config::settings::LoggingSettings;
use glidinghorse::config::AgentSettings;
use glidinghorse::config::GatewaySettings;
use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::gateway::UnifiedGateway;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::templates::template_engine::TemplateEngine;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::utils::init_logging;
use glidinghorse::CoreConfig;
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
                prefix: "e2e_autonomous".to_string(),
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

fn build_autonomous_system(max_iterations: u32, model: &str) -> (SupervisorAgent, TempDir) {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let base_url = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

    let settings = GatewaySettings {
        base_url,
        api_key,
        default_model: model.to_string(),
        timeout_seconds: 300,
        max_retries: 3,
        retry_base_ms: 500,
        use_responses_api: false,
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

#[derive(Debug)]
struct AutonomousTestResult {
    status: String,
    summary: String,
    turn_count: u32,
    tool_call_count: u32,
    errors: Vec<String>,
    execution_time_secs: f64,
    #[allow(dead_code)]
    artifacts_created: Vec<String>,
}

async fn run_autonomous_task(
    sa: &mut SupervisorAgent,
    single_prompt: &str,
    task_iri: &str,
) -> AutonomousTestResult {
    let start_time = std::time::Instant::now();

    tracing::info!("========== 自主任务开始 ==========");
    tracing::info!("用户提示: {}", single_prompt);
    tracing::info!("任务 IRI: {}", task_iri);

    let result = sa.process_task(single_prompt, task_iri).await;
    let elapsed = start_time.elapsed();

    match result {
        Ok(task_result) => {
            tracing::info!("========== 任务完成 ==========");
            tracing::info!("状态: {}", task_result.status);
            tracing::info!("摘要: {}", task_result.summary);
            tracing::info!(
                "轮次: {}, 工具调用: {}",
                task_result.turn_count,
                task_result.tool_call_count
            );
            tracing::info!("执行时间: {:.2}s", elapsed.as_secs_f64());

            AutonomousTestResult {
                status: task_result.status.clone(),
                summary: task_result.summary.clone(),
                turn_count: task_result.turn_count,
                tool_call_count: task_result.tool_call_count,
                errors: task_result.errors.clone(),
                execution_time_secs: elapsed.as_secs_f64(),
                artifacts_created: vec![],
            }
        }
        Err(e) => {
            tracing::error!("任务执行失败: {:?}", e);
            AutonomousTestResult {
                status: "error".to_string(),
                summary: format!("执行错误: {:?}", e),
                turn_count: 0,
                tool_call_count: 0,
                errors: vec![format!("{:?}", e)],
                execution_time_secs: elapsed.as_secs_f64(),
                artifacts_created: vec![],
            }
        }
    }
}

fn check_artifact_exists(path: &str) -> bool {
    Path::new(path).exists()
}

fn read_artifact_content(path: &str) -> Option<String> {
    if Path::new(path).exists() {
        std::fs::read_to_string(path).ok()
    } else {
        None
    }
}

/// 真正的端到端自主测试 - 编程任务
/// 用户只输入一句话，系统自动完成：分析、规划、编码、测试、验证全流程
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_autonomous_programming_task() {
    init_e2e_logging();

    let (mut sa, _dir) = build_autonomous_system(30, "deepseek-v4-flash");

    let single_prompt = "帮我写一个 Python 贪吃蛇游戏，保存到 /tmp/agent_os_autonomous/snake.py，然后运行测试确保游戏能正常启动。";

    std::fs::create_dir_all("/tmp/agent_os_autonomous").ok();

    let result = run_autonomous_task(&mut sa, single_prompt, "iri://task/autonomous_snake").await;

    tracing::info!("========== 测试结果 ==========");
    tracing::info!("状态: {}", result.status);
    tracing::info!("摘要: {}", result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        result.turn_count,
        result.tool_call_count
    );
    tracing::info!("执行时间: {:.2}s", result.execution_time_secs);
    tracing::info!("错误: {:?}", result.errors);

    assert_ne!(result.status, "failed", "任务不应该完全失败");

    if check_artifact_exists("/tmp/agent_os_autonomous/snake.py") {
        tracing::info!("✅ 贪吃蛇游戏文件已创建");
        if let Some(content) = read_artifact_content("/tmp/agent_os_autonomous/snake.py") {
            tracing::info!("文件大小: {} 字节", content.len());
            assert!(content.len() > 100, "游戏代码应该有足够的内容");
        }
    } else {
        tracing::warn!("⚠️ 贪吃蛇游戏文件未创建");
    }

    tracing::info!("========== 测试完成 ==========");
}

/// 真正的端到端自主测试 - 数据分析任务
/// 用户只输入一句话，系统自动完成：读取数据、分析、生成报告
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_autonomous_data_analysis_task() {
    init_e2e_logging();

    let (mut sa, _dir) = build_autonomous_system(25, "deepseek-v4-flash");

    let single_prompt = "分析当前目录下所有 Rust 源文件的代码行数分布，生成一个统计报告保存到 /tmp/agent_os_autonomous/rust_analysis.md";

    std::fs::create_dir_all("/tmp/agent_os_autonomous").ok();

    let result =
        run_autonomous_task(&mut sa, single_prompt, "iri://task/autonomous_analysis").await;

    tracing::info!("========== 测试结果 ==========");
    tracing::info!("状态: {}", result.status);
    tracing::info!("摘要: {}", result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        result.turn_count,
        result.tool_call_count
    );
    tracing::info!("执行时间: {:.2}s", result.execution_time_secs);

    assert_ne!(result.status, "failed", "任务不应该完全失败");

    if check_artifact_exists("/tmp/agent_os_autonomous/rust_analysis.md") {
        tracing::info!("✅ 分析报告已创建");
        if let Some(content) = read_artifact_content("/tmp/agent_os_autonomous/rust_analysis.md") {
            tracing::info!("报告大小: {} 字节", content.len());
            assert!(content.len() > 50, "报告应该有足够的内容");
        }
    }

    tracing::info!("========== 测试完成 ==========");
}

/// 真正的端到端自主测试 - 系统运维任务
/// 用户只输入一句话，系统自动完成：检查系统状态、诊断问题、生成报告
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_autonomous_system_ops_task() {
    init_e2e_logging();

    let (mut sa, _dir) = build_autonomous_system(20, "deepseek-v4-flash");

    let single_prompt = "检查当前系统的内存使用情况、磁盘空间和运行中的进程，生成一份系统健康报告保存到 /tmp/agent_os_autonomous/system_health.txt";

    std::fs::create_dir_all("/tmp/agent_os_autonomous").ok();

    let result = run_autonomous_task(&mut sa, single_prompt, "iri://task/autonomous_sysops").await;

    tracing::info!("========== 测试结果 ==========");
    tracing::info!("状态: {}", result.status);
    tracing::info!("摘要: {}", result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        result.turn_count,
        result.tool_call_count
    );
    tracing::info!("执行时间: {:.2}s", result.execution_time_secs);

    assert_ne!(result.status, "failed", "任务不应该完全失败");

    if check_artifact_exists("/tmp/agent_os_autonomous/system_health.txt") {
        tracing::info!("✅ 系统健康报告已创建");
        if let Some(content) = read_artifact_content("/tmp/agent_os_autonomous/system_health.txt") {
            tracing::info!("报告大小: {} 字节", content.len());
        }
    }

    tracing::info!("========== 测试完成 ==========");
}

/// 真正的端到端自主测试 - 复杂多步骤任务
/// 用户只输入一句话，系统自动完成：创建项目、编写代码、配置文件、测试验证
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_autonomous_full_project_task() {
    init_e2e_logging();

    let (mut sa, _dir) = build_autonomous_system(40, "deepseek-v4-flash");

    let single_prompt = "创建一个完整的 Python 命令行计算器项目，支持加减乘除和括号运算，包含单元测试，项目放在 /tmp/agent_os_autonomous/calculator/";

    std::fs::create_dir_all("/tmp/agent_os_autonomous/calculator").ok();

    let result =
        run_autonomous_task(&mut sa, single_prompt, "iri://task/autonomous_calculator").await;

    tracing::info!("========== 测试结果 ==========");
    tracing::info!("状态: {}", result.status);
    tracing::info!("摘要: {}", result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        result.turn_count,
        result.tool_call_count
    );
    tracing::info!("执行时间: {:.2}s", result.execution_time_secs);
    tracing::info!("错误: {:?}", result.errors);

    assert_ne!(result.status, "failed", "复杂任务不应该完全失败");

    let calculator_main = "/tmp/agent_os_autonomous/calculator/calculator.py";
    let calculator_test = "/tmp/agent_os_autonomous/calculator/test_calculator.py";

    let mut files_created = 0;

    if check_artifact_exists(calculator_main) {
        files_created += 1;
        tracing::info!("✅ calculator.py 已创建");
        if let Some(content) = read_artifact_content(calculator_main) {
            assert!(
                content.contains("def ") || content.contains("class "),
                "应该包含函数或类定义"
            );
        }
    }

    if check_artifact_exists(calculator_test) {
        files_created += 1;
        tracing::info!("✅ test_calculator.py 已创建");
    }

    tracing::info!("文件创建统计: {} 个核心文件", files_created);

    tracing::info!("========== 测试完成 ==========");
}

/// 真正的端到端自主测试 - 调研任务
/// 用户只输入一句话，系统自动完成：搜索、整理、生成报告
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_autonomous_research_task() {
    init_e2e_logging();

    let (mut sa, _dir) = build_autonomous_system(25, "deepseek-v4-flash");

    let single_prompt = "调研 Rust 异步编程的主流框架，对比 tokio、async-std 和 smol 的优缺点，生成调研报告保存到 /tmp/agent_os_autonomous/async_rust_report.md";

    std::fs::create_dir_all("/tmp/agent_os_autonomous").ok();

    let result =
        run_autonomous_task(&mut sa, single_prompt, "iri://task/autonomous_research").await;

    tracing::info!("========== 测试结果 ==========");
    tracing::info!("状态: {}", result.status);
    tracing::info!("摘要: {}", result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        result.turn_count,
        result.tool_call_count
    );
    tracing::info!("执行时间: {:.2}s", result.execution_time_secs);

    assert_ne!(result.status, "failed", "调研任务不应该完全失败");

    if check_artifact_exists("/tmp/agent_os_autonomous/async_rust_report.md") {
        tracing::info!("✅ 调研报告已创建");
        if let Some(content) =
            read_artifact_content("/tmp/agent_os_autonomous/async_rust_report.md")
        {
            tracing::info!("报告大小: {} 字节", content.len());
            assert!(content.len() > 100, "报告应该有足够的内容");
        }
    }

    tracing::info!("========== 测试完成 ==========");
}

/// 真正的端到端自主测试 - 极简提示词
/// 测试系统对最简单提示词的理解和执行能力
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_autonomous_minimal_prompt() {
    init_e2e_logging();

    let (mut sa, _dir) = build_autonomous_system(15, "deepseek-v4-flash");

    let single_prompt = "列出 src/core 目录下所有文件名并统计总行数";

    let result = run_autonomous_task(&mut sa, single_prompt, "iri://task/autonomous_minimal").await;

    tracing::info!("========== 测试结果 ==========");
    tracing::info!("状态: {}", result.status);
    tracing::info!("摘要: {}", result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        result.turn_count,
        result.tool_call_count
    );

    assert_ne!(result.status, "failed", "简单任务不应该失败");

    tracing::info!("========== 测试完成 ==========");
}
