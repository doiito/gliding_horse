#![cfg_attr(not(feature = "live-tests"), allow(dead_code, unused_imports))]

use std::path::Path;
use std::sync::Arc;

use glidinghorse::config::settings::LoggingSettings;
use glidinghorse::config::AgentSettings;
use glidinghorse::config::GatewaySettings;
use glidinghorse::core::agent_runner::AgentRunner;
use glidinghorse::core::event_bus::EventBus;
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::core::validation::ValidationEngine;
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
                prefix: "e2e_programming".to_string(),
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

#[allow(dead_code)]
fn validate_jsonld_with_context(json_str: &str) -> Result<Value, String> {
    let parsed = validate_jsonld_basic(json_str)?;

    if parsed.get("@context").is_none() {
        return Err("缺少 @context 字段".to_string());
    }

    Ok(parsed)
}

fn validate_agent_output_jsonld(
    blackboard: &Blackboard,
    task_iri: &str,
    expected_role: &str,
) -> Result<Vec<Value>, String> {
    let nodes = blackboard
        .query_nodes(task_iri)
        .map_err(|e| format!("查询节点失败: {}", e))?;

    if nodes.is_empty() {
        return Err(format!("任务 {} 没有找到任何节点", task_iri));
    }

    let mut valid_nodes = Vec::new();
    for node in nodes {
        match validate_jsonld_basic(&node.json_ld) {
            Ok(parsed) => {
                if let Some(node_type) = node.node_type.as_ref() {
                    if node_type.contains(expected_role) || expected_role.is_empty() {
                        valid_nodes.push(parsed);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("节点 {} JSON-LD验证失败: {}", node.iri, e);
            }
        }
    }

    Ok(valid_nodes)
}

fn verify_entity_alignment(blackboard: &Blackboard, iri: &str) -> Result<bool, String> {
    let nodes = blackboard
        .query_nodes(iri)
        .map_err(|e| format!("查询节点失败: {}", e))?;

    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for node in &nodes {
        if let Ok(parsed) = serde_json::from_str::<Value>(&node.json_ld) {
            if let Some(id) = parsed.get("@id").and_then(|i| i.as_str()) {
                if seen_ids.contains(id) {
                    return Ok(true);
                }
                seen_ids.insert(id.to_string());
            }
        }
    }

    Ok(false)
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

/// 最复杂的端到端编程任务测试
/// 任务：创建一个完整的计算器程序，包含：
/// 1. 主程序 calculator.py（支持加减乘除和括号）
/// 2. 测试文件 test_calculator.py（使用 pytest）
/// 3. 运行测试验证功能
/// 4. 验证所有Agent输出为有效的JSON-LD格式
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_e2e_complex_programming_task() {
    init_e2e_logging();

    std::fs::create_dir_all("/tmp/agent_os_e2e/calculator").ok();

    let (mut sa, _dir) = build_system(50);

    let user_input = r#"
创建一个完整的 Python 计算器项目，要求如下：

1. 创建主程序 /tmp/agent_os_e2e/calculator/calculator.py：
   - 实现一个 Calculator 类
   - 支持 add, subtract, multiply, divide 四个方法
   - 支持 evaluate 方法计算带括号的表达式，如 "2 + 3 * 4" 或 "(2 + 3) * 4"
   - 必须处理除以零的错误

2. 创建测试文件 /tmp/agent_os_e2e/calculator/test_calculator.py：
   - 使用 pytest 框架
   - 测试所有四个基本运算
   - 测试表达式求值
   - 测试除以零错误处理
   - 至少包含 10 个测试用例

3. 运行 pytest 验证所有测试通过

注意：必须确保代码质量，添加适当的文档字符串和类型注解。
"#;

    let task_iri = "iri://task/e2e_complex_programming";

    tracing::info!("========== 开始复杂编程任务测试 ==========");
    tracing::info!("任务: {}", user_input.replace('\n', " "));

    let result = sa.process_task(user_input, task_iri).await;

    assert!(
        result.is_ok(),
        "任务应该完成，不应该报错: {:?}",
        result.err()
    );
    let task_result = result.unwrap();

    tracing::info!("========== 任务完成 ==========");
    tracing::info!("状态: {}", task_result.status);
    tracing::info!("摘要: {}", task_result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        task_result.turn_count,
        task_result.tool_call_count
    );
    tracing::info!("错误: {:?}", task_result.errors);

    // 验证文件是否创建
    let calculator_path = Path::new("/tmp/agent_os_e2e/calculator/calculator.py");
    let test_path = Path::new("/tmp/agent_os_e2e/calculator/test_calculator.py");

    if calculator_path.exists() {
        let content = std::fs::read_to_string(calculator_path).unwrap();
        tracing::info!("========== calculator.py 内容 ==========\n{}", content);
        assert!(
            content.contains("Calculator") || content.contains("calculator"),
            "应该包含 Calculator 类"
        );
    } else {
        tracing::warn!("calculator.py 未创建");
    }

    if test_path.exists() {
        let content = std::fs::read_to_string(test_path).unwrap();
        tracing::info!("========== test_calculator.py 内容 ==========\n{}", content);
        assert!(
            content.contains("pytest") || content.contains("test_"),
            "应该包含测试代码"
        );
    } else {
        tracing::warn!("test_calculator.py 未创建");
    }

    assert_ne!(task_result.status, "failed", "任务不应该完全失败");

    // ===== JSON-LD 验证 =====
    tracing::info!("========== 开始 JSON-LD 验证 ==========");

    if let Some(blackboard) = sa.blackboard() {
        // 验证所有Agent输出节点是否为有效JSON-LD
        match validate_agent_output_jsonld(blackboard, task_iri, "") {
            Ok(nodes) => {
                tracing::info!("JSON-LD 验证通过: {} 个有效节点", nodes.len());
                assert!(!nodes.is_empty(), "应该至少有一个有效的JSON-LD节点");

                // 验证每个节点都有 @id, @type
                for node in &nodes {
                    assert!(node.get("@id").is_some(), "节点必须有 @id");
                    assert!(node.get("@type").is_some(), "节点必须有 @type");
                }
            }
            Err(e) => {
                tracing::warn!("JSON-LD 验证失败: {}", e);
            }
        }

        // 验证PA输出包含正确的@type
        let pa_nodes = blackboard
            .query_by_types(&["PlanNode".to_string(), "Plan".to_string()])
            .unwrap_or_default();
        if !pa_nodes.is_empty() {
            tracing::info!("找到 {} 个 PA 节点", pa_nodes.len());
            for node in &pa_nodes {
                if node.parent_task.as_deref() == Some(task_iri) {
                    assert!(
                        node.jsonld_types.contains(&"PlanNode".to_string())
                            || node.jsonld_types.contains(&"Plan".to_string()),
                        "PA节点应该有正确的@type"
                    );
                }
            }
        }

        // 验证DA输出包含正确的IRI引用
        let da_nodes = blackboard
            .query_by_types(&["ExecutionResult".to_string(), "Do".to_string()])
            .unwrap_or_default();
        if !da_nodes.is_empty() {
            tracing::info!("找到 {} 个 DA 节点", da_nodes.len());
            for node in &da_nodes {
                if node.parent_task.as_deref() == Some(task_iri) {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&node.json_ld) {
                        assert!(parsed.get("@id").is_some(), "DA节点必须有 @id");
                    }
                }
            }
        }

        // 验证CA能查询到DA创建的节点
        let all_nodes = blackboard.query_nodes(task_iri).unwrap_or_default();
        let da_node_count = all_nodes
            .iter()
            .filter(|n| {
                n.jsonld_types
                    .iter()
                    .any(|t| t.contains("Execution") || t.contains("Do"))
            })
            .count();
        tracing::info!("DA创建的节点数: {}", da_node_count);

        // 验证AA能访问所有Agent的输出
        let total_nodes = blackboard.node_count();
        tracing::info!("总节点数: {}", total_nodes);

        // 验证实体对齐（相同@id的节点合并）
        match verify_entity_alignment(blackboard, task_iri) {
            Ok(has_alignment) => {
                tracing::info!(
                    "实体对齐检查: {}",
                    if has_alignment {
                        "发现重复@id"
                    } else {
                        "无重复@id"
                    }
                );
            }
            Err(e) => {
                tracing::warn!("实体对齐检查失败: {}", e);
            }
        }

        // 使用 ValidationEngine 验证
        let validator = ValidationEngine::new(2048);
        for node in &all_nodes {
            match validator.validate_json_ld(&node.json_ld) {
                Ok(()) => {
                    tracing::debug!("节点 {} 通过 ValidationEngine 验证", node.iri);
                }
                Err(e) => {
                    tracing::warn!("节点 {} ValidationEngine 验证失败: {}", node.iri, e);
                }
            }
        }
    }
}

/// 测试 SA 使用 LLM 生成详细计划
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_e2e_sa_llm_plan_generation() {
    init_e2e_logging();

    let (mut sa, _dir) = build_system(20);

    let user_input =
        "分析当前目录结构，找出所有 Rust 源文件，统计每个文件的行数，并生成一个报告文件。";

    let task_iri = "iri://task/e2e_llm_plan";

    tracing::info!("========== 开始 LLM 计划生成测试 ==========");

    let result = sa.process_task(user_input, task_iri).await;

    assert!(result.is_ok(), "任务应该完成: {:?}", result.err());
    let task_result = result.unwrap();

    tracing::info!("========== LLM 计划生成测试完成 ==========");
    tracing::info!("状态: {}", task_result.status);
    tracing::info!("摘要: {}", task_result.summary);

    assert_ne!(task_result.status, "failed", "任务不应该失败");

    // JSON-LD 验证
    if let Some(blackboard) = sa.blackboard() {
        let nodes = blackboard.query_nodes(task_iri).unwrap_or_default();
        tracing::info!("任务节点数: {}", nodes.len());

        for node in &nodes {
            if let Ok(parsed) = validate_jsonld_basic(&node.json_ld) {
                tracing::debug!(
                    "节点 {} JSON-LD 有效, @type: {:?}",
                    node.iri,
                    parsed.get("@type")
                );
            }
        }
    }
}

/// 测试 Agent 隔离执行
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_e2e_agent_isolation() {
    init_e2e_logging();

    let (mut sa, _dir) = build_system(20);

    // 这个任务需要多个步骤，测试 Agent 隔离
    let user_input = r#"
完成以下任务：
1. 创建一个文件 /tmp/agent_os_e2e/data.txt，写入 "Hello, World!"
2. 读取该文件内容并打印
3. 将文件内容转换为大写后写回文件
4. 再次读取并验证内容是 "HELLO, WORLD!"
"#;

    let task_iri = "iri://task/e2e_isolation";

    tracing::info!("========== 开始 Agent 隔离测试 ==========");

    let result = sa.process_task(user_input, task_iri).await;

    assert!(result.is_ok(), "任务应该完成: {:?}", result.err());
    let task_result = result.unwrap();

    tracing::info!("========== Agent 隔离测试完成 ==========");
    tracing::info!("状态: {}", task_result.status);
    tracing::info!("摘要: {}", task_result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        task_result.turn_count,
        task_result.tool_call_count
    );

    // 验证最终文件内容
    let data_path = Path::new("/tmp/agent_os_e2e/data.txt");
    if data_path.exists() {
        let content = std::fs::read_to_string(data_path).unwrap();
        tracing::info!("最终文件内容: {}", content);
        assert!(content.contains("HELLO"), "内容应该是大写");
    }

    // JSON-LD 验证
    if let Some(blackboard) = sa.blackboard() {
        let nodes: Vec<_> = blackboard.query_nodes(task_iri).unwrap_or_default();
        tracing::info!("任务节点数: {}", nodes.len());

        // 验证每个Agent的输出都有正确的JSON-LD格式
        for node in &nodes {
            match validate_jsonld_basic(&node.json_ld) {
                Ok(parsed) => {
                    tracing::debug!(
                        "节点 {} 验证通过, @id: {:?}, @type: {:?}",
                        node.iri,
                        parsed.get("@id"),
                        parsed.get("@type")
                    );
                }
                Err(e) => {
                    tracing::warn!("节点 {} JSON-LD验证失败: {}", node.iri, e);
                }
            }
        }

        // 验证跨Agent实体对齐
        match verify_entity_alignment(blackboard, task_iri) {
            Ok(_) => {
                tracing::info!("实体对齐检查完成");
            }
            Err(e) => {
                tracing::warn!("实体对齐检查失败: {}", e);
            }
        }
    }
}

/// 最复杂的端到端测试：创建完整的 Python Web API 项目
/// 任务：创建一个完整的 FastAPI 项目，包含：
/// 1. 项目目录结构 (myapi/)
/// 2. 主应用文件 (app/main.py)
/// 3. 模型定义 (app/models.py)
/// 4. 路由处理 (app/routes.py)
/// 5. 配置文件 (pyproject.toml, requirements.txt)
/// 6. README.md 文件
/// 7. 测试文件 (tests/test_api.py)
/// 8. 运行测试验证
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "live-tests")]
async fn test_e2e_full_project_creation() {
    init_e2e_logging();

    let project_dir = "/tmp/agent_os_e2e/myapi";
    std::fs::create_dir_all(project_dir).ok();
    std::fs::create_dir_all(format!("{}/app", project_dir)).ok();
    std::fs::create_dir_all(format!("{}/tests", project_dir)).ok();

    let (mut sa, _dir) = build_system(40);

    let user_input = r#"
创建一个完整的 Python FastAPI Web API 项目，项目目录: /tmp/agent_os_e2e/myapi

## 项目结构要求
```
myapi/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 主应用
│   ├── models.py        # Pydantic 数据模型
│   └── routes.py        # API 路由定义
├── tests/
│   ├── __init__.py
│   └── test_api.py      # pytest 测试文件
├── pyproject.toml       # 项目配置
├── requirements.txt     # 依赖列表
└── README.md            # 项目文档
```

## 功能要求

### 1. app/models.py
定义以下 Pydantic 模型：
- Item: 包含 id (int), name (str), price (float), is_offer (bool, 可选)
- User: 包含 id (int), username (str), email (str)

### 2. app/routes.py
实现以下 API 端点：
- GET /items/ - 获取所有 items 列表
- GET /items/{item_id} - 获取单个 item
- POST /items/ - 创建新 item
- PUT /items/{item_id} - 更新 item
- DELETE /items/{item_id} - 删除 item
- GET /users/ - 获取所有 users 列表
- POST /users/ - 创建新 user

### 3. app/main.py
- 创建 FastAPI 应用实例
- 包含 CORS 中间件配置
- 挂载路由
- 包含健康检查端点 GET /health

### 4. tests/test_api.py
使用 pytest 和 httpx 编写测试：
- 测试健康检查端点
- 测试 CRUD 操作
- 至少 8 个测试用例

### 5. pyproject.toml
包含项目元数据和依赖配置

### 6. requirements.txt
列出所有依赖：fastapi, uvicorn, pydantic, pytest, httpx

### 7. README.md
包含：
- 项目名称和描述
- 安装说明
- 运行说明
- API 文档概览

## 质量要求
- 所有代码添加类型注解
- 添加 docstring 文档
- 代码格式规范

完成后运行 pytest 验证测试通过。
"#;

    let task_iri = "iri://task/e2e_full_project";

    tracing::info!("========== 开始完整项目创建测试 ==========");
    tracing::info!("任务: 创建 FastAPI Web API 项目");

    let start_time = std::time::Instant::now();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(600),
        sa.process_task(user_input, task_iri),
    )
    .await;
    let elapsed = start_time.elapsed();

    tracing::info!("执行时间: {:?}", elapsed);

    let task_result = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => panic!("任务执行失败: {:?}", e),
        Err(_) => panic!("任务执行超时 (600秒)"),
    };

    tracing::info!("========== 项目创建完成 ==========");
    tracing::info!("状态: {}", task_result.status);
    tracing::info!("摘要: {}", task_result.summary);
    tracing::info!(
        "轮次: {}, 工具调用: {}",
        task_result.turn_count,
        task_result.tool_call_count
    );
    tracing::info!("错误: {:?}", task_result.errors);

    // 验证项目结构
    let expected_files = vec![
        (
            format!("{}/app/__init__.py", project_dir),
            "app/__init__.py",
        ),
        (format!("{}/app/main.py", project_dir), "app/main.py"),
        (format!("{}/app/models.py", project_dir), "app/models.py"),
        (format!("{}/app/routes.py", project_dir), "app/routes.py"),
        (
            format!("{}/tests/__init__.py", project_dir),
            "tests/__init__.py",
        ),
        (
            format!("{}/tests/test_api.py", project_dir),
            "tests/test_api.py",
        ),
        (format!("{}/pyproject.toml", project_dir), "pyproject.toml"),
        (
            format!("{}/requirements.txt", project_dir),
            "requirements.txt",
        ),
        (format!("{}/README.md", project_dir), "README.md"),
    ];

    let mut created_count = 0;
    let mut missing_files = Vec::new();

    for (path, name) in &expected_files {
        if Path::new(path).exists() {
            created_count += 1;
            let content = std::fs::read_to_string(path).unwrap_or_default();
            tracing::info!("✅ {} 已创建 ({} 字节)", name, content.len());
        } else {
            missing_files.push(*name);
            tracing::warn!("❌ {} 未创建", name);
        }
    }

    tracing::info!("文件创建统计: {}/{}", created_count, expected_files.len());

    // 验证关键文件内容
    let main_path = format!("{}/app/main.py", project_dir);
    if Path::new(&main_path).exists() {
        let content = std::fs::read_to_string(&main_path).unwrap();
        assert!(content.contains("FastAPI"), "main.py 应该包含 FastAPI");
        assert!(
            content.contains("/health") || content.contains("health"),
            "main.py 应该包含健康检查端点"
        );
        tracing::info!("✅ main.py 内容验证通过");
    }

    let models_path = format!("{}/app/models.py", project_dir);
    if Path::new(&models_path).exists() {
        let content = std::fs::read_to_string(&models_path).unwrap();
        assert!(
            content.contains("Item") || content.contains("item"),
            "models.py 应该包含 Item 模型"
        );
        tracing::info!("✅ models.py 内容验证通过");
    }

    let routes_path = format!("{}/app/routes.py", project_dir);
    if Path::new(&routes_path).exists() {
        let content = std::fs::read_to_string(&routes_path).unwrap();
        assert!(
            content.contains("GET") || content.contains("POST") || content.contains("router"),
            "routes.py 应该包含路由定义"
        );
        tracing::info!("✅ routes.py 内容验证通过");
    }

    let test_path = format!("{}/tests/test_api.py", project_dir);
    if Path::new(&test_path).exists() {
        let content = std::fs::read_to_string(&test_path).unwrap();
        assert!(
            content.contains("pytest") || content.contains("test_") || content.contains("def test"),
            "test_api.py 应该包含测试代码"
        );
        tracing::info!("✅ test_api.py 内容验证通过");
    }

    let readme_path = format!("{}/README.md", project_dir);
    if Path::new(&readme_path).exists() {
        let content = std::fs::read_to_string(&readme_path).unwrap();
        assert!(content.len() > 50, "README.md 应该有足够的内容");
        tracing::info!("✅ README.md 内容验证通过 ({} 字节)", content.len());
    }

    // 至少创建 6 个文件才算成功
    assert!(
        created_count >= 6,
        "至少应该创建 6 个文件，实际创建了 {} 个",
        created_count
    );

    // JSON-LD 验证
    if let Some(blackboard) = sa.blackboard() {
        let nodes = blackboard.query_nodes(task_iri).unwrap_or_default();
        tracing::info!("任务节点数: {}", nodes.len());

        // 验证 JSON-LD 格式
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
    }

    tracing::info!("========== 测试完成 ==========");
}
