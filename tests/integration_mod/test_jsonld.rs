use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use glidinghorse::core::validation::{JsonLdValidator, MetaValidator, ValidationEngine};
use glidinghorse::memory::l2_blackboard::{Blackboard, GraphPermission};
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::CoreConfig;
use serde_json::json;

fn create_test_node(iri: &str, node_type: &str, properties: HashMap<&str, serde_json::Value>) -> String {
    let mut node = json!({
        "@id": iri,
        "@type": node_type,
        "@context": "https://agent-os.org/context/test"
    });
    
    if let Some(obj) = node.as_object_mut() {
        for (key, value) in properties {
            obj.insert(key.to_string(), value);
        }
    }
    
    serde_json::to_string(&node).unwrap()
}

#[test]
fn test_jsonld_node_creation_and_validation() {
    let validator = JsonLdValidator::default();
    
    let valid_node = create_test_node(
        "iri://task/test/1",
        "PlanNode",
        HashMap::from([
            ("summary", json!("测试计划")),
            ("confidence", json!(0.85)),
        ])
    );
    
    let result = validator.validate(&valid_node);
    assert!(result.valid, "有效节点应该通过验证: {:?}", result.errors);
    assert!(result.warnings.is_empty() || result.warnings.iter().any(|w| w.contains("@context")));
    
    let invalid_node = json!({
        "summary": "缺少 @id 和 @type"
    }).to_string();
    
    let result = validator.validate(&invalid_node);
    assert!(!result.warnings.is_empty(), "缺少关键字段应该有警告");
}

#[test]
fn test_jsonld_multi_type_node() {
    let validator = JsonLdValidator::default();
    
    let multi_type_node = json!({
        "@id": "iri://task/test/multi",
        "@type": ["PlanNode", "Urgent", "Priority"],
        "@context": "https://agent-os.org/context/test",
        "summary": "多类型节点"
    }).to_string();
    
    let result = validator.validate(&multi_type_node);
    assert!(result.valid, "多类型节点应该有效");
}

#[test]
fn test_jsonld_context_validation() {
    let validator = JsonLdValidator::default();
    
    let with_string_context = json!({
        "@id": "iri://test/1",
        "@type": "Test",
        "@context": "https://schema.org"
    }).to_string();
    let result = validator.validate(&with_string_context);
    assert!(result.valid);
    
    let with_array_context = json!({
        "@id": "iri://test/2",
        "@type": "Test",
        "@context": ["https://schema.org", {"ex": "https://example.org/"}]
    }).to_string();
    let result = validator.validate(&with_array_context);
    assert!(result.valid);
    
    let with_object_context = json!({
        "@id": "iri://test/3",
        "@type": "Test",
        "@context": {"name": "http://schema.org/name"}
    }).to_string();
    let result = validator.validate(&with_object_context);
    assert!(result.valid);
}

#[test]
fn test_entity_alignment_and_graph_merge() {
    let blackboard = Blackboard::new().unwrap();
    let config = CoreConfig::default();
    
    let shared_id = "iri://entity/shared";
    
    let node1 = json!({
        "@id": shared_id,
        "@type": "Task",
        "status": "running",
        "created_by": "PA"
    }).to_string();
    
    let node2 = json!({
        "@id": shared_id,
        "@type": "Task",
        "status": "completed",
        "updated_by": "DA",
        "result": "成功"
    }).to_string();
    
    blackboard.write_node(shared_id, &node1, &config).unwrap();
    blackboard.write_node(shared_id, &node2, &config).unwrap();

    // Wait for background Oxigraph sync before SPARQL querying
    blackboard.flush_oxigraph();

    let nodes = blackboard.query_nodes(shared_id).unwrap();
    assert!(!nodes.is_empty(), "应该能查询到写入的节点");
    
    let sparql = format!(
        "SELECT ?p ?o WHERE {{ <{}> ?p ?o . }}",
        shared_id
    );
    let results = blackboard.query(&sparql).unwrap();
    assert!(!results.is_empty(), "应该能查询到共享实体的三元组");
}

#[test]
fn test_multi_type_query() {
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let config = CoreConfig::default();
    
    let plan_node = json!({
        "@id": "iri://test/plan",
        "@type": "PlanNode",
        "summary": "计划节点"
    }).to_string();
    
    let exec_node = json!({
        "@id": "iri://test/exec",
        "@type": "ExecutionResult",
        "summary": "执行节点"
    }).to_string();
    
    let multi_node = json!({
        "@id": "iri://test/multi",
        "@type": ["PlanNode", "Urgent"],
        "summary": "多类型节点"
    }).to_string();
    
    blackboard.write_node("iri://test/plan", &plan_node, &config).unwrap();
    blackboard.write_node("iri://test/exec", &exec_node, &config).unwrap();
    blackboard.write_node("iri://test/multi", &multi_node, &config).unwrap();

    blackboard.flush_oxigraph();

    let plan_nodes = blackboard.query_by_types(&["PlanNode".to_string()]).unwrap();
    assert_eq!(plan_nodes.len(), 2, "应该找到2个PlanNode类型的节点");
    
    let exec_nodes = blackboard.query_by_types(&["ExecutionResult".to_string()]).unwrap();
    assert_eq!(exec_nodes.len(), 1, "应该找到1个ExecutionResult类型的节点");
    
    let urgent_nodes = blackboard.query_by_types(&["Urgent".to_string()]).unwrap();
    assert_eq!(urgent_nodes.len(), 1, "应该找到1个Urgent类型的节点");
}

#[test]
fn test_named_graph_isolation() {
    let blackboard = Blackboard::new().unwrap();
    let config = CoreConfig::default();
    
    let plan_node = json!({
        "@id": "iri://test/plan",
        "@type": "Plan",
        "status": "draft"
    }).to_string();
    
    let exec_node = json!({
        "@id": "iri://test/exec",
        "@type": "Execution",
        "status": "running"
    }).to_string();
    
    blackboard.write_node_to_graph("iri://test/plan", &plan_node, "system:plan", &config).unwrap();
    blackboard.write_node_to_graph("iri://test/exec", &exec_node, "system:execution", &config).unwrap();

    blackboard.flush_oxigraph();

    let plan_results = blackboard.query_graph("system:plan", "?s ?p ?o").unwrap();
    assert!(!plan_results.is_empty(), "plan图应该有节点");
    
    let exec_results = blackboard.query_graph("system:execution", "?s ?p ?o").unwrap();
    assert!(!exec_results.is_empty(), "execution图应该有节点");
    
    let all_results = blackboard.query(
        "SELECT ?s WHERE { { ?s a ?type } UNION { GRAPH ?g { ?s a ?type } } }"
    ).unwrap();
    assert!(all_results.len() >= 2, "全局查询应该返回所有节点, 实际: {}", all_results.len());
}

#[test]
fn test_token_budget_control() {
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let projection = ProjectionEngine::new(blackboard.clone(), 200);
    let config = CoreConfig::default();
    
    for i in 0..10 {
        let node = json!({
            "@id": format!("iri://test/node/{}", i),
            "@type": "TestNode",
            "summary": "x".repeat(100),
            "data": "y".repeat(100)
        }).to_string();
        blackboard.write_node(&format!("iri://test/node/{}", i), &node, &config).unwrap();
    }
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        projection.project("iri://test", "reference_only", HashMap::new()).await
    }).unwrap();
    
    assert!(result.len() <= 200, "投影结果应该在max_size预算内, 实际长度: {}", result.len());
}

#[test]
fn test_skill_semantic_discovery() {
    let registry = SkillRegistry::new();
    
    let basic_skills = registry.list_skills_basic();
    assert!(!basic_skills.is_empty(), "应该有内置技能");
    
    for skill in &basic_skills {
        assert!(!skill.name.is_empty(), "技能应该有名称");
        assert!(!skill.description.is_empty(), "技能应该有描述");
    }
    
    let da_skills = registry.list_skills_for_role("DA");
    assert!(!da_skills.is_empty(), "DA角色应该有可用技能");
    
    let pa_skills = registry.list_skills_for_role("PA");
    assert!(!pa_skills.is_empty(), "PA角色应该有可用技能");
}

#[test]
fn test_meta_validator_plan_conversion() {
    let validator = MetaValidator::new();
    
    let plan_meta = json!({
        "summary": "创建用户认证系统",
        "goal": "实现安全的用户登录",
        "approach": "使用JWT和bcrypt",
        "sub_tasks": ["设计数据库", "实现API", "添加测试"],
        "priority": "high",
        "confidence": 0.9
    });
    
    let result = validator.validate_and_convert("plan", &plan_meta);
    assert!(result.is_ok(), "计划元数据验证应该成功");
    
    let json_ld = result.unwrap();
    assert_eq!(json_ld.get("@type").and_then(|t| t.as_str()), Some("PlanNode"));
    assert!(json_ld.get("@id").is_some());
    assert!(json_ld.get("@context").is_some());
}

#[test]
fn test_meta_validator_execution_conversion() {
    let validator = MetaValidator::new();
    
    let exec_meta = json!({
        "summary": "用户认证系统实现完成",
        "result_type": "code",
        "output_location": "/src/auth/",
        "steps_completed": ["数据库设计", "API实现", "测试编写"],
        "confidence": 0.85
    });
    
    let result = validator.validate_and_convert("execution", &exec_meta);
    assert!(result.is_ok());
    
    let json_ld = result.unwrap();
    assert_eq!(json_ld.get("@type").and_then(|t| t.as_str()), Some("ExecutionResult"));
}

#[test]
fn test_meta_validator_check_conversion() {
    let validator = MetaValidator::new();
    
    let check_meta = json!({
        "summary": "代码质量检查通过",
        "verdict": "pass",
        "quality_score": 92,
        "strengths": ["代码结构清晰", "测试覆盖完整"],
        "recommendations": ["添加更多边界测试"],
        "confidence": 0.88
    });
    
    let result = validator.validate_and_convert("check", &check_meta);
    assert!(result.is_ok());
    
    let json_ld = result.unwrap();
    assert_eq!(json_ld.get("@type").and_then(|t| t.as_str()), Some("CheckResult"));
}

#[test]
fn test_meta_validator_decision_conversion() {
    let validator = MetaValidator::new();
    
    let decision_meta = json!({
        "summary": "批准部署到生产环境",
        "action": "continue",
        "reasoning": "所有测试通过，代码质量达标",
        "next_steps": ["部署", "监控", "收集反馈"],
        "confidence": 0.95
    });
    
    let result = validator.validate_and_convert("decision", &decision_meta);
    assert!(result.is_ok());
    
    let json_ld = result.unwrap();
    assert_eq!(json_ld.get("@type").and_then(|t| t.as_str()), Some("DecisionNode"));
}

#[test]
fn test_validation_engine_integration() {
    let engine = ValidationEngine::new(2048);
    
    let valid_jsonld = json!({
        "@id": "iri://test/valid",
        "@type": "TestNode",
        "@context": "https://agent-os.org/context/test"
    }).to_string();
    
    let result = engine.validate_json_ld(&valid_jsonld);
    assert!(result.is_ok());
    
    let invalid_jsonld = "not valid json";
    let result = engine.validate_json_ld(invalid_jsonld);
    assert!(result.is_err());
}

#[test]
fn test_permission_matrix() {
    let blackboard = Blackboard::new().unwrap();
    
    assert!(blackboard.check_permission("Plan", "system:plan", GraphPermission::Read));
    assert!(blackboard.check_permission("Plan", "system:plan", GraphPermission::Write));
    assert!(blackboard.check_permission("Plan", "system:knowledge", GraphPermission::Read));
    assert!(!blackboard.check_permission("Plan", "system:knowledge", GraphPermission::Write));
    
    assert!(blackboard.check_permission("Do", "system:execution", GraphPermission::Write));
    assert!(!blackboard.check_permission("Do", "system:plan", GraphPermission::Write));
    
    assert!(blackboard.check_permission("Check", "system:review", GraphPermission::Write));
    assert!(blackboard.check_permission("Act", "system:decision", GraphPermission::Write));
}

#[test]
fn test_sparql_query_on_jsonld_nodes() {
    let blackboard = Blackboard::new().unwrap();
    let config = CoreConfig::default();
    
    let node1 = json!({
        "@id": "iri://task/1",
        "@type": "Task",
        "status": "running",
        "priority": "high"
    }).to_string();
    
    let node2 = json!({
        "@id": "iri://task/2",
        "@type": "Task",
        "status": "completed",
        "priority": "low"
    }).to_string();
    
    blackboard.write_node("iri://task/1", &node1, &config).unwrap();
    blackboard.write_node("iri://task/2", &node2, &config).unwrap();

    blackboard.flush_oxigraph();

    let sparql = r#"
        SELECT ?s ?status WHERE {
            ?s a <http://agent-os.org/type/Task> .
            ?s <http://agent-os.org/prop/status> ?status .
        }
    "#;
    
    let results = blackboard.query(sparql).unwrap();
    assert!(results.len() >= 2, "应该查询到至少2个Task节点");
}

#[test]
fn test_projection_frame_templates() {
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let projection = ProjectionEngine::new(blackboard, 1024);
    
    let frames = projection.list_frames();
    assert!(!frames.is_empty(), "应该有预定义的Frame模板");
    
    let frame_names: Vec<&str> = frames.iter().map(|f| f.name.as_str()).collect();
    assert!(frame_names.contains(&"summary_only"));
    assert!(frame_names.contains(&"pa_init"));
    assert!(frame_names.contains(&"da_input"));
    assert!(frame_names.contains(&"ca_review"));
    assert!(frame_names.contains(&"aa_decision"));
}

#[test]
fn test_jsonld_size_limit() {
    let blackboard = Blackboard::new().unwrap();
    let config = CoreConfig {
        max_node_size: 100,
        ..Default::default()
    };
    
    let large_node = json!({
        "@id": "iri://test/large",
        "@type": "Test",
        "data": "x".repeat(200)
    }).to_string();
    
    let result = blackboard.write_node("iri://test/large", &large_node, &config);
    assert!(result.is_err(), "超大节点应该被拒绝");
}

#[test]
fn test_batch_write_to_graphs() {
    let blackboard = Blackboard::new().unwrap();
    let config = CoreConfig::default();
    
    let nodes = vec![
        ("iri://test/1".to_string(), json!({"@id": "iri://test/1", "@type": "Plan"}).to_string(), "system:plan".to_string()),
        ("iri://test/2".to_string(), json!({"@id": "iri://test/2", "@type": "Execution"}).to_string(), "system:execution".to_string()),
        ("iri://test/3".to_string(), json!({"@id": "iri://test/3", "@type": "Review"}).to_string(), "system:review".to_string()),
    ];
    
    let count = blackboard.write_batch_to_graphs(nodes, &config).unwrap();
    assert_eq!(count, 3, "应该成功写入3个节点");
    
    assert_eq!(blackboard.node_count(), 3);
}

#[test]
fn test_node_tags_and_metadata() {
    let blackboard = Blackboard::new().unwrap();
    let config = CoreConfig::default();
    
    let node = json!({
        "@id": "iri://test/tagged",
        "@type": "Task",
        "tags": ["important", "urgent", "backend"],
        "created_by": "test_user"
    }).to_string();
    
    blackboard.write_node("iri://test/tagged", &node, &config).unwrap();
    
    let stored = blackboard.read_node("iri://test/tagged").unwrap().unwrap();
    assert_eq!(stored.tags, vec!["important", "urgent", "backend"]);
    assert_eq!(stored.created_by, Some("test_user".to_string()));
}

#[test]
fn test_cache_invalidation() {
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let projection = ProjectionEngine::new(blackboard.clone(), 1024);
    let config = CoreConfig::default();
    
    let node = json!({
        "@id": "iri://test/cache",
        "@type": "Test",
        "value": "initial"
    }).to_string();
    blackboard.write_node("iri://test/cache", &node, &config).unwrap();
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _result1 = rt.block_on(async {
        projection.project("iri://test", "reference_only", HashMap::new()).await
    }).unwrap();
    
    projection.invalidate_view("reference_only", "iri://test");
    
    let stats = projection.cache_stats();
    assert!(stats.invalid_views > 0, "应该有无效的缓存视图");
}

#[test]
fn test_performance_jsonld_serialization() {
    let iterations = 100;
    let mut total_time = 0u64;
    
    for i in 0..iterations {
        let start = Instant::now();
        
        let node = json!({
            "@id": format!("iri://test/{}", i),
            "@type": "PerformanceTest",
            "@context": "https://agent-os.org/context/test",
            "summary": "性能测试节点",
            "data": {
                "field1": "value1",
                "field2": 42,
                "field3": [1, 2, 3]
            }
        });
        
        let _serialized = serde_json::to_string(&node).unwrap();
        let _deserialized: serde_json::Value = serde_json::from_str(&_serialized).unwrap();
        
        total_time += start.elapsed().as_micros() as u64;
    }
    
    let avg_time = total_time / iterations;
    println!("JSON-LD 序列化/反序列化平均时间: {} μs", avg_time);
    assert!(avg_time < 1000, "平均序列化时间应该小于1ms");
}

#[test]
fn test_performance_sparql_query() {
    let blackboard = Blackboard::new().unwrap();
    let config = CoreConfig::default();
    
    for i in 0..50 {
        let node = json!({
            "@id": format!("iri://test/node/{}", i),
            "@type": if i % 2 == 0 { "TypeA" } else { "TypeB" },
            "index": i
        }).to_string();
        blackboard.write_node(&format!("iri://test/node/{}", i), &node, &config).unwrap();
    }
    
    let start = Instant::now();
    let sparql = "SELECT ?s WHERE { ?s a <http://agent-os.org/type/TypeA> }";
    let results = blackboard.query(sparql).unwrap();
    let query_time = start.elapsed().as_millis();
    
    println!("SPARQL 查询时间: {} ms, 结果数: {}", query_time, results.len());
    assert!(query_time < 100, "SPARQL查询应该小于100ms");
}

#[test]
fn test_performance_projection() {
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let projection = ProjectionEngine::new(blackboard.clone(), 5000);
    let config = CoreConfig::default();
    
    for i in 0..30 {
        let node = json!({
            "@id": format!("iri://task/perf/node/{}", i),
            "@type": "TestNode",
            "summary": format!("节点 {}", i),
            "data": "x".repeat(50)
        }).to_string();
        blackboard.write_node(&format!("iri://task/perf/node/{}", i), &node, &config).unwrap();
    }
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let start = Instant::now();
    let result = rt.block_on(async {
        projection.project("iri://task/perf", "summary_only", HashMap::new()).await
    }).unwrap();
    let projection_time = start.elapsed().as_millis();
    
    println!("投影生成时间: {} ms, 结果大小: {} bytes", projection_time, result.len());
    assert!(projection_time < 50, "投影生成应该小于50ms");
}
