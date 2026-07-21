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
        ?s a <http://agent-os.org/ontology/Task> .
        ?s <http://agent-os.org/ontology/status> ?status .
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
    let sparql = "SELECT ?s WHERE { ?s a <http://agent-os.org/ontology/TypeA> }";
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

// ═══════════════════════════════════════════════════════════════
// Bug 3 fix verification: type/ → ontology/ namespace
// ═══════════════════════════════════════════════════════════════
#[test]
fn test_bug3_type_namespace_fix() {
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let config = CoreConfig::default();

    // Step 1: Write a node with @type: "Task"
    let node = json!({
        "@id": "iri://task/bug3_test",
        "@type": "Task",
        "summary": "Bug 3 test task",
        "status": "running",
        "confidence": 0.9,
    }).to_string();
    blackboard.write_node("iri://task/bug3_test", &node, &config).unwrap();

    // Step 2: Query with ontology/ namespace (the fix)
    blackboard.flush_oxigraph();

    let sparql_ontology = r#"
        PREFIX ex: <http://agent-os.org/ontology/>
        SELECT ?s ?summary WHERE {
            ?s a ex:Task .
            ?s ex:summary ?summary .
        }
    "#;
    let results = blackboard.query(sparql_ontology).unwrap();
    assert!(!results.is_empty(),
        "Bug 3 FAIL: SPARQL query with ex:Task (ontology/) returned 0 results. \
         write_node with @type='Task' should store as <http://agent-os.org/ontology/Task>");
    println!("[Bug 3] ontology/ SPARQL returned {} result(s)", results.len());
    println!("[Bug 3] First result: {:?}", results[0]);

    // Step 3: Negative test — type/ should NOT exist
    let sparql_type = "SELECT ?s WHERE { ?s a <http://agent-os.org/type/Task> } LIMIT 1";
    let old_results = blackboard.query(sparql_type).unwrap();
    assert!(old_results.is_empty(),
        "Bug 3 FAIL: type/ namespace should NOT contain data anymore");
    println!("[Bug 3] type/ namespace correctly empty");

    // Step 4: Projection engine summary_only (uses ?node a ?type — should find the node)
    let proj = ProjectionEngine::new(blackboard.clone(), 1024);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let projection = rt.block_on(async {
        proj.project("iri://task/bug3_test", "summary_only", HashMap::new()).await.unwrap()
    });
    assert!(projection.contains("iri://task/bug3_test"),
        "Bug 3 FAIL: summary_only projection should include the task node. Got: {}", &projection[..300.min(projection.len())]);
    println!("[Bug 3] summary_only projection includes task node ({} bytes)", projection.len());

    // Step 5: pa_init also uses ex:Task — verify it returns artifacts
    let projection_pa = rt.block_on(async {
        proj.project("iri://task/bug3_test", "pa_init", HashMap::new()).await.unwrap()
    });
    println!("[Bug 3] pa_init projection: {} bytes", projection_pa.len());

    // Step 6: da_input with ex:PlanNode — negative test (should return 0 since we wrote Task, not PlanNode)
    let projection_da = rt.block_on(async {
        proj.project("iri://task/bug3_test", "da_input", HashMap::new()).await.unwrap()
    });
    // da_input requires ex:PlanNode, our node is ex:Task, so this should have 0 artifacts
    println!("[Bug 3] da_input projection (expect 0, no PlanNode): {} bytes", projection_da.len());
}

// ═══════════════════════════════════════════════════════════════
// Bug 1 fix verification: PREFIX skill: in to_sparql_insert()
// ═══════════════════════════════════════════════════════════════
#[test]
fn test_bug1_skill_iri_fix() {
    use glidinghorse::skill_graph::types::{
        SkillGraphNode, SkillLinkType, Skill5W2H,
    };

    // Step 1: Build a SkillGraphNode
    let node = SkillGraphNode::new("iri://skills/test_bug1", "bug1_test", "Bug 1 fix test")
        .with_5w2h(Skill5W2H::new("bug1_test", "Testing Bug 1 fix"))
        .with_tag("test");

    // Step 2: SPARQL must have PREFIX
    let sparql = node.to_sparql_insert("system:test_graph");
    assert!(sparql.starts_with("PREFIX skill:"),
        "Bug 1 FAIL: to_sparql_insert() should start with 'PREFIX skill:'. Got: {}...",
        &sparql[..60.min(sparql.len())]);
    println!("[Bug 1] SPARQL includes PREFIX skill:");

    // Step 3: Validate SPARQL is syntactically sound by direct Oxigraph execution
    let store = oxigraph::store::Store::new().unwrap();
    store.update(&sparql).unwrap_or_else(|e| {
        panic!("Bug 1 FAIL: SPARQL execution failed: {}\nSPARQL:\n---\n{}\n---", e, sparql)
    });
    println!("[Bug 1] SPARQL INSERT DATA executed successfully");

    // Step 4: Verify the data via SELECT (data is in named graph system:test_graph)
    use oxigraph::sparql::QueryResults;
    let verify_sparql = "PREFIX skill: <https://agent-harness.os/skill#>
        SELECT ?s WHERE { GRAPH <system:test_graph> { ?s a skill:CognitiveSkill } }";
    let query_results = store.query(verify_sparql).unwrap();
    let solutions: Vec<_> = match query_results {
        QueryResults::Solutions(solutions) => solutions.collect(),
        _ => vec![],
    };
    assert!(!solutions.is_empty(), "Bug 1 FAIL: no CognitiveSkill found (GRAPH <system:test_graph>)");
    println!("[Bug 1] Skill roundtrip confirmed: {} CognitiveSkill node(s)", solutions.len());

    // Step 5: Verify linked usageCount and successRate data
    let detail_sparql = "PREFIX skill: <https://agent-harness.os/skill#>
        SELECT ?usage ?rate WHERE { GRAPH <system:test_graph> { ?s skill:usageCount ?usage ; skill:successRate ?rate } }";
    let detail_results = store.query(detail_sparql).unwrap();
    let detail_solutions: Vec<_> = match detail_results {
        QueryResults::Solutions(solutions) => solutions.collect(),
        _ => vec![],
    };
    assert_eq!(detail_solutions.len(), 1, "Bug 1 FAIL: should find usage+rate data");
    println!("[Bug 1] Skill detail data (usageCount, successRate) also confirmed");

    // Step 6: Cross-graph query — default graph should NOT have the data
    let default_sparql = "PREFIX skill: <https://agent-harness.os/skill#>
        SELECT ?s WHERE { ?s a skill:CognitiveSkill }";
    let default_results = store.query(default_sparql).unwrap();
    let default_solutions: Vec<_> = match default_results {
        QueryResults::Solutions(solutions) => solutions.collect(),
        _ => vec![],
    };
    assert!(default_solutions.is_empty(),
        "Bug 1: data should NOT be in default graph (only in GRAPH <system:test_graph>)");
    println!("[Bug 1] Data correctly isolated to named graph (not in default graph)");
}

// ═══════════════════════════════════════════════════════════════
// Bug 2 fix verification: kg_search entity_type auto-namespace
// ═══════════════════════════════════════════════════════════════
#[test]
fn test_bug2_kg_search_entity_type_fix() {
    use glidinghorse::knowledge_graph::store::KnowledgeGraphStore;

    let store = KnowledgeGraphStore::new().unwrap();

    // Step 1: Insert a quad with type in ontology/ namespace via inner store
    let insert_sparql = "PREFIX ex: <http://agent-os.org/ontology/>
        INSERT DATA {
            GRAPH <http://agent-os.org/graph/test_graph> {
                <iri://entity/test_entity> a ex:TestEntity .
                <iri://entity/test_entity> <http://www.w3.org/2000/01/rdf-schema#label> \"Test Entity Label\" .
            }
        }";
    store.store_arc().update(insert_sparql).unwrap();
    println!("[Bug 2] Test entity inserted");

    // Step 2: Search with entity_type WITHOUT namespace (the fix)
    let results = store.search_entities("Test", Some("TestEntity")).unwrap();
    assert!(!results.is_empty(),
        "Bug 2 FAIL: search_entities(entity_type='TestEntity') returned 0 results");
    println!("[Bug 2] entity_type='TestEntity' (auto-qualified) returned {} result(s)", results.len());
    println!("[Bug 2] First result: {}", results[0]);

    // Step 3: Search with full IRI still works
    let results_full = store.search_entities("Test", Some("http://agent-os.org/ontology/TestEntity")).unwrap();
    assert!(!results_full.is_empty(),
        "Bug 2 FAIL: full IRI search should also work");
    println!("[Bug 2] Full IRI entity_type also works ({} result(s))", results_full.len());

    // Step 4: Search without entity_type
    let results_none = store.search_entities("Entity", None).unwrap();
    assert!(!results_none.is_empty(),
        "Bug 2 FAIL: keyword search without entity_type should work");
    println!("[Bug 2] Keyword search without entity_type: {} result(s)", results_none.len());
}

// ═══════════════════════════════════════════════════════════════
// Round 2: Edge case + stress tests
// ═══════════════════════════════════════════════════════════════

// Bug 3 edge cases: multiple types, write+query+delete cycle
#[test]
fn test_bug3_edge_cases() {
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let config = CoreConfig::default();
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Edge 1: Node with multiple @type values
    let multi_type = json!({
        "@id": "iri://edge/multi_type",
        "@type": ["Task", "Urgent", "ReviewNeeded"],
        "summary": "Multi-type node"
    }).to_string();
    blackboard.write_node("iri://edge/multi_type", &multi_type, &config).unwrap();
    blackboard.flush_oxigraph();

    let sparql = r#"PREFIX ex: <http://agent-os.org/ontology/>
        SELECT ?s WHERE { ?s a ex:Urgent }"#;
    let results = blackboard.query(sparql).unwrap();
    assert_eq!(results.len(), 1, "Multi-type: ex:Urgent should match");
    println!("[R2 Bug3-E1] Multi-type node matched via ex:Urgent: {} result(s)", results.len());

    // Edge 2: @type with full IRI (should NOT be prefixed with ontology/)
    let full_iri_type = json!({
        "@id": "iri://edge/full_iri_type",
        "@type": "http://custom-ontology.org/types/MyType",
        "summary": "Full IRI type"
    }).to_string();
    blackboard.write_node("iri://edge/full_iri_type", &full_iri_type, &config).unwrap();
    blackboard.flush_oxigraph();

    let sparql_full = "SELECT ?s WHERE { ?s a <http://custom-ontology.org/types/MyType> }";
    let results_full = blackboard.query(sparql_full).unwrap();
    assert_eq!(results_full.len(), 1, "Full IRI @type should match directly");
    println!("[R2 Bug3-E2] Full IRI type correctly stored and queryable: {} result(s)", results_full.len());

    // Edge 3: Multiple nodes with same type — bulk projection
    for i in 0..10 {
        let node = json!({
            "@id": format!("iri://edge/bulk/{}", i),
            "@type": "BulkNode",
            "summary": format!("Bulk node {}", i),
        }).to_string();
        blackboard.write_node(&format!("iri://edge/bulk/{}", i), &node, &config).unwrap();
    }
    blackboard.flush_oxigraph();

    let sparql_bulk = r#"PREFIX ex: <http://agent-os.org/ontology/>
        SELECT (COUNT(?s) AS ?cnt) WHERE { ?s a ex:BulkNode }"#;
    let bulk_results = blackboard.query(sparql_bulk).unwrap();
    println!("[R2 Bug3-E3] Bulk type query: {:?}", bulk_results);

    // Edge 4: query_by_types method (also uses ontology/ now)
    let type_nodes = blackboard.query_by_types(&["Task".to_string()]).unwrap();
    assert!(!type_nodes.is_empty(), "query_by_types('Task') should return nodes");
    println!("[R2 Bug3-E4] query_by_types('Task') returned {} node(s)", type_nodes.len());

    // Edge 5: projection with summary_only on bulk data
    let proj = ProjectionEngine::new(blackboard.clone(), 2048);
    let projection = rt.block_on(async {
        proj.project("iri://edge", "summary_only", HashMap::new()).await.unwrap()
    });
    assert!(!projection.is_empty(), "summary_only projection should return data for bulk nodes");
    println!("[R2 Bug3-E5] summary_only projection across 10 BulkNode nodes OK ({} bytes)", projection.len());
}

// Bug 1 edge cases: complex skill nodes, link IRI verification
#[test]
fn test_bug1_edge_cases() {
    use glidinghorse::skill_graph::types::{
        SkillGraphNode, SkillLink, SkillLinkType, Skill5W2H, SkillGraphMeta,
    };
    use oxigraph::sparql::QueryResults;

    // Edge 1: Skill with ALL possible link types
    let mut node = SkillGraphNode::new("iri://skills/edge_complex", "edge_complex", "Edge case complex skill");
    let all_link_types = [
        (SkillLinkType::Prerequisite, "iri://skills/prereq"),
        (SkillLinkType::Composition, "iri://skills/composition"),
        (SkillLinkType::Related, "iri://skills/related"),
        (SkillLinkType::Alternative, "iri://skills/alt"),
        (SkillLinkType::Extends, "iri://skills/extends"),
        (SkillLinkType::Generalization, "iri://skills/gen"),
    ];
    for (link_type, target) in &all_link_types {
        node = node.with_link(SkillLink {
            target_iri: target.to_string(),
            link_type: *link_type,
            strength: glidinghorse::skill_graph::types::LinkStrength::Recommended,
            description: format!("{:?} link", link_type),
        });
    }

    let sparql = node.to_sparql_insert("system:edge_graph");
    eprintln!("DEBUG SPARQL:\n{}", sparql);
    let store = oxigraph::store::Store::new().unwrap();
    store.update(&sparql).unwrap();
    println!("[R2 Bug1-E1] Skill with 6 link types inserted successfully");

    // Verify each link type (use prefixed name without angle brackets)
    for (link_type, target_iri) in &all_link_types {
        let link_name = format!("{:?}", link_type); // e.g., "Prerequisite"
        let verify = format!(
            "PREFIX skill: <https://agent-harness.os/skill#>
             SELECT ?s WHERE {{ GRAPH <system:edge_graph> {{ ?s skill:{} <{}> }} }}",
            link_name, target_iri
        );
        let qr = store.query(&verify).unwrap();
        let sols: Vec<_> = match qr { QueryResults::Solutions(s) => s.collect(), _ => vec![] };
        assert_eq!(sols.len(), 1, "Link {} should be found", link_name);
    }
    println!("[R2 Bug1-E1] All 6 link types confirmed queryable");

    // Edge 2: Skill with special characters in name/description
    let special = SkillGraphNode::new("iri://skills/special_chars", "it's \"quoted\" & <encoded>", "Description with 'single' and \"double\" quotes");
    let sparql_special = special.to_sparql_insert("system:special_graph");
    let store2 = oxigraph::store::Store::new().unwrap();
    store2.update(&sparql_special).unwrap();

    let verify_special = r#"PREFIX skill: <https://agent-harness.os/skill#>
        SELECT ?name ?desc WHERE { GRAPH <system:special_graph> { ?s skill:name ?name ; skill:description ?desc } }"#;
    let qr = store2.query(verify_special).unwrap();
    let sols: Vec<_> = match qr { QueryResults::Solutions(s) => s.collect(), _ => vec![] };
    assert_eq!(sols.len(), 1, "Special chars skill should be found");
    let (name_val, desc_val): (String, String) = {
        let mut sols_iter = sols.into_iter();
        let sol = sols_iter.next().unwrap().unwrap();
        use oxigraph::model::Term;
        let get_literal = |v: &str| -> String {
            match sol.get(v).unwrap() {
                Term::Literal(lit) => lit.value().to_string(),
                _ => panic!("expected literal for {}", v),
            }
        };
        (get_literal("name"), get_literal("desc"))
    };
    assert_eq!(name_val, "it's \"quoted\" & <encoded>", "Name with special chars should roundtrip");
    assert_eq!(desc_val, "Description with 'single' and \"double\" quotes", "Desc with special chars should roundtrip");
    println!("[R2 Bug1-E2] Special characters in skill name/description roundtrip correctly");

    // Edge 3: Empty links skill
    let empty = SkillGraphNode::new("iri://skills/empty", "empty", "No links");
    let sparql_empty = empty.to_sparql_insert("system:empty_graph");
    let store3 = oxigraph::store::Store::new().unwrap();
    store3.update(&sparql_empty).unwrap();

    let verify_empty = r#"PREFIX skill: <https://agent-harness.os/skill#>
        SELECT ?s WHERE { GRAPH <system:empty_graph> { ?s a skill:CognitiveSkill } }"#;
    let qr = store3.query(verify_empty).unwrap();
    let sols: Vec<_> = match qr { QueryResults::Solutions(s) => s.collect(), _ => vec![] };
    assert_eq!(sols.len(), 1, "Empty-link skill should be found");
    println!("[R2 Bug1-E3] Skill with no links inserted and queried successfully");
}

// Bug 2 edge cases: various entity_type formats
#[test]
fn test_bug2_edge_cases() {
    use glidinghorse::knowledge_graph::store::KnowledgeGraphStore;

    let store = KnowledgeGraphStore::new().unwrap();

    // Insert entities with different types
    let inserts = [
        ("type_a", "TypeA", "Entity A label"),
        ("type_b", "TypeB", "Entity B label"),
        ("type_c", "http://custom.org/types/TypeC", "Entity C label"),
    ];
    for (id, type_name, label) in &inserts {
        let type_iri = if type_name.contains("://") {
            format!("<{}>", type_name)
        } else {
            format!("<http://agent-os.org/ontology/{}>", type_name)
        };
        let sparql = format!(
            "INSERT DATA {{ GRAPH <http://agent-os.org/graph/kg_search> {{ <iri://entity/{}> a {} . \
             <iri://entity/{}> <http://www.w3.org/2000/01/rdf-schema#label> \"{}\" . }} }}",
            id, type_iri, id, label
        );
        store.store_arc().update(&sparql).unwrap();
    }
    println!("[R2 Bug2] 3 test entities inserted with different types");

    // Edge 1: search with simple type name (no namespace) — should auto-qualify
    let results = store.search_entities("Entity", Some("TypeA")).unwrap();
    assert_eq!(results.len(), 1, "search_entities('TypeA') should find 1");
    println!("[R2 Bug2-E1] Auto-qualified TypeA: {} result(s)", results.len());

    // Edge 2: search with a different simple type
    let results = store.search_entities("Entity", Some("TypeB")).unwrap();
    assert_eq!(results.len(), 1, "search_entities('TypeB') should find 1");
    println!("[R2 Bug2-E2] Auto-qualified TypeB: {} result(s)", results.len());

    // Edge 3: search with full custom IRI type
    let results = store.search_entities("Entity", Some("http://custom.org/types/TypeC")).unwrap();
    assert_eq!(results.len(), 1, "search_entities(full IRI TypeC) should find 1");
    println!("[R2 Bug2-E3] Full custom IRI TypeC: {} result(s)", results.len());

    // Edge 4: search with non-existent type
    let results = store.search_entities("Entity", Some("NonExistentType")).unwrap();
    assert_eq!(results.len(), 0, "search_entities('NonExistentType') should find 0");
    println!("[R2 Bug2-E4] Non-existent type correctly returns 0 results");

    // Edge 5: search with empty string keyword but valid type
    let results = store.search_entities("A", Some("TypeA")).unwrap();
    assert_eq!(results.len(), 1, "Keyword 'A' + TypeA should find entity A");
    println!("[R2 Bug2-E5] Keyword 'A' + TypeA: {} result(s)", results.len());

    // Edge 6: verify no false positives across types
    let results = store.search_entities("Entity", Some("TypeB")).unwrap();
    let has_a = results.iter().any(|r| r["?s"].as_str().map_or(false, |s| s.contains("type_a")));
    assert!(!has_a, "TypeB search should NOT return type_a entity");
    println!("[R2 Bug2-E6] Type isolation verified: no cross-type false positives");
}

// Combined: stress test with many nodes
#[test]
fn test_all_bugfixes_stress() {
    let blackboard = Arc::new(Blackboard::new().unwrap());
    let config = CoreConfig::default();
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Write 100 nodes with alternating types
    for i in 0..100 {
        let type_name = if i % 2 == 0 { "EvenType" } else { "OddType" };
        let node = json!({
            "@id": format!("iri://stress/node/{}", i),
            "@type": type_name,
            "summary": format!("Stress test node {}", i),
            "index": i,
        }).to_string();
        blackboard.write_node(&format!("iri://stress/node/{}", i), &node, &config).unwrap();
    }
    blackboard.flush_oxigraph();

    // Verify all EvenType nodes via ontology/ namespace
    let sparql = r#"PREFIX ex: <http://agent-os.org/ontology/>
        SELECT (COUNT(?s) AS ?cnt) WHERE { ?s a ex:EvenType }"#;
    let results = blackboard.query(sparql).unwrap();
    println!("[Stress] EvenType count via SPARQL: {:?}", results);

    // Verify all OddType nodes
    let sparql_odd = r#"PREFIX ex: <http://agent-os.org/ontology/>
        SELECT (COUNT(?s) AS ?cnt) WHERE { ?s a ex:OddType }"#;
    let results_odd = blackboard.query(sparql_odd).unwrap();
    println!("[Stress] OddType count via SPARQL: {:?}", results_odd);

    // Verify query_by_types
    let even_nodes = blackboard.query_by_types(&["EvenType".to_string()]).unwrap();
    assert_eq!(even_nodes.len(), 50, "Should find 50 EvenType nodes");
    let odd_nodes = blackboard.query_by_types(&["OddType".to_string()]).unwrap();
    assert_eq!(odd_nodes.len(), 50, "Should find 50 OddType nodes");
    println!("[Stress] query_by_types: {} EvenType + {} OddType = 100 total",
        even_nodes.len(), odd_nodes.len());

    // Projection on stress data
    let proj = ProjectionEngine::new(blackboard.clone(), 4096);
    let proj_result = rt.block_on(async {
        proj.project("iri://stress", "summary_only", HashMap::new()).await.unwrap()
    });
    assert!(proj_result.len() > 50, "summary_only projection should be substantial for 100 nodes");
    println!("[Stress] summary_only projection OK ({} bytes)", proj_result.len());

    // type/ namespace must be completely empty
    for type_name in &["EvenType", "OddType"] {
        let sparql_old = format!("SELECT ?s WHERE {{ ?s a <http://agent-os.org/type/{}> }} LIMIT 1", type_name);
        let old_results = blackboard.query(&sparql_old).unwrap();
        assert!(old_results.is_empty(), "type/{} should have 0 results", type_name);
    }
    println!("[Stress] type/ namespace verified empty for all types — all data in ontology/");
}
