use glidinghorse::config::settings::ToolResultRouterSettings;
use glidinghorse::tools::result_router::graphify::GraphifyEngine;
use glidinghorse::tools::result_router::micro_tools::MicroToolGenerator;
use glidinghorse::tools::result_router::router::ResultRouter;
use glidinghorse::tools::result_router::summary;
use glidinghorse::tools::result_router::{RouteDecision, SchemaAnalysis};
use serde_json::json;

#[test]
fn test_router_small_result_passthrough() {
    let settings = ToolResultRouterSettings::default();
    let router = ResultRouter::new(&settings);
    let result = "small result";
    let decision = router.route(result, "test_tool", "call_1");
    assert_eq!(decision, RouteDecision::PassThrough);
}

#[test]
fn test_router_medium_result_summarize() {
    let settings = ToolResultRouterSettings::default();
    let router = ResultRouter::new(&settings);
    // threshold_small=16384, micro_tool_threshold=16384 → need >= 16384 for Summarize
    let result = "x".repeat(20_000);
    let decision = router.route(&result, "test_tool", "call_2");
    assert!(matches!(decision, RouteDecision::Summarize { .. }));
}

#[test]
fn test_router_large_json_graphify() {
    let settings = ToolResultRouterSettings::default();
    let router = ResultRouter::new(&settings);
    // Must exceed threshold_large (32768) and be structured JSON → Graphify
    let items: Vec<serde_json::Value> = (0..1200)
        .map(|i| json!({"id": format!("item_{}", i), "name": format!("item_{}", i), "value": i * 10}))
        .collect();
    let result = serde_json::to_string(&items).unwrap();
    assert!(
        result.len() > 32768,
        "result size {} should exceed threshold_large",
        result.len()
    );
    let decision = router.route(&result, "test_tool", "call_3");
    assert!(matches!(decision, RouteDecision::Graphify { .. }));
}

#[test]
fn test_router_large_text_summarize() {
    let settings = ToolResultRouterSettings::default();
    let router = ResultRouter::new(&settings);
    // Must exceed threshold_large (32768) and not be structured JSON → Summarize
    let result = "line\n".repeat(8000);
    let decision = router.route(&result, "test_tool", "call_4");
    assert!(matches!(decision, RouteDecision::Summarize { .. }));
}

#[test]
fn test_router_disabled_fallback() {
    let mut settings = ToolResultRouterSettings::default();
    settings.enabled = false;
    let router = ResultRouter::new(&settings);
    let result = "x".repeat(10000);
    let decision = router.route(&result, "test_tool", "call_5");
    assert_eq!(decision, RouteDecision::Truncate { max_chars: 8000 });
}

#[test]
fn test_json_smart_truncate_array() {
    let items: Vec<serde_json::Value> = (0..50)
        .map(|i| json!({"id": i, "name": format!("item_{}", i)}))
        .collect();
    let json = serde_json::to_string(&items).unwrap();
    let result = summary::smart_truncate(&json, 500);
    assert!(result.len() < 600);
    assert!(result.contains("truncated"));
}

#[test]
fn test_json_smart_truncate_text() {
    let text = "line1\nline2\nline3\nline4\nline5\n".repeat(200);
    let result = summary::smart_truncate(&text, 1000);
    assert!(result.len() < 1100);
    assert!(result.contains("truncated"));
}

#[test]
fn test_text_summary_generation() {
    let text = "line\n".repeat(1000);
    let summary_text = summary::generate_text_summary(&text, "test_tool", 200);
    assert!(summary_text.contains("test_tool"));
    assert!(summary_text.contains("1000 lines"));
    assert!(summary_text.contains("read_full_result"));
}

#[test]
fn test_graphify_json_array() {
    let mut engine = GraphifyEngine::new(100).unwrap();
    let items: Vec<serde_json::Value> = (0..10)
        .map(|i| json!({"id": format!("item_{}", i), "name": format!("item_{}", i), "value": i * 10}))
        .collect();
    let result = engine.graphify_json(&json!(items), "integ_call_1", 100);
    assert!(result.entity_count > 0);
    assert!(result.graph_name.contains("integ_call_1"));
    assert!(!result.summary.is_empty());
}

#[test]
fn test_graphify_entity_limit() {
    let mut engine = GraphifyEngine::new(100).unwrap();
    let items: Vec<serde_json::Value> = (0..50)
        .map(|i| json!({"id": format!("item_{}", i), "name": format!("item_{}", i)}))
        .collect();
    let result = engine.graphify_json(&json!(items), "integ_call_2", 5);
    assert!(result.entity_count <= 5);
}

#[test]
fn test_micro_tools_generation() {
    let analysis = SchemaAnalysis {
        entity_types: vec![("Person".to_string(), 10)],
        relation_types: vec!["works_for".to_string()],
        property_names: vec!["name".to_string()],
        total_entities: 10,
        total_relations: 5,
    };
    let tools = MicroToolGenerator::generate_from_schema(&analysis, "integ_call_3", 5);
    assert!(tools.len() >= 2);
    assert!(tools.iter().any(|t| t.name == "query_person"));
    assert!(tools.iter().any(|t| t.name == "get_entity_details"));

    let msg = MicroToolGenerator::format_tool_injection_message("测试摘要", &tools);
    assert!(msg.contains("测试摘要"));
    assert!(msg.contains("query_person"));
}

#[test]
fn test_micro_tools_read_full() {
    let tool = MicroToolGenerator::generate_read_full_tool("integ_call_4", "storage_key", 2000);
    assert_eq!(tool.name, "read_full_result");
}

#[test]
fn test_config_default_values() {
    let settings = ToolResultRouterSettings::default();
    assert!(settings.enabled);
    assert_eq!(settings.threshold_small, 16384);
    assert_eq!(settings.threshold_large, 32768);
    assert_eq!(settings.preview_size, 2000);
    assert_eq!(settings.max_graph_entities, 500);
    assert_eq!(settings.max_micro_tools, 5);
    assert_eq!(settings.sparql_query_timeout_ms, 100);
    assert!(settings.auto_cleanup);
}

#[test]
fn test_full_pipeline_json_graphify() {
    let settings = ToolResultRouterSettings::default();
    let router = ResultRouter::new(&settings);

    // Must exceed threshold_large (32768) and be structured JSON → Graphify
    let items: Vec<serde_json::Value> = (0..1200)
        .map(|i| json!({"id": format!("item_{}", i), "name": format!("item_{}", i), "value": i * 10}))
        .collect();
    let result_str = serde_json::to_string(&items).unwrap();
    assert!(
        result_str.len() > 32768,
        "result size {} should exceed threshold_large",
        result_str.len()
    );

    let decision = router.route(&result_str, "test_tool", "pipeline_1");
    assert!(matches!(decision, RouteDecision::Graphify { .. }));

    if let RouteDecision::Graphify { call_id, .. } = decision {
        let mut engine = GraphifyEngine::new(settings.max_graph_entities).unwrap();
        let graphify_result =
            engine.graphify_json(&json!(items), &call_id, settings.max_graph_entities);
        assert!(graphify_result.entity_count > 0);

        let analysis = SchemaAnalysis {
            entity_types: graphify_result
                .entity_types
                .iter()
                .map(|t| (t.clone(), 0))
                .collect(),
            relation_types: vec![],
            property_names: vec![],
            total_entities: graphify_result.entity_count,
            total_relations: graphify_result.relation_count,
        };
        let tools =
            MicroToolGenerator::generate_from_schema(&analysis, &call_id, settings.max_micro_tools);
        assert!(!tools.is_empty());

        let msg =
            MicroToolGenerator::format_tool_injection_message(&graphify_result.summary, &tools);
        assert!(msg.contains("Data summary"));
    }
}

#[test]
fn test_full_pipeline_text_summarize() {
    let settings = ToolResultRouterSettings::default();
    let router = ResultRouter::new(&settings);

    // Must exceed threshold_large (32768) and not be structured JSON → Summarize
    let result_str = "line\n".repeat(8000);
    let decision = router.route(&result_str, "test_tool", "pipeline_2");
    assert!(matches!(decision, RouteDecision::Summarize { .. }));

    if let RouteDecision::Summarize { preview_size, .. } = decision {
        let preview = summary::generate_text_summary(&result_str, "test_tool", preview_size);
        assert!(preview.contains("test_tool"));
        assert!(preview.contains("read_full_result"));
    }
}
