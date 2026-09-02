use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::core::core_types::SemanticCore;
use crate::knowledge_graph::rdf_mapper::RdfMapper;
use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::knowledge_graph::types::{EdgeDef, LLMExtractionOutput, NodeDef};
use crate::tools::tool_guard::{GuardAuditEntry, GUARD_AUDIT_LOG};

pub struct AppState {
    pub core: Arc<SemanticCore>,
    pub kg_store: Arc<oxigraph::store::Store>,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Deserialize)]
pub struct TaskRequest {
    pub user_input: String,
}

#[derive(Deserialize)]
pub struct NodeWriteRequest {
    pub task_iri: String,
    pub json_ld: String,
    pub created_by: Option<String>,
}

#[derive(Deserialize)]
pub struct ProjectionRequest {
    pub task_iri: String,
    pub frame_name: Option<String>,
    pub params: Option<HashMap<String, String>>,
}

#[derive(Deserialize)]
pub struct StreamTaskRequest {
    pub prompt: String,
    pub task_iri: Option<String>,
    pub include_thought: Option<bool>,
    pub include_tool_calls: Option<bool>,
}

#[derive(Deserialize)]
pub struct RealtimeStatusRequest {
    pub task_iri: String,
}

#[derive(Deserialize)]
pub struct KgImportRequest {
    pub nodes: Vec<NodeDef>,
    #[serde(default)]
    pub edges: Vec<EdgeDef>,
    pub graph: String,
    #[serde(default = "default_true")]
    pub clear_before: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
pub struct KgQueryRequest {
    pub sparql: String,
    pub named_graph: Option<String>,
}

#[derive(Serialize)]
pub struct StreamEventResponse {
    pub event_type: String,
    pub data: Value,
}

pub fn build_router(
    core: Arc<SemanticCore>,
    kg_store: Arc<oxigraph::store::Store>,
    auth_token: Option<String>,
) -> Router {
    let state = Arc::new(AppState { core, kg_store });

    let protected = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/api/v1/tasks", post(create_task_handler))
        .route("/api/v1/tasks/:task_iri", get(get_task_handler))
        .route("/api/v1/tasks/stream", post(stream_task_handler))
        .route(
            "/api/v1/tasks/:task_iri/status",
            get(get_realtime_status_handler),
        )
        .route(
            "/api/v1/tasks/:task_iri/details",
            get(get_execution_details_handler),
        )
        .route("/api/v1/nodes", post(write_node_handler))
        .route("/api/v1/nodes/:node_iri", get(read_node_handler))
        .route("/api/v1/projections", post(get_projection_handler))
        .route("/api/v1/events", post(emit_event_handler))
        .route("/api/v1/batch/events", get(stream_batch_events_handler))
        .route("/api/v1/skills", get(list_skills_handler))
        .route("/api/v1/guard/audit", get(guard_audit_handler))
        .route("/api/v1/guard/stats", get(guard_stats_handler))
        .route("/api/v1/kg/import", post(kg_import_handler))
        .route("/api/v1/kg/query", post(kg_query_handler));
    let protected = if let Some(token) = auth_token.filter(|token| !token.trim().is_empty()) {
        protected.layer(middleware::from_fn_with_state(
            Arc::new(token),
            require_bearer,
        ))
    } else {
        protected
    };

    Router::new()
        .route("/health", get(health_handler))
        .merge(protected)
        .with_state(state)
}

async fn require_bearer(
    State(expected_token): State<Arc<String>>,
    request: Request,
    next: Next,
) -> Response {
    let header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    if crate::api::auth::valid_bearer_header(header, expected_token.as_str()) {
        next.run(request).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

async fn health_handler() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "l2_nodes": state.core.blackboard.node_count(),
        "l2_bytes": state.core.blackboard.total_bytes(),
        "events": state.core.events.event_count(),
        "subscribers": state.core.events.subscriber_count(),
        "skills": state.core.skills.skill_count(),
        "checkpoints": state.core.checkpoints.checkpoint_count(),
    }))
}

async fn guard_audit_handler() -> impl IntoResponse {
    let log = GUARD_AUDIT_LOG.read();
    let entries: Vec<GuardAuditEntry> = log.clone();
    Json(json!({
        "total": entries.len(),
        "entries": entries,
    }))
}

async fn guard_stats_handler() -> impl IntoResponse {
    let log = GUARD_AUDIT_LOG.read();
    let total = log.len();
    if total == 0 {
        return Json(json!({
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "pass_rate": 1.0,
        }));
    }
    let passed = log.iter().filter(|e| e.validation_passed).count();
    Json(json!({
        "total_checks": total,
        "passed_checks": passed,
        "failed_checks": total - passed,
        "pass_rate": passed as f64 / total as f64,
    }))
}

async fn create_task_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TaskRequest>,
) -> impl IntoResponse {
    match state.core.init_task(&req.user_input, None, None).await {
        Ok(task_iri) => (
            StatusCode::CREATED,
            Json(json!({"task_iri": task_iri, "status": "created"})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        ),
    }
}

async fn get_task_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(task_iri): axum::extract::Path<String>,
) -> impl IntoResponse {
    match state.core.read_node(&task_iri).await {
        Ok(Some(node)) => Json(json!({
            "task_iri": task_iri,
            "found": true,
            "node": node,
        })),
        Ok(None) => Json(json!({
            "task_iri": task_iri,
            "found": false,
        })),
        Err(e) => Json(json!({
            "task_iri": task_iri,
            "error": e.to_string(),
        })),
    }
}

async fn stream_task_handler(
    State(_state): State<Arc<AppState>>,
    Json(_req): Json<StreamTaskRequest>,
) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "HTTP task execution is not wired; use the gRPC ExecuteTaskStream API"
        })),
    )
        .into_response()
}

async fn get_realtime_status_handler(
    State(_state): State<Arc<AppState>>,
    axum::extract::Path(task_iri): axum::extract::Path<String>,
) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "task_iri": task_iri,
            "error": "HTTP execution state is not wired; use the gRPC GetRealtimeStatus API"
        })),
    )
}

async fn get_execution_details_handler(
    State(_state): State<Arc<AppState>>,
    axum::extract::Path(task_iri): axum::extract::Path<String>,
) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "task_iri": task_iri,
            "error": "HTTP execution details are not wired; use the gRPC GetExecutionDetails API"
        })),
    )
}

async fn write_node_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NodeWriteRequest>,
) -> impl IntoResponse {
    match state
        .core
        .write_node(&req.task_iri, &req.json_ld, None, req.created_by.as_deref())
        .await
    {
        Ok(node_iri) => (
            StatusCode::CREATED,
            Json(json!({"node_iri": node_iri, "accepted": true})),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"accepted": false, "error": e.to_string()})),
        ),
    }
}

async fn get_projection_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProjectionRequest>,
) -> impl IntoResponse {
    let frame = req
        .frame_name
        .unwrap_or_else(|| "reference_only".to_string());
    let params = req.params.unwrap_or_default();
    match state
        .core
        .projection
        .project(&req.task_iri, &frame, params)
        .await
    {
        Ok(projection) => Json(json!({
            "projection": serde_json::from_str::<Value>(&projection).ok(),
            "frame": frame,
            "task_iri": req.task_iri,
        })),
        Err(e) => Json(json!({"error": e.to_string(), "task_iri": req.task_iri})),
    }
}

async fn read_node_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(node_iri): axum::extract::Path<String>,
) -> impl IntoResponse {
    match state.core.read_node(&node_iri).await {
        Ok(Some(node)) => Json(json!({
            "found": true,
            "json_ld": node.json_ld,
        })),
        Ok(None) => Json(json!({"found": false})),
        Err(e) => Json(json!({"found": false, "error": e.to_string()})),
    }
}

async fn emit_event_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let task_iri = payload
        .get("task_iri")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let event_type = payload
        .get("event_type")
        .and_then(|v| v.as_str())
        .unwrap_or("CUSTOM");
    let source = payload
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("http_api");
    let event_id = state
        .core
        .emit_event(task_iri, event_type, source, &payload.to_string())
        .await;
    Json(json!({"event_id": event_id, "status": "emitted"}))
}

async fn list_skills_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let skills = state.core.skills.list_all_skills();
    Json(json!({
        "count": skills.len(),
        "skills": skills,
    }))
}

async fn stream_batch_events_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let event_bus = state.core.events.clone();
    let mut rx = event_bus.subscribe();

    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    if !event.event_type.starts_with("BATCH_") {
                        continue;
                    }
                    let payload: Value =
                        serde_json::from_str(&event.payload).unwrap_or(Value::Null);
                    let data = json!({
                        "channel": "batch",
                        "event_type": event.event_type,
                        "source": event.source_agent_iri,
                        "task_iri": event.task_iri,
                        "timestamp": event.timestamp.to_rfc3339(),
                        "payload": payload,
                    });
                    yield Ok::<Event, Infallible>(
                        Event::default()
                            .event("batch")
                            .data(data.to_string()),
                    );
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Expand short namespace prefixes to absolute IRIs for Oxigraph.
/// e.g. "aps:Bench" → "http://aps.local/ontology/Bench"
///      "graph:aps/benches" → "http://aps.local/graph/benches"
///      "rdfs:subClassOf" → "http://www.w3.org/2000/01/rdf-schema#subClassOf"
fn expand_iri(s: &str) -> String {
    if s.contains('/') && (s.starts_with("http://") || s.starts_with("https://")) {
        return s.to_string();
    }
    if let Some(rest) = s.strip_prefix("aps:") {
        format!("http://aps.local/ontology/{}", rest)
    } else if let Some(rest) = s.strip_prefix("graph:aps/") {
        format!("http://aps.local/graph/{}", rest)
    } else if let Some(rest) = s.strip_prefix("rdfs:") {
        format!("http://www.w3.org/2000/01/rdf-schema#{}", rest)
    } else if let Some(rest) = s.strip_prefix("rdf:") {
        format!("http://www.w3.org/1999/02/22-rdf-syntax-ns#{}", rest)
    } else {
        s.to_string()
    }
}

fn expand_extraction(mut extraction: LLMExtractionOutput) -> LLMExtractionOutput {
    for node in &mut extraction.nodes {
        node.node_type = expand_iri(&node.node_type);
    }
    for edge in &mut extraction.edges {
        edge.relation = expand_iri(&edge.relation);
    }
    extraction
}

async fn kg_import_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<KgImportRequest>,
) -> impl IntoResponse {
    let store = state.kg_store.clone();
    let graph_iri = expand_iri(&req.graph);

    if req.clear_before {
        let clear = format!("DELETE WHERE {{ GRAPH <{}> {{ ?s ?p ?o . }} }}", graph_iri);
        if let Err(e) = store.update(&clear) {
            tracing::warn!(graph = %graph_iri, "KG clear skipped: {}", e);
        }
    }

    let extraction = expand_extraction(LLMExtractionOutput {
        nodes: req.nodes,
        edges: req.edges,
    });
    let result = RdfMapper::map_extraction(&extraction, &graph_iri);

    let kg = match KnowledgeGraphStore::with_shared_store(store) {
        Ok(kg) => kg,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))),
    };

    match kg.write_quads(&result.quads, &graph_iri) {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "entity_count": result.entity_count,
                "relation_count": result.relation_count,
                "quad_count": result.quads.len(),
                "graph": req.graph,
            })),
        ),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))),
    }
}

async fn kg_query_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<KgQueryRequest>,
) -> impl IntoResponse {
    let store = state.kg_store.clone();
    let kg = match KnowledgeGraphStore::with_shared_store(store) {
        Ok(kg) => kg,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))),
    };

    let named_graph = req.named_graph.as_deref().map(|g| expand_iri(g));
    match kg.query_sparql(&req.sparql, named_graph.as_deref()) {
        Ok(results) => (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "results": results,
                "count": results.len(),
            })),
        ),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e}))),
    }
}
