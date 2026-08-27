use axum::{routing::post, Json, Router};
use glidinghorse::skill_graph::graph_store::SkillGraphStore;
use glidinghorse::skill_graph::{MCPIntegration, MCPRegistry, MCPServerConfig, MCPToolInfo};
use glidinghorse::tools::mcp_client::McpClient;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::tools::ToolExecutor;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::net::TcpListener;

// ── Mock MCP Server ──────────────────────────────────────────────

/// Standard mock MCP handler: responds to tools/list with 3 browser tools,
/// tools/call with a mock result, and other methods with error.
async fn mock_mcp_handler(Json(body): Json<Value>) -> Json<Value> {
    let method = body.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = body.get("id").and_then(|v| v.as_u64()).unwrap_or(0);

    match method {
        "tools/list" => Json(json!({
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": "browser_navigate",
                        "description": "Navigate to a URL in the browser",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "The URL to navigate to"}
                            },
                            "required": ["url"]
                        }
                    },
                    {
                        "name": "browser_click",
                        "description": "Click an element on the page",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "selector": {"type": "string", "description": "CSS selector"}
                            },
                            "required": ["selector"]
                        }
                    },
                    {
                        "name": "browser_snapshot",
                        "description": "Take a screenshot of the current page",
                        "input_schema": {"type": "object", "properties": {}}
                    }
                ]
            },
            "id": id
        })),
        "tools/call" => {
            let tool_name = body["params"]["name"].as_str().unwrap_or("unknown");
            Json(json!({
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string(&json!({
                            "status": "ok",
                            "tool": tool_name,
                            "result": "mock execution successful"
                        })).unwrap_or_default()
                    }]
                },
                "id": id
            }))
        }
        _ => Json(json!({
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found"},
            "id": id
        })),
    }
}

/// Handler that returns a single tool named "tool_a"
async fn mock_mcp_handler_a(Json(body): Json<Value>) -> Json<Value> {
    let id = body.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
    Json(json!({
        "jsonrpc": "2.0",
        "result": {
            "tools": [{"name": "tool_a", "description": "Tool A", "input_schema": {}}]
        },
        "id": id
    }))
}

/// Handler that returns a single tool named "tool_b"
async fn mock_mcp_handler_b(Json(body): Json<Value>) -> Json<Value> {
    let id = body.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
    Json(json!({
        "jsonrpc": "2.0",
        "result": {
            "tools": [{"name": "tool_b", "description": "Tool B", "input_schema": {}}]
        },
        "id": id
    }))
}

/// Start a mock MCP server on a random port and return its address.
async fn start_mock_server(router: Router) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    addr
}

fn chrome_router() -> Router {
    Router::new().route("/mcp", post(mock_mcp_handler))
}

// ── Tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mcp_client_register_and_connect() {
    let addr = start_mock_server(chrome_router()).await;
    let url = format!("http://{}/mcp", addr);

    let mut client = McpClient::new();
    assert!(client.list_servers().is_empty());

    // Register
    client.register_server("chrome", &url);
    let state = client.get_server("chrome").unwrap();
    assert_eq!(state.status, "registered");
    assert_eq!(state.tools.len(), 0);

    // Connect
    let tools = client.connect("chrome").await.unwrap();
    assert_eq!(tools.len(), 3);

    let state = client.get_server("chrome").unwrap();
    assert_eq!(state.status, "connected");
    assert_eq!(state.tools.len(), 3);

    // Tool names
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"browser_navigate"));
    assert!(names.contains(&"browser_click"));
    assert!(names.contains(&"browser_snapshot"));

    // List servers
    assert_eq!(client.list_servers().len(), 1);
}

#[tokio::test]
async fn test_mcp_client_call_tool() {
    let addr = start_mock_server(chrome_router()).await;
    let url = format!("http://{}/mcp", addr);

    let mut client = McpClient::new();
    client.register_server("chrome", &url);
    client.connect("chrome").await.unwrap();

    // Call navigate tool
    let result = client
        .call_tool(
            "chrome",
            "browser_navigate",
            &json!({"url": "https://example.com"}),
        )
        .await;
    assert!(result.is_ok());
    let content = result.unwrap();
    let content_str = content.to_string();
    assert!(
        content_str.contains("browser_navigate"),
        "result should mention tool name: {}",
        content_str
    );

    // Call unknown tool → should error
    let err = client.call_tool("chrome", "nonexistent", &json!({})).await;
    assert!(err.is_err(), "unknown tool should return error");
}

#[tokio::test]
async fn test_mcp_client_all_tools() {
    let addr = start_mock_server(chrome_router()).await;
    let url = format!("http://{}/mcp", addr);

    let mut client = McpClient::new();
    client.register_server("chrome", &url);
    client.connect("chrome").await.unwrap();

    let all = client.all_tools();
    assert_eq!(all.len(), 3);
    // Each entry is (server_name, McpTool)
    assert!(all.iter().all(|(s, _)| s == "chrome"));
}

#[tokio::test]
async fn test_mcp_client_register_to_skill_registry() {
    let addr = start_mock_server(chrome_router()).await;
    let url = format!("http://{}/mcp", addr);

    let mut client = McpClient::new();
    client.register_server("chrome", &url);
    client.connect("chrome").await.unwrap();

    let registry = SkillRegistry::new();
    client.register_tools_to_skill_registry(&registry);

    // The registry should now have 3 MCP skills.
    // There's no public API to query count, so we verify no panic/error occurred.
}

#[tokio::test]
async fn test_mcp_client_connection_fallback() {
    // Connect to an unreachable address — should get fallback tools
    let mut client = McpClient::new();
    client.register_server("offline", "http://127.0.0.1:1/mcp");

    let result = client.connect("offline").await;
    assert!(result.is_ok());
    let tools = result.unwrap();
    // Fallback tools: list_resources, read_resource
    assert_eq!(tools.len(), 2);

    let state = client.get_server("offline").unwrap();
    assert_eq!(state.status, "connected_fallback");
}

#[tokio::test]
async fn test_multiple_mcp_servers() {
    let addr_a = start_mock_server(Router::new().route("/mcp", post(mock_mcp_handler_a))).await;
    let addr_b = start_mock_server(Router::new().route("/mcp", post(mock_mcp_handler_b))).await;

    let mut client = McpClient::new();
    client.register_server("server_a", &format!("http://{}/mcp", addr_a));
    client.register_server("server_b", &format!("http://{}/mcp", addr_b));

    client.connect("server_a").await.unwrap();
    client.connect("server_b").await.unwrap();

    assert_eq!(client.list_servers().len(), 2);

    let all = client.all_tools();
    assert_eq!(all.len(), 2);

    let names: Vec<&str> = all.iter().map(|(_, t)| t.name.as_str()).collect();
    assert!(names.contains(&"tool_a"));
    assert!(names.contains(&"tool_b"));
}

#[tokio::test]
async fn test_mcp_client_idempotent_connect() {
    let addr = start_mock_server(chrome_router()).await;
    let url = format!("http://{}/mcp", addr);

    let mut client = McpClient::new();
    client.register_server("chrome", &url);

    // Connect twice — second call should still work
    client.connect("chrome").await.unwrap();
    let tools = client.connect("chrome").await.unwrap();
    assert_eq!(tools.len(), 3);
}

#[tokio::test]
async fn test_mcp_client_unknown_server_error() {
    let mut client = McpClient::new();
    let result = client.connect("nonexistent").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_mcp_client_tool_not_found_error() {
    let addr = start_mock_server(chrome_router()).await;
    let url = format!("http://{}/mcp", addr);

    let mut client = McpClient::new();
    client.register_server("chrome", &url);

    // Call before connect — should error because status != connected
    let result = client
        .call_tool("chrome", "browser_navigate", &json!({}))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_register_tools_to_tool_executor() {
    let addr = start_mock_server(chrome_router()).await;
    let url = format!("http://{}/mcp", addr);

    let mut client = McpClient::new();
    client.register_server("chrome", &url);
    client.connect("chrome").await.unwrap();

    let handle = Arc::new(tokio::sync::Mutex::new(Some(client)));
    let mut executor = ToolExecutor::new();
    {
        let guard = handle.lock().await;
        guard
            .as_ref()
            .unwrap()
            .register_tools_to_tool_executor(&mut executor, handle.clone());
    }

    let result = executor
        .execute("browser_navigate", json!({"url": "https://example.com"}))
        .await;
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(
        output.to_string().contains("browser_navigate"),
        "real dispatch should reach the MCP server, got: {}",
        output
    );
    assert!(
        !output.to_string().contains("simulated"),
        "registered MCP tool must not return a simulated result"
    );
}

#[tokio::test]
async fn test_mcp_integration_invoke_tool_real_dispatch() {
    let addr = start_mock_server(chrome_router()).await;
    let url = format!("http://{}/mcp", addr);

    let mut client = McpClient::new();
    client.register_server("chrome", &url);
    client.connect("chrome").await.unwrap();
    let handle = Arc::new(tokio::sync::Mutex::new(Some(client)));

    let registry = Arc::new(MCPRegistry::new());
    let store = Arc::new(SkillGraphStore::new());
    let integration = MCPIntegration::new(registry.clone(), store.clone()).with_client(handle);

    let server_config = MCPServerConfig {
        server_id: "chrome".to_string(),
        server_name: "Chrome".to_string(),
        endpoint: Some(url),
        trust_level: glidinghorse::skill_graph::TrustLevel::High,
        ..Default::default()
    };
    registry.register_server(server_config).await.unwrap();
    registry
        .register_tool(MCPToolInfo::new(
            "chrome",
            "browser_navigate",
            "Navigate to a URL",
        ))
        .await
        .unwrap();

    let sync = integration.sync_tools_to_skills("chrome").await.unwrap();
    assert_eq!(sync.tools_added, 1);

    let result = integration
        .invoke_tool(
            "iri://skills/mcp/chrome-browser_navigate",
            json!({"url": "https://example.com"}),
        )
        .await;
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(
        output.to_string().contains("browser_navigate"),
        "invoke_tool should reach the real MCP server, got: {}",
        output
    );
    assert!(
        !output.to_string().contains("simulated"),
        "invoke_tool must not return a simulated result"
    );
}
