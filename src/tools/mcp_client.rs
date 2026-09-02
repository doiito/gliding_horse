use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::{debug, info, warn};

use crate::config::{McpServerConfig, McpStdioServerConfig};
use crate::CoreError;

static JSON_RPC_VERSION: &str = "2.0";

// ── Data types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerState {
    pub name: String,
    pub url: String,
    pub transport: String, // "http" or "stdio"
    pub status: String,
    pub tools: Vec<McpTool>,
    pub server_info: Option<Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    params: Value,
    id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    result: Option<Value>,
    error: Option<JsonRpcError>,
    id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    data: Option<Value>,
}

// ── Stdio process management ──────────────────────────────────────

/// Manages a spawned MCP server subprocess with stdin/stdout JSON-RPC transport.
struct StdioProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    buffer: String,
    timeout: Duration,
    pending_responses: HashMap<u64, JsonRpcResponse>,
}

impl StdioProcess {
    /// Spawn a new MCP server process.
    async fn spawn(config: &McpStdioServerConfig) -> Result<Self, CoreError> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args);
        cmd.env_clear();
        cmd.envs(crate::tools::process_env::sanitized_child_environment(
            false,
        ));
        crate::tools::process_env::overlay_explicit_environment(&mut cmd, &config.env);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        // Discard stderr — MCP server logs (startup banners, usage stats) would
        // corrupt the TUI display if inherited. Errors surface via JSON-RPC.
        cmd.stderr(std::process::Stdio::null());
        cmd.kill_on_drop(true);

        let mut child = cmd.spawn().map_err(|e| CoreError::Internal {
            message: format!("Failed to start MCP server '{}': {}", config.command, e),
        })?;

        let stdin = child.stdin.take().ok_or_else(|| CoreError::Internal {
            message: "Failed to get MCP server stdin".to_string(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| CoreError::Internal {
            message: "Failed to get MCP server stdout".to_string(),
        })?;

        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            buffer: String::new(),
            timeout: Duration::from_millis(config.tool_call_timeout_ms.unwrap_or(30_000).max(1)),
            pending_responses: HashMap::new(),
        })
    }

    /// Send a JSON-RPC request and read the matching response.
    async fn send_request(
        &mut self,
        request: &JsonRpcRequest,
    ) -> Result<JsonRpcResponse, CoreError> {
        let timeout = self.timeout;
        match tokio::time::timeout(timeout, self.send_request_inner(request)).await {
            Ok(result) => result,
            Err(_) => Err(CoreError::Internal {
                message: format!(
                    "MCP stdio request '{}' timed out after {}ms",
                    request.method,
                    timeout.as_millis()
                ),
            }),
        }
    }

    async fn send_request_inner(
        &mut self,
        request: &JsonRpcRequest,
    ) -> Result<JsonRpcResponse, CoreError> {
        if let Some(response) = self.pending_responses.remove(&request.id) {
            return Ok(response);
        }
        let json_str = serde_json::to_string(request).map_err(|e| CoreError::Internal {
            message: format!("JSON serialization failed: {}", e),
        })?;

        // Write request to stdin (newline-delimited JSON)
        self.stdin
            .write_all(json_str.as_bytes())
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Failed to write to MCP stdin: {}", e),
            })?;
        self.stdin
            .write_all(b"\n")
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Failed to write newline to MCP stdin: {}", e),
            })?;
        self.stdin.flush().await.map_err(|e| CoreError::Internal {
            message: format!("Failed to flush MCP stdin: {}", e),
        })?;

        loop {
            self.buffer.clear();
            self.stdout
                .read_line(&mut self.buffer)
                .await
                .map_err(|e| CoreError::Internal {
                    message: format!("Failed to read MCP stdout: {}", e),
                })?;
            if self.buffer.is_empty() {
                return Err(CoreError::Internal {
                    message: "MCP server stdout closed".to_string(),
                });
            }
            let value: Value =
                serde_json::from_str(self.buffer.trim()).map_err(|e| CoreError::Internal {
                    message: format!(
                        "Failed to parse MCP message: {} (raw: {})",
                        e,
                        self.buffer.trim()
                    ),
                })?;
            let Some(response_id) = value.get("id").and_then(Value::as_u64) else {
                debug!(method = ?value.get("method"), "Ignoring MCP notification");
                continue;
            };
            let response: JsonRpcResponse =
                serde_json::from_value(value).map_err(|e| CoreError::Internal {
                    message: format!("Failed to parse MCP response: {e}"),
                })?;
            if response_id == request.id {
                return Ok(response);
            }
            if self.pending_responses.len() >= 128 {
                return Err(CoreError::Internal {
                    message: "Too many out-of-order MCP responses".to_string(),
                });
            }
            self.pending_responses.insert(response_id, response);
        }
    }

    /// Check if the process is still alive.
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

// ── McpClient ─────────────────────────────────────────────────────

pub struct McpClient {
    servers: HashMap<String, McpServerState>,
    processes: HashMap<String, StdioProcess>,
    stdio_configs: HashMap<String, McpStdioServerConfig>,
    http_headers: HashMap<String, BTreeMap<String, String>>,
    http_client: Client,
    request_timeout: Duration,
    next_id: std::sync::atomic::AtomicU64,
}

impl McpClient {
    pub fn new() -> Self {
        Self::with_timeout(30)
    }

    pub fn with_timeout(timeout_secs: u64) -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .unwrap_or_default();
        Self {
            servers: HashMap::new(),
            processes: HashMap::new(),
            stdio_configs: HashMap::new(),
            http_headers: HashMap::new(),
            http_client,
            request_timeout: Duration::from_secs(timeout_secs.max(1)),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    fn next_request_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Register an HTTP MCP server by URL.
    pub fn register_server(&mut self, name: &str, server_url: &str) {
        self.register_http_server(name, server_url, BTreeMap::new());
    }

    fn register_http_server(
        &mut self,
        name: &str,
        server_url: &str,
        headers: BTreeMap<String, String>,
    ) {
        info!(server = %name, url = %server_url, transport = "http", "registering MCP server");
        self.servers.insert(
            name.to_string(),
            McpServerState {
                name: name.to_string(),
                url: server_url.to_string(),
                transport: "http".to_string(),
                status: "registered".to_string(),
                tools: Vec::new(),
                server_info: None,
                error: None,
            },
        );
        self.http_headers.insert(name.to_string(), headers);
    }

    /// Register a stdio MCP server (spawns subprocess on connect).
    pub fn register_stdio_server(&mut self, name: &str, config: &McpStdioServerConfig) {
        info!(server = %name, command = %config.command, transport = "stdio", "registering MCP Stdio server");
        self.servers.insert(
            name.to_string(),
            McpServerState {
                name: name.to_string(),
                url: String::new(),
                transport: "stdio".to_string(),
                status: "registered".to_string(),
                tools: Vec::new(),
                server_info: None,
                error: None,
            },
        );
        // Store config alongside server state for later spawning
        let mut config = config.clone();
        config
            .tool_call_timeout_ms
            .get_or_insert(self.request_timeout.as_millis().min(u128::from(u64::MAX)) as u64);
        self.stdio_configs.insert(name.to_string(), config);
    }

    /// Register an MCP server from a generic `McpServerConfig` enum.
    pub fn register_from_config(&mut self, name: &str, config: &McpServerConfig) {
        match config {
            McpServerConfig::Http(http_cfg) => {
                self.register_http_server(name, &http_cfg.url, http_cfg.headers.clone());
            }
            McpServerConfig::Stdio(stdio_cfg) => {
                self.register_stdio_server(name, stdio_cfg);
            }
        }
    }

    // ── Connection ────────────────────────────────────────────────

    pub async fn connect(&mut self, name: &str) -> Result<Vec<McpTool>, CoreError> {
        let transport = {
            let state = self.servers.get(name).ok_or_else(|| CoreError::Internal {
                message: format!("MCP server not registered: {}", name),
            })?;
            state.transport.clone()
        };

        match transport.as_str() {
            "http" => self.connect_http(name).await,
            "stdio" => self.connect_stdio(name).await,
            _ => Err(CoreError::Internal {
                message: format!("Unknown MCP transport type: {}", transport),
            }),
        }
    }

    async fn connect_http(&mut self, name: &str) -> Result<Vec<McpTool>, CoreError> {
        let (url, headers) = {
            let state = self
                .servers
                .get_mut(name)
                .ok_or_else(|| CoreError::Internal {
                    message: format!("MCP server not registered: {}", name),
                })?;
            state.status = "connecting".to_string();
            (
                state.url.clone(),
                self.http_headers.get(name).cloned().unwrap_or_default(),
            )
        };

        let request = JsonRpcRequest {
            jsonrpc: JSON_RPC_VERSION.to_string(),
            method: "tools/list".to_string(),
            params: json!({}),
            id: self.next_request_id(),
        };

        let tools = match self.send_rpc_http(&url, &headers, &request).await {
            Ok(response) => self.handle_connect_response(name, response).await,
            Err(e) => self.handle_connect_fallback(name, e).await,
        };

        Ok(tools)
    }

    async fn connect_stdio(&mut self, name: &str) -> Result<Vec<McpTool>, CoreError> {
        // Get the stdio config
        let config = self
            .stdio_configs
            .get(name)
            .cloned()
            .ok_or_else(|| CoreError::Internal {
                message: format!("MCP Stdio server config not found: {}", name),
            })?;

        // Update status
        if let Some(state) = self.servers.get_mut(name) {
            state.status = "connecting".to_string();
        }

        // Spawn the subprocess
        match StdioProcess::spawn(&config).await {
            Ok(mut process) => {
                let request = JsonRpcRequest {
                    jsonrpc: JSON_RPC_VERSION.to_string(),
                    method: "tools/list".to_string(),
                    params: json!({}),
                    id: self.next_request_id(),
                };

                match process.send_request(&request).await {
                    Ok(response) => {
                        let tools = self
                            .parse_tools_from_response(name, &response)
                            .unwrap_or_default();
                        self.processes.insert(name.to_string(), process);

                        if let Some(state) = self.servers.get_mut(name) {
                            state.tools = tools.clone();
                            state.status = "connected".to_string();
                        }
                        info!(server = %name, tool_count = tools.len(), "MCP Stdio server connected successfully");
                        Ok(tools)
                    }
                    Err(e) => {
                        let _ = process.child.kill().await;
                        let _ = process.child.wait().await;
                        Ok(self.handle_connect_fallback(name, e).await)
                    }
                }
            }
            Err(e) => Ok(self.handle_connect_fallback(name, e).await),
        }
    }

    /// Parse tools from a JSON-RPC tools/list response.
    fn parse_tools_from_response(
        &self,
        name: &str,
        response: &JsonRpcResponse,
    ) -> Result<Vec<McpTool>, CoreError> {
        if let Some(ref result) = response.result {
            let tools: Vec<McpTool> = result
                .get("tools")
                .and_then(|t| serde_json::from_value(t.clone()).ok())
                .unwrap_or_default();
            Ok(tools)
        } else if let Some(ref error) = response.error {
            Err(CoreError::Internal {
                message: format!(
                    "MCP server '{}' returned error: {} ({})",
                    name, error.message, error.code
                ),
            })
        } else {
            Ok(Vec::new())
        }
    }

    async fn handle_connect_response(
        &mut self,
        name: &str,
        response: JsonRpcResponse,
    ) -> Vec<McpTool> {
        let tools = self
            .parse_tools_from_response(name, &response)
            .unwrap_or_default();
        if let Some(state) = self.servers.get_mut(name) {
            state.tools = tools.clone();
            state.status = "connected".to_string();
        }
        info!(server = %name, tool_count = tools.len(), "MCP server connected successfully");
        tools
    }

    async fn handle_connect_fallback(&mut self, name: &str, error: CoreError) -> Vec<McpTool> {
        let tools = vec![
            McpTool {
                name: "list_resources".to_string(),
                description: Some("List available resources".to_string()),
                input_schema: None,
            },
            McpTool {
                name: "read_resource".to_string(),
                description: Some("Read resource by URI".to_string()),
                input_schema: Some(json!({
                    "type": "object",
                    "properties": { "uri": {"type": "string"} },
                    "required": ["uri"]
                })),
            },
        ];
        if let Some(state) = self.servers.get_mut(name) {
            state.tools = tools.clone();
            state.status = "connected_fallback".to_string();
            state.error = Some(error.to_string());
        }
        warn!(server = %name, error = %error, "MCP server connection failed, using fallback tools");
        tools
    }

    // ── Tool execution ────────────────────────────────────────────

    pub async fn call_tool(
        &mut self,
        server: &str,
        tool: &str,
        arguments: &Value,
    ) -> Result<Value, CoreError> {
        let transport = {
            let state = self
                .servers
                .get(server)
                .ok_or_else(|| CoreError::Internal {
                    message: format!("MCP server not found: {}", server),
                })?;
            if state.status.starts_with("error") {
                return Err(CoreError::Internal {
                    message: format!("MCP server {} status abnormal: {}", server, state.status),
                });
            }
            state
                .tools
                .iter()
                .find(|t| t.name == tool)
                .ok_or_else(|| CoreError::Internal {
                    message: format!("Tool {} not found on server {}", tool, server),
                })?;
            state.transport.clone()
        };

        debug!(server = %server, tool = %tool, transport = %transport, "MCP tool call");

        let request = JsonRpcRequest {
            jsonrpc: JSON_RPC_VERSION.to_string(),
            method: "tools/call".to_string(),
            params: json!({
                "name": tool,
                "arguments": arguments,
            }),
            id: self.next_request_id(),
        };

        match transport.as_str() {
            "http" => {
                let url = self
                    .servers
                    .get(server)
                    .map(|s| s.url.clone())
                    .unwrap_or_default();
                let headers = self.http_headers.get(server).cloned().unwrap_or_default();
                self.call_tool_http(&url, &headers, &request).await
            }
            "stdio" => self.call_tool_stdio(server, &request).await,
            _ => Err(CoreError::Internal {
                message: format!("Unknown MCP transport type: {}", transport),
            }),
        }
    }

    async fn call_tool_http(
        &self,
        url: &str,
        headers: &BTreeMap<String, String>,
        request: &JsonRpcRequest,
    ) -> Result<Value, CoreError> {
        match self.send_rpc_http(url, headers, request).await {
            Ok(response) => Self::handle_call_response(response),
            Err(e) => Err(e),
        }
    }

    async fn call_tool_stdio(
        &mut self,
        server: &str,
        request: &JsonRpcRequest,
    ) -> Result<Value, CoreError> {
        let result = {
            let process = self
                .processes
                .get_mut(server)
                .ok_or_else(|| CoreError::Internal {
                    message: format!("MCP Stdio process not found: {}", server),
                })?;
            if !process.is_alive() {
                return Err(CoreError::Internal {
                    message: format!("MCP Stdio process for server '{}' has exited", server),
                });
            }
            process.send_request(request).await
        };
        if result
            .as_ref()
            .is_err_and(|error| error.to_string().contains("timed out"))
        {
            if let Some(mut process) = self.processes.remove(server) {
                let _ = process.child.kill().await;
                let _ = process.child.wait().await;
            }
            if let Some(state) = self.servers.get_mut(server) {
                state.status = "error_timeout".to_string();
                state.error = Some("MCP stdio request timed out; process terminated".to_string());
            }
        }
        result.and_then(Self::handle_call_response)
    }

    fn handle_call_response(response: JsonRpcResponse) -> Result<Value, CoreError> {
        if let Some(result) = response.result {
            Ok(result)
        } else if let Some(error) = response.error {
            Err(CoreError::Internal {
                message: format!("MCP tool call error: {} ({})", error.message, error.code),
            })
        } else {
            Ok(json!({"status": "ok"}))
        }
    }

    // ── Transport layer ───────────────────────────────────────────

    async fn send_rpc_http(
        &self,
        url: &str,
        headers: &BTreeMap<String, String>,
        request: &JsonRpcRequest,
    ) -> Result<JsonRpcResponse, CoreError> {
        let mut request_builder = self.http_client.post(url);
        for (name, value) in headers {
            request_builder = request_builder.header(name, value);
        }
        let response =
            request_builder
                .json(request)
                .send()
                .await
                .map_err(|e| CoreError::Internal {
                    message: format!("MCP HTTP request failed: {}", e),
                })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(CoreError::Internal {
                message: format!("MCP HTTP request failed with {status}: {body}"),
            });
        }
        let rpc_response: JsonRpcResponse =
            response.json().await.map_err(|e| CoreError::Internal {
                message: format!("MCP response parse failed: {}", e),
            })?;

        Ok(rpc_response)
    }

    // ── Query methods ─────────────────────────────────────────────

    pub fn list_servers(&self) -> Vec<&McpServerState> {
        self.servers.values().collect()
    }

    pub fn get_server(&self, name: &str) -> Option<&McpServerState> {
        self.servers.get(name)
    }

    pub fn all_tools(&self) -> Vec<(String, McpTool)> {
        let mut result = Vec::new();
        for (server_name, state) in &self.servers {
            for tool in &state.tools {
                result.push((server_name.clone(), tool.clone()));
            }
        }
        result
    }

    pub fn register_tools_to_skill_registry(
        &self,
        registry: &crate::tools::skill_registry::SkillRegistry,
    ) {
        for (server_name, state) in &self.servers {
            for tool in &state.tools {
                let iri = format!("iri://mcp/{}/{}", server_name, tool.name);
                let input_schema = tool
                    .input_schema
                    .clone()
                    .unwrap_or(json!({"type":"object","properties":{}}));
                let skill = crate::tools::skill_registry::SkillMeta {
                    skill_iri: iri.clone(),
                    name: tool.name.clone(),
                    description: tool.description.clone().unwrap_or_default(),
                    version: "0.1.0".to_string(),
                    category: "mcp".to_string(),
                    security_level: "normal".to_string(),
                    allowed_roles: vec![
                        "Plan".to_string(),
                        "Do".to_string(),
                        "Check".to_string(),
                        "Act".to_string(),
                    ],
                    input_schema,
                    output_schema: json!({"type":"object"}),
                    compiled_template: String::new(),
                    signature: None,
                    signature_algorithm: None,
                    input_mapping: Default::default(),
                    output_mapping: Default::default(),
                    skill_types: vec!["skill-types/MCPOperation".to_string()],
                    discovery_5w2h: None,
                };
                registry.register_skill(skill);
                debug!(iri = %iri, "MCP tool registered in SkillRegistry");
            }
        }
    }

    /// Register every connected tool of every server into the ToolExecutor.
    pub fn register_tools_to_tool_executor(
        &self,
        executor: &mut crate::tools::tool_executor::ToolExecutor,
        handle: Arc<tokio::sync::Mutex<Option<McpClient>>>,
    ) {
        for (server_name, tool) in self.all_tools() {
            let server = server_name.clone();
            let tool_name = tool.name.clone();
            let handle = handle.clone();
            let description = tool.description.clone().unwrap_or_default();
            let input_schema = tool
                .input_schema
                .clone()
                .unwrap_or(json!({"type":"object","properties":{}}));

            let server_for_fn = server.clone();
            let tool_name_for_fn = tool_name.clone();
            executor.register(
                &tool_name,
                &description,
                input_schema,
                Arc::new(move |input: Value| {
                    let handle = handle.clone();
                    let server = server_for_fn.clone();
                    let tool_name = tool_name_for_fn.clone();
                    Box::pin(async move {
                        let mut guard = handle.lock().await;
                        let client = guard.as_mut().ok_or_else(|| {
                            "MCP client not connected for tool dispatch".to_string()
                        })?;
                        client
                            .call_tool(&server, &tool_name, &input)
                            .await
                            .map_err(|e| e.to_string())
                    })
                }),
                &["Plan", "Do", "Check", "Act"],
            );
            debug!(server = %server, tool = %tool_name, "MCP tool registered in ToolExecutor");
        }
    }

    pub async fn kill_all_processes(&mut self) {
        let names: Vec<String> = self.processes.keys().cloned().collect();
        for name in names {
            if let Some(mut process) = self.processes.remove(&name) {
                let _ = process.child.kill().await;
                let _ = process.child.wait().await;
                info!(server = %name, "MCP Stdio process terminated");
            }
        }
    }
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new()
    }
}

// Engine shutdown performs kill + wait. `kill_on_drop(true)` on every stdio
// command is the abnormal-exit fallback when async shutdown cannot run.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{McpRemoteServerConfig, McpServerConfig};

    #[tokio::test]
    async fn test_mcp_client_register() {
        let mut client = McpClient::new();
        client.register_server("test", "http://localhost:8080/mcp");
        assert!(client.get_server("test").is_some());
        assert_eq!(client.get_server("test").unwrap().status, "registered");
    }

    #[test]
    fn test_unknown_server() {
        let client = McpClient::new();
        assert!(client.get_server("nonexistent").is_none());
    }

    #[test]
    fn test_all_tools_empty() {
        let client = McpClient::new();
        assert!(client.all_tools().is_empty());
    }

    #[test]
    fn test_register_to_skill_registry() {
        let mut client = McpClient::new();
        client.register_server("test", "http://localhost:8080/mcp");
        client.servers.get_mut("test").unwrap().tools = vec![McpTool {
            name: "test_tool".to_string(),
            description: Some("Test tool".to_string()),
            input_schema: Some(json!({"type":"object"})),
        }];
        let registry = crate::tools::skill_registry::SkillRegistry::new();
        client.register_tools_to_skill_registry(&registry);
    }

    #[tokio::test]
    async fn test_register_from_config_http() {
        let headers = std::collections::BTreeMap::from([(
            "Authorization".to_string(),
            "Bearer test-token".to_string(),
        )]);
        let config = McpServerConfig::Http(McpRemoteServerConfig {
            url: "http://localhost:9999/mcp".to_string(),
            headers: headers.clone(),
        });
        let mut client = McpClient::new();
        client.register_from_config("test-http", &config);
        let state = client.get_server("test-http").unwrap();
        assert_eq!(state.transport, "http");
        assert_eq!(state.url, "http://localhost:9999/mcp");
        assert_eq!(client.http_headers.get("test-http"), Some(&headers));
    }

    #[tokio::test]
    async fn test_register_from_config_stdio() {
        let config = McpServerConfig::Stdio(McpStdioServerConfig {
            command: "echo".to_string(),
            args: vec!["{}".to_string()],
            env: std::collections::BTreeMap::new(),
            tool_call_timeout_ms: None,
        });
        let mut client = McpClient::new();
        client.register_from_config("test-stdio", &config);
        let state = client.get_server("test-stdio").unwrap();
        assert_eq!(state.transport, "stdio");
        assert!(client.stdio_configs.contains_key("test-stdio"));
        assert_eq!(
            client.stdio_configs["test-stdio"].tool_call_timeout_ms,
            Some(30_000)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stdio_request_timeout_is_enforced_and_bounded() {
        let config = McpStdioServerConfig {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), "read _; sleep 5".to_string()],
            env: BTreeMap::new(),
            tool_call_timeout_ms: Some(40),
        };
        let mut process = StdioProcess::spawn(&config).await.unwrap();
        let request = JsonRpcRequest {
            jsonrpc: JSON_RPC_VERSION.to_string(),
            method: "tools/list".to_string(),
            params: json!({}),
            id: 1,
        };
        let started = std::time::Instant::now();
        let error = process.send_request(&request).await.unwrap_err();
        assert!(error.to_string().contains("timed out after 40ms"));
        assert!(started.elapsed() < Duration::from_secs(1));
        let _ = process.child.kill().await;
        let _ = process.child.wait().await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stdio_request_ignores_notifications_and_matches_response_id() {
        let script = concat!(
            "read _; ",
            "printf '%s\\n' ",
            "'{\"jsonrpc\":\"2.0\",\"method\":\"progress\",\"params\":{}}' ",
            "'{\"jsonrpc\":\"2.0\",\"result\":{\"wrong\":true},\"id\":99}' ",
            "'{\"jsonrpc\":\"2.0\",\"result\":{\"ok\":true},\"id\":42}'"
        );
        let config = McpStdioServerConfig {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: BTreeMap::new(),
            tool_call_timeout_ms: Some(1_000),
        };
        let mut process = StdioProcess::spawn(&config).await.unwrap();
        let request = JsonRpcRequest {
            jsonrpc: JSON_RPC_VERSION.to_string(),
            method: "tools/call".to_string(),
            params: json!({}),
            id: 42,
        };
        let response = process.send_request(&request).await.unwrap();
        assert_eq!(response.id, 42);
        assert_eq!(response.result, Some(json!({"ok": true})));
        assert!(process.pending_responses.contains_key(&99));
        let _ = process.child.wait().await;
    }

    #[tokio::test]
    async fn http_mcp_sends_configured_headers() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = vec![0u8; 8192];
            let read = stream.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..read]).to_ascii_lowercase();
            assert!(request.contains("authorization: bearer test-token"));
            let body = r#"{"jsonrpc":"2.0","result":{"tools":[]},"id":1}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let mut client = McpClient::new();
        client.register_from_config(
            "authenticated",
            &McpServerConfig::Http(McpRemoteServerConfig {
                url: format!("http://{address}/mcp"),
                headers: BTreeMap::from([(
                    "Authorization".to_string(),
                    "Bearer test-token".to_string(),
                )]),
            }),
        );
        let tools = client.connect("authenticated").await.unwrap();
        assert!(tools.is_empty());
        server.await.unwrap();
    }
}
