use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::CoreError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MCPMessageType {
    Request,
    Response,
    Notification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPMessage {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<MCPError>,
}

impl MCPMessage {
    pub fn request(method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::String(uuid::Uuid::new_v4().hyphenated().to_string())),
            method: Some(method.to_string()),
            params,
            result: None,
            error: None,
        }
    }

    pub fn response(result: Value, id: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: None,
            params: None,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(code: i32, message: &str, id: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: None,
            params: None,
            result: None,
            error: Some(MCPError {
                code,
                message: message.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPError {
    pub code: i32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

impl MCPTool {
    pub fn new(name: &str, description: &str, input_schema: Value) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema,
            output_schema: None,
        }
    }

    pub fn to_mcp_format(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(default = "default_mime_type")]
    pub mime_type: String,
}

fn default_mime_type() -> String {
    "text/plain".to_string()
}

impl MCPResource {
    pub fn new(uri: &str, name: &str, description: &str) -> Self {
        Self {
            uri: uri.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            mime_type: "text/plain".to_string(),
        }
    }

    pub fn to_mcp_format(&self) -> Value {
        json!({
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPPrompt {
    pub name: String,
    pub description: String,
    pub arguments: Vec<Value>,
}

impl MCPPrompt {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            arguments: Vec::new(),
        }
    }

    pub fn to_mcp_format(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        })
    }
}

#[async_trait]
pub trait ToolHandler: Send + Sync {
    async fn execute(&self, arguments: Value) -> Result<Value, CoreError>;
}

pub struct FunctionToolHandler<F>(pub F);

#[async_trait]
impl<F, Fut> ToolHandler for FunctionToolHandler<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Value, CoreError>> + Send,
{
    async fn execute(&self, arguments: Value) -> Result<Value, CoreError> {
        (self.0)(arguments).await
    }
}

pub struct MCPToolRegistry {
    tools: RwLock<HashMap<String, MCPTool>>,
    handlers: RwLock<HashMap<String, Arc<dyn ToolHandler>>>,
}

impl MCPToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            handlers: RwLock::new(HashMap::new()),
        }
    }

    pub fn register<H: ToolHandler + 'static>(&self, tool: MCPTool, handler: H) {
        let name = tool.name.clone();
        self.tools.write().insert(name.clone(), tool);
        self.handlers.write().insert(name, Arc::new(handler));
    }

    pub fn register_fn<F, Fut>(&self, tool: MCPTool, handler: F)
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Value, CoreError>> + Send + 'static,
    {
        self.register(tool, FunctionToolHandler(handler));
    }

    pub fn list_tools(&self) -> Vec<MCPTool> {
        self.tools.read().values().cloned().collect()
    }

    pub fn get_tool(&self, name: &str) -> Option<MCPTool> {
        self.tools.read().get(name).cloned()
    }

    pub async fn execute(&self, name: &str, arguments: Value) -> Result<Value, CoreError> {
        let handler = self.handlers.read().get(name).cloned();

        match handler {
            Some(h) => h.execute(arguments).await,
            None => Err(CoreError::Internal {
                message: format!("Tool not found: {}", name),
            }),
        }
    }
}

impl Default for MCPToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MCPServer {
    #[allow(dead_code)]
    name: String,
    tools: Arc<MCPToolRegistry>,
    resources: RwLock<HashMap<String, MCPResource>>,
    prompts: RwLock<HashMap<String, MCPPrompt>>,
}

impl MCPServer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tools: Arc::new(MCPToolRegistry::new()),
            resources: RwLock::new(HashMap::new()),
            prompts: RwLock::new(HashMap::new()),
        }
    }

    pub fn register_tool<H: ToolHandler + 'static>(&self, tool: MCPTool, handler: H) {
        self.tools.register(tool, handler);
    }

    pub fn register_resource(&self, resource: MCPResource) {
        self.resources
            .write()
            .insert(resource.uri.clone(), resource);
    }

    pub fn register_prompt(&self, prompt: MCPPrompt) {
        self.prompts.write().insert(prompt.name.clone(), prompt);
    }

    pub fn tools(&self) -> &MCPToolRegistry {
        &self.tools
    }

    pub async fn handle_message(&self, message: MCPMessage) -> MCPMessage {
        let method = match &message.method {
            Some(m) => m,
            None => return MCPMessage::error(-32600, "Invalid request", message.id.clone()),
        };

        match method.as_str() {
            "tools/list" => {
                let tools: Vec<Value> = self
                    .tools
                    .list_tools()
                    .iter()
                    .map(|t| t.to_mcp_format())
                    .collect();
                MCPMessage::response(json!({"tools": tools}), message.id.unwrap_or(Value::Null))
            }

            "tools/call" => {
                let params = message.params.clone().unwrap_or(Value::Null);
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

                match self.tools.execute(tool_name, arguments).await {
                    Ok(result) => MCPMessage::response(
                        json!({"content": [{"type": "text", "text": result.to_string()}]}),
                        message.id.unwrap_or(Value::Null),
                    ),
                    Err(e) => MCPMessage::error(-32603, &e.to_string(), message.id),
                }
            }

            "resources/list" => {
                let resources: Vec<Value> = self
                    .resources
                    .read()
                    .values()
                    .map(|r| r.to_mcp_format())
                    .collect();
                MCPMessage::response(
                    json!({"resources": resources}),
                    message.id.unwrap_or(Value::Null),
                )
            }

            "prompts/list" => {
                let prompts: Vec<Value> = self
                    .prompts
                    .read()
                    .values()
                    .map(|p| p.to_mcp_format())
                    .collect();
                MCPMessage::response(
                    json!({"prompts": prompts}),
                    message.id.unwrap_or(Value::Null),
                )
            }

            _ => MCPMessage::error(-32601, &format!("Method not found: {}", method), message.id),
        }
    }
}

pub struct MCPClient {
    server: Option<Arc<MCPServer>>,
}

impl MCPClient {
    pub fn new() -> Self {
        Self { server: None }
    }

    pub fn connect(&mut self, server: Arc<MCPServer>) {
        self.server = Some(server);
    }

    pub async fn list_tools(&self) -> Result<Vec<MCPTool>, CoreError> {
        let server = self.server.as_ref().ok_or_else(|| CoreError::Internal {
            message: "Not connected to server".to_string(),
        })?;

        let message = MCPMessage::request("tools/list", None);
        let response = server.handle_message(message).await;

        if let Some(error) = response.error {
            return Err(CoreError::Internal {
                message: error.message,
            });
        }

        let tools_data = response
            .result
            .and_then(|r| r.get("tools").cloned())
            .unwrap_or(Value::Array(Vec::new()));

        let tools: Vec<MCPTool> = serde_json::from_value(tools_data).unwrap_or_default();
        Ok(tools)
    }

    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, CoreError> {
        let server = self.server.as_ref().ok_or_else(|| CoreError::Internal {
            message: "Not connected to server".to_string(),
        })?;

        let message = MCPMessage::request(
            "tools/call",
            Some(json!({
                "name": name,
                "arguments": arguments,
            })),
        );

        let response = server.handle_message(message).await;

        if let Some(error) = response.error {
            return Err(CoreError::Internal {
                message: error.message,
            });
        }

        let content = response
            .result
            .and_then(|r| r.get("content").cloned())
            .unwrap_or(Value::Array(Vec::new()));

        if let Some(first) = content.as_array().and_then(|a| a.first()) {
            if first.get("type").and_then(|t| t.as_str()) == Some("text") {
                let text = first.get("text").and_then(|t| t.as_str()).unwrap_or("");
                return serde_json::from_str(text).or_else(|_| Ok(Value::String(text.to_string())));
            }
        }

        Ok(content)
    }

    pub async fn list_resources(&self) -> Result<Vec<MCPResource>, CoreError> {
        let server = self.server.as_ref().ok_or_else(|| CoreError::Internal {
            message: "Not connected to server".to_string(),
        })?;

        let message = MCPMessage::request("resources/list", None);
        let response = server.handle_message(message).await;

        if let Some(error) = response.error {
            return Err(CoreError::Internal {
                message: error.message,
            });
        }

        let resources_data = response
            .result
            .and_then(|r| r.get("resources").cloned())
            .unwrap_or(Value::Array(Vec::new()));

        let resources: Vec<MCPResource> =
            serde_json::from_value(resources_data).unwrap_or_default();
        Ok(resources)
    }
}

impl Default for MCPClient {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_default_mcp_server() -> MCPServer {
    let server = MCPServer::new("agent-os");

    server.tools.register_fn(
        MCPTool::new(
            "file_read",
            "Read content from a file",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "default": "utf-8"},
                },
                "required": ["path"],
            }),
        ),
        |args| async move {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let encoding = args
                .get("encoding")
                .and_then(|v| v.as_str())
                .unwrap_or("utf-8");

            match std::fs::read_to_string(path) {
                Ok(content) => Ok(json!({"content": content, "path": path, "encoding": encoding})),
                Err(e) => Err(CoreError::Internal {
                    message: e.to_string(),
                }),
            }
        },
    );

    server.tools.register_fn(
        MCPTool::new(
            "file_write",
            "Write content to a file",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            }),
        ),
        |args| async move {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");

            match std::fs::write(path, content) {
                Ok(_) => Ok(json!({"success": true, "path": path})),
                Err(e) => Err(CoreError::Internal {
                    message: e.to_string(),
                }),
            }
        },
    );

    server.tools.register_fn(
        MCPTool::new(
            "http_request",
            "Make an HTTP request",
            json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "method": {"type": "string", "default": "GET"},
                    "headers": {"type": "object"},
                    "body": {"type": "object"},
                    "timeout": {"type": "number", "default": 30},
                },
                "required": ["url"],
            }),
        ),
        |args| async move {
            let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
            let method = args.get("method").and_then(|v| v.as_str()).unwrap_or("GET");

            Ok(json!({
                "url": url,
                "method": method,
                "status": "simulated",
                "note": "HTTP client requires reqwest integration"
            }))
        },
    );

    server
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_server() {
        let server = MCPServer::new("test");

        server.tools.register_fn(
            MCPTool::new("echo", "Echo input", json!({"type": "object"})),
            |args| async move { Ok(args) },
        );

        let message = MCPMessage::request("tools/list", None);
        let response = server.handle_message(message).await;

        assert!(response.result.is_some());
        let result = response.result.as_ref().unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 1);
    }

    #[tokio::test]
    async fn test_mcp_client() {
        let server = Arc::new(create_default_mcp_server());
        let mut client = MCPClient::new();
        client.connect(server);

        let tools = client.list_tools().await.unwrap();
        assert!(!tools.is_empty());
    }

    #[test]
    fn test_mcp_message() {
        let req = MCPMessage::request("test/method", Some(json!({"arg": "value"})));
        assert_eq!(req.method, Some("test/method".to_string()));
        assert!(req.id.is_some());

        let resp = MCPMessage::response(json!({"result": "ok"}), Value::String("1".to_string()));
        assert!(resp.result.is_some());

        let err = MCPMessage::error(-32600, "Invalid request", None);
        assert!(err.error.is_some());
    }
}
