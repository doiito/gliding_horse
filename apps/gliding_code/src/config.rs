use glidinghorse::config::{GatewaySettings, McpStdioServerConfig};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct McpServerEntry {
    pub name: String,
    pub url: String,
}

/// A parsed entry for a stdio MCP server (from env or CLI args).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct McpStdioServerEntry {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub tool_call_timeout_ms: Option<u64>,
}

impl From<McpStdioServerEntry> for McpStdioServerConfig {
    fn from(entry: McpStdioServerEntry) -> Self {
        McpStdioServerConfig {
            command: entry.command,
            args: entry.args,
            env: entry.env,
            tool_call_timeout_ms: entry.tool_call_timeout_ms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CliConfig {
    pub gateway: GatewaySettings,
    pub model: String,
    pub workspace: String,
    pub max_iterations: u32,
    pub max_pdca_cycles: u32,
    pub max_l1_mb: u64,
    pub max_l2_mb: u64,
    pub max_l3_mb: u64,
    pub data_dir: Option<String>,
    /// JSON-LD 工作流文件路径（可选，替代 LLM 生成的 plan）
    pub workflow_path: Option<String>,
    /// MCP 服务器配置（名称→URL）
    pub mcp_servers: Vec<McpServerEntry>,
    /// MCP Stdio 服务器配置（名称→命令+参数）
    pub mcp_stdio_servers: Vec<(String, McpStdioServerEntry)>,
}

impl CliConfig {
    pub fn from_env_and_args(
        model: String,
        workspace: String,
        max_iterations: u32,
        max_pdca_cycles: u32,
        workflow_path: Option<String>,
    ) -> Self {
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .or_else(|_| std::env::var("AGENT_OS_GATEWAY_API_KEY"))
            .unwrap_or_else(|_| {
                eprintln!("错误: 请设置 DEEPSEEK_API_KEY 或 AGENT_OS_GATEWAY_API_KEY 环境变量");
                std::process::exit(1);
            });

        let base_url = std::env::var("DEEPSEEK_API_URL")
            .or_else(|_| std::env::var("AGENT_OS_GATEWAY_BASE_URL"))
            .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

        let model = if model.is_empty() {
            "deepseek-v4-flash".to_string()
        } else {
            model
        };

        // Responses API only affects deepseek-v4-flash (the gateway falls back to
        // chat completions for other models), so it is safe to enable by default.
        let use_responses_api = std::env::var("USE_RESPONSES_API")
            .or_else(|_| std::env::var("AGENT_OS_GATEWAY_USE_RESPONSES_API"))
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);

        let gateway = GatewaySettings {
            base_url,
            api_key,
            default_model: model.clone(),
            timeout_seconds: 300,
            max_retries: 2,
            retry_base_ms: 500,
            use_responses_api,
            model_mapping: HashMap::from([
                ("planning".to_string(), model.clone()),
                ("execution".to_string(), model.clone()),
                ("analysis".to_string(), model.clone()),
                ("default".to_string(), model.clone()),
            ]),
        };

        // Try to load memory limits from agent_os config file, fall back to env vars, then defaults
        let (max_l1_mb, max_l2_mb, max_l3_mb) = Self::load_memory_limits();
        let data_dir = std::env::var("GLIDING_HORSE_DATA").ok().or_else(|| {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .ok()
                .map(|home| format!("{}/.gliding_horse/data", home))
        });

        let mcp_servers = Self::load_mcp_servers();
        let mcp_stdio_servers = Self::load_mcp_stdio_servers();

        Self {
            gateway,
            model,
            workspace,
            max_iterations,
            max_pdca_cycles,
            max_l1_mb,
            max_l2_mb,
            max_l3_mb,
            data_dir,
            workflow_path,
            mcp_servers,
            mcp_stdio_servers,
        }
    }

    /// Load memory limits from agent_os Settings (config file / env vars) or use defaults.
    fn load_memory_limits() -> (u64, u64, u64) {
        // First, try to load from the agent_os Settings config file
        if let Ok(settings) = glidinghorse::config::Settings::load() {
            return (
                settings.memory.l1.max_memory_mb,
                settings.memory.l2.max_memory_mb,
                settings.memory.l3.max_memory_mb,
            );
        }
        // Fall back to environment variables
        let l1 = std::env::var("AGENT_OS_L1_MEMORY_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let l2 = std::env::var("AGENT_OS_L2_MEMORY_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let l3 = std::env::var("AGENT_OS_L3_MEMORY_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        (l1, l2, l3)
    }

    /// 从环境变量加载 MCP 服务器配置。
    /// 支持两种格式：
    /// 1. `GLIDING_HORSE_MCP_SERVERS` JSON 数组：
    ///    [{"name":"chrome","url":"http://localhost:3000/sse"}]
    /// 2. 独立的 `MCP_SERVER__{NAME}` 环境变量（值=URL），优先级更高：
    ///    MCP_SERVER__chrome=http://localhost:3000/sse
    fn load_mcp_servers() -> Vec<McpServerEntry> {
        let mut servers = Vec::new();

        // 优先从 JSON 环境变量加载
        if let Ok(json_str) = std::env::var("GLIDING_HORSE_MCP_SERVERS") {
            if let Ok(parsed) = serde_json::from_str::<Vec<McpServerEntry>>(&json_str) {
                servers = parsed;
            }
        }

        // 独立的 MCP_SERVER__{NAME} 环境变量覆盖/追加
        for (key, val) in std::env::vars() {
            if let Some(name) = key.strip_prefix("MCP_SERVER__") {
                if !name.is_empty() {
                    // 如果同名已存在则替换，否则追加
                    let name = name.to_lowercase();
                    if let Some(pos) = servers.iter().position(|s| s.name == name) {
                        servers[pos].url = val;
                    } else {
                        servers.push(McpServerEntry { name, url: val });
                    }
                }
            }
        }

        servers
    }

    /// 从 `MCP_STDIO__{NAME}` 环境变量加载 stdio MCP 服务器配置。
    /// 值必须为 JSON 格式：{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem"],"env":{}}
    fn load_mcp_stdio_servers() -> Vec<(String, McpStdioServerEntry)> {
        let mut servers = Vec::new();
        for (key, val) in std::env::vars() {
            if let Some(name) = key.strip_prefix("MCP_STDIO__") {
                if !name.is_empty() {
                    if let Ok(entry) = serde_json::from_str::<McpStdioServerEntry>(&val) {
                        servers.push((name.to_lowercase(), entry));
                    }
                }
            }
        }
        servers
    }

    pub fn clone_with_model(&self, model: String) -> Self {
        let gateway = GatewaySettings {
            default_model: model.clone(),
            model_mapping: HashMap::from([
                ("planning".to_string(), model.clone()),
                ("execution".to_string(), model.clone()),
                ("analysis".to_string(), model.clone()),
                ("default".to_string(), model.clone()),
            ]),
            ..self.gateway.clone()
        };

        Self {
            gateway,
            model,
            workspace: self.workspace.clone(),
            max_iterations: self.max_iterations,
            max_pdca_cycles: self.max_pdca_cycles,
            max_l1_mb: self.max_l1_mb,
            max_l2_mb: self.max_l2_mb,
            max_l3_mb: self.max_l3_mb,
            data_dir: self.data_dir.clone(),
            workflow_path: self.workflow_path.clone(),
            mcp_servers: self.mcp_servers.clone(),
            mcp_stdio_servers: self.mcp_stdio_servers.clone(),
        }
    }

    pub fn clone_with_api_key(&self, api_key: String) -> Self {
        let mut gateway = self.gateway.clone();
        gateway.api_key = api_key;
        Self {
            gateway,
            model: self.model.clone(),
            workspace: self.workspace.clone(),
            max_iterations: self.max_iterations,
            max_pdca_cycles: self.max_pdca_cycles,
            max_l1_mb: self.max_l1_mb,
            max_l2_mb: self.max_l2_mb,
            max_l3_mb: self.max_l3_mb,
            data_dir: self.data_dir.clone(),
            workflow_path: self.workflow_path.clone(),
            mcp_servers: self.mcp_servers.clone(),
            mcp_stdio_servers: self.mcp_stdio_servers.clone(),
        }
    }

    pub fn clone_with_api_url(&self, api_url: String) -> Self {
        let mut gateway = self.gateway.clone();
        gateway.base_url = api_url;
        Self {
            gateway,
            model: self.model.clone(),
            workspace: self.workspace.clone(),
            max_iterations: self.max_iterations,
            max_pdca_cycles: self.max_pdca_cycles,
            max_l1_mb: self.max_l1_mb,
            max_l2_mb: self.max_l2_mb,
            max_l3_mb: self.max_l3_mb,
            data_dir: self.data_dir.clone(),
            workflow_path: self.workflow_path.clone(),
            mcp_servers: self.mcp_servers.clone(),
            mcp_stdio_servers: self.mcp_stdio_servers.clone(),
        }
    }
}
