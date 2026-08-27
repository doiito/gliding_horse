use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::*;
use crate::tools::mcp_client::McpClient;
use crate::CoreError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolInfo {
    pub tool_name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub output_schema: Option<serde_json::Value>,
    pub server_id: String,
    pub discovered_at: DateTime<Utc>,
}

impl MCPToolInfo {
    pub fn new(server_id: &str, tool_name: &str, description: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            description: description.to_string(),
            input_schema: serde_json::json!({}),
            output_schema: None,
            server_id: server_id.to_string(),
            discovered_at: Utc::now(),
        }
    }

    pub fn with_input_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = schema;
        self
    }

    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolSyncResult {
    pub server_id: String,
    pub tools_added: u32,
    pub tools_updated: u32,
    pub tools_removed: u32,
    pub synced_at: DateTime<Utc>,
    pub errors: Vec<String>,
}

impl MCPToolSyncResult {
    pub fn new(server_id: &str) -> Self {
        Self {
            server_id: server_id.to_string(),
            tools_added: 0,
            tools_updated: 0,
            tools_removed: 0,
            synced_at: Utc::now(),
            errors: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: &str) {
        self.errors.push(error.to_string());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    pub server_id: String,
    pub server_name: String,
    pub endpoint: Option<String>,
    pub auth_type: MCPAuthType,
    pub enabled: bool,
    pub trust_level: TrustLevel,
    pub auto_sync: bool,
    pub sync_interval_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MCPAuthType {
    None,
    BearerToken,
    OAuth2,
    ApiKey,
}

impl Default for MCPServerConfig {
    fn default() -> Self {
        Self {
            server_id: String::new(),
            server_name: String::new(),
            endpoint: None,
            auth_type: MCPAuthType::None,
            enabled: true,
            trust_level: TrustLevel::Medium,
            auto_sync: true,
            sync_interval_seconds: 300,
        }
    }
}

pub struct MCPRegistry {
    servers: RwLock<HashMap<String, MCPServerConfig>>,
    tools: RwLock<HashMap<String, MCPToolInfo>>,
    mappings: RwLock<HashMap<String, MCPSkillMapping>>,
}

impl MCPRegistry {
    pub fn new() -> Self {
        Self {
            servers: RwLock::new(HashMap::new()),
            tools: RwLock::new(HashMap::new()),
            mappings: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register_server(&self, config: MCPServerConfig) -> Result<(), CoreError> {
        let server_id = config.server_id.clone();
        info!(
            "Registering MCP server: {} ({})",
            config.server_name, server_id
        );

        let mut servers = self.servers.write().await;
        servers.insert(server_id, config);
        Ok(())
    }

    pub async fn unregister_server(&self, server_id: &str) -> Result<bool, CoreError> {
        let mut servers = self.servers.write().await;
        let removed = servers.remove(server_id).is_some();

        if removed {
            let mut tools = self.tools.write().await;
            tools.retain(|_, tool| tool.server_id != server_id);

            let mut mappings = self.mappings.write().await;
            mappings.retain(|_, mapping| mapping.mcp_server_id != server_id);
        }

        Ok(removed)
    }

    pub async fn get_server(&self, server_id: &str) -> Option<MCPServerConfig> {
        let servers = self.servers.read().await;
        servers.get(server_id).cloned()
    }

    pub async fn list_servers(&self) -> Vec<MCPServerConfig> {
        let servers = self.servers.read().await;
        servers.values().cloned().collect()
    }

    pub async fn register_tool(&self, tool: MCPToolInfo) -> Result<(), CoreError> {
        let tool_key = format!("{}:{}", tool.server_id, tool.tool_name);
        debug!("Registering MCP tool: {}", tool_key);

        let mut tools = self.tools.write().await;
        tools.insert(tool_key, tool);
        Ok(())
    }

    pub async fn get_tool(&self, server_id: &str, tool_name: &str) -> Option<MCPToolInfo> {
        let tool_key = format!("{}:{}", server_id, tool_name);
        let tools = self.tools.read().await;
        tools.get(&tool_key).cloned()
    }

    pub async fn list_tools(&self, server_id: Option<&str>) -> Vec<MCPToolInfo> {
        let tools = self.tools.read().await;
        match server_id {
            Some(sid) => tools
                .values()
                .filter(|t| t.server_id == sid)
                .cloned()
                .collect(),
            None => tools.values().cloned().collect(),
        }
    }

    pub async fn create_mapping(&self, mapping: MCPSkillMapping) -> Result<(), CoreError> {
        info!(
            "Creating MCP skill mapping: {} -> {}:{}",
            mapping.skill_iri, mapping.mcp_server_id, mapping.mcp_tool_name
        );

        let mut mappings = self.mappings.write().await;
        mappings.insert(mapping.skill_iri.clone(), mapping);
        Ok(())
    }

    pub async fn get_mapping(&self, skill_iri: &str) -> Option<MCPSkillMapping> {
        let mappings = self.mappings.read().await;
        mappings.get(skill_iri).cloned()
    }

    pub async fn remove_mapping(&self, skill_iri: &str) -> bool {
        let mut mappings = self.mappings.write().await;
        mappings.remove(skill_iri).is_some()
    }

    pub async fn list_mappings(&self) -> Vec<MCPSkillMapping> {
        let mappings = self.mappings.read().await;
        mappings.values().cloned().collect()
    }
}

impl Default for MCPRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MCPIntegration {
    registry: Arc<MCPRegistry>,
    graph_store: Arc<SkillGraphStore>,
    client: Option<Arc<tokio::sync::Mutex<Option<McpClient>>>>,
}

impl MCPIntegration {
    pub fn new(registry: Arc<MCPRegistry>, graph_store: Arc<SkillGraphStore>) -> Self {
        Self {
            registry,
            graph_store,
            client: None,
        }
    }

    /// Attach the shared MCP client handle so invoke_tool dispatches real tool calls.
    pub fn with_client(mut self, client: Arc<tokio::sync::Mutex<Option<McpClient>>>) -> Self {
        self.client = Some(client);
        self
    }

    pub async fn sync_tools_to_skills(
        &self,
        server_id: &str,
    ) -> Result<MCPToolSyncResult, CoreError> {
        let mut result = MCPToolSyncResult::new(server_id);

        let server = self.registry.get_server(server_id).await;
        if server.is_none() {
            return Err(CoreError::ValidationFailed {
                message: format!("MCP server not found: {}", server_id),
            });
        }
        let server = server.expect("server validated non-none above");

        let tools = self.registry.list_tools(Some(server_id)).await;

        for tool in tools {
            match self.create_skill_from_tool(&tool, &server).await {
                Ok(true) => result.tools_added += 1,
                Ok(false) => result.tools_updated += 1,
                Err(e) => {
                    warn!("Failed to create skill: {} - {}", tool.tool_name, e);
                    result.add_error(&format!("{}: {}", tool.tool_name, e));
                }
            }
        }

        info!(
            "MCP tool sync complete: {} (added={}, updated={}, errors={})",
            server_id,
            result.tools_added,
            result.tools_updated,
            result.errors.len()
        );

        Ok(result)
    }

    async fn create_skill_from_tool(
        &self,
        tool: &MCPToolInfo,
        server: &MCPServerConfig,
    ) -> Result<bool, CoreError> {
        let skill_iri = format!("iri://skills/mcp/{}-{}", tool.server_id, tool.tool_name);

        let existing = self.graph_store.get_skill(&skill_iri);
        let is_new = existing.is_none();

        let w2h = Skill5W2H {
            what: tool.tool_name.clone(),
            why: tool.description.clone(),
            who: SkillRole {
                role_name: "MCP Tool".to_string(),
                required_agent_role: Some("DA".to_string()),
            },
            when: SkillTrigger {
                applicable_phases: vec!["Do".to_string()],
                trigger_condition: None,
                deadline_constraint: None,
            },
            where_: SkillContext {
                target_stack: vec![],
                repo_pattern: None,
            },
            how: SkillApproach {
                approach: format!("MCP tool call to {}", tool.tool_name),
                plan_iri: None,
            },
            how_much: SkillCost {
                avg_token_cost: 500,
                avg_duration_seconds: 5,
                max_sub_agents: 0,
            },
        };

        let security_info =
            SkillSecurityInfo::new(SkillSource::MCPExternal).with_trust_level(server.trust_level);

        let skill = SkillGraphNode {
            skill_iri: skill_iri.clone(),
            name: tool.tool_name.clone(),
            description: tool.description.clone(),
            version: "1.0.0".to_string(),
            node_type: SkillNodeType::MCPTool,
            maturity: "stable".to_string(),
            tags: vec!["mcp".to_string(), format!("server:{}", tool.server_id)],
            w2h,
            links: Vec::new(),
            graph_meta: SkillGraphMeta::new(),
            content: Some(SkillContent {
                summary: tool.description.clone(),
                steps: vec![SkillStep::new(
                    "call-mcp",
                    1,
                    &format!("Call MCP tool {}", tool.tool_name),
                )],
                validation: Some(SkillValidation {
                    method: "MCP tool execution".to_string(),
                    success_condition: "Tool returns success".to_string(),
                }),
            }),
            attached_to: None,
            security_info: Some(security_info),
            mcp_server_id: Some(tool.server_id.clone()),
            storage_tier: StorageTier::L2Blackboard,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_used_at: None,
        };

        self.graph_store.register_skill(skill)?;

        let mapping = MCPSkillMapping::new(&skill_iri, &tool.server_id, &tool.tool_name);
        self.registry.create_mapping(mapping).await?;

        Ok(is_new)
    }

    pub async fn invoke_tool(
        &self,
        skill_iri: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, CoreError> {
        let mapping = self.registry.get_mapping(skill_iri).await.ok_or_else(|| {
            CoreError::ValidationFailed {
                message: format!("No MCP mapping for skill: {}", skill_iri),
            }
        })?;

        if !mapping.enabled {
            return Err(CoreError::ValidationFailed {
                message: format!("MCP mapping disabled for skill: {}", skill_iri),
            });
        }

        let mcp_params = self.transform_params(&mapping, params)?;

        debug!(
            "Invoking MCP tool: {}:{} with params {:?}",
            mapping.mcp_server_id, mapping.mcp_tool_name, mcp_params
        );

        let handle = self
            .client
            .as_ref()
            .ok_or_else(|| CoreError::ValidationFailed {
                message: format!("No MCP client attached for skill: {}", skill_iri),
            })?;
        let mut guard = handle.lock().await;
        let client = guard.as_mut().ok_or_else(|| CoreError::ValidationFailed {
            message: format!("MCP client not connected for skill: {}", skill_iri),
        })?;

        let result = client
            .call_tool(&mapping.mcp_server_id, &mapping.mcp_tool_name, &mcp_params)
            .await?;

        let transformed_result = self.transform_result(&mapping, result)?;

        Ok(transformed_result)
    }

    fn transform_params(
        &self,
        mapping: &MCPSkillMapping,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, CoreError> {
        if mapping.parameter_mapping.is_empty() {
            return Ok(params);
        }

        let params_obj = params
            .as_object()
            .ok_or_else(|| CoreError::ValidationFailed {
                message: "Params must be a JSON object".to_string(),
            })?;

        let mut transformed = serde_json::Map::new();
        for (key, value) in params_obj {
            let mcp_key = mapping.parameter_mapping.get(key).unwrap_or(key);
            transformed.insert(mcp_key.clone(), value.clone());
        }

        Ok(serde_json::Value::Object(transformed))
    }

    fn transform_result(
        &self,
        mapping: &MCPSkillMapping,
        result: serde_json::Value,
    ) -> Result<serde_json::Value, CoreError> {
        if mapping.result_mapping.is_empty() {
            return Ok(result);
        }

        let result_obj = result
            .as_object()
            .ok_or_else(|| CoreError::ValidationFailed {
                message: "Result must be a JSON object".to_string(),
            })?;

        let mut transformed = serde_json::Map::new();
        for (key, value) in result_obj {
            let skill_key = mapping
                .result_mapping
                .iter()
                .find(|(_, v)| v == &key)
                .map(|(k, _)| k.as_str())
                .unwrap_or(&key);
            transformed.insert(skill_key.to_string(), value.clone());
        }

        Ok(serde_json::Value::Object(transformed))
    }

    pub async fn check_tool_availability(&self, skill_iri: &str) -> Result<bool, CoreError> {
        let mapping = self.registry.get_mapping(skill_iri).await;

        if let Some(mapping) = mapping {
            let server = self.registry.get_server(&mapping.mcp_server_id).await;
            Ok(server.map(|s| s.enabled).unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    pub async fn get_tool_schema(&self, skill_iri: &str) -> Option<serde_json::Value> {
        let mapping = self.registry.get_mapping(skill_iri).await?;
        let tool = self
            .registry
            .get_tool(&mapping.mcp_server_id, &mapping.mcp_tool_name)
            .await?;

        Some(serde_json::json!({
            "input": tool.input_schema,
            "output": tool.output_schema,
            "mapping": {
                "parameters": mapping.parameter_mapping,
                "results": mapping.result_mapping
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_registry_server() {
        let registry = MCPRegistry::new();

        let config = MCPServerConfig {
            server_id: "test-server".to_string(),
            server_name: "Test Server".to_string(),
            endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        registry.register_server(config.clone()).await.unwrap();

        let retrieved = registry.get_server("test-server").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().server_name, "Test Server");

        let removed = registry.unregister_server("test-server").await.unwrap();
        assert!(removed);

        let retrieved = registry.get_server("test-server").await;
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_mcp_registry_tool() {
        let registry = MCPRegistry::new();

        let tool = MCPToolInfo::new("server-1", "read_file", "Read a file from filesystem")
            .with_input_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                }
            }));

        registry.register_tool(tool).await.unwrap();

        let retrieved = registry.get_tool("server-1", "read_file").await;
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().description,
            "Read a file from filesystem"
        );

        let tools = registry.list_tools(Some("server-1")).await;
        assert_eq!(tools.len(), 1);
    }

    #[tokio::test]
    async fn test_mcp_registry_mapping() {
        let registry = MCPRegistry::new();

        let mapping =
            MCPSkillMapping::new("iri://skills/file-read", "filesystem-server", "read_file")
                .with_param_mapping("file_path", "path")
                .with_result_mapping("content", "data");

        registry.create_mapping(mapping).await.unwrap();

        let retrieved = registry.get_mapping("iri://skills/file-read").await;
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().enabled);

        let removed = registry.remove_mapping("iri://skills/file-read").await;
        assert!(removed);
    }

    #[test]
    fn test_mcp_tool_info() {
        let tool = MCPToolInfo::new("server-1", "test_tool", "A test tool")
            .with_input_schema(serde_json::json!({"type": "object"}))
            .with_output_schema(serde_json::json!({"type": "string"}));

        assert_eq!(tool.tool_name, "test_tool");
        assert_eq!(tool.server_id, "server-1");
        assert!(tool.input_schema.is_object());
        assert!(tool.output_schema.is_some());
    }

    #[test]
    fn test_mcp_tool_sync_result() {
        let mut result = MCPToolSyncResult::new("test-server");
        result.tools_added = 5;
        result.tools_updated = 3;
        result.add_error("Test error");

        assert_eq!(result.tools_added, 5);
        assert_eq!(result.tools_updated, 3);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_mcp_server_config_default() {
        let config = MCPServerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.trust_level, TrustLevel::Medium);
        assert!(config.auto_sync);
    }
}
