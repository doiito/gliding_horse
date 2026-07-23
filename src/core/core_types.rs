//! Core types - CoreError, CoreConfig, SemanticCore
//!
//! Ported from rust-core (pdca-core v2.0.0)

use std::sync::Arc;
use thiserror::Error;
use tracing::info;

use crate::core::event_bus::EventBus;
use crate::core::validation::ValidationEngine;
use crate::core::CheckpointManager;
use crate::memory::{l0_store, l2_blackboard, l3_projection, EvictionConfig};
use crate::tools::skill_registry::SkillRegistry;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Node size {size} exceeds maximum {max}")]
    NodeTooLarge { size: usize, max: usize },

    #[error("Projection size {size} exceeds maximum {max}")]
    ProjectionTooLarge { size: usize, max: usize },

    #[error("Invalid JSON-LD: {message}")]
    InvalidJsonLd { message: String },

    #[error("Node not found: {iri}")]
    NodeNotFound { iri: String },

    #[error("Task not found: {iri}")]
    TaskNotFound { iri: String },

    #[error("Skill not found: {iri}")]
    SkillNotFound { iri: String },

    #[error("Frame not found: {name}")]
    FrameNotFound { name: String },

    #[error("Validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("SPARQL query error: {message}")]
    SparqlError { message: String },

    #[error("Storage error: {message}")]
    StorageError { message: String },

    #[error("Oxigraph sync failed: {message}")]
    OxigraphSyncFailed { message: String },

    #[error("Internal error: {message}")]
    Internal { message: String },

    #[error("Permission denied: agent '{agent}' cannot {action} on '{resource}'")]
    PermissionDenied {
        agent: String,
        resource: String,
        action: String,
    },
}

impl From<oxigraph::store::StorageError> for CoreError {
    fn from(e: oxigraph::store::StorageError) -> Self {
        CoreError::StorageError {
            message: e.to_string(),
        }
    }
}

impl From<oxigraph::sparql::QueryEvaluationError> for CoreError {
    fn from(e: oxigraph::sparql::QueryEvaluationError) -> Self {
        CoreError::SparqlError {
            message: e.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoreConfig {
    pub max_node_size: usize,
    pub max_projection_size: usize,
    pub l0_storage_path: String,
    pub event_buffer_size: usize,
    pub enable_metrics: bool,
    /// Global L1 eviction policy config (None = auto-select by Agent Role)
    pub eviction_config: Option<EvictionConfig>,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            max_node_size: 5_242_880, // 5MB: agent full content can reach hundreds of KB
            max_projection_size: 500,
            l0_storage_path: "./data/l0".to_string(),
            event_buffer_size: 1000,
            enable_metrics: true,
            eviction_config: None,
        }
    }
}

impl CoreConfig {
    pub fn from_env() -> Self {
        Self {
            max_node_size: std::env::var("PDCA_MAX_NODE_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5_242_880),
            max_projection_size: std::env::var("PDCA_MAX_PROJECTION_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(500),
            l0_storage_path: std::env::var("PDCA_L0_STORAGE_PATH")
                .unwrap_or_else(|_| "./data/l0".to_string()),
            event_buffer_size: std::env::var("PDCA_EVENT_BUFFER_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000),
            enable_metrics: std::env::var("PDCA_ENABLE_METRICS")
                .ok()
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            eviction_config: None,
        }
    }
}

/// SemanticCore — unified entrypoint combining all subsystems.
///
/// Ported from rust-core (pdca-core v2.0.0).
/// Wraps Blackboard, L0Store, ProjectionEngine, SkillRegistry,
/// EventBus, ValidationEngine, and CheckpointManager.
pub struct SemanticCore {
    pub blackboard: Arc<l2_blackboard::Blackboard>,
    pub l0_store: Arc<l0_store::L0Store>,
    pub projection: Arc<l3_projection::ProjectionEngine>,
    pub skills: Arc<SkillRegistry>,
    pub events: Arc<EventBus>,
    pub validation: Arc<ValidationEngine>,
    pub checkpoints: Arc<CheckpointManager>,
    pub config: CoreConfig,
}

impl SemanticCore {
    pub fn new(config: CoreConfig) -> Result<Self, CoreError> {
        info!("Initializing SemanticCore");

        let blackboard = Arc::new(l2_blackboard::Blackboard::new()?);
        let l0_store = Arc::new(l0_store::L0Store::new(&config.l0_storage_path)?);
        let events = Arc::new(EventBus::new(config.event_buffer_size));
        let validation = Arc::new(ValidationEngine::new(config.max_node_size));
        let skills = Arc::new(SkillRegistry::new());
        let checkpoints = Arc::new(CheckpointManager::new());
        let projection = Arc::new(l3_projection::ProjectionEngine::new(
            blackboard.clone(),
            config.max_projection_size,
        ));

        Ok(Self {
            blackboard,
            l0_store,
            projection,
            skills,
            events,
            validation,
            checkpoints,
            config,
        })
    }

    pub async fn init_task(
        &self,
        user_input: &str,
        _agent_md_path: Option<&str>,
        parent_task_iri: Option<&str>,
    ) -> Result<String, CoreError> {
        let task_iri = format!("iri://task_{}", uuid::Uuid::new_v4().hyphenated());
        let task_node = serde_json::json!({
            "@id": &task_iri,
            "@type": "Task",
            "user_input": user_input,
            "status": "pending",
            "created_at": chrono::Utc::now().to_rfc3339(),
            "parent_task": parent_task_iri,
        });
        let json_ld = serde_json::to_string(&task_node).map_err(|e| CoreError::Internal {
            message: e.to_string(),
        })?;
        self.blackboard
            .write_node(&task_iri, &json_ld, &self.config)?;
        self.events
            .emit(&task_iri, "TASK_CREATED", "system", &json_ld)
            .await;
        Ok(task_iri)
    }

    pub async fn write_node(
        &self,
        task_iri: &str,
        json_ld: &str,
        node_type: Option<&str>,
        created_by: Option<&str>,
    ) -> Result<String, CoreError> {
        self.validation.validate_json_ld(json_ld)?;
        let node_iri = format!(
            "iri://{}/node_{}",
            task_iri.strip_prefix("iri://").unwrap_or(task_iri),
            uuid::Uuid::new_v4().hyphenated()
        );
        self.blackboard
            .write_node(&node_iri, json_ld, &self.config)?;
        self.projection.invalidate_cache_for_task(task_iri);
        self.events
            .emit(
                task_iri,
                "NODE_CREATED",
                created_by.unwrap_or("unknown"),
                &serde_json::json!({
                    "node_iri": &node_iri,
                    "node_type": node_type,
                })
                .to_string(),
            )
            .await;
        Ok(node_iri)
    }

    pub async fn read_node(
        &self,
        node_iri: &str,
    ) -> Result<Option<std::sync::Arc<l2_blackboard::Node>>, CoreError> {
        self.blackboard.read_node(node_iri)
    }

    pub async fn emit_event(
        &self,
        task_iri: &str,
        event_type: &str,
        source_agent_iri: &str,
        payload: &str,
    ) -> String {
        self.events
            .emit(task_iri, event_type, source_agent_iri, payload)
            .await
    }

    pub fn list_skills(&self, agent_role: &str) -> Vec<crate::tools::skill_registry::SkillMeta> {
        self.skills.list_skills_for_role(agent_role)
    }

    pub fn health_check(&self) -> String {
        "healthy".to_string()
    }
}
