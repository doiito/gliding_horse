pub mod api;
pub mod batch;
pub mod config;
pub mod core;
pub mod gateway;
pub mod jsonld;
pub mod knowledge_graph;
pub mod llm;
pub mod memory;
pub mod methodology;
pub mod perception;
pub mod permissions;
pub mod root_cause;
pub mod skill_graph;
pub mod templates;
pub mod tools;
pub mod utils;
pub mod worker;

pub mod causal;
/// Skill graph versioned snapshots & temporal hyperedges.
///
/// **Experimental** — API may change without notice.
/// Renamed from `temporal` to `snapshots` for clarity.
pub mod snapshots;

pub mod graph_backend;
/// Topological feature extraction & neighborhood aggregation.
///
/// **Experimental** — API may change without notice.
/// Renamed from `gnn` to `graph_features` for clarity.
pub mod graph_features;

#[cfg(feature = "ontology")]
pub mod ontology;

/// Bridge types for ontology embedding storage (OntologyEmbedStore, HyperspaceEmbedStore,
/// OntologySearchBridge, OntologyBridgeManager, StructuralEmbeddingService).
/// This module was moved out of crates/hyperspace-engine/src/open_ontologies.rs
/// because it bridges two independent subsystems; it belongs at the application level.
#[cfg(feature = "ontology")]
pub mod ontology_bridge;

#[cfg(feature = "ontology")]
pub use ontology_bridge::{
    CrossSpaceProjection, FallbackStructuralEmbeddingService, HyperspaceEmbedStore,
    LinearCrossSpaceProjection, LinearCrossSpaceProjectionConfig, OntologyBridgeConfig,
    OntologyBridgeManager, OntologyEmbedStore, OntologySearchBridge, StructuralEmbeddingService,
    StructuralFeatures,
};

pub use config::Settings;
pub use core::{
    agent_instance::{AgentRole, AgentStatus},
    agent_runner::{TaskContext, TaskResult},
    sa::{CyclePhase, CycleState, ExecutionPlan, PlanStep, TaskComplexity},
    AgentInstance, AgentRunner, CoreConfig, CoreError, SupervisorAgent,
};
pub use gateway::UnifiedGateway;
pub use jsonld::JsonLdContext;
pub use memory::{Blackboard, L0Store, L1Session, ProjectionEngine};
pub use tools::SkillRegistry;
