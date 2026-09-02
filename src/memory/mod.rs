pub mod consistency_engine;
pub mod context_recall;
pub mod embedding_service;
pub mod hyperspace_store;
pub mod l0_store;
pub mod l1_session;
pub mod l2_blackboard;
pub mod l3_projection;
pub mod memory_bus;
pub mod memory_manager;
pub mod prefetch_engine;
pub mod scheduler;
pub mod unified_graph;

pub use consistency_engine::{ConsistencyEngine, WriteStrategy};
pub use context_recall::{ContextRecallQuery, CONTEXT_RECALL_QUERY_VERSION};
pub use embedding_service::{
    create_embedding_service_from_config, EmbeddingService, FallbackEmbeddingService,
    OneApiEmbeddingService,
};
pub use hyperspace_store::{
    AnnHealthProbeConfig, AnnHealthReport, AnnMaintenanceRecommendation, HybridSearchFilter,
    HyperspaceStore, ScoredEntry, ANN_HEALTH_REPORT_SCHEMA_VERSION,
};
pub use l0_store::{
    L0RecordKind, L0Store, MesiState, RetentionClass, TaskEvidenceAppendOutcome,
    TaskEvidenceFrameRecord, TaskEvidenceHead, TaskEvidenceSealOutcome, TaskEvidenceSealRecord,
};
pub use l1_session::{cosine_similarity, EvictionConfig, L1Session, L1Turn};
pub use l2_blackboard::{
    AgentActivity, AgentStatus, Blackboard, DurabilityClass, LockType, ResourceLock, TaskTreeNode,
};
pub use l3_projection::ProjectionEngine;
pub use memory_bus::MemoryBus;
pub use memory_manager::MemoryManager;
pub use prefetch_engine::PrefetchEngine;
pub use scheduler::MemoryScheduler;
