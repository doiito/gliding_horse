use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tracing::{debug, info};

use crate::memory::hyperspace_store::HyperspaceStore;
use crate::memory::l0_store::L0Store;
use crate::memory::l1_session::{L1Session, SessionSummary};
use crate::memory::l2_blackboard::Blackboard;
use crate::memory::l3_projection::ProjectionEngine;
use crate::memory::scheduler::MemoryScheduler;
use crate::core::tracked_action::TrackedAction;
#[cfg(feature = "ontology")]
use crate::ontology_bridge::OntologyBridgeManager;
use crate::{CoreConfig, CoreError};

/// Coordinates all four memory layers (L0/L1/L2/L3)
///
/// Memory lifecycle:
/// L1 Session → (compress) → L2 Blackboard → (archive) → L0 persistence
///                                                      → L3 projection (on demand)
pub struct MemoryManager {
    l0: Arc<L0Store>,
    l2: Arc<Blackboard>,
    projection: Arc<ProjectionEngine>,
    config: CoreConfig,
    sessions: HashMap<String, L1Session>,
    scheduler: Option<Arc<MemoryScheduler>>,
    l1_active_count: AtomicU64,
    /// HyperspaceEngine-backed vector store for semantic search.
    /// Available to all memory layers for embedding-based retrieval.
    vector_store: Option<Arc<HyperspaceStore>>,
    /// OntologyBridge dual-space embedding store (text Cosine + struct Poincaré).
    #[cfg(feature = "ontology")]
    ontology_bridge: Option<Arc<OntologyBridgeManager>>,
}

impl MemoryManager {
    pub fn new(
        l0: Arc<L0Store>,
        l2: Arc<Blackboard>,
        projection: Arc<ProjectionEngine>,
        config: CoreConfig,
    ) -> Self {
        Self::with_vector_store(l0, l2, projection, config, None)
    }

    /// Construct MemoryManager with an optional vector store.
    pub fn with_vector_store(
        l0: Arc<L0Store>,
        l2: Arc<Blackboard>,
        projection: Arc<ProjectionEngine>,
        config: CoreConfig,
        vector_store: Option<Arc<HyperspaceStore>>,
    ) -> Self {
        info!("MemoryManager initialized");
        Self {
            l0,
            l2,
            projection,
            config,
            sessions: HashMap::new(),
            scheduler: None,
            l1_active_count: AtomicU64::new(0),
            vector_store,
            #[cfg(feature = "ontology")]
            ontology_bridge: None,
        }
    }

    /// Construct MemoryManager with a MemoryScheduler
    ///
    /// When scheduler exists, session changes are synced to the scheduler,
    /// enabling it to perform context requests, overflow handling, etc.
    pub fn with_scheduler(
        l0: Arc<L0Store>,
        l2: Arc<Blackboard>,
        projection: Arc<ProjectionEngine>,
        config: CoreConfig,
        scheduler: Arc<MemoryScheduler>,
    ) -> Self {
        Self::with_scheduler_and_vector_store(l0, l2, projection, config, scheduler, None)
    }

    pub fn with_scheduler_and_vector_store(
        l0: Arc<L0Store>,
        l2: Arc<Blackboard>,
        projection: Arc<ProjectionEngine>,
        config: CoreConfig,
        scheduler: Arc<MemoryScheduler>,
        vector_store: Option<Arc<HyperspaceStore>>,
    ) -> Self {
        info!("MemoryManager initialized (with scheduler)");
        Self {
            l0,
            l2,
            projection,
            config,
            sessions: HashMap::new(),
            scheduler: Some(scheduler),
            l1_active_count: AtomicU64::new(0),
            vector_store,
            #[cfg(feature = "ontology")]
            ontology_bridge: None,
        }
    }

    /// Set scheduler at runtime (for delayed injection scenarios)
    pub fn set_scheduler(&mut self, scheduler: Arc<MemoryScheduler>) {
        self.scheduler = Some(scheduler);
    }

    /// Get scheduler reference
    pub fn scheduler(&self) -> Option<&Arc<MemoryScheduler>> {
        self.scheduler.as_ref()
    }

    /// Get L3 ProjectionEngine reference
    pub fn projection(&self) -> &Arc<ProjectionEngine> {
        &self.projection
    }

    /// Get HyperspaceEngine vector store reference (if configured)
    pub fn vector_store(&self) -> Option<&Arc<HyperspaceStore>> {
        self.vector_store.as_ref()
    }

    /// Attach an OntologyBridgeManager at runtime.
    #[cfg(feature = "ontology")]
    pub fn set_ontology_bridge(&mut self, ob: Arc<OntologyBridgeManager>) {
        self.ontology_bridge = Some(ob);
    }

    /// Get OntologyBridgeManager reference (if configured).
    #[cfg(feature = "ontology")]
    pub fn ontology_bridge(&self) -> Option<&Arc<OntologyBridgeManager>> {
        self.ontology_bridge.as_ref()
    }

    // ========== L1 Session Management ==========

    /// Create new L1 session
    pub fn create_session(&mut self, agent_id: &str, agent_role: &str, task_iri: &str) -> L1Session {
        let session = match self.config.eviction_config {
            Some(cfg) => L1Session::with_config(agent_id, agent_role, task_iri, 2000, cfg),
            None => L1Session::new(agent_id, agent_role, task_iri),
        };
        self.l1_active_count.fetch_add(1, Ordering::Relaxed);
        debug!(
            session_id = %session.session_id(),
            agent_id = %agent_id,
            "L1 session created"
        );
        session
    }

    /// Register session with manager, returns session_id
    ///
    /// When scheduler exists, also syncs to scheduler for its high-level operations.
    pub fn track_session(&mut self, session: L1Session) -> String {
        let id = session.session_id().to_string();
        if let Some(ref scheduler) = self.scheduler {
            scheduler.insert_session(session);
        } else {
            self.sessions.insert(id.clone(), session);
        }
        self.l1_active_count.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Get immutable session reference by ID
    pub fn get_session(&self, session_id: &str) -> Option<&L1Session> {
        self.sessions.get(session_id)
    }

    /// Get mutable session reference by ID
    pub fn get_session_mut(&mut self, session_id: &str) -> Option<&mut L1Session> {
        self.sessions.get_mut(session_id)
    }

    /// Compress and close session, returns session summary
    pub fn close_session(&mut self, session_id: &str) -> Result<SessionSummary, CoreError> {
        let result = if let Some(ref scheduler) = self.scheduler {
            let session = scheduler.remove_session(session_id).ok_or_else(|| CoreError::Internal {
                message: format!("Session not found: {}", session_id),
            })?;
            let summary = session.summarize();
            info!(
                session_id = %session_id,
                turn_count = summary.turn_count,
                "L1 session closed (via scheduler)"
            );
            Ok(summary)
        } else {
            let session = self.sessions.remove(session_id).ok_or_else(|| CoreError::Internal {
                message: format!("Session not found: {}", session_id),
            })?;
            let summary = session.summarize();
            info!(
                session_id = %session_id,
                turn_count = summary.turn_count,
                "L1 session closed"
            );
            Ok(summary)
        };
        if result.is_ok() {
            self.l1_active_count.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    /// Current number of active sessions
    pub fn session_count(&self) -> usize {
        if let Some(ref scheduler) = self.scheduler {
            scheduler.session_count()
        } else {
            self.sessions.len()
        }
    }

    /// Lock-free active session count (maintained via atomic counter)
    pub fn l1_session_count(&self) -> u64 {
        self.l1_active_count.load(Ordering::Relaxed)
    }

    // ========== L2/L0 Archival ==========

    /// Archive session summary to L2 blackboard
    pub fn archive_to_l2(&self, task_iri: &str, summary: &SessionSummary) -> Result<(), CoreError> {
        // Use task-prefixed IRI so extract_task_iri maps this node to the correct task
        let node_iri = format!("{}/session/{}", task_iri, summary.session_id);
        let json_ld = serde_json::json!({
            "@context": "https://pdca-agent.org/context/memory",
            "@id": &node_iri,
            "@type": "SessionSummary",
            "session_id": summary.session_id,
            "agent_id": summary.agent_id,
            "agent_role": summary.agent_role,
            "task_iri": summary.task_iri,
            "turn_count": summary.turn_count,
            "summary_text": summary.summary_text,
        })
        .to_string();

        self.l2.write_node(&node_iri, &json_ld, &self.config)
    }

    pub fn archive_session_actions(&self, task_iri: &str, actions: &[TrackedAction], summary: &str) -> Result<(), CoreError> {
        if actions.is_empty() { return Ok(()); }
        let task_id = format!("iri://task/{}", task_iri);
        let mut produces = vec![];
        for a in actions {
            for fc in &a.files_created {
                produces.push(serde_json::json!({
                    "@id": format!("iri://file/{}", fc.path.replace('/', "_")),
                    "@type": "https://agentos.ontology/core/File",
                    "https://agentos.ontology/core/filePath": fc.path,
                }));
            }
        }
        let json_ld = serde_json::json!({
            "@context": {"aos": "https://agentos.ontology/core/"},
            "@id": task_id,
            "@type": "aos:Task",
            "aos:hasStatus": "completed",
            "aos:produces": produces,
            "aos:summary": summary,
            "aos:actionCount": actions.len(),
        }).to_string();
        self.l2.write_node(&task_id, &json_ld, &self.config)
    }

    pub fn archive_experience(&self, task_iri: &str, agent_role: &str, summary: &str, success_rate: f32) -> Result<(), CoreError> {
        let exp = serde_json::json!({
            "experience_id": format!("exp_{}", uuid::Uuid::new_v4().hyphenated()),
            "scenario": summary,
            "pattern": if success_rate < 0.5 { "had_failures" } else { "all_success" },
            "success_rating": success_rate,
            "tags": ["experience", agent_role],
            "task_iri": task_iri,
            "created_at": chrono::Utc::now().to_rfc3339(),
        }).to_string();
        let iri = format!("iri://experience/{}", uuid::Uuid::new_v4().hyphenated());
        self.l0.store(&iri, &exp)
    }

    /// Archive summary to L0 permanent storage
    pub fn archive_to_l0(&self, summary: &SessionSummary) -> Result<(), CoreError> {
        let iri = format!("iri://archive/session/{}", summary.session_id);
        let content = serde_json::json!({
            "session_id": summary.session_id,
            "agent_id": summary.agent_id,
            "agent_role": summary.agent_role,
            "task_iri": summary.task_iri,
            "turn_count": summary.turn_count,
            "summary_text": summary.summary_text,
        })
        .to_string();

        self.l0.store(&iri, &content)
    }

    // ========== L3 Projection ==========

    /// Get projection for the specified agent role (sync wrapper, async internally)
    pub fn get_projection(
        &self,
        task_iri: &str,
        frame_name: &str,
    ) -> Result<Option<String>, CoreError> {
        let params = HashMap::new();
        let handle = tokio::runtime::Handle::try_current();

        match handle {
            Ok(_h) => {
                let frame = self.projection.get_frame(frame_name);
                let actual_frame = if frame.is_some() { frame_name } else { "reference_only" };
                let proj = self.projection.clone();
                let task_iri = task_iri.to_string();
                let actual_frame = actual_frame.to_string();
                
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        proj.project(&task_iri, &actual_frame, params).await
                    })
                })?;
                Ok(Some(result))
            }
            Err(_) => {
                let frames: Vec<String> = self.projection.list_frames().iter().map(|f| f.name.clone()).collect();
                let result = serde_json::json!({
                    "@context": "https://pdca-agent.org/context/projection",
                    "note": "Async runtime not available, returning frame list",
                    "available_frames": frames,
                }).to_string();
                Ok(Some(result))
            }
        }
    }

    // ========== Unified Storage Interface ==========

    /// Unified storage interface: store data by layer
    pub fn store(&self, agent_id: &str, key: &str, value: &str, layer: &str) -> Result<String, CoreError> {
        match layer {
            "L0" | "l0" => {
                let iri = format!("iri://{}/{}", agent_id, key);
                self.l0.store(&iri, value)?;
                Ok(iri)
            }
            "L1" | "l1" => {
                Err(CoreError::Internal {
                    message: "L1 layer does not support direct key-value storage; use session APIs instead".to_string(),
                })
            }
            "L2" | "l2" => {
                let iri = format!("iri://{}/{}", agent_id, key);
                self.l2.write_node(&iri, value, &self.config)?;
                Ok(iri)
            }
            _ => Err(CoreError::Internal {
                message: format!("Unsupported storage layer: {}", layer),
            }),
        }
    }

    /// Unified retrieval interface: retrieve data from specified layer
    pub fn retrieve(&self, key: &str, layers: &[&str]) -> Result<Option<String>, CoreError> {
        for layer in layers {
            match *layer {
                "L0" | "l0" => {
                    if let Some(entry) = self.l0.retrieve(key)? {
                        return Ok(Some(entry.content));
                    }
                }
                "L2" | "l2" => {
                    if let Some(node) = self.l2.read_node(key)? {
                        return Ok(Some(node.json_ld.clone()));
                    }
                }
                _ => {}
            }
        }
        Ok(None)
    }

    /// Archive L1 session to L0
    ///
    /// If scheduler exists, executes archival through scheduler's `on_session_close`,
    /// ensuring consistency engine invalidation propagation and projection cache cleanup.
    pub fn archive_session(&self, session_id: &str) -> Result<(), CoreError> {
        if let Some(ref scheduler) = self.scheduler {
            let session = scheduler.remove_session(session_id).ok_or_else(|| CoreError::Internal {
                message: format!("Session not found: {}", session_id),
            })?;
            let summary = session.summarize();
            self.archive_to_l0(&summary)?;
            self.archive_to_l2(&summary.task_iri, &summary)?;
            Ok(())
        } else {
            let session = self.sessions.get(session_id)
                .ok_or_else(|| CoreError::Internal {
                    message: format!("Session not found: {}", session_id),
                })?;
            let summary = session.summarize();
            self.archive_to_l0(&summary)?;
            self.archive_to_l2(&summary.task_iri, &summary)?;
            Ok(())
        }
    }

    /// Finalize and archive an externally held L1Session (skips track_session/close_session flow)
    ///
    /// Suitable for callers like AgentRunner that directly own the session.
    /// Automates: track → close → archive_to_l2 → archive_to_l0
    pub fn finalize_session(&mut self, session: L1Session, task_iri: &str) -> Result<(), CoreError> {
        let session_id = session.session_id().to_string();
        self.track_session(session);
        let summary = self.close_session(&session_id)?;
        self.archive_to_l2(task_iri, &summary)?;
        self.archive_to_l0(&summary)?;
        info!(
            session_id = %session_id,
            task_iri = %task_iri,
            "Session finalized and archived"
        );
        Ok(())
    }

    /// Sync cross-layer data
    pub fn sync_layers(&self, iri: &str) -> Result<(), CoreError> {
        if let Some(entry) = self.l0.retrieve(iri)? {
            self.l2.write_node(iri, &entry.content, &self.config)?;
        }
        Ok(())
    }

    // ========== Memory Statistics ==========

    /// Get memory system statistics
    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "l0_entries": self.l0.count().unwrap_or(0),
            "l2_nodes": self.l2.node_count(),
            "l2_bytes": self.l2.total_bytes(),
            "active_sessions": self.session_count(),
        })
    }

    // ========== Agent Situational Awareness Delegation ==========

    /// Register Agent to battle map
    pub fn register_agent(&self, agent_id: &str, role: &str, task_iri: &str) {
        self.l2.register_agent(agent_id, role, task_iri);
    }

    /// Update Agent heartbeat
    pub fn update_agent_heartbeat(&self, agent_id: &str) {
        self.l2.update_agent_heartbeat(agent_id);
    }

    /// Update Agent status
    pub fn update_agent_status(&self, agent_id: &str, status: crate::memory::AgentActivity, operation: Option<&str>) {
        self.l2.update_agent_status(agent_id, status, operation);
    }

    /// Get Agent status
    pub fn get_agent_status(&self, agent_id: &str) -> Option<crate::memory::AgentStatus> {
        self.l2.get_agent_status(agent_id)
    }

    /// List active agents
    pub fn list_active_agents(&self) -> Vec<crate::memory::AgentStatus> {
        self.l2.list_active_agents()
    }

    /// Unregister Agent
    pub fn unregister_agent(&self, agent_id: &str) {
        self.l2.unregister_agent(agent_id);
    }

    /// Detect heartbeat-timeout agents
    pub fn detect_stale_agents(&self, max_idle_seconds: i64) -> Vec<String> {
        self.l2.detect_stale_agents(max_idle_seconds)
    }

    /// Get Blackboard reference
    pub fn blackboard(&self) -> &Arc<Blackboard> {
        &self.l2
    }
}
