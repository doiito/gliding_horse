use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tracing::{debug, info};

use crate::core::tracked_action::TrackedAction;
use crate::memory::hyperspace_store::HyperspaceStore;
use crate::memory::l0_store::L0Store;
use crate::memory::l1_session::{EvictionConfig, L1Session, SessionSummary};
use crate::memory::l2_blackboard::Blackboard;
use crate::memory::l3_projection::ProjectionEngine;
use crate::memory::scheduler::MemoryScheduler;
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
    active_session_ids: std::collections::HashSet<String>,
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
            active_session_ids: std::collections::HashSet::new(),
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
            active_session_ids: std::collections::HashSet::new(),
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
    pub fn create_session(
        &mut self,
        agent_id: &str,
        agent_role: &str,
        task_iri: &str,
    ) -> L1Session {
        let budget = self.config.l1_token_budget.max(1);
        let mut eviction_config = self
            .config
            .eviction_config
            .unwrap_or_else(|| EvictionConfig::for_role(agent_role));
        if let Some(value) = self.config.l1_max_low_relevance_refs {
            eviction_config.max_low_relevance_refs = value;
        }
        if let Some(value) = self.config.l1_reload_preview_chars {
            eviction_config.reload_preview_chars = value;
        }
        let session =
            L1Session::with_config(agent_id, agent_role, task_iri, budget, eviction_config);
        if self
            .active_session_ids
            .insert(session.session_id().to_string())
        {
            self.l1_active_count.fetch_add(1, Ordering::Relaxed);
        }
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
        if self.active_session_ids.insert(id.clone()) {
            self.l1_active_count.fetch_add(1, Ordering::Relaxed);
        }
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
            let session =
                scheduler
                    .remove_session(session_id)
                    .ok_or_else(|| CoreError::Internal {
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
            let session = self
                .sessions
                .remove(session_id)
                .ok_or_else(|| CoreError::Internal {
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
        if result.is_ok() && self.active_session_ids.remove(session_id) {
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
            "@context": "https://agent-os.org/context/memory",
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

    pub fn archive_session_actions(
        &self,
        task_iri: &str,
        actions: &[TrackedAction],
        summary: &str,
    ) -> Result<(), CoreError> {
        if actions.is_empty() {
            return Ok(());
        }
        let task_id = format!("iri://task/{}", task_iri);
        let mut produces = vec![];
        for a in actions {
            for fc in &a.files_created {
                produces.push(serde_json::json!({
                    "@id": format!("iri://file/{}", fc.path.replace('/', "_")),
                    "@type": "https://agent-os.org/ontology/core/File",
                    "https://agent-os.org/ontology/core/filePath": fc.path,
                }));
            }
        }
        let json_ld = serde_json::json!({
            "@context": {"aos": "https://agent-os.org/ontology/core/"},
            "@id": task_id,
            "@type": "aos:Task",
            "aos:hasStatus": "completed",
            "aos:produces": produces,
            "aos:summary": summary,
            "aos:actionCount": actions.len(),
        })
        .to_string();
        self.l2.write_node(&task_id, &json_ld, &self.config)
    }

    /// Preserve one BizAgent's tool-level execution trace for diagnostics.
    /// This is deliberately not tagged as a reusable task `experience`:
    /// only SA's terminal user-task outcome may enter that retrieval pool.
    pub fn archive_agent_execution(
        &self,
        task_iri: &str,
        agent_role: &str,
        summary: &str,
        success_rate: f32,
    ) -> Result<(), CoreError> {
        let exp = serde_json::json!({
            "@type": "AgentExecutionTrace",
            "experience_id": format!("exp_{}", uuid::Uuid::new_v4().hyphenated()),
            "scenario": summary,
            "pattern": if success_rate < 0.5 { "had_failures" } else { "all_success" },
            "success_rating": success_rate,
            "tags": ["agent_execution", agent_role],
            "task_iri": task_iri,
            "created_at": chrono::Utc::now().to_rfc3339(),
        })
        .to_string();
        let iri = format!(
            "iri://execution-trace/{}",
            uuid::Uuid::new_v4().hyphenated()
        );
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
                let actual_frame = if frame.is_some() {
                    frame_name
                } else {
                    "reference_only"
                };
                let proj = self.projection.clone();
                let task_iri = task_iri.to_string();
                let actual_frame = actual_frame.to_string();

                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(async { proj.project(&task_iri, &actual_frame, params).await })
                })?;
                Ok(Some(result))
            }
            Err(_) => {
                let frames: Vec<String> = self
                    .projection
                    .list_frames()
                    .iter()
                    .map(|f| f.name.clone())
                    .collect();
                let result = serde_json::json!({
                    "@context": "https://agent-os.org/context/projection",
                    "note": "Async runtime not available, returning frame list",
                    "available_frames": frames,
                })
                .to_string();
                Ok(Some(result))
            }
        }
    }

    // ========== Unified Storage Interface ==========

    /// Unified storage interface: store data by layer
    pub fn store(
        &self,
        agent_id: &str,
        key: &str,
        value: &str,
        layer: &str,
    ) -> Result<String, CoreError> {
        match layer {
            "L0" | "l0" => {
                let iri = format!("iri://{}/{}", agent_id, key);
                self.l0.store(&iri, value)?;
                Ok(iri)
            }
            "L1" | "l1" => Err(CoreError::Internal {
                message:
                    "L1 layer does not support direct key-value storage; use session APIs instead"
                        .to_string(),
            }),
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
            let session =
                scheduler
                    .remove_session(session_id)
                    .ok_or_else(|| CoreError::Internal {
                        message: format!("Session not found: {}", session_id),
                    })?;
            let summary = session.summarize();
            self.archive_to_l0(&summary)?;
            self.archive_to_l2(&summary.task_iri, &summary)?;
            Ok(())
        } else {
            let session = self
                .sessions
                .get(session_id)
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
    pub fn finalize_session(
        &mut self,
        session: L1Session,
        task_iri: &str,
    ) -> Result<(), CoreError> {
        let session_id = session.session_id().to_string();
        self.track_session(session);
        let summary = self.close_session(&session_id)?;
        // Durable archive first; L2 is a rebuildable active projection.
        self.archive_to_l0(&summary)?;
        self.archive_to_l2(task_iri, &summary)?;
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
    pub fn update_agent_status(
        &self,
        agent_id: &str,
        status: crate::memory::AgentActivity,
        operation: Option<&str>,
    ) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::EvictionConfig;

    fn manager_with_config(config: CoreConfig) -> (MemoryManager, Arc<L0Store>, Arc<Blackboard>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.keep();
        let l0 = Arc::new(L0Store::new(path.to_string_lossy().as_ref()).unwrap());
        let l2 = Arc::new(Blackboard::new().unwrap());
        let projection = Arc::new(ProjectionEngine::new(l2.clone(), 4096));
        (
            MemoryManager::new(l0.clone(), l2.clone(), projection, config),
            l0,
            l2,
        )
    }

    #[test]
    fn configured_l1_budget_is_used_with_and_without_custom_eviction() {
        let mut config = CoreConfig::default();
        config.l1_token_budget = 321;
        let (mut manager, _, _) = manager_with_config(config.clone());
        assert_eq!(
            manager
                .create_session("da", "DA", "iri://task/budget")
                .token_budget(),
            321
        );

        config.eviction_config = Some(EvictionConfig::default_sa());
        let (mut manager, _, _) = manager_with_config(config);
        assert_eq!(
            manager
                .create_session("ca", "CA", "iri://task/budget")
                .token_budget(),
            321
        );
    }

    #[test]
    fn l1_reference_overrides_preserve_role_specific_weights() {
        let mut config = CoreConfig::default();
        config.l1_max_low_relevance_refs = Some(9);
        config.l1_reload_preview_chars = Some(777);
        let (mut manager, _, _) = manager_with_config(config);
        let session = manager.create_session("ca", "CA", "iri://task/l1-overrides");
        assert!(session.eviction_config().relevance_weight > 0.6);
        assert_eq!(session.eviction_config().max_low_relevance_refs, 9);
        assert_eq!(session.eviction_config().reload_preview_chars, 777);
    }

    #[test]
    fn finalize_does_not_leak_active_session_count() {
        let (mut manager, l0, l2) = manager_with_config(CoreConfig::default());
        let mut session = manager.create_session("da", "DA", "iri://task/finalize-count");
        session.add_summary("DA", "compact historical summary", None);
        assert_eq!(manager.l1_session_count(), 1);

        manager
            .finalize_session(session, "iri://task/finalize-count")
            .unwrap();
        assert_eq!(manager.l1_session_count(), 0);
        assert_eq!(manager.session_count(), 0);
        assert_eq!(l0.count().unwrap(), 1);
        assert_eq!(l2.get_task_nodes("iri://task/finalize-count").len(), 1);
    }
}
