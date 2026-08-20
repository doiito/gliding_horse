use std::collections::HashMap;
use std::sync::Arc;

use tracing::debug;

use crate::core::agent_instance::AgentRole;
use crate::core::timeline::TimeRange;
use crate::memory::consistency_engine::ConsistencyEngine;
use crate::memory::hyperspace_store::HyperspaceStore;
use crate::memory::l0_store::L0Store;
use crate::memory::l1_session::L1Session;
use crate::memory::l2_blackboard::Blackboard;
use crate::memory::l3_projection::ProjectionEngine;
use crate::memory::memory_bus::MemoryBus;
use crate::CoreError;

pub struct MemoryScheduler {
    l0_store: Arc<L0Store>,
    blackboard: Arc<Blackboard>,
    projection: Arc<ProjectionEngine>,
    consistency: Arc<ConsistencyEngine>,
    memory_bus: Arc<MemoryBus>,
    sessions: parking_lot::RwLock<HashMap<String, L1Session>>,
    /// Optional HyperspaceStore for time-decayed vector search
    hyperspace: Option<Arc<HyperspaceStore>>,
}

impl MemoryScheduler {
    pub fn new(
        l0_store: Arc<L0Store>,
        blackboard: Arc<Blackboard>,
        projection: Arc<ProjectionEngine>,
        consistency: Arc<ConsistencyEngine>,
        memory_bus: Arc<MemoryBus>,
    ) -> Self {
        Self::with_hyperspace(
            l0_store,
            blackboard,
            projection,
            consistency,
            memory_bus,
            None,
        )
    }

    pub fn with_hyperspace(
        l0_store: Arc<L0Store>,
        blackboard: Arc<Blackboard>,
        projection: Arc<ProjectionEngine>,
        consistency: Arc<ConsistencyEngine>,
        memory_bus: Arc<MemoryBus>,
        hyperspace: Option<Arc<HyperspaceStore>>,
    ) -> Self {
        Self {
            l0_store,
            blackboard,
            projection,
            consistency,
            memory_bus,
            sessions: parking_lot::RwLock::new(HashMap::new()),
            hyperspace,
        }
    }

    pub async fn on_context_request(
        &self,
        agent_role: AgentRole,
        task_iri: &str,
    ) -> Result<String, CoreError> {
        let frame_name = match agent_role {
            AgentRole::Plan => "pa_init",
            AgentRole::Do => "da_input",
            AgentRole::Check => "ca_review",
            AgentRole::Act => "aa_decision",
        };

        let params = HashMap::new();
        let projection_result = self.projection.project(task_iri, frame_name, params).await;

        match projection_result {
            Ok(result) => {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&result) {
                    if let Some(artifacts) = parsed.get("artifacts").and_then(|a| a.as_array()) {
                        if !artifacts.is_empty() {
                            return Ok(result);
                        }
                    }
                } else {
                    return Ok(result);
                }
            }
            Err(_) => {}
        }

        let nodes = self.blackboard.query_nodes(task_iri)?;
        if !nodes.is_empty() {
            for n in &nodes {
                let _ = self.consistency.on_l2_read(&n.iri);
            }
            let contents: Vec<String> = nodes.iter().map(|n| n.json_ld.clone()).collect();
            return Ok(contents.join("\n"));
        }

        // Healthy HyperspaceStore → time-windowed vector recall before the pure L0 scan
        if let Some(ref hs) = self.hyperspace {
            if hs.embedding_provider() != "fallback" {
                let now = chrono::Utc::now();
                let window = TimeRange::after(now - chrono::Duration::hours(24));
                if let Ok(entries) = hs.search_by_time(task_iri, &window, 10).await {
                    if !entries.is_empty() {
                        let contents: Vec<String> = entries.iter().map(|r| r.text.clone()).collect();
                        return Ok(contents.join("\n"));
                    }
                }
            }
        }

        let results = self.l0_store.search(task_iri, 10)?;
        if !results.is_empty() {
            let contents: Vec<String> = results.iter().map(|r| r.content.clone()).collect();
            return Ok(contents.join("\n"));
        }

        Ok(String::new())
    }

    pub fn on_l1_overflow(&self, session_id: &str) -> Result<usize, CoreError> {
        let mut sessions = self.sessions.write();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| CoreError::Internal {
                message: format!("Session not found: {}", session_id),
            })?;
        Ok(session.evict_by_policy())
    }

    pub async fn on_task_complete(&self, task_iri: &str) -> Result<(), CoreError> {
        self.blackboard.flush_dirty_nodes(&self.l0_store)?;
        self.blackboard.release_subtree(task_iri)?;
        self.memory_bus
            .publish("TASK_COMPLETED", task_iri, "{}")
            .await;
        Ok(())
    }

    /// Context request with time-decayed search at the vector-store level.
    ///
    /// Falls back to `on_context_request` when HyperspaceStore is not available.
    pub async fn context_request_with_decay(
        &self,
        agent_role: AgentRole,
        task_iri: &str,
        decay_lambda: f64,
    ) -> Result<String, CoreError> {
        if let Some(ref hs) = self.hyperspace {
            let filter = crate::memory::hyperspace_store::HybridSearchFilter::new();
            let results = hs
                .search_with_time_decay(task_iri, &filter, decay_lambda, 10)
                .await?;
            if !results.is_empty() {
                let contents: Vec<String> = results.iter().map(|r| r.text.clone()).collect();
                return Ok(contents.join("\n"));
            }
        }
        self.on_context_request(agent_role, task_iri).await
    }

    /// Attach a HyperspaceStore at runtime (for delayed injection).
    pub fn with_hyperspace_store(&mut self, hs: Arc<HyperspaceStore>) {
        self.hyperspace = Some(hs);
    }

    pub fn on_session_close(&self, session_id: &str) -> Result<(), CoreError> {
        let session = {
            let mut sessions = self.sessions.write();
            sessions
                .remove(session_id)
                .ok_or_else(|| CoreError::Internal {
                    message: format!("Session not found: {}", session_id),
                })?
        };

        let summary = session.summarize();
        let config = crate::CoreConfig::default();

        let json_ld = serde_json::json!({
            "@context": "https://agent-os.org/context/memory",
            "@id": format!("iri://memory/{}", uuid::Uuid::new_v4().hyphenated()),
            "@type": "SessionSummary",
            "session_id": summary.session_id,
            "agent_id": summary.agent_id,
            "agent_role": summary.agent_role,
            "task_iri": summary.task_iri,
            "turn_count": summary.turn_count,
            "summary_text": summary.summary_text,
        })
        .to_string();

        self.blackboard.write_node(
            &format!("iri://session/{}", summary.session_id),
            &json_ld,
            &config,
        )?;

        let l0_iri = format!("iri://archive/session/{}", summary.session_id);
        let content = serde_json::json!({
            "session_id": summary.session_id,
            "agent_id": summary.agent_id,
            "agent_role": summary.agent_role,
            "task_iri": summary.task_iri,
            "turn_count": summary.turn_count,
            "summary_text": summary.summary_text,
        })
        .to_string();
        self.l0_store.store(&l0_iri, &content)?;

        debug!(session_id = %session_id, "Session closed and archived to L2+L0");
        Ok(())
    }

    pub fn create_session(
        &self,
        agent_id: &str,
        agent_role: &str,
        task_iri: &str,
        token_budget: usize,
    ) -> String {
        let session = L1Session::with_budget(agent_id, agent_role, task_iri, token_budget);
        let session_id = session.session_id().to_string();
        self.sessions.write().insert(session_id.clone(), session);
        session_id
    }

    pub fn get_session(&self, session_id: &str) -> Option<L1Session> {
        self.sessions.read().get(session_id).cloned()
    }

    pub fn add_summary_to_session(
        &self,
        session_id: &str,
        role: &str,
        summary: &str,
        l0_archive_iri: Option<String>,
    ) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(session_id) {
            session.add_summary(role, summary, l0_archive_iri);
        }
    }

    pub async fn archive_to_l0(
        &self,
        session_id: &str,
        role: &str,
        thought: &str,
        content: &str,
    ) -> Result<String, CoreError> {
        let sessions = self.sessions.read();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| CoreError::Internal {
                message: format!("Session not found: {}", session_id),
            })?;
        let iri = session.archive_full_to_l0(&self.l0_store, role, thought, content)?;
        if let Err(e) = self.consistency.on_l0_update(&iri).await {
            tracing::warn!("Consistency on_l0_update failed: {}", e);
        }
        Ok(iri)
    }

    /// Insert an existing session (called by MemoryManager for synchronization)
    pub fn insert_session(&self, session: L1Session) {
        let id = session.session_id().to_string();
        self.sessions.write().insert(id, session);
    }

    /// Remove and return the specified session (called by MemoryManager for synchronous shutdown)
    pub fn remove_session(&self, session_id: &str) -> Option<L1Session> {
        self.sessions.write().remove(session_id)
    }

    /// Return the current session count
    pub fn session_count(&self) -> usize {
        self.sessions.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_bus::EventBus;
    use crate::memory::consistency_engine::ConsistencyEngine;
    use crate::memory::l0_store::L0Store;
    use crate::memory::l2_blackboard::Blackboard;
    use crate::memory::l3_projection::ProjectionEngine;
    use crate::memory::memory_bus::MemoryBus;
    use tempfile::tempdir;

    fn setup_scheduler() -> (Arc<MemoryScheduler>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("l0_sched");
        let l0_store = Arc::new(L0Store::new(path.to_str().unwrap()).unwrap());
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let projection = Arc::new(ProjectionEngine::new(blackboard.clone(), 1024));
        let event_bus = Arc::new(EventBus::new(100));
        let memory_bus = Arc::new(MemoryBus::new(event_bus));
        let consistency = Arc::new(ConsistencyEngine::new(
            memory_bus.clone(),
            l0_store.clone(),
            blackboard.clone(),
            projection.clone(),
        ));
        let scheduler = Arc::new(MemoryScheduler::new(
            l0_store,
            blackboard,
            projection,
            consistency,
            memory_bus,
        ));
        (scheduler, dir)
    }

    #[test]
    fn test_create_and_get_session() {
        let (scheduler, _dir) = setup_scheduler();
        let id = scheduler.create_session("agent1", "PA", "iri://task1", 1000);
        let session = scheduler.get_session(&id);
        assert!(session.is_some());
        assert_eq!(session.unwrap().agent_id(), "agent1");
    }

    #[test]
    fn test_session_count() {
        let (scheduler, _dir) = setup_scheduler();
        assert_eq!(scheduler.session_count(), 0);
        scheduler.create_session("a1", "PA", "iri://t1", 500);
        assert_eq!(scheduler.session_count(), 1);
        scheduler.create_session("a2", "DO", "iri://t2", 500);
        assert_eq!(scheduler.session_count(), 2);
    }

    #[test]
    fn test_insert_and_remove_session() {
        let (scheduler, _dir) = setup_scheduler();
        let session = L1Session::with_budget("ext", "PA", "iri://ext", 500);
        let id = session.session_id().to_string();

        scheduler.insert_session(session);
        assert_eq!(scheduler.session_count(), 1);

        let removed = scheduler.remove_session(&id);
        assert!(removed.is_some());
        assert_eq!(scheduler.session_count(), 0);
    }

    #[test]
    fn test_on_l1_overflow() {
        let (scheduler, _dir) = setup_scheduler();
        let id = scheduler.create_session("a1", "PA", "iri://t1", 10);

        let evicted = scheduler.on_l1_overflow(&id);
        assert!(evicted.is_ok());
    }

    #[test]
    fn test_add_summary_to_session() {
        let (scheduler, _dir) = setup_scheduler();
        let id = scheduler.create_session("a1", "PA", "iri://t1", 1000);

        scheduler.add_summary_to_session(&id, "PA", "Summary text", None);
        let session = scheduler.get_session(&id).unwrap();
        assert_eq!(session.turn_count(), 1);
    }

    #[test]
    fn test_on_session_close() {
        let (scheduler, _dir) = setup_scheduler();
        let id = scheduler.create_session("a1", "PA", "iri://t1", 1000);

        let result = scheduler.on_session_close(&id);
        assert!(result.is_ok());
        assert!(scheduler.get_session(&id).is_none());
    }

    #[test]
    fn test_on_session_close_nonexistent() {
        let (scheduler, _dir) = setup_scheduler();
        let result = scheduler.on_session_close("iri://nonexistent");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_on_task_complete() {
        let (scheduler, _dir) = setup_scheduler();
        let result = scheduler.on_task_complete("iri://task_complete").await;
        assert!(result.is_ok());
    }
}
