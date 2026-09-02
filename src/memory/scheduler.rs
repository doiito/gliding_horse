use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tracing::debug;

use crate::core::agent_instance::AgentRole;
use crate::memory::consistency_engine::ConsistencyEngine;
use crate::memory::context_recall::ContextRecallQuery;
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
    /// P1 lexical+dense fusion starts in shadow mode. An explicit operator
    /// admission can enable it after independent retrieval evaluation.
    fused_recall_enabled: AtomicBool,
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
            fused_recall_enabled: AtomicBool::new(false),
        }
    }

    pub async fn on_context_request(
        &self,
        agent_role: AgentRole,
        task_iri: &str,
    ) -> Result<String, CoreError> {
        self.on_context_request_with_query(agent_role, &ContextRecallQuery::legacy(task_iri))
            .await
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
        self.context_request_with_decay_query(
            agent_role,
            &ContextRecallQuery::legacy(task_iri),
            decay_lambda,
        )
        .await
    }

    /// Typed recall path. The task IRI remains a scope/audit key; only
    /// `semantic_text` is passed to vector or lexical retrieval.
    pub async fn context_request_with_decay_query(
        &self,
        agent_role: AgentRole,
        query: &ContextRecallQuery,
        decay_lambda: f64,
    ) -> Result<String, CoreError> {
        if let Some(ref hs) = self.hyperspace {
            let filter = crate::memory::hyperspace_store::HybridSearchFilter::new();
            if !query.is_empty() {
                let fused_results = hs
                    .search_fused_recall(&query.semantic_text, &filter, decay_lambda, 10)
                    .await?;
                if self.fused_recall_enabled.load(Ordering::Acquire) {
                    if !fused_results.is_empty() {
                        let contents: Vec<String> =
                            fused_results.iter().map(|r| r.text.clone()).collect();
                        return Ok(contents.join("\n"));
                    }
                } else {
                    debug!(
                        task_iri = %query.task_iri,
                        query_version = query.query_version,
                        field_sources = ?query.field_sources,
                        fused_candidate_count = fused_results.len(),
                        "Fused retrieval evaluated in shadow mode"
                    );
                    let baseline = hs
                        .search_with_time_decay(&query.semantic_text, &filter, decay_lambda, 10)
                        .await?;
                    if !baseline.is_empty() {
                        return Ok(baseline
                            .iter()
                            .map(|entry| entry.text.clone())
                            .collect::<Vec<_>>()
                            .join("\n"));
                    }
                }
            }
        }
        // Legacy read paths still use task IRI for graph scope. Query text is
        // used for content matching when it is available.
        self.on_context_request_with_query(agent_role, query).await
    }

    async fn on_context_request_with_query(
        &self,
        agent_role: AgentRole,
        query: &ContextRecallQuery,
    ) -> Result<String, CoreError> {
        let task_iri = &query.task_iri;
        let frame_name = match agent_role {
            AgentRole::Plan => "pa_init",
            AgentRole::Do => "da_input",
            AgentRole::Check => "ca_review",
            AgentRole::Act => "aa_decision",
        };
        let existing_nodes = self.blackboard.query_nodes(task_iri)?;
        for node in &existing_nodes {
            self.consistency.on_l2_read(&node.iri)?;
        }
        let params = HashMap::new();
        if let Ok(result) = self.projection.project(task_iri, frame_name, params).await {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&result) {
                if parsed
                    .get("artifacts")
                    .and_then(|artifacts| artifacts.as_array())
                    .is_some_and(|artifacts| !artifacts.is_empty())
                {
                    return Ok(result);
                }
            } else {
                return Ok(result);
            }
        }
        let nodes = self.blackboard.query_nodes(task_iri)?;
        if !nodes.is_empty() {
            return Ok(nodes
                .iter()
                .map(|node| node.json_ld.clone())
                .collect::<Vec<_>>()
                .join("\n"));
        }
        let content_query = if query.is_empty() {
            task_iri.as_str()
        } else {
            query.semantic_text.as_str()
        };
        let results = self.l0_store.search(content_query, 10)?;
        Ok(results
            .iter()
            .map(|entry| entry.content.clone())
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// Attach a HyperspaceStore at runtime (for delayed injection).
    pub fn with_hyperspace_store(&mut self, hs: Arc<HyperspaceStore>) {
        self.hyperspace = Some(hs);
    }

    /// Enable only after a separately labelled offline evaluation and explicit
    /// review. Disabling immediately returns the scheduler to semantic-only
    /// baseline while retaining shadow diagnostics.
    pub fn set_fused_recall_enabled(&self, enabled: bool) {
        self.fused_recall_enabled.store(enabled, Ordering::Release);
    }

    pub fn fused_recall_enabled(&self) -> bool {
        self.fused_recall_enabled.load(Ordering::Acquire)
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
        self.archive_to_l0_with_update(session_id, role, thought, content, |iri| async move {
            self.consistency.on_l0_update(&iri).await
        })
        .await
    }

    async fn archive_to_l0_with_update<F, Fut>(
        &self,
        session_id: &str,
        role: &str,
        thought: &str,
        content: &str,
        on_update: F,
    ) -> Result<String, CoreError>
    where
        F: FnOnce(String) -> Fut,
        Fut: Future<Output = Result<(), CoreError>>,
    {
        // Keep synchronous session access in its own scope. In particular, the
        // parking_lot read guard must not survive into the asynchronous
        // consistency notification below: doing so blocks session writers and
        // makes this future non-Send if that notification ever suspends.
        let iri = {
            let sessions = self.sessions.read();
            let session = sessions
                .get(session_id)
                .ok_or_else(|| CoreError::Internal {
                    message: format!("Session not found: {}", session_id),
                })?;
            session.archive_full_to_l0(&self.l0_store, role, thought, content)?
        };

        if let Err(e) = on_update(iri.clone()).await {
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
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::sync::Notify;

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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn archive_releases_session_lock_before_async_consistency_update() {
        let (scheduler, _dir) = setup_scheduler();
        let session_id = scheduler.create_session("a1", "DA", "iri://task/archive", 1000);
        let update_started = Arc::new(Notify::new());
        let release_update = Arc::new(Notify::new());

        let archive_scheduler = scheduler.clone();
        let archive_session_id = session_id.clone();
        let update_started_for_task = update_started.clone();
        let release_update_for_task = release_update.clone();
        let archive_task = tokio::spawn(async move {
            archive_scheduler
                .archive_to_l0_with_update(
                    &archive_session_id,
                    "DA",
                    "reasoning",
                    "archived content",
                    move |_| async move {
                        update_started_for_task.notify_one();
                        release_update_for_task.notified().await;
                        Ok(())
                    },
                )
                .await
        });

        tokio::time::timeout(Duration::from_secs(1), update_started.notified())
            .await
            .expect("archive did not reach the asynchronous consistency update");

        // Hold the consistency future at an actual suspension point. A writer
        // must still acquire the sessions lock while that future is pending.
        let writer_scheduler = scheduler.clone();
        let writer_session_id = session_id.clone();
        let mut writer_task = tokio::task::spawn_blocking(move || {
            writer_scheduler.remove_session(&writer_session_id)
        });
        let writer_result = tokio::time::timeout(Duration::from_secs(1), &mut writer_task).await;

        // Always release the archive task before asserting so a regression
        // cannot leave the test runtime waiting on a blocked task.
        release_update.notify_one();
        let archive_iri = archive_task
            .await
            .expect("archive task panicked")
            .expect("archive failed");

        let removed = match writer_result {
            Ok(join_result) => join_result.expect("session writer panicked"),
            Err(_) => {
                let _ = writer_task.await;
                panic!("session writer was blocked by the pending consistency update");
            }
        };

        assert!(removed.is_some());
        assert!(archive_iri.starts_with("iri://archive/task/archive/DA/"));
    }

    #[tokio::test]
    async fn context_request_repairs_stale_l2_when_invalidation_event_is_missed() {
        let (scheduler, _dir) = setup_scheduler();
        let iri = "iri://task/read-repair/node_1";
        let task = "iri://task/read-repair";
        let config = crate::CoreConfig::default();
        let version_one = serde_json::json!({
            "@id": iri,
            "@type": "TaskArtifact",
            "summary": "canonical-version-one"
        })
        .to_string();
        let version_two = serde_json::json!({
            "@id": iri,
            "@type": "TaskArtifact",
            "summary": "canonical-version-two"
        })
        .to_string();

        scheduler.l0_store.store(iri, &version_one).unwrap();
        scheduler
            .blackboard
            .write_node(iri, &version_one, &config)
            .unwrap();
        assert_eq!(scheduler.blackboard.node_generation(iri), Some(1));

        // Deliberately bypass ConsistencyEngine::on_l0_update to model a
        // dropped or delayed invalidation event.
        scheduler.l0_store.store(iri, &version_two).unwrap();
        let context = scheduler
            .on_context_request(AgentRole::Do, task)
            .await
            .unwrap();

        let repaired = scheduler.blackboard.read_node(iri).unwrap().unwrap();
        assert_eq!(repaired.generation, 2);
        assert!(repaired.json_ld.contains("canonical-version-two"));
        assert!(!repaired.json_ld.contains("canonical-version-one"));
        assert!(context.contains("canonical-version-two"));
    }
}
