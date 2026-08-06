use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{debug, instrument, warn};

use crate::memory::l0_store::{L0Store, MesiState};
use crate::memory::l2_blackboard::Blackboard;
use crate::memory::l3_projection::ProjectionEngine;
use crate::memory::memory_bus::MemoryBus;
use crate::{CoreConfig, CoreError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteStrategy {
    WriteThrough,
    WriteBack,
}

pub struct ConsistencyEngine {
    memory_bus: Arc<MemoryBus>,
    l0_store: Arc<L0Store>,
    blackboard: Arc<Blackboard>,
    projection: Arc<ProjectionEngine>,
    critical_tags: Arc<RwLock<HashSet<String>>>,
}

impl ConsistencyEngine {
    pub fn new(
        memory_bus: Arc<MemoryBus>,
        l0_store: Arc<L0Store>,
        blackboard: Arc<Blackboard>,
        projection: Arc<ProjectionEngine>,
    ) -> Self {
        let critical_tags = Arc::new(RwLock::new(HashSet::from([
            "emphasis".to_string(),
            "user_intent".to_string(),
            "confirmed_fact".to_string(),
        ])));

        let result = Self {
            memory_bus,
            l0_store,
            blackboard,
            projection,
            critical_tags,
        };
        let hook_blackboard = result.blackboard.clone();
        let hook_l0 = result.l0_store.clone();
        let hook_critical = result.critical_tags.clone();
        let hook_projection = result.projection.clone();
        let closure_blackboard = hook_blackboard.clone();
        hook_blackboard.set_write_hook(move |node_iri, tags| {
            hook_projection.invalidate_by_node(node_iri);
            let has_critical = tags.iter().any(|t| hook_critical.read().contains(t));
            if has_critical {
                let _ = closure_blackboard.persist_node(node_iri, &hook_l0);
            }
        });
        result
    }

    #[instrument(skip(self, tags))]
    pub async fn on_l2_write(
        &self,
        node_iri: &str,
        task_iri: &str,
        tags: &[String],
    ) -> Result<(), CoreError> {
        let strategy = self.determine_write_strategy(tags);

        self.blackboard.mark_dirty(node_iri);
        debug!(node_iri = %node_iri, strategy = ?strategy, "L2 write: marking dirty node");

        if strategy == WriteStrategy::WriteThrough {
            let flushed = self.blackboard.flush_dirty_nodes(&self.l0_store)?;
            debug!(node_iri = %node_iri, flushed = flushed, "WriteThrough: dirty node flushed to L0");
        }

        self.memory_bus.publish_invalidate(node_iri, task_iri).await;

        self.projection.invalidate_by_node(node_iri);

        Ok(())
    }

    #[instrument(skip(self))]
    pub fn on_l2_read(&self, node_iri: &str) -> Result<(), CoreError> {
        if let Some(node) = self.blackboard.read_node(node_iri)? {
            match node.mesi_state {
                MesiState::Modified | MesiState::Exclusive | MesiState::Shared => {
                    debug!(node_iri = %node_iri, state = ?node.mesi_state, "L2 read: cache hit");
                }
                MesiState::Invalid => {
                    debug!(node_iri = %node_iri, "L2 read: Invalid state, reloading from L0");
                    self.blackboard.delete_node(node_iri)?;
                    if let Some(entry) = self.l0_store.retrieve(node_iri)? {
                        let config = CoreConfig::default();
                        self.blackboard
                            .write_node(node_iri, &entry.content, &config)?;
                        debug!(node_iri = %node_iri, "L2 read: node reloaded from L0");
                    } else {
                        warn!(node_iri = %node_iri, "L2 read: no corresponding entry in L0");
                    }
                }
            }
        }
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn on_l0_update(&self, iri: &str) -> Result<(), CoreError> {
        self.l0_store.update_mesi_state(iri, MesiState::Modified)?;
        debug!(iri = %iri, "L0 update: MESI state set to Modified");

        self.memory_bus.publish_invalidate(iri, iri).await;

        self.projection.invalidate_by_node(iri);

        Ok(())
    }

    pub fn determine_write_strategy(&self, tags: &[String]) -> WriteStrategy {
        let critical = self.critical_tags.read();
        let has_critical = tags.iter().any(|t| critical.contains(t));
        if has_critical {
            WriteStrategy::WriteThrough
        } else {
            WriteStrategy::WriteBack
        }
    }

    pub fn add_critical_tag(&self, tag: &str) {
        self.critical_tags.write().insert(tag.to_string());
    }

    pub fn remove_critical_tag(&self, tag: &str) {
        self.critical_tags.write().remove(tag);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_bus::EventBus;
    use crate::memory::l0_store::L0Store;
    use crate::memory::l2_blackboard::Blackboard;
    use crate::memory::l3_projection::ProjectionEngine;
    use crate::memory::memory_bus::MemoryBus;
    use crate::CoreConfig;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn setup() -> (
        Arc<ConsistencyEngine>,
        Arc<Blackboard>,
        Arc<L0Store>,
        Arc<MemoryBus>,
    ) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("l0_test");
        let l0_store = Arc::new(L0Store::new(path.to_str().unwrap()).unwrap());
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let projection_engine = Arc::new(ProjectionEngine::new(blackboard.clone(), 1024));
        let event_bus = Arc::new(EventBus::new(100));
        let memory_bus = Arc::new(MemoryBus::new(event_bus));
        let consistency = Arc::new(ConsistencyEngine::new(
            memory_bus.clone(),
            l0_store.clone(),
            blackboard.clone(),
            projection_engine,
        ));
        (consistency, blackboard, l0_store, memory_bus)
    }

    #[tokio::test]
    async fn test_on_l2_write_write_through_for_critical() {
        let (consistency, blackboard, _l0, _bus) = setup();
        let config = CoreConfig::default();
        blackboard
            .write_node("iri://node1", r#"{"@id":"iri://node1"}"#, &config)
            .unwrap();

        consistency
            .on_l2_write("iri://node1", "iri://task", &["emphasis".to_string()])
            .await
            .unwrap();

        let node = blackboard.read_node("iri://node1").unwrap().unwrap();
        // After WriteThrough flush to L0, state resets to Shared (clean, in-sync)
        assert_eq!(node.mesi_state, MesiState::Shared);
        assert!(!node.dirty);
    }

    #[tokio::test]
    async fn test_on_l2_write_write_back_for_normal() {
        let (consistency, blackboard, _l0, _bus) = setup();
        let config = CoreConfig::default();
        blackboard
            .write_node("iri://node2", r#"{"@id":"iri://node2"}"#, &config)
            .unwrap();

        consistency
            .on_l2_write("iri://node2", "iri://task", &["normal_tag".to_string()])
            .await
            .unwrap();

        let node = blackboard.read_node("iri://node2").unwrap().unwrap();
        assert_eq!(node.mesi_state, MesiState::Modified);
        assert!(node.dirty);
    }

    #[test]
    fn test_determine_write_strategy() {
        let (consistency, _bb, _l0, _bus) = setup();

        assert_eq!(
            consistency.determine_write_strategy(&["emphasis".to_string()]),
            WriteStrategy::WriteThrough
        );
        assert_eq!(
            consistency.determine_write_strategy(&["normal".to_string()]),
            WriteStrategy::WriteBack
        );
        assert_eq!(
            consistency.determine_write_strategy(&[]),
            WriteStrategy::WriteBack
        );
    }

    #[test]
    fn test_add_remove_critical_tag() {
        let (consistency, _bb, _l0, _bus) = setup();

        consistency.add_critical_tag("urgent");
        assert_eq!(
            consistency.determine_write_strategy(&["urgent".to_string()]),
            WriteStrategy::WriteThrough
        );

        consistency.remove_critical_tag("urgent");
        assert_eq!(
            consistency.determine_write_strategy(&["urgent".to_string()]),
            WriteStrategy::WriteBack
        );
    }

    #[test]
    fn test_on_l2_read_modified_no_op() {
        let (consistency, blackboard, _l0, _bus) = setup();
        let config = CoreConfig::default();

        blackboard
            .write_node("iri://m1", r#"{"@id":"iri://m1"}"#, &config)
            .unwrap();
        blackboard.mark_dirty("iri://m1");

        let result = consistency.on_l2_read("iri://m1");
        assert!(result.is_ok());

        let node = blackboard.read_node("iri://m1").unwrap().unwrap();
        assert_eq!(node.mesi_state, MesiState::Modified);
    }

    #[test]
    fn test_on_l2_read_shared_no_op() {
        let (consistency, blackboard, _l0, _bus) = setup();
        let config = CoreConfig::default();

        blackboard
            .write_node("iri://s1", r#"{"@id":"iri://s1"}"#, &config)
            .unwrap();

        let result = consistency.on_l2_read("iri://s1");
        assert!(result.is_ok());

        let node = blackboard.read_node("iri://s1").unwrap().unwrap();
        assert_eq!(node.mesi_state, MesiState::Shared);
    }

    #[tokio::test]
    async fn test_on_l0_update() {
        let (consistency, blackboard, l0_store, _bus) = setup();
        let config = CoreConfig::default();

        l0_store
            .store("iri://l0_1", r#"{"@id":"iri://l0_1"}"#)
            .unwrap();
        blackboard
            .write_node("iri://l0_1", r#"{"@id":"iri://l0_1"}"#, &config)
            .unwrap();

        consistency.on_l0_update("iri://l0_1").await.unwrap();
        let entry = l0_store.retrieve("iri://l0_1").unwrap().unwrap();
        assert_eq!(entry.mesi_state, MesiState::Modified);
    }

    #[test]
    fn test_on_l2_read_exclusive_no_op() {
        let (consistency, blackboard, _l0, _bus) = setup();
        let config = CoreConfig::default();

        blackboard
            .write_node("iri://e1", r#"{"@id":"iri://e1"}"#, &config)
            .unwrap();
        let node = blackboard.read_node("iri://e1").unwrap().unwrap();
        assert_eq!(node.mesi_state, MesiState::Shared);

        let result = consistency.on_l2_read("iri://e1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_on_l2_read_nonexistent_ok() {
        let (consistency, _bb, _l0, _bus) = setup();
        let result = consistency.on_l2_read("iri://nonexistent");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_on_l2_write_invalidates_projection() {
        let (consistency, blackboard, _l0, _bus) = setup();
        let config = CoreConfig::default();

        blackboard
            .write_node("iri://proj_test", r#"{"@id":"iri://proj_test"}"#, &config)
            .unwrap();

        consistency
            .on_l2_write("iri://proj_test", "iri://task", &[])
            .await
            .unwrap();

        let node = blackboard.read_node("iri://proj_test").unwrap().unwrap();
        assert_eq!(node.mesi_state, MesiState::Modified);
    }

    #[tokio::test]
    async fn test_write_node_with_critical_tag_flushes_to_l0() {
        let (_consistency, blackboard, l0_store, _bus) = setup();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://critical/1","tags":["emphasis"]}"#;
        blackboard
            .write_node("iri://critical/1", json_ld, &config)
            .unwrap();
        assert!(l0_store.retrieve("iri://critical/1").unwrap().is_some());
    }

    #[tokio::test]
    async fn test_write_node_without_critical_tag_stays_l2_only() {
        let (_consistency, blackboard, l0_store, _bus) = setup();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://normal/1","tags":["ordinary"]}"#;
        blackboard
            .write_node("iri://normal/1", json_ld, &config)
            .unwrap();
        assert!(l0_store.retrieve("iri://normal/1").unwrap().is_none());
    }

    #[tokio::test]
    async fn test_write_node_invalidates_l3_projection_view() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("l0_test");
        let l0_store = Arc::new(L0Store::new(path.to_str().unwrap()).unwrap());
        let blackboard = Arc::new(Blackboard::new().unwrap());
        let projection = Arc::new(ProjectionEngine::new(blackboard.clone(), 1024));
        let event_bus = Arc::new(EventBus::new(100));
        let memory_bus = Arc::new(MemoryBus::new(event_bus));
        let _consistency = Arc::new(ConsistencyEngine::new(
            memory_bus.clone(),
            l0_store.clone(),
            blackboard.clone(),
            projection.clone(),
        ));

        let config = CoreConfig::default();
        let node_iri = "iri://task/inval_proj/node_1";
        let json_ld = r#"{"@id":"iri://task/inval_proj/node_1","@type":"Artifact","status":"created"}"#;
        blackboard
            .write_node(node_iri, json_ld, &config)
            .unwrap();

        projection
            .project("iri://task/inval_proj", "reference_only", HashMap::new())
            .await
            .unwrap();
        assert_eq!(projection.cleanup_invalid(), 0);

        blackboard
            .write_node(node_iri, json_ld, &config)
            .unwrap();
        assert_eq!(
            projection.cleanup_invalid(),
            1,
            "write hook 应使 L3 投影视图失效(回归:P0 删除 on_l2_write 后投影失效路径断裂)"
        );
    }
}
