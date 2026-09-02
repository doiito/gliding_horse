use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

/// Supplementary input entry
#[derive(Debug, Clone)]
pub struct SupplementEntry {
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub relevance_score: f64,
    pub timestamp: DateTime<Utc>,
    pub consumed: bool,
}

/// Supplementary input shared store
///
/// SA (producer) stores supplementary inputs after classification, also writes to L1Session.
/// AgentRunner (consumer) checks and injects into messages[] on each CycleStart.
///
/// Design goals:
/// - Decouple SA event handling from AgentRunner execution loop
/// - Ensure supplementary inputs are not lost (even if AgentRunner is not at CycleStart)
/// - Support batch consumption (multiple supplementary inputs injected in one CycleStart)
pub struct SupplementaryInputStore {
    /// task_iri → supplementary input list
    pending: Arc<Mutex<HashMap<String, Vec<SupplementEntry>>>>,
}

impl SupplementaryInputStore {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Called by SA: store a supplementary input
    pub fn store(
        &self,
        task_iri: &str,
        content: &str,
        embedding: Option<Vec<f32>>,
        relevance_score: f64,
    ) {
        let entry = SupplementEntry {
            content: content.to_string(),
            embedding,
            relevance_score,
            timestamp: Utc::now(),
            consumed: false,
        };
        let mut map = self
            .pending
            .lock()
            .expect("SupplementaryInputStore lock poisoned");
        map.entry(task_iri.to_string()).or_default().push(entry);
        tracing::info!(
            task_iri = %task_iri,
            content = %content.chars().take(80).collect::<String>(),
            score = relevance_score,
            "Supplementary input stored in SupplementaryInputStore"
        );
    }

    /// Called by AgentRunner: fetch all unconsumed supplementary inputs for the current task
    ///
    /// Atomic operation: fetched entries are marked consumed but kept in the list for auditing.
    /// Returns Vec instead of iterator to minimize lock hold time.
    pub fn take_pending(&self, task_iri: &str) -> Vec<SupplementEntry> {
        let mut map = self
            .pending
            .lock()
            .expect("SupplementaryInputStore lock poisoned");
        let entries = map.entry(task_iri.to_string()).or_default();
        let pending: Vec<_> = entries
            .iter_mut()
            .filter(|e| !e.consumed)
            .map(|e| {
                e.consumed = true;
                e.clone()
            })
            .collect();
        pending
    }

    /// Return an audit snapshot of inputs that have not yet been injected
    /// into an AgentRunner turn.
    ///
    /// Checkpointing must use this instead of [`Self::take_pending`]. A
    /// checkpoint is observational; consuming here would silently discard a
    /// user's instruction before the next agent receives it.
    pub fn snapshot_pending(&self, task_iri: &str) -> Vec<SupplementEntry> {
        let map = self
            .pending
            .lock()
            .expect("SupplementaryInputStore lock poisoned");
        map.get(task_iri)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|entry| !entry.consumed)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if there are unconsumed supplementary inputs
    pub fn has_pending(&self, task_iri: &str) -> bool {
        let map = self
            .pending
            .lock()
            .expect("SupplementaryInputStore lock poisoned");
        map.get(task_iri)
            .map(|entries| entries.iter().any(|e| !e.consumed))
            .unwrap_or(false)
    }

    /// Get the count of unconsumed entries for the specified task
    pub fn pending_count(&self, task_iri: &str) -> usize {
        let map = self
            .pending
            .lock()
            .expect("SupplementaryInputStore lock poisoned");
        map.get(task_iri)
            .map(|entries| entries.iter().filter(|e| !e.consumed).count())
            .unwrap_or(0)
    }

    /// Clean up data for completed tasks
    pub fn cleanup(&self, task_iri: &str) {
        let mut map = self
            .pending
            .lock()
            .expect("SupplementaryInputStore lock poisoned");
        map.remove(task_iri);
    }
}

impl Default for SupplementaryInputStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SupplementaryInputStore {
    fn clone(&self) -> Self {
        Self {
            pending: self.pending.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_take_pending() {
        let store = SupplementaryInputStore::new();
        store.store("iri://task/test1", "supplement info 1", None, 0.85);
        store.store(
            "iri://task/test1",
            "supplement info 2",
            Some(vec![1.0, 2.0]),
            0.45,
        );

        let pending = store.take_pending("iri://task/test1");
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].content, "supplement info 1");
        assert_eq!(pending[1].content, "supplement info 2");
        assert!(pending[0].consumed);
    }

    #[test]
    fn test_take_pending_only_once() {
        let store = SupplementaryInputStore::new();
        store.store("iri://task/test2", "single", None, 0.9);

        let first = store.take_pending("iri://task/test2");
        assert_eq!(first.len(), 1);

        let second = store.take_pending("iri://task/test2");
        assert_eq!(
            second.len(),
            0,
            "consumed entries should not be returned again"
        );
    }

    #[test]
    fn snapshot_pending_does_not_consume_instruction() {
        let store = SupplementaryInputStore::new();
        store.store(
            "iri://task/checkpoint",
            "write report to workspace",
            None,
            0.9,
        );

        let snapshot = store.snapshot_pending("iri://task/checkpoint");
        assert_eq!(snapshot.len(), 1);
        assert!(store.has_pending("iri://task/checkpoint"));

        let injected = store.take_pending("iri://task/checkpoint");
        assert_eq!(injected.len(), 1);
        assert_eq!(injected[0].content, "write report to workspace");
    }

    #[test]
    fn test_has_pending() {
        let store = SupplementaryInputStore::new();
        assert!(!store.has_pending("iri://task/empty"));

        store.store("iri://task/t", "x", None, 0.5);
        assert!(store.has_pending("iri://task/t"));

        store.take_pending("iri://task/t");
        assert!(!store.has_pending("iri://task/t"));
    }

    #[test]
    fn test_cleanup() {
        let store = SupplementaryInputStore::new();
        store.store("iri://task/t", "x", None, 0.5);
        assert!(store.has_pending("iri://task/t"));

        store.cleanup("iri://task/t");
        assert!(!store.has_pending("iri://task/t"));
    }

    #[test]
    fn test_pending_count() {
        let store = SupplementaryInputStore::new();
        assert_eq!(store.pending_count("iri://task/t"), 0);

        store.store("iri://task/t", "a", None, 0.5);
        store.store("iri://task/t", "b", None, 0.6);
        assert_eq!(store.pending_count("iri://task/t"), 2);

        store.take_pending("iri://task/t");
        assert_eq!(store.pending_count("iri://task/t"), 0);
    }
}
