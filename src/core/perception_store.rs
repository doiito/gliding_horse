use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Source type for perception entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerceptionSource {
    WorkspaceMonitor,
    BatchAgent,
    PerceptionEngine,
    Environment,
    System,
}

impl PerceptionSource {
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::WorkspaceMonitor => "📁 Workspace",
            Self::BatchAgent => "📊 Batch",
            Self::PerceptionEngine => "⚠️ Alert",
            Self::Environment => "🔌 Environment",
            Self::System => "ℹ️ System",
        }
    }
}

/// A single perception entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionEntry {
    pub id: String,
    pub source: PerceptionSource,
    pub content: String,
    pub detail_iri: Option<String>,
    pub priority: u8, // 0-9, 9=highest
    pub timestamp: DateTime<Utc>,
    pub consumed: bool,
}

impl PerceptionEntry {
    pub fn new(source: PerceptionSource, content: impl Into<String>) -> Self {
        Self {
            id: format!("pe_{}", uuid::Uuid::new_v4().hyphenated()),
            source,
            content: content.into(),
            detail_iri: None,
            priority: 5,
            timestamp: Utc::now(),
            consumed: false,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.clamp(0, 9);
        self
    }

    pub fn with_detail_iri(mut self, iri: String) -> Self {
        self.detail_iri = Some(iri);
        self
    }

    /// Render as a single line of text for the perception area
    pub fn render(&self) -> String {
        let prefix = self.source.prefix();
        if let Some(ref iri) = self.detail_iri {
            format!("[{}] {} | details: {}", prefix, self.content, iri)
        } else {
            format!("[{}] {}", prefix, self.content)
        }
    }
}

/// Perception content store
///
/// Similar to SupplementaryInputStore, but targets system-level proactive perception rather than task-level supplementary input.
///
/// Differences:
///   - PerceptionStore: producers = system components (WorkspaceMonitor/BatchAgent/PerceptionEngine)
///   - SupplementaryInputStore: producer = SA (task-level supplementary input)
///   - PerceptionStore content is injected during exec() initial assembly (beginning of messages)
///   - SupplementaryInputStore is injected per CycleStart round (middle of messages)
///
/// # Lifecycle
/// 1. System components call `store(task_iri, entry)` to write perception data
/// 2. AgentRunner calls `take_perception_text()` during exec() initial assembly
/// 3. Text is injected as a `role: "system"` message after messages[0]
/// 4. `cleanup(task_iri)` is called when the task completes
pub struct PerceptionStore {
    /// task_iri → perception entry list
    pending: Arc<Mutex<HashMap<String, Vec<PerceptionEntry>>>>,
    /// Global perception entries (not bound to a specific task), visible to all tasks
    global: Arc<Mutex<Vec<PerceptionEntry>>>,
    /// Dedup cache: source+content → last write time, used for 60-second dedup
    dedup_cache: Arc<Mutex<HashMap<(PerceptionSource, String), DateTime<Utc>>>>,
}

impl PerceptionStore {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
            global: Arc::new(Mutex::new(Vec::new())),
            dedup_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Producer write: add a perception entry for a specific task
    /// Auto-dedup: same source+content within 60 seconds is not written again
    pub fn store(&self, task_iri: &str, entry: PerceptionEntry) {
        let dedup_key = (entry.source, entry.content.clone());
        {
            let mut dedup = self.dedup_cache.lock().expect("dedup_cache lock poisoned");
            let now = Utc::now();
            if let Some(last) = dedup.get(&dedup_key) {
                if now.signed_duration_since(*last).num_seconds() < 60 {
                    return; // dedup
                }
            }
            dedup.insert(dedup_key, now);
        }

        let mut map = self.pending.lock().expect("PerceptionStore lock poisoned");
        map.entry(task_iri.to_string()).or_default().push(entry);
    }

    /// Producer write: add a global perception entry (visible to all tasks)
    pub fn store_global(&self, entry: PerceptionEntry) {
        let dedup_key = (entry.source, entry.content.clone());
        {
            let mut dedup = self.dedup_cache.lock().expect("dedup_cache lock poisoned");
            let now = Utc::now();
            if let Some(last) = dedup.get(&dedup_key) {
                if now.signed_duration_since(*last).num_seconds() < 60 {
                    return;
                }
            }
            dedup.insert(dedup_key, now);
        }

        let mut global = self
            .global
            .lock()
            .expect("PerceptionStore global lock poisoned");
        global.push(entry);
    }

    /// Clear all global perception entries (called on topic shift to prevent cross-task context pollution)
    pub fn clear_global(&self) {
        let mut global = self
            .global
            .lock()
            .expect("PerceptionStore global lock poisoned");
        global.clear();
        let mut dedup = self.dedup_cache.lock().expect("dedup_cache lock poisoned");
        dedup.retain(|(source, _), _| *source != PerceptionSource::WorkspaceMonitor);
    }

    /// Consumer pull: get all unconsumed perception entries (global + task level), merged into text
    ///
    /// Returns formatted perception text, or empty string if nothing new.
    /// Consumed entries are marked consumed but retained in the list for audit.
    pub fn take_perception_text(&self, task_iri: &str) -> String {
        let mut entries: Vec<PerceptionEntry> = Vec::new();

        // 1. Get global unconsumed entries
        {
            let mut global = self
                .global
                .lock()
                .expect("PerceptionStore global lock poisoned");
            for entry in global.iter_mut() {
                if !entry.consumed {
                    entry.consumed = true;
                    entries.push(entry.clone());
                }
            }
        }

        // 2. Get task-level unconsumed entries
        {
            let mut map = self.pending.lock().expect("PerceptionStore lock poisoned");
            if let Some(task_entries) = map.get_mut(task_iri) {
                for entry in task_entries.iter_mut() {
                    if !entry.consumed {
                        entry.consumed = true;
                        entries.push(entry.clone());
                    }
                }
            }
        }

        if entries.is_empty() {
            return String::new();
        }

        // Sort by priority, highest first
        entries.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.timestamp.cmp(&b.timestamp))
        });

        // Limit to 10 entries max to prevent bloat
        if entries.len() > 10 {
            entries.truncate(10);
        }

        let lines: Vec<String> = entries.iter().map(|e| e.render()).collect();
        lines.join("\n")
    }

    /// Peek at global unconsumed entries for a specific source (non-consuming read).
    /// Used by ProactiveEngine to check for workspace file events during task analysis
    /// without consuming the entries (they remain available for agent context injection).
    pub fn peek_unconsumed_for_source(&self, source: PerceptionSource) -> Vec<String> {
        let global = self
            .global
            .lock()
            .expect("PerceptionStore global lock poisoned");
        global
            .iter()
            .filter(|e| e.source == source && !e.consumed)
            .map(|e| e.render())
            .collect()
    }

    /// Check whether a given task has unconsumed perception entries
    pub fn has_new(&self, task_iri: &str) -> bool {
        // Check global
        {
            let global = self
                .global
                .lock()
                .expect("PerceptionStore global lock poisoned");
            if global.iter().any(|e| !e.consumed) {
                return true;
            }
        }
        // Check task-level
        {
            let map = self.pending.lock().expect("PerceptionStore lock poisoned");
            if let Some(entries) = map.get(task_iri) {
                if entries.iter().any(|e| !e.consumed) {
                    return true;
                }
            }
        }
        false
    }

    /// Clean up data for a completed task
    pub fn cleanup(&self, task_iri: &str) {
        let mut map = self.pending.lock().expect("PerceptionStore lock poisoned");
        map.remove(task_iri);
    }

    /// Evict expired dedup cache entries (older than 120 seconds)
    pub fn evict_dedup_cache(&self) {
        let mut dedup = self.dedup_cache.lock().expect("dedup_cache lock poisoned");
        let now = Utc::now();
        dedup.retain(|_, last| now.signed_duration_since(*last).num_seconds() < 120);
    }
}

impl Default for PerceptionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for PerceptionStore {
    fn clone(&self) -> Self {
        Self {
            pending: self.pending.clone(),
            global: self.global.clone(),
            dedup_cache: self.dedup_cache.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_store_empty() {
        let store = PerceptionStore::new();
        let text = store.take_perception_text("iri://task/test1");
        assert!(text.is_empty(), "New store should have no entries");
    }

    #[test]
    fn test_store_and_take() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/test1",
            PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, "3 stale files detected"),
        );
        store.store(
            "iri://task/test1",
            PerceptionEntry::new(
                PerceptionSource::BatchAgent,
                "Intent shift detected: db→cache",
            ),
        );

        let text = store.take_perception_text("iri://task/test1");
        assert!(!text.is_empty(), "Should have perception text");
        assert!(
            text.contains("Workspace"),
            "Should contain WorkspaceMonitor prefix"
        );
        assert!(text.contains("Batch"), "Should contain BatchAgent prefix");
    }

    #[test]
    fn test_store_global() {
        let store = PerceptionStore::new();
        store.store_global(PerceptionEntry::new(
            PerceptionSource::System,
            "System initialized",
        ));

        // Global entries visible to any task
        let text1 = store.take_perception_text("iri://task/a");
        assert!(
            !text1.is_empty(),
            "Global entry should be visible to task_a"
        );
        assert!(text1.contains("System"), "Should contain System prefix");

        let text2 = store.take_perception_text("iri://task/b");
        assert!(
            text2.is_empty(),
            "Global entry should be consumed after first take"
        );
    }

    #[test]
    fn test_dedup() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/test1",
            PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, "dedup test"),
        );
        store.store(
            "iri://task/test1",
            PerceptionEntry::new(
                PerceptionSource::WorkspaceMonitor,
                "dedup test", // duplicate
            ),
        );

        let text = store.take_perception_text("iri://task/test1");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines.len(),
            1,
            "Duplicate entries should be dedup'd to 1 line"
        );
    }

    #[test]
    fn test_priority_order() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, "low priority")
                .with_priority(1),
        );
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::PerceptionEngine, "high priority alert!")
                .with_priority(9),
        );

        let text = store.take_perception_text("iri://task/t");
        // High priority should come first
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].contains("Alert"),
            "High priority alert should come first"
        );
    }

    #[test]
    fn test_max_10_entries() {
        let store = PerceptionStore::new();
        for i in 0..15 {
            store.store(
                "iri://task/t",
                PerceptionEntry::new(PerceptionSource::System, format!("entry {}", i))
                    .with_priority(1),
            );
        }

        let text = store.take_perception_text("iri://task/t");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 10, "Should max at 10 perception entries");
    }

    #[test]
    fn test_cleanup() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::System, "test"),
        );
        store.cleanup("iri://task/t");

        let text = store.take_perception_text("iri://task/t");
        assert!(text.is_empty(), "Should be empty after cleanup");
    }

    #[test]
    fn test_has_new() {
        let store = PerceptionStore::new();
        assert!(!store.has_new("iri://task/t"));

        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::System, "new entry"),
        );
        assert!(store.has_new("iri://task/t"));

        store.take_perception_text("iri://task/t");
        assert!(!store.has_new("iri://task/t"));
    }

    #[test]
    fn test_perception_entry_render() {
        let entry = PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, "file changed")
            .with_detail_iri("iri://detail/1".to_string());
        let text = entry.render();
        assert!(text.contains("Workspace"));
        assert!(text.contains("file changed"));
        assert!(text.contains("iri://detail/1"));
    }

    #[test]
    fn test_store_different_tasks_same_source_content_dedup() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/a",
            PerceptionEntry::new(PerceptionSource::System, "same content"),
        );
        // Same source+content to different task — dedup cache IS global so this is blocked
        store.store(
            "iri://task/b",
            PerceptionEntry::new(PerceptionSource::System, "same content"),
        );

        let text_a = store.take_perception_text("iri://task/a");
        assert!(!text_a.is_empty(), "task_a should have perception");
        let text_b = store.take_perception_text("iri://task/b");
        // task_b's store was dedup'd — so has_new should still be false
        assert!(
            text_b.is_empty(),
            "task_b should be empty because store was dedup'd"
        );
    }

    #[test]
    fn test_same_content_different_sources_allowed_across_tasks() {
        let store = PerceptionStore::new();
        // Same content but from different sources → NOT dedup'd
        store.store(
            "iri://task/a",
            PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, "file changed"),
        );
        store.store(
            "iri://task/b",
            PerceptionEntry::new(PerceptionSource::PerceptionEngine, "file changed"),
        );

        let text_a = store.take_perception_text("iri://task/a");
        assert!(!text_a.is_empty(), "task_a should have entry");
        let text_b = store.take_perception_text("iri://task/b");
        assert!(
            !text_b.is_empty(),
            "task_b should have entry (different source)"
        );
    }

    #[test]
    fn test_global_not_duplicated_by_task_store() {
        let store = PerceptionStore::new();
        store.store_global(PerceptionEntry::new(PerceptionSource::System, "global msg"));

        // global + task store with same content — should keep both
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, "task specific"),
        );

        let text = store.take_perception_text("iri://task/t");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "Global + task entries should both appear");
    }

    #[test]
    fn test_take_empty_after_consumed() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::System, "msg"),
        );

        let first = store.take_perception_text("iri://task/t");
        assert!(!first.is_empty(), "First take should have content");
        assert!(
            !store.has_new("iri://task/t"),
            "has_new should be false after take"
        );

        let second = store.take_perception_text("iri://task/t");
        assert!(second.is_empty(), "Second take should be empty");
    }

    #[test]
    fn test_has_new_global_only() {
        let store = PerceptionStore::new();
        assert!(!store.has_new("iri://task/x"), "No entries initially");
        store.store_global(PerceptionEntry::new(
            PerceptionSource::Environment,
            "env event",
        ));
        assert!(
            store.has_new("iri://task/x"),
            "Should detect new global entry"
        );
        store.take_perception_text("iri://task/x");
        assert!(
            !store.has_new("iri://task/x"),
            "Should be consumed after take"
        );
    }

    #[test]
    fn test_priority_clamping() {
        let entry = PerceptionEntry::new(PerceptionSource::System, "test").with_priority(99);
        assert_eq!(entry.priority, 9, "Priority should clamp to 9");

        let entry = PerceptionEntry::new(PerceptionSource::System, "test").with_priority(0);
        assert_eq!(entry.priority, 0, "Priority 0 should stay 0");

        let entry = PerceptionEntry::new(PerceptionSource::System, "test").with_priority(5);
        assert_eq!(entry.priority, 5, "Priority 5 should stay 5");
    }

    #[test]
    fn test_dedup_cross_task_global() {
        let store = PerceptionStore::new();
        // Same source+content to global → should dedup
        store.store_global(PerceptionEntry::new(PerceptionSource::System, "dedup me"));
        store.store_global(PerceptionEntry::new(PerceptionSource::System, "dedup me"));

        let text = store.take_perception_text("iri://task/t");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1, "Global dedup should work");
    }

    #[test]
    fn test_different_sources_no_dedup() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, "same text"),
        );
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::BatchAgent, "same text"),
        );

        let text = store.take_perception_text("iri://task/t");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "Different sources with same content should NOT dedup"
        );
    }

    #[test]
    fn test_dedup_cache_prevents_immediate_repeat() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::System, "repeat"),
        );
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::System, "repeat"),
        );

        let text = store.take_perception_text("iri://task/t");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1, "Immediate repeat should be dedup'd");
    }

    #[test]
    fn test_cleanup_removes_only_target_task() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/a",
            PerceptionEntry::new(PerceptionSource::System, "a_data"),
        );
        store.store(
            "iri://task/b",
            PerceptionEntry::new(PerceptionSource::System, "b_data"),
        );

        store.cleanup("iri://task/a");

        // task_a should be gone
        let text_a = store.take_perception_text("iri://task/a");
        assert!(text_a.is_empty(), "Cleaned task should have no entries");

        // task_b should still have its entry (not consumed)
        assert!(
            store.has_new("iri://task/b"),
            "Task b should still have entries"
        );
        let text_b = store.take_perception_text("iri://task/b");
        assert!(!text_b.is_empty(), "Task b should have perception");
    }

    #[test]
    fn test_evict_dedup_cache_removes_old() {
        let store = PerceptionStore::new();
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::System, "old"),
        );

        // Manually set the dedup entry to be old (>120s)
        {
            let mut dedup = store.dedup_cache.lock().unwrap();
            dedup.insert(
                (PerceptionSource::System, "old".to_string()),
                Utc::now() - chrono::Duration::seconds(130),
            );
        }

        store.evict_dedup_cache();
        {
            let dedup = store.dedup_cache.lock().unwrap();
            assert!(dedup.is_empty(), "Evict should remove expired entries");
        }

        // After eviction the dedup cache is clean, but the original entry is still in pending.
        // Re-storing the same content adds a second entry (dedup check passes since cache was cleared).
        // Then take returns both entries.
        store.store(
            "iri://task/t",
            PerceptionEntry::new(PerceptionSource::System, "old"),
        );
        let text = store.take_perception_text("iri://task/t");
        let lines: Vec<&str> = text.lines().collect();
        // Two entries: original (step 1) + re-stored (after eviction)
        assert_eq!(
            lines.len(),
            2,
            "After eviction+restore, both entries should be present"
        );
    }

    #[test]
    fn test_perception_entry_default_id_is_unique() {
        let e1 = PerceptionEntry::new(PerceptionSource::System, "a");
        let e2 = PerceptionEntry::new(PerceptionSource::System, "a");
        assert_ne!(e1.id, e2.id, "Each entry should have a unique id");
    }

    #[test]
    fn test_entry_render_no_detail_iri() {
        let entry = PerceptionEntry::new(PerceptionSource::BatchAgent, "simple message");
        let text = entry.render();
        assert!(text.contains("Batch"), "Should contain Batch prefix");
        assert!(text.contains("simple message"), "Should contain message");
        assert!(
            !text.contains("details:"),
            "Should NOT contain details: without detail_iri"
        );
    }

    #[test]
    fn test_store_with_high_priority_comes_first() {
        let store = PerceptionStore::new();
        // 3 entries with different priorities
        store.store_global(PerceptionEntry::new(PerceptionSource::System, "low").with_priority(1));
        store.store_global(
            PerceptionEntry::new(PerceptionSource::PerceptionEngine, "critical!").with_priority(9),
        );
        store.store_global(
            PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, "mid").with_priority(5),
        );

        let text = store.take_perception_text("iri://task/t");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "All 3 entries should appear");
        // First line should be the high-priority one (Alert prefix)
        assert!(
            lines[0].contains("Alert"),
            "Highest priority should come first: got {}",
            lines[0]
        );
        // Last line should be the low priority
        assert!(
            lines[2].contains("Workspace") || lines[2].contains("System"),
            "Lower priority should be last"
        );
    }

    #[test]
    fn test_take_perception_respects_10_limit_across_global_and_task() {
        let store = PerceptionStore::new();
        for i in 0..8 {
            store.store_global(PerceptionEntry::new(
                PerceptionSource::System,
                format!("global_{}", i),
            ));
        }
        for i in 0..8 {
            store.store(
                "iri://task/t",
                PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, format!("task_{}", i)),
            );
        }
        let text = store.take_perception_text("iri://task/t");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines.len(),
            10,
            "Should cap at 10 even with 16 total entries"
        );
    }
}
