use chrono::Utc;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::graph_backend::{SnapshotBackend, SnapshotNode};
use crate::memory::l0_store::{L0Entry, L0Store, MesiState};
use crate::skill_graph::types::{Hyperedge, KnowledgeFragment, MOCNode, SkillGraphNode};
use crate::snapshots::types::{GraphDiff, GraphMutation, GraphSnapshot, SnapshotMeta};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TimelineMutationRecord {
    sequence: u64,
    mutation: GraphMutation,
}

/// TimelineStore — versioned snapshots of the skill graph.
///
/// # Storage Strategy
///
/// Uses **chunked snapshot + mutation log**:
/// - **Full snapshots** are taken every N mutations (configurable, default=100).
///   Stored as a complete clone of the graph at that point.
/// - **Incremental mutations** between full snapshots are stored separately.
///   To reconstruct, replay mutations on top of the last full snapshot.
/// - **Copy-on-write**: snapshots clone the graph state only when explicitly
///   triggered or on the configured interval.
///
/// # GC Policy
///
/// Keeps the last `max_full_snapshots` full snapshots and all incremental
/// mutations since the earliest retained snapshot. Older snapshots are dropped.
pub struct TimelineStore {
    /// All snapshots (full + incremental markers)
    snapshots: RwLock<Vec<SnapshotMeta>>,
    /// Mutation log since last full snapshot
    mutations_since_last_full: RwLock<Vec<TimelineMutationRecord>>,
    /// Mutation counter (global, monotonically increasing)
    mutation_count: AtomicU64,
    /// Snapshot frequency (full snapshot every N mutations)
    snapshot_frequency: u64,
    /// Maximum number of full snapshots to retain
    max_full_snapshots: usize,
    l0_store: Option<Arc<L0Store>>,
}

impl Default for TimelineStore {
    fn default() -> Self {
        Self::new(100, 10)
    }
}

impl TimelineStore {
    /// Create a new TimelineStore.
    ///
    /// `snapshot_frequency`: take a full snapshot every N mutations (default 100).
    /// `max_full_snapshots`: maximum full snapshots to retain (default 10).
    pub fn new(snapshot_frequency: u64, max_full_snapshots: usize) -> Self {
        Self {
            snapshots: RwLock::new(Vec::new()),
            mutations_since_last_full: RwLock::new(Vec::new()),
            mutation_count: AtomicU64::new(0),
            snapshot_frequency,
            max_full_snapshots,
            l0_store: None,
        }
    }

    pub fn with_l0_store(mut self, l0_store: Arc<L0Store>) -> Self {
        self.l0_store = Some(l0_store);
        self
    }

    fn snapshot_key(snapshot_id: &str) -> String {
        format!("iri://timeline/snapshot/{}", snapshot_id)
    }

    fn mutation_key(sequence: u64) -> String {
        format!("iri://timeline/mutation/{sequence:020}")
    }

    fn persist_mutation(&self, record: &TimelineMutationRecord) {
        let Some(l0_store) = &self.l0_store else {
            return;
        };
        let content = match serde_json::to_string(record) {
            Ok(content) => content,
            Err(error) => {
                tracing::warn!(%error, "Failed to serialize timeline mutation");
                return;
            }
        };
        let now = Utc::now();
        let entry = L0Entry {
            iri: Self::mutation_key(record.sequence),
            content,
            importance: 0.3,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            tags: vec!["timeline".to_string(), "mutation".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: Some("system:timeline".to_string()),
            jsonld_context: None,
            jsonld_types: vec!["timeline:GraphMutation".to_string()],
            hyperspace_point_id: None,
        };
        if let Err(error) = l0_store.store_entry(&entry) {
            tracing::warn!(%error, sequence = record.sequence, "Failed to persist timeline mutation");
        }
    }

    fn persist_snapshot(&self, snapshot: &GraphSnapshot) {
        let Some(l0_store) = &self.l0_store else {
            return;
        };
        let content = match serde_json::to_string(snapshot) {
            Ok(content) => content,
            Err(error) => {
                tracing::warn!(%error, "Failed to serialize timeline snapshot");
                return;
            }
        };
        let now = Utc::now();
        let entry = L0Entry {
            iri: Self::snapshot_key(&snapshot.snapshot_id),
            content,
            importance: 0.4,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            tags: vec!["timeline".to_string(), "snapshot".to_string()],
            metadata: serde_json::Map::new(),
            mesi_state: MesiState::Shared,
            content_hash: String::new(),
            named_graph: Some("system:timeline".to_string()),
            jsonld_context: None,
            jsonld_types: vec!["timeline:GraphSnapshot".to_string()],
            hyperspace_point_id: None,
        };
        if let Err(error) = l0_store.store_entry(&entry) {
            tracing::warn!(%error, "Failed to persist timeline snapshot");
        }
    }

    /// Rebuild in-memory metadata from durable full snapshots.
    pub fn load_persisted(&self) -> Result<usize, crate::CoreError> {
        let Some(l0_store) = &self.l0_store else {
            return Ok(0);
        };
        let entries = l0_store.scan_iri_prefix("iri://timeline/snapshot/", usize::MAX)?;
        let mut snapshots = Vec::new();
        let mut max_count = 0;
        for entry in entries {
            if let Ok(snapshot) = serde_json::from_str::<GraphSnapshot>(&entry.content) {
                max_count = max_count.max(snapshot.mutation_count);
                snapshots.push(SnapshotMeta {
                    snapshot_id: snapshot.snapshot_id,
                    timestamp: snapshot.timestamp,
                    label: snapshot.label,
                    skill_count: snapshot.skills.len(),
                    hyperedge_count: snapshot.hyperedges.len(),
                    mutation_count: snapshot.mutation_count,
                    parent_snapshot_id: snapshot.parent_snapshot_id,
                    size_bytes: entry.content.len() as u64,
                });
            }
        }
        snapshots.sort_by_key(|meta| meta.timestamp);
        let mutation_entries = l0_store.scan_iri_prefix("iri://timeline/mutation/", usize::MAX)?;
        let mut mutations: Vec<TimelineMutationRecord> = mutation_entries
            .into_iter()
            .filter_map(|entry| serde_json::from_str(&entry.content).ok())
            .filter(|record: &TimelineMutationRecord| record.sequence > max_count)
            .collect();
        mutations.sort_by_key(|record| record.sequence);
        if let Some(record) = mutations.last() {
            max_count = max_count.max(record.sequence);
        }
        let count = snapshots.len();
        *self.snapshots.write() = snapshots;
        *self.mutations_since_last_full.write() = mutations;
        self.mutation_count.store(max_count, Ordering::SeqCst);
        Ok(count)
    }

    pub fn load_snapshot(
        &self,
        snapshot_id: &str,
    ) -> Result<Option<GraphSnapshot>, crate::CoreError> {
        let Some(l0_store) = &self.l0_store else {
            return Ok(None);
        };
        let entry = l0_store.retrieve(&Self::snapshot_key(snapshot_id))?;
        entry
            .map(|entry| {
                serde_json::from_str(&entry.content).map_err(|error| {
                    crate::CoreError::StorageError {
                        message: format!("Failed to deserialize timeline snapshot: {}", error),
                    }
                })
            })
            .transpose()
    }

    /// Reconstruct the newest durable graph state by applying the mutation
    /// records after the latest full snapshot.  A snapshot is required because
    /// fragments and any pre-snapshot history are intentionally compacted.
    pub fn reconstruct_latest(&self) -> Result<Option<GraphSnapshot>, crate::CoreError> {
        let Some(meta) = self
            .snapshots
            .read()
            .iter()
            .max_by_key(|meta| meta.mutation_count)
            .cloned()
        else {
            return Ok(None);
        };
        let Some(mut snapshot) = self.load_snapshot(&meta.snapshot_id)? else {
            return Ok(None);
        };
        let records = self.mutations_since_last_full.read().clone();
        let base_mutation_count = snapshot.mutation_count;
        for record in records
            .iter()
            .filter(|record| record.sequence > base_mutation_count)
        {
            Self::apply_mutation(&mut snapshot, &record.mutation);
            snapshot.mutation_count = record.sequence;
        }
        Ok(Some(snapshot))
    }

    fn apply_mutation(snapshot: &mut GraphSnapshot, mutation: &GraphMutation) {
        match mutation {
            GraphMutation::SkillRegistered(skill) => {
                snapshot
                    .skills
                    .retain(|candidate| candidate.skill_iri != skill.skill_iri);
                snapshot.skills.push(skill.clone());
            }
            GraphMutation::SkillUpdated { new, .. } => {
                snapshot
                    .skills
                    .retain(|candidate| candidate.skill_iri != new.skill_iri);
                snapshot.skills.push(new.clone());
            }
            GraphMutation::SkillRemoved(skill) => snapshot
                .skills
                .retain(|candidate| candidate.skill_iri != skill.skill_iri),
            GraphMutation::LinkAdded { source_after, .. } => {
                snapshot
                    .skills
                    .retain(|candidate| candidate.skill_iri != source_after.skill_iri);
                snapshot.skills.push(source_after.clone());
            }
            GraphMutation::LinkRemoved {
                source,
                target,
                link_type,
            } => {
                if let Some(skill) = snapshot
                    .skills
                    .iter_mut()
                    .find(|skill| skill.skill_iri == *source)
                {
                    skill
                        .links
                        .retain(|link| link.target_iri != *target || link.link_type != *link_type);
                }
            }
            GraphMutation::HyperedgeAdded(edge) => {
                snapshot
                    .hyperedges
                    .retain(|candidate| candidate.hyperedge_id != edge.hyperedge_id);
                snapshot.hyperedges.push(edge.clone());
            }
            GraphMutation::HyperedgeRemoved(id) => snapshot
                .hyperedges
                .retain(|candidate| candidate.hyperedge_id != *id),
            GraphMutation::MOCAdded(moc) => {
                snapshot
                    .mocs
                    .retain(|candidate| candidate.moc_iri != moc.moc_iri);
                snapshot.mocs.push(moc.clone());
            }
            GraphMutation::MOCRemoved(id) => {
                snapshot.mocs.retain(|candidate| candidate.moc_iri != *id)
            }
        }
    }

    /// Record a mutation. If the mutation count exceeds `snapshot_frequency`,
    /// automatically triggers a full snapshot (optional, requires backend ref).
    pub fn record_mutation(&self, mutation: GraphMutation, backend: Option<&dyn SnapshotBackend>) {
        let count = self.mutation_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Always store the mutation
        let record = TimelineMutationRecord {
            sequence: count,
            mutation,
        };
        self.persist_mutation(&record);
        self.mutations_since_last_full.write().push(record);

        // Check if we need a full snapshot
        if count % self.snapshot_frequency == 0 {
            if let Some(backend) = backend {
                self.create_snapshot(backend, &format!("auto-{}", count));
            }
        }
    }

    /// Create an explicit full snapshot of the current graph state.
    /// Returns the snapshot_id.
    pub fn create_snapshot(&self, backend: &dyn SnapshotBackend, label: &str) -> String {
        let nodes = backend.snapshot();

        // Deserialize SnapshotNodes into typed collections
        let mut skills = Vec::new();
        let mut hyperedges = Vec::new();
        let mut mocs = Vec::new();
        let mut fragments = Vec::new();

        for node in &nodes {
            match node.node_type.as_str() {
                "SkillGraphNode" => {
                    if let Ok(skill) = serde_json::from_value::<SkillGraphNode>(node.data.clone()) {
                        skills.push(skill);
                    }
                }
                "Hyperedge" => {
                    if let Ok(he) = serde_json::from_value::<Hyperedge>(node.data.clone()) {
                        hyperedges.push(he);
                    }
                }
                "MOCNode" => {
                    if let Ok(moc) = serde_json::from_value::<MOCNode>(node.data.clone()) {
                        mocs.push(moc);
                    }
                }
                "KnowledgeFragment" => {
                    if let Ok(frag) = serde_json::from_value::<KnowledgeFragment>(node.data.clone())
                    {
                        fragments.push(frag);
                    }
                }
                _ => {}
            }
        }

        let snapshot_id = format!("snap_{}", Uuid::new_v4().hyphenated());
        let parent_id = self.snapshots.read().last().map(|m| m.snapshot_id.clone());

        let meta = SnapshotMeta {
            snapshot_id: snapshot_id.clone(),
            timestamp: Utc::now(),
            label: label.to_string(),
            skill_count: skills.len(),
            hyperedge_count: hyperedges.len(),
            mutation_count: self.mutation_count.load(Ordering::SeqCst),
            parent_snapshot_id: parent_id,
            size_bytes: 0, // computed on serialization
        };

        let snapshot = GraphSnapshot {
            snapshot_id: snapshot_id.clone(),
            timestamp: Utc::now(),
            label: label.to_string(),
            skills,
            hyperedges,
            mocs,
            fragments: fragments,
            parent_snapshot_id: meta.parent_snapshot_id.clone(),
            mutation_count: meta.mutation_count,
            compressed: false,
        };
        self.persist_snapshot(&snapshot);

        info!(
            "TimelineStore: created snapshot {} ({} skills, {} hyperedges, label={})",
            snapshot_id,
            snapshot.skills.len(),
            snapshot.hyperedges.len(),
            label
        );

        // Store metadata
        self.snapshots.write().push(meta);

        // Clear incremental mutations
        self.mutations_since_last_full.write().clear();
        if let Some(l0_store) = &self.l0_store {
            for record in l0_store
                .scan_iri_prefix("iri://timeline/mutation/", usize::MAX)
                .unwrap_or_default()
            {
                if let Ok(record) = serde_json::from_str::<TimelineMutationRecord>(&record.content)
                {
                    if record.sequence <= snapshot.mutation_count {
                        if let Err(error) = l0_store.delete(&Self::mutation_key(record.sequence)) {
                            tracing::warn!(%error, sequence = record.sequence, "Failed to compact timeline mutation");
                        }
                    }
                }
            }
        }

        // GC old snapshots
        self.gc();

        snapshot_id
    }

    /// Push a full snapshot directly (for loading from persistence).
    pub fn push_snapshot(&self, snapshot: GraphSnapshot) {
        let meta = SnapshotMeta {
            snapshot_id: snapshot.snapshot_id,
            timestamp: snapshot.timestamp,
            label: snapshot.label,
            skill_count: snapshot.skills.len(),
            hyperedge_count: snapshot.hyperedges.len(),
            mutation_count: snapshot.mutation_count,
            parent_snapshot_id: snapshot.parent_snapshot_id,
            size_bytes: 0,
        };
        self.snapshots.write().push(meta);
    }

    /// List all snapshots (most recent first).
    pub fn list_snapshots(&self) -> Vec<SnapshotMeta> {
        let mut metas = self.snapshots.read().clone();
        metas.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        metas
    }

    /// Get snapshot metadata by ID.
    pub fn get_snapshot_meta(&self, snapshot_id: &str) -> Option<SnapshotMeta> {
        self.snapshots
            .read()
            .iter()
            .find(|m| m.snapshot_id == snapshot_id)
            .cloned()
    }

    /// Rollback the backend to a previous snapshot.
    /// Clears current state and restores the snapshot's data via the backend.
    pub fn rollback(
        &self,
        target_snapshot: &GraphSnapshot,
        backend: &dyn SnapshotBackend,
    ) -> Result<(), String> {
        info!(
            "TimelineStore: rolling back to snapshot {} ({} skills, {} hyperedges)",
            target_snapshot.snapshot_id,
            target_snapshot.skills.len(),
            target_snapshot.hyperedges.len()
        );

        // Convert GraphSnapshot to Vec<SnapshotNode>
        let mut nodes: Vec<SnapshotNode> = Vec::new();
        for skill in &target_snapshot.skills {
            if let Ok(data) = serde_json::to_value(skill) {
                nodes.push(SnapshotNode::new(&skill.skill_iri, data, "SkillGraphNode"));
            }
        }
        for he in &target_snapshot.hyperedges {
            if let Ok(data) = serde_json::to_value(he) {
                nodes.push(SnapshotNode::new(&he.hyperedge_id, data, "Hyperedge"));
            }
        }
        for moc in &target_snapshot.mocs {
            if let Ok(data) = serde_json::to_value(moc) {
                nodes.push(SnapshotNode::new(&moc.moc_iri, data, "MOCNode"));
            }
        }
        for frag in &target_snapshot.fragments {
            if let Ok(data) = serde_json::to_value(frag) {
                nodes.push(SnapshotNode::new(
                    &frag.fragment_iri,
                    data,
                    "KnowledgeFragment",
                ));
            }
        }

        // Capture the current state first.  `SnapshotBackend` is intentionally
        // generic and may span non-transactional stores, so this provides
        // best-effort compensation if target application fails after clear.
        let previous = backend.snapshot();

        // Clear existing state and apply snapshot
        backend.clear().map_err(|e| format!("clear: {}", e))?;
        if let Err(error) = backend.apply_snapshot(&nodes) {
            let recovery = backend
                .clear()
                .and_then(|_| backend.apply_snapshot(&previous));
            return match recovery {
                Ok(()) => Err(format!(
                    "apply_snapshot: {}; previous state restored by compensation",
                    error
                )),
                Err(recovery_error) => Err(format!(
                    "apply_snapshot: {}; compensation failed: {}",
                    error, recovery_error
                )),
            };
        }

        info!(
            "TimelineStore: rollback to {} complete",
            target_snapshot.snapshot_id
        );
        Ok(())
    }

    /// Compute the diff between two snapshots.
    /// If `from_id` is None, diffs from the first snapshot.
    pub fn diff(&self, from_id: Option<&str>, to_id: &str) -> Option<GraphDiff> {
        let snapshots = self.snapshots.read();
        let to_snap = snapshots.iter().find(|m| m.snapshot_id == to_id)?;

        let from_snap = from_id
            .and_then(|id| snapshots.iter().find(|m| m.snapshot_id == id))
            .or_else(|| snapshots.first());

        // If same snapshot, return empty diff
        if Some(to_id) == from_id {
            return Some(GraphDiff {
                from_snapshot_id: from_snap.map(|m| m.snapshot_id.clone()).unwrap_or_default(),
                to_snapshot_id: to_id.to_string(),
                from_timestamp: from_snap.map(|m| m.timestamp).unwrap_or_default(),
                to_timestamp: to_snap.timestamp,
                skills_added: Vec::new(),
                skills_removed: Vec::new(),
                skills_modified: Vec::new(),
                hyperedges_added: Vec::new(),
                hyperedges_removed: Vec::new(),
                mocs_added: Vec::new(),
                mocs_removed: Vec::new(),
                total_mutations: 0,
            });
        }

        // For a real diff, we need the actual snapshot data.
        // This returns metadata only; the caller must provide snapshots for full diff.
        Some(GraphDiff {
            from_snapshot_id: from_snap.map(|m| m.snapshot_id.clone()).unwrap_or_default(),
            to_snapshot_id: to_id.to_string(),
            from_timestamp: from_snap.map(|m| m.timestamp).unwrap_or_default(),
            to_timestamp: to_snap.timestamp,
            skills_added: Vec::new(),
            skills_removed: Vec::new(),
            skills_modified: Vec::new(),
            hyperedges_added: Vec::new(),
            hyperedges_removed: Vec::new(),
            mocs_added: Vec::new(),
            mocs_removed: Vec::new(),
            total_mutations: 0,
        })
    }

    /// Compute a full diff given the actual snapshot data.
    pub fn full_diff(from: &GraphSnapshot, to: &GraphSnapshot) -> GraphDiff {
        let from_skills: HashMap<&str, &SkillGraphNode> = from
            .skills
            .iter()
            .map(|s| (s.skill_iri.as_str(), s))
            .collect();
        let to_skills: HashMap<&str, &SkillGraphNode> = to
            .skills
            .iter()
            .map(|s| (s.skill_iri.as_str(), s))
            .collect();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        for (iri, skill) in &to_skills {
            match from_skills.get(iri) {
                None => added.push((*skill).clone()),
                Some(old) => {
                    if old.version != skill.version
                        || old.graph_meta.usage_count != skill.graph_meta.usage_count
                    {
                        modified.push(((*old).clone(), (*skill).clone()));
                    }
                }
            }
        }

        for (iri, skill) in &from_skills {
            if !to_skills.contains_key(iri) {
                removed.push((*skill).clone());
            }
        }

        let from_hyperedges: HashSet<&str> = from
            .hyperedges
            .iter()
            .map(|h| h.hyperedge_id.as_str())
            .collect();
        let to_hyperedges: HashSet<&str> = to
            .hyperedges
            .iter()
            .map(|h| h.hyperedge_id.as_str())
            .collect();

        let hyperedges_added: Vec<Hyperedge> = to
            .hyperedges
            .iter()
            .filter(|h| !from_hyperedges.contains(h.hyperedge_id.as_str()))
            .cloned()
            .collect();
        let hyperedges_removed: Vec<String> = from
            .hyperedges
            .iter()
            .filter(|h| !to_hyperedges.contains(h.hyperedge_id.as_str()))
            .map(|h| h.hyperedge_id.clone())
            .collect();

        let from_mocs: HashSet<&str> = from.mocs.iter().map(|m| m.moc_iri.as_str()).collect();
        let to_mocs: HashSet<&str> = to.mocs.iter().map(|m| m.moc_iri.as_str()).collect();

        let mocs_added: Vec<MOCNode> = to
            .mocs
            .iter()
            .filter(|m| !from_mocs.contains(m.moc_iri.as_str()))
            .cloned()
            .collect();
        let mocs_removed: Vec<String> = from
            .mocs
            .iter()
            .filter(|m| !to_mocs.contains(m.moc_iri.as_str()))
            .map(|m| m.moc_iri.clone())
            .collect();

        GraphDiff {
            from_snapshot_id: from.snapshot_id.clone(),
            to_snapshot_id: to.snapshot_id.clone(),
            from_timestamp: from.timestamp,
            to_timestamp: to.timestamp,
            skills_added: added,
            skills_removed: removed,
            skills_modified: modified,
            hyperedges_added,
            hyperedges_removed,
            mocs_added,
            mocs_removed,
            total_mutations: (to.mutation_count - from.mutation_count) as usize,
        }
    }

    // ── Internal ──

    /// Garbage collect old snapshots.
    fn gc(&self) {
        let mut snapshots = self.snapshots.write();
        let mut full_count = 0;

        // Count full snapshots (those with labels not starting with "incr-")
        for meta in snapshots.iter().rev() {
            if !meta.label.starts_with("incr-") {
                full_count += 1;
            }
        }

        if full_count > self.max_full_snapshots {
            let to_remove = full_count - self.max_full_snapshots;
            let mut removed = 0;
            snapshots.retain(|meta| {
                if removed < to_remove && !meta.label.starts_with("incr-") {
                    removed += 1;
                    false
                } else {
                    true
                }
            });
            debug!("TimelineStore GC: removed {} old snapshots", removed);
        }
    }

    // ── Stats ──

    pub fn total_mutations(&self) -> u64 {
        self.mutation_count.load(Ordering::SeqCst)
    }

    pub fn pending_mutations(&self) -> usize {
        self.mutations_since_last_full.read().len()
    }

    pub fn snapshot_count(&self) -> usize {
        self.snapshots.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_backend::SkillGraphSnapshotBackend;
    use crate::memory::l0_store::L0Store;
    use crate::skill_graph::graph_store::SkillGraphStore;
    use crate::skill_graph::types::SkillGraphNode;
    use std::sync::Arc;

    fn test_backend() -> SkillGraphSnapshotBackend {
        let store = Arc::new(SkillGraphStore::new());
        let s1 = SkillGraphNode::new("iri://skills/a", "Skill A", "Test A");
        let s2 = SkillGraphNode::new("iri://skills/b", "Skill B", "Test B");
        store.register_skill(s1).unwrap();
        store.register_skill(s2).unwrap();
        SkillGraphSnapshotBackend::new(store)
    }

    #[test]
    fn test_create_snapshot() {
        let backend = test_backend();
        let timeline = TimelineStore::new(100, 10);
        let id = timeline.create_snapshot(&backend, "test-snap");
        assert!(!id.is_empty());

        let metas = timeline.list_snapshots();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].skill_count, 2);
    }

    #[test]
    fn persisted_snapshot_can_be_loaded_after_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let backend = test_backend();
        let timeline = TimelineStore::new(100, 10).with_l0_store(l0.clone());
        let id = timeline.create_snapshot(&backend, "persistent");

        let rebuilt = TimelineStore::new(100, 10).with_l0_store(l0);
        assert_eq!(rebuilt.load_persisted().unwrap(), 1);
        let snapshot = rebuilt.load_snapshot(&id).unwrap().unwrap();
        assert_eq!(snapshot.skills.len(), 2);
    }

    #[test]
    fn persisted_mutation_reconstructs_after_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let backend = test_backend();
        let timeline = TimelineStore::new(100, 10).with_l0_store(l0.clone());
        timeline.create_snapshot(&backend, "base");
        timeline.record_mutation(
            GraphMutation::SkillRegistered(SkillGraphNode::new(
                "iri://skills/recovered",
                "Recovered",
                "from log",
            )),
            None,
        );

        let rebuilt = TimelineStore::new(100, 10).with_l0_store(l0);
        rebuilt.load_persisted().unwrap();
        let reconstructed = rebuilt.reconstruct_latest().unwrap().unwrap();
        assert!(reconstructed
            .skills
            .iter()
            .any(|skill| skill.skill_iri == "iri://skills/recovered"));
        assert_eq!(rebuilt.pending_mutations(), 1);
    }

    #[test]
    fn persisted_link_mutation_reconstructs_complete_source_state() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let timeline = Arc::new(TimelineStore::new(100, 10).with_l0_store(l0.clone()));
        let store = Arc::new(SkillGraphStore::new().with_timeline(timeline.clone()));
        store
            .register_skill(SkillGraphNode::new(
                "iri://skills/source",
                "Source",
                "source",
            ))
            .unwrap();
        store
            .register_skill(SkillGraphNode::new(
                "iri://skills/target",
                "Target",
                "target",
            ))
            .unwrap();
        let backend = SkillGraphSnapshotBackend::new(store.clone());
        timeline.create_snapshot(&backend, "base");

        store
            .add_link(
                "iri://skills/source",
                "iri://skills/target",
                crate::skill_graph::types::SkillLinkType::Prerequisite,
                crate::skill_graph::types::LinkStrength::Required,
                "must be completed first",
            )
            .unwrap();

        let rebuilt = TimelineStore::new(100, 10).with_l0_store(l0);
        rebuilt.load_persisted().unwrap();
        let reconstructed = rebuilt.reconstruct_latest().unwrap().unwrap();
        let source = reconstructed
            .skills
            .iter()
            .find(|skill| skill.skill_iri == "iri://skills/source")
            .unwrap();
        assert_eq!(source.links.len(), 1);
        assert_eq!(source.links[0].target_iri, "iri://skills/target");
        assert_eq!(
            source.links[0].link_type,
            crate::skill_graph::types::SkillLinkType::Prerequisite
        );
        assert_eq!(
            source.links[0].strength,
            crate::skill_graph::types::LinkStrength::Required
        );
        assert_eq!(source.links[0].description, "must be completed first");
    }

    #[test]
    fn test_record_mutation_auto_snapshot() {
        let backend = test_backend();
        // frequency=2 → every 2 mutations triggers auto snapshot
        let timeline = TimelineStore::new(2, 10);

        timeline.record_mutation(
            GraphMutation::SkillRegistered(SkillGraphNode::new("iri://skills/c", "C", "C")),
            Some(&backend),
        );
        assert_eq!(timeline.snapshot_count(), 0); // not yet (count=1)

        timeline.record_mutation(
            GraphMutation::SkillRegistered(SkillGraphNode::new("iri://skills/d", "D", "D")),
            Some(&backend),
        );
        assert_eq!(timeline.snapshot_count(), 1); // triggered at count=2
    }

    #[test]
    fn test_rollback() {
        let backend = test_backend();
        let timeline = TimelineStore::new(100, 10);
        let target_moc = crate::skill_graph::types::MOCNode::new(
            "iri://mocs/target",
            "Target",
            "target snapshot moc",
        );
        backend.store().register_moc(target_moc).unwrap();
        backend
            .store()
            .create_fragment(
                "iri://fragments/target",
                "iri://skills/a",
                "target failure",
                "target mitigation",
                None,
            )
            .unwrap();
        let snap = GraphSnapshot {
            snapshot_id: "rollback-target".to_string(),
            timestamp: Utc::now(),
            label: "pre-rollback".to_string(),
            skills: backend.store().list_all_skills(),
            hyperedges: backend.store().list_hyperedges(),
            mocs: backend.store().list_mocs(),
            fragments: backend.store().list_fragments(),
            parent_snapshot_id: None,
            mutation_count: 0,
            compressed: false,
        };

        backend
            .store()
            .register_skill(SkillGraphNode::new("iri://skills/c", "Skill C", "Test C"))
            .unwrap();
        backend
            .store()
            .register_moc(crate::skill_graph::types::MOCNode::new(
                "iri://mocs/current",
                "Current",
                "current-only moc",
            ))
            .unwrap();
        backend
            .store()
            .create_fragment(
                "iri://fragments/current",
                "iri://skills/b",
                "current failure",
                "current mitigation",
                None,
            )
            .unwrap();
        assert_eq!(backend.store().skill_count(), 3);

        timeline.rollback(&snap, &backend).unwrap();
        assert_eq!(backend.store().skill_count(), 2);
        assert_eq!(backend.store().list_mocs().len(), 1);
        assert_eq!(backend.store().list_mocs()[0].moc_iri, "iri://mocs/target");
        assert_eq!(backend.store().list_fragments().len(), 1);
        assert_eq!(
            backend.store().list_fragments()[0].fragment_iri,
            "iri://fragments/target"
        );
        // Restoring the full fragment must not add the failure mode a second time.
        assert_eq!(
            backend
                .store()
                .get_skill("iri://skills/a")
                .unwrap()
                .graph_meta
                .known_failure_modes
                .len(),
            1
        );
    }

    #[test]
    fn test_diff_tracking() {
        let timeline = TimelineStore::new(100, 10);
        timeline.record_mutation(
            GraphMutation::SkillRegistered(SkillGraphNode::new("iri://skills/a", "A", "A")),
            None,
        );
        assert_eq!(timeline.total_mutations(), 1);
        assert_eq!(timeline.pending_mutations(), 1);
    }

    #[test]
    fn test_gc() {
        let backend = test_backend();
        let timeline = TimelineStore::new(1, 2);

        for i in 0..5 {
            backend
                .store()
                .register_skill(SkillGraphNode::new(
                    &format!("iri://skills/gc-{}", i),
                    &format!("GC {}", i),
                    "",
                ))
                .unwrap();
            timeline.create_snapshot(&backend, &format!("snap-{}", i));
        }

        assert!(timeline.snapshot_count() <= 3);
    }

    #[test]
    fn test_full_diff() {
        let from = GraphSnapshot {
            snapshot_id: "snap-1".to_string(),
            timestamp: Utc::now(),
            label: "from".to_string(),
            skills: vec![
                SkillGraphNode::new("iri://skills/a", "A", "A"),
                SkillGraphNode::new("iri://skills/b", "B", "B"),
            ],
            hyperedges: Vec::new(),
            mocs: Vec::new(),
            fragments: Vec::new(),
            parent_snapshot_id: None,
            mutation_count: 5,
            compressed: false,
        };

        let mut to = from.clone();
        to.snapshot_id = "snap-2".to_string();
        to.skills
            .push(SkillGraphNode::new("iri://skills/c", "C", "C"));
        to.mutation_count = 8;

        let diff = TimelineStore::full_diff(&from, &to);
        assert_eq!(diff.skills_added.len(), 1);
        assert_eq!(diff.skills_added[0].skill_iri, "iri://skills/c");
        assert_eq!(diff.total_mutations, 3);
    }
}
