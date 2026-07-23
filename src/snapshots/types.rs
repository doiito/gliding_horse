use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::skill_graph::types::{
    Hyperedge, KnowledgeFragment, MOCNode, SkillGraphNode, SkillLinkType,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Timeline / Versioning Types
// ═══════════════════════════════════════════════════════════════════════════════

/// A single mutation to the skill graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphMutation {
    SkillRegistered(SkillGraphNode),
    SkillUpdated {
        old: SkillGraphNode,
        new: SkillGraphNode,
    },
    SkillRemoved(SkillGraphNode),
    LinkAdded {
        source: String,
        target: String,
        link_type: SkillLinkType,
        /// Complete source state after the link mutation.  The link fields
        /// alone are insufficient to reconstruct strength/description.
        source_after: SkillGraphNode,
    },
    LinkRemoved {
        source: String,
        target: String,
        link_type: SkillLinkType,
    },
    HyperedgeAdded(Hyperedge),
    HyperedgeRemoved(String), // hyperedge_id
    MOCAdded(MOCNode),
    MOCRemoved(String), // moc_iri
}

/// A complete point-in-time snapshot of the skill graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub snapshot_id: String,
    pub timestamp: DateTime<Utc>,
    pub label: String,
    pub skills: Vec<SkillGraphNode>,
    pub hyperedges: Vec<Hyperedge>,
    pub mocs: Vec<MOCNode>,
    pub fragments: Vec<KnowledgeFragment>,
    pub parent_snapshot_id: Option<String>,
    /// Cumulated mutation count at this snapshot
    pub mutation_count: u64,
    /// Optional compact binary representation for storage
    pub compressed: bool,
}

/// Lightweight metadata for snapshot listing (avoids deserializing full snapshot).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub snapshot_id: String,
    pub timestamp: DateTime<Utc>,
    pub label: String,
    pub skill_count: usize,
    pub hyperedge_count: usize,
    pub mutation_count: u64,
    pub parent_snapshot_id: Option<String>,
    pub size_bytes: u64,
}

/// Diff between two snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDiff {
    pub from_snapshot_id: String,
    pub to_snapshot_id: String,
    pub from_timestamp: DateTime<Utc>,
    pub to_timestamp: DateTime<Utc>,
    pub skills_added: Vec<SkillGraphNode>,
    pub skills_removed: Vec<SkillGraphNode>,
    pub skills_modified: Vec<(SkillGraphNode, SkillGraphNode)>, // (old, new)
    pub hyperedges_added: Vec<Hyperedge>,
    pub hyperedges_removed: Vec<String>,
    pub mocs_added: Vec<MOCNode>,
    pub mocs_removed: Vec<String>,
    pub total_mutations: usize,
}

impl GraphDiff {
    pub fn is_empty(&self) -> bool {
        self.skills_added.is_empty()
            && self.skills_removed.is_empty()
            && self.skills_modified.is_empty()
            && self.hyperedges_added.is_empty()
            && self.hyperedges_removed.is_empty()
            && self.mocs_added.is_empty()
            && self.mocs_removed.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Temporal Hypergraph Types
// ═══════════════════════════════════════════════════════════════════════════════

/// A hyperedge that evolves over time.
///
/// Extends the existing `Hyperedge` with:
/// - `valid_from` / `valid_until`: lifetime window
/// - `intervals`: discontinuous active periods (gaps allowed)
/// - `version`: incremented on each update
/// - `supersedes` / `superseded_by`: version chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalHyperedge {
    // ── Base Hyperedge fields (mirrored from skill_graph::types::Hyperedge) ──
    pub hyperedge_id: String,
    pub name: String,
    pub description: String,
    pub components: Vec<String>,
    pub target_composite: Option<String>,
    pub composition_type: super::super::skill_graph::types::CompositionType,
    pub weight: f32,
    pub metadata: HashMap<String, String>,

    // ── Temporal extensions ──
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub intervals: Vec<TimeInterval>,
    pub version: TemporalVersion,
    pub supersedes: Option<String>,
    pub superseded_by: Option<String>,
}

/// Version identifier for a temporal hyperedge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TemporalVersion(pub u32);

impl Default for TemporalVersion {
    fn default() -> Self {
        Self(1)
    }
}

/// A continuous time interval during which a hyperedge is active.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeInterval {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub label: Option<String>,
}

/// Event type in the temporal index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalEventKind {
    Created,
    Modified,
    Activated,
    Deactivated,
    Superseded,
}

/// Entry in the temporal index for efficient time-range queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalIndexEntry {
    pub hyperedge_id: String,
    pub timestamp: DateTime<Utc>,
    pub event_kind: TemporalEventKind,
}

/// Binary-searchable temporal index.
///
/// Maintains a sorted Vec of (timestamp, entry) pairs.
/// Queries use binary search for O(log N) point lookups.
#[derive(Debug, Clone)]
pub struct TemporalIndex {
    entries: Vec<(DateTime<Utc>, TemporalIndexEntry)>,
}

impl Default for TemporalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert a new entry (maintains sorted order).
    pub fn insert(&mut self, timestamp: DateTime<Utc>, entry: TemporalIndexEntry) {
        let pos = self
            .entries
            .binary_search_by(|(t, _)| t.cmp(&timestamp))
            .unwrap_or_else(|e| e);
        self.entries.insert(pos, (timestamp, entry));
    }

    /// Find all entries at or after `start` and at or before `end`.
    pub fn range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&TemporalIndexEntry> {
        let lo = self
            .entries
            .binary_search_by(|(t, _)| t.cmp(&start))
            .unwrap_or_else(|e| e);
        let hi = self
            .entries
            .binary_search_by(|(t, _)| t.cmp(&end))
            .unwrap_or_else(|e| e);

        self.entries[lo..hi].iter().map(|(_, e)| e).collect()
    }

    /// Find all entries for a specific hyperedge, ordered by time.
    pub fn history_of(&self, hyperedge_id: &str) -> Vec<&TemporalIndexEntry> {
        self.entries
            .iter()
            .filter(|(_, e)| e.hyperedge_id == hyperedge_id)
            .map(|(_, e)| e)
            .collect()
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TemporalHypergraphStore
//
// Experimental library component.  It is intentionally not updated by the
// SkillGraph mutation path and CausalEngine does not consume it as a prior, so
// callers must opt in explicitly and must not describe it as runtime temporal
// reasoning.
// ═══════════════════════════════════════════════════════════════════════════════

/// Store for temporal hyperedges with versioning and time-range queries.
#[derive(Debug, Clone)]
pub struct TemporalHypergraphStore {
    /// All hyperedges (including historical versions), keyed by hyperedge_id
    hyperedges: std::sync::Arc<parking_lot::RwLock<HashMap<String, Vec<TemporalHyperedge>>>>,
    /// Temporal index for time-range queries
    index: std::sync::Arc<parking_lot::RwLock<TemporalIndex>>,
}

impl Default for TemporalHypergraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalHypergraphStore {
    pub fn new() -> Self {
        Self {
            hyperedges: std::sync::Arc::new(parking_lot::RwLock::new(HashMap::new())),
            index: std::sync::Arc::new(parking_lot::RwLock::new(TemporalIndex::new())),
        }
    }

    // ── CRUD ──

    /// Register a new temporal hyperedge (version 1).
    pub fn register(&self, mut he: TemporalHyperedge) {
        let now = Utc::now();
        he.valid_from = now;
        he.version = TemporalVersion(1);

        let id = he.hyperedge_id.clone();
        self.hyperedges
            .write()
            .entry(id.clone())
            .or_default()
            .push(he);

        self.index.write().insert(
            now,
            TemporalIndexEntry {
                hyperedge_id: id,
                timestamp: now,
                event_kind: TemporalEventKind::Created,
            },
        );
    }

    /// Update/create a new version of a hyperedge.
    /// Returns the new version number.
    pub fn update(&self, mut he: TemporalHyperedge) -> TemporalVersion {
        let now = Utc::now();
        let id = he.hyperedge_id.clone();

        let mut versions = self.hyperedges.write();
        let new_version = {
            let entry = versions.entry(id.clone()).or_default();
            let prev_version = entry.last().map(|v| v.version.0).unwrap_or(0);
            let next_ver = TemporalVersion(prev_version + 1);

            he.version = next_ver;
            he.valid_from = now;
            he.supersedes = entry.last().map(|v| v.hyperedge_id.clone());

            // Mark previous version as superseded
            if let Some(prev) = entry.last_mut() {
                prev.superseded_by = Some(id.clone());
                prev.valid_until = Some(now);
            }

            entry.push(he);
            next_ver
        };

        self.index.write().insert(
            now,
            TemporalIndexEntry {
                hyperedge_id: id,
                timestamp: now,
                event_kind: TemporalEventKind::Modified,
            },
        );

        new_version
    }

    /// Deactivate a hyperedge at a given time.
    pub fn deactivate(&self, hyperedge_id: &str, at: DateTime<Utc>) {
        let mut versions = self.hyperedges.write();
        if let Some(entries) = versions.get_mut(hyperedge_id) {
            if let Some(latest) = entries.last_mut() {
                latest.valid_until = Some(at);
            }
        }

        self.index.write().insert(
            at,
            TemporalIndexEntry {
                hyperedge_id: hyperedge_id.to_string(),
                timestamp: at,
                event_kind: TemporalEventKind::Deactivated,
            },
        );
    }

    // ── Queries ──

    /// Get the current (latest) version of a hyperedge.
    pub fn current(&self, hyperedge_id: &str) -> Option<TemporalHyperedge> {
        self.hyperedges
            .read()
            .get(hyperedge_id)
            .and_then(|v| v.last().cloned())
    }

    /// Get all versions of a hyperedge (historical).
    pub fn history(&self, hyperedge_id: &str) -> Vec<TemporalHyperedge> {
        self.hyperedges
            .read()
            .get(hyperedge_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Find all hyperedges active at a specific instant.
    pub fn active_at(&self, instant: DateTime<Utc>) -> Vec<TemporalHyperedge> {
        self.hyperedges
            .read()
            .values()
            .flat_map(|versions| {
                versions.last().filter(|he| {
                    he.valid_from <= instant
                        && he.valid_until.map_or(true, |until| until >= instant)
                })
            })
            .cloned()
            .collect()
    }

    /// Find hyperedges active within a time range.
    pub fn active_between(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<TemporalHyperedge> {
        self.hyperedges
            .read()
            .values()
            .flat_map(|versions| {
                versions.last().filter(|he| {
                    he.valid_from <= end && he.valid_until.map_or(true, |until| until >= start)
                })
            })
            .cloned()
            .collect()
    }

    /// Find temporally ordered hyperedge pairs for causal analysis.
    /// Returns pairs (earlier, later) where earlier's end ≤ later's start.
    pub fn find_causal_candidates(&self) -> Vec<(TemporalHyperedge, TemporalHyperedge)> {
        let all: Vec<TemporalHyperedge> = self
            .hyperedges
            .read()
            .values()
            .filter_map(|v| v.last().cloned())
            .collect();

        let mut candidates = Vec::new();
        for i in 0..all.len() {
            for j in 0..all.len() {
                if i == j {
                    continue;
                }
                if let Some(end_i) = all[i].valid_until {
                    if end_i <= all[j].valid_from {
                        // Check if they share components
                        let shared: Vec<&String> = all[i]
                            .components
                            .iter()
                            .filter(|c| all[j].components.contains(c))
                            .collect();
                        if !shared.is_empty() {
                            candidates.push((all[i].clone(), all[j].clone()));
                        }
                    }
                }
            }
        }
        candidates
    }

    /// Get temporal index entries for debug/audit.
    pub fn index_entries(&self) -> Vec<TemporalIndexEntry> {
        self.index
            .read()
            .entries
            .iter()
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// Number of unique hyperedges (latest version only).
    pub fn count(&self) -> usize {
        self.hyperedges.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.hyperedges.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill_graph::types::CompositionType;

    fn make_he(id: &str, components: Vec<&str>) -> TemporalHyperedge {
        TemporalHyperedge {
            hyperedge_id: id.to_string(),
            name: id.to_string(),
            description: "test".to_string(),
            components: components.into_iter().map(|s| s.to_string()).collect(),
            target_composite: None,
            composition_type: CompositionType::Conjunction,
            weight: 1.0,
            metadata: HashMap::new(),
            valid_from: Utc::now(),
            valid_until: None,
            intervals: Vec::new(),
            version: TemporalVersion(1),
            supersedes: None,
            superseded_by: None,
        }
    }

    #[test]
    fn test_register_and_current() {
        let store = TemporalHypergraphStore::new();
        store.register(make_he("he-1", vec!["a", "b"]));
        let current = store.current("he-1");
        assert!(current.is_some());
        assert_eq!(current.unwrap().version.0, 1);
    }

    #[test]
    fn test_update_creates_new_version() {
        let store = TemporalHypergraphStore::new();
        store.register(make_he("he-1", vec!["a"]));
        let v2 = store.update(make_he("he-1", vec!["a", "b", "c"]));
        assert_eq!(v2.0, 2);

        let current = store.current("he-1").unwrap();
        assert_eq!(current.components.len(), 3);
        assert!(current.supersedes.is_some());

        let history = store.history("he-1");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_active_at() {
        let store = TemporalHypergraphStore::new();
        store.register(make_he("he-1", vec!["a"]));
        let now = Utc::now();
        let active = store.active_at(now);
        assert!(!active.is_empty());
        assert_eq!(active[0].hyperedge_id, "he-1");
    }

    #[test]
    fn test_deactivate() {
        let store = TemporalHypergraphStore::new();
        store.register(make_he("he-1", vec!["a"]));
        let now = Utc::now();
        store.deactivate("he-1", now);

        let future = now + chrono::Duration::hours(1);
        assert!(store.active_at(future).is_empty());
    }

    #[test]
    fn test_temporal_index() {
        let mut idx = TemporalIndex::new();
        let now = Utc::now();
        idx.insert(
            now,
            TemporalIndexEntry {
                hyperedge_id: "he-1".to_string(),
                timestamp: now,
                event_kind: TemporalEventKind::Created,
            },
        );

        assert_eq!(idx.len(), 1);
        let range = idx.range(
            now - chrono::Duration::seconds(1),
            now + chrono::Duration::seconds(1),
        );
        assert_eq!(range.len(), 1);
    }

    #[test]
    fn test_find_causal_candidates() {
        let store = TemporalHypergraphStore::new();
        let mut he1 = make_he("he-1", vec!["a", "b"]);
        he1.valid_from = Utc::now() - chrono::Duration::hours(2);
        he1.valid_until = Some(Utc::now() - chrono::Duration::hours(1));
        store.register(he1);

        let he2 = make_he("he-2", vec!["b", "c"]);
        store.register(he2);

        let candidates = store.find_causal_candidates();
        assert!(
            !candidates.is_empty(),
            "Should find he-1 → he-2 via shared component b"
        );
    }
}
