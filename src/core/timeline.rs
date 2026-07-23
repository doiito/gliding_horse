use chrono::{DateTime, Utc};

use crate::memory::hyperspace_store::{HybridSearchFilter, HyperspaceStore, ScoredEntry};
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::SkillGraphNode;

/// Unified time range filter for timeline queries.
///
/// Can specify `after`, `before`, or both.  A `TimeRange` with no bounds
/// matches every entry (pass-through).
#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    /// Include entries created / stored at or after this instant.
    pub after: Option<DateTime<Utc>>,
    /// Include entries created / stored before or at this instant.
    pub before: Option<DateTime<Utc>>,
}

impl TimeRange {
    pub fn new() -> Self {
        Self {
            after: None,
            before: None,
        }
    }

    /// Only entries at or after `after`.
    pub fn after(after: DateTime<Utc>) -> Self {
        Self {
            after: Some(after),
            before: None,
        }
    }

    /// Only entries before or at `before`.
    pub fn before(before: DateTime<Utc>) -> Self {
        Self {
            after: None,
            before: Some(before),
        }
    }

    /// Entries whose timestamp lies in `[after, before]`.
    pub fn range(after: DateTime<Utc>, before: DateTime<Utc>) -> Self {
        Self {
            after: Some(after),
            before: Some(before),
        }
    }

    /// `after` bound as a Unix-epoch seconds f64 (for Hyperspace `stored_at`).
    pub fn after_unix_secs(&self) -> Option<f64> {
        self.after.map(|dt| dt.timestamp() as f64)
    }

    /// `before` bound as a Unix-epoch seconds f64 (for Hyperspace `stored_at`).
    pub fn before_unix_secs(&self) -> Option<f64> {
        self.before.map(|dt| dt.timestamp() as f64)
    }

    /// True when `dt` falls within the range (or the range side is unset).
    pub fn contains(&self, dt: &DateTime<Utc>) -> bool {
        if let Some(after) = self.after {
            if *dt < after {
                return false;
            }
        }
        if let Some(before) = self.before {
            if *dt > before {
                return false;
            }
        }
        true
    }
}

impl Default for TimeRange {
    fn default() -> Self {
        Self::new()
    }
}

/// A single entry on a unified timeline, boxing results from any subsystem.
#[derive(Debug, Clone)]
pub enum TimelineEntry {
    /// A skill-graph node.
    Skill(SkillGraphNode),
    /// A vector-store entry (semantic search hit).
    Vector(ScoredEntry),
}

impl TimelineEntry {
    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            TimelineEntry::Skill(s) => &s.name,
            TimelineEntry::Vector(v) => &v.iri,
        }
    }

    /// Best-effort timestamp for sort / display.
    pub fn timestamp(&self) -> Option<DateTime<Utc>> {
        match self {
            TimelineEntry::Skill(s) => Some(s.created_at),
            TimelineEntry::Vector(v) => v
                .stored_at
                .and_then(|ts| DateTime::from_timestamp(ts as i64, 0)),
        }
    }
}

// ── Timeline queries on SkillGraphStore ────────────────────────────────────

impl SkillGraphStore {
    /// Return skills whose `created_at` falls within `range`.
    pub fn query_skills_by_time(&self, range: &TimeRange) -> Vec<SkillGraphNode> {
        self.list_all_skills()
            .into_iter()
            .filter(|s| range.contains(&s.created_at))
            .collect()
    }

    /// Return skills whose `updated_at` falls within `range`.
    pub fn query_skills_by_update_time(&self, range: &TimeRange) -> Vec<SkillGraphNode> {
        self.list_all_skills()
            .into_iter()
            .filter(|s| range.contains(&s.updated_at))
            .collect()
    }

    /// Return skills that have **not** been used since `cutoff`.
    ///
    /// A skill with `last_used_at == None` is considered stale (never used).
    /// This is the primary API for LLRU cold-data discovery.
    pub fn query_unused_since(&self, cutoff: &DateTime<Utc>) -> Vec<SkillGraphNode> {
        self.list_all_skills()
            .into_iter()
            .filter(|s| {
                s.last_used_at
                    .as_ref()
                    .map(|used| used < cutoff)
                    .unwrap_or(true)
            })
            .collect()
    }
}

// ── Timeline queries on HyperspaceStore ────────────────────────────────────

impl HyperspaceStore {
    /// Search vector entries restricted to a `TimeRange` on `stored_at`.
    pub async fn search_by_time(
        &self,
        query: &str,
        range: &TimeRange,
        limit: u64,
    ) -> Result<Vec<ScoredEntry>, crate::CoreError> {
        let mut filter = HybridSearchFilter::new();
        if let Some(after) = range.after_unix_secs() {
            filter = filter.with_created_after(after);
        }
        if let Some(before) = range.before_unix_secs() {
            filter = filter.with_created_before(before);
        }
        self.search_with_filter(query, &filter, limit).await
    }

    /// All vector entries within `range` (empty-query match-all).
    pub async fn query_vectors_by_time(
        &self,
        range: &TimeRange,
        limit: u64,
    ) -> Result<Vec<ScoredEntry>, crate::CoreError> {
        self.search_by_time("", range, limit).await
    }
}

// ── Time-decayed re-ranking of vector results ────────────────────────────

/// Apply an exponential time-decay multiplier to scored entries.
///
/// `decay_lambda` controls the half-life:
/// - `λ = 0.0` → no decay (pass-through)
/// - `λ = 0.5` → score halves after ~1.39 hours
/// - `λ = 1.0` → score halves after ~0.69 hours
///
/// Each entry's score is multiplied by `exp(-λ * hours_since_stored)`.
/// Entries without a `stored_at` timestamp are not penalised.
pub fn apply_time_decay(entries: &mut [ScoredEntry], decay_lambda: f64) {
    let now = Utc::now();
    for entry in entries.iter_mut() {
        if let Some(stored_at) = entry.stored_at {
            let age_secs = now.timestamp() as f64 - stored_at;
            let age_hours = age_secs / 3600.0;
            if age_hours > 0.0 {
                entry.score *= (-decay_lambda * age_hours).exp() as f32;
            }
        }
    }
    // Re-sort descending by new score
    entries.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range_contains() {
        let now = Utc::now();
        let earlier = now - chrono::Duration::hours(2);
        let later = now + chrono::Duration::hours(2);

        let range = TimeRange::range(earlier, later);
        assert!(range.contains(&now));
        assert!(!range.contains(&(earlier - chrono::Duration::hours(1))));
        assert!(!range.contains(&(later + chrono::Duration::hours(1))));
    }

    #[test]
    fn test_time_range_after_only() {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::hours(1);
        let range = TimeRange::after(cutoff);
        assert!(range.contains(&now));
        assert!(!range.contains(&(cutoff - chrono::Duration::hours(1))));
    }

    #[test]
    fn test_time_range_before_only() {
        let now = Utc::now();
        let cutoff = now + chrono::Duration::hours(1);
        let range = TimeRange::before(cutoff);
        assert!(range.contains(&now));
        assert!(!range.contains(&(cutoff + chrono::Duration::hours(1))));
    }

    #[test]
    fn test_time_range_unix_secs() {
        let now = Utc::now();
        let range = TimeRange::after(now);
        let unix = range.after_unix_secs().unwrap();
        assert!((unix - now.timestamp() as f64).abs() < 1.0);
    }

    #[test]
    fn test_timeline_entry_label() {
        let skill = SkillGraphNode::new("iri://test", "Test Skill", "A test");
        let entry = TimelineEntry::Skill(skill);
        assert_eq!(entry.label(), "Test Skill");
    }

    #[test]
    fn test_query_unused_since() {
        let store = SkillGraphStore::new();
        let fresh = SkillGraphNode::new("iri://fresh", "Fresh", "Recently used").with_last_used();
        let _ = store.register_skill(fresh);

        let stale = SkillGraphNode::new("iri://stale", "Stale", "Never used");
        let _ = store.register_skill(stale);

        let cutoff = Utc::now() - chrono::Duration::minutes(1);
        let unused = store.query_unused_since(&cutoff);
        assert_eq!(unused.len(), 1);
        assert_eq!(unused[0].skill_iri, "iri://stale");
    }

    #[test]
    fn test_apply_time_decay_no_decay() {
        let mut entries = vec![ScoredEntry {
            iri: "x".into(),
            text: "".into(),
            score: 0.9,
            tags: vec![],
            importance: None,
            jsonld_types: vec![],
            stored_at: Some(Utc::now().timestamp() as f64),
        }];
        apply_time_decay(&mut entries, 0.0);
        assert!((entries[0].score - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_apply_time_decay_with_age() {
        let two_hours_ago = (Utc::now() - chrono::Duration::hours(2)).timestamp() as f64;
        let mut entries = vec![
            ScoredEntry {
                iri: "old".into(),
                text: "".into(),
                score: 0.9,
                tags: vec![],
                importance: None,
                jsonld_types: vec![],
                stored_at: Some(two_hours_ago),
            },
            ScoredEntry {
                iri: "new".into(),
                text: "".into(),
                score: 0.5,
                tags: vec![],
                importance: None,
                jsonld_types: vec![],
                stored_at: Some(Utc::now().timestamp() as f64),
            },
        ];
        apply_time_decay(&mut entries, 0.5);
        // "new" should now rank first despite lower raw score
        assert_eq!(entries[0].iri, "new");
    }
}
