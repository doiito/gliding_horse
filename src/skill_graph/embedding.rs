use std::sync::Arc;

use hyperspace_engine::hyper_vector::{EmbeddingVector, MetricKind};

use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::{SkillGraphNode, SkillLinkType};

/// Generates Poincaré structural embeddings from skill graph topology.
///
/// Each skill is embedded in Poincaré ball space where:
/// - **Radius** (dimension 1): encodes prerequisite depth — foundational skills
///   (no prerequisites) sit near the ball boundary, deep skills near the origin.
/// - **Angular dimensions** (dimensions 2–4): encode domain/cluster identity via
///   tag fingerprinting, so skills in the same domain cluster together.
///
/// Skills with similar topological positions (similar depth + related tags) have
/// nearby Poincaré embeddings; Poincaré distance naturally captures the
/// hierarchical structure of the skill dependency DAG.
pub struct SkillGraphEmbedder {
    store: Arc<SkillGraphStore>,
    dim: usize,
}

impl SkillGraphEmbedder {
    /// Create a new embedder over the given skill graph store.
    /// Uses a 4-dimensional Poincaré ball (default).
    pub fn new(store: Arc<SkillGraphStore>) -> Self {
        Self { store, dim: 4 }
    }

    /// Set a custom embedding dimension (minimum 3).
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim.max(3);
        self
    }

    /// Compute a Poincaré embedding for a skill from its graph topology.
    ///
    /// Returns `None` if the skill is not in the store.
    pub fn embed_skill(&self, skill_iri: &str) -> Option<EmbeddingVector> {
        let skill = self.store.get_skill(skill_iri)?;
        let coords = self.compute_coords(&skill);
        Some(EmbeddingVector::new_unchecked(coords, MetricKind::Poincare))
    }

    /// Rank all skills by structural similarity (Poincaré distance) to `skill_iri`.
    /// Returns up to `limit` results as `(iri, similarity_score)` pairs.
    pub fn rank_by_similarity(&self, skill_iri: &str, limit: usize) -> Vec<(String, f32)> {
        let target_emb = match self.embed_skill(skill_iri) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let all_skills = self.store.list_all_skills();
        let mut scored: Vec<(String, f32)> = all_skills
            .iter()
            .filter(|s| s.skill_iri != skill_iri)
            .filter_map(|s| {
                let emb = self.embed_skill(&s.skill_iri)?;
                let sim = self.poincare_similarity(&target_emb, &emb);
                Some((s.skill_iri.clone(), sim as f32))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    // ── Coordinate computation ───────────────────────────────────────────

    fn compute_coords(&self, skill: &SkillGraphNode) -> Vec<f64> {
        let all_skills = self.store.list_all_skills();

        let max_depth = all_skills
            .iter()
            .map(|s| self.prerequisite_depth(&s.skill_iri))
            .max()
            .unwrap_or(1)
            .max(1);

        let max_dependents = all_skills
            .iter()
            .map(|s| self.count_dependents(&s.skill_iri))
            .max()
            .unwrap_or(1)
            .max(1);

        let depth = self.prerequisite_depth(&skill.skill_iri);
        let dependents = self.count_dependents(&skill.skill_iri);

        // Dimension 1: foundational depth — encodes hierarchy level.
        // Foundational skills (no prereqs, depth_ratio ≈ 0) → 0.85 (near boundary)
        // Deep skills (many prereqs, depth_ratio ≈ 1) → 0.10 (near origin)
        let depth_ratio = depth as f64 / max_depth as f64;
        let d1 = 0.85 - depth_ratio * 0.75;

        // Dimension 2: breadth — skills with many dependents are broader
        let dep_ratio = dependents as f64 / max_dependents as f64;
        let d2 = dep_ratio * 0.3;

        // Dimensions 3+: tag domain fingerprint — clusters skills by domain.
        // Ensures skills with similar tags have similar angular positions.
        let mut coords = vec![d1, d2];
        let fp = self.tag_fingerprint(&skill.tags, self.dim.saturating_sub(2));

        // Remaining dimensions carry tag fingerprint scaled to stay within ball.
        // Scale ensures the Euclidean norm stays < 1.0:
        //   sqrt(0.85² + 0.3² + 0.2² * (dim-2)) = ?
        //   For dim=4: sqrt(0.7225 + 0.09 + 0.04 + 0.04) ≈ 0.94 < 1.0 ✓
        for i in 0..fp.len().min(self.dim.saturating_sub(2)) {
            let scaled = fp[i] * 0.2;
            coords.push(scaled);
        }

        // Pad remaining dims with zeros
        while coords.len() < self.dim {
            coords.push(0.0);
        }

        // Verify norm < 1.0 (debug-only to avoid perf hit)
        debug_assert!(
            {
                let sq: f64 = coords.iter().map(|c| c * c).sum();
                sq < 1.0
            },
            "Poincaré embedding norm must be < 1.0, got {:.4}",
            {
                let sq: f64 = coords.iter().map(|c| c * c).sum();
                sq.sqrt()
            }
        );

        coords
    }

    // ── Poincaré geometry ────────────────────────────────────────────────

    /// Poincaré similarity in [0, 1], where 1.0 = identical and 0.0 = infinitely far.
    fn poincare_similarity(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        let diff_sq: f64 = a
            .coords
            .iter()
            .zip(&b.coords)
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        let norm_a_sq: f64 = a.coords.iter().map(|x| x.powi(2)).sum();
        let norm_b_sq: f64 = b.coords.iter().map(|x| x.powi(2)).sum();

        let denom = (1.0 - norm_a_sq) * (1.0 - norm_b_sq);
        if denom <= 0.0 {
            return 0.0;
        }

        // Poincaré distance: d(u,v) = arccosh(1 + 2 * ||u-v||² / ((1-||u||²)(1-||v||²)))
        let arg = 1.0 + 2.0 * diff_sq / denom;
        let dist = if arg > 1.0 { arg.acosh() } else { 0.0 };

        // Convert distance to similarity: 1 / (1 + dist)
        1.0 / (1.0 + dist)
    }

    // ── Graph topology helpers ───────────────────────────────────────────

    fn prerequisite_depth(&self, iri: &str) -> usize {
        let deps = self.store.resolve_dependencies(iri);
        deps.len().saturating_sub(1) // deps includes self at end
    }

    fn count_dependents(&self, iri: &str) -> usize {
        self.store
            .list_all_skills()
            .iter()
            .filter(|s| {
                s.links
                    .iter()
                    .any(|l| l.link_type == SkillLinkType::Prerequisite && l.target_iri == iri)
            })
            .count()
    }

    // ── Domain fingerprinting ────────────────────────────────────────────

    /// Deterministic tag-based fingerprint projected onto `n` angular dimensions.
    /// Skills with similar tag sets produce nearby fingerprint vectors.
    fn tag_fingerprint(&self, tags: &[String], n: usize) -> Vec<f64> {
        if n == 0 || tags.is_empty() {
            return vec![0.0; n.max(0)];
        }

        let mut fp = vec![0.0f64; n];
        for tag in tags {
            for i in 0..n {
                let h = stable_hash(&format!("{}:{}", i, tag)) as f64;
                fp[i] += (h / u64::MAX as f64) * 2.0 - 1.0;
            }
        }

        // Normalize to unit length per dimension group (directional)
        let mag: f64 = fp.iter().map(|x| x * x).sum();
        if mag > 0.001 {
            let inv = 1.0 / mag.sqrt();
            for x in &mut fp {
                *x *= inv;
            }
        }

        fp
    }
}

/// Deterministic string hash (djb2) for tag fingerprinting.
fn stable_hash(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill_graph::types::SkillGraphNode;
    use std::sync::Arc;

    fn setup_graph() -> Arc<SkillGraphStore> {
        let store = Arc::new(SkillGraphStore::new());

        // Root/foundational skill (depth 0, many dependents)
        let rust_basics = SkillGraphNode::new(
            "iri://skills/rust-basics",
            "Rust Basics",
            "Fundamental Rust concepts",
        )
        .with_tag("rust")
        .with_tag("fundamentals");
        store.register_skill(rust_basics).unwrap();

        // Intermediate skill (depth 1)
        let mut rust_async = SkillGraphNode::new(
            "iri://skills/rust-async",
            "Rust Async",
            "Async programming in Rust",
        )
        .with_tag("rust")
        .with_tag("async");
        rust_async.add_prerequisite("iri://skills/rust-basics", "Needs Rust basics");
        store.register_skill(rust_async).unwrap();

        // Deep skill (depth 2)
        let mut rust_web = SkillGraphNode::new(
            "iri://skills/rust-web",
            "Rust Web",
            "Web programming in Rust",
        )
        .with_tag("rust")
        .with_tag("web");
        rust_web.add_prerequisite("iri://skills/rust-async", "Needs async Rust");
        store.register_skill(rust_web).unwrap();

        // Unrelated domain skill (depth 0)
        let python_basics = SkillGraphNode::new(
            "iri://skills/python-basics",
            "Python Basics",
            "Fundamental Python concepts",
        )
        .with_tag("python")
        .with_tag("fundamentals");
        store.register_skill(python_basics).unwrap();

        store
    }

    #[test]
    fn test_embed_skill_returns_valid_embedding() {
        let store = setup_graph();
        let embedder = SkillGraphEmbedder::new(store);

        let emb = embedder.embed_skill("iri://skills/rust-basics");
        assert!(emb.is_some());

        let emb = emb.unwrap();
        assert_eq!(emb.metric, MetricKind::Poincare);
        assert_eq!(emb.coords.len(), 4);

        // Verify Poincaré ball constraint: norm < 1.0
        let sq_norm: f64 = emb.coords.iter().map(|c| c * c).sum();
        assert!(sq_norm < 1.0, "Norm {} must be < 1.0", sq_norm.sqrt());
    }

    #[test]
    fn test_missing_skill_returns_none() {
        let store = setup_graph();
        let embedder = SkillGraphEmbedder::new(store);

        let emb = embedder.embed_skill("iri://skills/nonexistent");
        assert!(emb.is_none());
    }

    #[test]
    fn test_foundational_skill_near_boundary() {
        let store = setup_graph();
        let embedder = SkillGraphEmbedder::new(store);

        // Foundational (depth 0) should be near boundary (d1 ≈ 0.85)
        let emb = embedder.embed_skill("iri://skills/rust-basics").unwrap();
        assert!(
            emb.coords[0] > 0.8,
            "Foundational skill d1={} should be > 0.8",
            emb.coords[0]
        );

        // Deep skill (depth 2) should be nearer origin (d1 ≈ 0.10)
        let deep_emb = embedder.embed_skill("iri://skills/rust-web").unwrap();
        assert!(
            deep_emb.coords[0] < 0.3,
            "Deep skill d1={} should be < 0.3",
            deep_emb.coords[0]
        );
    }

    #[test]
    fn test_rank_by_similarity_returns_most_similar_first() {
        let store = setup_graph();
        let embedder = SkillGraphEmbedder::new(store);

        let ranked = embedder.rank_by_similarity("iri://skills/rust-basics", 10);
        assert!(!ranked.is_empty(), "Should return results");

        for w in ranked.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "Results must be sorted descending by score: {:?} >= {:?}",
                w[0].1,
                w[1].1
            );
        }

        let mut unique_iris: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (iri, _) in &ranked {
            assert!(unique_iris.insert(iri), "Duplicate IRI in results: {}", iri);
        }

        assert!(
            ranked
                .iter()
                .any(|(iri, _)| iri == "iri://skills/python-basics"),
            "Same-depth skill should appear in results"
        );
    }

    #[test]
    fn test_empty_store_handling() {
        let store = Arc::new(SkillGraphStore::new());
        let embedder = SkillGraphEmbedder::new(store);

        let ranked = embedder.rank_by_similarity("iri://skills/nonexistent", 10);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_different_dimensions() {
        let store = setup_graph();
        let embedder = SkillGraphEmbedder::new(store.clone()).with_dim(6);

        let emb = embedder.embed_skill("iri://skills/rust-basics").unwrap();
        assert_eq!(emb.coords.len(), 6);

        let sq_norm: f64 = emb.coords.iter().map(|c| c * c).sum();
        assert!(sq_norm < 1.0, "Norm {} must be < 1.0", sq_norm.sqrt());
    }

    #[test]
    fn test_same_skill_identity_similarity() {
        let store = setup_graph();
        let embedder = SkillGraphEmbedder::new(store);

        let emb1 = embedder.embed_skill("iri://skills/rust-basics").unwrap();
        let emb2 = embedder.embed_skill("iri://skills/rust-basics").unwrap();

        let sim = embedder.poincare_similarity(&emb1, &emb2);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Same skill should have similarity ≈ 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_rank_excludes_self() {
        let store = setup_graph();
        let embedder = SkillGraphEmbedder::new(store);

        let ranked = embedder.rank_by_similarity("iri://skills/rust-basics", 10);
        assert!(
            ranked
                .iter()
                .all(|(iri, _)| iri != "iri://skills/rust-basics"),
            "Self should not appear in results"
        );
    }
}
