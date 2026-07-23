use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::debug;

use crate::causal::types::{CausalObservation, PropagationEdge};

/// Persistent store for causal failure data.
///
/// Maintains:
/// - `error_index`: error_signature → [(skill_iri, count)] — fast lookup
/// - `error_profiles`: skill_iri → {error_signature → count} — per-skill profile
/// - `propagation_edges`: (from_iri, to_iri) → count — propagation statistics
/// - `prior_probability`: skill_iri → prior P(skill_is_root) based on failure rate
///
/// All data is held in memory for hot-path performance, with optional
/// persistence via `L0Store` (integrated at the `SkillGraphStore` level).
#[derive(Debug, Clone)]
pub struct CausalModelStore {
    /// error_signature → Vec<(skill_iri, count)>
    error_index: Arc<RwLock<HashMap<String, Vec<(String, u32)>>>>,
    /// skill_iri → (error_signature → count)
    error_profiles: Arc<RwLock<HashMap<String, HashMap<String, u32>>>>,
    /// (from, to) → propagation count
    edges: Arc<RwLock<HashMap<(String, String), u32>>>,
    /// skill_iri → prior root-cause probability
    priors: Arc<RwLock<HashMap<String, f32>>>,
}

impl Default for CausalModelStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalModelStore {
    pub fn new() -> Self {
        Self {
            error_index: Arc::new(RwLock::new(HashMap::new())),
            error_profiles: Arc::new(RwLock::new(HashMap::new())),
            edges: Arc::new(RwLock::new(HashMap::new())),
            priors: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // ── Recording ─────────────────────────────────────────────────────────

    /// Record an error observation, updating statistical models.
    pub fn record_observation(&self, obs: &CausalObservation) {
        // 1. Update error_index
        {
            let mut idx = self.error_index.write();
            let entries = idx.entry(obs.error_signature.clone()).or_default();
            if let Some(pos) = entries.iter_mut().find(|(iri, _)| iri == &obs.skill_iri) {
                pos.1 += 1;
            } else {
                entries.push((obs.skill_iri.clone(), 1));
            }
        }

        // 2. Update error_profiles
        {
            let mut profiles = self.error_profiles.write();
            let profile = profiles.entry(obs.skill_iri.clone()).or_default();
            *profile.entry(obs.error_signature.clone()).or_insert(0) += 1;
        }

        // 3. Record propagation if known
        if let Some(ref from) = obs.propagation_from {
            let mut edges = self.edges.write();
            let key = (from.clone(), obs.skill_iri.clone());
            *edges.entry(key).or_insert(0) += 1;
        }

        debug!(
            "CausalModelStore: recorded {} on {} (propagation_from={:?})",
            obs.error_signature, obs.skill_iri, obs.propagation_from
        );
    }

    /// Record a propagation edge between two skills.
    pub fn record_propagation(&self, from: &str, to: &str) {
        let mut edges = self.edges.write();
        let key = (from.to_string(), to.to_string());
        *edges.entry(key).or_insert(0) += 1;
    }

    /// Update the prior probability that a skill is a root cause.
    /// Normally computed from (1.0 - success_rate) normalized across all skills.
    pub fn set_prior(&self, skill_iri: &str, probability: f32) {
        let prob = probability.clamp(0.001, 0.999);
        self.priors.write().insert(skill_iri.to_string(), prob);
    }

    /// Batch update priors from success rates.
    /// `success_rates`: skill_iri → success_rate (0.0–1.0)
    pub fn update_priors_from_success_rates(&self, success_rates: &[(&str, f32)]) {
        let total_failures: f32 = success_rates.iter().map(|(_, rate)| 1.0 - rate).sum();
        let total = total_failures.max(0.001);

        let mut priors = self.priors.write();
        for (iri, rate) in success_rates {
            let prior = (1.0 - rate) / total;
            priors.insert(iri.to_string(), prior.clamp(0.001, 0.999));
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────

    /// Find all skills that have exhibited a given error signature,
    /// ordered by frequency (most likely first).
    pub fn find_skills_by_error(&self, error_signature: &str) -> Vec<(String, u32)> {
        let idx = self.error_index.read();
        let mut results = idx.get(error_signature).cloned().unwrap_or_default();
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results
    }

    /// Get the error profile for a specific skill.
    pub fn error_profile(&self, skill_iri: &str) -> HashMap<String, u32> {
        self.error_profiles
            .read()
            .get(skill_iri)
            .cloned()
            .unwrap_or_default()
    }

    /// Get the propagation probability from `from` to `to`.
    /// Returns P(error_at_to | error_at_from) = count(from→to) / total_outgoing(from)
    pub fn propagation_probability(&self, from: &str, to: &str) -> f32 {
        let edges = self.edges.read();
        let key = (from.to_string(), to.to_string());
        let count = edges.get(&key).copied().unwrap_or(0) as f32;

        let total_out: u32 = edges
            .iter()
            .filter(|((f, _), _)| f == from)
            .map(|(_, c)| c)
            .sum();

        if total_out == 0 {
            0.0
        } else {
            count / total_out as f32
        }
    }

    /// Get all outgoing propagation edges from a skill.
    pub fn outgoing_edges(&self, from: &str) -> Vec<(String, PropagationEdge)> {
        let edges = self.edges.read();
        let mut total_out: u32 = 0;
        let mut targets: Vec<(String, u32)> = Vec::new();

        for ((f, t), count) in edges.iter() {
            if f == from {
                total_out += count;
                targets.push((t.clone(), *count));
            }
        }

        targets
            .into_iter()
            .map(|(to, count)| {
                let weight = if total_out > 0 {
                    count as f32 / total_out as f32
                } else {
                    0.0
                };
                (
                    to,
                    PropagationEdge {
                        weight,
                        observation_count: count,
                    },
                )
            })
            .collect()
    }

    /// Get prior probability for a skill.
    pub fn prior(&self, skill_iri: &str) -> f32 {
        self.priors.read().get(skill_iri).copied().unwrap_or(0.001)
    }

    /// List all skills that have recorded failures.
    pub fn known_failing_skills(&self) -> Vec<String> {
        self.error_profiles.read().keys().cloned().collect()
    }

    /// Return all recorded propagation edges as (from, to, weight) triples.
    pub fn all_propagation_edges(&self) -> Vec<(String, String, f32)> {
        let edges = self.edges.read();
        let mut total_out: HashMap<&str, u32> = HashMap::new();
        for ((from, _), count) in edges.iter() {
            *total_out.entry(from.as_str()).or_insert(0) += count;
        }
        edges
            .iter()
            .map(|((from, to), count)| {
                let total = total_out.get(from.as_str()).copied().unwrap_or(1).max(1);
                let weight = *count as f32 / total as f32;
                (from.clone(), to.clone(), weight)
            })
            .collect()
    }

    /// Total number of recorded observations.
    pub fn total_observations(&self) -> usize {
        let profiles = self.error_profiles.read();
        profiles
            .values()
            .map(|p| p.values().sum::<u32>() as usize)
            .sum()
    }

    /// Clear all data (for testing).
    pub fn clear(&self) {
        self.error_index.write().clear();
        self.error_profiles.write().clear();
        self.edges.write().clear();
        self.priors.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_query_observation() {
        let store = CausalModelStore::new();

        let obs = CausalObservation::new("evt-1", "iri://skills/a", "Timeout", "timeout:db:5s")
            .with_propagation("iri://skills/b");
        store.record_observation(&obs);

        let skills = store.find_skills_by_error("timeout:db:5s");
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].0, "iri://skills/a");
        assert_eq!(skills[0].1, 1);
    }

    #[test]
    fn test_propagation_probability() {
        let store = CausalModelStore::new();

        store.record_propagation("iri://skills/a", "iri://skills/b");
        store.record_propagation("iri://skills/a", "iri://skills/c");
        store.record_propagation("iri://skills/a", "iri://skills/b");

        let p_b = store.propagation_probability("iri://skills/a", "iri://skills/b");
        let p_c = store.propagation_probability("iri://skills/a", "iri://skills/c");

        assert!((p_b - 2.0 / 3.0).abs() < 0.01);
        assert!((p_c - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_outgoing_edges() {
        let store = CausalModelStore::new();
        store.record_propagation("root", "mid");
        store.record_propagation("root", "leaf");

        let edges = store.outgoing_edges("root");
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].1.observation_count, 1);
        assert_eq!(edges[1].1.observation_count, 1);
    }

    #[test]
    fn test_priors() {
        let store = CausalModelStore::new();
        store.set_prior("iri://skills/x", 0.8);
        assert!((store.prior("iri://skills/x") - 0.8).abs() < 0.01);
        assert!((store.prior("iri://skills/unknown") - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_clear() {
        let store = CausalModelStore::new();
        store.record_observation(&CausalObservation::new("e1", "s1", "Err", "err:1"));
        assert_eq!(store.total_observations(), 1);
        store.clear();
        assert_eq!(store.total_observations(), 0);
    }
}
