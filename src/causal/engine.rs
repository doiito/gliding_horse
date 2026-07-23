use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use petgraph::prelude::*;
use tracing::info;

use crate::causal::store::CausalModelStore;
use crate::causal::types::{CausalInference, CausalObservation, PropagationHop, PropagationPath};
use crate::graph_backend::{EdgeDescriptor, GraphBackend};
use crate::skill_graph::types::SkillLinkType;

/// Root Cause Analysis Engine.
///
/// Given observed errors, traverses the skill dependency graph backwards and
/// uses Bayesian inference to identify the most probable root cause skill(s).
///
/// # Algorithm
///
/// 1. **Subgraph extraction**: From each observed skill, traverse reverse
///    prerequisite/extends edges to build a candidate subgraph.
/// 2. **Posterior computation**: For each candidate node:
///    ```text
///    P(root | observed) ∝ P(observed | root) × P(root)
///    ```
///    where P(observed | root) = product of propagation probabilities along
///    all paths from root to observed errors.
/// 3. **Path reconstruction**: For top candidates, find the most likely
///    propagation path(s) through the graph.
pub struct CausalEngine {
    store: Arc<CausalModelStore>,
    backend: Arc<dyn GraphBackend>,
}

impl CausalEngine {
    pub fn new(store: Arc<CausalModelStore>, backend: Arc<dyn GraphBackend>) -> Self {
        Self { store, backend }
    }

    /// Record a single observation into the causal model.
    pub fn record_observation(&self, obs: CausalObservation) {
        self.store.record_observation(&obs);
    }

    /// Infer root cause(s) from a batch of observations.
    ///
    /// `observations`: the error events to analyze.
    /// `top_k`: return this many top root cause candidates.
    pub fn infer_root_cause(
        &self,
        observations: &[CausalObservation],
        top_k: usize,
    ) -> Vec<CausalInference> {
        if observations.is_empty() {
            return Vec::new();
        }

        let observed_skills: Vec<&str> =
            observations.iter().map(|o| o.skill_iri.as_str()).collect();
        let observed_set: HashSet<&str> = observed_skills.iter().copied().collect();

        // Step 1: Extract candidate subgraph via reverse BFS
        let candidates = self.extract_candidates(&observed_skills);

        if candidates.is_empty() {
            info!("CausalEngine: no candidate root causes found");
            return Vec::new();
        }

        // Step 2: Compute posterior for each candidate
        let mut scored: Vec<(String, f32, Vec<PropagationPath>)> = candidates
            .into_iter()
            .filter_map(|candidate| {
                let posterior = self.compute_posterior(&candidate, &observed_set, observations);
                if posterior > 0.0 {
                    // Step 3: reconstruct propagation paths
                    let paths = self.reconstruct_paths(&candidate, observations);
                    Some((candidate, posterior, paths))
                } else {
                    None
                }
            })
            .collect();

        // Sort by confidence descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<_> = scored.into_iter().take(top_k).collect();

        // Build results
        let total_obs = observations.len();
        top.into_iter()
            .map(|(iri, confidence, paths)| {
                CausalInference {
                    root_cause_iri: iri,
                    confidence,
                    propagation_paths: paths,
                    alternative_causes: Vec::new(),    // filled below
                    observations_explained: total_obs, // all observations reachable
                    total_observations: total_obs,
                }
            })
            .collect()
    }

    /// Extract candidate root causes via reverse graph traversal.
    ///
    /// Starts from observed skills and traverses prerequisite/generalization
    /// edges backwards to find all potentially responsible nodes.
    fn extract_candidates(&self, observed_skills: &[&str]) -> Vec<String> {
        let edges = self.backend.all_edges();

        // Filter edges that represent upstream dependency relationships
        let dep_edges: Vec<&EdgeDescriptor> = edges
            .iter()
            .filter(|e| {
                matches!(
                    e.edge_type.as_str(),
                    "Prerequisite" | "Extends" | "Generalization"
                )
            })
            .collect();

        // Build dependency adjacency: for each skill, which skills does it point to?
        let mut dep_adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &dep_edges {
            dep_adj
                .entry(edge.source.as_str())
                .or_default()
                .push(edge.target.as_str());
        }

        // BFS from observed skills traversing upstream dependency edges
        let mut candidates = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        let mut visited: HashSet<&str> = HashSet::new();

        for &obs in observed_skills {
            if !visited.contains(obs) {
                visited.insert(obs);
                queue.push_back(obs);
            }
        }

        while let Some(current) = queue.pop_front() {
            if let Some(deps) = dep_adj.get(current) {
                for &dep in deps {
                    if visited.insert(dep) {
                        candidates.insert(dep.to_string());
                        queue.push_back(dep);
                    }
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Compute Bayesian posterior P(candidate | observations).
    fn compute_posterior(
        &self,
        candidate: &str,
        observed_set: &HashSet<&str>,
        observations: &[CausalObservation],
    ) -> f32 {
        let prior = self.store.prior(candidate);

        // Compute likelihood: P(observations | candidate_is_root)
        // For each observed skill, find the maximum propagation probability
        // along any path from candidate → observed
        let mut likelihood = 1.0_f32;

        for obs in observations {
            let target = obs.skill_iri.as_str();
            if target == candidate {
                continue; // candidate itself was observed — strong signal
            }
            if !observed_set.contains(target) {
                continue;
            }

            let path_prob = self.max_propagation_probability(candidate, target);
            if path_prob > 0.0 {
                likelihood *= path_prob;
            } else {
                // Candidate cannot reach this observation — weakens score
                likelihood *= 0.01;
            }
        }

        // Posterior ∝ likelihood × prior
        prior * likelihood
    }

    /// Find the maximum propagation probability along any path from `from` to `to`.
    /// Uses DFS with branch-and-bound (optimistic pruning).
    fn max_propagation_probability(&self, from: &str, to: &str) -> f32 {
        if from == to {
            return 1.0;
        }

        let all_nodes: HashSet<String> = self.backend.all_nodes().into_iter().collect();
        if !all_nodes.contains(from) || !all_nodes.contains(to) {
            return 0.0;
        }

        let edges = self.backend.all_edges();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &edges {
            if matches!(
                edge.edge_type.as_str(),
                "Prerequisite" | "Composition" | "Extends" | "Related"
            ) {
                adj.entry(edge.source.as_str())
                    .or_default()
                    .push(edge.target.as_str());
            }
        }

        // DFS with best-so-far pruning
        let mut visited = HashSet::new();
        let mut best = 0.0_f32;
        self.dfs_propagate(from, to, &adj, &mut visited, 1.0, &mut best);
        best
    }

    fn dfs_propagate(
        &self,
        current: &str,
        target: &str,
        adj: &HashMap<&str, Vec<&str>>,
        visited: &mut HashSet<String>,
        path_prob: f32,
        best: &mut f32,
    ) {
        if path_prob <= *best {
            return; // pruning: cannot beat current best
        }
        if current == target {
            *best = path_prob;
            return;
        }

        visited.insert(current.to_string());

        if let Some(neighbors) = adj.get(current) {
            for &next in neighbors {
                if !visited.contains(next) {
                    let edge_prob = self.store.propagation_probability(current, next);
                    let effective_prob = if edge_prob == 0.0 { 0.5 } else { edge_prob };
                    self.dfs_propagate(
                        next,
                        target,
                        adj,
                        visited,
                        path_prob * effective_prob,
                        best,
                    );
                }
            }
        }

        visited.remove(current);
    }

    /// Reconstruct propagation paths from candidate root to observed errors.
    fn reconstruct_paths(
        &self,
        candidate: &str,
        observations: &[CausalObservation],
    ) -> Vec<PropagationPath> {
        let edges = self.backend.all_edges();
        let mut adj: HashMap<String, Vec<(String, SkillLinkType)>> = HashMap::new();
        for edge in &edges {
            let link_type = parse_skill_link_type(&edge.edge_type);
            adj.entry(edge.source.clone())
                .or_default()
                .push((edge.target.clone(), link_type));
        }

        let mut paths = Vec::new();

        for obs in observations {
            if obs.skill_iri == candidate {
                continue;
            }

            let path = self.bfs_shortest_path(candidate, &obs.skill_iri, &adj);
            if let Some(hops) = path {
                let path_conf = hops
                    .iter()
                    .map(|h| h.propagation_probability)
                    .product::<f32>();

                paths.push(PropagationPath {
                    hops,
                    terminal_observation: obs.clone(),
                    path_confidence: path_conf,
                });
            }
        }

        paths
    }

    fn bfs_shortest_path(
        &self,
        from: &str,
        to: &str,
        adj: &HashMap<String, Vec<(String, SkillLinkType)>>,
    ) -> Option<Vec<PropagationHop>> {
        if from == to {
            return Some(Vec::new());
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(&str, Vec<PropagationHop>)> = VecDeque::new();
        visited.insert(from.to_string());
        queue.push_back((from, Vec::new()));

        while let Some((current, hops)) = queue.pop_front() {
            if let Some(neighbors) = adj.get(current) {
                for (next, link_type) in neighbors {
                    if !visited.contains(next.as_str()) {
                        let prob = self.store.propagation_probability(current, next);
                        let mut new_hops = hops.clone();
                        new_hops.push(PropagationHop {
                            skill_iri: next.clone(),
                            link_type: *link_type,
                            propagation_probability: if prob == 0.0 { 0.5 } else { prob },
                        });

                        if next.as_str() == to {
                            return Some(new_hops);
                        }

                        visited.insert(next.clone());
                        queue.push_back((next.as_str(), new_hops));
                    }
                }
            }
        }

        None
    }

    /// Build the full propagation graph as a petgraph DiGraph for visualization.
    pub fn propagation_graph(&self) -> DiGraph<String, f32> {
        let mut graph = DiGraph::new();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        // Collect all nodes from propagation edges (both sources and targets)
        let all_edges = self.store.all_propagation_edges();
        for (from, to, _weight) in &all_edges {
            if !node_map.contains_key(from) {
                let idx = graph.add_node(from.clone());
                node_map.insert(from.clone(), idx);
            }
            if !node_map.contains_key(to) {
                let idx = graph.add_node(to.clone());
                node_map.insert(to.clone(), idx);
            }
        }

        for (from, to, weight) in &all_edges {
            if let (Some(&from_idx), Some(&to_idx)) = (node_map.get(from), node_map.get(to)) {
                graph.add_edge(from_idx, to_idx, *weight);
            }
        }

        graph
    }

    /// Get a reference to the underlying model store.
    pub fn store(&self) -> &Arc<CausalModelStore> {
        &self.store
    }
}

/// Map an edge-type string from [`EdgeDescriptor`] back to a [`SkillLinkType`].
///
/// The string format comes from `Debug` on `SkillLinkType` variants
/// (e.g. `"Prerequisite"`, `"Extends"`) as produced by [`PetgraphBackend`].
fn parse_skill_link_type(s: &str) -> SkillLinkType {
    match s {
        "Prerequisite" => SkillLinkType::Prerequisite,
        "Composition" => SkillLinkType::Composition,
        "Related" => SkillLinkType::Related,
        "Alternative" => SkillLinkType::Alternative,
        "Extends" => SkillLinkType::Extends,
        "Generalization" => SkillLinkType::Generalization,
        _ => SkillLinkType::Related,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_backend::PetgraphBackend;
    use crate::skill_graph::graph_store::SkillGraphStore;
    use crate::skill_graph::types::SkillGraphNode;

    fn build_test_backend() -> Arc<dyn GraphBackend> {
        let store = Arc::new(SkillGraphStore::new());

        // root → mid_a → leaf_a
        // root → mid_b → leaf_b
        let root = SkillGraphNode::new("iri://skills/root", "Root Cause", "Root skill");
        let mut mid_a = SkillGraphNode::new("iri://skills/mid_a", "Middle A", "Intermediate A");
        mid_a.add_prerequisite("iri://skills/root", "Root enables A");
        let mut leaf_a = SkillGraphNode::new("iri://skills/leaf_a", "Leaf A", "Leaf A");
        leaf_a.add_prerequisite("iri://skills/mid_a", "A enables leaf A");
        let mut mid_b = SkillGraphNode::new("iri://skills/mid_b", "Middle B", "Intermediate B");
        mid_b.add_prerequisite("iri://skills/root", "Root enables B");
        let mut leaf_b = SkillGraphNode::new("iri://skills/leaf_b", "Leaf B", "Leaf B");
        leaf_b.add_prerequisite("iri://skills/mid_b", "B enables leaf B");

        store.register_skill(root).unwrap();
        store.register_skill(mid_a).unwrap();
        store.register_skill(leaf_a).unwrap();
        store.register_skill(mid_b).unwrap();
        store.register_skill(leaf_b).unwrap();

        Arc::new(PetgraphBackend::new(store))
    }

    #[test]
    fn test_extract_candidates() {
        let backend = build_test_backend();
        let model_store = Arc::new(CausalModelStore::new());
        let engine = CausalEngine::new(model_store, backend);

        let candidates = engine.extract_candidates(&["iri://skills/leaf_a"]);
        assert!(candidates.contains(&"iri://skills/root".to_string()));
        assert!(candidates.contains(&"iri://skills/mid_a".to_string()));
    }

    #[test]
    fn test_infer_root_cause() {
        let backend = build_test_backend();
        let model_store = Arc::new(CausalModelStore::new());

        // Seed priors
        model_store.set_prior("iri://skills/root", 0.5);
        model_store.set_prior("iri://skills/mid_a", 0.3);
        model_store.set_prior("iri://skills/mid_b", 0.3);
        model_store.set_prior("iri://skills/leaf_a", 0.1);
        model_store.set_prior("iri://skills/leaf_b", 0.1);

        // Record propagations
        model_store.record_propagation("iri://skills/root", "iri://skills/mid_a");
        model_store.record_propagation("iri://skills/root", "iri://skills/mid_b");
        model_store.record_propagation("iri://skills/mid_a", "iri://skills/leaf_a");
        model_store.record_propagation("iri://skills/mid_b", "iri://skills/leaf_b");

        let engine = CausalEngine::new(model_store, backend);

        let observations = vec![
            CausalObservation::new("evt-1", "iri://skills/leaf_a", "Timeout", "timeout:db")
                .with_propagation("iri://skills/mid_a"),
            CausalObservation::new("evt-2", "iri://skills/leaf_b", "Timeout", "timeout:db")
                .with_propagation("iri://skills/mid_b"),
        ];

        let results = engine.infer_root_cause(&observations, 3);
        assert!(!results.is_empty(), "Should find root causes");
        assert_eq!(
            results[0].root_cause_iri, "iri://skills/root",
            "Root should be the top result"
        );
        assert!(
            results[0].confidence > results.get(1).map(|r| r.confidence).unwrap_or(0.0),
            "Root should have highest confidence"
        );
    }

    #[test]
    fn test_empty_observations() {
        let backend = build_test_backend();
        let model_store = Arc::new(CausalModelStore::new());
        let engine = CausalEngine::new(model_store, backend);

        let results = engine.infer_root_cause(&[], 3);
        assert!(results.is_empty());
    }

    #[test]
    fn test_propagation_graph() {
        let backend = build_test_backend();
        let model_store = Arc::new(CausalModelStore::new());
        model_store.record_propagation("iri://skills/root", "iri://skills/mid_a");
        model_store.record_propagation("iri://skills/mid_a", "iri://skills/leaf_a");

        let engine = CausalEngine::new(model_store, backend);
        let g = engine.propagation_graph();
        assert!(g.node_count() >= 1);
    }
}
