use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::algo;
use petgraph::prelude::*;
use petgraph::Direction;

use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::{ScoredNode, SkillLinkType};

pub struct SkillGraphAlgorithms {
    graph: DiGraph<String, SkillLinkType>,
    node_map: HashMap<String, NodeIndex>,
}

impl SkillGraphAlgorithms {
    pub fn from_store(store: &SkillGraphStore) -> Self {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        for skill in store.list_all_skills() {
            let idx = graph.add_node(skill.skill_iri.clone());
            node_map.insert(skill.skill_iri, idx);
        }

        for skill in store.list_all_skills() {
            if let Some(&from_idx) = node_map.get(&skill.skill_iri) {
                for link in &skill.links {
                    if let Some(&to_idx) = node_map.get(&link.target_iri) {
                        graph.add_edge(from_idx, to_idx, link.link_type);
                    }
                }
            }
        }

        Self { graph, node_map }
    }

    pub fn rebuild(&mut self, store: &SkillGraphStore) {
        *self = Self::from_store(store);
    }

    pub fn page_rank(&self, damping: f32) -> Vec<ScoredNode> {
        if self.graph.node_count() == 0 {
            return Vec::new();
        }
        let ranks = algo::page_rank(&self.graph, damping, 100);
        self.zip_scores(&ranks)
    }

    pub fn betweenness_centrality(&self) -> Vec<ScoredNode> {
        let n = self.graph.node_count();
        if n == 0 {
            return Vec::new();
        }
        let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();
        let mut centrality = vec![0.0f64; n];

        for &s in &nodes {
            Self::brandes_bfs(&self.graph, s, &nodes, &mut centrality);
        }

        if n > 2 {
            let norm = 1.0f64 / ((n - 1) * (n - 2)) as f64;
            for c in &mut centrality {
                *c *= norm;
            }
        }

        nodes
            .iter()
            .enumerate()
            .map(|(i, &node_idx)| ScoredNode {
                iri: self.graph[node_idx].clone(),
                score: centrality[i],
            })
            .collect()
    }

    fn brandes_bfs(
        graph: &DiGraph<String, SkillLinkType>,
        s: NodeIndex,
        nodes: &[NodeIndex],
        centrality: &mut [f64],
    ) {
        let mut stack: Vec<NodeIndex> = Vec::new();
        let mut predecessors: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
        let mut sigma: HashMap<NodeIndex, f64> = HashMap::new();
        let mut dist: HashMap<NodeIndex, i32> = HashMap::new();
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();

        sigma.insert(s, 1.0);
        dist.insert(s, 0);
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let v_dist = dist[&v];

            for w in graph.neighbors_directed(v, Direction::Outgoing) {
                let new_dist = v_dist + 1;
                let is_shorter = dist.get(&w).map_or(true, |&d| new_dist < d);

                if is_shorter {
                    dist.insert(w, new_dist);
                    queue.push_back(w);
                    predecessors.entry(w).or_default().clear();
                }

                if dist.get(&w) == Some(&new_dist) {
                    predecessors.entry(w).or_default().push(v);
                    let sv = sigma.get(&v).copied().unwrap_or(0.0);
                    *sigma.entry(w).or_insert(0.0) += sv;
                }
            }
        }

        let mut delta: HashMap<NodeIndex, f64> = HashMap::new();
        while let Some(w) = stack.pop() {
            if let Some(preds) = predecessors.get(&w) {
                let sw = sigma.get(&w).copied().unwrap_or(1.0);
                let dw = delta.get(&w).copied().unwrap_or(0.0);
                for &v in preds {
                    let sv = sigma.get(&v).copied().unwrap_or(0.0);
                    let contribution = (sv / sw) * (1.0 + dw);
                    *delta.entry(v).or_insert(0.0) += contribution;
                }
            }
            if w != s {
                if let Some(pos) = nodes.iter().position(|&n| n == w) {
                    if pos < centrality.len() {
                        centrality[pos] += delta.get(&w).copied().unwrap_or(0.0);
                    }
                }
            }
        }
    }

    pub fn detect_communities(&self) -> Vec<Vec<String>> {
        let n = self.graph.node_count();
        if n == 0 {
            return Vec::new();
        }
        let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();
        let node_to_idx: HashMap<NodeIndex, usize> =
            nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        let mut labels: Vec<usize> = (0..n).collect();

        for _ in 0..20 {
            let mut changed = false;
            for i in 0..n {
                let mut neighbor_labels: HashMap<usize, usize> = HashMap::new();

                for neighbor in self.graph.neighbors_directed(nodes[i], Direction::Outgoing) {
                    if let Some(&ni) = node_to_idx.get(&neighbor) {
                        *neighbor_labels.entry(labels[ni]).or_insert(0) += 1;
                    }
                }
                for neighbor in self.graph.neighbors_directed(nodes[i], Direction::Incoming) {
                    if let Some(&ni) = node_to_idx.get(&neighbor) {
                        *neighbor_labels.entry(labels[ni]).or_insert(0) += 1;
                    }
                }

                if neighbor_labels.is_empty() {
                    continue;
                }

                let best_label = neighbor_labels
                    .into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                    .map(|(label, _)| label)
                    .unwrap();

                if labels[i] != best_label {
                    labels[i] = best_label;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut community_map: HashMap<usize, Vec<String>> = HashMap::new();
        for (i, &node) in nodes.iter().enumerate() {
            community_map
                .entry(labels[i])
                .or_default()
                .push(self.graph[node].clone());
        }

        community_map.into_values().collect()
    }

    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        let from_idx = self.node_map.get(from).copied()?;
        let to_idx = self.node_map.get(to).copied()?;

        if from_idx == to_idx {
            return Some(vec![self.graph[from_idx].clone()]);
        }

        let mut visited = HashSet::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut queue: VecDeque<NodeIndex> = VecDeque::new();

        visited.insert(from_idx);
        queue.push_back(from_idx);

        while let Some(current) = queue.pop_front() {
            if current == to_idx {
                let mut path = Vec::new();
                let mut node = current;
                path.push(self.graph[node].clone());
                while let Some(&p) = parent.get(&node) {
                    path.push(self.graph[p].clone());
                    node = p;
                }
                path.reverse();
                return Some(path);
            }

            for neighbor in self.graph.neighbors_directed(current, Direction::Outgoing) {
                if visited.insert(neighbor) {
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    pub fn all_paths(&self, from: &str, to: &str, max_depth: usize) -> Vec<Vec<String>> {
        let from_idx = match self.node_map.get(from) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };
        let to_idx = match self.node_map.get(to) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        if max_depth == 0 || (max_depth == 1 && from_idx != to_idx) {
            return Vec::new();
        }

        if from_idx == to_idx {
            return vec![vec![self.graph[from_idx].clone()]];
        }

        let max_inter = if max_depth >= 10 {
            None
        } else {
            Some(max_depth - 1)
        };

        algo::all_simple_paths::<Vec<_>, _>(&self.graph, from_idx, to_idx, 0, max_inter)
            .map(|path: Vec<NodeIndex>| path.into_iter().map(|n| self.graph[n].clone()).collect())
            .collect()
    }

    pub fn prerequisite_chain(&self, skill_iri: &str) -> Vec<Vec<String>> {
        let start_idx = match self.node_map.get(skill_iri) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        let mut chains = Vec::new();
        let mut current_path = vec![start_idx];
        self.collect_prerequisite_chains(start_idx, &mut current_path, &mut chains);

        chains
            .into_iter()
            .map(|path| path.into_iter().map(|n| self.graph[n].clone()).collect())
            .collect()
    }

    fn collect_prerequisite_chains(
        &self,
        node: NodeIndex,
        current_path: &mut Vec<NodeIndex>,
        chains: &mut Vec<Vec<NodeIndex>>,
    ) {
        let prerequisite_edges: Vec<NodeIndex> = self
            .graph
            // A prerequisite link is stored as `skill -> prerequisite`.
            // Traversing outgoing edges therefore follows the requirements of
            // the requested skill; incoming edges would instead find skills
            // that depend on it.
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| *e.weight() == SkillLinkType::Prerequisite)
            .map(|e| e.target())
            .collect();

        if prerequisite_edges.is_empty() {
            chains.push(current_path.clone());
            return;
        }

        for source in prerequisite_edges {
            if !current_path.contains(&source) {
                current_path.push(source);
                self.collect_prerequisite_chains(source, current_path, chains);
                current_path.pop();
            }
        }
    }

    pub fn detect_cycles(&self) -> Vec<Vec<String>> {
        let sccs = algo::tarjan_scc(&self.graph);

        sccs.into_iter()
            .filter(|scc| {
                if scc.len() > 1 {
                    return true;
                }
                if scc.len() == 1 {
                    let node = scc[0];
                    return self
                        .graph
                        .edges_directed(node, Direction::Outgoing)
                        .any(|e| e.target() == node);
                }
                false
            })
            .map(|scc| scc.into_iter().map(|n| self.graph[n].clone()).collect())
            .collect()
    }

    fn zip_scores(&self, scores: &[f32]) -> Vec<ScoredNode> {
        let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();
        nodes
            .iter()
            .enumerate()
            .filter_map(|(i, &node_idx)| {
                if i < scores.len() {
                    Some(ScoredNode {
                        iri: self.graph[node_idx].clone(),
                        score: scores[i] as f64,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill_graph::graph_store::SkillGraphStore;
    use crate::skill_graph::types::{SkillGraphNode, SkillLink};

    fn build_test_store() -> SkillGraphStore {
        let store = SkillGraphStore::new();

        let s1 = SkillGraphNode::new("iri://skills/a", "Skill A", "First skill").with_tag("core");
        let s2 = SkillGraphNode::new("iri://skills/b", "Skill B", "Second skill").with_tag("core");
        let s3 =
            SkillGraphNode::new("iri://skills/c", "Skill C", "Third skill").with_tag("advanced");
        let s4 =
            SkillGraphNode::new("iri://skills/d", "Skill D", "Fourth skill").with_tag("advanced");

        let mut s1_with_links = s1.clone();
        s1_with_links.add_prerequisite("iri://skills/b", "B is prerequisite of A");
        s1_with_links.add_related("iri://skills/c", "Related to C");

        let mut s2_with_links = s2.clone();
        s2_with_links.add_prerequisite("iri://skills/d", "D is prerequisite of B");

        store.register_skill(s1_with_links).unwrap();
        store.register_skill(s2_with_links).unwrap();
        store.register_skill(s3).unwrap();
        store.register_skill(s4).unwrap();

        store
    }

    #[test]
    fn test_from_store_and_rebuild() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);

        assert_eq!(algo.graph.node_count(), 4);
        assert_eq!(algo.graph.edge_count(), 3);
        assert!(algo.node_map.contains_key("iri://skills/a"));
        assert!(algo.node_map.contains_key("iri://skills/b"));
        assert!(algo.node_map.contains_key("iri://skills/c"));
        assert!(algo.node_map.contains_key("iri://skills/d"));

        let mut algo2 = SkillGraphAlgorithms::from_store(&store);
        store.remove_skill("iri://skills/d").unwrap();
        algo2.rebuild(&store);
        assert_eq!(algo2.graph.node_count(), 3);
        assert_eq!(algo2.graph.edge_count(), 2);
    }

    #[test]
    fn test_page_rank() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);
        let ranks = algo.page_rank(0.85);

        assert_eq!(ranks.len(), 4);
        for scored in &ranks {
            assert!(!scored.iri.is_empty());
            assert!(scored.score > 0.0);
        }
        // Nodes with more in-degree should have higher PageRank
        let rank_a = ranks.iter().find(|s| s.iri == "iri://skills/a").unwrap();
        let rank_b = ranks.iter().find(|s| s.iri == "iri://skills/b").unwrap();
        assert!(rank_b.score > rank_a.score); // B is pointed to by A's prerequisite link
    }

    #[test]
    fn test_betweenness_centrality() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);
        let centrality = algo.betweenness_centrality();

        assert_eq!(centrality.len(), 4);
        for scored in &centrality {
            assert!(!scored.iri.is_empty());
            assert!(scored.score >= 0.0);
        }
    }

    #[test]
    fn test_shortest_path() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);

        let path = algo.shortest_path("iri://skills/a", "iri://skills/d");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3); // a -> b -> d
        assert_eq!(path[0], "iri://skills/a");
        assert_eq!(path[1], "iri://skills/b");
        assert_eq!(path[2], "iri://skills/d");
    }

    #[test]
    fn test_shortest_path_same_node() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);

        let path = algo.shortest_path("iri://skills/a", "iri://skills/a");
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec!["iri://skills/a"]);
    }

    #[test]
    fn test_shortest_path_missing_node() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);

        assert!(algo
            .shortest_path("iri://skills/a", "iri://skills/missing")
            .is_none());
        assert!(algo
            .shortest_path("iri://skills/missing", "iri://skills/a")
            .is_none());
    }

    #[test]
    fn test_all_paths() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);

        let paths = algo.all_paths("iri://skills/a", "iri://skills/d", 5);
        assert!(!paths.is_empty());
        for p in &paths {
            assert_eq!(p.first().unwrap(), "iri://skills/a");
            assert_eq!(p.last().unwrap(), "iri://skills/d");
        }
    }

    #[test]
    fn test_all_paths_zero_depth() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);

        let paths = algo.all_paths("iri://skills/a", "iri://skills/d", 0);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_all_paths_same_node() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);

        let paths = algo.all_paths("iri://skills/a", "iri://skills/a", 5);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec!["iri://skills/a"]);
    }

    #[test]
    fn test_prerequisite_chain() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);

        let chains = algo.prerequisite_chain("iri://skills/a");
        assert_eq!(
            chains,
            vec![vec![
                "iri://skills/a".to_string(),
                "iri://skills/b".to_string(),
                "iri://skills/d".to_string(),
            ]]
        );
    }

    #[test]
    fn test_prerequisite_chain_missing() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);
        assert!(algo.prerequisite_chain("iri://skills/missing").is_empty());
    }

    #[test]
    fn test_detect_cycles_acyclic() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);
        let cycles = algo.detect_cycles();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_detect_cycles_with_cycle() {
        let store = SkillGraphStore::new();

        let mut s1 = SkillGraphNode::new("iri://skills/x", "Skill X", "X");
        s1.add_prerequisite("iri://skills/y", "X -> Y");
        let mut s2 = SkillGraphNode::new("iri://skills/y", "Skill Y", "Y");
        s2.add_prerequisite("iri://skills/z", "Y -> Z");
        let mut s3 = SkillGraphNode::new("iri://skills/z", "Skill Z", "Z");
        s3.add_prerequisite("iri://skills/x", "Z -> X (cycle)");

        store.register_skill(s1).unwrap();
        store.register_skill(s2).unwrap();
        store.register_skill(s3).unwrap();

        let algo = SkillGraphAlgorithms::from_store(&store);
        let cycles = algo.detect_cycles();
        assert!(!cycles.is_empty());
        // All three nodes should be in one cycle
        assert_eq!(cycles[0].len(), 3);
    }

    #[test]
    fn test_detect_communities() {
        let store = build_test_store();
        let algo = SkillGraphAlgorithms::from_store(&store);
        let communities = algo.detect_communities();

        assert!(!communities.is_empty());
        let total: usize = communities.iter().map(|c| c.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_empty_graph() {
        let store = SkillGraphStore::new();
        let algo = SkillGraphAlgorithms::from_store(&store);

        assert!(algo.page_rank(0.85).is_empty());
        assert!(algo.betweenness_centrality().is_empty());
        assert!(algo.detect_communities().is_empty());
        assert!(algo.detect_cycles().is_empty());
        assert!(algo.shortest_path("a", "b").is_none());
        assert!(algo.all_paths("a", "b", 5).is_empty());
        assert!(algo.prerequisite_chain("a").is_empty());
    }

    #[test]
    fn test_no_path_between_unconnected() {
        let store = SkillGraphStore::new();
        let s1 = SkillGraphNode::new("iri://skills/a", "A", "Node A");
        let s2 = SkillGraphNode::new("iri://skills/b", "B", "Node B");
        store.register_skill(s1).unwrap();
        store.register_skill(s2).unwrap();

        let algo = SkillGraphAlgorithms::from_store(&store);
        assert!(algo
            .shortest_path("iri://skills/a", "iri://skills/b")
            .is_none());
    }

    #[test]
    fn test_self_loop_detected_as_cycle() {
        let store = SkillGraphStore::new();
        let mut s1 = SkillGraphNode::new("iri://skills/self", "Self", "Has self-loop");
        s1.links.push(SkillLink::new(
            SkillLinkType::Related,
            "iri://skills/self".to_string(),
        ));
        store.register_skill(s1).unwrap();

        let algo = SkillGraphAlgorithms::from_store(&store);
        let cycles = algo.detect_cycles();
        assert!(!cycles.is_empty());
        assert_eq!(cycles[0], vec!["iri://skills/self"]);
    }
}
