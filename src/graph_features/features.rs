use std::collections::HashMap;
use std::sync::Arc;

use crate::graph_backend::{Direction, FeatureGraph};

/// Numeric feature vector for a skill graph node.
///
/// All features are normalized to [0.0, 1.0] or [-1.0, 1.0] for compatibility.
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    pub skill_iri: String,

    // ── Structural (from SkillGraphEmbedder) ──
    /// Poincaré ball radius (prerequisite depth encoding)
    pub poincare_radius: f64,
    /// Poincaré angular coordinates (domain cluster encoding)
    pub poincare_angles: Vec<f64>,

    // ── Graph metrics (from SkillGraphAlgorithms) ──
    pub in_degree: f32,
    pub out_degree: f32,
    pub page_rank: f32,
    pub betweenness: f32,
    /// Community ID from label propagation (0 = unassigned)
    pub community_id: i32,

    // ── Skill attributes ──
    /// Node type encoded as float: Atomic=0.0, Composite=0.2, MOC=0.4, KnowledgeFragment=0.6, MCPTool=0.8, Bootstrap=1.0
    pub node_type: f32,
    /// Maturity encoded as float: experimental=0.0, beta=0.33, stable=0.66, deprecated=1.0
    pub maturity: f32,
    pub success_rate: f32,
    pub security_level: f32,
    pub tag_count: f32,
    /// Average success rate of neighbors (0.0–1.0)
    pub avg_neighbor_success_rate: f32,
}

impl NodeFeatures {
    /// Convert to a flat f32 vector for similarity computation.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(16);
        v.push(self.poincare_radius as f32);
        for &a in self.poincare_angles.iter().take(3) {
            v.push(a as f32);
        }
        // Pad to exactly three angular dimensions.  Use the output length,
        // rather than the immutable source length, so short inputs terminate.
        while v.len() < 4 {
            v.push(0.0);
        }
        v.push(self.in_degree);
        v.push(self.out_degree);
        v.push(self.page_rank);
        v.push(self.betweenness);
        v.push(self.node_type);
        v.push(self.maturity);
        v.push(self.success_rate);
        v.push(self.security_level);
        v.push(self.tag_count / 20.0); // normalize tag count
        v.push(self.avg_neighbor_success_rate);
        v
    }
}

fn parse_node_type(raw: &str) -> f32 {
    match raw {
        "Atomic" => 0.0,
        "Composite" => 0.2,
        "MOC" => 0.4,
        "KnowledgeFragment" => 0.6,
        "MCPTool" => 0.8,
        "Bootstrap" => 1.0,
        _ => 0.5,
    }
}

fn parse_maturity(raw: &str) -> f32 {
    match raw {
        "experimental" => 0.0,
        "beta" => 0.33,
        "stable" => 0.66,
        "deprecated" => 1.0,
        _ => 0.5,
    }
}

/// Converts graph topology (via `FeatureGraph` trait) → numeric feature vectors.
pub struct FeatureExtractor {
    graph: Arc<dyn FeatureGraph>,
}

impl FeatureExtractor {
    pub fn new(graph: Arc<dyn FeatureGraph>) -> Self {
        Self { graph }
    }

    /// Extract features for a single skill.
    pub fn extract(&self, skill_iri: &str) -> Option<NodeFeatures> {
        let data = self.graph.node_data(skill_iri)?;

        let in_degree = self.graph.degree(skill_iri, Direction::Incoming) as f32;
        let out_degree = self.graph.degree(skill_iri, Direction::Outgoing) as f32;

        let page_rank = self
            .graph
            .page_rank(0.85)
            .into_iter()
            .find(|(iri, _)| iri == skill_iri)
            .map(|(_, s)| s as f32)
            .unwrap_or(0.0);

        let betweenness = self
            .graph
            .betweenness_centrality()
            .into_iter()
            .find(|(iri, _)| iri == skill_iri)
            .map(|(_, s)| s as f32)
            .unwrap_or(0.0);

        let communities = self.graph.detect_communities();
        let community_id = communities
            .iter()
            .position(|c| c.iter().any(|iri| iri == skill_iri))
            .map(|i| i as i32)
            .unwrap_or(-1);

        let node_type = data
            .get("node_type")
            .and_then(|v| v.as_str())
            .map(parse_node_type)
            .unwrap_or(0.5);

        let maturity = data
            .get("maturity")
            .and_then(|v| v.as_str())
            .map(parse_maturity)
            .unwrap_or(0.5);

        let success_rate = data
            .get("success_rate")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.5);

        let security_level = data
            .get("security_level")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32 / 4.0)
            .unwrap_or(0.5);

        let tag_count = data
            .get("tag_count")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.0);

        // Average neighbor success rate
        let neighbor_iris = self.graph.neighbors(skill_iri, Direction::Outgoing);
        let total_success: f32 = neighbor_iris
            .iter()
            .filter_map(|n_iri| {
                self.graph
                    .node_data(n_iri)
                    .and_then(|d| d.get("success_rate").and_then(|v| v.as_f64()))
                    .map(|v| v as f32)
            })
            .sum();
        let avg_neighbor_success_rate = if neighbor_iris.is_empty() {
            0.5
        } else {
            total_success / neighbor_iris.len() as f32
        };

        Some(NodeFeatures {
            skill_iri: skill_iri.to_string(),
            poincare_radius: 0.0,
            poincare_angles: vec![0.0, 0.0, 0.0],
            in_degree,
            out_degree,
            page_rank,
            betweenness,
            community_id,
            node_type,
            maturity,
            success_rate,
            security_level,
            tag_count,
            avg_neighbor_success_rate,
        })
    }

    /// Extract features for all nodes in the graph.
    pub fn extract_all(&self) -> HashMap<String, NodeFeatures> {
        let iris = self.graph.all_nodes();
        iris.iter()
            .filter_map(|iri| {
                let features = self.extract(iri)?;
                Some((iri.clone(), features))
            })
            .collect()
    }
}

/// Iterative neighborhood aggregation (simplified GraphSAGE).
///
/// For each node, computes the mean of its neighbors' features and
/// combines with its own features. After K rounds, each node's embedding
/// incorporates information from its K-hop neighborhood.
pub struct NeighborhoodAggregator;

impl NeighborhoodAggregator {
    /// Run K rounds of mean-pool neighborhood aggregation.
    ///
    /// `features`: initial node features (keyed by IRI).
    /// `adjacency`: adjacency list — for each node IRI, the list of neighbor IRIs.
    /// `rounds`: number of aggregation rounds (default: 2).
    /// `self_weight`: weight for the node's own features vs neighbors (0.0–1.0).
    ///
    /// Returns aggregated embeddings keyed by IRI.
    pub fn aggregate(
        features: &HashMap<String, NodeFeatures>,
        adjacency: &HashMap<String, Vec<String>>,
        rounds: usize,
        self_weight: f32,
    ) -> HashMap<String, Vec<f32>> {
        let mut current: HashMap<String, Vec<f32>> = features
            .iter()
            .map(|(iri, f)| (iri.clone(), f.to_dense()))
            .collect();

        for _round in 0..rounds {
            let mut next = HashMap::new();

            for (iri, embedding) in &current {
                let neighbors = adjacency.get(iri.as_str()).cloned().unwrap_or_default();

                // Mean-pool neighbors
                let mut neighbor_mean = vec![0.0_f32; embedding.len()];
                let mut neighbor_count = 0;

                for n_iri in &neighbors {
                    if let Some(n_emb) = current.get(n_iri) {
                        for (i, &v) in n_emb.iter().enumerate() {
                            neighbor_mean[i] += v;
                        }
                        neighbor_count += 1;
                    }
                }

                if neighbor_count > 0 {
                    for v in &mut neighbor_mean {
                        *v /= neighbor_count as f32;
                    }
                }

                // Combine self + neighbors
                let mut combined = Vec::with_capacity(embedding.len());
                for i in 0..embedding.len() {
                    let val = self_weight * embedding[i] + (1.0 - self_weight) * neighbor_mean[i];
                    combined.push(val);
                }

                next.insert(iri.clone(), combined);
            }

            current = next;
        }

        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_backend::SkillGraphFeatureGraph;
    use crate::skill_graph::graph_algorithms::SkillGraphAlgorithms;
    use crate::skill_graph::graph_store::SkillGraphStore;
    use crate::skill_graph::types::SkillGraphNode;

    /// Create a shared store + FeatureGraph pair so registered skills are visible.
    fn test_fixture() -> (Arc<dyn FeatureGraph>, Arc<SkillGraphStore>) {
        let store = Arc::new(SkillGraphStore::new());
        let algo = Arc::new(SkillGraphAlgorithms::from_store(&store));
        let graph = Arc::new(SkillGraphFeatureGraph::new(store.clone(), algo));
        (graph, store)
    }

    #[test]
    fn test_extract_features() {
        let (graph, store) = test_fixture();
        store
            .register_skill(SkillGraphNode::new(
                "iri://skills/test",
                "Test",
                "A test skill",
            ))
            .unwrap();

        let extractor = FeatureExtractor::new(graph);
        let features = extractor.extract("iri://skills/test");
        assert!(features.is_some());
        let f = features.unwrap();
        assert_eq!(f.skill_iri, "iri://skills/test");
        assert_eq!(f.in_degree, 0.0);
        assert_eq!(f.out_degree, 0.0);
        assert_eq!(f.node_type, 0.0); // Atomic
    }

    #[test]
    fn test_dense_vector_length() {
        let (graph, store) = test_fixture();
        store
            .register_skill(SkillGraphNode::new("iri://skills/t1", "T1", "Test 1"))
            .unwrap();
        store
            .register_skill(SkillGraphNode::new("iri://skills/t2", "T2", "Test 2"))
            .unwrap();

        let extractor = FeatureExtractor::new(graph);
        let features = extractor.extract_all();
        assert_eq!(features.len(), 2);
        for (_, f) in &features {
            let dense = f.to_dense();
            assert!(dense.len() >= 12);
        }
    }

    #[test]
    fn test_dense_vector_pads_short_angle_vectors_without_looping() {
        let features = NodeFeatures {
            skill_iri: "iri://skills/short".to_string(),
            poincare_radius: 0.5,
            poincare_angles: vec![0.25],
            in_degree: 0.0,
            out_degree: 0.0,
            page_rank: 0.0,
            betweenness: 0.0,
            community_id: 0,
            node_type: 0.0,
            maturity: 0.0,
            success_rate: 0.0,
            security_level: 0.0,
            tag_count: 0.0,
            avg_neighbor_success_rate: 0.0,
        };

        let dense = features.to_dense();
        assert_eq!(&dense[..4], &[0.5, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn test_neighborhood_aggregation() {
        let (graph, store) = test_fixture();

        let mut a = SkillGraphNode::new("iri://skills/a", "A", "Node A");
        a.add_related("iri://skills/b", "related");
        store.register_skill(a).unwrap();
        let mut b = SkillGraphNode::new("iri://skills/b", "B", "Node B");
        b.add_related("iri://skills/a", "related");
        store.register_skill(b).unwrap();

        let extractor = FeatureExtractor::new(graph);
        let features = extractor.extract_all();

        let mut adj = HashMap::new();
        adj.insert(
            "iri://skills/a".to_string(),
            vec!["iri://skills/b".to_string()],
        );
        adj.insert(
            "iri://skills/b".to_string(),
            vec!["iri://skills/a".to_string()],
        );

        let result = NeighborhoodAggregator::aggregate(&features, &adj, 1, 0.5);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("iri://skills/a"));
        assert!(result.contains_key("iri://skills/b"));
    }
}
