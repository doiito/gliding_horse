use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::causal::types::CausalInference;
use crate::graph_backend::{EdgeDescriptor, GraphBackend};
use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::root_cause::types::TraceChain;

// ════════════════════════════════════════════════════════════════════════
// Fused Root Cause — three-dimensional analysis container
// ════════════════════════════════════════════════════════════════════════

/// Three-dimensional fused root cause analysis result.
///
/// Merges:
/// - **Execution** (L0): 5-why backward trace from RootCauseEngine
/// - **Structural** (L1): Dependency-graph propagation via GraphBackend
/// - **Semantic** (L2): RDF semantic context via KnowledgeGraphStore SPARQL
#[derive(Debug, Clone)]
pub struct FusedRootCause {
    /// 5-why trace chain from RootCauseEngine
    pub trace_chain: TraceChain,
    /// Graph propagation Bayesian inferences (structural dimension)
    pub causal_inferences: Vec<CausalInference>,
    /// RDF semantic neighbor context (semantic dimension)
    pub semantic_context: RdfSemanticContext,
    /// Weighted fusion result across all three dimensions
    pub fused_root: FusedRootCauseResult,
}

/// Weighted fusion output identifying the most probable root cause.
#[derive(Debug, Clone)]
pub struct FusedRootCauseResult {
    pub primary_iri: String,
    pub confidence: f64,
    pub contributing_factors: Vec<ContributingFactor>,
    pub recommended_actions: Vec<String>,
}

/// A single contributing factor from one of the three dimensions.
#[derive(Debug, Clone)]
pub struct ContributingFactor {
    pub dimension: CausalDimension,
    pub iri: String,
    pub weight: f64,
    pub evidence: String,
}

/// Which analysis dimension a contributing factor originates from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalDimension {
    Execution,
    Structural,
    Semantic,
}

impl CausalDimension {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Execution => "execution (5-why trace)",
            Self::Structural => "structural (dependency graph)",
            Self::Semantic => "semantic (RDF context)",
        }
    }
}

/// RDF semantic context gathered via SPARQL traversal.
#[derive(Debug, Clone, Default)]
pub struct RdfSemanticContext {
    /// Direct and transitive neighbors found by BFS
    pub neighbors: Vec<String>,
    /// Paths from error IRI to semantically related entities
    pub paths: Vec<Vec<String>>,
}

// ════════════════════════════════════════════════════════════════════════
// FusedRootCauseEngine
// ════════════════════════════════════════════════════════════════════════

/// Engine that enriches a 5-why trace with structural (graph) and
/// semantic (RDF) analysis, then fuses all three dimensions into a
/// single weighted diagnosis.
#[derive(Clone)]
pub struct FusedRootCauseEngine {
    graph_backend: Option<Arc<dyn GraphBackend>>,
    knowledge_store: Option<Arc<KnowledgeGraphStore>>,
}

impl FusedRootCauseEngine {
    pub fn new(
        graph_backend: Option<Arc<dyn GraphBackend>>,
        knowledge_store: Option<Arc<KnowledgeGraphStore>>,
    ) -> Self {
        Self {
            graph_backend,
            knowledge_store,
        }
    }

    /// Run the full three-dimensional fusion pipeline.
    ///
    /// 1. **Structural** — BFS on the dependency graph from error IRI
    /// 2. **Semantic** — SPARQL neighbor traversal on the RDF store
    /// 3. **Fusion** — weight and combine all dimensions
    pub fn fuse(&self, trace_chain: &TraceChain, error_iri: &str) -> FusedRootCause {
        // ── Structural dimension (dependency graph) ──
        let mut structural_factors = Vec::new();
        if let Some(ref gb) = self.graph_backend {
            let edges = gb.all_edges();
            let reachable = bfs_from(error_iri, &edges, 3);
            if !reachable.is_empty() {
                for iri in reachable.iter().take(3) {
                    structural_factors.push(ContributingFactor {
                        dimension: CausalDimension::Structural,
                        iri: iri.clone(),
                        weight: 0.35,
                        evidence: format!("Reachable via dependency graph BFS from {}", error_iri),
                    });
                }
            }

            let out_deg = edges.iter().filter(|e| e.source == error_iri).count();
            if out_deg > 0 {
                structural_factors.push(ContributingFactor {
                    dimension: CausalDimension::Structural,
                    iri: error_iri.to_string(),
                    weight: 0.35,
                    evidence: format!("Node has {} outgoing edges", out_deg),
                });
            }
        }

        // ── Semantic dimension (RDF context) ──
        let mut semantic_factors = Vec::new();
        let semantic_context = if let Some(ref ks) = self.knowledge_store {
            let ctx = self.collect_semantic_context(ks, error_iri);
            for neighbor in ctx.neighbors.iter().take(3) {
                semantic_factors.push(ContributingFactor {
                    dimension: CausalDimension::Semantic,
                    iri: neighbor.clone(),
                    weight: 0.25,
                    evidence: format!("RDF semantic neighbor of {}", error_iri),
                });
            }
            ctx
        } else {
            RdfSemanticContext::default()
        };

        // ── Fusion — combine all dimensions ──
        let mut all_factors: Vec<ContributingFactor> = structural_factors;
        all_factors.extend(semantic_factors);

        // Determine primary IRI from the highest-weighted factor, or fall back to
        // the trace chain's root cause level.
        let primary_iri = all_factors
            .first()
            .map(|f| f.iri.clone())
            .or_else(|| {
                trace_chain
                    .root_level()
                    .map(|l| format!("root_cause:{}", l.description.clone()))
            })
            .unwrap_or_else(|| error_iri.to_string());

        // Fuse confidence: execution 0.40 + structural 0.35 + semantic 0.25
        let execution_conf = trace_chain
            .root_level()
            .map(|l| l.evidence.confidence)
            .unwrap_or(0.5);
        let structural_conf = if self.graph_backend.is_some() {
            all_factors
                .iter()
                .filter(|f| f.dimension == CausalDimension::Structural)
                .map(|f| f.weight)
                .sum::<f64>()
                .min(0.85)
        } else {
            0.0
        };
        let semantic_conf =
            if self.knowledge_store.is_some() && !semantic_context.neighbors.is_empty() {
                0.25
            } else {
                0.0
            };
        let fused_confidence =
            0.40 * execution_conf + 0.35 * structural_conf + 0.25 * semantic_conf;
        let fused_confidence = fused_confidence.min(1.0);

        // Recommended actions from each dimension
        let mut recommended_actions = Vec::new();
        if let Some(ref gb) = self.graph_backend {
            let out_deg = gb
                .all_edges()
                .iter()
                .filter(|e| e.source == error_iri)
                .count();
            if out_deg > 3 {
                recommended_actions.push(
                    "High out-degree node: consider adding defensive checks at all dependents"
                        .into(),
                );
            }
        }
        if !semantic_context.neighbors.is_empty() {
            recommended_actions.push(
                "Semantic neighbors found: verify related entities for cascading effects".into(),
            );
        }
        if execution_conf < 0.5 {
            recommended_actions.push(
                "Low trace confidence: increase observability at the identified call sites".into(),
            );
        }
        if recommended_actions.is_empty() {
            recommended_actions
                .push("No additional actions recommended beyond standard remediation".into());
        }

        FusedRootCause {
            trace_chain: trace_chain.clone(),
            causal_inferences: Vec::new(),
            semantic_context,
            fused_root: FusedRootCauseResult {
                primary_iri,
                confidence: fused_confidence,
                contributing_factors: all_factors,
                recommended_actions,
            },
        }
    }

    /// Collect RDF semantic context via SPARQL neighbor traversal.
    fn collect_semantic_context(&self, ks: &KnowledgeGraphStore, iri: &str) -> RdfSemanticContext {
        let mut ctx = RdfSemanticContext::default();

        // Outgoing edges: SELECT ?o WHERE { <iri> ?p ?o }
        let sparql = format!("SELECT ?o WHERE {{ <{}> ?p ?o }} LIMIT 20", iri);
        if let Ok(results) = ks.query_sparql(&sparql, None) {
            for row in &results {
                if let Some(val) = row.get("?o").and_then(|v| v.as_str()) {
                    if val.starts_with("iri:")
                        || val.starts_with("http")
                        || val.starts_with("system:")
                    {
                        ctx.neighbors.push(val.to_string());
                    }
                }
            }
        }

        // Incoming edges: SELECT ?s WHERE { ?s ?p <iri> }
        let sparql_in = format!("SELECT ?s WHERE {{ ?s ?p <{}> }} LIMIT 10", iri);
        if let Ok(results) = ks.query_sparql(&sparql_in, None) {
            for row in &results {
                if let Some(val) = row.get("?s").and_then(|v| v.as_str()) {
                    if val.starts_with("iri:")
                        || val.starts_with("http")
                        || val.starts_with("system:")
                    {
                        ctx.neighbors.push(val.to_string());
                    }
                }
            }
        }

        ctx
    }

    // ── Accessors ──

    pub fn has_graph_backend(&self) -> bool {
        self.graph_backend.is_some()
    }

    pub fn has_knowledge_store(&self) -> bool {
        self.knowledge_store.is_some()
    }
}

// ════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════

/// Simple BFS traversal on an edge list. Returns all nodes reachable
/// from the seed within `depth` hops, excluding the seed itself.
fn bfs_from(seed: &str, edges: &[EdgeDescriptor], depth: usize) -> Vec<String> {
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for e in edges {
        adj.entry(e.source.clone())
            .or_default()
            .push(e.target.clone());
    }
    if !adj.contains_key(seed) {
        return Vec::new();
    }
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    let mut result = Vec::new();

    visited.insert(seed.to_string());
    queue.push_back(seed.to_string());

    for _ in 0..depth {
        let level_len = queue.len();
        for _ in 0..level_len {
            let current = queue.pop_front().unwrap();
            if let Some(neighbors) = adj.get(&current) {
                for n in neighbors {
                    if visited.insert(n.clone()) {
                        queue.push_back(n.clone());
                        result.push(n.clone());
                    }
                }
            }
        }
    }
    result
}

const FUSE_EXECUTION_WEIGHT: f64 = 0.40;
const FUSE_STRUCTURAL_WEIGHT: f64 = 0.35;
const FUSE_SEMANTIC_WEIGHT: f64 = 0.25;

/// Compute fused confidence from the three dimensions.
pub fn fuse_confidence(
    trace_confidence: f64,
    causal_confidence: f64,
    semantic_relevance: f64,
) -> f64 {
    (FUSE_EXECUTION_WEIGHT * trace_confidence
        + FUSE_STRUCTURAL_WEIGHT * causal_confidence
        + FUSE_SEMANTIC_WEIGHT * semantic_relevance)
        .min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_backend::PetgraphBackend;
    use crate::root_cause::types::*;
    use crate::skill_graph::graph_store::SkillGraphStore;
    use crate::skill_graph::types::SkillGraphNode;
    use std::sync::Arc;

    fn dummy_trace_chain() -> TraceChain {
        let mut chain = TraceChain::new("trace_test_001", "test_agent");
        chain.add_level(TraceLevel {
            level: 1,
            label: "symptom".into(),
            description: "Failed to connect to database".into(),
            source_location: "db.rs:42".into(),
            is_root_cause: false,
            evidence: Evidence::new("error log", serde_json::json!("connection refused"), 0.9),
        });
        chain.add_level(TraceLevel {
            level: 5,
            label: "root_cause".into(),
            description: "Database server not running".into(),
            source_location: "db.rs:42".into(),
            is_root_cause: true,
            evidence: Evidence::new("systemctl status", serde_json::json!("inactive"), 0.85),
        });
        chain
    }

    #[test]
    fn test_fuse_without_backends() {
        let engine = FusedRootCauseEngine::new(None, None);
        let chain = dummy_trace_chain();
        let result = engine.fuse(&chain, "iri://skills/db");

        assert_eq!(result.trace_chain.trace_id, "trace_test_001");
        assert!(result.causal_inferences.is_empty());
        assert!(result.semantic_context.neighbors.is_empty());
        assert!(result.fused_root.confidence > 0.0);
        assert!(result.fused_root.confidence <= 1.0);
    }

    #[test]
    fn test_fuse_confidence_formula() {
        let confidence = fuse_confidence(0.9, 0.7, 0.5);
        let expected = 0.40 * 0.9 + 0.35 * 0.7 + 0.25 * 0.5;
        assert!((confidence - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fuse_confidence_caps_at_one() {
        let confidence = fuse_confidence(1.0, 1.0, 1.0);
        assert!((confidence - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fuse_with_petgraph_backend() {
        let store = Arc::new(SkillGraphStore::new());
        let mut a = SkillGraphNode::new("iri://skills/a", "Service A", "");
        a.add_prerequisite("iri://skills/b", "B provides DB");
        a.add_prerequisite("iri://skills/c", "C provides cache");
        store.register_skill(a).unwrap();
        store
            .register_skill(SkillGraphNode::new("iri://skills/b", "Service B", ""))
            .unwrap();
        store
            .register_skill(SkillGraphNode::new("iri://skills/c", "Service C", ""))
            .unwrap();

        let backend = PetgraphBackend::new(store.clone());
        let engine = FusedRootCauseEngine::new(Some(Arc::new(backend)), None);

        let chain = dummy_trace_chain();
        let result = engine.fuse(&chain, "iri://skills/a");

        assert!(
            !result.fused_root.contributing_factors.is_empty(),
            "Expected contributing factors from PetgraphBackend, got {}",
            result.fused_root.contributing_factors.len()
        );
        assert!(
            result.fused_root.primary_iri == "iri://skills/b"
                || result.fused_root.primary_iri == "iri://skills/c",
            "Expected one of the prerequisite IRIs, got '{}'",
            result.fused_root.primary_iri
        );
        assert!(result.fused_root.confidence > 0.0);
        assert!(result.fused_root.confidence <= 1.0);
    }

    #[test]
    fn test_bfs_from_empty_edges() {
        let result = bfs_from("iri://skills/a", &[], 3);
        assert!(
            result.is_empty(),
            "BFS from empty edges should return empty"
        );
    }

    #[test]
    fn test_bfs_from_nonexistent_seed() {
        let edges = vec![EdgeDescriptor {
            source: "iri://skills/a".into(),
            target: "iri://skills/b".into(),
            edge_type: "related".into(),
        }];
        let result = bfs_from("iri://skills/nonexistent", &edges, 3);
        assert!(
            result.is_empty(),
            "BFS from nonexistent seed should return empty"
        );
    }

    #[test]
    fn test_bfs_from_three_hops() {
        let edges = vec![
            EdgeDescriptor {
                source: "iri://skills/a".into(),
                target: "iri://skills/b".into(),
                edge_type: "related".into(),
            },
            EdgeDescriptor {
                source: "iri://skills/b".into(),
                target: "iri://skills/c".into(),
                edge_type: "related".into(),
            },
            EdgeDescriptor {
                source: "iri://skills/c".into(),
                target: "iri://skills/d".into(),
                edge_type: "related".into(),
            },
            EdgeDescriptor {
                source: "iri://skills/d".into(),
                target: "iri://skills/e".into(),
                edge_type: "related".into(),
            },
        ];

        // Depth 1 should reach b only
        let d1 = bfs_from("iri://skills/a", &edges, 1);
        assert_eq!(d1.len(), 1);
        assert_eq!(d1[0], "iri://skills/b");

        // Depth 2 should reach b, c
        let d2 = bfs_from("iri://skills/a", &edges, 2);
        assert_eq!(d2.len(), 2);
        assert!(d2.contains(&"iri://skills/b".into()));
        assert!(d2.contains(&"iri://skills/c".into()));

        // Depth 3 should reach b, c, d
        let d3 = bfs_from("iri://skills/a", &edges, 3);
        assert_eq!(d3.len(), 3);
        assert!(d3.contains(&"iri://skills/d".into()));

        // Depth 10 should cap at available nodes (b,c,d,e)
        let d10 = bfs_from("iri://skills/a", &edges, 10);
        assert_eq!(d10.len(), 4);
    }

    #[test]
    fn test_fuse_with_petgraph_backend_chain() {
        let store = Arc::new(SkillGraphStore::new());
        let mut a = SkillGraphNode::new("iri://skills/a", "A", "");
        a.add_prerequisite("iri://skills/b", "chain");
        let mut b = SkillGraphNode::new("iri://skills/b", "B", "");
        b.add_prerequisite("iri://skills/c", "chain");
        let mut c = SkillGraphNode::new("iri://skills/c", "C", "");
        c.add_prerequisite("iri://skills/d", "chain");
        store.register_skill(a).unwrap();
        store.register_skill(b).unwrap();
        store.register_skill(c).unwrap();
        store
            .register_skill(SkillGraphNode::new("iri://skills/d", "D", ""))
            .unwrap();

        let backend = PetgraphBackend::new(store);
        let engine = FusedRootCauseEngine::new(Some(Arc::new(backend)), None);
        let chain = dummy_trace_chain();
        let result = engine.fuse(&chain, "iri://skills/a");

        assert!(
            result.fused_root.contributing_factors.len() >= 3,
            "Expected at least 3 contributing factors (a→b→c→d), got {}",
            result.fused_root.contributing_factors.len()
        );
    }
}
