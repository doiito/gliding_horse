//! Cross-domain graph abstraction layer.
//!
//! Three traits that decouple causal analysis, snapshot versioning, and
//! topological feature extraction from any specific graph storage backend:
//!
//! - [`GraphBackend`] — causal traversal (used by [`CausalEngine`])
//! - [`SnapshotBackend`] — versioning via snapshot/restore (used by [`TimelineStore`])
//! - [`FeatureGraph`] — topological feature extraction (used by [`FeatureExtractor`])
//!
//! Each trait has at minimum a `PetgraphBackend` / `SkillGraph*Backend` wrapping
//! the in-memory [`SkillGraphStore`], and a `SparqlBackend` / `Oxigraph*Backend`
//! that queries the shared [`KnowledgeGraphStore`] (Oxigraph RDF store) over SPARQL.
//!
//! [`CausalEngine`]: crate::causal::engine::CausalEngine
//! [`TimelineStore`]: crate::snapshots::timeline::TimelineStore
//! [`FeatureExtractor`]: crate::graph_features::features::FeatureExtractor
//! [`SkillGraphStore`]: crate::skill_graph::graph_store::SkillGraphStore
//! [`KnowledgeGraphStore`]: crate::knowledge_graph::store::KnowledgeGraphStore

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::types::{Hyperedge, KnowledgeFragment, MOCNode, SkillGraphNode};
use crate::CoreError;

// ════════════════════════════════════════════════════════════════════════
// GraphBackend — causal traversal
// ════════════════════════════════════════════════════════════════════════

/// A single directed edge with a semantic type label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDescriptor {
    pub source: String,
    pub target: String,
    pub edge_type: String,
}

/// Abstract graph backend for causal traversal and feature extraction.
///
/// The [`CausalEngine`](crate::causal::engine::CausalEngine) builds adjacency
/// structures entirely from the three methods below — it never touches
/// petgraph or SPARQL directly.
pub trait GraphBackend: Send + Sync {
    /// All node IRIs in the graph.
    fn all_nodes(&self) -> Vec<String>;

    /// All directed edges as (source, target, edge_type) triples.
    fn all_edges(&self) -> Vec<EdgeDescriptor>;

    /// Check whether a node exists.
    fn has_node(&self, iri: &str) -> bool;

    /// Total node count.
    fn node_count(&self) -> usize;
}

// ════════════════════════════════════════════════════════════════════════
// SnapshotBackend — snapshot versioning
// ════════════════════════════════════════════════════════════════════════

/// A serializable node that a [`SnapshotBackend`] can read/write.
///
/// Generic enough to represent a skill-graph node, an RDF entity, or any
/// named-graph fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotNode {
    pub iri: String,
    pub data: serde_json::Value,
    pub node_type: String,
}

impl SnapshotNode {
    pub fn new(iri: &str, data: serde_json::Value, node_type: &str) -> Self {
        Self {
            iri: iri.to_string(),
            data,
            node_type: node_type.to_string(),
        }
    }
}

/// Abstract backend for snapshot/restore operations.
///
/// Used by [`TimelineStore`](crate::snapshots::timeline::TimelineStore) to
/// decouple versioning from any specific store.
pub trait SnapshotBackend: Send + Sync {
    /// Read the current state as a list of serializable nodes.
    fn snapshot(&self) -> Vec<SnapshotNode>;

    /// Restore state from a previously-captured snapshot.
    fn apply_snapshot(&self, nodes: &[SnapshotNode]) -> Result<(), CoreError>;

    /// Clear all state in the backend (default no-op).
    fn clear(&self) -> Result<(), CoreError> {
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════
// FeatureGraph — GNN-style topological features
// ════════════════════════════════════════════════════════════════════════

/// Direction for neighbor/degree queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Incoming,
    Outgoing,
    Both,
}

/// Abstract graph for topological feature extraction.
///
/// Used by [`FeatureExtractor`](crate::graph_features::features::FeatureExtractor)
/// so it can compute degree, PageRank, betweenness, and community metrics
/// over any graph — not just the skill-graph petgraph.
pub trait FeatureGraph: Send + Sync {
    /// Neighbor IRIs of a node in the given direction.
    fn neighbors(&self, iri: &str, direction: Direction) -> Vec<String>;

    /// Degree of a node in the given direction.
    fn degree(&self, iri: &str, direction: Direction) -> usize;

    /// All node IRIs.
    fn all_nodes(&self) -> Vec<String>;

    /// Total node count.
    fn node_count(&self) -> usize;

    /// Approximate PageRank scores (or empty if not implemented).
    fn page_rank(&self, _damping: f32) -> Vec<(String, f64)> {
        Vec::new()
    }

    /// Approximate betweenness centrality scores (or empty if not implemented).
    fn betweenness_centrality(&self) -> Vec<(String, f64)> {
        Vec::new()
    }

    /// Community detection result (or empty if not implemented).
    fn detect_communities(&self) -> Vec<Vec<String>> {
        Vec::new()
    }

    /// Get a node's serializable data (e.g. tags, type, success rate).
    fn node_data(&self, iri: &str) -> Option<serde_json::Value>;
}

// ════════════════════════════════════════════════════════════════════════
// PetgraphBackend — wraps SkillGraphStore
// ════════════════════════════════════════════════════════════════════════

pub struct PetgraphBackend {
    store: Arc<SkillGraphStore>,
}

impl PetgraphBackend {
    pub fn new(store: Arc<SkillGraphStore>) -> Self {
        Self { store }
    }
}

impl GraphBackend for PetgraphBackend {
    fn all_nodes(&self) -> Vec<String> {
        self.store
            .list_all_skills()
            .into_iter()
            .map(|s| s.skill_iri)
            .collect()
    }

    fn all_edges(&self) -> Vec<EdgeDescriptor> {
        let mut edges = Vec::new();
        for skill in self.store.list_all_skills() {
            for link in &skill.links {
                edges.push(EdgeDescriptor {
                    source: skill.skill_iri.clone(),
                    target: link.target_iri.clone(),
                    edge_type: format!("{:?}", link.link_type),
                });
            }
        }
        edges
    }

    fn has_node(&self, iri: &str) -> bool {
        self.store.get_skill(iri).is_some()
    }

    fn node_count(&self) -> usize {
        self.store.list_all_skills().len()
    }
}

// ════════════════════════════════════════════════════════════════════════
// SparqlBackend — wraps KnowledgeGraphStore via SPARQL
// ════════════════════════════════════════════════════════════════════════

pub struct SparqlBackend {
    store: Arc<KnowledgeGraphStore>,
    /// Optional named graph filter (None = all graphs).
    named_graph: Option<String>,
    /// Edge predicate filter — only edges whose predicate matches are returned.
    edge_predicates: Option<Vec<String>>,
}

impl SparqlBackend {
    pub fn new(store: Arc<KnowledgeGraphStore>) -> Self {
        Self {
            store,
            named_graph: None,
            edge_predicates: None,
        }
    }

    pub fn with_named_graph(mut self, graph: &str) -> Self {
        self.named_graph = Some(graph.to_string());
        self
    }

    pub fn with_edge_predicates(mut self, predicates: Vec<String>) -> Self {
        self.edge_predicates = Some(predicates);
        self
    }
}

impl GraphBackend for SparqlBackend {
    fn all_nodes(&self) -> Vec<String> {
        let graph_clause = self
            .named_graph
            .as_ref()
            .map(|g| format!("GRAPH <{}>", g))
            .unwrap_or_else(|| "GRAPH ?g".to_string());

        let sparql = format!(
            "SELECT DISTINCT ?node WHERE {{ {} {{ {{ ?node ?p ?o }} UNION {{ ?s ?p ?node }} }} }}",
            graph_clause
        );
        match self.store.query_sparql(&sparql, None) {
            Ok(rows) => {
                let mut nodes: Vec<String> = rows
                    .iter()
                    .filter_map(|row| row.get("?node").and_then(|v| v.as_str()))
                    .map(|s| s.to_string())
                    .filter(|s| {
                        !s.starts_with("http://www.w3.org/")
                            && !s.starts_with("https://agent-os.org/ontology/")
                    })
                    .collect();
                nodes.sort();
                nodes.dedup();
                nodes
            }
            Err(_) => Vec::new(),
        }
    }

    fn all_edges(&self) -> Vec<EdgeDescriptor> {
        let graph_clause = self
            .named_graph
            .as_ref()
            .map(|g| format!("GRAPH <{}>", g))
            .unwrap_or_else(|| "GRAPH ?g".to_string());

        let filter_clause = self
            .edge_predicates
            .as_ref()
            .map(|preds| {
                let joined = preds
                    .iter()
                    .map(|p| format!("?p = <{}>", p))
                    .collect::<Vec<_>>()
                    .join(" || ");
                format!("FILTER({})", joined)
            })
            .unwrap_or_default();

        let sparql = format!(
            "SELECT ?s ?p ?o WHERE {{ {} {{ ?s ?p ?o }} {} }}",
            graph_clause, filter_clause
        );

        match self.store.query_sparql(&sparql, None) {
            Ok(rows) => rows
                .iter()
                .filter_map(|row| {
                    let s = row.get("?s")?.as_str()?;
                    let p = row.get("?p")?.as_str()?;
                    let o = row.get("?o")?.as_str()?;
                    // Only IRI objects (skip literal objects)
                    if o.starts_with("iri:")
                        || o.starts_with("http://")
                        || o.starts_with("https://")
                    {
                        Some(EdgeDescriptor {
                            source: s.to_string(),
                            target: o.to_string(),
                            edge_type: p.to_string(),
                        })
                    } else {
                        None
                    }
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    fn has_node(&self, iri: &str) -> bool {
        let graph_clause = self
            .named_graph
            .as_ref()
            .map(|g| format!("GRAPH <{}>", g))
            .unwrap_or_else(|| "GRAPH ?g".to_string());

        let sparql = format!(
            "ASK {{ {} {{ {{ <{}> ?p ?o }} UNION {{ ?s ?p <{}> }} }} }}",
            graph_clause,
            iri.replace('>', "\\u003E"),
            iri.replace('>', "\\u003E")
        );
        match self.store.query_sparql(&sparql, None) {
            Ok(rows) => rows
                .first()
                .and_then(|r| r.get("result"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Err(_) => false,
        }
    }

    fn node_count(&self) -> usize {
        self.all_nodes().len()
    }
}

// ════════════════════════════════════════════════════════════════════════
// SkillGraphSnapshotBackend — wraps SkillGraphStore for TimelineStore
// ════════════════════════════════════════════════════════════════════════

pub struct SkillGraphSnapshotBackend {
    store: Arc<SkillGraphStore>,
}

impl SkillGraphSnapshotBackend {
    pub fn new(store: Arc<SkillGraphStore>) -> Self {
        Self { store }
    }

    /// Access the underlying store (for test and advanced usage).
    pub fn store(&self) -> &Arc<SkillGraphStore> {
        &self.store
    }
}

impl SnapshotBackend for SkillGraphSnapshotBackend {
    fn snapshot(&self) -> Vec<SnapshotNode> {
        let mut nodes = Vec::new();

        // Export all skill graph nodes as JSON
        for skill in self.store.list_all_skills() {
            if let Ok(data) = serde_json::to_value(&skill) {
                nodes.push(SnapshotNode::new(&skill.skill_iri, data, "SkillGraphNode"));
            }
        }

        // Export hyperedges
        for he in self.store.list_hyperedges() {
            if let Ok(data) = serde_json::to_value(&he) {
                nodes.push(SnapshotNode::new(&he.hyperedge_id, data, "Hyperedge"));
            }
        }

        // Export knowledge fragments
        for frag in self.store.list_fragments() {
            if let Ok(data) = serde_json::to_value(&frag) {
                nodes.push(SnapshotNode::new(
                    &frag.fragment_iri,
                    data,
                    "KnowledgeFragment",
                ));
            }
        }

        // Export MOC nodes
        for moc in self.store.list_mocs() {
            if let Ok(data) = serde_json::to_value(&moc) {
                nodes.push(SnapshotNode::new(&moc.moc_iri, data, "MOCNode"));
            }
        }

        nodes
    }

    fn apply_snapshot(&self, nodes: &[SnapshotNode]) -> Result<(), CoreError> {
        // Validate the complete snapshot before changing live state.  A
        // malformed node must not turn a rollback into a partial restore.
        enum RestoredNode {
            Skill(SkillGraphNode),
            Hyperedge(Hyperedge),
            Fragment(KnowledgeFragment),
            Moc(MOCNode),
        }
        let restored = nodes
            .iter()
            .map(|node| {
                let decode = |kind: &str| CoreError::StorageError {
                    message: format!(
                        "Invalid {} snapshot node '{}': invalid serialized data",
                        kind, node.iri
                    ),
                };
                match node.node_type.as_str() {
                    "SkillGraphNode" => serde_json::from_value(node.data.clone())
                        .map(RestoredNode::Skill)
                        .map_err(|_| decode("SkillGraphNode")),
                    "Hyperedge" => serde_json::from_value(node.data.clone())
                        .map(RestoredNode::Hyperedge)
                        .map_err(|_| decode("Hyperedge")),
                    "KnowledgeFragment" => serde_json::from_value(node.data.clone())
                        .map(RestoredNode::Fragment)
                        .map_err(|_| decode("KnowledgeFragment")),
                    "MOCNode" => serde_json::from_value(node.data.clone())
                        .map(RestoredNode::Moc)
                        .map_err(|_| decode("MOCNode")),
                    other => Err(CoreError::ValidationFailed {
                        message: format!(
                            "Unsupported snapshot node type '{}' for '{}'",
                            other, node.iri
                        ),
                    }),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        for node in restored {
            match node {
                RestoredNode::Skill(skill) => self.store.register_skill(skill)?,
                RestoredNode::Hyperedge(hyperedge) => self.store.register_hyperedge(hyperedge)?,
                RestoredNode::Fragment(fragment) => self.store.register_fragment(fragment)?,
                RestoredNode::Moc(moc) => self.store.register_moc(moc)?,
            }
        }
        Ok(())
    }

    fn clear(&self) -> Result<(), CoreError> {
        let skill_iris: Vec<String> = self
            .store
            .list_all_skills()
            .into_iter()
            .map(|s| s.skill_iri)
            .collect();
        for iri in &skill_iris {
            self.store.remove_skill(iri)?;
        }
        let he_ids: Vec<String> = self
            .store
            .list_hyperedges()
            .into_iter()
            .map(|h| h.hyperedge_id)
            .collect();
        for id in &he_ids {
            self.store.remove_hyperedge(id)?;
        }
        let moc_iris: Vec<String> = self
            .store
            .list_mocs()
            .into_iter()
            .map(|moc| moc.moc_iri)
            .collect();
        for iri in &moc_iris {
            self.store.remove_moc(iri)?;
        }
        let fragment_iris: Vec<String> = self
            .store
            .list_fragments()
            .into_iter()
            .map(|fragment| fragment.fragment_iri)
            .collect();
        for iri in &fragment_iris {
            self.store.remove_fragment(iri)?;
        }
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════
// OxigraphSnapshotBackend — wraps KnowledgeGraphStore
// ════════════════════════════════════════════════════════════════════════

pub struct OxigraphSnapshotBackend {
    store: Arc<KnowledgeGraphStore>,
    named_graph: Option<String>,
}

impl OxigraphSnapshotBackend {
    pub fn new(store: Arc<KnowledgeGraphStore>) -> Self {
        Self {
            store,
            named_graph: None,
        }
    }

    pub fn with_named_graph(mut self, graph: &str) -> Self {
        self.named_graph = Some(graph.to_string());
        self
    }
}

impl SnapshotBackend for OxigraphSnapshotBackend {
    fn snapshot(&self) -> Vec<SnapshotNode> {
        let graph_clause = self
            .named_graph
            .as_ref()
            .map(|g| format!("GRAPH <{}>", g))
            .unwrap_or_else(|| "GRAPH ?g".to_string());

        // Export each distinct subject as a snapshot node with its properties
        let sparql = format!(
            "SELECT DISTINCT ?s WHERE {{ {} {{ ?s ?p ?o }} }}",
            graph_clause
        );

        let iris = match self.store.query_sparql(&sparql, None) {
            Ok(rows) => rows
                .iter()
                .filter_map(|row| row.get("?s").and_then(|v| v.as_str()))
                .map(|s| s.to_string())
                .filter(|s| !s.starts_with("http://www.w3.org/"))
                .collect::<Vec<_>>(),
            Err(_) => return Vec::new(),
        };

        let mut nodes = Vec::new();
        for iri in &iris {
            let prop_sparql = format!(
                "SELECT ?p ?o WHERE {{ {} {{ <{}> ?p ?o }} }}",
                graph_clause,
                iri.replace('>', "\\u003E")
            );

            let props: serde_json::Map<String, serde_json::Value> =
                match self.store.query_sparql(&prop_sparql, None) {
                    Ok(rows) => rows
                        .iter()
                        .filter_map(|row| {
                            let p = row.get("?p")?.as_str()?;
                            let o = row.get("?o")?.as_str()?;
                            Some((p.to_string(), serde_json::Value::String(o.to_string())))
                        })
                        .collect(),
                    Err(_) => serde_json::Map::new(),
                };

            nodes.push(SnapshotNode::new(
                iri,
                serde_json::Value::Object(props),
                "RdfEntity",
            ));
        }

        nodes
    }

    fn apply_snapshot(&self, nodes: &[SnapshotNode]) -> Result<(), CoreError> {
        // Determine target graph
        let graph = self.named_graph.as_deref().unwrap_or("graph:world");

        // Clear the existing graph content
        let clear_sparql = format!("DROP SILENT GRAPH <{}>", graph);
        let _ = self.store.query_sparql(&clear_sparql, None);

        // Re-insert all triples from snapshot nodes
        for node in nodes {
            if let Some(props) = node.data.as_object() {
                for (predicate, value) in props {
                    let obj_str = match value.as_str() {
                        Some(v)
                            if v.starts_with("iri:")
                                || v.starts_with("http://")
                                || v.starts_with("https://") =>
                        {
                            format!("<{}>", v)
                        }
                        Some(v) => format!("\"{}\"", v.replace('"', "\\\"")),
                        None => continue,
                    };

                    let insert = format!(
                        "INSERT DATA {{ GRAPH <{}> {{ <{}> <{}> {} }} }}",
                        graph, node.iri, predicate, obj_str
                    );
                    let _ = self.store.query_sparql(&insert, None);
                }
            }
        }

        Ok(())
    }

    fn clear(&self) -> Result<(), CoreError> {
        if let Some(ref g) = self.named_graph {
            let sparql = format!("DROP SILENT GRAPH <{}>", g);
            let _ = self.store.query_sparql(&sparql, None);
        }
        // Without a named graph, no-op to avoid clearing shared store data
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════
// SkillGraphFeatureGraph — wraps SkillGraphStore + SkillGraphAlgorithms
// ════════════════════════════════════════════════════════════════════════

use crate::skill_graph::graph_algorithms::SkillGraphAlgorithms;

pub struct SkillGraphFeatureGraph {
    store: Arc<SkillGraphStore>,
    // Kept in the constructor for API compatibility; metrics below rebuild
    // from the authoritative store so graph mutations cannot leave a stale
    // algorithm snapshot in the feature pipeline.
    _algorithms: Arc<SkillGraphAlgorithms>,
}

impl SkillGraphFeatureGraph {
    pub fn new(store: Arc<SkillGraphStore>, algorithms: Arc<SkillGraphAlgorithms>) -> Self {
        Self {
            store,
            _algorithms: algorithms,
        }
    }
}

impl FeatureGraph for SkillGraphFeatureGraph {
    fn neighbors(&self, iri: &str, direction: Direction) -> Vec<String> {
        let all_skills = self.store.list_all_skills();
        match direction {
            Direction::Outgoing => all_skills
                .iter()
                .find(|s| s.skill_iri == iri)
                .map(|s| s.links.iter().map(|l| l.target_iri.clone()).collect())
                .unwrap_or_default(),
            Direction::Incoming => all_skills
                .iter()
                .filter(|s| s.links.iter().any(|l| l.target_iri == iri))
                .map(|s| s.skill_iri.clone())
                .collect(),
            Direction::Both => {
                let mut result: Vec<String> = all_skills
                    .iter()
                    .find(|s| s.skill_iri == iri)
                    .map(|s| s.links.iter().map(|l| l.target_iri.clone()).collect())
                    .unwrap_or_default();
                result.extend(
                    all_skills
                        .iter()
                        .filter(|s| s.links.iter().any(|l| l.target_iri == iri))
                        .map(|s| s.skill_iri.clone()),
                );
                result.sort();
                result.dedup();
                result
            }
        }
    }

    fn degree(&self, iri: &str, direction: Direction) -> usize {
        match direction {
            Direction::Outgoing => self
                .store
                .get_skill(iri)
                .map(|s| s.links.len())
                .unwrap_or(0),
            Direction::Incoming => self
                .store
                .list_all_skills()
                .iter()
                .filter(|s| s.links.iter().any(|l| l.target_iri == iri))
                .count(),
            Direction::Both => {
                let out = self
                    .store
                    .get_skill(iri)
                    .map(|s| s.links.len())
                    .unwrap_or(0);
                let inc = self
                    .store
                    .list_all_skills()
                    .iter()
                    .filter(|s| s.links.iter().any(|l| l.target_iri == iri))
                    .count();
                out + inc
            }
        }
    }

    fn all_nodes(&self) -> Vec<String> {
        self.store
            .list_all_skills()
            .into_iter()
            .map(|s| s.skill_iri)
            .collect()
    }

    fn node_count(&self) -> usize {
        self.store.list_all_skills().len()
    }

    fn page_rank(&self, damping: f32) -> Vec<(String, f64)> {
        SkillGraphAlgorithms::from_store(&self.store)
            .page_rank(damping)
            .into_iter()
            .map(|s| (s.iri, s.score))
            .collect()
    }

    fn betweenness_centrality(&self) -> Vec<(String, f64)> {
        SkillGraphAlgorithms::from_store(&self.store)
            .betweenness_centrality()
            .into_iter()
            .map(|s| (s.iri, s.score))
            .collect()
    }

    fn detect_communities(&self) -> Vec<Vec<String>> {
        SkillGraphAlgorithms::from_store(&self.store).detect_communities()
    }

    fn node_data(&self, iri: &str) -> Option<serde_json::Value> {
        let skill = self.store.get_skill(iri)?;
        let mut map = serde_json::Map::new();
        map.insert(
            "node_type".to_string(),
            serde_json::Value::String(format!("{:?}", skill.node_type)),
        );
        map.insert(
            "maturity".to_string(),
            serde_json::Value::String(skill.maturity.clone()),
        );
        map.insert(
            "success_rate".to_string(),
            serde_json::json!(skill.graph_meta.success_rate),
        );
        map.insert("tag_count".to_string(), serde_json::json!(skill.tags.len()));
        if let Some(ref sec) = skill.security_info {
            map.insert(
                "security_level".to_string(),
                serde_json::json!(sec.trust_level as u8),
            );
        }
        Some(serde_json::Value::Object(map))
    }
}

// ════════════════════════════════════════════════════════════════════════
// SparqlFeatureGraph — wraps KnowledgeGraphStore
// ════════════════════════════════════════════════════════════════════════

pub struct SparqlFeatureGraph {
    store: Arc<KnowledgeGraphStore>,
    named_graph: Option<String>,
}

impl SparqlFeatureGraph {
    pub fn new(store: Arc<KnowledgeGraphStore>) -> Self {
        Self {
            store,
            named_graph: None,
        }
    }

    pub fn with_named_graph(mut self, graph: &str) -> Self {
        self.named_graph = Some(graph.to_string());
        self
    }

    fn graph_clause(&self) -> String {
        self.named_graph
            .as_ref()
            .map(|g| format!("GRAPH <{}>", g))
            .unwrap_or_else(|| "GRAPH ?g".to_string())
    }

    fn exec(&self, sparql: &str) -> Vec<serde_json::Value> {
        self.store.query_sparql(sparql, None).unwrap_or_default()
    }
}

impl FeatureGraph for SparqlFeatureGraph {
    fn neighbors(&self, iri: &str, direction: Direction) -> Vec<String> {
        let gc = self.graph_clause();
        let safe_iri = iri.replace('>', "\\u003E");
        match direction {
            Direction::Outgoing | Direction::Both => {
                let q = format!(
                    "SELECT ?o WHERE {{ {} {{ <{}> ?p ?o . FILTER(isIRI(?o)) }} }}",
                    gc, safe_iri
                );
                let mut result: Vec<String> = self
                    .exec(&q)
                    .iter()
                    .filter_map(|r| r.get("?o").and_then(|v| v.as_str()))
                    .map(|s| s.to_string())
                    .collect();
                if direction == Direction::Both {
                    let inc = self.neighbors(iri, Direction::Incoming);
                    result.extend(inc);
                    result.sort();
                    result.dedup();
                }
                result
            }
            Direction::Incoming => {
                let q = format!("SELECT ?s WHERE {{ {} {{ ?s ?p <{}> . }} }}", gc, safe_iri);
                self.exec(&q)
                    .iter()
                    .filter_map(|r| r.get("?s").and_then(|v| v.as_str()))
                    .map(|s| s.to_string())
                    .collect()
            }
        }
    }

    fn degree(&self, iri: &str, direction: Direction) -> usize {
        let safe_iri = iri.replace('>', "\\u003E");
        let gc = self.graph_clause();
        match direction {
            Direction::Outgoing => {
                let q = format!(
                    "SELECT (COUNT(?o) AS ?cnt) WHERE {{ {} {{ <{}> ?p ?o }} }}",
                    gc, safe_iri
                );
                self.exec(&q)
                    .first()
                    .and_then(|r| r.get("?cnt").and_then(|v| v.as_str()))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0)
            }
            Direction::Incoming => {
                let q = format!(
                    "SELECT (COUNT(?s) AS ?cnt) WHERE {{ {} {{ ?s ?p <{}> . }} }}",
                    gc, safe_iri
                );
                self.exec(&q)
                    .first()
                    .and_then(|r| r.get("?cnt").and_then(|v| v.as_str()))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0)
            }
            Direction::Both => {
                self.degree(iri, Direction::Outgoing) + self.degree(iri, Direction::Incoming)
            }
        }
    }

    fn all_nodes(&self) -> Vec<String> {
        let gc = self.graph_clause();
        // Exclude ontology/vocabulary IRIs
        self.exec(&format!(
            "SELECT DISTINCT ?s WHERE {{ {} {{ ?s ?p ?o . FILTER(isIRI(?s)) }} }}",
            gc
        ))
        .iter()
        .filter_map(|r| r.get("?s").and_then(|v| v.as_str()))
        .filter(|s| {
            !s.starts_with("http://www.w3.org/") && !s.starts_with("https://agent-os.org/ontology/")
        })
        .map(|s| s.to_string())
        .collect()
    }

    fn node_count(&self) -> usize {
        self.all_nodes().len()
    }

    fn node_data(&self, iri: &str) -> Option<serde_json::Value> {
        let gc = self.graph_clause();
        let safe_iri = iri.replace('>', "\\u003E");
        let q = format!("SELECT ?p ?o WHERE {{ {} {{ <{}> ?p ?o }} }}", gc, safe_iri);
        let rows = self.exec(&q);
        if rows.is_empty() {
            return None;
        }
        let mut map = serde_json::Map::new();
        for row in &rows {
            if let (Some(p), Some(o)) = (
                row.get("?p").and_then(|v| v.as_str()),
                row.get("?o").and_then(|v| v.as_str()),
            ) {
                let key = p.rsplit('/').last().unwrap_or(p);
                map.insert(key.to_string(), serde_json::Value::String(o.to_string()));
            }
        }
        Some(serde_json::Value::Object(map))
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_graph::store::KnowledgeGraphStore;
    use crate::knowledge_graph::types::{RdfQuad, RdfValue};
    use crate::skill_graph::types::SkillGraphNode;

    fn setup_petgraph_backend() -> PetgraphBackend {
        let store = Arc::new(SkillGraphStore::new());
        let mut a = SkillGraphNode::new("iri://skills/a", "A", "Node A");
        a.add_prerequisite("iri://skills/b", "B enables A");
        a.add_related("iri://skills/c", "Related to C");
        store.register_skill(a).unwrap();
        store
            .register_skill(SkillGraphNode::new("iri://skills/b", "B", "Node B"))
            .unwrap();
        store
            .register_skill(SkillGraphNode::new("iri://skills/c", "C", "Node C"))
            .unwrap();
        PetgraphBackend::new(store)
    }

    #[test]
    fn test_petgraph_all_nodes() {
        let backend = setup_petgraph_backend();
        let nodes = backend.all_nodes();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.contains(&"iri://skills/a".to_string()));
    }

    #[test]
    fn test_petgraph_all_edges() {
        let backend = setup_petgraph_backend();
        let edges = backend.all_edges();
        assert_eq!(edges.len(), 2);
        let a_to_b = edges
            .iter()
            .find(|e| e.source == "iri://skills/a" && e.target == "iri://skills/b");
        assert!(a_to_b.is_some());
        assert!(a_to_b.unwrap().edge_type.contains("Prerequisite"));
    }

    #[test]
    fn test_petgraph_has_node() {
        let backend = setup_petgraph_backend();
        assert!(backend.has_node("iri://skills/a"));
        assert!(!backend.has_node("iri://skills/nonexistent"));
    }

    #[test]
    fn test_petgraph_node_count() {
        let backend = setup_petgraph_backend();
        assert_eq!(backend.node_count(), 3);
    }

    #[test]
    fn test_skill_graph_snapshot_backend() {
        let store = Arc::new(SkillGraphStore::new());
        store
            .register_skill(SkillGraphNode::new("iri://skills/x", "X", "Test X"))
            .unwrap();
        let backend = SkillGraphSnapshotBackend::new(store.clone());

        let snap = backend.snapshot();
        assert!(!snap.is_empty());
        assert!(snap.iter().any(|n| n.iri == "iri://skills/x"));
    }

    #[test]
    fn snapshot_backend_rejects_invalid_data_without_mutating_store() {
        let store = Arc::new(SkillGraphStore::new());
        store
            .register_skill(SkillGraphNode::new("iri://skills/kept", "Kept", "existing"))
            .unwrap();
        let backend = SkillGraphSnapshotBackend::new(store.clone());

        let result = backend.apply_snapshot(&[SnapshotNode::new(
            "iri://skills/bad",
            serde_json::json!({"not": "a skill graph node"}),
            "SkillGraphNode",
        )]);

        assert!(result.is_err());
        assert_eq!(store.skill_count(), 1);
        assert!(store.get_skill("iri://skills/kept").is_some());
    }

    #[test]
    fn test_feature_graph_degree() {
        let store = Arc::new(SkillGraphStore::new());
        let algo = Arc::new(SkillGraphAlgorithms::from_store(&store));
        let mut a = SkillGraphNode::new("iri://skills/a", "A", "Node A");
        a.add_prerequisite("iri://skills/b", "B enables A");
        a.add_related("iri://skills/c", "Related to C");
        store.register_skill(a).unwrap();
        store
            .register_skill(SkillGraphNode::new("iri://skills/b", "B", "Node B"))
            .unwrap();
        store
            .register_skill(SkillGraphNode::new("iri://skills/c", "C", "Node C"))
            .unwrap();

        let fg = SkillGraphFeatureGraph::new(store, algo);
        assert_eq!(fg.degree("iri://skills/a", Direction::Outgoing), 2);
        assert_eq!(fg.degree("iri://skills/b", Direction::Incoming), 1);
        assert_eq!(fg.neighbors("iri://skills/a", Direction::Outgoing).len(), 2);
    }

    #[test]
    fn test_feature_graph_metrics_refresh_after_store_mutation() {
        let store = Arc::new(SkillGraphStore::new());
        store
            .register_skill(SkillGraphNode::new("iri://skills/a", "A", "A"))
            .unwrap();
        let stale = Arc::new(SkillGraphAlgorithms::from_store(&store));
        let fg = SkillGraphFeatureGraph::new(store.clone(), stale);
        store
            .register_skill(SkillGraphNode::new("iri://skills/b", "B", "B"))
            .unwrap();
        assert!(fg
            .page_rank(0.85)
            .iter()
            .any(|(iri, _)| iri == "iri://skills/b"));
    }

    // ── SparqlBackend tests ────────────────────────────────────────────

    const SPARQL_TEST_GRAPH: &str = "http://test/graph";

    fn make_quad(s: &str, p: &str, o: RdfValue) -> RdfQuad {
        RdfQuad {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o,
            graph: Some(SPARQL_TEST_GRAPH.to_string()),
        }
    }

    fn setup_sparql_backend() -> SparqlBackend {
        let kg = KnowledgeGraphStore::new().unwrap();
        let quads = vec![
            make_quad(
                "iri://skills/a",
                "iri://predicate/prerequisite",
                RdfValue::Iri("iri://skills/b".to_string()),
            ),
            make_quad(
                "iri://skills/a",
                "iri://predicate/related",
                RdfValue::Iri("iri://skills/c".to_string()),
            ),
            make_quad(
                "iri://skills/b",
                "iri://predicate/related",
                RdfValue::Iri("iri://skills/d".to_string()),
            ),
        ];
        kg.write_quads(&quads, SPARQL_TEST_GRAPH).unwrap();
        SparqlBackend::new(Arc::new(kg)).with_named_graph(SPARQL_TEST_GRAPH)
    }

    #[test]
    fn test_sparql_all_nodes() {
        let backend = setup_sparql_backend();
        let nodes = backend.all_nodes();
        assert_eq!(nodes.len(), 4);
        assert!(nodes.contains(&"iri://skills/a".to_string()));
        assert!(nodes.contains(&"iri://skills/b".to_string()));
    }

    #[test]
    fn test_sparql_all_edges() {
        let backend = setup_sparql_backend();
        let edges = backend.all_edges();
        assert_eq!(edges.len(), 3);
        let a_to_b = edges
            .iter()
            .find(|e| e.source == "iri://skills/a" && e.target == "iri://skills/b");
        assert!(a_to_b.is_some());
    }

    #[test]
    fn test_sparql_has_node() {
        let backend = setup_sparql_backend();
        assert!(backend.has_node("iri://skills/a"));
        assert!(backend.has_node("iri://skills/d"));
        assert!(!backend.has_node("iri://skills/nonexistent"));
    }

    #[test]
    fn test_sparql_node_count() {
        let backend = setup_sparql_backend();
        assert_eq!(backend.node_count(), 4);
    }

    #[test]
    fn test_sparql_with_edge_predicates() {
        let kg = KnowledgeGraphStore::new().unwrap();
        let quads = vec![
            make_quad(
                "iri://skills/x",
                "http://example.org/related",
                RdfValue::Iri("iri://skills/y".to_string()),
            ),
            make_quad(
                "iri://skills/x",
                "http://example.org/ignore-me",
                RdfValue::Iri("iri://skills/z".to_string()),
            ),
        ];
        kg.write_quads(&quads, SPARQL_TEST_GRAPH).unwrap();
        let backend = SparqlBackend::new(Arc::new(kg))
            .with_named_graph(SPARQL_TEST_GRAPH)
            .with_edge_predicates(vec!["http://example.org/related".to_string()]);
        let edges = backend.all_edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, "iri://skills/y");
    }

    // ── SparqlFeatureGraph tests ──────────────────────────────────────

    fn setup_sparql_feature_graph() -> SparqlFeatureGraph {
        let kg = KnowledgeGraphStore::new().unwrap();
        let quads = vec![
            make_quad(
                "iri://skills/a",
                "iri://predicate/related",
                RdfValue::Iri("iri://skills/b".to_string()),
            ),
            make_quad(
                "iri://skills/a",
                "iri://predicate/related",
                RdfValue::Iri("iri://skills/c".to_string()),
            ),
            make_quad(
                "iri://skills/c",
                "iri://predicate/prerequisite",
                RdfValue::Iri("iri://skills/d".to_string()),
            ),
            make_quad(
                "iri://skills/e",
                "iri://predicate/label",
                RdfValue::Literal("leaf node".to_string()),
            ),
        ];
        kg.write_quads(&quads, SPARQL_TEST_GRAPH).unwrap();
        SparqlFeatureGraph::new(Arc::new(kg)).with_named_graph(SPARQL_TEST_GRAPH)
    }

    #[test]
    fn test_sparql_feature_neighbors_outgoing() {
        let fg = setup_sparql_feature_graph();
        let neighbors = fg.neighbors("iri://skills/a", Direction::Outgoing);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&"iri://skills/b".to_string()));
        assert!(neighbors.contains(&"iri://skills/c".to_string()));
    }

    #[test]
    fn test_sparql_feature_neighbors_incoming() {
        let fg = setup_sparql_feature_graph();
        let neighbors = fg.neighbors("iri://skills/b", Direction::Incoming);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&"iri://skills/a".to_string()));
    }

    #[test]
    fn test_sparql_feature_degree_outgoing() {
        let fg = setup_sparql_feature_graph();
        assert_eq!(fg.degree("iri://skills/a", Direction::Outgoing), 2);
        assert_eq!(fg.degree("iri://skills/c", Direction::Outgoing), 1);
        assert_eq!(fg.degree("iri://skills/b", Direction::Outgoing), 0);
    }

    #[test]
    fn test_sparql_feature_degree_incoming() {
        let fg = setup_sparql_feature_graph();
        assert_eq!(fg.degree("iri://skills/b", Direction::Incoming), 1);
        assert_eq!(fg.degree("iri://skills/d", Direction::Incoming), 1);
        assert_eq!(fg.degree("iri://skills/a", Direction::Incoming), 0);
    }

    #[test]
    fn test_sparql_feature_node_data() {
        let fg = setup_sparql_feature_graph();
        let data = fg.node_data("iri://skills/a");
        assert!(data.is_some(), "Expected node_data for iri://skills/a");
    }

    #[test]
    fn test_sparql_feature_node_data_missing() {
        let fg = setup_sparql_feature_graph();
        let data = fg.node_data("iri://skills/nonexistent");
        assert!(data.is_none(), "Expected None for nonexistent node");
    }

    #[test]
    fn test_sparql_feature_neighbors_nonexistent() {
        let fg = setup_sparql_feature_graph();
        let out = fg.neighbors("iri://skills/nonexistent", Direction::Outgoing);
        let inc = fg.neighbors("iri://skills/nonexistent", Direction::Incoming);
        assert!(
            out.is_empty(),
            "Expected empty outgoing for nonexistent IRI"
        );
        assert!(
            inc.is_empty(),
            "Expected empty incoming for nonexistent IRI"
        );
    }

    #[test]
    fn test_sparql_feature_degree_nonexistent() {
        let fg = setup_sparql_feature_graph();
        assert_eq!(
            fg.degree("iri://skills/nonexistent", Direction::Outgoing),
            0
        );
        assert_eq!(
            fg.degree("iri://skills/nonexistent", Direction::Incoming),
            0
        );
        assert_eq!(fg.degree("iri://skills/nonexistent", Direction::Both), 0);
    }

    // ── Empty graph tests ─────────────────────────────────────────────

    fn empty_sparql_backend() -> SparqlBackend {
        let kg = KnowledgeGraphStore::new().unwrap();
        SparqlBackend::new(Arc::new(kg))
    }

    fn empty_sparql_feature_graph() -> SparqlFeatureGraph {
        let kg = KnowledgeGraphStore::new().unwrap();
        SparqlFeatureGraph::new(Arc::new(kg))
    }

    fn empty_petgraph_backend() -> PetgraphBackend {
        PetgraphBackend::new(Arc::new(SkillGraphStore::new()))
    }

    #[test]
    fn test_empty_sparql_all_nodes() {
        let b = empty_sparql_backend();
        assert!(b.all_nodes().is_empty());
    }

    #[test]
    fn test_empty_sparql_all_edges() {
        let b = empty_sparql_backend();
        assert!(b.all_edges().is_empty());
    }

    #[test]
    fn test_empty_sparql_has_node() {
        let b = empty_sparql_backend();
        assert!(!b.has_node("iri://skills/a"));
    }

    #[test]
    fn test_empty_sparql_node_count() {
        let b = empty_sparql_backend();
        assert_eq!(b.node_count(), 0);
    }

    #[test]
    fn test_empty_sparql_feature_neighbors() {
        let fg = empty_sparql_feature_graph();
        assert!(fg
            .neighbors("iri://skills/a", Direction::Outgoing)
            .is_empty());
        assert!(fg
            .neighbors("iri://skills/a", Direction::Incoming)
            .is_empty());
    }

    #[test]
    fn test_empty_sparql_feature_degree() {
        let fg = empty_sparql_feature_graph();
        assert_eq!(fg.degree("iri://skills/a", Direction::Outgoing), 0);
    }

    #[test]
    fn test_empty_sparql_feature_node_count() {
        let fg = empty_sparql_feature_graph();
        assert_eq!(fg.node_count(), 0);
    }

    #[test]
    fn test_empty_petgraph_all_nodes() {
        let b = empty_petgraph_backend();
        assert!(b.all_nodes().is_empty());
    }

    #[test]
    fn test_empty_petgraph_all_edges() {
        let b = empty_petgraph_backend();
        assert!(b.all_edges().is_empty());
    }

    #[test]
    fn test_empty_petgraph_has_node() {
        let b = empty_petgraph_backend();
        assert!(!b.has_node("iri://skills/a"));
    }
}
