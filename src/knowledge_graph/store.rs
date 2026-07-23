use oxigraph::sparql::QueryResults;
use oxigraph::store::Store;
use std::sync::Arc;

use super::rdf_mapper::RdfMapper;
use super::types::RdfQuad;

pub struct KnowledgeGraphStore {
    store: Arc<Store>,
    default_graph: String,
}

impl KnowledgeGraphStore {
    /// Create KG Store using a unified shared Oxigraph Store
    pub fn with_shared_store(store: Arc<Store>) -> Result<Self, String> {
        Ok(Self {
            store,
            default_graph: "graph:world".to_string(),
        })
    }

    pub fn new() -> Result<Self, String> {
        let store = Store::new().map_err(|e| format!("failed to create Oxigraph Store: {}", e))?;
        Ok(Self {
            store: Arc::new(store),
            default_graph: "graph:world".to_string(),
        })
    }

    /// Expose the inner `Arc<Store>` for GH ↔ OO store sharing.
    pub fn store_arc(&self) -> &Arc<Store> {
        &self.store
    }

    /// Create from an OO `SharedGraphStore` (integration point).
    #[cfg(feature = "ontology")]
    pub fn with_shared_graph_store(
        shared: &crate::ontology::SharedGraphStore,
    ) -> Result<Self, String> {
        Ok(Self {
            store: Arc::clone(shared.inner()),
            default_graph: "graph:world".to_string(),
        })
    }

    pub fn with_graph(graph_name: &str) -> Result<Self, String> {
        let store = Store::new().map_err(|e| format!("failed to create Oxigraph Store: {}", e))?;
        Ok(Self {
            store: Arc::new(store),
            default_graph: graph_name.to_string(),
        })
    }

    pub fn write_quads(&self, quads: &[RdfQuad], graph: &str) -> Result<(), String> {
        if quads.is_empty() {
            return Ok(());
        }
        let sparql = RdfMapper::quads_to_sparql_insert(quads, graph);
        self.store
            .update(&sparql)
            .map_err(|e| format!("SPARQL INSERT failed: {}", e))
    }

    pub fn delete_quads_for_source(&self, source_file: &str, graph: &str) -> Result<usize, String> {
        let safe_file = RdfMapper::sanitize_id(source_file);
        let subject_iri = format!("iri://entity/file:{}", safe_file);
        let delete_sparql = format!(
            "DELETE WHERE {{ GRAPH <{}> {{ <{}> ?p ?o . }} }}",
            graph, subject_iri
        );
        self.store
            .update(&delete_sparql)
            .map_err(|e| format!("SPARQL DELETE failed: {}", e))?;

        let related_delete = format!(
            "DELETE WHERE {{ GRAPH <{}> {{ ?s <https://agentos.ontology/code/contains> <{}> . }} }}",
            graph, subject_iri
        );
        let _ = self.store.update(&related_delete);

        Ok(0)
    }

    #[allow(deprecated)]
    pub fn delete_quads_by_subject_prefix(
        &self,
        prefix: &str,
        graph: &str,
    ) -> Result<usize, String> {
        let sparql = format!(
            "SELECT DISTINCT ?s WHERE {{ GRAPH <{}> {{ ?s ?p ?o . FILTER(STRSTARTS(STR(?s), \"{}\")) }} }}",
            graph, Self::escape_sparql_string(prefix)
        );

        let subjects: Vec<String> = match self.store.query(&sparql) {
            Ok(QueryResults::Solutions(solutions)) => solutions
                .filter_map(|sol| sol.ok())
                .filter_map(|sol| {
                    sol.get(0).map(|v| {
                        v.to_string()
                            .trim_start_matches('<')
                            .trim_end_matches('>')
                            .to_string()
                    })
                })
                .collect(),
            _ => return Ok(0),
        };

        let count = subjects.len();
        for subject in &subjects {
            let s = format!("<{}>", subject);
            let del = format!("DELETE WHERE {{ GRAPH <{}> {{ {} ?p ?o . }} }}", graph, s);
            let _ = self.store.update(&del);
            let del_in = format!("DELETE WHERE {{ GRAPH <{}> {{ ?s ?p {} . }} }}", graph, s);
            let _ = self.store.update(&del_in);
        }

        Ok(count)
    }

    /// Remove the source-file node and every declaration currently owned by
    /// it. AST declaration IRIs are `fn:/class:/…` rather than descendants of
    /// the file IRI, so prefix deletion alone leaves stale declarations after
    /// an incremental re-extraction.
    #[allow(deprecated)]
    pub fn delete_code_file_subgraph(&self, file_iri: &str, graph: &str) -> Result<usize, String> {
        let contains = "https://agentos.ontology/code/contains";
        let query = format!(
            "SELECT DISTINCT ?node WHERE {{ GRAPH <{}> {{ <{}> <{}> ?node . }} }}",
            graph, file_iri, contains
        );
        let mut subjects = vec![file_iri.to_string()];
        if let Ok(QueryResults::Solutions(rows)) = self.store.query(&query) {
            subjects.extend(rows.filter_map(Result::ok).filter_map(|row| {
                row.get(0).map(|term| {
                    term.to_string()
                        .trim_start_matches('<')
                        .trim_end_matches('>')
                        .to_string()
                })
            }));
        }
        subjects.sort();
        subjects.dedup();

        for subject in &subjects {
            let escaped = format!("<{}>", subject);
            self.store
                .update(&format!(
                    "DELETE WHERE {{ GRAPH <{}> {{ {} ?p ?o . }} }}",
                    graph, escaped
                ))
                .map_err(|e| format!("SPARQL DELETE failed: {}", e))?;
            self.store
                .update(&format!(
                    "DELETE WHERE {{ GRAPH <{}> {{ ?s ?p {} . }} }}",
                    graph, escaped
                ))
                .map_err(|e| format!("SPARQL DELETE failed: {}", e))?;
        }
        Ok(subjects.len())
    }

    #[allow(deprecated)]
    pub fn query_sparql(
        &self,
        sparql: &str,
        named_graph: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, String> {
        let final_sparql = match named_graph {
            Some(graph) if !sparql.to_uppercase().contains("GRAPH") => {
                let g = format!("<{}>", graph);
                let upper = sparql.to_uppercase();
                if let Some(where_pos) = upper.find("WHERE") {
                    let after_where = &sparql[where_pos + 5..];
                    if let Some(brace_pos) = after_where.find('{') {
                        let prefix = &sparql[..where_pos + 5 + brace_pos + 1];
                        let inner = after_where[brace_pos + 1..].trim_end_matches('}').trim();
                        format!("{} GRAPH {} {{ {} }} }}", prefix, g, inner)
                    } else {
                        sparql.to_string()
                    }
                } else {
                    format!("SELECT * WHERE {{ GRAPH {} {{ {} }} }}", g, sparql)
                }
            }
            _ => sparql.to_string(),
        };

        let results = self
            .store
            .query(&final_sparql)
            .map_err(|e| format!("SPARQL query failed: {}", e))?;

        let mut values = Vec::new();
        match results {
            QueryResults::Solutions(solutions) => {
                for solution in solutions {
                    let solution =
                        solution.map_err(|e| format!("Failed to read query result: {}", e))?;
                    let mut obj = serde_json::Map::new();
                    for (var, value) in solution.iter() {
                        obj.insert(
                            var.to_string(),
                            serde_json::Value::String(normalize_term(&value.to_string())),
                        );
                    }
                    values.push(serde_json::Value::Object(obj));
                }
            }
            QueryResults::Graph(graph) => {
                for triple in graph {
                    let triple =
                        triple.map_err(|e| format!("Failed to read graph result: {}", e))?;
                    let mut obj = serde_json::Map::new();
                    obj.insert(
                        "subject".to_string(),
                        serde_json::Value::String(triple.subject.to_string()),
                    );
                    obj.insert(
                        "predicate".to_string(),
                        serde_json::Value::String(triple.predicate.to_string()),
                    );
                    obj.insert(
                        "object".to_string(),
                        serde_json::Value::String(triple.object.to_string()),
                    );
                    values.push(serde_json::Value::Object(obj));
                }
            }
            QueryResults::Boolean(b) => {
                values.push(serde_json::json!({"result": b}));
            }
        }
        Ok(values)
    }

    pub fn search_entities(
        &self,
        keyword: &str,
        entity_type: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, String> {
        let escaped = Self::escape_sparql_string(keyword);

        let type_filter = match entity_type {
            Some(t) => {
                let type_iri = if t.contains("://") {
                    t.to_string()
                } else {
                    format!("http://agent-os.org/ontology/{}", t)
                };
                format!(
                    "?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .",
                    type_iri
                )
            }
            None => String::new(),
        };

        let sparql = format!(
            "SELECT DISTINCT ?s ?label WHERE {{ \
               GRAPH ?g {{ \
                 ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label . \
                 {} \
                 FILTER(CONTAINS(LCASE(STR(?label)), LCASE(\"{}\"))) \
               }} \
             }}",
            type_filter, escaped
        );

        self.query_sparql(&sparql, None)
    }

    pub fn get_neighbors(
        &self,
        entity_id: &str,
        depth: usize,
    ) -> Result<serde_json::Value, String> {
        if depth == 0 || depth > 3 {
            return Ok(serde_json::json!({
                "entity": entity_id,
                "neighbors": [],
                "depth": depth
            }));
        }

        let mut all_neighbors = Vec::new();
        let mut visited = std::collections::HashSet::new();
        visited.insert(entity_id.to_string());
        let mut current_level = vec![entity_id.to_string()];

        for level in 0..depth {
            let mut next_level = Vec::new();

            for node_id in &current_level {
                let node = format!("<{}>", node_id);

                let out_sparql =
                    format!("SELECT ?p ?o WHERE {{ GRAPH ?g {{ {} ?p ?o . }} }}", node);
                let out_results = self.query_sparql(&out_sparql, None)?;

                for row in &out_results {
                    if let (Some(pred), Some(obj)) = (
                        row.get("?p").and_then(|v| v.as_str()),
                        row.get("?o").and_then(|v| v.as_str()),
                    ) {
                        let obj_clean = obj.trim_start_matches('<').trim_end_matches('>');
                        all_neighbors.push(serde_json::json!({
                            "source": node_id,
                            "predicate": pred,
                            "target": obj_clean,
                            "direction": "outgoing",
                            "level": level + 1
                        }));
                        if !visited.contains(obj_clean) && level + 1 < depth {
                            next_level.push(obj_clean.to_string());
                        }
                        visited.insert(obj_clean.to_string());
                    }
                }

                let in_sparql = format!("SELECT ?s ?p WHERE {{ GRAPH ?g {{ ?s ?p {} . }} }}", node);
                let in_results = self.query_sparql(&in_sparql, None)?;

                for row in &in_results {
                    if let (Some(subj), Some(pred)) = (
                        row.get("?s").and_then(|v| v.as_str()),
                        row.get("?p").and_then(|v| v.as_str()),
                    ) {
                        let subj_clean = subj.trim_start_matches('<').trim_end_matches('>');
                        all_neighbors.push(serde_json::json!({
                            "source": subj_clean,
                            "predicate": pred,
                            "target": node_id,
                            "direction": "incoming",
                            "level": level + 1
                        }));
                        if !visited.contains(subj_clean) && level + 1 < depth {
                            next_level.push(subj_clean.to_string());
                        }
                        visited.insert(subj_clean.to_string());
                    }
                }
            }

            current_level = next_level;
        }

        Ok(serde_json::json!({
            "entity": entity_id,
            "neighbors": all_neighbors,
            "depth": depth,
            "total_found": all_neighbors.len()
        }))
    }

    fn escape_sparql_string(s: &str) -> String {
        let mut escaped = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '\\' => escaped.push_str("\\\\"),
                '"' => escaped.push_str("\\\""),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                c if c.is_control() => {
                    escaped.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => escaped.push(c),
            }
        }
        escaped
    }

    pub fn default_graph(&self) -> &str {
        &self.default_graph
    }

    // Knowledge graph → vector index sync is handled externally via HyperspaceEngine.
    // See src/memory/hyperspace_store.rs for the vector store API.
    // To index KG entities into vector search, iterate results from SPARQL queries
    // and call HyperspaceStore::upsert() with the entity IRI and text content.
}

fn normalize_term(s: &str) -> String {
    if s.starts_with('<') && s.ends_with('>') {
        s[1..s.len() - 1].to_string()
    } else if s.starts_with('"') {
        if let Some(pos) = s.rfind("\"^^<") {
            s[1..pos].to_string()
        } else if let Some(pos) = s.rfind("\"@") {
            s[1..pos].to_string()
        } else if s.ends_with('"') && s.len() > 1 {
            s[1..s.len() - 1].to_string()
        } else {
            s.to_string()
        }
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_graph::types::RdfValue;

    static RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    static RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
    static PERSON: &str = "http://example.org/Person";
    static VEHICLE: &str = "http://example.org/Vehicle";
    static KNOWS: &str = "http://example.org/knows";
    static TEST_GRAPH: &str = "http://test/graph";

    fn make_quad(s: &str, p: &str, o: RdfValue) -> RdfQuad {
        RdfQuad {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o,
            graph: Some(TEST_GRAPH.to_string()),
        }
    }

    #[test]
    fn test_write_and_query_quads() {
        let store = KnowledgeGraphStore::new().unwrap();

        let quads = vec![
            make_quad(
                "http://example.org/alice",
                RDF_TYPE,
                RdfValue::Iri(PERSON.to_string()),
            ),
            make_quad(
                "http://example.org/alice",
                RDFS_LABEL,
                RdfValue::Literal("Alice".to_string()),
            ),
            make_quad(
                "http://example.org/alice",
                "http://example.org/age",
                RdfValue::TypedLiteral(
                    "30".to_string(),
                    "http://www.w3.org/2001/XMLSchema#integer".to_string(),
                ),
            ),
        ];

        store.write_quads(&quads, TEST_GRAPH).unwrap();

        let results = store
            .query_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }", Some(TEST_GRAPH))
            .unwrap();

        assert_eq!(results.len(), 3, "should return 3 triples");

        let labels: Vec<&str> = results
            .iter()
            .filter_map(|r| {
                if r.get("?p").and_then(|v| v.as_str()) == Some(RDFS_LABEL) {
                    r.get("?o").and_then(|v| v.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], "Alice");
    }

    #[test]
    fn test_write_empty_quads() {
        let store = KnowledgeGraphStore::new().unwrap();
        let result = store.write_quads(&[], TEST_GRAPH);
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_entities() {
        let store = KnowledgeGraphStore::new().unwrap();

        let quads = vec![
            make_quad(
                "http://example.org/alice",
                RDF_TYPE,
                RdfValue::Iri(PERSON.to_string()),
            ),
            make_quad(
                "http://example.org/alice",
                RDFS_LABEL,
                RdfValue::Literal("Alice Johnson".to_string()),
            ),
            make_quad(
                "http://example.org/bob",
                RDF_TYPE,
                RdfValue::Iri(PERSON.to_string()),
            ),
            make_quad(
                "http://example.org/bob",
                RDFS_LABEL,
                RdfValue::Literal("Bob Smith".to_string()),
            ),
            make_quad(
                "http://example.org/car",
                RDF_TYPE,
                RdfValue::Iri(VEHICLE.to_string()),
            ),
            make_quad(
                "http://example.org/car",
                RDFS_LABEL,
                RdfValue::Literal("Toyota Car".to_string()),
            ),
        ];

        store.write_quads(&quads, TEST_GRAPH).unwrap();

        let results = store.search_entities("alice", None).unwrap();
        assert_eq!(results.len(), 1, "fuzzy search should find Alice");
        let label = results[0].get("?label").and_then(|v| v.as_str()).unwrap();
        assert!(label.contains("Alice"));

        let person_results = store.search_entities("o", Some(PERSON)).unwrap();
        assert!(
            person_results.len() >= 2,
            "search by type Person should find at least 2"
        );

        let vehicle_results = store.search_entities("alice", Some(VEHICLE)).unwrap();
        assert_eq!(vehicle_results.len(), 0, "Alice is not Vehicle type");
    }

    #[test]
    fn test_search_entities_case_insensitive() {
        let store = KnowledgeGraphStore::new().unwrap();

        let quads = vec![make_quad(
            "http://example.org/alice",
            RDFS_LABEL,
            RdfValue::Literal("Alice".to_string()),
        )];

        store.write_quads(&quads, TEST_GRAPH).unwrap();

        let upper = store.search_entities("ALICE", None).unwrap();
        assert_eq!(
            upper.len(),
            1,
            "case-insensitive search should find results"
        );

        let lower = store.search_entities("alice", None).unwrap();
        assert_eq!(lower.len(), 1);

        let mixed = store.search_entities("AlIcE", None).unwrap();
        assert_eq!(mixed.len(), 1);
    }

    #[test]
    fn test_get_neighbors() {
        let store = KnowledgeGraphStore::new().unwrap();

        let quads = vec![
            make_quad(
                "http://example.org/alice",
                RDF_TYPE,
                RdfValue::Iri(PERSON.to_string()),
            ),
            make_quad(
                "http://example.org/alice",
                RDFS_LABEL,
                RdfValue::Literal("Alice".to_string()),
            ),
            make_quad(
                "http://example.org/alice",
                KNOWS,
                RdfValue::Iri("http://example.org/bob".to_string()),
            ),
            make_quad(
                "http://example.org/bob",
                RDF_TYPE,
                RdfValue::Iri(PERSON.to_string()),
            ),
            make_quad(
                "http://example.org/bob",
                RDFS_LABEL,
                RdfValue::Literal("Bob".to_string()),
            ),
            make_quad(
                "http://example.org/bob",
                KNOWS,
                RdfValue::Iri("http://example.org/charlie".to_string()),
            ),
            make_quad(
                "http://example.org/charlie",
                RDFS_LABEL,
                RdfValue::Literal("Charlie".to_string()),
            ),
        ];

        store.write_quads(&quads, TEST_GRAPH).unwrap();

        let result = store.get_neighbors("http://example.org/alice", 1).unwrap();

        let neighbors = result.get("neighbors").unwrap().as_array().unwrap();
        assert!(
            neighbors.len() >= 2,
            "1-hop traversal should find at least 2 neighbors (type + label + knows)"
        );

        let knows: Vec<_> = neighbors
            .iter()
            .filter(|n| n.get("predicate").and_then(|v| v.as_str()) == Some(KNOWS))
            .collect();
        assert_eq!(knows.len(), 1, "should find 1 knows relation");
        assert_eq!(
            knows[0].get("target").and_then(|v| v.as_str()),
            Some("http://example.org/bob")
        );
    }

    #[test]
    fn test_get_neighbors_depth_2() {
        let store = KnowledgeGraphStore::new().unwrap();

        let quads = vec![
            make_quad(
                "http://example.org/alice",
                KNOWS,
                RdfValue::Iri("http://example.org/bob".to_string()),
            ),
            make_quad(
                "http://example.org/bob",
                KNOWS,
                RdfValue::Iri("http://example.org/charlie".to_string()),
            ),
        ];

        store.write_quads(&quads, TEST_GRAPH).unwrap();

        let result = store.get_neighbors("http://example.org/alice", 2).unwrap();

        let neighbors = result.get("neighbors").unwrap().as_array().unwrap();
        assert!(
            neighbors.len() >= 2,
            "2-hop traversal should find alice->bob and bob->charlie"
        );

        let targets: Vec<_> = neighbors
            .iter()
            .filter_map(|n| {
                if n.get("predicate").and_then(|v| v.as_str()) == Some(KNOWS) {
                    n.get("target").and_then(|v| v.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            targets.contains(&"http://example.org/bob"),
            "should include bob as direct neighbor"
        );
        assert!(
            targets.contains(&"http://example.org/charlie"),
            "should include charlie as 2-hop neighbor"
        );
    }

    #[test]
    fn test_get_neighbors_zero_depth() {
        let store = KnowledgeGraphStore::new().unwrap();
        let result = store.get_neighbors("http://example.org/alice", 0).unwrap();
        let neighbors = result.get("neighbors").unwrap().as_array().unwrap();
        assert!(
            neighbors.is_empty(),
            "depth=0 should return empty neighbor list"
        );
    }

    #[test]
    fn test_with_graph_constructor() {
        let store = KnowledgeGraphStore::with_graph("http://test/custom").unwrap();
        assert_eq!(store.default_graph(), "http://test/custom");
    }

    #[test]
    fn test_default_graph() {
        let store = KnowledgeGraphStore::new().unwrap();
        assert_eq!(store.default_graph(), "graph:world");
    }

    #[test]
    fn test_query_sparql_no_named_graph() {
        let store = KnowledgeGraphStore::new().unwrap();

        let quads = vec![make_quad(
            "http://example.org/x",
            RDFS_LABEL,
            RdfValue::Literal("X".to_string()),
        )];

        store.write_quads(&quads, TEST_GRAPH).unwrap();

        let results = store
            .query_sparql("SELECT ?s ?p ?o WHERE { GRAPH ?g { ?s ?p ?o } }", None)
            .unwrap();
        assert!(
            !results.is_empty(),
            "using GRAPH ?g should query triples in named graph"
        );
    }

    #[test]
    fn test_query_sparql_with_graph_clause() {
        let store = KnowledgeGraphStore::new().unwrap();

        let quads = vec![make_quad(
            "http://example.org/x",
            RDFS_LABEL,
            RdfValue::Literal("X".to_string()),
        )];

        store.write_quads(&quads, TEST_GRAPH).unwrap();

        let sparql = format!(
            "SELECT ?s ?p ?o WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
            TEST_GRAPH
        );
        let results = store.query_sparql(&sparql, Some(TEST_GRAPH)).unwrap();
        assert_eq!(
            results.len(),
            1,
            "should not double-wrap when GRAPH clause already exists"
        );
    }

    #[test]
    fn test_incoming_neighbors() {
        let store = KnowledgeGraphStore::new().unwrap();

        let quads = vec![make_quad(
            "http://example.org/alice",
            KNOWS,
            RdfValue::Iri("http://example.org/bob".to_string()),
        )];

        store.write_quads(&quads, TEST_GRAPH).unwrap();

        let result = store.get_neighbors("http://example.org/bob", 1).unwrap();

        let neighbors = result.get("neighbors").unwrap().as_array().unwrap();
        let incoming: Vec<_> = neighbors
            .iter()
            .filter(|n| n.get("direction").and_then(|v| v.as_str()) == Some("incoming"))
            .collect();
        assert_eq!(
            incoming.len(),
            1,
            "bob should have 1 incoming edge (from alice)"
        );
        assert_eq!(
            incoming[0].get("source").and_then(|v| v.as_str()),
            Some("http://example.org/alice")
        );
    }
}
