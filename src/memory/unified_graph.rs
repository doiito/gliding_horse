use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use oxigraph::model::{Literal, NamedNode, Quad, Term};
use oxigraph::sparql::QueryResults;
use oxigraph::store::{Store, Transaction};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// A knowledge graph entity with temporal metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub types: Vec<String>,
    pub properties: HashMap<String, PropertyValue>,
    /// When this entity was created
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    /// When this entity was last modified
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Reference(String),
    Array(Vec<PropertyValue>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub subject: String,
    pub predicate: String,
    pub object: RelationObject,
    /// When this relation was created
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationObject {
    Node(String),
    Value(PropertyValue),
}

#[derive(Debug, Clone)]
pub struct SparqlQueryResult {
    pub variables: Vec<String>,
    pub bindings: Vec<HashMap<String, SparqlValue>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparqlValue {
    Uri(String),
    Literal(String, Option<String>),
    BlankNode(String),
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_triples: usize,
    pub named_graphs: usize,
    pub entities: usize,
}

pub struct UnifiedGraphStore {
    store: Arc<Store>,
    default_graph: String,
}

impl UnifiedGraphStore {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing Unified Oxigraph Store (memory)");
        Ok(Self {
            store: Arc::new(Store::new()?),
            default_graph: "http://agent-os.org/graph/default".to_string(),
        })
    }

    pub fn new_persistent<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        info!(path = %path.as_ref().display(), "Initializing persistent Unified Oxigraph Store");
        Ok(Self {
            store: Arc::new(Store::open(path)?),
            default_graph: "http://agent-os.org/graph/default".to_string(),
        })
    }

    /// Create a UnifiedGraphStore backed by a shared Oxigraph `Arc<Store>`.
    ///
    /// All subsystems (skill graph, knowledge bridge, blackboard) that receive
    /// the same `Arc<Store>` will share the underlying Oxigraph store, enabling
    /// cross-subsystem SPARQL joins via named graphs.
    ///
    /// This is the production pattern used by the gRPC server — it creates one
    /// `UnifiedGraphStore` and passes `Arc::clone(&store.store())` to all consumers
    /// via their respective `with_shared_store()` / `with_oxi_store()` builders.
    pub fn with_shared_store(store: Arc<Store>) -> Self {
        Self {
            store,
            default_graph: "http://agent-os.org/graph/default".to_string(),
        }
    }

    pub fn store(&self) -> Arc<Store> {
        self.store.clone()
    }

    /// Execute a real Oxigraph transaction.  The closure's writes are visible
    /// only if it returns `Ok`; returning an error aborts all writes.
    pub fn transaction<T>(
        &self,
        operation: impl FnOnce(&mut Transaction<'_>) -> Result<T, Box<dyn std::error::Error>>,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let mut transaction = self.store.start_transaction()?;
        let result = operation(&mut transaction)?;
        transaction.commit()?;
        Ok(result)
    }

    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.store)
    }

    fn parse_uri(&self, uri: &str) -> NamedNode {
        NamedNode::new_unchecked(uri)
    }

    pub fn add_entity(
        &self,
        entity: &Entity,
        graph: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.add_entities_atomic(std::slice::from_ref(entity), graph)
    }

    /// Atomically insert one or more complete entities into the selected graph.
    ///
    /// Oxigraph commits the closure only when every quad insert succeeds.  This
    /// replaces the former begin/commit/rollback façade whose writes had
    /// already escaped to the store before rollback was requested.
    pub fn add_entities_atomic(
        &self,
        entities: &[Entity],
        graph: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let graph_uri = graph.unwrap_or(&self.default_graph);
        let graph_node = self.parse_uri(graph_uri);

        self.transaction(|transaction| {
            for entity in entities {
                let subject = self.parse_uri(&entity.id);
                for type_uri in &entity.types {
                    transaction.insert(&Quad::new(
                        subject.clone(),
                        NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                        self.parse_uri(type_uri),
                        graph_node.clone(),
                    ));
                }
                for (predicate, value) in &entity.properties {
                    transaction.insert(&Quad::new(
                        subject.clone(),
                        NamedNode::new_unchecked(predicate),
                        self.property_value_to_term(value)?,
                        graph_node.clone(),
                    ));
                }
            }
            Ok(())
        })?;

        debug!(entities = entities.len(), graph = %graph_uri, "Entities added atomically");
        Ok(())
    }

    fn property_value_to_term(
        &self,
        value: &PropertyValue,
    ) -> Result<Term, Box<dyn std::error::Error>> {
        Ok(match value {
            PropertyValue::String(s) => Term::Literal(Literal::new_simple_literal(s)),
            PropertyValue::Integer(n) => Term::Literal(Literal::new_typed_literal(
                n.to_string(),
                NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
            )),
            PropertyValue::Float(f) => Term::Literal(Literal::new_typed_literal(
                f.to_string(),
                NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#double"),
            )),
            PropertyValue::Boolean(b) => Term::Literal(Literal::new_typed_literal(
                b.to_string(),
                NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#boolean"),
            )),
            PropertyValue::Reference(uri) => Term::NamedNode(self.parse_uri(uri)),
            PropertyValue::Array(items) => {
                // A repeated RDF predicate would lose the distinction between
                // a scalar and a one-element array. `rdf:JSON` is the RDF 1.1
                // typed-literal representation and preserves nested arrays
                // without creating orphaned RDF list blank nodes on updates.
                let serialized = serde_json::to_string(items)?;
                Term::Literal(Literal::new_typed_literal(
                    serialized,
                    NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON"),
                ))
            }
        })
    }

    pub fn get_entity(&self, id: &str, graph: Option<&str>) -> Option<Entity> {
        let subject = self.parse_uri(id);
        let mut types = Vec::new();
        let mut property_values: HashMap<String, Vec<PropertyValue>> = HashMap::new();

        let rdf_type = NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");

        let graph_node = graph.map(|g| self.parse_uri(g));
        let graph_name: Option<oxigraph::model::GraphNameRef<'_>> =
            graph_node.as_ref().map(|node| node.as_ref().into());

        let mut results: Vec<Quad> = self
            .store
            .quads_for_pattern(Some(subject.as_ref().into()), None, None, graph_name)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        // RDF does not define an insertion order for repeated predicates.
        // Canonicalize external multi-value reads while retaining the exact
        // order of `PropertyValue::Array`, which is stored as one rdf:JSON
        // literal rather than repeated predicates.
        results.sort_by_key(|quad| (quad.predicate.as_str().to_string(), quad.object.to_string()));

        for quad in &results {
            if quad.predicate == rdf_type {
                if let Term::NamedNode(node) = &quad.object {
                    types.push(node.as_str().to_string());
                }
            } else {
                let value = self.term_to_property_value(&quad.object);
                property_values
                    .entry(quad.predicate.as_str().to_string())
                    .or_default()
                    .push(value);
            }
        }

        let properties = property_values
            .into_iter()
            .map(|(predicate, mut values)| {
                let value = if values.len() == 1 {
                    values.pop().expect("single property value must exist")
                } else {
                    PropertyValue::Array(values)
                };
                (predicate, value)
            })
            .collect::<HashMap<_, _>>();

        if types.is_empty() && properties.is_empty() {
            None
        } else {
            let now = Utc::now();
            Some(Entity {
                id: id.to_string(),
                types,
                properties,
                created_at: now,
                updated_at: now,
            })
        }
    }

    fn term_to_property_value(&self, term: &Term) -> PropertyValue {
        match term {
            Term::NamedNode(node) => PropertyValue::Reference(node.as_str().to_string()),
            Term::Literal(lit) => {
                let value = lit.value();
                if lit.language().is_some() {
                    return PropertyValue::String(value.to_string());
                }
                let dtype = lit.datatype().as_str();
                if dtype == "http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON" {
                    return serde_json::from_str::<Vec<PropertyValue>>(value)
                        .map(PropertyValue::Array)
                        .unwrap_or_else(|_| PropertyValue::String(value.to_string()));
                }
                if dtype.contains("integer") {
                    value
                        .parse::<i64>()
                        .map(PropertyValue::Integer)
                        .unwrap_or_else(|_| PropertyValue::String(value.to_string()))
                } else if dtype.contains("double")
                    || dtype.contains("float")
                    || dtype.contains("decimal")
                {
                    value
                        .parse::<f64>()
                        .map(PropertyValue::Float)
                        .unwrap_or_else(|_| PropertyValue::String(value.to_string()))
                } else if dtype.contains("boolean") {
                    value
                        .parse::<bool>()
                        .map(PropertyValue::Boolean)
                        .unwrap_or_else(|_| PropertyValue::String(value.to_string()))
                } else {
                    PropertyValue::String(value.to_string())
                }
            }
            _ => PropertyValue::String(term.to_string()),
        }
    }

    pub fn update_entity(
        &self,
        entity: &Entity,
        graph: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let graph_uri = graph.unwrap_or(&self.default_graph);
        let graph_node = self.parse_uri(graph_uri);
        let subject = self.parse_uri(&entity.id);
        self.transaction(|transaction| {
            let old_quads: Vec<Quad> = transaction
                .quads_for_pattern(
                    Some(subject.as_ref().into()),
                    None,
                    None,
                    Some(graph_node.as_ref().into()),
                )
                .collect::<Result<_, _>>()?;
            for quad in old_quads {
                transaction.remove(&quad);
            }
            for type_uri in &entity.types {
                transaction.insert(&Quad::new(
                    subject.clone(),
                    NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                    self.parse_uri(type_uri),
                    graph_node.clone(),
                ));
            }
            for (predicate, value) in &entity.properties {
                transaction.insert(&Quad::new(
                    subject.clone(),
                    NamedNode::new_unchecked(predicate),
                    self.property_value_to_term(value)?,
                    graph_node.clone(),
                ));
            }
            Ok(())
        })?;
        debug!(entity_id = %entity.id, "Entity updated");
        Ok(())
    }

    pub fn delete_entity(
        &self,
        id: &str,
        graph: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let subject = self.parse_uri(id);
        let graph_uri = graph.unwrap_or(&self.default_graph);
        let graph_ref = self.parse_uri(graph_uri);

        let quads_to_remove: Vec<Quad> = self
            .store
            .quads_for_pattern(
                Some(subject.as_ref().into()),
                None,
                None,
                Some(graph_ref.as_ref().into()),
            )
            .collect::<Result<Vec<_>, _>>()?;

        for quad in &quads_to_remove {
            self.store.remove(quad)?;
        }

        debug!(entity_id = %id, removed = quads_to_remove.len(), "Entity deleted");
        Ok(())
    }

    pub fn add_relation(
        &self,
        relation: &Relation,
        graph: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let graph_uri = graph.unwrap_or(&self.default_graph);
        let graph_node = self.parse_uri(graph_uri);
        let subject = self.parse_uri(&relation.subject);
        let predicate = NamedNode::new_unchecked(&relation.predicate);
        let object = match &relation.object {
            RelationObject::Node(uri) => self.parse_uri(uri).into(),
            RelationObject::Value(v) => self.property_value_to_term(v)?,
        };

        let quad = Quad::new(subject, predicate, object, graph_node);
        self.store.insert(&quad)?;

        debug!(subject = %relation.subject, predicate = %relation.predicate, "Relation added");
        Ok(())
    }

    pub fn delete_relation(
        &self,
        relation: &Relation,
        graph: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let graph_uri = graph.unwrap_or(&self.default_graph);
        let graph_node = self.parse_uri(graph_uri);
        let subject = self.parse_uri(&relation.subject);
        let predicate = NamedNode::new_unchecked(&relation.predicate);
        let object = match &relation.object {
            RelationObject::Node(uri) => self.parse_uri(uri).into(),
            RelationObject::Value(v) => self.property_value_to_term(v)?,
        };

        let quad = Quad::new(subject, predicate, object, graph_node);
        self.store.remove(&quad)?;

        debug!(subject = %relation.subject, predicate = %relation.predicate, "Relation deleted");
        Ok(())
    }

    #[allow(deprecated)]
    pub fn query(&self, sparql: &str) -> Result<SparqlQueryResult, Box<dyn std::error::Error>> {
        debug!(sparql_len = sparql.len(), "Executing SPARQL query");

        let results = self.store.query(sparql)?;
        let mut variables = Vec::new();
        let mut bindings = Vec::new();

        match results {
            QueryResults::Solutions(solutions) => {
                for result in solutions {
                    let result = result?;
                    if variables.is_empty() {
                        variables = result
                            .variables()
                            .iter()
                            .map(|v| v.as_str().to_string())
                            .collect();
                    }

                    let mut row = HashMap::new();
                    for (var, value) in result.iter() {
                        let sparql_value = match value {
                            Term::NamedNode(node) => SparqlValue::Uri(node.as_str().to_string()),
                            Term::Literal(lit) => {
                                let lang = lit.language().map(|l| l.to_string());
                                SparqlValue::Literal(lit.value().to_string(), lang)
                            }
                            Term::BlankNode(node) => {
                                SparqlValue::BlankNode(node.as_str().to_string())
                            }
                        };
                        row.insert(var.as_str().to_string(), sparql_value);
                    }
                    bindings.push(row);
                }
            }
            QueryResults::Boolean(b) => {
                debug!(result = b, "SPARQL ASK query completed");
            }
            QueryResults::Graph(_graph) => {
                debug!("SPARQL CONSTRUCT/DESCRIBE query completed");
            }
        }

        debug!(
            variables = variables.len(),
            bindings = bindings.len(),
            "SPARQL query completed"
        );
        Ok(SparqlQueryResult {
            variables,
            bindings,
        })
    }

    pub fn query_as_json(&self, sparql: &str) -> Result<String, Box<dyn std::error::Error>> {
        let result = self.query(sparql)?;
        let json = serde_json::to_string(&serde_json::json!({
            "variables": result.variables,
            "bindings": result.bindings.iter().map(|row| {
                let mut map = serde_json::Map::new();
                for (k, v) in row {
                    let value = match v {
                        SparqlValue::Uri(uri) => serde_json::json!({"type": "uri", "value": uri}),
                        SparqlValue::Literal(val, Some(lang)) => serde_json::json!({"type": "literal", "value": val, "lang": lang}),
                        SparqlValue::Literal(val, None) => serde_json::json!({"type": "literal", "value": val}),
                        SparqlValue::BlankNode(id) => serde_json::json!({"type": "bnode", "value": id}),
                    };
                    map.insert(k.clone(), value);
                }
                serde_json::Value::Object(map)
            }).collect::<Vec<_>>()
        }))?;
        Ok(json)
    }

    pub fn update(&self, sparql: &str) -> Result<(), Box<dyn std::error::Error>> {
        debug!(sparql_len = sparql.len(), "Executing SPARQL update");
        self.store.update(sparql)?;
        debug!("SPARQL update completed");
        Ok(())
    }

    pub fn create_named_graph(&self, graph_uri: &str) -> Result<(), Box<dyn std::error::Error>> {
        let graph_node = self.parse_uri(graph_uri);
        let quad = Quad::new(
            graph_node.clone(),
            NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            NamedNode::new_unchecked("http://www.w3.org/2002/07/owl#NamedGraph"),
            graph_node,
        );
        self.store.insert(&quad)?;
        info!(graph = %graph_uri, "Named graph created");
        Ok(())
    }

    pub fn drop_named_graph(&self, graph_uri: &str) -> Result<(), Box<dyn std::error::Error>> {
        let graph_node = self.parse_uri(graph_uri);
        let quads_to_remove: Vec<Quad> = self
            .store
            .quads_for_pattern(None, None, None, Some(graph_node.as_ref().into()))
            .collect::<Result<Vec<_>, _>>()?;

        for quad in &quads_to_remove {
            self.store.remove(quad)?;
        }

        info!(graph = %graph_uri, removed = quads_to_remove.len(), "Named graph dropped");
        Ok(())
    }

    pub fn list_named_graphs(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let graphs: Vec<String> = self
            .store
            .named_graphs()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|g| g.to_string())
            .collect();
        Ok(graphs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_store_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let graph_path = dir.path().join("unified-graph");
        let entity = Entity {
            id: "https://example.org/entity".to_string(),
            types: vec!["https://example.org/Type".to_string()],
            properties: HashMap::from([(
                "https://example.org/name".to_string(),
                PropertyValue::String("persisted".to_string()),
            )]),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        {
            let store = UnifiedGraphStore::new_persistent(&graph_path).unwrap();
            store
                .add_entity(&entity, Some("https://example.org/graph"))
                .unwrap();
        }

        let reopened = UnifiedGraphStore::new_persistent(&graph_path).unwrap();
        let loaded = reopened
            .get_entity(
                "https://example.org/entity",
                Some("https://example.org/graph"),
            )
            .expect("entity should be present after reopening persistent store");
        assert_eq!(loaded.types, entity.types);
        assert!(matches!(
            loaded.properties.get("https://example.org/name"),
            Some(PropertyValue::String(value)) if value == "persisted"
        ));
    }

    #[test]
    fn transaction_error_rolls_back_all_written_quads() {
        let store = UnifiedGraphStore::new().unwrap();
        let subject = NamedNode::new("https://example.org/transactional").unwrap();
        let predicate = NamedNode::new("https://example.org/name").unwrap();
        let graph = NamedNode::new("https://example.org/graph").unwrap();
        let quad = Quad::new(
            subject,
            predicate,
            Literal::new_simple_literal("temporary"),
            graph,
        );

        let result: Result<(), Box<dyn std::error::Error>> = store.transaction(|transaction| {
            transaction.insert(&quad);
            Err(Box::new(std::io::Error::other("abort")))
        });
        assert!(result.is_err());
        assert!(store
            .get_entity(
                "https://example.org/transactional",
                Some("https://example.org/graph")
            )
            .is_none());
    }

    #[test]
    fn update_entity_replaces_the_complete_entity_atomically() {
        let store = UnifiedGraphStore::new().unwrap();
        let graph = "https://example.org/graph";
        let mut first = Entity {
            id: "https://example.org/entity".to_string(),
            types: vec!["https://example.org/OldType".to_string()],
            properties: HashMap::from([(
                "https://example.org/name".to_string(),
                PropertyValue::String("old".to_string()),
            )]),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        store.add_entity(&first, Some(graph)).unwrap();
        first.types = vec!["https://example.org/NewType".to_string()];
        first.properties.insert(
            "https://example.org/name".to_string(),
            PropertyValue::String("new".to_string()),
        );
        store.update_entity(&first, Some(graph)).unwrap();

        let updated = store.get_entity(&first.id, Some(graph)).unwrap();
        assert_eq!(updated.types, first.types);
        assert!(matches!(
            updated.properties.get("https://example.org/name"),
            Some(PropertyValue::String(value)) if value == "new"
        ));
    }

    #[test]
    fn array_property_round_trips_without_losing_values_or_shape() {
        let store = UnifiedGraphStore::new().unwrap();
        let graph = "https://example.org/graph";
        let values = vec![
            PropertyValue::String("alpha".to_string()),
            PropertyValue::Integer(7),
            PropertyValue::Reference("https://example.org/related".to_string()),
            PropertyValue::Array(vec![PropertyValue::Boolean(true)]),
        ];
        let entity = Entity {
            id: "https://example.org/array-entity".to_string(),
            types: vec!["https://example.org/Type".to_string()],
            properties: HashMap::from([(
                "https://example.org/values".to_string(),
                PropertyValue::Array(values.clone()),
            )]),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        store.add_entity(&entity, Some(graph)).unwrap();
        let loaded = store.get_entity(&entity.id, Some(graph)).unwrap();
        assert_eq!(
            loaded.properties.get("https://example.org/values"),
            Some(&PropertyValue::Array(values)),
        );
    }

    #[test]
    fn repeated_rdf_predicates_are_exposed_as_an_array() {
        let store = UnifiedGraphStore::new().unwrap();
        let graph = "https://example.org/graph";
        let relation = |value: &str| Relation {
            subject: "https://example.org/repeated-values".to_string(),
            predicate: "https://example.org/tag".to_string(),
            object: RelationObject::Value(PropertyValue::String(value.to_string())),
            created_at: Utc::now(),
        };

        store.add_relation(&relation("first"), Some(graph)).unwrap();
        store
            .add_relation(&relation("second"), Some(graph))
            .unwrap();

        let loaded = store
            .get_entity("https://example.org/repeated-values", Some(graph))
            .unwrap();
        assert_eq!(
            loaded.properties.get("https://example.org/tag"),
            Some(&PropertyValue::Array(vec![
                PropertyValue::String("first".to_string()),
                PropertyValue::String("second".to_string()),
            ])),
        );
    }
}
