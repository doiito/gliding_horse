use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::knowledge_graph::rdf_mapper::RdfMapper;
use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::knowledge_graph::types::{EdgeDef, LLMExtractionOutput, NodeDef};

use super::{GraphifyResult, SchemaAnalysis};

pub struct GraphifyEngine {
    store: KnowledgeGraphStore,
    #[allow(dead_code)]
    max_entities: usize,
}

impl GraphifyEngine {
    pub fn new(max_entities: usize) -> Result<Self, String> {
        let store = KnowledgeGraphStore::new()?;
        Ok(Self {
            store,
            max_entities,
        })
    }

    pub fn with_shared_store(
        store: Arc<oxigraph::store::Store>,
        max_entities: usize,
    ) -> Result<Self, String> {
        let store = KnowledgeGraphStore::with_shared_store(store)?;
        Ok(Self {
            store,
            max_entities,
        })
    }

    pub fn graphify_json(
        &mut self,
        json: &Value,
        call_id: &str,
        max_entities: usize,
    ) -> GraphifyResult {
        let graph_name = format!("graph:tool-result:{}", call_id);

        let (mut nodes, mut edges) = Self::json_to_graph(json, call_id, max_entities);

        Self::normalize_iris(&mut nodes, &mut edges);

        let entity_count = nodes.len();
        let relation_count = edges.len();

        let output = LLMExtractionOutput { nodes, edges };
        let mapping = RdfMapper::map_extraction(&output, &graph_name);

        if let Err(e) = self.store.write_quads(&mapping.quads, &graph_name) {
            return GraphifyResult {
                graph_name,
                entity_count: 0,
                relation_count: 0,
                entity_types: vec![],
                summary: format!("Graphify failed: {}", e),
                micro_tools: vec![],
            };
        }

        let analysis = Self::analyze_schema(&mapping.quads);
        let summary = Self::generate_data_summary(&analysis);

        GraphifyResult {
            graph_name,
            entity_count,
            relation_count,
            entity_types: analysis
                .entity_types
                .iter()
                .map(|(t, _)| t.clone())
                .collect(),
            summary,
            micro_tools: vec![],
        }
    }

    fn json_to_graph(
        json: &Value,
        call_id: &str,
        max_entities: usize,
    ) -> (Vec<NodeDef>, Vec<EdgeDef>) {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        match json {
            Value::Array(arr) => {
                for (idx, item) in arr.iter().enumerate() {
                    if nodes.len() >= max_entities {
                        break;
                    }
                    if let Value::Object(obj) = item {
                        let id = Self::extract_id(obj, idx, call_id);
                        let label = Self::extract_label(obj, &id);
                        let node_type = Self::extract_type(obj);

                        let mut properties = HashMap::new();
                        let mut child_edges = Vec::new();

                        for (key, value) in obj {
                            match value {
                                Value::Object(child) => {
                                    let child_id = format!("{}_{}", id, key);
                                    let child_label = Self::extract_label(child, &child_id);
                                    let child_type = format!(
                                        "https://agent-os.org/ontology/tool-result/{}_{}",
                                        Self::to_iri_short(&node_type),
                                        key
                                    );

                                    let mut child_props = HashMap::new();
                                    for (ck, cv) in child {
                                        if !cv.is_object() && !cv.is_array() {
                                            child_props.insert(ck.clone(), cv.clone());
                                        }
                                    }

                                    if nodes.len() < max_entities {
                                        nodes.push(NodeDef {
                                            id: child_id.clone(),
                                            node_type: child_type,
                                            label: child_label,
                                            description: None,
                                            properties: child_props,
                                        });
                                        child_edges.push(EdgeDef {
                                            source: id.clone(),
                                            target: child_id,
                                            relation: format!("has_{}", key),
                                            properties: HashMap::new(),
                                        });
                                    }
                                }
                                Value::Array(arr_val) => {
                                    for (ai, av) in arr_val.iter().enumerate() {
                                        if nodes.len() >= max_entities {
                                            break;
                                        }
                                        if av.is_object() {
                                            let arr_item_id = format!("{}_{}_{}", id, key, ai);
                                            let arr_item_label = Self::extract_label(
                                                av.as_object().expect("checked is_object above"),
                                                &arr_item_id,
                                            );
                                            let arr_item_type = format!(
                                                "https://agent-os.org/ontology/tool-result/{}_{}_item",
                                                Self::to_iri_short(&node_type),
                                                key
                                            );

                                            let mut arr_props = HashMap::new();
                                            if let Some(obj) = av.as_object() {
                                                for (ak, av_val) in obj {
                                                    if !av_val.is_object() && !av_val.is_array() {
                                                        arr_props
                                                            .insert(ak.clone(), av_val.clone());
                                                    }
                                                }
                                            }

                                            nodes.push(NodeDef {
                                                id: arr_item_id.clone(),
                                                node_type: arr_item_type,
                                                label: arr_item_label,
                                                description: None,
                                                properties: arr_props,
                                            });
                                            child_edges.push(EdgeDef {
                                                source: id.clone(),
                                                target: arr_item_id,
                                                relation: format!("has_{}", key),
                                                properties: HashMap::new(),
                                            });
                                        }
                                    }
                                }
                                _ => {
                                    properties.insert(key.clone(), value.clone());
                                }
                            }
                        }

                        nodes.insert(
                            0,
                            NodeDef {
                                id: id.clone(),
                                node_type,
                                label,
                                description: None,
                                properties,
                            },
                        );
                        edges.extend(child_edges);
                    } else {
                        let id = format!("{}_{}", call_id, idx);
                        nodes.push(NodeDef {
                            id: id.clone(),
                            node_type: "https://agent-os.org/ontology/tool-result/ScalarItem"
                                .to_string(),
                            label: format!("Item {}", idx),
                            description: None,
                            properties: {
                                let mut p = HashMap::new();
                                p.insert("value".to_string(), item.clone());
                                p
                            },
                        });
                    }
                }
            }
            Value::Object(obj) => {
                let id = format!("{}_root", call_id);
                let label = Self::extract_label(obj, &id);
                let node_type = "https://agent-os.org/ontology/tool-result/RootObject".to_string();

                let mut properties = HashMap::new();
                let mut child_edges = Vec::new();

                for (key, value) in obj {
                    match value {
                        Value::Object(child) => {
                            if nodes.len() < max_entities {
                                let child_id = format!("{}_{}", id, key);
                                let child_label = Self::extract_label(child, &child_id);
                                let child_type = format!(
                                    "https://agent-os.org/ontology/tool-result/Child_{}",
                                    key
                                );

                                let mut child_props = HashMap::new();
                                for (ck, cv) in child {
                                    if !cv.is_object() && !cv.is_array() {
                                        child_props.insert(ck.clone(), cv.clone());
                                    }
                                }

                                nodes.push(NodeDef {
                                    id: child_id.clone(),
                                    node_type: child_type,
                                    label: child_label,
                                    description: None,
                                    properties: child_props,
                                });
                                child_edges.push(EdgeDef {
                                    source: id.clone(),
                                    target: child_id,
                                    relation: format!("has_{}", key),
                                    properties: HashMap::new(),
                                });
                            }
                        }
                        Value::Array(arr_val) => {
                            for (ai, av) in arr_val.iter().enumerate() {
                                if nodes.len() >= max_entities {
                                    break;
                                }
                                if let Some(child_obj) = av.as_object() {
                                    let arr_item_id = format!("{}_{}_{}", id, key, ai);
                                    let arr_item_label =
                                        Self::extract_label(child_obj, &arr_item_id);
                                    let arr_item_type = format!(
                                        "https://agent-os.org/ontology/tool-result/{}_item",
                                        key
                                    );

                                    let mut arr_props = HashMap::new();
                                    for (ak, av_val) in child_obj {
                                        if !av_val.is_object() && !av_val.is_array() {
                                            arr_props.insert(ak.clone(), av_val.clone());
                                        }
                                    }

                                    nodes.push(NodeDef {
                                        id: arr_item_id.clone(),
                                        node_type: arr_item_type,
                                        label: arr_item_label,
                                        description: None,
                                        properties: arr_props,
                                    });
                                    child_edges.push(EdgeDef {
                                        source: id.clone(),
                                        target: arr_item_id,
                                        relation: format!("has_{}", key),
                                        properties: HashMap::new(),
                                    });
                                }
                            }
                        }
                        _ => {
                            properties.insert(key.clone(), value.clone());
                        }
                    }
                }

                nodes.insert(
                    0,
                    NodeDef {
                        id: id.clone(),
                        node_type,
                        label,
                        description: None,
                        properties,
                    },
                );
                edges.extend(child_edges);
            }
            _ => {
                nodes.push(NodeDef {
                    id: format!("{}_value", call_id),
                    node_type: "https://agent-os.org/ontology/tool-result/ScalarValue".to_string(),
                    label: "Value".to_string(),
                    description: None,
                    properties: {
                        let mut p = HashMap::new();
                        p.insert("value".to_string(), json.clone());
                        p
                    },
                });
            }
        }

        (nodes, edges)
    }

    fn extract_id(obj: &serde_json::Map<String, Value>, idx: usize, call_id: &str) -> String {
        for key in &["id", "ID", "_id", "uid", "key"] {
            if let Some(Value::String(s)) = obj.get(*key) {
                return s.clone();
            }
            if let Some(Value::Number(n)) = obj.get(*key) {
                return format!("{}", n);
            }
        }
        format!("{}_{}", call_id, idx)
    }

    fn extract_label(obj: &serde_json::Map<String, Value>, fallback: &str) -> String {
        for key in &["name", "title", "label", "displayName"] {
            if let Some(Value::String(s)) = obj.get(*key) {
                return s.clone();
            }
        }
        fallback.to_string()
    }

    fn extract_type(obj: &serde_json::Map<String, Value>) -> String {
        for key in &["@type", "type", "kind", "category"] {
            if let Some(Value::String(s)) = obj.get(*key) {
                return Self::to_iri(s);
            }
        }
        "https://agent-os.org/ontology/tool-result/Entity".to_string()
    }

    fn to_iri(s: &str) -> String {
        if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("iri://") {
            s.to_string()
        } else {
            format!("https://agent-os.org/ontology/tool-result/{}", s)
        }
    }

    fn to_iri_short(s: &str) -> String {
        s.split('/').last().unwrap_or(s).to_string()
    }

    fn normalize_iris(nodes: &mut [NodeDef], edges: &mut [EdgeDef]) {
        for node in nodes.iter_mut() {
            let mut normalized = HashMap::new();
            for (key, value) in node.properties.drain() {
                let iri_key = if key.starts_with("http://")
                    || key.starts_with("https://")
                    || key.starts_with("iri://")
                {
                    key
                } else {
                    format!("https://agent-os.org/ontology/property/{}", key)
                };
                normalized.insert(iri_key, value);
            }
            node.properties = normalized;
        }

        for edge in edges.iter_mut() {
            if !edge.relation.starts_with("http://")
                && !edge.relation.starts_with("https://")
                && !edge.relation.starts_with("iri://")
            {
                edge.relation = format!("https://agent-os.org/ontology/relation/{}", edge.relation);
            }
        }
    }

    pub fn analyze_schema(quads: &[crate::knowledge_graph::types::RdfQuad]) -> SchemaAnalysis {
        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label";

        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut relation_types: Vec<String> = Vec::new();
        let mut property_names: Vec<String> = Vec::new();
        let mut entity_count = 0;
        let mut relation_count = 0;

        for quad in quads {
            if quad.predicate == rdf_type {
                if let crate::knowledge_graph::types::RdfValue::Iri(ref type_iri) = quad.object {
                    *type_counts.entry(type_iri.clone()).or_insert(0) += 1;
                    entity_count += 1;
                }
            } else if quad.predicate == rdfs_label {
                // skip
            } else {
                let pred_short = quad
                    .predicate
                    .split('/')
                    .last()
                    .unwrap_or(&quad.predicate)
                    .to_string();

                if matches!(quad.object, crate::knowledge_graph::types::RdfValue::Iri(_)) {
                    if !relation_types.contains(&pred_short) {
                        relation_types.push(pred_short);
                    }
                    relation_count += 1;
                } else {
                    if !property_names.contains(&pred_short) {
                        property_names.push(pred_short);
                    }
                }
            }
        }

        let mut entity_types: Vec<(String, usize)> = type_counts.into_iter().collect();
        entity_types.sort_by(|a, b| b.1.cmp(&a.1));

        SchemaAnalysis {
            entity_types,
            relation_types,
            property_names,
            total_entities: entity_count,
            total_relations: relation_count,
        }
    }

    pub fn generate_data_summary(analysis: &SchemaAnalysis) -> String {
        let mut summary = format!(
            "Data summary: {} entities, {} relations\n",
            analysis.total_entities, analysis.total_relations
        );

        if !analysis.entity_types.is_empty() {
            summary.push_str("Entity type distribution:\n");
            for (type_name, count) in &analysis.entity_types {
                let short = type_name.split('/').last().unwrap_or(type_name);
                summary.push_str(&format!("  - {}: {} count\n", short, count));
            }
        }

        if !analysis.relation_types.is_empty() {
            summary.push_str(&format!(
                "Relation types: {}\n",
                analysis.relation_types.join(", ")
            ));
        }

        if !analysis.property_names.is_empty() {
            summary.push_str(&format!(
                "Property fields: {} (total {})\n",
                analysis
                    .property_names
                    .iter()
                    .take(10)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                analysis.property_names.len()
            ));
        }

        summary
    }

    pub fn get_store(&self) -> &KnowledgeGraphStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_graphify_json_array() {
        let mut engine = GraphifyEngine::new(100).unwrap();
        let items: Vec<Value> = (0..10)
            .map(|i| json!({"id": format!("item_{}", i), "name": format!("item_{}", i), "value": i * 10}))
            .collect();
        let result = engine.graphify_json(&Value::Array(items), "test_call_1", 100);

        assert!(
            result.entity_count > 0,
            "entity_count should be > 0, summary: {}",
            result.summary
        );
        assert!(result.graph_name.contains("test_call_1"));
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_graphify_json_object() {
        let mut engine = GraphifyEngine::new(100).unwrap();
        let obj = json!({
            "name": "root",
            "items": [
                {"id": "child1", "name": "child1"},
                {"id": "child2", "name": "child2"}
            ]
        });
        let result = engine.graphify_json(&obj, "test_call_2", 100);

        assert!(
            result.entity_count >= 2,
            "entity_count should be >= 2, got: {}, summary: {}",
            result.entity_count,
            result.summary
        );
    }

    #[test]
    fn test_graphify_entity_limit() {
        let mut engine = GraphifyEngine::new(100).unwrap();
        let items: Vec<Value> = (0..50)
            .map(|i| json!({"id": format!("item_{}", i), "name": format!("item_{}", i)}))
            .collect();
        let result = engine.graphify_json(&Value::Array(items), "test_call_3", 5);

        assert!(result.entity_count <= 5);
    }

    #[test]
    fn test_analyze_schema() {
        use crate::knowledge_graph::types::{RdfQuad, RdfValue};

        let quads = vec![
            RdfQuad {
                subject: "iri://entity/a".into(),
                predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(),
                object: RdfValue::Iri("Person".into()),
                graph: Some("g".into()),
            },
            RdfQuad {
                subject: "iri://entity/b".into(),
                predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(),
                object: RdfValue::Iri("Person".into()),
                graph: Some("g".into()),
            },
            RdfQuad {
                subject: "iri://entity/a".into(),
                predicate: "knows".into(),
                object: RdfValue::Iri("iri://entity/b".into()),
                graph: Some("g".into()),
            },
            RdfQuad {
                subject: "iri://entity/a".into(),
                predicate: "age".into(),
                object: RdfValue::Literal("30".into()),
                graph: Some("g".into()),
            },
        ];

        let analysis = GraphifyEngine::analyze_schema(&quads);

        assert_eq!(analysis.total_entities, 2);
        assert_eq!(analysis.entity_types.len(), 1);
        assert_eq!(analysis.entity_types[0], ("Person".to_string(), 2));
        assert!(analysis.relation_types.contains(&"knows".to_string()));
        assert!(analysis.property_names.contains(&"age".to_string()));
    }

    #[test]
    fn test_generate_data_summary() {
        let analysis = SchemaAnalysis {
            entity_types: vec![("Person".to_string(), 10), ("Org".to_string(), 3)],
            relation_types: vec!["works_for".to_string()],
            property_names: vec!["name".to_string(), "age".to_string()],
            total_entities: 13,
            total_relations: 5,
        };

        let summary = GraphifyEngine::generate_data_summary(&analysis);
        assert!(summary.contains("13 entities"));
        assert!(summary.contains("Person"));
        assert!(summary.contains("works_for"));
    }
}
