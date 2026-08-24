use super::types::{EdgeDef, LLMExtractionOutput, NodeDef, RdfMappingResult, RdfQuad, RdfValue};

pub struct RdfMapper;

impl RdfMapper {
    pub fn new() -> Self {
        Self
    }

    /// Replace or percent-encode characters not allowed in SPARQL IRIs.
    /// Allowed unreserved chars: A-Z a-z 0-9 - . _ ~
    /// Allowed sub-delims:      ! $ & ' ( ) * + , ; =
    /// Also preserved:          : / @ and already-valid `%XX` escapes.
    /// Everything else (spaces, brackets, etc.) → percent-encoded.
    pub(crate) fn sanitize_id(id: &str) -> String {
        let mut out = String::with_capacity(id.len() + 8);
        let bytes = id.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            let b = bytes[index];
            match b {
                b'A'..=b'Z'
                | b'a'..=b'z'
                | b'0'..=b'9'
                | b'-'
                | b'.'
                | b'_'
                | b'~'
                | b'!'
                | b'$'
                | b'&'
                | b'\''
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b','
                | b';'
                | b'='
                | b':'
                | b'/'
                | b'@' => out.push(b as char),
                b'%' if index + 2 < bytes.len()
                    && bytes[index + 1].is_ascii_hexdigit()
                    && bytes[index + 2].is_ascii_hexdigit() =>
                {
                    out.push('%');
                    out.push(bytes[index + 1] as char);
                    out.push(bytes[index + 2] as char);
                    index += 2;
                }
                b'%' => out.push_str("%25"),
                b' ' | b'\n' | b'\r' | b'\t' => out.push('_'),
                _ => out.push_str(&format!("%{:02X}", b)),
            }
            index += 1;
        }
        out
    }

    fn make_entity_iri(id: &str) -> String {
        format!("iri://entity/{}", Self::sanitize_id(id))
    }

    fn json_value_to_rdf(value: &serde_json::Value) -> Option<RdfValue> {
        match value {
            serde_json::Value::Null => None,
            serde_json::Value::Bool(b) => Some(RdfValue::TypedLiteral(
                b.to_string(),
                "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
            )),
            serde_json::Value::Number(n) => {
                let xsd_type = if n.is_i64() || n.is_u64() {
                    "http://www.w3.org/2001/XMLSchema#integer"
                } else {
                    "http://www.w3.org/2001/XMLSchema#decimal"
                };
                Some(RdfValue::TypedLiteral(n.to_string(), xsd_type.to_string()))
            }
            serde_json::Value::String(s) => Some(RdfValue::Literal(s.clone())),
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                Some(RdfValue::Literal(value.to_string()))
            }
        }
    }

    pub fn map_nodes(nodes: &[NodeDef], graph: &str) -> Vec<RdfQuad> {
        let mut quads = Vec::new();
        for node in nodes {
            let iri = Self::make_entity_iri(&node.id);

            quads.push(RdfQuad {
                subject: iri.clone(),
                predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                object: RdfValue::Iri(node.node_type.clone()),
                graph: Some(graph.to_string()),
            });

            quads.push(RdfQuad {
                subject: iri.clone(),
                predicate: "http://www.w3.org/2000/01/rdf-schema#label".to_string(),
                object: RdfValue::Literal(node.label.clone()),
                graph: Some(graph.to_string()),
            });

            for (key, value) in &node.properties {
                if let Some(rdf_value) = Self::json_value_to_rdf(value) {
                    let predicate = if key.contains('/') || key.contains('#') || key.contains(':') {
                        key.clone()
                    } else {
                        format!("https://agent-os.org/ontology/meta/{}", key)
                    };
                    quads.push(RdfQuad {
                        subject: iri.clone(),
                        predicate,
                        object: rdf_value,
                        graph: Some(graph.to_string()),
                    });
                }
            }
        }
        quads
    }

    pub fn map_edges(edges: &[EdgeDef], graph: &str) -> Vec<RdfQuad> {
        let mut quads = Vec::new();
        for edge in edges {
            let source_iri = Self::make_entity_iri(&edge.source);
            let target_iri = Self::make_entity_iri(&edge.target);

            quads.push(RdfQuad {
                subject: source_iri,
                predicate: edge.relation.clone(),
                object: RdfValue::Iri(target_iri),
                graph: Some(graph.to_string()),
            });
        }
        quads
    }

    pub fn map_edge_properties(edges: &[EdgeDef], graph: &str) -> Vec<RdfQuad> {
        let mut quads = Vec::new();
        for (idx, edge) in edges.iter().enumerate() {
            if edge.properties.is_empty() {
                continue;
            }

            let stmt_id = format!("stmt_{}_{}", Self::sanitize_id(&edge.source), idx);
            let stmt_iri = format!("iri://statement/{}", stmt_id);
            let source_iri = Self::make_entity_iri(&edge.source);
            let target_iri = Self::make_entity_iri(&edge.target);

            let rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

            quads.push(RdfQuad {
                subject: stmt_iri.clone(),
                predicate: format!("{}type", rdf),
                object: RdfValue::Iri(format!("{}Statement", rdf)),
                graph: Some(graph.to_string()),
            });
            quads.push(RdfQuad {
                subject: stmt_iri.clone(),
                predicate: format!("{}subject", rdf),
                object: RdfValue::Iri(source_iri),
                graph: Some(graph.to_string()),
            });
            quads.push(RdfQuad {
                subject: stmt_iri.clone(),
                predicate: format!("{}predicate", rdf),
                object: RdfValue::Iri(edge.relation.clone()),
                graph: Some(graph.to_string()),
            });
            quads.push(RdfQuad {
                subject: stmt_iri.clone(),
                predicate: format!("{}object", rdf),
                object: RdfValue::Iri(target_iri),
                graph: Some(graph.to_string()),
            });

            for (key, value) in &edge.properties {
                if let Some(rdf_value) = Self::json_value_to_rdf(value) {
                    quads.push(RdfQuad {
                        subject: stmt_iri.clone(),
                        predicate: key.clone(),
                        object: rdf_value,
                        graph: Some(graph.to_string()),
                    });
                }
            }
        }
        quads
    }

    pub fn map_extraction(output: &LLMExtractionOutput, graph: &str) -> RdfMappingResult {
        let mut quads = Vec::new();
        quads.extend(Self::map_nodes(&output.nodes, graph));
        quads.extend(Self::map_edges(&output.edges, graph));
        quads.extend(Self::map_edge_properties(&output.edges, graph));

        RdfMappingResult {
            entity_count: output.nodes.len(),
            relation_count: output.edges.len(),
            quads,
        }
    }

    pub fn map_to_rdf(&self, extraction: &LLMExtractionOutput) -> RdfMappingResult {
        Self::map_extraction(extraction, "default")
    }

    fn escape_sparql_literal(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }

    fn format_rdf_value(value: &RdfValue) -> String {
        match value {
            RdfValue::Iri(iri) => format!("<{}>", iri),
            RdfValue::Literal(s) => format!("\"{}\"", Self::escape_sparql_literal(s)),
            RdfValue::TypedLiteral(s, datatype) => {
                format!("\"{}\"^^<{}>", Self::escape_sparql_literal(s), datatype)
            }
        }
    }

    pub fn quads_to_sparql_insert(quads: &[RdfQuad], graph: &str) -> String {
        let mut triples = Vec::new();
        for quad in quads {
            let s = format!("<{}>", quad.subject);
            let p = format!("<{}>", quad.predicate);
            let o = Self::format_rdf_value(&quad.object);
            triples.push(format!("{} {} {} .", s, p, o));
        }
        let body = triples.join("\n  ");
        format!("INSERT DATA {{ GRAPH <{}> {{\n  {}\n}} }}", graph, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_node(id: &str, node_type: &str, label: &str) -> NodeDef {
        NodeDef {
            id: id.to_string(),
            node_type: node_type.to_string(),
            label: label.to_string(),
            description: None,
            properties: HashMap::new(),
        }
    }

    fn make_edge(source: &str, target: &str, relation: &str) -> EdgeDef {
        EdgeDef {
            source: source.to_string(),
            target: target.to_string(),
            relation: relation.to_string(),
            properties: HashMap::new(),
        }
    }

    #[test]
    fn test_map_node() {
        let mut props = HashMap::new();
        props.insert("age".to_string(), serde_json::json!(30));
        props.insert("active".to_string(), serde_json::json!(true));

        let node = NodeDef {
            id: "person_1".to_string(),
            node_type: "Person".to_string(),
            label: "Alice".to_string(),
            description: None,
            properties: props,
        };

        let quads = RdfMapper::map_nodes(&[node], "test_graph");
        assert_eq!(quads.len(), 4);

        assert_eq!(quads[0].subject, "iri://entity/person_1");
        assert_eq!(
            quads[0].predicate,
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );
        assert_eq!(quads[0].object, RdfValue::Iri("Person".to_string()));
        assert_eq!(quads[0].graph, Some("test_graph".to_string()));

        assert_eq!(quads[1].subject, "iri://entity/person_1");
        assert_eq!(
            quads[1].predicate,
            "http://www.w3.org/2000/01/rdf-schema#label"
        );
        assert_eq!(quads[1].object, RdfValue::Literal("Alice".to_string()));

        let age_quad = quads
            .iter()
            .find(|q| q.predicate == "https://agent-os.org/ontology/meta/age")
            .unwrap();
        assert_eq!(
            age_quad.object,
            RdfValue::TypedLiteral(
                "30".to_string(),
                "http://www.w3.org/2001/XMLSchema#integer".to_string()
            )
        );

        let active_quad = quads
            .iter()
            .find(|q| q.predicate == "https://agent-os.org/ontology/meta/active")
            .unwrap();
        assert_eq!(
            active_quad.object,
            RdfValue::TypedLiteral(
                "true".to_string(),
                "http://www.w3.org/2001/XMLSchema#boolean".to_string()
            )
        );
    }

    #[test]
    fn test_map_node_space_in_id() {
        let node = make_node("my node", "Concept", "My Node");
        let quads = RdfMapper::map_nodes(&[node], "g");
        assert_eq!(quads[0].subject, "iri://entity/my_node");
    }

    #[test]
    fn test_sanitize_id_encodes_invalid_percent_sequences() {
        assert_eq!(
            RdfMapper::sanitize_id("chrono::Local::now().format(%H:%M:%S)"),
            "chrono::Local::now().format(%25H:%25M:%25S)"
        );
        assert_eq!(RdfMapper::sanitize_id("already%2Fencoded"), "already%2Fencoded");
        assert_eq!(RdfMapper::sanitize_id("literal%%7B"), "literal%25%7B");
        assert_eq!(RdfMapper::sanitize_id("raw#fragment?query"), "raw%23fragment%3Fquery");
    }

    #[test]
    fn test_map_edge() {
        let edge = make_edge("a", "b", "knows");
        let quads = RdfMapper::map_edges(&[edge], "g");

        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].subject, "iri://entity/a");
        assert_eq!(quads[0].predicate, "knows");
        assert_eq!(quads[0].object, RdfValue::Iri("iri://entity/b".to_string()));
        assert_eq!(quads[0].graph, Some("g".to_string()));
    }

    #[test]
    fn test_map_edge_properties() {
        let mut props = HashMap::new();
        props.insert("since".to_string(), serde_json::json!("2020"));
        props.insert("weight".to_string(), serde_json::json!(0.8));

        let edge = EdgeDef {
            source: "a".to_string(),
            target: "b".to_string(),
            relation: "knows".to_string(),
            properties: props,
        };

        let quads = RdfMapper::map_edge_properties(&[edge], "g");

        assert!(quads.len() >= 6);

        let stmt_iri = "iri://statement/stmt_a_0";

        let type_quad = quads
            .iter()
            .find(|q| q.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .unwrap();
        assert_eq!(type_quad.subject, stmt_iri);
        assert_eq!(
            type_quad.object,
            RdfValue::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement".to_string())
        );

        let subj_quad = quads
            .iter()
            .find(|q| q.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")
            .unwrap();
        assert_eq!(
            subj_quad.object,
            RdfValue::Iri("iri://entity/a".to_string())
        );

        let pred_quad = quads
            .iter()
            .find(|q| q.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate")
            .unwrap();
        assert_eq!(pred_quad.object, RdfValue::Iri("knows".to_string()));

        let obj_quad = quads
            .iter()
            .find(|q| q.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#object")
            .unwrap();
        assert_eq!(obj_quad.object, RdfValue::Iri("iri://entity/b".to_string()));

        let since_quad = quads.iter().find(|q| q.predicate == "since").unwrap();
        assert_eq!(since_quad.object, RdfValue::Literal("2020".to_string()));

        let weight_quad = quads.iter().find(|q| q.predicate == "weight").unwrap();
        assert_eq!(
            weight_quad.object,
            RdfValue::TypedLiteral(
                "0.8".to_string(),
                "http://www.w3.org/2001/XMLSchema#decimal".to_string()
            )
        );
    }

    #[test]
    fn test_map_edge_properties_empty() {
        let edge = make_edge("a", "b", "knows");
        let quads = RdfMapper::map_edge_properties(&[edge], "g");
        assert!(quads.is_empty());
    }

    #[test]
    fn test_map_extraction() {
        let node = make_node("x", "Item", "X");
        let edge = make_edge("x", "y", "related_to");

        let output = LLMExtractionOutput {
            nodes: vec![node],
            edges: vec![edge],
        };

        let result = RdfMapper::map_extraction(&output, "kg");

        assert_eq!(result.entity_count, 1);
        assert_eq!(result.relation_count, 1);

        assert!(result.quads.iter().any(|q| q.subject == "iri://entity/x"
            && q.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            && q.object == RdfValue::Iri("Item".to_string())));
        assert!(result
            .quads
            .iter()
            .any(|q| q.subject == "iri://entity/x" && q.predicate == "related_to"));
    }

    #[test]
    fn test_quads_to_sparql_insert() {
        let quads = vec![
            RdfQuad {
                subject: "iri://entity/a".to_string(),
                predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                object: RdfValue::Iri("Person".to_string()),
                graph: Some("g".to_string()),
            },
            RdfQuad {
                subject: "iri://entity/a".to_string(),
                predicate: "http://www.w3.org/2000/01/rdf-schema#label".to_string(),
                object: RdfValue::Literal("Alice".to_string()),
                graph: Some("g".to_string()),
            },
        ];

        let sparql = RdfMapper::quads_to_sparql_insert(&quads, "my_graph");

        assert!(sparql.starts_with("INSERT DATA { GRAPH <my_graph> {"));
        assert!(sparql.contains(
            "<iri://entity/a> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <Person> ."
        ));
        assert!(sparql.contains(
            r#"<iri://entity/a> <http://www.w3.org/2000/01/rdf-schema#label> "Alice" ."#
        ));
        assert!(sparql.ends_with("} }"));
    }

    #[test]
    fn test_quads_to_sparql_insert_typed_literal() {
        let quads = vec![RdfQuad {
            subject: "iri://entity/a".to_string(),
            predicate: "age".to_string(),
            object: RdfValue::TypedLiteral(
                "30".to_string(),
                "http://www.w3.org/2001/XMLSchema#integer".to_string(),
            ),
            graph: Some("g".to_string()),
        }];

        let sparql = RdfMapper::quads_to_sparql_insert(&quads, "g");
        assert!(sparql.contains(r#""30"^^<http://www.w3.org/2001/XMLSchema#integer>"#));
    }

    #[test]
    fn test_escape_sparql_literal() {
        assert_eq!(RdfMapper::escape_sparql_literal("hello"), "hello");
        assert_eq!(
            RdfMapper::escape_sparql_literal(r#"say "hi""#),
            r#"say \"hi\""#
        );
        assert_eq!(RdfMapper::escape_sparql_literal("a\nb"), "a\\nb");
        assert_eq!(RdfMapper::escape_sparql_literal("a\\b"), "a\\\\b");
    }

    #[test]
    fn test_json_value_to_rdf_null() {
        assert_eq!(RdfMapper::json_value_to_rdf(&serde_json::Value::Null), None);
    }

    #[test]
    fn test_json_value_to_rdf_string() {
        assert_eq!(
            RdfMapper::json_value_to_rdf(&serde_json::json!("hello")),
            Some(RdfValue::Literal("hello".to_string()))
        );
    }

    #[test]
    fn test_json_value_to_rdf_integer() {
        assert_eq!(
            RdfMapper::json_value_to_rdf(&serde_json::json!(42)),
            Some(RdfValue::TypedLiteral(
                "42".to_string(),
                "http://www.w3.org/2001/XMLSchema#integer".to_string()
            ))
        );
    }

    #[test]
    fn test_json_value_to_rdf_float() {
        assert_eq!(
            RdfMapper::json_value_to_rdf(&serde_json::json!(2.5)),
            Some(RdfValue::TypedLiteral(
                "2.5".to_string(),
                "http://www.w3.org/2001/XMLSchema#decimal".to_string()
            ))
        );
    }

    #[test]
    fn test_map_to_rdf_instance_method() {
        let mapper = RdfMapper::new();
        let output = LLMExtractionOutput {
            nodes: vec![make_node("n1", "T", "N")],
            edges: vec![],
        };
        let result = mapper.map_to_rdf(&output);
        assert_eq!(result.entity_count, 1);
        assert_eq!(result.relation_count, 0);
        assert_eq!(result.quads[0].graph, Some("default".to_string()));
    }
}
