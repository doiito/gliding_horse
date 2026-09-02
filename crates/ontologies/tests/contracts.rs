use std::sync::Arc;

use ontologies::graph::GraphStore;
use ontologies::ontology::OntologyService;
use ontologies::reason::Reasoner;
use ontologies::shacl::ShaclValidator;

const PREFIXES: &str = r#"
    @prefix ex: <https://example.test/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
"#;

#[test]
fn syntax_validation_reports_valid_and_invalid_turtle() {
    let valid =
        OntologyService::validate_string(&format!("{PREFIXES} ex:Person a rdfs:Class .")).unwrap();
    let valid: serde_json::Value = serde_json::from_str(&valid).unwrap();
    assert_eq!(valid["valid"], true);
    assert_eq!(valid["triple_count"], 1);

    let invalid = OntologyService::validate_string("@prefix ex: <broken ex:a ex:b ex:c .")
        .expect("invalid RDF is represented by a validation report");
    let invalid: serde_json::Value = serde_json::from_str(&invalid).unwrap();
    assert_eq!(invalid["valid"], false);
    assert!(invalid["errors"].as_array().unwrap().len() == 1);
}

#[test]
fn diff_reports_added_and_removed_triples() {
    let old = format!("{PREFIXES} ex:Alice a ex:Person .");
    let new = format!("{PREFIXES} ex:Bob a ex:Person .");
    let report: serde_json::Value =
        serde_json::from_str(&OntologyService::diff(&old, &new).unwrap()).unwrap();
    assert_eq!(report["added"], 1);
    assert_eq!(report["removed"], 1);
}

#[test]
fn shacl_min_count_violation_and_conformance_are_distinguished() {
    let graph = Arc::new(GraphStore::new());
    graph
        .load_turtle(&format!("{PREFIXES} ex:Alice a ex:Person ."), None)
        .unwrap();
    let shapes = format!(
        r#"{PREFIXES}
        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [ sh:path ex:name ; sh:minCount 1 ] .
        "#
    );
    let report: serde_json::Value =
        serde_json::from_str(&ShaclValidator::validate(&graph, &shapes).unwrap()).unwrap();
    assert_eq!(report["conforms"], false);
    assert_eq!(report["violation_count"], 1);

    graph
        .load_turtle(&format!(r#"{PREFIXES} ex:Alice ex:name "Alice" ."#), None)
        .unwrap();
    let report: serde_json::Value =
        serde_json::from_str(&ShaclValidator::validate(&graph, &shapes).unwrap()).unwrap();
    assert_eq!(report["conforms"], true);
}

#[test]
fn rdfs_reasoner_materializes_subclass_membership() {
    let graph = Arc::new(GraphStore::new());
    graph
        .load_turtle(
            &format!("{PREFIXES} ex:Engineer rdfs:subClassOf ex:Person . ex:Alice a ex:Engineer ."),
            None,
        )
        .unwrap();
    let report: serde_json::Value =
        serde_json::from_str(&Reasoner::run(&graph, "rdfs", true).unwrap()).unwrap();
    assert!(report["inferred_count"].as_u64().unwrap() >= 1);
    let query: serde_json::Value = serde_json::from_str(
        &graph
            .sparql_select("SELECT ?who WHERE { ?who a <https://example.test/Person> }")
            .unwrap(),
    )
    .unwrap();
    assert_eq!(query["results"].as_array().unwrap().len(), 1);
}
