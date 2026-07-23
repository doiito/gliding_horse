use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyTerm {
    pub iri: String,
    pub label: String,
    pub description: String,
    pub term_type: OntologyTermType,
}

impl OntologyTerm {
    pub fn class(
        iri: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            iri: iri.into(),
            label: label.into(),
            description: description.into(),
            term_type: OntologyTermType::Class,
        }
    }
    pub fn relation(
        iri: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            iri: iri.into(),
            label: label.into(),
            description: description.into(),
            term_type: OntologyTermType::Relation,
        }
    }
    pub fn property(iri: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            iri: iri.into(),
            label: label.into(),
            description: String::new(),
            term_type: OntologyTermType::Property,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OntologyTermType {
    Class,
    Property,
    Relation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDef {
    pub id: String,
    pub node_type: String,
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub properties: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDef {
    pub source: String,
    pub target: String,
    pub relation: String,
    #[serde(default)]
    pub properties: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMExtractionOutput {
    pub nodes: Vec<NodeDef>,
    pub edges: Vec<EdgeDef>,
}

#[derive(Debug, Clone)]
pub struct RdfQuad {
    pub subject: String,
    pub predicate: String,
    pub object: RdfValue,
    pub graph: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RdfValue {
    Iri(String),
    Literal(String),
    TypedLiteral(String, String),
}

#[derive(Debug, Clone)]
pub struct RdfMappingResult {
    pub quads: Vec<RdfQuad>,
    pub entity_count: usize,
    pub relation_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRelation {
    pub entity_id: String,
    pub skill_iri: String,
    pub relation_type: BridgeRelationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeRelationType {
    HasSkill,
    ApplicableIn,
    RelatedTo,
}
