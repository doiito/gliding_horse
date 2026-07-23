//! JSON-LD type definitions
//!
//! Defines JSON-LD keywords, node structures, and related methods

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JsonLdKeyword {
    #[serde(rename = "@context")]
    Context,

    #[serde(rename = "@id")]
    Id,

    #[serde(rename = "@type")]
    Type,

    #[serde(rename = "@graph")]
    Graph,

    #[serde(rename = "@value")]
    Value,

    #[serde(rename = "@list")]
    List,

    #[serde(rename = "@set")]
    Set,

    #[serde(rename = "@reverse")]
    Reverse,

    #[serde(rename = "@container")]
    Container,

    #[serde(rename = "@language")]
    Language,
}

impl JsonLdKeyword {
    pub fn as_str(&self) -> &'static str {
        match self {
            JsonLdKeyword::Context => "@context",
            JsonLdKeyword::Id => "@id",
            JsonLdKeyword::Type => "@type",
            JsonLdKeyword::Graph => "@graph",
            JsonLdKeyword::Value => "@value",
            JsonLdKeyword::List => "@list",
            JsonLdKeyword::Set => "@set",
            JsonLdKeyword::Reverse => "@reverse",
            JsonLdKeyword::Container => "@container",
            JsonLdKeyword::Language => "@language",
        }
    }
}

impl std::fmt::Display for JsonLdKeyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonLdNode {
    #[serde(rename = "@context")]
    pub context: Value,

    #[serde(rename = "@id")]
    pub id: String,

    #[serde(rename = "@type")]
    pub node_type: Value,

    #[serde(flatten)]
    pub properties: HashMap<String, Value>,
}

impl JsonLdNode {
    pub fn new(id: String, node_type: impl Into<Value>) -> Self {
        Self {
            context: Value::String("https://pdca-agent.org/context/task".to_string()),
            id,
            node_type: node_type.into(),
            properties: HashMap::new(),
        }
    }

    pub fn with_context(mut self, context: Value) -> Self {
        self.context = context;
        self
    }

    pub fn with_property(mut self, key: String, value: Value) -> Self {
        self.properties.insert(key, value);
        self
    }

    pub fn with_properties(mut self, properties: HashMap<String, Value>) -> Self {
        self.properties = properties;
        self
    }

    pub fn to_json(&self) -> Result<Value, serde_json::Error> {
        serde_json::to_value(self)
    }

    pub fn from_json(json: &Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(json.clone())
    }

    pub fn get_property(&self, key: &str) -> Option<&Value> {
        self.properties.get(key)
    }

    pub fn set_property(&mut self, key: String, value: Value) {
        self.properties.insert(key, value);
    }
}

pub fn extract_iri(node: &JsonLdNode) -> &str {
    &node.id
}

pub fn extract_type(node: &JsonLdNode) -> Option<String> {
    match &node.node_type {
        Value::String(s) => Some(s.clone()),
        Value::Array(arr) if !arr.is_empty() => {
            arr.first().and_then(|v| v.as_str().map(|s| s.to_string()))
        }
        _ => None,
    }
}

pub fn extract_types(node: &JsonLdNode) -> Vec<String> {
    match &node.node_type {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => vec![],
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRIReference {
    #[serde(rename = "@id")]
    pub iri: String,
}

impl IRIReference {
    pub fn new(iri: String) -> Self {
        Self { iri }
    }

    pub fn from_string(iri: &str) -> Self {
        if iri.starts_with("iri://") {
            Self {
                iri: iri.to_string(),
            }
        } else {
            Self {
                iri: format!("iri://{}", iri),
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryNode {
    #[serde(rename = "@context")]
    pub context: String,

    #[serde(rename = "@id")]
    pub iri: String,

    #[serde(rename = "@type")]
    pub node_type: String,

    pub summary: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

impl SummaryNode {
    pub fn new(iri: String, node_type: String, summary: String) -> Self {
        Self {
            context: "https://pdca-agent.org/context/task".to_string(),
            iri,
            node_type,
            summary,
            status: None,
            confidence: None,
        }
    }

    pub fn with_status(mut self, status: String) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn to_json(&self) -> Result<Value, serde_json::Error> {
        serde_json::to_value(self)
    }

    pub fn from_json(json: &Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(json.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullNode {
    #[serde(rename = "@context")]
    pub context: String,

    #[serde(rename = "@id")]
    pub iri: String,

    #[serde(rename = "@type")]
    pub node_type: Value,

    #[serde(flatten)]
    pub properties: HashMap<String, Value>,
}

impl FullNode {
    pub fn new(iri: String, node_type: impl Into<Value>) -> Self {
        Self {
            context: "https://pdca-agent.org/context/task".to_string(),
            iri,
            node_type: node_type.into(),
            properties: HashMap::new(),
        }
    }

    pub fn with_property(mut self, key: String, value: Value) -> Self {
        self.properties.insert(key, value);
        self
    }

    pub fn to_json(&self) -> Result<Value, serde_json::Error> {
        serde_json::to_value(self)
    }

    pub fn from_json(json: &Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(json.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_jsonld_keyword_as_str() {
        assert_eq!(JsonLdKeyword::Context.as_str(), "@context");
        assert_eq!(JsonLdKeyword::Id.as_str(), "@id");
        assert_eq!(JsonLdKeyword::Type.as_str(), "@type");
    }

    #[test]
    fn test_jsonld_node_new() {
        let node = JsonLdNode::new("iri://task/123".to_string(), "TaskNode");
        assert_eq!(node.id, "iri://task/123");
        assert_eq!(node.node_type, Value::String("TaskNode".to_string()));
    }

    #[test]
    fn test_jsonld_node_with_property() {
        let node = JsonLdNode::new("iri://task/123".to_string(), "TaskNode")
            .with_property("summary".to_string(), json!("Test task"));

        assert_eq!(node.get_property("summary"), Some(&json!("Test task")));
    }

    #[test]
    fn test_jsonld_node_to_json() {
        let node = JsonLdNode::new("iri://task/123".to_string(), "TaskNode")
            .with_property("status".to_string(), json!("running"));

        let json = node.to_json().unwrap();
        assert_eq!(json["@id"], "iri://task/123");
        assert_eq!(json["@type"], "TaskNode");
        assert_eq!(json["status"], "running");
    }

    #[test]
    fn test_jsonld_node_from_json() {
        let json = json!({
            "@context": "https://pdca-agent.org/context/task",
            "@id": "iri://task/456",
            "@type": "TaskNode",
            "summary": "Test task"
        });

        let node = JsonLdNode::from_json(&json).unwrap();
        assert_eq!(node.id, "iri://task/456");
        assert_eq!(node.node_type, Value::String("TaskNode".to_string()));
    }

    #[test]
    fn test_extract_iri() {
        let node = JsonLdNode::new("iri://task/789".to_string(), "TaskNode");
        assert_eq!(extract_iri(&node), "iri://task/789");
    }

    #[test]
    fn test_extract_type() {
        let node = JsonLdNode::new("iri://task/123".to_string(), "TaskNode");
        assert_eq!(extract_type(&node), Some("TaskNode".to_string()));

        let node_multi = JsonLdNode::new(
            "iri://task/456".to_string(),
            json!(["TaskNode", "PlanNode"]),
        );
        assert_eq!(extract_type(&node_multi), Some("TaskNode".to_string()));
    }

    #[test]
    fn test_iri_reference() {
        let iri_ref = IRIReference::from_string("task/123");
        assert_eq!(iri_ref.iri, "iri://task/123");

        let iri_ref2 = IRIReference::from_string("iri://task/456");
        assert_eq!(iri_ref2.iri, "iri://task/456");
    }

    #[test]
    fn test_summary_node() {
        let summary = SummaryNode::new(
            "iri://task/123".to_string(),
            "TaskNode".to_string(),
            "Test summary".to_string(),
        )
        .with_status("running".to_string())
        .with_confidence(0.95);

        assert_eq!(summary.iri, "iri://task/123");
        assert_eq!(summary.node_type, "TaskNode");
        assert_eq!(summary.summary, "Test summary");
        assert_eq!(summary.status, Some("running".to_string()));
        assert_eq!(summary.confidence, Some(0.95));
    }

    #[test]
    fn test_full_node() {
        let full = FullNode::new("iri://task/123".to_string(), "TaskNode")
            .with_property("priority".to_string(), json!("high"))
            .with_property("assignee".to_string(), json!("agent-001"));

        assert_eq!(full.iri, "iri://task/123");
        assert_eq!(full.properties.get("priority"), Some(&json!("high")));
    }
}
