use serde_json::{json, Value};

use crate::jsonld::JsonLdContext;

/// Trait for types that can be serialized to/from JSON-LD format
pub trait JsonLdNode: Sized {
    fn to_json_ld(&self) -> Value;
    fn from_json_ld(value: &Value) -> Option<Self>;
    fn node_type() -> &'static str;
    fn node_id(&self) -> String;

    fn to_json_ld_doc(&self) -> Value {
        let mut doc = json!({
            "@id": self.node_id(),
            "@type": Self::node_type(),
            "data": self.to_json_ld(),
        });
        JsonLdContext::inject(&mut doc);
        doc
    }
}

/// Helper to extract @id from a JSON-LD document
pub fn extract_id(value: &Value) -> Option<String> {
    value.get("@id").and_then(|v| v.as_str()).map(String::from)
}

/// Helper to extract @type from a JSON-LD document
pub fn extract_type(value: &Value) -> Option<String> {
    value
        .get("@type")
        .and_then(|v| v.as_str())
        .map(String::from)
}
