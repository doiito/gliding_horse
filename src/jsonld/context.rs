//! JSON-LD Context Definition
//!
//! Provides a global unified @context definition, including system namespaces and field mappings
//!
//! # Design Notes
//!
//! ## Single @context Source
//!
//! context.json is the authoritative @context definition source, embedded at compile time.
//! `context_value()` provides lazy `Arc<Value>` access.
//! No separate context HashMap is maintained beyond context.json.
//!
//! ## map_field_to_iri() = Lightweight Expansion
//!
//! This is a deliberately simplified JSON-LD key expansion — reads field→IRI mappings from context_value(),
//! O(1) lookup instead of the O(n) recursive tree traversal of the full JSON-LD 1.1 Expansion algorithm.
//!
//! ## No Remote @context Loading
//!
//! In a closed system, @context is known at compile time, avoiding remote loading latency and failure modes.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

pub const NS_AGENT: &str = "agent";
pub const NS_TASK: &str = "task";
pub const NS_SKILL: &str = "skill";
pub const NS_MEM: &str = "mem";
pub const NS_SEC: &str = "sec";
pub const NS_MON: &str = "mon";
pub const NS_TMPL: &str = "tmpl";
pub const NS_EXP: &str = "exp";
pub const NS_ADV: &str = "adv";
pub const NS_NODE: &str = "node";
pub const NS_CORE: &str = "core";

pub const URI_CORE: &str = "https://agent-os.org/ontology/core/";
pub const URI_AGENT: &str = "https://agent-os.org/ontology/agent#";
pub const URI_TASK: &str = "https://agent-os.org/ontology/task#";
pub const URI_SKILL: &str = "https://agent-os.org/ontology/skill#";
pub const URI_MEM: &str = "https://agent-os.org/ontology/memory#";
pub const URI_SEC: &str = "https://agent-os.org/ontology/security#";
pub const URI_MON: &str = "https://agent-os.org/ontology/monitoring#";
pub const URI_TMPL: &str = "https://agent-os.org/ontology/template#";
pub const URI_EXP: &str = "https://agent-os.org/ontology/experience#";
pub const URI_ADV: &str = "https://agent-os.org/ontology/advisory#";
pub const URI_NODE: &str = "https://agent-os.org/ontology/node#";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonLdContext {
    pub prefix: String,
    pub uri: String,
    pub mappings: HashMap<String, String>,
}

static UNIFIED_CONTEXT: OnceLock<Arc<Value>> = OnceLock::new();

impl JsonLdContext {
    pub fn new(prefix: String, uri: String) -> Self {
        Self {
            prefix,
            uri,
            mappings: HashMap::new(),
        }
    }

    pub fn with_mappings(prefix: String, uri: String, mappings: HashMap<String, String>) -> Self {
        Self {
            prefix,
            uri,
            mappings,
        }
    }

    pub fn to_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut result = HashMap::new();
        result.insert(
            self.prefix.clone(),
            serde_json::Value::String(self.uri.clone()),
        );

        for (key, value) in &self.mappings {
            result.insert(key.clone(), serde_json::Value::String(value.clone()));
        }

        result
    }

    pub fn add_mapping(&mut self, field: String, iri: String) {
        self.mappings.insert(field, iri);
    }

    /// Build the unified @context object
    ///
    /// Loads base definitions (type mappings + field mappings + domain namespaces) from context.json,
    /// then appends programmatically injected namespace prefixes as a safety fallback.
    ///
    /// The result is cached by OnceLock, providing O(1) access thereafter.
    fn build_unified_context() -> Arc<Value> {
        let raw: Value = serde_json::from_str(include_str!("context.json"))
            .expect("failed to parse context.json");

        let mut context_map = raw
            .get("@context")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        // context.json already contains these namespace definitions (as base fields),
        // re-injecting them via named constants ensures they aren't broken by accidentally deleted context.json fields.
        // Values are identical (guaranteed by const URI_* constants), duplicate insertion is a no-op.
        for (prefix, uri) in [
            ("core", URI_CORE),
            ("agent", URI_AGENT),
            ("task", URI_TASK),
            ("skill", URI_SKILL),
            ("mem", URI_MEM),
            ("sec", URI_SEC),
            ("mon", URI_MON),
            ("tmpl", URI_TMPL),
            ("exp", URI_EXP),
            ("adv", URI_ADV),
            ("node", URI_NODE),
        ] {
            context_map
                .entry(prefix.to_string())
                .or_insert(Value::String(uri.to_string()));
        }

        Arc::new(Value::Object(context_map))
    }

    pub fn context_value() -> Arc<Value> {
        UNIFIED_CONTEXT
            .get_or_init(Self::build_unified_context)
            .clone()
    }

    /// Inject @context into a JSON value (if not already present)
    pub fn inject(value: &mut Value) {
        if let Some(obj) = value.as_object_mut() {
            if !obj.contains_key("@context") {
                obj.insert("@context".to_string(), (*Self::context_value()).clone());
            }
        }
    }
}

/// Map a field name to its full IRI
///
/// # Design Intent: Lightweight Expansion
///
/// This is NOT the JSON-LD 1.1 standard Expansion algorithm (recursive tree traversal + value expansion),
/// but rather a flat lookup of field→IRI mappings from context.json.
/// For a closed system (@context known at compile time) this provides equivalent semantics,
/// at a cost reduced from O(n) recursion to O(1) lookup.
///
/// Fields not defined in context.json fall back to `node:{field}`.
pub fn map_field_to_iri(field: &str) -> String {
    if field == "@id" || field == "@type" {
        return field.to_string();
    }

    let ctx = JsonLdContext::context_value();
    ctx.get(field)
        .and_then(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Object(obj) => obj.get("@id").and_then(|id| id.as_str().map(String::from)),
            _ => None,
        })
        .unwrap_or_else(|| format!("node:{}", field))
}

/// Create a Skill-specific @context (appends skill_name / skill_version)
pub fn create_context_for_skill(
    _skill_name: &str,
    _skill_version: &str,
) -> HashMap<String, serde_json::Value> {
    let ctx = JsonLdContext::context_value();
    let mut context: HashMap<String, Value> = ctx
        .as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    let name_iri = map_field_to_iri("skill_name");
    let version_iri = map_field_to_iri("skill_version");

    context.insert("skill_name".to_string(), Value::String(name_iri));
    context.insert("skill_version".to_string(), Value::String(version_iri));

    context
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonld_context_new() {
        let ctx = JsonLdContext::new("task".to_string(), URI_TASK.to_string());
        assert_eq!(ctx.prefix, "task");
        assert_eq!(ctx.uri, URI_TASK);
        assert!(ctx.mappings.is_empty());
    }

    #[test]
    fn test_jsonld_context_to_dict() {
        let mut mappings = HashMap::new();
        mappings.insert("summary".to_string(), "task:summary".to_string());
        mappings.insert("status".to_string(), "task:status".to_string());

        let ctx = JsonLdContext::with_mappings("task".to_string(), URI_TASK.to_string(), mappings);

        let dict = ctx.to_dict();
        assert_eq!(
            dict.get("task"),
            Some(&serde_json::Value::String(URI_TASK.to_string()))
        );
        assert_eq!(
            dict.get("summary"),
            Some(&serde_json::Value::String("task:summary".to_string()))
        );
    }

    #[test]
    fn test_context_value_contains_namespaces() {
        let ctx = JsonLdContext::context_value();
        assert_eq!(ctx.get("core").and_then(Value::as_str), Some(URI_CORE));
        assert!(ctx.get("agent").is_some());
        assert!(ctx.get("task").is_some());
        assert!(ctx.get("skill").is_some());
    }

    #[test]
    fn test_map_field_to_iri() {
        assert_eq!(map_field_to_iri("summary"), "core:summary");
        assert_eq!(map_field_to_iri("status"), "core:status");
        assert_eq!(map_field_to_iri("confidence"), "core:confidence");
        assert_eq!(map_field_to_iri("created_at"), "core:createdAt");
        assert_eq!(map_field_to_iri("what"), "core:what");
        assert_eq!(map_field_to_iri("plan_iri"), "core:planIRI");
        assert_eq!(map_field_to_iri("@id"), "@id");
        assert_eq!(map_field_to_iri("unknown_field"), "node:unknown_field");
    }

    #[test]
    fn test_create_context_for_skill() {
        let ctx = create_context_for_skill("file_read", "1.0.0");
        assert!(ctx.contains_key("skill_name"));
        assert!(ctx.contains_key("skill_version"));
        assert!(ctx.contains_key("summary"));
        assert!(ctx.contains_key("status"));
    }

    #[test]
    fn test_context_uses_canonical_namespace() {
        let ctx = JsonLdContext::context_value();
        for prefix in [
            "core", "agent", "task", "skill", "mem", "sec", "mon", "tmpl", "exp", "adv", "node",
        ] {
            let uri = ctx
                .get(prefix)
                .and_then(Value::as_str)
                .expect("namespace must be a string");
            assert!(
                uri.starts_with("https://agent-os.org/ontology/"),
                "{prefix} namespace is not canonical: {uri}"
            );
        }
    }

    #[test]
    fn test_inject_context() {
        let mut node = serde_json::json!({
            "@id": "iri://task/123",
            "summary": "test"
        });
        JsonLdContext::inject(&mut node);
        assert!(node.get("@context").is_some());
        JsonLdContext::inject(&mut node);
        assert!(node.get("@context").is_some());
    }
}
