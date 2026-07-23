//! JSON-LD utility functions
//!
//! Provides IRI generation, parsing, validation, and other utility functions

use serde_json::Value;

pub fn generate_iri(namespace: &str, id: &str) -> String {
    format!("iri://{}/{}", namespace, id)
}

pub fn parse_iri(iri: &str) -> Option<(String, String)> {
    if !iri.starts_with("iri://") {
        return None;
    }

    let without_prefix = &iri[6..];
    let parts: Vec<&str> = without_prefix.splitn(2, '/').collect();

    if parts.len() == 2 {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

pub fn is_iri_reference(value: &str) -> bool {
    value.starts_with("iri://")
}

pub fn validate_jsonld_node(node: &Value) -> Result<(), String> {
    if !node.is_object() {
        return Err("Node must be a JSON object".to_string());
    }

    let obj = node.as_object().ok_or("Node must be a JSON object")?;

    if !obj.contains_key("@id") {
        return Err("Node must contain @id field".to_string());
    }

    let id = obj.get("@id").ok_or("Node must contain @id field")?;
    if !id.is_string() {
        return Err("@id must be a string".to_string());
    }

    if let Some(id_str) = id.as_str() {
        if !id_str.starts_with("iri://") {
            return Err("@id must be a valid IRI starting with iri://".to_string());
        }
    }

    if !obj.contains_key("@type") {
        return Err("Node must contain @type field".to_string());
    }

    let node_type = obj.get("@type").ok_or("Node must contain @type field")?;
    match node_type {
        Value::String(_) => Ok(()),
        Value::Array(arr) => {
            if arr.is_empty() {
                return Err("@type array must not be empty".to_string());
            }
            if !arr.iter().all(|v| v.is_string()) {
                return Err("@type array must contain only strings".to_string());
            }
            Ok(())
        }
        _ => Err("@type must be a string or array of strings".to_string()),
    }
}

pub fn extract_id_from_iri(iri: &str) -> Option<String> {
    parse_iri(iri).map(|(_, id)| id)
}

pub fn extract_namespace_from_iri(iri: &str) -> Option<String> {
    parse_iri(iri).map(|(namespace, _)| namespace)
}

/// All known IRI namespaces (should match format!("iri://{}/...", ns) in code)
pub const KNOWN_NAMESPACES: &[&str] = &[
    "agent",
    "approval",
    "archive",
    "chat",
    "checkpoint",
    "concept",
    "emphasis",
    "entity",
    "experience",
    "fragment",
    "item",
    "knowledge",
    "mcp",
    "mem",
    "memory",
    "moc",
    "node",
    "plan",
    "project",
    "rag",
    "schema",
    "sec",
    "session",
    "share",
    "skill",
    "skills",
    "skill-types",
    "stage",
    "statement",
    "stream",
    "system",
    "task",
    "template",
    "test",
    "tmpl",
    "exp",
    "adv",
    "tool-result",
    "whitelist",
];

pub fn is_valid_namespace(namespace: &str) -> bool {
    KNOWN_NAMESPACES.contains(&namespace)
}

pub fn normalize_iri(iri: &str) -> String {
    if iri.starts_with("iri://") {
        iri.to_string()
    } else {
        format!("iri://{}", iri)
    }
}

pub fn create_node_id(namespace: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_nanos();
    generate_iri(namespace, &format!("{}", timestamp))
}

pub fn merge_contexts(contexts: Vec<Value>) -> Value {
    if contexts.is_empty() {
        return Value::Object(serde_json::Map::new());
    }

    if contexts.len() == 1 {
        return contexts[0].clone();
    }

    let mut merged = serde_json::Map::new();

    for ctx in contexts {
        match ctx {
            Value::Object(map) => {
                for (key, value) in map {
                    merged.insert(key, value);
                }
            }
            Value::String(s) => {
                merged.insert("@base".to_string(), Value::String(s));
            }
            Value::Array(arr) => {
                for item in arr {
                    if let Value::Object(map) = item {
                        for (key, value) in map {
                            merged.insert(key, value);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Value::Object(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_generate_iri() {
        assert_eq!(generate_iri("task", "123"), "iri://task/123");
        assert_eq!(generate_iri("agent", "agent-001"), "iri://agent/agent-001");
    }

    #[test]
    fn test_parse_iri() {
        let result = parse_iri("iri://task/123");
        assert_eq!(result, Some(("task".to_string(), "123".to_string())));

        let result2 = parse_iri("iri://agent/agent-001");
        assert_eq!(
            result2,
            Some(("agent".to_string(), "agent-001".to_string()))
        );

        let result3 = parse_iri("invalid_iri");
        assert_eq!(result3, None);

        let result4 = parse_iri("http://example.com");
        assert_eq!(result4, None);
    }

    #[test]
    fn test_is_iri_reference() {
        assert!(is_iri_reference("iri://task/123"));
        assert!(is_iri_reference("iri://agent/agent-001"));
        assert!(!is_iri_reference("http://example.com"));
        assert!(!is_iri_reference("task/123"));
    }

    #[test]
    fn test_validate_jsonld_node_valid() {
        let node = json!({
            "@id": "iri://task/123",
            "@type": "TaskNode",
            "summary": "Test task"
        });
        assert!(validate_jsonld_node(&node).is_ok());
    }

    #[test]
    fn test_validate_jsonld_node_with_type_array() {
        let node = json!({
            "@id": "iri://task/123",
            "@type": ["TaskNode", "PlanNode"],
            "summary": "Test task"
        });
        assert!(validate_jsonld_node(&node).is_ok());
    }

    #[test]
    fn test_validate_jsonld_node_missing_id() {
        let node = json!({
            "@type": "TaskNode",
            "summary": "Test task"
        });
        assert!(validate_jsonld_node(&node).is_err());
    }

    #[test]
    fn test_validate_jsonld_node_missing_type() {
        let node = json!({
            "@id": "iri://task/123",
            "summary": "Test task"
        });
        assert!(validate_jsonld_node(&node).is_err());
    }

    #[test]
    fn test_validate_jsonld_node_invalid_id() {
        let node = json!({
            "@id": "invalid_id",
            "@type": "TaskNode",
            "summary": "Test task"
        });
        assert!(validate_jsonld_node(&node).is_err());
    }

    #[test]
    fn test_extract_id_from_iri() {
        assert_eq!(
            extract_id_from_iri("iri://task/123"),
            Some("123".to_string())
        );
        assert_eq!(
            extract_id_from_iri("iri://agent/agent-001"),
            Some("agent-001".to_string())
        );
        assert_eq!(extract_id_from_iri("invalid"), None);
    }

    #[test]
    fn test_extract_namespace_from_iri() {
        assert_eq!(
            extract_namespace_from_iri("iri://task/123"),
            Some("task".to_string())
        );
        assert_eq!(
            extract_namespace_from_iri("iri://agent/agent-001"),
            Some("agent".to_string())
        );
        assert_eq!(extract_namespace_from_iri("invalid"), None);
    }

    #[test]
    fn test_is_valid_namespace() {
        assert!(is_valid_namespace("task"));
        assert!(is_valid_namespace("agent"));
        assert!(is_valid_namespace("skill"));
        assert!(
            is_valid_namespace("entity"),
            "Extended namespace should be recognized"
        );
        assert!(
            is_valid_namespace("share"),
            "Extended namespace should be recognized"
        );
        assert!(
            is_valid_namespace("knowledge"),
            "Extended namespace should be recognized"
        );
        assert!(!is_valid_namespace("invalid_namespace"));
    }

    #[test]
    fn test_normalize_iri() {
        assert_eq!(normalize_iri("task/123"), "iri://task/123");
        assert_eq!(normalize_iri("iri://task/123"), "iri://task/123");
    }

    #[test]
    fn test_create_node_id() {
        let id = create_node_id("task");
        assert!(id.starts_with("iri://task/"));
        assert!(id.len() > "iri://task/".len());
    }

    #[test]
    fn test_merge_contexts_empty() {
        let result = merge_contexts(vec![]);
        assert!(result.is_object());
    }

    #[test]
    fn test_merge_contexts_single() {
        let ctx = json!({"task": "https://example.org/task#"});
        let result = merge_contexts(vec![ctx.clone()]);
        assert_eq!(result, ctx);
    }

    #[test]
    fn test_merge_contexts_multiple() {
        let ctx1 = json!({"task": "https://example.org/task#"});
        let ctx2 = json!({"agent": "https://example.org/agent#"});

        let result = merge_contexts(vec![ctx1, ctx2]);
        let obj = result.as_object().unwrap();

        assert!(obj.contains_key("task"));
        assert!(obj.contains_key("agent"));
    }
}
