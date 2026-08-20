//! JSON-LD Framing Implementation
//!
//! Provides on-demand projection and context trimming with token budget control

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EmbedDirective {
    #[serde(rename = "@always")]
    Always,

    #[serde(rename = "@link")]
    Link,

    #[serde(rename = "@never")]
    Never,
}

impl EmbedDirective {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbedDirective::Always => "@always",
            EmbedDirective::Link => "@link",
            EmbedDirective::Never => "@never",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "@always" => Some(EmbedDirective::Always),
            "@link" => Some(EmbedDirective::Link),
            "@never" => Some(EmbedDirective::Never),
            _ => None,
        }
    }
}

impl Default for EmbedDirective {
    fn default() -> Self {
        EmbedDirective::Always
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameTemplate {
    #[serde(rename = "@context")]
    pub context: Value,

    #[serde(skip)]
    pub embed_rules: HashMap<String, EmbedDirective>,

    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub include_properties: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
}

impl FrameTemplate {
    pub fn new(context: Value) -> Self {
        Self {
            context,
            embed_rules: HashMap::new(),
            include_properties: Vec::new(),
            max_depth: None,
        }
    }

    pub fn with_embed_rule(mut self, property: String, directive: EmbedDirective) -> Self {
        self.embed_rules.insert(property, directive);
        self
    }

    pub fn with_include_properties(mut self, properties: Vec<String>) -> Self {
        self.include_properties = properties;
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn get_embed_directive(&self, property: &str) -> &EmbedDirective {
        self.embed_rules
            .get(property)
            .unwrap_or(&EmbedDirective::Always)
    }
}

pub fn apply_frame(node: &Value, frame: &FrameTemplate) -> Value {
    if !node.is_object() {
        return node.clone();
    }

    let obj = match node.as_object() {
        Some(o) => o,
        None => return node.clone(),
    };

    let mut result = serde_json::Map::new();

    if let Some(id) = obj.get("@id") {
        result.insert("@id".to_string(), id.clone());
    }

    if let Some(node_type) = obj.get("@type") {
        result.insert("@type".to_string(), node_type.clone());
    }

    if !frame.context.is_null() {
        result.insert("@context".to_string(), frame.context.clone());
    }

    let properties = if frame.include_properties.is_empty() {
        obj.iter()
            .filter(|(k, _)| !k.starts_with('@'))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
    } else {
        frame
            .include_properties
            .iter()
            .filter_map(|prop| obj.get(prop).map(|v| (prop.clone(), v.clone())))
            .collect()
    };

    for (key, value) in properties {
        let directive = frame.get_embed_directive(&key);
        let processed_value =
            embed_node(&value, directive, frame.max_depth.unwrap_or(usize::MAX), 0);
        result.insert(key, processed_value);
    }

    Value::Object(result)
}

pub fn embed_node(
    node: &Value,
    directive: &EmbedDirective,
    max_depth: usize,
    current_depth: usize,
) -> Value {
    match directive {
        EmbedDirective::Never => {
            if let Some(obj) = node.as_object() {
                if let Some(id) = obj.get("@id") {
                    return json!({ "@id": id });
                }
            }
            return node.clone();
        }

        EmbedDirective::Link => {
            if let Some(obj) = node.as_object() {
                let has_nested_structure = obj.values().any(|v| {
                    v.is_object()
                        || (v.is_array()
                            && !v.as_array().expect("checked is_array above").is_empty())
                });

                if !has_nested_structure {
                    if let Some(id) = obj.get("@id") {
                        return json!({ "@id": id });
                    }
                }

                let mut result = serde_json::Map::new();

                for (key, value) in obj {
                    if key == "@id" || key == "@type" || key == "@context" {
                        result.insert(key.clone(), value.clone());
                    } else if let Value::Object(nested) = value {
                        if let Some(nested_id) = nested.get("@id") {
                            result.insert(key.clone(), json!({ "@id": nested_id }));
                        } else {
                            result.insert(key.clone(), Value::Object(nested.clone()));
                        }
                    } else if let Value::Array(arr) = value {
                        let linked_arr: Vec<Value> = arr
                            .iter()
                            .filter_map(|item| {
                                if let Some(item_obj) = item.as_object() {
                                    item_obj.get("@id").map(|id| json!({ "@id": id }))
                                } else {
                                    Some(item.clone())
                                }
                            })
                            .collect();
                        result.insert(key.clone(), Value::Array(linked_arr));
                    } else {
                        result.insert(key.clone(), value.clone());
                    }
                }

                Value::Object(result)
            } else if let Some(arr) = node.as_array() {
                let linked: Vec<Value> = arr
                    .iter()
                    .filter_map(|item| {
                        if let Some(obj) = item.as_object() {
                            obj.get("@id").map(|id| json!({ "@id": id }))
                        } else {
                            Some(item.clone())
                        }
                    })
                    .collect();
                Value::Array(linked)
            } else {
                node.clone()
            }
        }

        EmbedDirective::Always => {
            if current_depth >= max_depth {
                if let Some(obj) = node.as_object() {
                    if let Some(id) = obj.get("@id") {
                        return json!({ "@id": id });
                    }
                }
                return node.clone();
            }

            if let Some(obj) = node.as_object() {
                let mut expanded = serde_json::Map::new();

                for (key, value) in obj {
                    if key == "@id" || key == "@type" || key == "@context" {
                        expanded.insert(key.clone(), value.clone());
                    } else if let Value::Object(nested) = value {
                        let nested_expanded = embed_node(
                            &Value::Object(nested.clone()),
                            &EmbedDirective::Always,
                            max_depth,
                            current_depth + 1,
                        );
                        expanded.insert(key.clone(), nested_expanded);
                    } else if let Value::Array(arr) = value {
                        let expanded_arr: Vec<Value> = arr
                            .iter()
                            .map(|item| {
                                embed_node(
                                    item,
                                    &EmbedDirective::Always,
                                    max_depth,
                                    current_depth + 1,
                                )
                            })
                            .collect();
                        expanded.insert(key.clone(), Value::Array(expanded_arr));
                    } else {
                        expanded.insert(key.clone(), value.clone());
                    }
                }

                Value::Object(expanded)
            } else if let Some(arr) = node.as_array() {
                let expanded_arr: Vec<Value> = arr
                    .iter()
                    .map(|item| embed_node(item, &EmbedDirective::Always, max_depth, current_depth))
                    .collect();
                Value::Array(expanded_arr)
            } else {
                node.clone()
            }
        }
    }
}

pub fn filter_properties(node: &Value, props: &[String]) -> Value {
    if !node.is_object() {
        return node.clone();
    }

    let obj = match node.as_object() {
        Some(o) => o,
        None => return node.clone(),
    };

    let mut result = serde_json::Map::new();

    for key in ["@id", "@type", "@context"] {
        if let Some(value) = obj.get(key) {
            result.insert(key.to_string(), value.clone());
        }
    }

    for prop in props {
        if let Some(value) = obj.get(prop) {
            result.insert(prop.clone(), value.clone());
        }
    }

    Value::Object(result)
}

pub fn estimate_tokens(node: &Value) -> usize {
    match node {
        Value::Null => 1,
        Value::Bool(_) => 1,
        Value::Number(n) => n.to_string().len(),
        Value::String(s) => {
            let char_count = s.chars().count();
            char_count / 4 + 1
        }
        Value::Array(arr) => {
            let mut total = 2;
            for item in arr {
                total += estimate_tokens(item) + 1;
            }
            total
        }
        Value::Object(obj) => {
            let mut total = 2;
            for (key, value) in obj {
                total += key.chars().count() / 4 + 1;
                total += estimate_tokens(value) + 1;
            }
            total
        }
    }
}

pub fn fit_to_budget(node: &Value, budget: usize, frame: &FrameTemplate) -> Value {
    let estimated = estimate_tokens(node);

    if estimated <= budget {
        return apply_frame(node, frame);
    }

    let mut adjusted_frame = frame.clone();

    if adjusted_frame.max_depth.is_none()
        || adjusted_frame
            .max_depth
            .expect("Some when is_none is false")
            > 2
    {
        adjusted_frame.max_depth = Some(2);
    }

    let result = apply_frame(node, &adjusted_frame);
    let new_estimated = estimate_tokens(&result);

    if new_estimated <= budget {
        return result;
    }

    adjusted_frame.max_depth = Some(1);
    let result = apply_frame(node, &adjusted_frame);
    let new_estimated = estimate_tokens(&result);

    if new_estimated <= budget {
        return result;
    }

    for (key, _) in frame.embed_rules.clone() {
        adjusted_frame.embed_rules.insert(key, EmbedDirective::Link);
    }

    let result = apply_frame(node, &adjusted_frame);
    let new_estimated = estimate_tokens(&result);

    if new_estimated <= budget {
        return result;
    }

    filter_to_summary(node)
}

fn filter_to_summary(node: &Value) -> Value {
    if !node.is_object() {
        return node.clone();
    }

    let obj = match node.as_object() {
        Some(o) => o,
        None => return node.clone(),
    };

    let mut result = serde_json::Map::new();

    if let Some(id) = obj.get("@id") {
        result.insert("@id".to_string(), id.clone());
    }

    if let Some(node_type) = obj.get("@type") {
        result.insert("@type".to_string(), node_type.clone());
    }

    if let Some(summary) = obj.get("summary") {
        result.insert("summary".to_string(), summary.clone());
    } else if let Some(desc) = obj.get("description") {
        result.insert("summary".to_string(), desc.clone());
    } else if let Some(content) = obj.get("content") {
        let content_str = content.as_str().unwrap_or("");
        let summary = if content_str.len() > 200 {
            let mut end = 200;
            while end > 0 && !content_str.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}...", &content_str[..end])
        } else {
            content_str.to_string()
        };
        result.insert("summary".to_string(), Value::String(summary));
    }

    Value::Object(result)
}

pub static PLAN_CONTEXT_FRAME: Lazy<FrameTemplate> = Lazy::new(|| {
    FrameTemplate::new(json!({
        "exec": "https://agent-os.org/ontology/exec#",
        "task": "https://agent-os.org/ontology/task#"
    }))
    .with_embed_rule("task:subTasks".to_string(), EmbedDirective::Always)
    .with_embed_rule("exec:assignedTo".to_string(), EmbedDirective::Link)
    .with_embed_rule("task:relatedHistory".to_string(), EmbedDirective::Link)
    .with_max_depth(3)
});

pub static DA_INPUT_FRAME: Lazy<FrameTemplate> = Lazy::new(|| {
    FrameTemplate::new(json!({
        "exec": "https://agent-os.org/ontology/exec#",
        "task": "https://agent-os.org/ontology/task#"
    }))
    .with_embed_rule("task:inputData".to_string(), EmbedDirective::Always)
    .with_embed_rule("task:resources".to_string(), EmbedDirective::Link)
    .with_embed_rule("task:dependencies".to_string(), EmbedDirective::Link)
    .with_max_depth(4)
});

pub static CA_REVIEW_FRAME: Lazy<FrameTemplate> = Lazy::new(|| {
    FrameTemplate::new(json!({
        "exec": "https://agent-os.org/ontology/exec#",
        "task": "https://agent-os.org/ontology/task#"
    }))
    .with_embed_rule("exec:results".to_string(), EmbedDirective::Always)
    .with_embed_rule("exec:validationRules".to_string(), EmbedDirective::Always)
    .with_embed_rule("exec:previousResults".to_string(), EmbedDirective::Link)
    .with_max_depth(3)
});

pub static AA_DECISION_FRAME: Lazy<FrameTemplate> = Lazy::new(|| {
    FrameTemplate::new(json!({
        "exec": "https://agent-os.org/ontology/exec#",
        "task": "https://agent-os.org/ontology/task#"
    }))
    .with_embed_rule("exec:reviewResults".to_string(), EmbedDirective::Always)
    .with_embed_rule("exec:alternatives".to_string(), EmbedDirective::Link)
    .with_embed_rule("exec:historicalDecisions".to_string(), EmbedDirective::Link)
    .with_max_depth(2)
});

pub static SUMMARY_ONLY_FRAME: Lazy<FrameTemplate> = Lazy::new(|| {
    FrameTemplate::new(json!({
        "agent": "https://agent-os.org/ontology/agent#"
    }))
    .with_embed_rule("agent:summary".to_string(), EmbedDirective::Always)
    .with_embed_rule("agent:pinnedIRIs".to_string(), EmbedDirective::Link)
    .with_include_properties(vec!["summary".to_string(), "status".to_string()])
    .with_max_depth(1)
});

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_embed_directive_as_str() {
        assert_eq!(EmbedDirective::Always.as_str(), "@always");
        assert_eq!(EmbedDirective::Link.as_str(), "@link");
        assert_eq!(EmbedDirective::Never.as_str(), "@never");
    }

    #[test]
    fn test_embed_directive_from_str() {
        assert_eq!(
            EmbedDirective::from_str("@always"),
            Some(EmbedDirective::Always)
        );
        assert_eq!(
            EmbedDirective::from_str("@link"),
            Some(EmbedDirective::Link)
        );
        assert_eq!(
            EmbedDirective::from_str("@never"),
            Some(EmbedDirective::Never)
        );
        assert_eq!(EmbedDirective::from_str("invalid"), None);
    }

    #[test]
    fn test_frame_template_new() {
        let frame = FrameTemplate::new(json!({"task": "https://example.org/task#"}));
        assert!(frame.embed_rules.is_empty());
        assert!(frame.include_properties.is_empty());
        assert!(frame.max_depth.is_none());
    }

    #[test]
    fn test_frame_template_with_methods() {
        let frame = FrameTemplate::new(json!({}))
            .with_embed_rule("subTasks".to_string(), EmbedDirective::Always)
            .with_include_properties(vec!["summary".to_string()])
            .with_max_depth(3);

        assert_eq!(frame.embed_rules.len(), 1);
        assert_eq!(frame.include_properties.len(), 1);
        assert_eq!(frame.max_depth, Some(3));
    }

    #[test]
    fn test_apply_frame_basic() {
        let node = json!({
            "@id": "iri://task/123",
            "@type": "TaskNode",
            "summary": "Test task",
            "description": "A longer description",
            "status": "running"
        });

        let frame =
            FrameTemplate::new(json!({})).with_include_properties(vec!["summary".to_string()]);

        let result = apply_frame(&node, &frame);

        assert_eq!(result["@id"], "iri://task/123");
        assert_eq!(result["@type"], "TaskNode");
        assert_eq!(result["summary"], "Test task");
        assert!(!result.as_object().unwrap().contains_key("description"));
    }

    #[test]
    fn test_embed_node_always() {
        let node = json!({
            "@id": "iri://task/123",
            "summary": "Test",
            "nested": {
                "@id": "iri://task/456",
                "value": "nested value"
            }
        });

        let result = embed_node(&node, &EmbedDirective::Always, 10, 0);

        assert!(result.is_object());
        let obj = result.as_object().unwrap();
        assert!(obj.contains_key("nested"));
        assert!(obj.get("nested").unwrap().is_object());
    }

    #[test]
    fn test_embed_node_link() {
        let node = json!({
            "@id": "iri://task/123",
            "summary": "Test",
            "nested": {
                "@id": "iri://task/456",
                "value": "nested value"
            }
        });

        let result = embed_node(&node, &EmbedDirective::Link, 10, 0);

        assert!(result.is_object());
        let obj = result.as_object().unwrap();
        assert!(obj.contains_key("nested"));
        let nested = obj.get("nested").unwrap();
        assert!(nested.is_object());
        assert_eq!(nested["@id"], "iri://task/456");
        assert!(!nested.as_object().unwrap().contains_key("value"));
    }

    #[test]
    fn test_embed_node_never() {
        let node = json!({
            "@id": "iri://task/123",
            "summary": "Test",
            "nested": {
                "@id": "iri://task/456",
                "value": "nested value"
            }
        });

        let result = embed_node(&node, &EmbedDirective::Never, 10, 0);

        assert!(result.is_object());
        let obj = result.as_object().unwrap();
        assert_eq!(obj.get("@id"), Some(&json!("iri://task/123")));
        assert!(!obj.contains_key("summary"));
        assert!(!obj.contains_key("nested"));
    }

    #[test]
    fn test_embed_node_max_depth() {
        let node = json!({
            "@id": "iri://task/123",
            "level1": {
                "@id": "iri://task/456",
                "level2": {
                    "@id": "iri://task/789",
                    "value": "deep"
                }
            }
        });

        let result = embed_node(&node, &EmbedDirective::Always, 2, 0);

        assert!(result.is_object());
        let obj = result.as_object().unwrap();
        let level1 = obj.get("level1").unwrap().as_object().unwrap();
        let level2 = level1.get("level2").unwrap().as_object().unwrap();

        assert_eq!(level2.get("@id"), Some(&json!("iri://task/789")));
        assert!(!level2.contains_key("value"));
    }

    #[test]
    fn test_filter_properties() {
        let node = json!({
            "@id": "iri://task/123",
            "@type": "TaskNode",
            "summary": "Test",
            "description": "Long desc",
            "status": "running",
            "priority": "high"
        });

        let result = filter_properties(&node, &["summary".to_string(), "status".to_string()]);

        assert!(result.is_object());
        let obj = result.as_object().unwrap();
        assert!(obj.contains_key("@id"));
        assert!(obj.contains_key("@type"));
        assert!(obj.contains_key("summary"));
        assert!(obj.contains_key("status"));
        assert!(!obj.contains_key("description"));
        assert!(!obj.contains_key("priority"));
    }

    #[test]
    fn test_estimate_tokens_string() {
        let value = json!("This is a test string");
        let tokens = estimate_tokens(&value);
        assert!(tokens > 0);
    }

    #[test]
    fn test_estimate_tokens_object() {
        let value = json!({
            "key1": "value1",
            "key2": "value2"
        });
        let tokens = estimate_tokens(&value);
        assert!(tokens > 0);
    }

    #[test]
    fn test_estimate_tokens_array() {
        let value = json!([1, 2, 3, "four"]);
        let tokens = estimate_tokens(&value);
        assert!(tokens > 0);
    }

    #[test]
    fn test_fit_to_budget_within_budget() {
        let node = json!({
            "@id": "iri://task/123",
            "summary": "Test"
        });

        let frame = FrameTemplate::new(json!({}));
        let result = fit_to_budget(&node, 100, &frame);

        assert_eq!(result["@id"], "iri://task/123");
    }

    #[test]
    fn test_fit_to_budget_exceeds_budget() {
        let node = json!({
            "@id": "iri://task/123",
            "summary": "Test summary",
            "description": "A very long description that should be truncated",
            "nested": {
                "@id": "iri://task/456",
                "value": "nested value",
                "deep": {
                    "@id": "iri://task/789",
                    "value": "deep nested"
                }
            }
        });

        let frame = FrameTemplate::new(json!({})).with_max_depth(5);

        let result = fit_to_budget(&node, 10, &frame);

        assert!(result.is_object());
        let obj = result.as_object().unwrap();
        assert!(obj.contains_key("@id"));
    }

    #[test]
    fn test_predefined_frames() {
        assert!(PLAN_CONTEXT_FRAME.max_depth.is_some());
        assert!(DA_INPUT_FRAME.max_depth.is_some());
        assert!(CA_REVIEW_FRAME.max_depth.is_some());
        assert!(AA_DECISION_FRAME.max_depth.is_some());
        assert!(SUMMARY_ONLY_FRAME.max_depth.is_some());

        assert!(!PLAN_CONTEXT_FRAME.embed_rules.is_empty());
        assert!(!DA_INPUT_FRAME.embed_rules.is_empty());
        assert!(!CA_REVIEW_FRAME.embed_rules.is_empty());
        assert!(!AA_DECISION_FRAME.embed_rules.is_empty());
        assert!(!SUMMARY_ONLY_FRAME.include_properties.is_empty());
    }

    #[test]
    fn test_apply_frame_with_embed_rules() {
        let node = json!({
            "@id": "iri://task/123",
            "@type": "TaskNode",
            "subTasks": [
                {
                    "@id": "iri://task/456",
                    "summary": "Subtask 1",
                    "details": "Details here"
                }
            ],
            "assignedTo": {
                "@id": "iri://agent/001",
                "name": "Agent 1",
                "role": "Plan"
            }
        });

        let frame = FrameTemplate::new(json!({}))
            .with_embed_rule("subTasks".to_string(), EmbedDirective::Always)
            .with_embed_rule("assignedTo".to_string(), EmbedDirective::Link);

        let result = apply_frame(&node, &frame);

        let sub_tasks = result.get("subTasks").unwrap().as_array().unwrap();
        let first_subtask = sub_tasks[0].as_object().unwrap();
        assert!(first_subtask.contains_key("summary"));

        let assigned = result.get("assignedTo").unwrap().as_object().unwrap();
        assert_eq!(assigned.get("@id"), Some(&json!("iri://agent/001")));
        assert!(!assigned.contains_key("name"));
    }

    #[test]
    fn test_embed_node_array() {
        let node = json!({
            "@id": "iri://task/123",
            "items": [
                {
                    "@id": "iri://item/1",
                    "name": "Item 1"
                },
                {
                    "@id": "iri://item/2",
                    "name": "Item 2"
                }
            ]
        });

        let result = embed_node(&node, &EmbedDirective::Link, 10, 0);

        let items = result.get("items").unwrap().as_array().unwrap();
        assert_eq!(items.len(), 2);

        for item in items {
            assert!(item.is_object());
            assert!(item.as_object().unwrap().contains_key("@id"));
            assert!(!item.as_object().unwrap().contains_key("name"));
        }
    }

    #[test]
    fn test_filter_to_summary() {
        let node = json!({
            "@id": "iri://task/123",
            "@type": "TaskNode",
            "content": "This is a long content that should be summarized because it exceeds the limit"
        });

        let result = filter_to_summary(&node);

        assert!(result.is_object());
        let obj = result.as_object().unwrap();
        assert!(obj.contains_key("@id"));
        assert!(obj.contains_key("@type"));
        assert!(obj.contains_key("summary"));
    }
}
