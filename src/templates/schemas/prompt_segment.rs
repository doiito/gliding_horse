use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentType {
    Fixed,
    Variable,
    Dynamic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSegment {
    #[serde(rename = "segment_type")]
    pub segment_type: SegmentType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<String>,
}

impl PromptSegment {
    pub fn fixed(content: impl Into<String>) -> Self {
        Self {
            segment_type: SegmentType::Fixed,
            content: Some(content.into()),
            source: None,
            transform: None,
        }
    }

    pub fn variable(source: impl Into<String>) -> Self {
        Self {
            segment_type: SegmentType::Variable,
            content: None,
            source: Some(source.into()),
            transform: None,
        }
    }

    pub fn dynamic(source: impl Into<String>, transform: Option<String>) -> Self {
        Self {
            segment_type: SegmentType::Dynamic,
            content: None,
            source: Some(source.into()),
            transform,
        }
    }

    pub fn render(&self, context: &serde_json::Map<String, serde_json::Value>) -> String {
        match self.segment_type {
            SegmentType::Fixed => self.content.clone().unwrap_or_default(),

            SegmentType::Variable => {
                if let Some(source) = &self.source {
                    context
                        .get(source)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                } else {
                    String::new()
                }
            }

            SegmentType::Dynamic => {
                if let Some(source) = &self.source {
                    let value = context
                        .get(source)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);

                    if let Some(transform_fn) = &self.transform {
                        self.apply_transform(&value, transform_fn)
                    } else {
                        serde_json::to_string_pretty(&value).unwrap_or_default()
                    }
                } else {
                    String::new()
                }
            }
        }
    }

    fn apply_transform(&self, value: &serde_json::Value, transform: &str) -> String {
        match transform {
            "json_pretty" => serde_json::to_string_pretty(value).unwrap_or_default(),
            "json_compact" => serde_json::to_string(value).unwrap_or_default(),
            "join_comma" => {
                if let Some(arr) = value.as_array() {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                } else {
                    value.to_string()
                }
            }
            "join_newline" => {
                if let Some(arr) = value.as_array() {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    value.to_string()
                }
            }
            _ => value.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_fixed_segment() {
        let segment = PromptSegment::fixed("Hello, World!");
        assert_eq!(segment.segment_type, SegmentType::Fixed);
        assert_eq!(segment.content, Some("Hello, World!".to_string()));

        let ctx = serde_json::Map::new();
        assert_eq!(segment.render(&ctx), "Hello, World!");
    }

    #[test]
    fn test_variable_segment() {
        let segment = PromptSegment::variable("name");
        assert_eq!(segment.segment_type, SegmentType::Variable);
        assert_eq!(segment.source, Some("name".to_string()));

        let mut ctx = serde_json::Map::new();
        ctx.insert("name".to_string(), json!("Alice"));
        assert_eq!(segment.render(&ctx), "Alice");
    }

    #[test]
    fn test_dynamic_segment_with_transform() {
        let segment = PromptSegment::dynamic("items", Some("join_comma".to_string()));

        let mut ctx = serde_json::Map::new();
        ctx.insert("items".to_string(), json!(["a", "b", "c"]));
        assert_eq!(segment.render(&ctx), "a, b, c");
    }

    #[test]
    fn test_serialization() {
        let segment = PromptSegment::fixed("Test");
        let json = serde_json::to_string(&segment).unwrap();
        assert!(json.contains("\"segment_type\":\"fixed\""));
        assert!(json.contains("\"content\":\"Test\""));
    }

    #[test]
    fn test_deserialization() {
        let json = r#"{"segment_type":"variable","source":"task"}"#;
        let segment: PromptSegment = serde_json::from_str(json).unwrap();
        assert_eq!(segment.segment_type, SegmentType::Variable);
        assert_eq!(segment.source, Some("task".to_string()));
    }
}
