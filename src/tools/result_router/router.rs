use super::{RouteDecision, ToolResultMeta};
use crate::config::settings::ToolResultRouterSettings;

pub struct ResultRouter {
    enabled: bool,
    threshold_small: usize,
    threshold_large: usize,
    micro_tool_threshold: usize,
    preview_size: usize,
}

impl ResultRouter {
    pub fn new(settings: &ToolResultRouterSettings) -> Self {
        Self {
            enabled: settings.enabled,
            threshold_small: settings.threshold_small,
            threshold_large: settings.threshold_large,
            micro_tool_threshold: settings.micro_tool_threshold,
            preview_size: settings.preview_size,
        }
    }

    pub fn route(&self, result_str: &str, tool_name: &str, call_id: &str) -> RouteDecision {
        if !self.enabled {
            return RouteDecision::Truncate { max_chars: 8000 };
        }

        let size = result_str.len();

        // file_read: multi-line ≤1000 lines passes through, single-line ≤32KB passes through
        if tool_name == "file_read" {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(result_str) {
                if let Some(total_lines) = val.get("total_lines").and_then(|v| v.as_u64()) {
                    let is_large = if total_lines > 1 {
                        total_lines > 1000
                    } else {
                        size > 32768
                    };
                    if !is_large {
                        return RouteDecision::PassThrough;
                    }
                }
            }
        }

        if size < self.threshold_small {
            return RouteDecision::PassThrough;
        }

        // >= micro_tool_threshold: generate IRI + micro-tools (original Truncate path changed to Summarize)
        if size >= self.micro_tool_threshold && size <= self.threshold_large {
            return RouteDecision::Summarize {
                call_id: call_id.to_string(),
                preview_size: self.preview_size,
            };
        }

        if size <= self.threshold_large {
            return RouteDecision::Truncate {
                max_chars: self.threshold_large,
            };
        }

        if Self::is_structured_json(result_str) {
            let graph_name = format!("graph:tool-result:{}", call_id);
            RouteDecision::Graphify {
                call_id: call_id.to_string(),
                graph_name,
            }
        } else {
            RouteDecision::Summarize {
                call_id: call_id.to_string(),
                preview_size: self.preview_size,
            }
        }
    }

    pub fn analyze(&self, result_str: &str, tool_name: &str, call_id: &str) -> ToolResultMeta {
        let size_bytes = result_str.len();
        let is_json = Self::try_parse_json(result_str).is_some();
        let is_structured = is_json && Self::has_complex_structure(result_str);

        ToolResultMeta {
            tool_name: tool_name.to_string(),
            call_id: call_id.to_string(),
            size_bytes,
            is_json,
            is_structured,
        }
    }

    fn is_structured_json(s: &str) -> bool {
        let trimmed = s.trim();
        if !trimmed.starts_with('[') && !trimmed.starts_with('{') {
            return false;
        }

        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return Self::has_complex_value(&val);
        }
        false
    }

    fn has_complex_structure(s: &str) -> bool {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(s.trim()) {
            return Self::has_complex_value(&val);
        }
        false
    }

    fn has_complex_value(val: &serde_json::Value) -> bool {
        match val {
            serde_json::Value::Array(arr) => {
                if arr.is_empty() {
                    return false;
                }
                if let Some(first) = arr.first() {
                    matches!(
                        first,
                        serde_json::Value::Object(_) | serde_json::Value::Array(_)
                    )
                } else {
                    false
                }
            }
            serde_json::Value::Object(obj) => {
                obj.values().any(|v| {
                    matches!(
                        v,
                        serde_json::Value::Object(_) | serde_json::Value::Array(_)
                    )
                }) || obj.len() > 5
            }
            _ => false,
        }
    }

    fn try_parse_json(s: &str) -> Option<serde_json::Value> {
        serde_json::from_str(s.trim()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_settings() -> ToolResultRouterSettings {
        ToolResultRouterSettings::default()
    }

    fn disabled_settings() -> ToolResultRouterSettings {
        let mut s = ToolResultRouterSettings::default();
        s.enabled = false;
        s
    }

    #[test]
    fn test_small_result_passthrough() {
        let router = ResultRouter::new(&default_settings());
        let result = "small result";
        let decision = router.route(result, "test_tool", "call_1");
        assert_eq!(decision, RouteDecision::PassThrough);
    }

    #[test]
    fn test_medium_result_summarize() {
        // threshold_small=16384, micro_tool_threshold=16384, threshold_large=32768
        // size must be >= micro_tool_threshold for Summarize path
        let router = ResultRouter::new(&default_settings());
        let result = "x".repeat(20_000);
        let decision = router.route(&result, "test_tool", "call_2");
        assert!(matches!(decision, RouteDecision::Summarize { .. }));
    }

    #[test]
    fn test_large_json_graphify() {
        let router = ResultRouter::new(&default_settings());
        let items: Vec<serde_json::Value> = (0..1200)
            .map(|i| serde_json::json!({"id": i, "name": format!("item_{}", i), "value": i * 10}))
            .collect();
        let result = serde_json::to_string(&items).unwrap();
        // must be > threshold_large (32768) + structured JSON to trigger Graphify
        assert!(
            result.len() > 32768,
            "result size {} should exceed threshold_large",
            result.len()
        );

        let decision = router.route(&result, "test_tool", "call_3");
        assert!(matches!(decision, RouteDecision::Graphify { .. }));
    }

    #[test]
    fn test_large_text_summarize() {
        let router = ResultRouter::new(&default_settings());
        let result = "line\n".repeat(8000);
        assert!(result.len() > 32768);

        let decision = router.route(&result, "test_tool", "call_4");
        assert!(matches!(decision, RouteDecision::Summarize { .. }));
    }

    #[test]
    fn test_large_simple_json_summarize() {
        let router = ResultRouter::new(&default_settings());
        let result = format!("{{\"data\": \"{}\"}}", "x".repeat(35000));
        assert!(result.len() > 32768);

        let decision = router.route(&result, "test_tool", "call_6");
        // simple JSON is not "structured" (no nested objects/arrays, keys <= 5)
        assert!(matches!(decision, RouteDecision::Summarize { .. }));
    }

    #[test]
    fn test_disabled_fallback_truncate() {
        let router = ResultRouter::new(&disabled_settings());
        let result = "x".repeat(10000);
        let decision = router.route(&result, "test_tool", "call_5");
        assert_eq!(decision, RouteDecision::Truncate { max_chars: 8000 });
    }

    #[test]
    fn test_analyze_meta() {
        let router = ResultRouter::new(&default_settings());
        let meta = router.analyze("{\"key\": \"value\"}", "test_tool", "call_7");
        assert_eq!(meta.tool_name, "test_tool");
        assert_eq!(meta.call_id, "call_7");
        assert!(meta.is_json);
    }
}
