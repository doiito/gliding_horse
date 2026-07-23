use crate::utils::text;
use serde_json::Value;

/// Preview-size limit for JSON values embedded in a summary.
const JSON_VALUE_PREVIEW_WIDTH: usize = 200;

pub fn smart_truncate(result: &str, max_bytes: usize) -> String {
    let trimmed = result.trim();

    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
        return smart_truncate_json_value(&val, max_bytes);
    }

    text::smart_truncate_text(result, max_bytes)
}

pub fn smart_truncate_json(json_str: &str, max_bytes: usize) -> String {
    if let Ok(val) = serde_json::from_str::<Value>(json_str.trim()) {
        return smart_truncate_json_value(&val, max_bytes);
    }
    text::smart_truncate_text(json_str, max_bytes)
}

fn smart_truncate_json_value(val: &Value, max_bytes: usize) -> String {
    match val {
        Value::Array(arr) => truncate_json_array(arr, max_bytes),
        Value::Object(obj) => truncate_json_object(obj, max_bytes),
        _ => {
            let s = val.to_string();
            if s.len() <= max_bytes {
                s
            } else {
                text::smart_truncate_text(&s, max_bytes)
            }
        }
    }
}

fn truncate_json_array(arr: &[Value], max_bytes: usize) -> String {
    let total = arr.len();
    let mut kept = Vec::new();
    let mut current_size = 2;

    for item in arr {
        let item_str = item.to_string();
        let needed = if kept.is_empty() {
            item_str.len()
        } else {
            item_str.len() + 2
        };

        if current_size + needed + 50 > max_bytes {
            break;
        }

        kept.push(item_str);
        current_size += needed;
    }

    let mut result = String::from("[");
    result.push_str(&kept.join(", "));
    result.push(']');

    if kept.len() < total {
        result.push_str(&format!(
            "\n\n[truncated: {} total elements, {} kept]",
            total,
            kept.len()
        ));
    }

    result
}

fn truncate_json_object(obj: &serde_json::Map<String, Value>, max_bytes: usize) -> String {
    let mut result_obj = serde_json::Map::new();
    let mut current_size = 2;

    for (key, value) in obj {
        let truncated_value = if let Value::String(s) = value {
            if text::display_width(s) > JSON_VALUE_PREVIEW_WIDTH {
                let preview = text::truncate_preview(s, JSON_VALUE_PREVIEW_WIDTH);
                Value::String(format!(
                    "{} [truncated: original {} characters]",
                    preview,
                    text::display_width(s)
                ))
            } else {
                value.clone()
            }
        } else if let Value::Array(arr) = value {
            if arr.len() > 10 {
                let truncated: Vec<Value> = arr.iter().take(10).cloned().collect();
                Value::Array(truncated)
            } else {
                value.clone()
            }
        } else {
            value.clone()
        };

        let entry_size = key.len() + truncated_value.to_string().len() + 4;
        if current_size + entry_size > max_bytes {
            break;
        }

        current_size += entry_size;
        result_obj.insert(key.clone(), truncated_value);
    }

    let mut result = serde_json::to_string_pretty(&Value::Object(result_obj)).unwrap_or_default();

    if result.len() > max_bytes {
        result = text::smart_truncate_text(&result, max_bytes);
    }

    result
}

pub fn format_iri_message(
    tool_name: &str,
    call_id: &str,
    result_summary: &str,
    result_size: usize,
) -> String {
    let threshold_small: usize = 2048;
    let threshold_large: usize = 8192;

    let size_mark = if result_size < threshold_small {
        ""
    } else if result_size < threshold_large {
        " [compressed]"
    } else {
        " [archived]"
    };

    let iri = format!("iri://tool-result/{}", call_id);
    let summary_preview = text::truncate_preview(result_summary, 200);

    format!(
        "[{tool}{mark}] {summary}\nIRI: {iri}",
        tool = tool_name,
        mark = size_mark,
        summary = summary_preview,
        iri = iri,
    )
}

pub fn generate_text_summary(result_str: &str, tool_name: &str, preview_bytes: usize) -> String {
    let size = result_str.len();
    let lines: Vec<&str> = result_str.lines().collect();
    let line_count = lines.len();

    let preview = if result_str.len() > preview_bytes {
        text::safe_truncate(result_str, preview_bytes).to_string()
    } else {
        result_str.to_string()
    };

    let mut summary = format!(
        "Tool [{}] returned large text result ({} bytes, {} lines):\n\n--- Preview ---\n{}\n",
        tool_name, size, line_count, preview
    );

    if size > preview_bytes {
        let tail_chars = 200usize;
        let tail_start = size.saturating_sub(tail_chars);
        let tail_start_adjusted = text::safe_truncate(result_str, tail_start).len();
        let tail = text::safe_truncate(&result_str[tail_start_adjusted..], tail_chars);
        summary.push_str(&format!("\n--- End preview ---\n{}\n", tail));
        summary.push_str("\n[Full result stored, use read_full_result to read on demand]");
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_json_array() {
        let items: Vec<Value> = (0..50)
            .map(|i| serde_json::json!({"id": i, "name": format!("item_{}", i)}))
            .collect();
        let json = serde_json::to_string(&items).unwrap();

        let result = smart_truncate(&json, 500);
        assert!(result.len() < 600);
        assert!(result.contains("truncated"));
        assert!(result.contains("50 total elements"));
    }

    #[test]
    fn test_truncate_json_object() {
        let mut obj = serde_json::Map::new();
        for i in 0..20 {
            obj.insert(format!("key_{}", i), Value::String("x".repeat(500)));
        }
        let json = serde_json::to_string(&Value::Object(obj)).unwrap();

        let result = smart_truncate(&json, 1000);
        assert!(result.len() < 1100);
    }

    #[test]
    fn test_truncate_invalid_json_fallback() {
        let text = "not json\n".repeat(500);
        let result = smart_truncate(&text, 1000);
        assert!(result.contains("truncated"));
        assert!(result.contains("lines"));
    }

    #[test]
    fn test_generate_summary_utf8() {
        let text = "Chinese content\n".repeat(1000);
        let summary = generate_text_summary(&text, "test_tool", 200);
        assert!(summary.contains("test_tool"));
        assert!(summary.contains("read_full_result"));
        assert!(summary.is_char_boundary(summary.len()));
    }

    #[test]
    fn test_generate_summary() {
        let text = "line\n".repeat(1000);
        let summary = generate_text_summary(&text, "test_tool", 200);
        assert!(summary.contains("test_tool"));
        assert!(summary.contains("1000 lines"));
        assert!(summary.contains("read_full_result"));
    }
}
