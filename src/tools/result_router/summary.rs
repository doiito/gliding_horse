use crate::utils::text;
use serde_json::Value;

use super::ResultRoutingIdentity;

/// Preview-size limit for JSON values embedded in a summary.
const JSON_VALUE_PREVIEW_WIDTH: usize = 200;
/// Final inline budget for an already-bounded routed result.  Routing branches
/// deliberately spend up to 2--4 KiB producing a useful preview; collapsing
/// that preview to 200 columns here discarded the evidence and caused models
/// to request the full result unnecessarily.
const ROUTED_RESULT_PREVIEW_BYTES: usize = 4 * 1024;

fn bounded_utf8_preview(text: &str, max_bytes: usize) -> String {
    bounded_utf8_preview_with_consumed(text, max_bytes).0
}

fn bounded_utf8_preview_with_consumed(text: &str, max_bytes: usize) -> (String, usize) {
    if text.len() <= max_bytes {
        return (text.to_string(), text.chars().count());
    }
    if max_bytes <= 3 {
        return (".".repeat(max_bytes), 0);
    }
    let prefix = text::safe_truncate(text, max_bytes - 3);
    (format!("{prefix}..."), prefix.chars().count())
}

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

fn file_read_preview_envelope(
    object: &serde_json::Map<String, Value>,
    routing: &ResultRoutingIdentity,
    result_size: usize,
    max_bytes: usize,
) -> String {
    const PATH_PREVIEW_BYTES: usize = 512;
    const PARTIAL_LINE_MARKER: &str = "...[line preview truncated]";

    let source_offset = object.get("offset").and_then(Value::as_u64).unwrap_or(0);
    let source_lines = object
        .get("lines")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let total_lines = object
        .get("total_lines")
        .and_then(Value::as_u64)
        .unwrap_or(source_offset.saturating_add(source_lines.len() as u64));
    let path = object
        .get("path")
        .and_then(Value::as_str)
        .map(|path| bounded_utf8_preview(path, PATH_PREVIEW_BYTES));
    let content_sha256 = object
        .get("content_sha256")
        .and_then(Value::as_str)
        .map(str::to_string);

    let render = |lines: &[String], complete_lines: usize, partial_prefix_chars: Option<usize>| {
        let partial_line = partial_prefix_chars.is_some();
        let next_offset = source_offset.saturating_add(complete_lines as u64);
        let has_more =
            partial_line || complete_lines < source_lines.len() || next_offset < total_lines;
        let message = if partial_line {
            format!(
                "The line at file offset {next_offset} is only partially previewed. Re-read from offset {next_offset} with a bounded file_read call, or call {} with reader_cursor only while that exact reader is advertised; do not skip this line.",
                routing.reader_name
            )
        } else if has_more {
            format!(
                "Continue file_read at offset {next_offset} with a bounded limit, or call {} with reader_cursor only while that exact reader is advertised.",
                routing.reader_name
            )
        } else {
            format!(
                "All reported file lines are represented in this preview. {} remains valid only while that exact reader is advertised.",
                routing.reader_name
            )
        };
        let reader_cursor = has_more.then(|| {
            partial_prefix_chars.map_or_else(
                || {
                    serde_json::json!({
                        "offset": next_offset,
                        "char_offset": 0,
                    })
                },
                |char_offset| {
                    serde_json::json!({
                        "offset": next_offset,
                        "limit": 1,
                        "char_offset": char_offset,
                    })
                },
            )
        });
        serde_json::json!({
            "path": path,
            "content_sha256": content_sha256,
            "total_lines": total_lines,
            "offset": source_offset,
            "lines": lines,
            "returned": complete_lines,
            "previewed_lines": lines.len(),
            "next_offset": has_more.then_some(next_offset),
            "reader_cursor": reader_cursor,
            "partial_line_preview": partial_line,
            "preview": true,
            "archived": true,
            "original_size_bytes": result_size,
            "result_iri": routing.storage_iri,
            "session_reader": routing.reader_name,
            "message": message,
        })
    };

    let mut lines = Vec::<String>::new();
    let mut complete_lines = 0usize;
    let mut partial_line = false;
    for value in source_lines {
        let line = value
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| value.to_string());
        let mut candidate_lines = lines.clone();
        candidate_lines.push(line.clone());
        let candidate = render(&candidate_lines, complete_lines + 1, None).to_string();
        if candidate.len() <= max_bytes {
            lines = candidate_lines;
            complete_lines += 1;
            continue;
        }

        // Keep a useful prefix of a single wide line without splitting a
        // Unicode scalar or claiming that the line was completely returned.
        let line_chars = line.chars().count();
        let mut low = 0usize;
        let mut high = line_chars;
        while low < high {
            let midpoint = low + (high - low).div_ceil(2);
            let prefix = line.chars().take(midpoint).collect::<String>();
            let mut partial_lines = lines.clone();
            partial_lines.push(format!("{prefix}{PARTIAL_LINE_MARKER}"));
            if render(&partial_lines, complete_lines, Some(midpoint))
                .to_string()
                .len()
                <= max_bytes
            {
                low = midpoint;
            } else {
                high = midpoint - 1;
            }
        }
        if low > 0 {
            let prefix = line.chars().take(low).collect::<String>();
            lines.push(format!("{prefix}{PARTIAL_LINE_MARKER}"));
            partial_line = true;
        }
        break;
    }

    let rendered = render(
        &lines,
        complete_lines,
        partial_line.then_some(
            lines
                .last()
                .and_then(|line| line.strip_suffix(PARTIAL_LINE_MARKER))
                .map(|prefix| prefix.chars().count())
                .unwrap_or(0),
        ),
    )
    .to_string();
    if rendered.len() <= max_bytes {
        return rendered;
    }

    // Metadata alone is expected to fit, but keep the contract fail-safe if a
    // future field grows: return a minimal valid JSON envelope, never a byte-
    // sliced JSON fragment.
    serde_json::json!({
        "path": path,
        "content_sha256": content_sha256,
        "total_lines": total_lines,
        "offset": source_offset,
        "returned": 0,
        "next_offset": source_offset,
        "reader_cursor": {"offset": source_offset, "char_offset": 0},
        "preview": true,
        "archived": true,
        "session_reader": routing.reader_name,
        "message": format!(
            "Preview metadata exceeded its budget. Continue file_read at offset {source_offset}, or call {} only while advertised.",
            routing.reader_name
        ),
    })
    .to_string()
}

/// Preserve the outcome-bearing fields of a large execution envelope. The
/// The byte-exact result remains archived for audit, and any view not fully
/// represented inline remains available through `session_reader`. The preview
/// is deliberately valid JSON so both the model and TUI can inspect exit
/// status and output without being dominated by a long command.
pub fn execution_preview_envelope(
    tool_name: &str,
    result: &str,
    routing: &ResultRoutingIdentity,
    result_size: usize,
    max_bytes: usize,
) -> String {
    let parsed = serde_json::from_str::<Value>(result).ok();
    let object = parsed.as_ref().and_then(Value::as_object);
    let string_field = |name: &str| {
        object
            .and_then(|value| value.get(name))
            .and_then(Value::as_str)
            .unwrap_or_default()
    };
    let command = object
        .and_then(|value| value.get("command").or_else(|| value.get("code")))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let stdout = string_field("stdout");
    let stderr = string_field("stderr");
    let exit_code = object
        .and_then(|value| value.get("exit_code"))
        .cloned()
        .unwrap_or(Value::Null);
    let duration_ms = object
        .and_then(|value| value.get("duration_ms"))
        .cloned()
        .unwrap_or(Value::Null);

    let render = |command_budget: usize, stdout_budget: usize, stderr_budget: usize| {
        let (command_preview, command_next) =
            bounded_utf8_preview_with_consumed(command, command_budget);
        let (stdout_preview, stdout_next) =
            bounded_utf8_preview_with_consumed(stdout, stdout_budget);
        let (stderr_preview, stderr_next) =
            bounded_utf8_preview_with_consumed(stderr, stderr_budget);
        serde_json::json!({
            "tool": tool_name,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "command_preview": command_preview,
            "command_sha256": crate::utils::CryptoUtils::sha256_hex(command),
            "command_chars": command.chars().count(),
            "stdout": stdout_preview,
            "stdout_chars": stdout.chars().count(),
            "stderr": stderr_preview,
            "stderr_chars": stderr.chars().count(),
            "reader_view": "execution_stream",
            "reader_cursors": {
                "stdout": (stdout_next < stdout.chars().count()).then(|| serde_json::json!({
                    "stream": "stdout",
                    "char_offset": stdout_next,
                })),
                "stderr": (stderr_next < stderr.chars().count()).then(|| serde_json::json!({
                    "stream": "stderr",
                    "char_offset": stderr_next,
                })),
                "command": (command_next < command.chars().count()).then(|| serde_json::json!({
                    "stream": "command",
                    "char_offset": command_next,
                })),
            },
            "archived": true,
            "original_size_bytes": result_size,
            "result_iri": routing.storage_iri,
            "session_reader": routing.reader_name,
            "message": format!(
                "Execution outcome is inline. Use {} only while advertised. It defaults to stdout; copy a reader_cursors entry exactly and use stream/char_offset. offset/limit are invalid for execution results.",
                routing.reader_name
            ),
        })
        .to_string()
    };

    let max_bytes = max_bytes.clamp(1_024, ROUTED_RESULT_PREVIEW_BYTES);
    let mut command_budget = 384usize;
    let mut stdout_budget = max_bytes.saturating_mul(5) / 8;
    let mut stderr_budget = max_bytes.saturating_mul(1) / 8;
    loop {
        let rendered = render(command_budget, stdout_budget, stderr_budget);
        if rendered.len() <= max_bytes {
            return rendered;
        }
        if command_budget == 0 && stdout_budget == 0 && stderr_budget == 0 {
            break;
        }
        command_budget /= 2;
        stdout_budget /= 2;
        stderr_budget /= 2;
    }

    // Metadata must remain parseable even for pathological escaped strings.
    serde_json::json!({
        "tool": tool_name,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "command_sha256": crate::utils::CryptoUtils::sha256_hex(command),
        "command_chars": command.chars().count(),
        "stdout_chars": stdout.chars().count(),
        "stderr_chars": stderr.chars().count(),
        "reader_view": "execution_stream",
        "reader_cursors": {
            "stdout": (!stdout.is_empty()).then(|| serde_json::json!({"stream": "stdout", "char_offset": 0})),
            "stderr": (!stderr.is_empty()).then(|| serde_json::json!({"stream": "stderr", "char_offset": 0})),
            "command": (!command.is_empty()).then(|| serde_json::json!({"stream": "command", "char_offset": 0})),
        },
        "archived": true,
        "original_size_bytes": result_size,
        "result_iri": routing.storage_iri,
        "session_reader": routing.reader_name,
        "message": format!(
            "Use {} only while advertised with stream and char_offset from reader_cursors; offset/limit are invalid for execution results.",
            routing.reader_name
        ),
    })
    .to_string()
}

pub fn format_iri_message(
    tool_name: &str,
    routing: &ResultRoutingIdentity,
    result_summary: &str,
    result_size: usize,
) -> String {
    if tool_name == "file_read" {
        if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(result_summary) {
            return file_read_preview_envelope(
                &object,
                routing,
                result_size,
                ROUTED_RESULT_PREVIEW_BYTES,
            );
        }
    }

    let threshold_small: usize = 2048;
    let threshold_large: usize = 8192;

    let size_mark = if result_size < threshold_small {
        ""
    } else if result_size < threshold_large {
        " [compressed]"
    } else {
        " [archived]"
    };

    let summary_preview = bounded_utf8_preview(result_summary, ROUTED_RESULT_PREVIEW_BYTES);

    format!(
        "[{tool}{mark}] {summary}\nIRI: {iri}\nSession reader: {reader}\nCall this reader only while the exact name is advertised in the current turn's tool schemas.",
        tool = tool_name,
        mark = size_mark,
        summary = summary_preview,
        iri = routing.storage_iri,
        reader = routing.reader_name,
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
        summary.push_str(
            "\n[Full result stored; use only the exact session reader named by the routing envelope when it is currently advertised]",
        );
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
        assert!(summary.contains("exact session reader"));
        assert!(!summary.contains("read_full_result_*"));
        assert!(summary.is_char_boundary(summary.len()));
    }

    #[test]
    fn test_generate_summary() {
        let text = "line\n".repeat(1000);
        let summary = generate_text_summary(&text, "test_tool", 200);
        assert!(summary.contains("test_tool"));
        assert!(summary.contains("1000 lines"));
        assert!(summary.contains("exact session reader"));
        assert!(!summary.contains("read_full_result_*"));
    }

    #[test]
    fn routed_result_preserves_useful_preview_and_names_exact_reader() {
        let preview = format!("{}END", "x".repeat(3_000));
        let identity = ResultRoutingIdentity::new("l1-summary", "call_exact");
        let routed = format_iri_message("file_read", &identity, &preview, 9_000);

        assert!(routed.contains("END"));
        assert!(routed.contains(&identity.reader_name));
        assert!(routed.contains(&identity.storage_iri));
        assert!(routed.contains("exact name is advertised"));
    }

    #[test]
    fn routed_result_preview_is_utf8_safe_and_bounded_to_four_kib() {
        let preview = "界".repeat(3_000);
        let identity = ResultRoutingIdentity::new("l1-summary", "call_utf8");
        let routed = format_iri_message("file_read", &identity, &preview, preview.len());
        let inline = routed
            .strip_prefix("[file_read [archived]] ")
            .unwrap()
            .split("\nIRI:")
            .next()
            .unwrap();

        assert!(inline.is_char_boundary(inline.len()));
        assert!(inline.len() <= ROUTED_RESULT_PREVIEW_BYTES);
        assert!(inline.len() > 200);
    }

    #[test]
    fn file_read_ascii_wide_line_stays_valid_json_and_does_not_skip_partial_line() {
        let identity = ResultRoutingIdentity::new("l1-file-wide-ascii", "call_0");
        let preview = serde_json::json!({
            "path": "/workspace/project/generated.txt",
            "content_sha256": "a".repeat(64),
            "total_lines": 17,
            "offset": 6,
            "lines": ["A".repeat(100_000)],
            "returned": 1,
            "preview": true,
        })
        .to_string();

        let routed = format_iri_message("file_read", &identity, &preview, 100_000);
        assert!(routed.len() <= ROUTED_RESULT_PREVIEW_BYTES);
        let envelope: Value =
            serde_json::from_str(&routed).expect("preview must remain valid JSON");
        assert_eq!(envelope["offset"], 6);
        assert_eq!(envelope["content_sha256"], "a".repeat(64));
        assert_eq!(envelope["returned"], 0);
        assert_eq!(envelope["previewed_lines"], 1);
        assert_eq!(envelope["next_offset"], 6);
        assert_eq!(envelope["partial_line_preview"], true);
        assert_eq!(envelope["session_reader"], identity.reader_name);
        assert_eq!(envelope["result_iri"], identity.storage_iri);
        let line = envelope["lines"][0].as_str().unwrap();
        assert!(line.starts_with('A'));
        assert!(line.ends_with("...[line preview truncated]"));
        let prefix_chars = line
            .strip_suffix("...[line preview truncated]")
            .unwrap()
            .chars()
            .count();
        assert!(prefix_chars > 0);
        assert_eq!(envelope["reader_cursor"]["offset"], 6);
        assert_eq!(envelope["reader_cursor"]["limit"], 1);
        assert_eq!(
            envelope["reader_cursor"]["char_offset"],
            prefix_chars as u64
        );
        assert!(envelope["message"].as_str().unwrap().contains("offset 6"));
    }

    #[test]
    fn file_read_cjk_wide_line_obeys_byte_budget_at_utf8_boundary() {
        let identity = ResultRoutingIdentity::new("l1-file-wide-cjk", "call_0");
        let preview = serde_json::json!({
            "path": "/workspace/报告/趋势.md",
            "total_lines": 9,
            "offset": 3,
            "lines": ["趋势与场景".repeat(20_000)],
            "returned": 1,
            "preview": true,
        })
        .to_string();

        let routed = format_iri_message("file_read", &identity, &preview, preview.len());
        assert!(routed.len() <= ROUTED_RESULT_PREVIEW_BYTES);
        assert!(routed.is_char_boundary(routed.len()));
        let envelope: Value = serde_json::from_str(&routed).expect("UTF-8 JSON must be complete");
        let line = envelope["lines"][0].as_str().unwrap();
        assert!(line.starts_with("趋势与场景"));
        assert!(!line.contains('\u{fffd}'));
        let prefix_chars = line
            .strip_suffix("...[line preview truncated]")
            .unwrap()
            .chars()
            .count();
        assert_eq!(envelope["returned"], 0);
        assert_eq!(envelope["next_offset"], 3);
        assert_eq!(envelope["reader_cursor"]["limit"], 1);
        assert_eq!(
            envelope["reader_cursor"]["char_offset"],
            prefix_chars as u64
        );
        assert_eq!(envelope["partial_line_preview"], true);
        assert!(envelope["message"]
            .as_str()
            .unwrap()
            .contains(&identity.reader_name));
    }

    #[test]
    fn execution_preview_keeps_outcome_ahead_of_large_command_and_is_bounded() {
        let identity = ResultRoutingIdentity::new("l1-shell-preview", "call_verify");
        let raw = serde_json::json!({
            "command": format!("python3 -m pytest {}", "very_long_argument ".repeat(2_000)),
            "exit_code": 0,
            "duration_ms": 731,
            "stdout": format!("29 passed\n{}", "detail\n".repeat(3_000)),
            "stderr": "",
        })
        .to_string();

        let preview = execution_preview_envelope("bash", &raw, &identity, raw.len(), 4_096);
        assert!(preview.len() <= 4_096);
        let value: Value = serde_json::from_str(&preview).expect("valid JSON execution preview");
        assert_eq!(value["exit_code"], 0);
        assert_eq!(value["duration_ms"], 731);
        assert!(value["stdout"].as_str().unwrap().contains("29 passed"));
        assert!(value["command_preview"].as_str().unwrap().len() < raw.len());
        assert_eq!(value["session_reader"], identity.reader_name);
        assert_eq!(value["result_iri"], identity.storage_iri);
    }

    #[test]
    fn execution_preview_preserves_utf8_boundaries() {
        let identity = ResultRoutingIdentity::new("l1-shell-cjk", "call_verify");
        let raw = serde_json::json!({
            "command": "python3 -m pytest",
            "exit_code": 1,
            "duration_ms": 12,
            "stdout": "测试通过".repeat(3_000),
            "stderr": "失败详情".repeat(3_000),
        })
        .to_string();
        let preview = execution_preview_envelope("bash", &raw, &identity, raw.len(), 4_096);
        assert!(preview.len() <= 4_096);
        assert!(preview.is_char_boundary(preview.len()));
        let value: Value = serde_json::from_str(&preview).expect("valid UTF-8 JSON");
        assert!(!value["stdout"].as_str().unwrap().contains('\u{fffd}'));
        assert!(!value["stderr"].as_str().unwrap().contains('\u{fffd}'));
    }
}
