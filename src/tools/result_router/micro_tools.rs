use serde_json::json;

use super::{MicroToolSchema, MicroToolType, ResultRoutingIdentity, SchemaAnalysis};

/// The model-facing coordinate system of one archived result reader.
///
/// Tool results are stored as byte-exact envelopes for auditability, but an
/// envelope is not necessarily the text the model saw in its routed preview.
/// In particular, JSON escapes newlines inside `file_read.lines` and shell
/// `stdout`.  Keeping the view explicit prevents a line cursor from being
/// applied to the single physical line of serialized JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArchivedReaderView {
    RawText,
    FileLines,
    ExecutionStream,
}

impl ArchivedReaderView {
    pub(crate) fn for_source_tool(tool_name: &str) -> Self {
        match tool_name {
            "file_read" => Self::FileLines,
            "bash" | "powershell" | "code_execute" => Self::ExecutionStream,
            _ => Self::RawText,
        }
    }

    /// Select a typed view only when the archived byte-exact envelope has the
    /// corresponding shape. Custom tools may reuse a built-in name while
    /// returning plain text; those results must remain readable as raw text
    /// instead of being trapped behind an incompatible schema.
    pub(crate) fn for_result(tool_name: &str, content: &str) -> Self {
        let candidate = Self::for_source_tool(tool_name);
        let parsed = serde_json::from_str::<serde_json::Value>(content).ok();
        match candidate {
            Self::FileLines
                if parsed
                    .as_ref()
                    .and_then(|value| value.get("lines"))
                    .is_some_and(serde_json::Value::is_array) =>
            {
                Self::FileLines
            }
            Self::ExecutionStream
                if parsed.as_ref().is_some_and(|value| {
                    ["stdout", "stderr"]
                        .iter()
                        .all(|field| value.get(*field).is_some_and(serde_json::Value::is_string))
                }) =>
            {
                Self::ExecutionStream
            }
            _ => Self::RawText,
        }
    }
}

/// Build the one canonical schema used by registered and reconstructed
/// `read_full_result_*` tools. Cursor units intentionally differ by typed
/// view and therefore use different field names.
pub(crate) fn archived_reader_parameters(
    view: ArchivedReaderView,
    max_line_limit: usize,
    max_char_limit: usize,
) -> serde_json::Value {
    match view {
        ArchivedReaderView::FileLines => json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Absolute 0-based source-file line offset; use next_cursor.offset exactly"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_line_limit.max(1),
                    "description": "Maximum source-file lines to select"
                },
                "char_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset within the selected line page; use only when next_cursor repeats the same line offset"
                }
            }
        }),
        ArchivedReaderView::ExecutionStream => json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "stream": {
                    "type": "string",
                    "enum": ["stdout", "stderr", "command", "raw"],
                    "default": "stdout",
                    "description": "Archived execution field to page; stdout is the default"
                },
                "char_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "0-based Unicode character offset; copy next_cursor.char_offset exactly"
                },
                "char_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_char_limit.max(1),
                    "description": "Maximum Unicode characters to return"
                }
            }
        }),
        ArchivedReaderView::RawText => json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "char_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "0-based Unicode character offset; copy next_cursor.char_offset exactly"
                },
                "char_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_char_limit.max(1),
                    "description": "Maximum Unicode characters to return"
                }
            }
        }),
    }
}

pub struct MicroToolGenerator;

impl MicroToolGenerator {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_from_schema(
        analysis: &SchemaAnalysis,
        routing: &ResultRoutingIdentity,
        max_tools: usize,
    ) -> Vec<MicroToolSchema> {
        let graph_name = routing.graph_name.clone();
        let mut tools = Vec::new();

        for (type_name, count) in &analysis.entity_types {
            if tools.len() >= max_tools {
                break;
            }
            let short_name = type_name.split('/').last().unwrap_or(type_name);
            let tool_name = routing.query_name(type_name);

            tools.push(MicroToolSchema {
                name: tool_name,
                description: format!(
                    "Query entities of type {} (total {}). Supports property filtering.",
                    short_name, count
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "filter_property": {
                            "type": "string",
                            "description": "Property name to filter on"
                        },
                        "filter_value": {
                            "description": "Exact JSON value to match"
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Matching-result offset",
                            "default": 0
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum results to return",
                            "default": 10
                        }
                    }
                }),
                tool_type: MicroToolType::EntityTypeQuery {
                    entity_type: type_name.clone(),
                    graph_name: graph_name.clone(),
                },
            });
        }

        if tools.len() < max_tools {
            tools.push(MicroToolSchema {
                name: routing.entity_details_name(),
                description: "Get all properties and relations of a specific entity".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID"
                        }
                    },
                    "required": ["entity_id"]
                }),
                tool_type: MicroToolType::EntityDetails {
                    graph_name: graph_name.clone(),
                },
            });
        }

        if tools.len() < max_tools && !analysis.relation_types.is_empty() {
            tools.push(MicroToolSchema {
                name: routing.relation_expansion_name(),
                description: format!(
                    "Traverse along relation edges. Available relations: {}",
                    analysis.relation_types.join(", ")
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Starting entity ID"
                        },
                        "relation": {
                            "type": "string",
                            "description": "Relation type"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Traversal depth",
                            "default": 1
                        }
                    },
                    "required": ["entity_id"]
                }),
                tool_type: MicroToolType::RelationTraversal {
                    graph_name: graph_name.clone(),
                },
            });
        }

        tools.truncate(max_tools);
        tools
    }

    pub fn generate_read_full_tool(
        routing: &ResultRoutingIdentity,
        storage_key: &str,
        preview_size: usize,
    ) -> MicroToolSchema {
        MicroToolSchema {
            name: routing.reader_name.clone(),
            description: format!(
                "Read full tool result (preview shows first {} characters)",
                preview_size
            ),
            parameters: archived_reader_parameters(
                ArchivedReaderView::RawText,
                preview_size,
                preview_size,
            ),
            tool_type: MicroToolType::FullTextRead {
                storage_key: storage_key.to_string(),
            },
        }
    }

    pub fn format_tool_injection_message(summary: &str, tools: &[MicroToolSchema]) -> String {
        const MAX_ENVELOPE_BYTES: usize = 3_000;
        const MAX_DESCRIPTION_BYTES: usize = 180;
        if tools.is_empty() {
            return crate::utils::text::safe_truncate(summary, MAX_ENVELOPE_BYTES).to_string();
        }
        // Capability names come first so the outer routed-preview bound can
        // never trim them after a multi-byte/CJK summary.
        let mut msg = "Available session-scoped graph tools:\n".to_string();
        let mut listed = 0usize;
        for tool in tools {
            let description =
                crate::utils::text::safe_truncate(&tool.description, MAX_DESCRIPTION_BYTES);
            let line = format!("- `{}`: {}\n", tool.name, description);
            if msg.len().saturating_add(line.len()) > MAX_ENVELOPE_BYTES {
                break;
            }
            msg.push_str(&line);
            listed += 1;
        }
        if listed < tools.len() {
            msg.push_str(&format!(
                "- [{} additional graph tools omitted by envelope bound]\n",
                tools.len() - listed
            ));
        }
        msg.push_str(
            "\nThese tools are valid only while their exact names appear in the current turn's schemas.",
        );
        let remaining = MAX_ENVELOPE_BYTES.saturating_sub(msg.len() + 2);
        if remaining > 0 && !summary.is_empty() {
            msg.push_str("\n\n");
            msg.push_str(crate::utils::text::safe_truncate(summary, remaining));
        }
        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_analysis() -> SchemaAnalysis {
        SchemaAnalysis {
            entity_types: vec![("Person".to_string(), 10), ("Organization".to_string(), 3)],
            relation_types: vec!["works_for".to_string()],
            property_names: vec!["name".to_string(), "age".to_string()],
            total_entities: 13,
            total_relations: 5,
        }
    }

    #[test]
    fn test_generate_from_schema() {
        let analysis = make_analysis();
        let routing = ResultRoutingIdentity::new("l1-session-one", "call_1");
        let tools = MicroToolGenerator::generate_from_schema(&analysis, &routing, 5);

        assert!(tools.len() >= 2);
        assert!(tools.iter().any(|t| t.name == routing.query_name("Person")));
        assert!(tools
            .iter()
            .any(|t| t.name == routing.query_name("Organization")));
        assert!(tools
            .iter()
            .any(|t| t.name == routing.entity_details_name()));
        assert!(tools
            .iter()
            .any(|t| t.name == routing.relation_expansion_name()));
        assert!(tools.iter().all(|tool| tool.name.len() <= 64));
        assert!(tools.iter().all(|tool| tool
            .name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))));
    }

    #[test]
    fn test_generate_respects_max_tools() {
        let analysis = make_analysis();
        let routing = ResultRoutingIdentity::new("l1-session-two", "call_2");
        let tools = MicroToolGenerator::generate_from_schema(&analysis, &routing, 2);

        assert!(tools.len() <= 2);
    }

    #[test]
    fn test_generate_read_full_tool() {
        let routing = ResultRoutingIdentity::new("l1-session-three", "call_3");
        let tool = MicroToolGenerator::generate_read_full_tool(&routing, "storage_key_1", 2000);

        assert_eq!(tool.name, routing.reader_name);
        assert!(tool.description.contains("2000"));
        assert!(matches!(tool.tool_type, MicroToolType::FullTextRead { .. }));
    }

    #[test]
    fn archived_execution_view_accepts_all_supported_execution_envelopes() {
        let shell = json!({
            "command": "printf ok",
            "stdout": "ok",
            "stderr": "",
            "exit_code": 0,
        })
        .to_string();
        let code = json!({
            "stdout": "ok",
            "stderr": "",
            "exit_code": 0,
        })
        .to_string();

        for tool_name in ["bash", "powershell"] {
            assert_eq!(
                ArchivedReaderView::for_result(tool_name, &shell),
                ArchivedReaderView::ExecutionStream
            );
        }
        assert_eq!(
            ArchivedReaderView::for_result("code_execute", &code),
            ArchivedReaderView::ExecutionStream
        );
        assert_eq!(
            ArchivedReaderView::for_result("code_execute", r#"{"stdout":"ok"}"#),
            ArchivedReaderView::RawText
        );
    }

    #[test]
    fn test_format_injection_message() {
        let analysis = make_analysis();
        let routing = ResultRoutingIdentity::new("l1-session-four", "call_4");
        let tools = MicroToolGenerator::generate_from_schema(&analysis, &routing, 5);
        let msg = MicroToolGenerator::format_tool_injection_message("Test summary", &tools);

        assert!(msg.contains("Test summary"));
        assert!(msg.contains("Available session-scoped graph tools:"));
        assert!(tools.iter().all(|tool| msg.contains(&tool.name)));
        assert!(msg.contains("only while their exact names appear"));
        assert!(!msg.contains("read_full_result_*"));
        assert!(!msg.contains("query_*"));
    }
}
