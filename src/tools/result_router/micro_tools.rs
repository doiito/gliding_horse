use serde_json::json;

use super::{MicroToolSchema, MicroToolType, SchemaAnalysis};

pub struct MicroToolGenerator;

impl MicroToolGenerator {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_from_schema(
        analysis: &SchemaAnalysis,
        call_id: &str,
        max_tools: usize,
    ) -> Vec<MicroToolSchema> {
        let graph_name = format!("graph:tool-result:{}", call_id);
        let mut tools = Vec::new();

        for (type_name, count) in &analysis.entity_types {
            if tools.len() >= max_tools {
                break;
            }
            let short_name = type_name.split('/').last().unwrap_or(type_name);
            let tool_name = format!("query_{}", short_name.to_lowercase());

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
                            "type": "string",
                            "description": "Filter value"
                        },
                        "limit": {
                            "type": "integer",
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
                name: "get_entity_details".to_string(),
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
                name: "expand_relation".to_string(),
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
        _call_id: &str,
        storage_key: &str,
        preview_size: usize,
    ) -> MicroToolSchema {
        MicroToolSchema {
            name: "read_full_result".to_string(),
            description: format!(
                "Read full tool result (preview shows first {} characters)",
                preview_size
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "offset": {
                        "type": "integer",
                        "description": "Starting offset (characters)",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Read length (characters)",
                        "default": 4000
                    }
                }
            }),
            tool_type: MicroToolType::FullTextRead {
                storage_key: storage_key.to_string(),
            },
        }
    }

    pub fn format_tool_injection_message(summary: &str, tools: &[MicroToolSchema]) -> String {
        let mut msg = format!("{}\n\nAvailable query tools:\n", summary);
        for tool in tools {
            msg.push_str(&format!("- **{}**: {}\n", tool.name, tool.description));
        }
        msg.push_str("\nYou can call these tools to query the complete data.");
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
        let tools = MicroToolGenerator::generate_from_schema(&analysis, "call_1", 5);

        assert!(tools.len() >= 2);
        assert!(tools.iter().any(|t| t.name == "query_person"));
        assert!(tools.iter().any(|t| t.name == "query_organization"));
        assert!(tools.iter().any(|t| t.name == "get_entity_details"));
        assert!(tools.iter().any(|t| t.name == "expand_relation"));
    }

    #[test]
    fn test_generate_respects_max_tools() {
        let analysis = make_analysis();
        let tools = MicroToolGenerator::generate_from_schema(&analysis, "call_2", 2);

        assert!(tools.len() <= 2);
    }

    #[test]
    fn test_generate_read_full_tool() {
        let tool = MicroToolGenerator::generate_read_full_tool("call_3", "storage_key_1", 2000);

        assert_eq!(tool.name, "read_full_result");
        assert!(tool.description.contains("2000"));
        assert!(matches!(tool.tool_type, MicroToolType::FullTextRead { .. }));
    }

    #[test]
    fn test_format_injection_message() {
        let analysis = make_analysis();
        let tools = MicroToolGenerator::generate_from_schema(&analysis, "call_4", 5);
        let msg = MicroToolGenerator::format_tool_injection_message("Test summary", &tools);

        assert!(msg.contains("Test summary"));
        assert!(msg.contains("query_person"));
        assert!(msg.contains("query tools"));
    }
}
