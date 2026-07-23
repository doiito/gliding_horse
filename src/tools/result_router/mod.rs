pub mod graphify;
pub mod micro_tools;
pub mod router;
pub mod summary;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum RouteDecision {
    PassThrough,
    Truncate {
        max_chars: usize,
    },
    Graphify {
        call_id: String,
        graph_name: String,
    },
    Summarize {
        call_id: String,
        preview_size: usize,
    },
}

#[derive(Debug, Clone)]
pub struct ToolResultMeta {
    pub tool_name: String,
    pub call_id: String,
    pub size_bytes: usize,
    pub is_json: bool,
    pub is_structured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub tool_type: MicroToolType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MicroToolType {
    EntityTypeQuery {
        entity_type: String,
        graph_name: String,
    },
    EntityDetails {
        graph_name: String,
    },
    RelationTraversal {
        graph_name: String,
    },
    FullTextRead {
        storage_key: String,
    },
}

#[derive(Debug, Clone)]
pub struct GraphifyResult {
    pub graph_name: String,
    pub entity_count: usize,
    pub relation_count: usize,
    pub entity_types: Vec<String>,
    pub summary: String,
    pub micro_tools: Vec<MicroToolSchema>,
}

#[derive(Debug, Clone)]
pub struct SchemaAnalysis {
    pub entity_types: Vec<(String, usize)>,
    pub relation_types: Vec<String>,
    pub property_names: Vec<String>,
    pub total_entities: usize,
    pub total_relations: usize,
}
