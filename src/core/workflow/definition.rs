//! JSON-LD workflow definition types
//!
//! Defines DAG workflow structures editable in the Web UI and persisted in JSON-LD format.
//! Unlike petgraph DiGraph, these are deserialized "blueprint" types.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Conditional branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchCondition {
    /// Condition expression, e.g., "$.result.status == 'failed'"
    pub condition: String,
    /// Target node ID to jump to when condition is met
    pub target: String,
}

/// Input mapping: JSONPath expression → context key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMapping {
    /// JSONPath mapping table, e.g., { "prev_summary": "$.nodes.step_1.summary" }
    #[serde(default)]
    pub mappings: HashMap<String, String>,
    /// Context template, referencing keys from mappings
    #[serde(default)]
    pub context_template: String,
}

/// Single Agent node in a workflow (JSON-LD serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNodeDef {
    /// Unique node ID, corresponding to json-ld @id
    #[serde(rename = "@id")]
    pub id: String,
    /// Node type
    #[serde(rename = "@type", default = "default_node_type")]
    pub node_type: String,
    /// Agent role
    pub agent_role: String,
    /// Node objective description
    #[serde(default)]
    pub objective: String,
    /// Next node ID (single step)
    #[serde(default)]
    pub next: Option<String>,
    /// Fork: multiple next node IDs (parallel)
    #[serde(default)]
    pub next_nodes: Vec<String>,
    /// Dependency node IDs (prerequisites)
    #[serde(default)]
    pub dependencies: Vec<String>,
    /// Allowed tools list
    #[serde(default)]
    pub tools: Vec<String>,
    /// Expected output (corresponds to PlanStep.expected_output)
    #[serde(default)]
    pub expected_output: String,
    /// Success criteria
    #[serde(default)]
    pub success_criteria: String,
    /// Input mapping
    #[serde(default)]
    pub input_mapping: Option<InputMapping>,
    /// Branch condition (jump on failure)
    #[serde(default)]
    pub branch_on_failure: Option<BranchCondition>,
    /// Retry count
    #[serde(default)]
    pub retry_count: u32,
    /// Retry delay in seconds
    #[serde(default)]
    pub retry_delay_secs: u64,
    /// Node timeout in seconds. Zero inherits the Supervisor's configured
    /// per-agent default; a positive value overrides it for this node.
    #[serde(default)]
    pub timeout_secs: u64,
    /// Whether this is a final node
    #[serde(default)]
    pub final_node: bool,
    /// Human approval prompt (for HumanApprovalNode)
    #[serde(default)]
    pub approval_prompt: String,
    /// Node ID to jump to after approval (optional, defaults to next/next_nodes)
    #[serde(default)]
    pub approval_next_on_approve: Option<String>,
    /// Node ID to jump to after rejection (optional, defaults to stopping subsequent execution)
    #[serde(default)]
    pub approval_next_on_reject: Option<String>,
    /// Custom attributes
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

fn default_node_type() -> String {
    "AgentNode".to_string()
}

/// Full workflow definition (JSON-LD @graph format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    /// Workflow IRI
    #[serde(rename = "@id")]
    pub id: String,
    /// Workflow name
    #[serde(default)]
    pub name: String,
    /// Description
    #[serde(default)]
    pub description: String,
    /// Version
    #[serde(default)]
    pub version: String,
    /// Entry node ID
    pub entry_node: String,
    /// All node definitions
    pub nodes: Vec<WorkflowNodeDef>,
}

/// JSON-LD container (parsing @graph array)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowContainer {
    #[serde(rename = "@context", default)]
    pub context: Value,
    #[serde(default)]
    pub graph: Vec<Value>,
}

/// Node execution result (in-memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResult {
    pub node_id: String,
    pub status: String,
    pub summary: String,
    pub archive_iri: Option<String>,
    pub turn_count: u32,
    pub tool_call_count: u32,
    pub error: Option<String>,
    pub output: Option<Value>,
    pub artifacts: Vec<Value>,
}
