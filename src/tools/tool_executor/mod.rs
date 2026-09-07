use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::knowledge_graph::store::KnowledgeGraphStore;
use crate::memory::hyperspace_store::HyperspaceStore;
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::skill_graph::security::{SecurityContext, SecurityDecision, SecurityEngine};
use crate::tools::builtin::hooks::HookRunner;
use crate::tools::builtin::knowledge;
#[cfg(feature = "ontology")]
use crate::tools::builtin::ontology_tools;
use crate::tools::builtin::permissions::{
    PermissionContext, PermissionMode, PermissionOutcome, PermissionOverride, PermissionPolicy,
};
use crate::tools::builtin::rag;
use crate::tools::skill_registry::SkillRegistry;
use crate::tools::tool_groups::ToolGroupManager;
use crate::tools::workspace_monitor::{FileState, WorkspaceMonitor};

mod builtins;

#[cfg(test)]
mod tests;

/// Kernel-owned execution semantics which must never be accepted from model
/// tool arguments or an external Hook. The AgentRunner selects one of the
/// clean verification profiles only after its attributable-command parser has
/// accepted the final SkillBefore arguments; ToolExecutor injects the marker
/// after all remaining argument-facing trust boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToolExecutionProfile {
    #[default]
    Standard,
    CleanPythonVerification,
    CleanPytestVerification,
    CleanMermaidVerification,
}

impl ToolExecutionProfile {
    const fn is_clean_verification(self) -> bool {
        matches!(
            self,
            Self::CleanPythonVerification
                | Self::CleanPytestVerification
                | Self::CleanMermaidVerification
        )
    }

    pub(crate) const fn is_python_verification(self) -> bool {
        matches!(
            self,
            Self::CleanPythonVerification | Self::CleanPytestVerification
        )
    }
}

/// Tool input structs
#[derive(Debug, Deserialize)]
pub struct GlobSearchInput {
    pub pattern: String,
    pub path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GrepSearchInput {
    pub pattern: String,
    pub path: Option<String>,
    pub glob: Option<String>,
    pub output_mode: Option<String>,
    pub before: Option<usize>,
    pub after: Option<usize>,
    pub context: Option<usize>,
    pub line_numbers: Option<bool>,
    pub head_limit: Option<usize>,
    pub offset: Option<usize>,
    #[serde(rename = "-i")]
    pub case_insensitive: Option<bool>,
    pub multiline: Option<bool>,
    pub file_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WebFetchInput {
    pub url: String,
    pub prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WebSearchInput {
    pub query: String,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct ToolSearchInput {
    pub query: String,
    pub max_results: Option<usize>,
}
pub type ToolFn =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> + Send + Sync>;
type ContextualToolFn = Arc<
    dyn Fn(
            Value,
            Option<crate::llm::LlmInteractionScope>,
        ) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>>
        + Send
        + Sync,
>;

/// These exact names have kernel-owned execution and evidence semantics.
/// An external handler may use a namespaced spelling, but can never acquire
/// one of these identities or replace its schema/role metadata.
const EVIDENCE_CRITICAL_BUILTIN_NAMES: &[&str] = &[
    "file_read",
    "file_write",
    "file_edit",
    "bash",
    "powershell",
    "code_execute",
    "jsonld_validate",
    "ontology_validate_turtle",
    "ontology_validate_shacl",
    "ontology_lint_turtle",
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolProvenance {
    Builtin,
    TrustedInternal,
    External { namespace: String },
}

impl ToolProvenance {
    fn label(&self) -> &str {
        match self {
            Self::Builtin => "builtin",
            Self::TrustedInternal => "trusted_internal",
            Self::External { namespace } => namespace,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolRegistrationError {
    InvalidName {
        name: String,
    },
    InvalidNamespace {
        namespace: String,
    },
    ReservedBuiltinName {
        name: String,
    },
    NameConflict {
        name: String,
        existing_provenance: String,
    },
}

impl std::fmt::Display for ToolRegistrationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidName { name } => {
                write!(formatter, "external tool has an invalid name: {name:?}")
            }
            Self::InvalidNamespace { namespace } => write!(
                formatter,
                "external tool has an invalid provenance namespace: {namespace:?}"
            ),
            Self::ReservedBuiltinName { name } => write!(
                formatter,
                "external tool cannot claim reserved built-in name '{name}'"
            ),
            Self::NameConflict {
                name,
                existing_provenance,
            } => write!(
                formatter,
                "external tool name '{name}' is already owned by {existing_provenance}"
            ),
        }
    }
}

impl std::error::Error for ToolRegistrationError {}

/// Wrap a synchronous tool function (takes &Value) as an async ToolFn
fn sync_tool_ref<F>(f: F) -> ToolFn
where
    F: Fn(&Value) -> Result<Value, String> + Send + Sync + 'static,
{
    let f = Arc::new(f);
    Arc::new(move |input| {
        let f = Arc::clone(&f);

        Box::pin(async move { f(&input) })
    })
}

fn background_workspace_mutation_rejection(name: &str, input: &Value) -> Option<Value> {
    (input.get("run_in_background").and_then(Value::as_bool) == Some(true)
        && (matches!(name, "bash" | "powershell" | "code_execute")
            || crate::core::effect::is_workspace_mutation_candidate(name, input)))
    .then(|| {
        json!({
            "error": format!(
                "Background workspace mutation rejected: '{name}' cannot settle its workspace delta before returning"
            ),
            "tool": name,
            "run_in_background": true,
            "reason": "background_workspace_mutation_unsettled",
        })
    })
}

/// Validate a filesystem argument against the executor's current workspace
/// and replace it with its resolved path before handing it to a read handler.
/// This keeps every registered knowledge reader on the same fail-closed path
/// policy as the built-in file tools and avoids re-following a caller-supplied
/// symlink alias in the downstream implementation.
fn resolve_workspace_path_argument(mut input: Value, field: &str) -> Result<Value, String> {
    let supplied = input
        .get(field)
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| format!("{field} parameter cannot be empty"))?;
    let (_, resolved) = builtins::resolve_path_in_workspace(supplied)?;
    let resolved = resolved
        .to_str()
        .ok_or_else(|| format!("Resolved {field} path is not valid UTF-8"))?
        .to_string();
    let object = input
        .as_object_mut()
        .ok_or_else(|| "Tool input must be a JSON object".to_string())?;
    object.insert(field.to_string(), Value::String(resolved));
    Ok(input)
}

/// Micro-tool context
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MicroToolContext {
    /// Session-scoped internal key. Never use the provider call ID as a
    /// process-wide lookup key because providers routinely reuse values such
    /// as `call_0` across independent AgentInstances.
    pub routing_call_key: String,
    /// Original provider identifier retained only for observability and the
    /// bounded reader response.
    pub provider_call_id: String,
    pub storage_key: String,
    pub tool_name: String,
    pub entity_types: Vec<String>,
    pub preview_size: usize,
}

/// Kernel-owned delivery progress for a session-scoped archived-result
/// reader.  A reader name already contains the Agent/L1/request/call routing
/// identity, so keeping this state by exact name preserves fresh-Agent
/// isolation even when `ToolExecutor` clones share the underlying registry.
#[derive(Debug, Default)]
struct ArchivedReaderConsumption {
    views: HashMap<String, ArchivedReaderViewConsumption>,
    retired: bool,
}

#[derive(Debug)]
struct ArchivedReaderViewConsumption {
    progress: ArchivedReaderProgress,
    completed_receipt: Option<Value>,
}

#[derive(Debug)]
enum ArchivedReaderProgress {
    Characters {
        contiguous_end: usize,
        /// The initial routed preview advances the delivery cursor before the
        /// session reader is ever called. Provider-history compression may
        /// later remove that preview, so one explicit restart from zero must
        /// be allowed to replay it. Pages returned by the reader itself do
        /// not receive this allowance.
        seeded_from_preview: bool,
        preview_replay_used: bool,
    },
    FileLines {
        started: bool,
        next_offset: usize,
        next_char_offset: usize,
        /// A character cursor is relative to the exact selected line page,
        /// so its limit must remain stable until that page is exhausted.
        continuation_limit: Option<usize>,
        seeded_from_preview: bool,
        preview_replay_used: bool,
    },
}

/// Unified tool executor with built-in tools
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkspaceSettlementStamp {
    pub coordinator_id: String,
    pub settlement_sequence: u64,
    pub mutation_epoch: u64,
    pub manifest_sha256: Option<String>,
    pub manifest_drift_observed: bool,
}

#[derive(Debug)]
pub(crate) struct WorkspaceMutationCoordinatorState {
    coordinator_id: String,
    settlement_sequence: u64,
    mutation_epoch: u64,
    current_manifest_sha256: Option<String>,
}

impl WorkspaceMutationCoordinatorState {
    fn new() -> Self {
        Self {
            coordinator_id: format!(
                "workspace-coordinator:{}",
                uuid::Uuid::new_v4().hyphenated()
            ),
            settlement_sequence: 0,
            mutation_epoch: 0,
            current_manifest_sha256: None,
        }
    }

    /// Finalize one action while the runner still owns the workspace gate.
    /// The returned stamp is kernel-owned and therefore cannot be supplied by
    /// model arguments or post-execution hook output.
    pub(crate) fn settle_action(
        &mut self,
        invalidates_verification: bool,
        manifest_sha256: Option<&str>,
    ) -> WorkspaceSettlementStamp {
        let valid_manifest_sha256 = manifest_sha256
            .filter(|digest| {
                digest.len() == 71
                    && digest.starts_with("sha256:")
                    && digest[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            .map(str::to_ascii_lowercase);
        let manifest_drift_observed = !invalidates_verification
            && self.current_manifest_sha256.is_some()
            && valid_manifest_sha256.is_some()
            && self.current_manifest_sha256 != valid_manifest_sha256;
        self.settlement_sequence = self
            .settlement_sequence
            .checked_add(1)
            .expect("workspace settlement sequence exhausted");
        if invalidates_verification || manifest_drift_observed {
            self.mutation_epoch = self
                .mutation_epoch
                .checked_add(1)
                .expect("workspace mutation epoch exhausted");
        }
        if let Some(digest) = valid_manifest_sha256.as_ref() {
            self.current_manifest_sha256 = Some(digest.clone());
        }
        WorkspaceSettlementStamp {
            coordinator_id: self.coordinator_id.clone(),
            settlement_sequence: self.settlement_sequence,
            mutation_epoch: self.mutation_epoch,
            manifest_sha256: valid_manifest_sha256,
            manifest_drift_observed,
        }
    }
}

#[derive(Clone)]
pub struct ToolExecutor {
    tools: HashMap<String, ToolFn>,
    /// Internal handlers that accept kernel-issued invocation metadata. Only
    /// crate-trusted registration may replace one; external registration is
    /// provenance checked and never reaches this map on collision.
    contextual_tools: HashMap<String, ContextualToolFn>,
    tool_provenance: HashMap<String, ToolProvenance>,
    registering_builtins: bool,
    tool_descriptions: Vec<ToolDescription>,
    kg_store: Arc<std::sync::RwLock<KnowledgeGraphStore>>,
    projection_engine:
        Arc<parking_lot::RwLock<Option<Arc<crate::memory::l3_projection::ProjectionEngine>>>>,
    micro_tool_contexts: Arc<parking_lot::RwLock<HashMap<String, MicroToolContext>>>,
    micro_tool_data: Arc<parking_lot::RwLock<HashMap<String, serde_json::Value>>>,
    /// Delivery state is shared across executor clones but remains scoped by
    /// the composite reader name.  It prevents a successful whole-result read
    /// from being replayed into the same Agent/L1 context while leaving other
    /// fresh AgentInstances completely independent.
    archived_reader_consumption:
        Arc<parking_lot::RwLock<HashMap<String, ArchivedReaderConsumption>>>,
    syscall_gate: Option<crate::core::syscall_gate::SyscallGate>,
    permission_policy: Option<PermissionPolicy>,
    hook_runner: Option<HookRunner>,
    /// External hook output is advisory by default. Replacing the effective
    /// tool input requires an explicit, runner-owned configuration switch.
    hook_allow_input_rewrite: bool,
    tool_group_manager: Option<ToolGroupManager>,
    workspace_monitor: Arc<parking_lot::RwLock<Option<Arc<WorkspaceMonitor>>>>,
    shared_skill_graph: Arc<parking_lot::RwLock<Option<Arc<SkillGraphStore>>>>,
    shared_skill_registry: Arc<parking_lot::RwLock<Option<Arc<SkillRegistry>>>>,
    shared_skill_vector_store: Arc<parking_lot::RwLock<Option<Arc<HyperspaceStore>>>>,
    /// Runner-owned LLM interaction plane for dynamic skill generation. This
    /// is deliberately executor-local: hooks, events and accounting must not
    /// fall back to mutable process-global task state.
    shared_skill_creator_interactions:
        Arc<parking_lot::RwLock<Option<Arc<crate::llm::LlmInteractionService>>>>,
    shared_skill_creator_gateway:
        Arc<parking_lot::RwLock<Option<Arc<crate::gateway::unified_gateway::UnifiedGateway>>>>,
    security_engine: Arc<parking_lot::RwLock<Option<Arc<SecurityEngine>>>>,
    /// Whole-file reads whose content has been returned to one Agent/L1
    /// execution context. The WorkspaceMonitor byte cache is shared by
    /// PA/DA/CA, but their LLM transcripts are not; a global cache hit
    /// therefore cannot imply that another Agent or L1 has the content.
    file_read_exposures: Arc<parking_lot::RwLock<HashSet<String>>>,
    /// One runner-owned mutation settlement window spans the pre-snapshot,
    /// shell execution, and post-snapshot phases. Clones must coordinate on
    /// the same lock, but `execute_internal` deliberately never acquires it:
    /// the runner owns the wider critical section and double-locking here
    /// would deadlock.
    workspace_mutation_coordinator: Arc<tokio::sync::Mutex<WorkspaceMutationCoordinatorState>>,
    max_micro_tool_descriptions: usize,
    micro_tool_page_size: usize,
    micro_tool_max_page_size: usize,
}

fn micro_tool_parameters(
    name: &str,
    reader_view: crate::tools::result_router::micro_tools::ArchivedReaderView,
    max_line_limit: usize,
    max_char_limit: usize,
) -> Value {
    if name.starts_with("query_") {
        json!({
            "type": "object",
            "properties": {
                "filter_property": {"type": "string", "description": "Optional exact property name"},
                "filter_value": {"description": "Optional exact JSON value for filter_property"},
                "offset": {"type": "integer", "minimum": 0, "description": "Matching-result offset"},
                "limit": {"type": "integer", "minimum": 1, "description": "Maximum matching results"}
            }
        })
    } else if name.starts_with("get_entity_details_") {
        json!({
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Exact entity id"},
                "char_offset": {"type": "integer", "minimum": 0, "description": "Character offset for a large serialized entity"}
            },
            "required": ["entity_id"]
        })
    } else if name.starts_with("expand_relation_") {
        json!({
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "relation": {"type": "string"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 3}
            },
            "required": ["entity_id"]
        })
    } else {
        crate::tools::result_router::micro_tools::archived_reader_parameters(
            reader_view,
            max_line_limit,
            max_char_limit,
        )
    }
}

fn archived_reader_view(
    context: &MicroToolContext,
    stored_data: Option<&Value>,
) -> crate::tools::result_router::micro_tools::ArchivedReaderView {
    stored_data
        .and_then(|data| data.get("content"))
        .and_then(Value::as_str)
        .map(|content| {
            crate::tools::result_router::micro_tools::ArchivedReaderView::for_result(
                &context.tool_name,
                content,
            )
        })
        .unwrap_or(crate::tools::result_router::micro_tools::ArchivedReaderView::RawText)
}

fn archived_item_matches_entity_type(item: &Value, expected_type: &str) -> bool {
    let expected_short = expected_type.rsplit('/').next().unwrap_or(expected_type);
    let actual = item.as_object().and_then(|object| {
        ["@type", "type", "kind", "category"]
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_str))
    });
    match actual {
        Some(actual) => {
            actual == expected_type || actual.rsplit('/').next() == Some(expected_short)
        }
        None => expected_short == "Entity",
    }
}

fn archived_item_matches_property(item: &Value, property: &str, expected: Option<&Value>) -> bool {
    if property.is_empty() {
        return true;
    }
    let Some(actual) = item.get(property) else {
        return false;
    };
    expected.map(|expected| actual == expected).unwrap_or(true)
}

fn archived_item_matches_id(item: &Value, expected: &str) -> bool {
    ["id", "ID", "_id", "uid", "key"].iter().any(|key| {
        item.get(*key).is_some_and(|value| match value {
            Value::String(value) => value == expected,
            Value::Number(value) => value.to_string() == expected,
            Value::Bool(value) => value.to_string() == expected,
            _ => false,
        })
    })
}

pub(crate) fn sanitize_session_tool_references(text: &str) -> (String, usize) {
    const PATTERNS: &[(&str, &str)] = &[
        (
            "read_full_result_",
            "[session-scoped result reader omitted]",
        ),
        ("iri://tool-result/", "[session-scoped tool result omitted]"),
        (
            "https://agent-os.org/ontology/tool-result/",
            "[session-scoped tool result omitted]",
        ),
        (
            "graph:tool-result:",
            "[session-scoped result graph omitted]",
        ),
    ];

    let mut sanitized = text.to_string();
    let mut total = 0;
    for (prefix, replacement) in PATTERNS {
        let mut remaining = sanitized.as_str();
        let mut output = String::with_capacity(sanitized.len());
        let mut redacted = 0;
        while let Some(start) = remaining.find(prefix) {
            output.push_str(&remaining[..start]);
            let token_start = start + prefix.len();
            if remaining[token_start..].starts_with('*') {
                output.push_str(replacement);
                remaining = &remaining[token_start + 1..];
                redacted += 1;
                continue;
            }
            let token_len = remaining[token_start..]
                .bytes()
                .take_while(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')
                })
                .count();
            if token_len == 0 {
                output.push_str(prefix);
                remaining = &remaining[token_start..];
                continue;
            }
            output.push_str(replacement);
            remaining = &remaining[token_start + token_len..];
            redacted += 1;
        }
        output.push_str(remaining);
        sanitized = output;
        total += redacted;
    }
    for prefix in ["query_", "get_entity_details_", "expand_relation_"] {
        let mut remaining = sanitized.as_str();
        let mut output = String::with_capacity(sanitized.len());
        let mut redacted = 0;
        while let Some(start) = remaining.find(prefix) {
            output.push_str(&remaining[..start]);
            let token_len = remaining[start..]
                .bytes()
                .take_while(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                .count();
            let token = &remaining[start..start + token_len];
            if crate::tools::result_router::is_session_scoped_micro_tool_name(token) {
                output.push_str("[session-scoped result tool omitted]");
                redacted += 1;
            } else {
                output.push_str(token);
            }
            remaining = &remaining[start + token_len..];
        }
        output.push_str(remaining);
        sanitized = output;
        total += redacted;
    }
    // Internal routing keys can otherwise be used to deterministically
    // reconstruct every generated name. Match only the exact kernel grammar
    // to avoid redacting ordinary identifiers beginning with `s`.
    let mut remaining = sanitized.as_str();
    let mut output = String::with_capacity(sanitized.len());
    let mut routing_keys_redacted = 0usize;
    while !remaining.is_empty() {
        let token_start = remaining
            .bytes()
            .position(|byte| byte.is_ascii_alphanumeric())
            .unwrap_or(remaining.len());
        output.push_str(&remaining[..token_start]);
        if token_start == remaining.len() {
            break;
        }
        let token_len = remaining[token_start..]
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
            .count();
        let token = &remaining[token_start..token_start + token_len];
        if crate::tools::result_router::valid_routing_call_key(token) {
            output.push_str("[session-scoped routing key omitted]");
            routing_keys_redacted += 1;
        } else {
            output.push_str(token);
        }
        remaining = &remaining[token_start + token_len..];
    }
    sanitized = output;
    total += routing_keys_redacted;
    (sanitized, total)
}

fn redact_session_tool_references(value: &mut Value) -> usize {
    match value {
        Value::String(text) => {
            let (redacted_text, count) = sanitize_session_tool_references(text);
            *text = redacted_text;
            count
        }
        Value::Array(items) => items.iter_mut().map(redact_session_tool_references).sum(),
        Value::Object(object) => {
            let entries = std::mem::take(object);
            let mut redacted = 0usize;
            for (key, mut nested) in entries {
                let (mut key, key_count) = sanitize_session_tool_references(&key);
                redacted += key_count + redact_session_tool_references(&mut nested);
                if object.contains_key(&key) {
                    let base = key;
                    let mut suffix = 2usize;
                    key = format!("{base} [redacted key {suffix}]");
                    while object.contains_key(&key) {
                        suffix += 1;
                        key = format!("{base} [redacted key {suffix}]");
                    }
                }
                object.insert(key, nested);
            }
            redacted
        }
        _ => 0,
    }
}

/// Return a detached, recursively sanitized copy suitable for crossing an
/// agent/session boundary. Stable AgentTurn archive IRIs and the
/// `read_agent_output` capability are deliberately untouched; only ephemeral
/// result-reader names and tool-result IRIs are removed.
pub(crate) fn sanitized_session_handoff_value(value: &Value) -> (Value, usize) {
    let mut sanitized = value.clone();
    let redacted = redact_session_tool_references(&mut sanitized);
    (sanitized, redacted)
}

fn agent_turn_content_page(node: &mut Value, input: &Value, node_iri: &str) -> Option<Value> {
    let redacted = redact_session_tool_references(node);
    let content = node.get("content")?.as_str()?;
    let char_offset = input
        .get("char_offset")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let char_limit = input
        .get("char_limit")
        .and_then(Value::as_u64)
        .unwrap_or(4_000)
        .clamp(1, 6_000) as usize;
    let total_chars = content.chars().count();
    let page = content
        .chars()
        .skip(char_offset)
        .take(char_limit)
        .collect::<String>();
    let next_char_offset = char_offset.saturating_add(page.chars().count());
    Some(json!({
        "content": page,
        "total_chars": total_chars,
        "char_offset": char_offset,
        "returned_chars": next_char_offset.saturating_sub(char_offset),
        "next_char_offset": (next_char_offset < total_chars).then_some(next_char_offset),
        "iri": node_iri,
        "role": node.get("role").cloned().unwrap_or(Value::Null),
        "cycle_id": node.get("cycle_id").cloned().unwrap_or(Value::Null),
        "session_tool_references_redacted": redacted,
    }))
}

fn reader_usize_argument(input: &Value, name: &str) -> Result<Option<usize>, String> {
    let Some(value) = input.get(name) else {
        return Ok(None);
    };
    let value = value
        .as_u64()
        .ok_or_else(|| format!("archived reader {name} must be a non-negative integer"))?;
    usize::try_from(value)
        .map(Some)
        .map_err(|_| format!("archived reader {name} is too large for this platform"))
}

fn validate_reader_arguments(input: &Value, allowed: &[&str]) -> Result<(), String> {
    let object = input
        .as_object()
        .ok_or_else(|| "archived reader input must be a JSON object".to_string())?;
    if let Some(name) = object.keys().find(|name| !allowed.contains(&name.as_str())) {
        return Err(format!(
            "archived reader argument '{name}' is not valid for this result view"
        ));
    }
    Ok(())
}

fn checked_character_page(
    content: &str,
    char_offset: usize,
    requested_limit: usize,
    max_chars: usize,
) -> Result<(String, usize, Option<usize>), String> {
    let total_chars = content.chars().count();
    // An exact EOF cursor is an idempotent completion probe, not malformed
    // input.  Only a cursor beyond EOF is invalid.  This also makes an empty
    // archived stream readable at its sole valid cursor (zero).
    if char_offset > total_chars {
        return Err(format!(
            "archived reader char_offset {char_offset} is outside content range 0..{total_chars}"
        ));
    }
    let limit = requested_limit.max(1).min(max_chars.max(1));
    let page = content
        .chars()
        .skip(char_offset)
        .take(limit)
        .collect::<String>();
    let returned_chars = page.chars().count();
    let consumed_chars = char_offset.saturating_add(returned_chars);
    let next = (consumed_chars < total_chars).then_some(consumed_chars);
    Ok((page, total_chars, next))
}

fn archived_raw_text_page(
    content: &str,
    input: &Value,
    max_chars: usize,
    provider_call_id: &str,
    routing_call_key: &str,
) -> Result<Value, String> {
    validate_reader_arguments(input, &["char_offset", "char_limit"])?;
    let char_offset = reader_usize_argument(input, "char_offset")?.unwrap_or(0);
    let char_limit = reader_usize_argument(input, "char_limit")?.unwrap_or(max_chars.max(1));
    if char_limit == 0 {
        return Err("archived reader char_limit must be greater than zero".to_string());
    }
    let (page, total_chars, next_char_offset) =
        checked_character_page(content, char_offset, char_limit, max_chars)?;
    let returned_chars = page.chars().count();
    Ok(json!({
        "reader_view": "raw_text",
        "content": page,
        "total_chars": total_chars,
        "char_offset": char_offset,
        "returned_chars": returned_chars,
        "next_char_offset": next_char_offset,
        "next_cursor": next_char_offset.map(|offset| json!({"char_offset": offset})),
        "truncated": next_char_offset.is_some(),
        "complete": next_char_offset.is_none(),
        "call_id": provider_call_id,
        "routing_call_key": routing_call_key,
    }))
}

fn archived_execution_stream_page(
    content: &str,
    input: &Value,
    max_chars: usize,
    provider_call_id: &str,
    routing_call_key: &str,
) -> Result<Value, String> {
    validate_reader_arguments(input, &["stream", "char_offset", "char_limit"])?;
    let stream = input
        .get("stream")
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| "archived reader stream must be a string".to_string())
        })
        .transpose()?
        .unwrap_or("stdout");
    if !matches!(stream, "stdout" | "stderr" | "command" | "raw") {
        return Err(format!(
            "archived reader stream '{stream}' is invalid; expected stdout, stderr, command, or raw"
        ));
    }
    let envelope = serde_json::from_str::<Value>(content)
        .map_err(|error| format!("archived execution envelope is invalid JSON: {error}"))?;
    let selected = if stream == "raw" {
        content
    } else if stream == "command" {
        envelope
            .get("command")
            .or_else(|| envelope.get("code"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                "archived execution envelope has no string field 'command' or 'code'".to_string()
            })?
    } else {
        envelope
            .get(stream)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("archived execution envelope has no string field '{stream}'"))?
    };
    let char_offset = reader_usize_argument(input, "char_offset")?.unwrap_or(0);
    let char_limit = reader_usize_argument(input, "char_limit")?.unwrap_or(max_chars.max(1));
    if char_limit == 0 {
        return Err("archived reader char_limit must be greater than zero".to_string());
    }
    let (page, total_chars, next_char_offset) =
        checked_character_page(selected, char_offset, char_limit, max_chars)?;
    let returned_chars = page.chars().count();
    Ok(json!({
        "reader_view": "execution_stream",
        "stream": stream,
        "content": page,
        "total_chars": total_chars,
        "char_offset": char_offset,
        "returned_chars": returned_chars,
        "next_char_offset": next_char_offset,
        "next_cursor": next_char_offset.map(|offset| json!({
            "stream": stream,
            "char_offset": offset,
        })),
        "truncated": next_char_offset.is_some(),
        "complete": next_char_offset.is_none(),
        "exit_code": envelope.get("exit_code").cloned().unwrap_or(Value::Null),
        "duration_ms": envelope.get("duration_ms").cloned().unwrap_or(Value::Null),
        "call_id": provider_call_id,
        "routing_call_key": routing_call_key,
    }))
}

fn archived_file_lines_page(
    content: &str,
    input: &Value,
    default_line_limit: usize,
    max_line_limit: usize,
    max_chars: usize,
    provider_call_id: &str,
    routing_call_key: &str,
) -> Result<Value, String> {
    validate_reader_arguments(input, &["offset", "limit", "char_offset"])?;
    let envelope = serde_json::from_str::<Value>(content)
        .map_err(|error| format!("archived file_read envelope is invalid JSON: {error}"))?;
    let lines = envelope
        .get("lines")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "archived file_read result has no line payload; issue a bounded file_read instead"
                .to_string()
        })?;
    let source_offset = envelope.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let archived_end = source_offset.saturating_add(lines.len());
    let line_offset = reader_usize_argument(input, "offset")?.unwrap_or(source_offset);
    // Like character readers, the exact end cursor is a valid, empty
    // completion page.  It is useful when a caller persists a cursor before
    // observing that the preceding page was terminal.
    if line_offset < source_offset || line_offset > archived_end {
        return Err(format!(
            "archived file cursor {line_offset} is outside archived source range {source_offset}..{archived_end}"
        ));
    }
    let line_limit = reader_usize_argument(input, "limit")?
        .unwrap_or(default_line_limit.max(1))
        .max(1)
        .min(max_line_limit.max(1));
    let char_offset = reader_usize_argument(input, "char_offset")?.unwrap_or(0);
    let local_offset = line_offset.saturating_sub(source_offset);
    let selected = lines
        .iter()
        .skip(local_offset)
        .take(line_limit)
        .map(|line| {
            line.as_str()
                .map(str::to_string)
                .unwrap_or_else(|| line.to_string())
        })
        .collect::<Vec<_>>();
    let selected_text = selected.join("\n");
    let (page, selected_chars, within_page_next) =
        checked_character_page(&selected_text, char_offset, max_chars.max(1), max_chars)?;
    let returned_chars = page.chars().count();
    let next_cursor = if let Some(next_char_offset) = within_page_next {
        Some(json!({
            "offset": line_offset,
            "limit": line_limit,
            "char_offset": next_char_offset,
        }))
    } else if local_offset.saturating_add(selected.len()) < lines.len() {
        Some(json!({
            "offset": line_offset.saturating_add(selected.len()),
            "limit": line_limit,
            "char_offset": 0,
        }))
    } else {
        None
    };
    let total_lines = envelope
        .get("total_lines")
        .and_then(Value::as_u64)
        .unwrap_or(archived_end as u64);
    let next_offset = next_cursor
        .as_ref()
        .and_then(|cursor| cursor.get("offset"))
        .cloned();
    let next_char_offset = next_cursor.as_ref().and_then(|cursor| {
        (cursor.get("offset").and_then(Value::as_u64) == Some(line_offset as u64))
            .then(|| cursor.get("char_offset").cloned())
            .flatten()
    });
    let truncated = next_cursor.is_some();
    Ok(json!({
        "reader_view": "file_lines",
        "path": envelope.get("path").cloned().unwrap_or(Value::Null),
        "content_sha256": envelope.get("content_sha256").cloned().unwrap_or(Value::Null),
        "content": page,
        "total_lines": total_lines,
        "archived_source_offset": source_offset,
        "archived_source_end": archived_end,
        "offset": line_offset,
        "limit": line_limit,
        "returned": selected.len(),
        "selected_lines": selected.len(),
        "char_offset": char_offset,
        "returned_chars": returned_chars,
        "selected_chars": selected_chars,
        "next_offset": next_offset,
        "next_char_offset": next_char_offset,
        "next_cursor": next_cursor,
        "source_has_more": archived_end < total_lines as usize,
        "truncated": truncated,
        "complete": !truncated,
        "call_id": provider_call_id,
        "routing_call_key": routing_call_key,
    }))
}

fn execute_archived_result_reader(
    input: &Value,
    context: &MicroToolContext,
    stored_data: &Value,
    default_line_limit: usize,
    max_line_limit: usize,
) -> Result<Value, String> {
    let content = stored_data
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            format!(
                "Micro-tool data has no text content: {}",
                context.storage_key
            )
        })?;
    let max_chars = context.preview_size.max(1);
    match crate::tools::result_router::micro_tools::ArchivedReaderView::for_result(
        &context.tool_name,
        content,
    ) {
        crate::tools::result_router::micro_tools::ArchivedReaderView::RawText => {
            archived_raw_text_page(
                content,
                input,
                max_chars,
                &context.provider_call_id,
                &context.routing_call_key,
            )
        }
        crate::tools::result_router::micro_tools::ArchivedReaderView::FileLines => {
            archived_file_lines_page(
                content,
                input,
                default_line_limit,
                max_line_limit,
                max_chars,
                &context.provider_call_id,
                &context.routing_call_key,
            )
        }
        crate::tools::result_router::micro_tools::ArchivedReaderView::ExecutionStream => {
            archived_execution_stream_page(
                content,
                input,
                max_chars,
                &context.provider_call_id,
                &context.routing_call_key,
            )
        }
    }
}

fn archived_reader_view_key(page: &Value) -> Option<String> {
    match page.get("reader_view").and_then(Value::as_str)? {
        "execution_stream" => page
            .get("stream")
            .and_then(Value::as_str)
            .map(|stream| format!("execution_stream:{stream}")),
        view @ ("raw_text" | "file_lines") => Some(view.to_string()),
        _ => None,
    }
}

fn archived_reader_completion_receipt(
    context: &MicroToolContext,
    archive_content: &str,
    page: &Value,
    view_key: &str,
) -> Value {
    let archive_sha256 = crate::utils::crypto::CryptoUtils::sha256_hex(archive_content);
    let receipt_material = format!(
        "{}\0{}\0{}",
        context.routing_call_key, view_key, archive_sha256
    );
    let mut receipt = json!({
        "schema_version": 1,
        "receipt_kind": "archived_result_consumed",
        "status": "already_consumed",
        "reader_view": page.get("reader_view").cloned().unwrap_or(Value::Null),
        "view_key": view_key,
        "content": "",
        "content_omitted": true,
        "complete": true,
        "truncated": false,
        "next_cursor": Value::Null,
        "archive_sha256": format!("sha256:{archive_sha256}"),
        "receipt_sha256": format!(
            "sha256:{}",
            crate::utils::crypto::CryptoUtils::sha256_hex(&receipt_material)
        ),
        // Keep the provider's raw identifier intact for observability.  The
        // composite routing key remains the authority and lookup identity.
        "call_id": context.provider_call_id,
        "routing_call_key": context.routing_call_key,
        "message": "This archived result view was already delivered completely to this Agent/L1 context; duplicate content is omitted. Use the receipt as evidence and do not call this reader again."
    });
    let Some(object) = receipt.as_object_mut() else {
        return receipt;
    };
    for field in [
        "stream",
        "path",
        "total_chars",
        "total_lines",
        "archived_source_offset",
        "archived_source_end",
        "source_has_more",
        "exit_code",
        "duration_ms",
    ] {
        if let Some(value) = page.get(field) {
            object.insert(field.to_string(), value.clone());
        }
    }
    receipt
}

/// Return a bounded control receipt when a caller asks for bytes/lines that
/// are already inside this reader's continuously delivered prefix.  The
/// receipt deliberately contains no archived body: it directs the caller to
/// the one cursor which can advance delivery without replaying context.
fn archived_reader_prefix_receipt(
    context: &MicroToolContext,
    page: &Value,
    view_key: &str,
    next_cursor: Value,
) -> Value {
    let cursor_material = serde_json::to_string(&next_cursor).unwrap_or_default();
    let receipt_material = format!(
        "{}\0{}\0{}",
        context.routing_call_key, view_key, cursor_material
    );
    let next_offset = next_cursor.get("offset").cloned().unwrap_or(Value::Null);
    let next_char_offset = next_cursor
        .get("char_offset")
        .cloned()
        .unwrap_or(Value::Null);
    let mut receipt = json!({
        "schema_version": 1,
        "receipt_kind": "archived_result_prefix_consumed",
        "status": "already_delivered_prefix",
        "reader_view": page.get("reader_view").cloned().unwrap_or(Value::Null),
        "view_key": view_key,
        "content": "",
        "content_omitted": true,
        "complete": false,
        "truncated": true,
        "next_offset": next_offset,
        "next_char_offset": next_char_offset,
        "next_cursor": next_cursor,
        "receipt_sha256": format!(
            "sha256:{}",
            crate::utils::crypto::CryptoUtils::sha256_hex(&receipt_material)
        ),
        // Preserve the raw provider identifier exactly; it remains
        // observability metadata and never becomes the registry key.
        "call_id": context.provider_call_id,
        "routing_call_key": context.routing_call_key,
        "message": "The requested range overlaps the prefix already delivered to this Agent/L1 context. Duplicate content was omitted; continue from next_cursor."
    });
    let Some(object) = receipt.as_object_mut() else {
        return receipt;
    };
    for field in [
        "stream",
        "path",
        "total_chars",
        "total_lines",
        "archived_source_offset",
        "archived_source_end",
        "source_has_more",
        "exit_code",
        "duration_ms",
    ] {
        if let Some(value) = page.get(field) {
            object.insert(field.to_string(), value.clone());
        }
    }
    receipt
}

/// Return a bounded correction receipt when a caller tries to skip over the
/// next continuously deliverable cursor. Unlike an overlapping-prefix
/// receipt, this does not claim that the skipped body was already delivered.
fn archived_reader_gap_receipt(
    context: &MicroToolContext,
    page: &Value,
    view_key: &str,
    next_cursor: Value,
) -> Value {
    let cursor_material = serde_json::to_string(&next_cursor).unwrap_or_default();
    let receipt_material = format!(
        "{}\0{}\0cursor_gap\0{}",
        context.routing_call_key, view_key, cursor_material
    );
    let mut receipt = archived_reader_prefix_receipt(context, page, view_key, next_cursor);
    if let Some(object) = receipt.as_object_mut() {
        object.insert(
            "receipt_kind".to_string(),
            Value::String("archived_result_cursor_required".to_string()),
        );
        object.insert(
            "status".to_string(),
            Value::String("cursor_gap_rejected".to_string()),
        );
        object.insert(
            "message".to_string(),
            Value::String(
                "The requested range skips archived content that has not been delivered to this Agent/L1 context. Duplicate-free delivery must continue from next_cursor."
                    .to_string(),
            ),
        );
        object.insert(
            "receipt_sha256".to_string(),
            Value::String(format!(
                "sha256:{}",
                crate::utils::crypto::CryptoUtils::sha256_hex(&receipt_material)
            )),
        );
    }
    receipt
}

fn archived_reader_view_is_completed(reader: &ArchivedReaderConsumption, view_key: &str) -> bool {
    reader
        .views
        .get(view_key)
        .and_then(|view| view.completed_receipt.as_ref())
        .is_some()
}

fn archived_reader_can_retire(
    reader: &ArchivedReaderConsumption,
    context: &MicroToolContext,
    archive_content: &str,
) -> bool {
    match crate::tools::result_router::micro_tools::ArchivedReaderView::for_result(
        &context.tool_name,
        archive_content,
    ) {
        crate::tools::result_router::micro_tools::ArchivedReaderView::RawText => {
            archived_reader_view_is_completed(reader, "raw_text")
        }
        crate::tools::result_router::micro_tools::ArchivedReaderView::FileLines => {
            archived_reader_view_is_completed(reader, "file_lines")
        }
        crate::tools::result_router::micro_tools::ArchivedReaderView::ExecutionStream => {
            // The raw view is the complete serialized execution envelope and
            // therefore subsumes every decoded stream. Once delivered from
            // start to finish there is nothing else this reader can reveal.
            if archived_reader_view_is_completed(reader, "execution_stream:raw") {
                return true;
            }
            let envelope = serde_json::from_str::<Value>(archive_content).ok();
            let mut required_streams = ["stdout", "stderr"]
                .into_iter()
                .filter(|stream| {
                    envelope
                        .as_ref()
                        .and_then(|value| value.get(*stream))
                        .and_then(Value::as_str)
                        .is_some_and(|content| !content.is_empty())
                })
                .collect::<Vec<_>>();
            // Even an execution with no textual output has meaningful exit
            // metadata.  Its default stdout page must be observed once.
            if required_streams.is_empty() {
                required_streams.push("stdout");
            }
            let baseline_complete = required_streams.into_iter().all(|stream| {
                archived_reader_view_is_completed(reader, &format!("execution_stream:{stream}"))
            });
            baseline_complete
                && reader.views.iter().all(|(view_key, view)| {
                    !view_key.starts_with("execution_stream:") || view.completed_receipt.is_some()
                })
        }
    }
}

/// Observe one successfully generated bounded page.  The first complete
/// delivery returns the real terminal page; later calls for that same logical
/// view receive only a content-addressed receipt.  Progress is intentionally
/// monotonic and bounded rather than retaining every requested interval.
fn record_archived_reader_delivery(
    registry: &parking_lot::RwLock<HashMap<String, ArchivedReaderConsumption>>,
    reader_name: &str,
    context: &MicroToolContext,
    archive_content: &str,
    page: Value,
) -> Value {
    let Some(view_key) = archived_reader_view_key(&page) else {
        return page;
    };
    let mut registry = registry.write();
    let reader = registry.entry(reader_name.to_string()).or_default();

    if let Some(receipt) = reader
        .views
        .get(&view_key)
        .and_then(|view| view.completed_receipt.clone())
    {
        return receipt;
    }

    let is_file_view = page.get("reader_view").and_then(Value::as_str) == Some("file_lines");
    let view =
        reader
            .views
            .entry(view_key.clone())
            .or_insert_with(|| ArchivedReaderViewConsumption {
                progress: if is_file_view {
                    ArchivedReaderProgress::FileLines {
                        started: false,
                        next_offset: page
                            .get("archived_source_offset")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as usize,
                        next_char_offset: 0,
                        continuation_limit: None,
                        seeded_from_preview: false,
                        preview_replay_used: false,
                    }
                } else {
                    ArchivedReaderProgress::Characters {
                        contiguous_end: 0,
                        seeded_from_preview: false,
                        preview_replay_used: false,
                    }
                },
                completed_receipt: None,
            });

    let completely_delivered = match &mut view.progress {
        ArchivedReaderProgress::Characters {
            contiguous_end,
            seeded_from_preview,
            preview_replay_used,
        } => {
            let char_offset = page.get("char_offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let returned_chars = page
                .get("returned_chars")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let total_chars = page.get("total_chars").and_then(Value::as_u64).unwrap_or(0) as usize;
            if char_offset < *contiguous_end {
                if *seeded_from_preview && !*preview_replay_used && char_offset == 0 {
                    // The caller explicitly asked to recover the initial
                    // preview. Return the real bounded page once without
                    // rewinding monotonic delivery progress.
                    *preview_replay_used = true;
                    return page;
                }
                let next_cursor = if page.get("reader_view").and_then(Value::as_str)
                    == Some("execution_stream")
                {
                    json!({
                        "stream": page.get("stream").cloned().unwrap_or_else(|| Value::String("stdout".to_string())),
                        "char_offset": *contiguous_end,
                    })
                } else {
                    json!({"char_offset": *contiguous_end})
                };
                return archived_reader_prefix_receipt(context, &page, &view_key, next_cursor);
            }
            if char_offset > *contiguous_end {
                let next_cursor = if page.get("reader_view").and_then(Value::as_str)
                    == Some("execution_stream")
                {
                    json!({
                        "stream": page.get("stream").cloned().unwrap_or_else(|| Value::String("stdout".to_string())),
                        "char_offset": *contiguous_end,
                    })
                } else {
                    json!({"char_offset": *contiguous_end})
                };
                return archived_reader_gap_receipt(context, &page, &view_key, next_cursor);
            }
            *contiguous_end = char_offset.saturating_add(returned_chars).min(total_chars);
            total_chars == 0 || *contiguous_end >= total_chars
        }
        ArchivedReaderProgress::FileLines {
            started,
            next_offset,
            next_char_offset,
            continuation_limit,
            seeded_from_preview,
            preview_replay_used,
        } => {
            let offset = page.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let char_offset = page.get("char_offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            let limit = page.get("limit").and_then(Value::as_u64).unwrap_or(1) as usize;
            let starts_at_archive_beginning = offset
                == page
                    .get("archived_source_offset")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize
                && char_offset == 0;
            let repeats_delivered_prefix = *started
                && (offset < *next_offset
                    || (offset == *next_offset && char_offset < *next_char_offset));
            let changes_active_page = *started
                && offset == *next_offset
                && char_offset == *next_char_offset
                && char_offset > 0
                && *continuation_limit != Some(limit);
            if repeats_delivered_prefix {
                if *seeded_from_preview && !*preview_replay_used && starts_at_archive_beginning {
                    // Context compaction is allowed to retire an old native
                    // tool message, but must not make the reader lie that its
                    // body remains model-visible. One explicit restart
                    // replays the bounded initial page; subsequent duplicate
                    // requests still receive the normal content-free receipt.
                    *preview_replay_used = true;
                    return page;
                }
                let next_limit = continuation_limit.unwrap_or(limit);
                return archived_reader_prefix_receipt(
                    context,
                    &page,
                    &view_key,
                    json!({
                        "offset": *next_offset,
                        "limit": next_limit,
                        "char_offset": *next_char_offset,
                    }),
                );
            }
            if changes_active_page {
                return archived_reader_gap_receipt(
                    context,
                    &page,
                    &view_key,
                    json!({
                        "offset": *next_offset,
                        "limit": continuation_limit.unwrap_or(limit),
                        "char_offset": *next_char_offset,
                    }),
                );
            }
            let follows_cursor = *started
                && offset == *next_offset
                && char_offset == *next_char_offset
                && (char_offset == 0 || *continuation_limit == Some(limit));
            if !*started && !starts_at_archive_beginning {
                return archived_reader_gap_receipt(
                    context,
                    &page,
                    &view_key,
                    json!({
                        "offset": *next_offset,
                        "limit": continuation_limit.unwrap_or(limit),
                        "char_offset": *next_char_offset,
                    }),
                );
            }
            if *started && !follows_cursor {
                return archived_reader_gap_receipt(
                    context,
                    &page,
                    &view_key,
                    json!({
                        "offset": *next_offset,
                        "limit": continuation_limit.unwrap_or(limit),
                        "char_offset": *next_char_offset,
                    }),
                );
            }
            if !*started && starts_at_archive_beginning || follows_cursor {
                *started = true;
                if let Some(cursor) = page.get("next_cursor").and_then(Value::as_object) {
                    *next_offset =
                        cursor.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
                    *next_char_offset = cursor
                        .get("char_offset")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    *continuation_limit = (*next_char_offset > 0).then_some(
                        cursor
                            .get("limit")
                            .and_then(Value::as_u64)
                            .unwrap_or(limit as u64) as usize,
                    );
                    false
                } else {
                    true
                }
            } else {
                false
            }
        }
    };

    if completely_delivered {
        view.completed_receipt = Some(archived_reader_completion_receipt(
            context,
            archive_content,
            &page,
            &view_key,
        ));
    }
    reader.retired = archived_reader_can_retire(reader, context, archive_content);
    page
}

/// Built-ins that are executor capabilities rather than independently
/// registered skills inherit a reviewed least-privilege SkillGraph policy.
/// Keep this table explicit: an unknown tool must still fail closed.
fn builtin_security_skill_iri(name: &str) -> Option<&'static str> {
    match name {
        "tool_search"
        | "glob_search"
        | "grep_search"
        | "file_read"
        | "file_list"
        | "workspace_status"
        | "rag_search"
        | "kg_search"
        | "codebase_search"
        | "knowledge_list"
        | "knowledge_search"
        | "knowledge_extract_code"
        | "knowledge_query"
        | "knowledge_neighbors"
        | "read_agent_output"
        | "read_full_result"
        | "get_entity_details"
        | "expand_relation" => Some("iri://skills/file_read"),
        "bash" | "file_write" | "file_edit" => Some("iri://skills/file_write"),
        "web_search" | "web_fetch" | "http_request" => Some("iri://skills/http_request"),
        "llm_chat" => Some("iri://skills/llm_chat"),
        _ => None,
    }
}

/// Tool role filter: empty = all roles, "PA"/"DA"/"CA"/"AA" = role-specific only
#[derive(Clone)]
pub struct ToolDescription {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub allowed_roles: Vec<String>, // empty = all roles allowed
}

impl ToolExecutor {
    pub fn new() -> Self {
        let kg_store = Arc::new(std::sync::RwLock::new(
            KnowledgeGraphStore::new().expect("Failed to create knowledge graph store"),
        ));
        let mut exe = Self {
            tools: HashMap::new(),
            contextual_tools: HashMap::new(),
            tool_provenance: HashMap::new(),
            registering_builtins: true,
            tool_descriptions: Vec::new(),
            kg_store,
            projection_engine: Arc::new(parking_lot::RwLock::new(None)),
            micro_tool_contexts: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            micro_tool_data: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            archived_reader_consumption: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            syscall_gate: None,
            permission_policy: None,
            hook_runner: None,
            hook_allow_input_rewrite: false,
            tool_group_manager: None,
            workspace_monitor: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_graph: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_registry: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_vector_store: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_creator_interactions: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_creator_gateway: Arc::new(parking_lot::RwLock::new(None)),
            security_engine: Arc::new(parking_lot::RwLock::new(None)),
            file_read_exposures: Arc::new(parking_lot::RwLock::new(HashSet::new())),
            workspace_mutation_coordinator: Arc::new(tokio::sync::Mutex::new(
                WorkspaceMutationCoordinatorState::new(),
            )),
            max_micro_tool_descriptions: 5,
            micro_tool_page_size: 100,
            micro_tool_max_page_size: 200,
        };
        exe.register_builtins();
        exe.registering_builtins = false;
        exe
    }

    /// Acquire the runner-wide mutation settlement window. The returned owned
    /// guard remains valid independently of this borrowed executor reference,
    /// allowing the runner to cover snapshot + execution + post-snapshot.
    pub(crate) async fn acquire_workspace_mutation_guard(
        &self,
    ) -> tokio::sync::OwnedMutexGuard<WorkspaceMutationCoordinatorState> {
        self.workspace_mutation_coordinator
            .clone()
            .lock_owned()
            .await
    }

    pub fn set_micro_tool_limits(
        &mut self,
        max_descriptions: usize,
        page_size: usize,
        max_page_size: usize,
    ) {
        self.max_micro_tool_descriptions = max_descriptions.max(1);
        self.micro_tool_page_size = page_size.max(1);
        self.micro_tool_max_page_size = max_page_size.max(self.micro_tool_page_size);
    }

    pub fn set_projection_engine(
        &mut self,
        engine: Arc<crate::memory::l3_projection::ProjectionEngine>,
    ) {
        *self.projection_engine.write() = Some(engine);
    }

    pub fn set_tool_group_manager(&mut self, manager: ToolGroupManager) {
        self.tool_group_manager = Some(manager);
    }

    pub fn clear_tool_group_manager(&mut self) {
        self.tool_group_manager = None;
    }

    /// Point the existing shared KG holder at a unified Oxigraph Store.
    ///
    /// Built-in KG tool handlers capture `self.kg_store` during registration.
    /// Replacing the Arc here would leave those handlers on the old isolated
    /// store, so preserve the Arc identity and replace only its inner value.
    pub fn set_unified_kg_store(&mut self, store: Arc<oxigraph::store::Store>) {
        let shared_store = KnowledgeGraphStore::with_shared_store(store)
            .expect("Failed to create shared KG Store");
        let mut guard = self
            .kg_store
            .write()
            .expect("Knowledge graph store lock poisoned");
        *guard = shared_store;
    }

    pub fn set_syscall_gate(&mut self, gate: crate::core::syscall_gate::SyscallGate) {
        self.syscall_gate = Some(gate);
    }

    pub fn set_permission_policy(&mut self, policy: PermissionPolicy) {
        self.permission_policy = Some(policy);
    }

    /// Enable enforced SkillGraph security decisions for contextual calls.
    /// Callers must supply the real agent/task context via
    /// `execute_with_security_context`.
    pub fn set_security_engine(&self, engine: Arc<SecurityEngine>) {
        *self.security_engine.write() = Some(engine);
    }

    pub fn set_hook_runner(&mut self, runner: HookRunner) {
        self.hook_runner = Some(runner);
        self.hook_allow_input_rewrite = false;
    }

    pub fn set_hook_runner_with_input_rewrite(
        &mut self,
        runner: HookRunner,
        allow_input_rewrite: bool,
    ) {
        self.hook_runner = Some(runner);
        self.hook_allow_input_rewrite = allow_input_rewrite;
    }

    pub fn set_workspace_monitor(&mut self, monitor: Arc<WorkspaceMonitor>) {
        *self.workspace_monitor.write() = Some(monitor);
    }

    pub fn get_workspace_monitor(&self) -> Option<Arc<WorkspaceMonitor>> {
        self.workspace_monitor.read().clone()
    }

    /// Get a reference to the internal KnowledgeGraphStore for shared use
    /// (e.g. by FusedRootCauseEngine for SPARQL semantic neighbor traversal).
    pub fn knowledge_graph_store(&self) -> Arc<std::sync::RwLock<KnowledgeGraphStore>> {
        self.kg_store.clone()
    }

    /// Inject the shared SkillGraphStore so that create_skill/convert_skill tools
    /// write into the live graph instead of an isolated temporary store.
    pub fn set_shared_skill_graph(&self, store: Arc<SkillGraphStore>) {
        *self.shared_skill_graph.write() = Some(store);
    }

    /// Inject the live registry used by planning and execution so dynamically
    /// created skills do not disappear into a short-lived private registry.
    pub fn set_shared_skill_registry(&self, registry: Arc<SkillRegistry>) {
        *self.shared_skill_registry.write() = Some(registry);
    }

    /// Inject the application-owned vector store used to index skills created
    /// through built-in tools.
    pub fn set_shared_skill_vector_store(&self, store: Arc<HyperspaceStore>) {
        *self.shared_skill_vector_store.write() = Some(store);
    }

    /// Inject the AgentRunner-owned model interaction plane. Dynamic skill
    /// tools then participate in the same Hook/EventBus/token-accounting
    /// lifecycle as the ReAct request that invoked them.
    pub fn set_shared_skill_creator_interactions(
        &self,
        interactions: Arc<crate::llm::LlmInteractionService>,
    ) {
        *self.shared_skill_creator_interactions.write() = Some(interactions);
    }

    #[cfg(test)]
    pub(crate) fn skill_creator_interactions(
        &self,
    ) -> Option<Arc<crate::llm::LlmInteractionService>> {
        self.shared_skill_creator_interactions.read().clone()
    }

    /// Inject the fallback application gateway used by standalone dynamic
    /// skill creation. AgentRunner also injects its exact interaction service,
    /// which takes precedence so runner-local Hooks and accounting are kept.
    pub fn set_shared_skill_creator_gateway(
        &self,
        gateway: Arc<crate::gateway::unified_gateway::UnifiedGateway>,
    ) {
        *self.shared_skill_creator_gateway.write() = Some(gateway);
    }

    /// Notify workspace_monitor that a file was read externally (e.g., via read_full_result).
    /// This helps the cache/diff system recognize the file as already-read on subsequent file_read.
    pub fn mark_file_external_read(&self, path: &str) {
        let guard = self.workspace_monitor.read();
        if let Some(ref wm) = *guard {
            wm.mark_file_read_external(path);
        }
    }

    /// Default tool requirements: bash/pwsh/code_exec→DangerFullAccess, file_write/edit→WorkspaceWrite, reads→ReadOnly
    pub fn set_default_permission_policy(&mut self) {
        let policy = PermissionPolicy::new(PermissionMode::Allow)
            .with_tool_requirement("bash", PermissionMode::DangerFullAccess)
            .with_tool_requirement("powershell", PermissionMode::DangerFullAccess)
            .with_tool_requirement("code_execute", PermissionMode::DangerFullAccess)
            .with_tool_requirement("file_write", PermissionMode::WorkspaceWrite)
            .with_tool_requirement("file_edit", PermissionMode::WorkspaceWrite)
            .with_tool_requirement("file_read", PermissionMode::ReadOnly)
            .with_tool_requirement("grep_search", PermissionMode::ReadOnly)
            .with_tool_requirement("glob_search", PermissionMode::ReadOnly)
            .with_tool_requirement("web_search", PermissionMode::ReadOnly)
            .with_tool_requirement("web_fetch", PermissionMode::ReadOnly);
        self.permission_policy = Some(policy);
    }

    fn register_builtins(&mut self) {
        // All tools open to all roles; LLM selects based on role description in agent.md
        let all: &[&str] = &[];
        self.register(
            "glob_search",
            "Find files by glob pattern.",
            json!({
                "properties": {"pattern": {"type":"string"},"path": {"type":"string"}},
                "required": ["pattern"]
            }),
            Arc::new(|input: Value| {
                Box::pin(async move { builtins::execute_glob_search(input).await })
            }),
            all,
        );
        self.register("grep_search", "Search file contents with regex.", json!({
            "properties": {
                "pattern": {"type":"string","description":"Regex pattern to search for"},
                "path": {"type":"string","description":"Directory to search in"},
                "glob": {"type":"string","description":"File glob pattern (e.g. *.rs)"},
                "output_mode": {"type":"string","description":"Output mode: files_with_matches | content | count"},
                "before": {"type":"integer","description":"Lines before match (-B)"},
                "after": {"type":"integer","description":"Lines after match (-A)"},
                "context": {"type":"integer","description":"Context lines around match (-C)"},
                "line_numbers": {"type":"boolean","description":"Show line numbers (default true)"},
                "head_limit": {"type":"integer","description":"Limit number of results (default 250)"},
                "offset": {"type":"integer","description":"Skip first N results"},
                "-i": {"type":"boolean","description":"Case insensitive search"},
                "multiline": {"type":"boolean","description":"Enable multiline mode"},
                "file_type": {"type":"string","description":"File type filter (rust, python, etc.)"}
            },
            "required": ["pattern"]
        }), Arc::new(|input: Value| Box::pin(async move { builtins::execute_grep_search(input).await })), all);
        self.register(
            "web_fetch",
            "Fetch a URL into readable text.",
            json!({
                "properties": {"url": {"type":"string"},"prompt": {"type":"string"}},
                "required": ["url"]
            }),
            Arc::new(|input: Value| {
                Box::pin(async move { builtins::execute_web_fetch(input).await })
            }),
            all,
        );
        self.register(
            "web_search",
            "Search the web for information.",
            json!({
                "properties": {"query": {"type":"string","minLength":2}},
                "required": ["query"]
            }),
            Arc::new(|input: Value| {
                Box::pin(async move { builtins::execute_web_search(input).await })
            }),
            all,
        );
        self.register(
            "tool_search",
            "Search available tools by name.",
            json!({
                "properties": {"query": {"type":"string"},"max_results": {"type":"integer"}},
                "required": ["query"]
            }),
            Arc::new(|input: Value| {
                Box::pin(async move { builtins::execute_tool_search(input).await })
            }),
            all,
        );
        let ws_read = self.workspace_monitor.clone();
        let read_exposures = self.file_read_exposures.clone();
        self.register("file_read", "Read a text file. Reads the entire file by default. On re-read of a changed file, returns a unified diff showing what changed. An unchanged whole-file auto re-read may return from_cache=true. Offset/limit reads always return the requested range, including when the file is cached. Use mode:full when earlier content is no longer visible, or mode:changed_only for changed lines.", json!({
            "properties": {
                "path": {"type":"string", "description": "File path to read"},
                "offset": {"type":"integer", "description": "Line offset to start from (0-indexed). Omit to read from beginning."},
                "limit": {"type":"integer", "description": "Number of lines to return. Omit to read all remaining lines from offset."},
                "mode": {"type":"string", "description": "Read mode: auto (default=use diff if previously read) | full | force_refresh | diff | changed_only"}
            },
            "required": ["path"]
        }), Arc::new(move |input: Value| {
            let ws = ws_read.clone();
            let read_exposures = read_exposures.clone();
            Box::pin(async move {
                let mode = input.get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("auto")
                    .to_string();
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if crate::tools::workspace_monitor::inventory::is_workspace_runtime_path(
                    std::path::Path::new(&path),
                ) {
                    return Err(
                        "Workspace runtime state is not readable as project content".to_string(),
                    );
                }
                let read_session = input.get("__gh_read_session")
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let exposure_key = read_session.as_ref()
                    .map(|session| format!("{session}\n{path}"));
                // Calls made through the legacy context-free execute() API
                // retain the historical global-cache behavior. Contextual
                // BizAgent calls require evidence that this exact
                // AgentInstance/L1 context has already received the whole
                // file.
                let already_exposed = exposure_key.as_ref()
                    .map(|key| read_exposures.read().contains(key))
                    .unwrap_or(true);
                // Extract offset/limit before input is moved into execute_file_read
                let has_offset = input.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) > 0;
                let has_limit = input.get("limit").is_some();

                // Fast path is safe only for a repeated whole-file `auto`
                // read. A previous implementation returned a cache marker for
                // offset/limit requests too, making it impossible to read the
                // second chunk of a large file. Explicit `full` also means the
                // caller needs content again (for example after context
                // compression), so it must bypass the marker.
                if !path.is_empty()
                    && mode == "auto"
                    && !has_offset
                    && !has_limit
                {
                    let guard = ws.read();
                    if let Some(ref wm) = *guard {
                        let normalized_path = wm.normalize_path(&path);
                        let entry = wm.inventory.read().get_entry(&normalized_path);
                        let should_cache = match entry {
                            Some(ref e) => e.state == crate::tools::workspace_monitor::FileState::ReadFresh
                                && e.current_version == e.last_read_version
                                && wm.content().try_get_cached(&normalized_path).is_some(),
                            None => false,
                        };
                        if should_cache && already_exposed {
                            return Ok(json!({
                                "path": path,
                                "from_cache": true,
                                "message": "Cache hit: file unchanged since the last whole-file read. Use mode:full or an offset/limit range if the earlier content is no longer visible."
                            }));
                        }
                    }
                    drop(guard);
                }

                // Slow path: read from disk once
                let result = builtins::execute_file_read(input).await?;
                let guard = ws.read();
                if let Some(ref wm) = *guard {
                    if let Some(path) = result.get("path").and_then(|v| v.as_str()) {
                        let read_mode = match mode.as_str() {
                            "force_refresh" => crate::tools::workspace_monitor::ReadMode::ForceRefresh,
                            "full" => crate::tools::workspace_monitor::ReadMode::Full,
                            "diff" => crate::tools::workspace_monitor::ReadMode::Diff,
                            "changed_only" => crate::tools::workspace_monitor::ReadMode::ChangedOnly,
                            _ => {
                                // auto: use diff if file is already cached, else full
                                let inv = wm.inventory.read();
                                let entry = inv.get_entry(&wm.normalize_path(path));
                                match entry {
                                    Some(e) if e.read_count > 0 => crate::tools::workspace_monitor::ReadMode::Diff,
                                    _ => crate::tools::workspace_monitor::ReadMode::Full,
                                }
                            }
                        };
                        if let Ok(read_result) = wm.read_file(path, read_mode) {
                                let mut result = result;
                                if let Some(diff) = &read_result.unified_diff {
                                    result.as_object_mut().map(|obj| {
                                        obj.insert("unified_diff".to_string(), Value::String(diff.clone()));
                                    });
                                }
                                if let Some(changed) = &read_result.changed_lines {
                                    result.as_object_mut().map(|obj| {
                                        obj.insert("changed_lines".to_string(), Value::Array(
                                            changed.iter().map(|l| Value::String(l.clone())).collect()
                                        ));
                                    });
                                }
                                if !read_result.changed && read_result.from_cache {
                                    // Cache hit: file unchanged since last read.
                                    // Strip full content to avoid token waste on re-read.
                                    if mode == "auto" && !has_offset && !has_limit && already_exposed {
                                        result.as_object_mut().map(|obj| {
                                            obj.remove("lines");
                                            obj.remove("returned");
                                            obj.insert("from_cache".to_string(), Value::Bool(true));
                                            obj.insert("message".to_string(), Value::String(
                                                "Cache hit: file unchanged since the last whole-file read. Use mode:full or an offset/limit range if the earlier content is no longer visible.".to_string()
                                            ));
                                        });
                                    } else {
                                        result.as_object_mut().map(|obj| {
                                            obj.insert("from_cache".to_string(), Value::Bool(true));
                                            obj.insert("message".to_string(), Value::String(
                                                if already_exposed {
                                                    "Cache hit: file unchanged; full content is included again because this call explicitly requested content.".to_string()
                                                } else {
                                                    "Shared content cache hit; full content is included because this Agent/L1 context had not observed it yet.".to_string()
                                                }
                                            ));
                                        });
                                    }
                                }
                                return Ok(result);
                            }
                        }
                    }
                Ok(result)
            })
        }), all);
        let ws_write = self.workspace_monitor.clone();
        self.register(
            "file_write",
            "Write content to a file.",
            json!({
                "properties": {"path": {"type":"string"},"content": {"type":"string"}},
                "required": ["path","content"]
            }),
            Arc::new(move |input: Value| {
                let ws = ws_write.clone();
                Box::pin(async move {
                    let result = builtins::execute_file_write(input).await?;
                    if result.get("success") == Some(&Value::Bool(true))
                        && result.get("changed") != Some(&Value::Bool(false))
                    {
                        let guard = ws.read();
                        if let Some(ref wm) = *guard {
                            if let Some(path) = result.get("path").and_then(|v| v.as_str()) {
                                wm.mark_file_written(path);
                            }
                        }
                    }
                    Ok(result)
                })
            }),
            all,
        );
        let ws_status = self.workspace_monitor.clone();
        self.register("workspace_status", "View workspace file status summary: stale files, written-unread files, counts by state and language.", json!({
            "properties": {},
            "required": []
        }), Arc::new(move |_: Value| {
            let ws = ws_status.clone();
            Box::pin(async move {
                let guard = ws.read();
                if let Some(ref wm) = *guard {
                    let inv = wm.inventory.read();
                        let all = inv.list_all();
                        let total = all.len();

                        let stale = inv.list_by_state(FileState::ReadStale);
                        let written_unread = inv.list_by_state(FileState::WrittenUnread);
                        let discovered = inv.list_by_state(FileState::Discovered);
                        let fresh = inv.list_by_state(FileState::ReadFresh);

                        // Group by language
                        let mut lang_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                        for entry in &all {
                            *lang_map.entry(entry.language.clone()).or_insert(0) += 1;
                        }
                        let mut by_language: Vec<serde_json::Value> = lang_map.into_iter()
                            .map(|(lang, count)| json!({"language": lang, "count": count}))
                            .collect();
                        by_language.sort_by(|a, b| {
                            b["count"].as_u64().unwrap_or(0).cmp(&a["count"].as_u64().unwrap_or(0))
                        });

                        return Ok(json!({
                            "total_files": total,
                            "stale_count": stale.len(),
                            "stale_files": stale.iter().take(20).map(|e| json!(e.path)).collect::<Vec<_>>(),
                            "written_unread_count": written_unread.len(),
                            "written_unread_files": written_unread.iter().take(20).map(|e| json!(e.path)).collect::<Vec<_>>(),
                            "discovered_count": discovered.len(),
                            "fresh_count": fresh.len(),
                            "by_language": by_language,
                        }));
                    }
                // Fallback if no workspace_monitor available
                Ok(json!({"total_files": 0, "stale_count": 0, "written_unread_count": 0, "message": "Workspace monitor not available"}))
            })
        }), all);
        let ws_list = self.workspace_monitor.clone();
        self.register(
            "file_list",
            "List files in a directory.",
            json!({
                "properties": {"path": {"type":"string"}},
                "required": []
            }),
            Arc::new(move |input: Value| {
                let ws = ws_list.clone();
                Box::pin(async move {
                    let monitor = ws.read().clone();
                    if let Some(ref wm) = monitor {
                        if wm.scan_complete() {
                            let requested =
                                input.get("path").and_then(Value::as_str).unwrap_or(".");
                            let requested_path = std::path::PathBuf::from(requested);
                            let directory_candidate = if requested == "." {
                                wm.config.workspace_root.clone()
                            } else if requested_path.is_absolute() {
                                requested_path
                            } else {
                                wm.config.workspace_root.join(requested_path)
                            };
                            let canonical_root = std::fs::canonicalize(&wm.config.workspace_root)
                                .unwrap_or_else(|_| wm.config.workspace_root.clone());
                            let Ok(directory) = std::fs::canonicalize(&directory_candidate) else {
                                // Preserve built-in validation/error semantics for
                                // missing or unresolved directories.
                                return builtins::execute_file_list(input).await;
                            };
                            if !directory.starts_with(&canonical_root) {
                                return Err(format!(
                                    "Path outside workspace is not allowed: {}",
                                    requested
                                ));
                            }
                            let mut listed =
                                std::collections::BTreeMap::<String, serde_json::Value>::new();
                            for file in wm.inventory.read().list_all() {
                                let file_path = std::path::Path::new(&file.path);
                                let Ok(relative) = file_path.strip_prefix(&directory) else {
                                    continue;
                                };
                                let mut components = relative.components();
                                let Some(first) = components.next() else {
                                    continue;
                                };
                                let name = first.as_os_str().to_string_lossy().to_string();
                                let is_directory = components.next().is_some();
                                let value = if is_directory {
                                    json!({"name": name, "type": "dir"})
                                } else {
                                    json!({
                                        "name": name,
                                        "type": "file",
                                        "state": file.state.as_str(),
                                        "language": file.language,
                                        "size": file.file_size,
                                        "version": file.current_version,
                                    })
                                };
                                match listed.get(&name) {
                                    Some(existing) if existing["type"] == "dir" => {}
                                    _ => {
                                        listed.insert(name, value);
                                    }
                                }
                            }
                            let entries = listed.into_values().collect::<Vec<_>>();
                            let count = entries.len();
                            return Ok(json!({
                                "path": requested,
                                "entries": entries,
                                "count": count,
                                "source": "workspace_inventory",
                                "generation": wm.generation(),
                            }));
                        }
                    }
                    builtins::execute_file_list(input).await
                })
            }),
            all,
        );
        let bash_desc = if cfg!(target_os = "windows") {
            "Execute a shell command via PowerShell. Use for running python, pytest, etc. Supports most common shell commands.\n\nOUTPUT MANAGEMENT (mandatory):\n- If the command may produce >100 lines of output, pipe through | head -N or | grep <keyword> to limit results\n- Use | tail -N for recent entries, | wc -l to count first, | grep -c to match-count\n- For file searches, constrain the path (e.g. grep ... path/) instead of searching the entire workspace\n- The output will be truncated at 16KB if too large; always filter proactively to avoid losing data"
        } else {
            "Execute a shell command. Use for running python, pytest, etc.\n\nOUTPUT MANAGEMENT (mandatory):\n- If the command may produce >100 lines of output, pipe through | head -N or | grep <keyword> to limit results\n- Use | tail -N for recent entries, | wc -l to count first, | grep -c to match-count\n- For file searches, constrain the path (e.g. grep ... path/) instead of searching the entire workspace\n- The output will be truncated at 16KB if too large; always filter proactively to avoid losing data"
        };
        self.register("bash", bash_desc, json!({
            "properties": {
                "command": {"type":"string","description":"Shell command to run"},
                "description": {"type":"string","description":"What this command does"},
                "timeout": {"type":"integer","description":"Timeout in milliseconds"},
                "run_in_background": {"type":"boolean","description":"Spawn detached and return a task id immediately (default false)"},
                "dangerouslyDisableSandbox": {"type":"boolean","description":"Run outside the sandbox. Only use when the command cannot work sandboxed and you are certain it is safe"},
                "namespaceRestrictions": {"type":"boolean","description":"Enable user/mount/pid namespace isolation via unshare (default true when sandbox enabled)"},
                "isolateNetwork": {"type":"boolean","description":"Isolate network via a new network namespace (default false)"},
                "filesystemMode": {"type":"string","enum":["off","workspace-only","allow-list"],"description":"Filesystem isolation level (default workspace-only)"},
                "allowedMounts": {"type":"array","items":{"type":"string"},"description":"Additional paths allowed when filesystemMode is allow-list"}
            },
            "required": ["command"]
        }), Arc::new(|input: Value| Box::pin(async move { builtins::execute_bash(input).await })), all);
        let ws_edit = self.workspace_monitor.clone();
        self.register("file_edit", "Edit a file by replacing old_string with new_string.", json!({
            "properties": {
                "path": {"type":"string","description":"File path to edit"},
                "old_string": {"type":"string","description":"Text to find and replace"},
                "new_string": {"type":"string","description":"Replacement text"},
                "replace_all": {"type":"boolean","description":"Replace all occurrences (default: false)"}
            },
            "required": ["path","old_string","new_string"]
        }), Arc::new(move |input: Value| {
            let ws = ws_edit.clone();
            Box::pin(async move {
                let result = builtins::execute_file_edit(input).await?;
                if result.get("success") == Some(&Value::Bool(true))
                    && result.get("changed") != Some(&Value::Bool(false))
                {
                    let guard = ws.read();
                    if let Some(ref wm) = *guard {
                        if let Some(path) = result.get("path").and_then(|v| v.as_str()) {
                            wm.mark_file_written(path);
                        }
                    }
                }
                Ok(result)
            })
        }), all);
        self.register(
            "powershell",
            "Execute a PowerShell command.",
            json!({
                "properties": {
                    "command": {"type":"string","description":"PowerShell command to run"},
                    "description": {"type":"string","description":"What this command does"},
                    "timeout": {"type":"integer","description":"Timeout in milliseconds"}
                },
                "required": ["command"]
            }),
            Arc::new(|input: Value| {
                Box::pin(async move { builtins::execute_powershell(input).await })
            }),
            all,
        );
        self.register("rag_search", "Semantic search for relevant documents using RAG (Retrieval-Augmented Generation).", json!({
            "properties": {"query": {"type":"string","description":"Search query"},"limit": {"type":"integer","description":"Max results"}},
            "required": ["query"]
        }), sync_tool_ref(rag::execute_rag_search), all);
        self.register("rag_index", "Index a document for RAG retrieval.", json!({
            "properties": {"content": {"type":"string","description":"Document content to index"},"iri": {"type":"string","description":"Optional IRI identifier"},"tags": {"type":"array","items":{"type":"string"},"description":"Optional tags"}},
            "required": ["content"]
        }), sync_tool_ref(rag::execute_rag_index), all);
        self.register("rag_chunk", "Split a document into chunks for indexing.", json!({
            "properties": {"content": {"type":"string","description":"Document content to chunk"},"chunk_size": {"type":"integer","description":"Chunk size in characters (default 500)"},"overlap": {"type":"integer","description":"Overlap between chunks (default 50)"}},
            "required": ["content"]
        }), sync_tool_ref(rag::execute_rag_chunk), all);

        // ========== Knowledge Import Tools ==========
        self.register("knowledge_import_file", "Import knowledge from a file (Markdown, TXT, HTML, JSON, etc.). Auto-chunks and indexes the content.", json!({
            "properties": {
                "path": {"type":"string","description":"File path to import"},
                "tags": {"type":"array","items":{"type":"string"},"description":"Tags for categorization"},
                "chunk_size": {"type":"integer","description":"Chunk size in characters (default 1000)"},
                "overlap": {"type":"integer","description":"Overlap between chunks (default 100)"},
                "auto_detect_title": {"type":"boolean","description":"Auto-detect title from content (default true)"}
            },
            "required": ["path"]
        }), Arc::new(|input: Value| Box::pin(async move {
            let input = resolve_workspace_path_argument(input, "path")?;
            knowledge::execute_knowledge_import_file(input).await
        })), all);

        self.register("knowledge_import_url", "Import knowledge from a URL. Fetches and extracts text content from web pages.", json!({
            "properties": {
                "url": {"type":"string","description":"URL to fetch and import"},
                "tags": {"type":"array","items":{"type":"string"},"description":"Tags for categorization"},
                "chunk_size": {"type":"integer","description":"Chunk size in characters (default 1000)"},
                "overlap": {"type":"integer","description":"Overlap between chunks (default 100)"},
                "selector": {"type":"string","description":"CSS selector or regex to extract specific content"}
            },
            "required": ["url"]
        }), Arc::new(|input: Value| Box::pin(async move { knowledge::execute_knowledge_import_url(input).await })), all);

        self.register("knowledge_import_directory", "Batch import knowledge from a directory. Recursively processes matching files.", json!({
            "properties": {
                "path": {"type":"string","description":"Directory path to import"},
                "pattern": {"type":"string","description":"File pattern (default: *.md,*.txt,*.html,*.json)"},
                "tags": {"type":"array","items":{"type":"string"},"description":"Tags for categorization"},
                "recursive": {"type":"boolean","description":"Recursively process subdirectories (default true)"},
                "chunk_size": {"type":"integer","description":"Chunk size in characters (default 1000)"},
                "overlap": {"type":"integer","description":"Overlap between chunks (default 100)"}
            },
            "required": ["path"]
        }), Arc::new(|input: Value| Box::pin(async move {
            let input = resolve_workspace_path_argument(input, "path")?;
            knowledge::execute_knowledge_import_directory(input).await
        })), all);

        self.register("knowledge_list", "List imported knowledge entries with optional filtering.", json!({
            "properties": {
                "tags": {"type":"array","items":{"type":"string"},"description":"Filter by tags"},
                "source_type": {"type":"string","description":"Filter by source type (file, url)"},
                "limit": {"type":"integer","description":"Max results (default 100)"},
                "offset": {"type":"integer","description":"Pagination offset"}
            }
        }), Arc::new(|input: Value| Box::pin(async move { knowledge::execute_knowledge_list(input).await })), all);

        self.register("knowledge_delete", "Delete imported knowledge entries by IRI or tags.", json!({
            "properties": {
                "iri": {"type":"string","description":"IRI of knowledge entry to delete"},
                "tags": {"type":"array","items":{"type":"string"},"description":"Delete all entries with these tags"},
                "all": {"type":"boolean","description":"Delete all knowledge entries"}
            }
        }), Arc::new(|input: Value| Box::pin(async move { knowledge::execute_knowledge_delete(input).await })), all);

        self.register("knowledge_search", "Search imported knowledge with keyword matching and optional tag filtering.", json!({
            "properties": {
                "query": {"type":"string","description":"Search query"},
                "tags": {"type":"array","items":{"type":"string"},"description":"Filter by tags"},
                "limit": {"type":"integer","description":"Max results (default 10)"},
                "min_score": {"type":"number","description":"Minimum relevance score (default 0.1)"}
            },
            "required": ["query"]
        }), Arc::new(|input: Value| Box::pin(async move { knowledge::execute_knowledge_search(input).await })), all);

        self.register("knowledge_update", "Update content or tags of an imported knowledge entry.", json!({
            "properties": {
                "iri": {"type":"string","description":"IRI of knowledge entry to update"},
                "content": {"type":"string","description":"New content"},
                "tags": {"type":"array","items":{"type":"string"},"description":"New or additional tags"},
                "append_tags": {"type":"boolean","description":"Append tags instead of replacing (default false)"}
            },
            "required": ["iri"]
        }), Arc::new(|input: Value| Box::pin(async move { knowledge::execute_knowledge_update(input).await })), all);

        // ========== Skill Creation Tools (with shared SkillGraphStore) ==========
        let sg_for_create = self.shared_skill_graph.clone();
        let registry_for_create = self.shared_skill_registry.clone();
        let vector_store_for_create = self.shared_skill_vector_store.clone();
        let interactions_for_create = self.shared_skill_creator_interactions.clone();
        let gateway_for_create = self.shared_skill_creator_gateway.clone();
        let create_skill_handler: ContextualToolFn = Arc::new(move |input: Value, scope| {
            let sg = sg_for_create.read().clone();
            let registry = registry_for_create.read().clone();
            let vector_store = vector_store_for_create.read().clone();
            let interactions = interactions_for_create.read().clone();
            let gateway = gateway_for_create.read().clone();
            Box::pin(async move {
                builtins::execute_create_skill(
                    input,
                    interactions,
                    gateway,
                    sg,
                    registry,
                    vector_store,
                    scope,
                )
                .await
            })
        });
        let create_skill_unscoped = create_skill_handler.clone();
        self.register("create_skill", "Create a new Skill definition from natural language using LLM. The definition is registered for review and discovery; it does not create an executable ToolExecutor handler.", json!({
            "properties": {
                "description": {"type":"string","description":"Natural language description of the skill to create"},
                "skill_name_hint": {"type":"string","description":"Suggested skill name (optional, lowercase with underscores)"},
                "category_hint": {"type":"string","description":"Suggested category (optional): file|network|ai|execution|validation|data|meta|system"},
                "security_level_override": {"type":"string","description":"Security level override (optional): low|normal|high|critical"}
            },
            "required": ["description"]
        }), Arc::new(move |input: Value| create_skill_unscoped(input, None)), &["DA"]);
        self.register_contextual_handler("create_skill", create_skill_handler);

        let sg_for_convert = self.shared_skill_graph.clone();
        let registry_for_convert = self.shared_skill_registry.clone();
        let vector_store_for_convert = self.shared_skill_vector_store.clone();
        let interactions_for_convert = self.shared_skill_creator_interactions.clone();
        let gateway_for_convert = self.shared_skill_creator_gateway.clone();
        let convert_skill_handler: ContextualToolFn = Arc::new(move |input: Value, scope| {
            let sg = sg_for_convert.read().clone();
            let registry = registry_for_convert.read().clone();
            let vector_store = vector_store_for_convert.read().clone();
            let interactions = interactions_for_convert.read().clone();
            let gateway = gateway_for_convert.read().clone();
            Box::pin(async move {
                builtins::execute_convert_skill(
                    input,
                    interactions,
                    gateway,
                    sg,
                    registry,
                    vector_store,
                    scope,
                )
                .await
            })
        });
        let convert_skill_unscoped = convert_skill_handler.clone();
        self.register("convert_skill", "Convert a Markdown-formatted skill description into a JSON-LD Skill definition. Parses the markdown structure and generates proper skill schema.", json!({
            "properties": {
                "markdown_content": {"type":"string","description":"Markdown content describing the skill"},
                "source_path": {"type":"string","description":"Source file path (optional)"}
            },
            "required": ["markdown_content"]
        }), Arc::new(move |input: Value| convert_skill_unscoped(input, None)), &["DA","CA"]);
        self.register_contextual_handler("convert_skill", convert_skill_handler);

        // ========== Knowledge Graph Tools ==========
        let kg_store_for_extract = self.kg_store.clone();
        self.register("knowledge_extract", "Extract entities and relations from unstructured text into the knowledge graph. Uses LLM for intelligent extraction.", json!({
            "properties": {
                "text": {"type":"string","description":"Text content to extract from."},
                "domain": {"type":"string","description":"Domain filter (optional, e.g. business/core)."}
            },
            "required": ["text"]
        }), Arc::new(move |input: Value| {
            let kg_store = kg_store_for_extract.clone();
            Box::pin(async move { builtins::execute_knowledge_extract(input, kg_store).await })
        }), all);

        let kg_store_for_query = self.kg_store.clone();
        self.register(
            "knowledge_query",
            "Execute a SPARQL SELECT query against the knowledge graph.",
            json!({
                "properties": {
                    "sparql": {"type":"string","description":"SPARQL SELECT query statement."},
                    "named_graph": {"type":"string","description":"Named graph IRI (optional)."}
                },
                "required": ["sparql"]
            }),
            Arc::new(move |input: Value| {
                let kg_store = kg_store_for_query.clone();
                Box::pin(async move { builtins::execute_knowledge_query(input, kg_store).await })
            }),
            all,
        );

        let kg_store_for_search = self.kg_store.clone();
        self.register("kg_search", "Fuzzy search entities in the knowledge graph.", json!({
            "properties": {
                "keyword": {"type":"string","description":"Search keyword."},
                "entity_type": {"type":"string","description":"Entity type IRI filter (optional)."}
            },
            "required": ["keyword"]
        }), Arc::new(move |input: Value| {
            let kg_store = kg_store_for_search.clone();
            Box::pin(async move { builtins::execute_knowledge_search(input, kg_store).await })
        }), all);

        let kg_store_for_neighbors = self.kg_store.clone();
        self.register(
            "knowledge_neighbors",
            "Get neighbor nodes and relations of a specified entity in the knowledge graph.",
            json!({
                "properties": {
                    "entity_id": {"type":"string","description":"Entity ID or IRI."},
                    "depth": {"type":"integer","description":"Traversal depth (1-3, default 1)."}
                },
                "required": ["entity_id"]
            }),
            Arc::new(move |input: Value| {
                let kg_store = kg_store_for_neighbors.clone();
                Box::pin(
                    async move { builtins::execute_knowledge_neighbors(input, kg_store).await },
                )
            }),
            all,
        );

        let kg_store_for_import = self.kg_store.clone();
        self.register("knowledge_import_json", "Map structured JSON data into knowledge graph nodes.", json!({
            "properties": {
                "json_data": {"type":"string","description":"JSON data (object or array)."},
                "mapping_config": {"type":"string","description":"Mapping config JSON: {id_field, type_field, label_field, relations:[{field, relation, target_prefix}]}."}
            },
            "required": ["json_data","mapping_config"]
        }), Arc::new(move |input: Value| {
            let kg_store = kg_store_for_import.clone();
            Box::pin(async move { builtins::execute_knowledge_import_json(input, kg_store).await })
        }), all);

        let kg_store_for_ontology = self.kg_store.clone();
        self.register("ontology_register", "Register custom ontology classes or properties to the knowledge graph.", json!({
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "iri": {"type":"string","description":"Ontology term IRI."},
                            "label": {"type":"string","description":"Term label."},
                            "description": {"type":"string","description":"Term description."},
                            "term_type": {"type":"string","description":"Type: Class | Property | Relation."}
                        },
                        "required": ["iri","label","description","term_type"]
                    },
                    "description": "Ontology term list."
                }
            },
            "required": ["terms"]
        }), Arc::new(move |input: Value| {
            let kg_store = kg_store_for_ontology.clone();
            Box::pin(async move { builtins::execute_ontology_register(input, kg_store).await })
        }), all);

        let kg_store_for_bridge = self.kg_store.clone();
        self.register("knowledge_bridge", "Create bridge relations between knowledge graph entities and skills.", json!({
            "properties": {
                "entity_id": {"type":"string","description":"Entity ID."},
                "skill_iri": {"type":"string","description":"Skill IRI."},
                "relation_type": {"type":"string","description":"Relation type: HasSkill | ApplicableIn | RelatedTo (default HasSkill)."}
            },
            "required": ["entity_id","skill_iri"]
        }), Arc::new(move |input: Value| {
            let kg_store = kg_store_for_bridge.clone();
            Box::pin(async move { builtins::execute_knowledge_bridge_with_store(input, kg_store).await })
        }), all);

        let kg_store_for_code = self.kg_store.clone();
        self.register("knowledge_extract_code", "Extract AST structure (functions, classes, imports, call relations etc.) from code files using tree-sitter and write to knowledge graph. Supports incremental updates: skips unchanged files automatically. Supports Rust/Python/JS/TS/Go/Java/C/C++.", json!({
            "properties": {
                "file_path": {"type":"string","description":"Code file path."},
                "named_graph": {"type":"string","description":"Named graph IRI (optional, default graph:code)."},
                "force": {"type":"boolean","description":"Force full extraction, ignore cache (optional, default false)."}
            },
            "required": ["file_path"]
        }), Arc::new(move |input: Value| {
            let kg_store = kg_store_for_code.clone();
            Box::pin(async move {
                let input = resolve_workspace_path_argument(input, "file_path")?;
                builtins::execute_knowledge_extract_code(input, kg_store).await
            })
        }), all);

        // ========== L3 Projection Query Tool ==========
        let proj_for_tool = self.projection_engine.clone();
        self.register("read_agent_output", "Read an archived AgentTurn owned by the current Agent/L1 session or explicitly granted by a typed task handoff (iri://task/.../session/.../turn_N; an exact granted legacy iri://task/.../turn_N is also readable). Continue with next_char_offset on the same IRI. Ephemeral tool-result IRIs must be read only through the exact session reader advertised in the current turn.", json!({
            "properties": {
                "node_iri": {"type":"string","description":"Stable AgentTurn archive IRI from the explicit task handoff."},
                "char_offset": {"type":"integer","description":"Starting Unicode character for an AgentTurn IRI (default 0)."},
                "char_limit": {"type":"integer","description":"Maximum Unicode characters for an AgentTurn page (default 4000, maximum 6000)."}
            },
            "required": ["node_iri"]
        }), Arc::new(move |input: Value| {
            let proj = proj_for_tool.clone();
            Box::pin(async move {
                let node_iri = input
                    .get("node_iri")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "Missing node_iri parameter".to_string())?;
                if node_iri.starts_with("iri://tool-result/") {
                    return Err(
                        "Tool-result IRIs are session-scoped and cannot be read through read_agent_output; use only the exact result reader advertised in the current turn"
                            .to_string(),
                    );
                }
                let guard = proj.read();
                let engine = guard.as_ref()
                    .ok_or_else(|| "Projection engine not initialized".to_string())?;
                let result = engine.read_node(node_iri)
                    .map_err(|e| format!("Failed to read L2 node: {}", e))?;
                match result {
                    Some(mut node) => {
                        if node.get("@id").and_then(Value::as_str) != Some(node_iri)
                            || node.get("@type").and_then(Value::as_str) != Some("AgentTurn")
                        {
                            return Err(format!(
                                "Node is not an exact archived AgentTurn: {node_iri}"
                            ));
                        }
                        if let Some(page) = agent_turn_content_page(&mut node, &input, node_iri) {
                            return Ok(page);
                        }
                        let redacted = redact_session_tool_references(&mut node);
                        if redacted > 0 {
                            if let Some(object) = node.as_object_mut() {
                                object.insert(
                                    "session_tool_references_redacted".to_string(),
                                    Value::Number((redacted as u64).into()),
                                );
                            }
                        }
                        Ok(node)
                    }
                    None => Err(format!("Node not found: {}", node_iri)),
                }
            })
        }), all);

        // ========== Ontology Tools ==========
        #[cfg(feature = "ontology")]
        {
            self.register(
                "ontology_validate_turtle",
                "Validate Turtle RDF syntax. Returns number of valid triples.",
                json!({
                    "properties": {
                        "ttl": {"type":"string","description":"Turtle content to validate"}
                    },
                    "required": ["ttl"]
                }),
                Arc::new(|input: Value| {
                    Box::pin(async move {
                        ontology_tools::execute_ontology_validate_turtle(input).await
                    })
                }),
                all,
            );

            self.register(
                "ontology_lint_turtle",
                "Lint Turtle content for best practices (labels, comments, domain/range).",
                json!({
                    "properties": {
                        "ttl": {"type":"string","description":"Turtle content to lint"}
                    },
                    "required": ["ttl"]
                }),
                Arc::new(|input: Value| {
                    Box::pin(
                        async move { ontology_tools::execute_ontology_lint_turtle(input).await },
                    )
                }),
                all,
            );

            self.register(
                "ontology_diff_turtle",
                "Diff two Turtle documents and report added/removed triples.",
                json!({
                    "properties": {
                        "old_ttl": {"type":"string","description":"Original Turtle content"},
                        "new_ttl": {"type":"string","description":"New Turtle content"}
                    },
                    "required": ["old_ttl","new_ttl"]
                }),
                Arc::new(|input: Value| {
                    Box::pin(
                        async move { ontology_tools::execute_ontology_diff_turtle(input).await },
                    )
                }),
                all,
            );

            self.register("ontology_validate_shacl", "Validate RDF data against SHACL shapes.", json!({
                "properties": {
                    "shapes_ttl": {"type":"string","description":"SHACL shapes in Turtle format"},
                    "data_ttl": {"type":"string","description":"Optional data Turtle to validate. If omitted, validates loaded store."}
                },
                "required": ["shapes_ttl"]
            }), Arc::new(|input: Value| Box::pin(async move { ontology_tools::execute_ontology_validate_shacl(input).await })), all);

            self.register("ontology_reason", "Run RDFS/OWL-RL reasoning on Turtle data. Returns inferred triples.", json!({
                "properties": {
                    "ttl": {"type":"string","description":"Turtle data to reason over"},
                    "profile": {"type":"string","description":"Reasoning profile: rdfs, owl-rl (default), owl-rl-ext, owl-dl"},
                    "materialize": {"type":"boolean","description":"Whether to materialize inferred triples (default: true)"}
                },
                "required": ["ttl"]
            }), Arc::new(|input: Value| Box::pin(async move { ontology_tools::execute_ontology_reason(input).await })), all);
        }
    }

    fn valid_external_registration_component(value: &str) -> bool {
        !value.is_empty() && value.chars().count() <= 256 && !value.chars().any(char::is_control)
    }

    pub fn is_evidence_critical_builtin_name(name: &str) -> bool {
        EVIDENCE_CRITICAL_BUILTIN_NAMES.contains(&name)
    }

    fn upsert_registered_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        handler: ToolFn,
        allowed_roles: &[&str],
        provenance: ToolProvenance,
    ) {
        let roles: Vec<String> = allowed_roles.iter().map(|s| s.to_string()).collect();
        if self.tools.contains_key(name) && !Self::is_micro_tool_name(name) {
            warn!(
                tool = name,
                old_provenance = self
                    .tool_provenance
                    .get(name)
                    .map(ToolProvenance::label)
                    .unwrap_or("unknown"),
                new_provenance = provenance.label(),
                "trusted tool registration overwrites existing handler with same name"
            );
        }
        // A crate-trusted replacement must completely replace any contextual
        // execution path associated with its name. Externals never reach this
        // point when the name is already owned.
        self.contextual_tools.remove(name);
        self.tools.insert(name.to_string(), handler);
        self.tool_provenance.insert(name.to_string(), provenance);

        if let Some(existing) = self.tool_descriptions.iter_mut().find(|td| td.name == name) {
            existing.description = description.to_string();
            existing.parameters = parameters.clone();
            existing.allowed_roles = roles;
        } else {
            self.tool_descriptions.push(ToolDescription {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
                allowed_roles: roles,
            });
            // Micro-tool description cap: removes oldest when exceeded
            if Self::is_micro_tool_name(name) {
                while self
                    .tool_descriptions
                    .iter()
                    .filter(|td| Self::is_micro_tool_name(&td.name))
                    .count()
                    > self.max_micro_tool_descriptions
                {
                    // position() returns the first match (oldest registered)
                    if let Some(pos) = self
                        .tool_descriptions
                        .iter()
                        .position(|td| Self::is_micro_tool_name(&td.name))
                    {
                        self.tool_descriptions.remove(pos);
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Crate-trusted registration used by built-ins and explicitly controlled
    /// runtime fixtures. External/application/MCP sources must use
    /// `register_external`, which cannot replace a kernel-owned name.
    pub(crate) fn register(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        handler: ToolFn,
        allowed_roles: &[&str],
    ) {
        let provenance = if self.registering_builtins {
            ToolProvenance::Builtin
        } else {
            ToolProvenance::TrustedInternal
        };
        self.upsert_registered_tool(
            name,
            description,
            parameters,
            handler,
            allowed_roles,
            provenance,
        );
    }

    /// Register an externally supplied handler without allowing identity
    /// spoofing or last-writer-wins replacement. Callers should namespace the
    /// public name (MCP uses `mcp__<server>__<tool>`); the namespace is also
    /// retained as provenance for diagnostics and collision decisions.
    pub fn register_external(
        &mut self,
        name: &str,
        provenance_namespace: &str,
        description: &str,
        parameters: Value,
        handler: ToolFn,
        allowed_roles: &[&str],
    ) -> Result<(), ToolRegistrationError> {
        if !Self::valid_external_registration_component(name)
            || !name
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "_-".contains(character))
        {
            return Err(ToolRegistrationError::InvalidName {
                name: name.to_string(),
            });
        }
        if !Self::valid_external_registration_component(provenance_namespace) {
            return Err(ToolRegistrationError::InvalidNamespace {
                namespace: provenance_namespace.to_string(),
            });
        }
        if Self::is_evidence_critical_builtin_name(name) {
            return Err(ToolRegistrationError::ReservedBuiltinName {
                name: name.to_string(),
            });
        }
        if self.tools.contains_key(name)
            || self.tool_descriptions.iter().any(|tool| tool.name == name)
        {
            return Err(ToolRegistrationError::NameConflict {
                name: name.to_string(),
                existing_provenance: self
                    .tool_provenance
                    .get(name)
                    .map(ToolProvenance::label)
                    .unwrap_or("unknown")
                    .to_string(),
            });
        }
        self.upsert_registered_tool(
            name,
            description,
            parameters,
            handler,
            allowed_roles,
            ToolProvenance::External {
                namespace: provenance_namespace.to_string(),
            },
        );
        Ok(())
    }

    fn register_contextual_handler(&mut self, name: &str, handler: ContextualToolFn) {
        debug_assert!(
            self.tools.contains_key(name),
            "contextual handler requires a registered public tool"
        );
        self.contextual_tools.insert(name.to_string(), handler);
    }

    pub(crate) fn is_micro_tool_name(name: &str) -> bool {
        crate::tools::result_router::is_session_scoped_micro_tool_name(name)
    }

    /// Register micro-tool (dynamically generated tool for querying large tool results)
    pub fn register_micro_tool(&mut self, tool_name: &str, context: MicroToolContext) {
        if !Self::is_micro_tool_name(tool_name) {
            warn!(
                tool = tool_name,
                "rejected non-session-scoped name at micro-tool registration boundary"
            );
            return;
        }
        let contexts = Arc::clone(&self.micro_tool_contexts);
        let data = Arc::clone(&self.micro_tool_data);
        let consumption = Arc::clone(&self.archived_reader_consumption);
        let tool_name_owned = tool_name.to_string();
        let default_page_size = self.micro_tool_page_size;
        let max_page_size = self.micro_tool_max_page_size;

        contexts
            .write()
            .insert(tool_name.to_string(), context.clone());
        // Registration is the capability-creation boundary.  A name is
        // normally unique, but clearing stale state here keeps explicit
        // re-registration deterministic in tests and recovery paths.
        self.archived_reader_consumption.write().remove(tool_name);

        let reader_view = archived_reader_view(
            &context,
            self.micro_tool_data.read().get(&context.storage_key),
        );
        let description = if tool_name.starts_with("read_full_result_") {
            let cursor_contract = match reader_view {
                crate::tools::result_router::micro_tools::ArchivedReaderView::FileLines => {
                    "Use the exact source-line next_cursor (offset/limit/char_offset)."
                }
                crate::tools::result_router::micro_tools::ArchivedReaderView::ExecutionStream => {
                    "Defaults to stdout; use stream plus next_cursor.char_offset. offset/limit are invalid."
                }
                crate::tools::result_router::micro_tools::ArchivedReaderView::RawText => {
                    "Use next_cursor.char_offset; offset/limit are invalid."
                }
            };
            format!(
                "Read a bounded typed view of an archived tool result. {cursor_contract} provider_call_id: {}",
                context.provider_call_id
            )
        } else if tool_name.starts_with("query_") {
            format!(
                "Query entity types: {:?}. provider_call_id: {}",
                context.entity_types, context.provider_call_id
            )
        } else if tool_name.starts_with("get_entity_details_") {
            format!(
                "Get entity details. provider_call_id: {}",
                context.provider_call_id
            )
        } else {
            format!("Micro-tool: {}", tool_name)
        };

        let params = micro_tool_parameters(
            tool_name,
            reader_view,
            max_page_size,
            context.preview_size.max(1),
        );

        self.register(
            tool_name,
            &description,
            params,
            Arc::new(move |input: Value| {
                let contexts = contexts.clone();
                let tool_name_owned = tool_name_owned.clone();
                let data = data.clone();
                let consumption = consumption.clone();
                Box::pin(async move {
                    let ctx_guard = contexts.read();
                    let ctx = ctx_guard.get(&tool_name_owned).ok_or_else(|| {
                        format!("Micro-tool context not found: {}", tool_name_owned)
                    })?;

                    let data_guard = data.read();
                    let stored_data = data_guard
                        .get(&ctx.storage_key)
                        .ok_or_else(|| format!("Micro-tool data not found: {}", ctx.storage_key))?;

                    if tool_name_owned.starts_with("read_full_result_") {
                        let page = execute_archived_result_reader(
                            &input,
                            ctx,
                            stored_data,
                            default_page_size,
                            max_page_size,
                        )?;
                        let archive_content = stored_data
                            .get("content")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                format!("Micro-tool data has no text content: {}", ctx.storage_key)
                            })?;
                        return Ok(record_archived_reader_delivery(
                            &consumption,
                            &tool_name_owned,
                            ctx,
                            archive_content,
                            page,
                        ));
                    } else if tool_name_owned.starts_with("query_") {
                        if let Some(content) = stored_data.get("content").and_then(|v| v.as_str()) {
                            let offset = input["offset"].as_u64().unwrap_or(0) as usize;
                            let limit = (input["limit"]
                                .as_u64()
                                .unwrap_or(default_page_size as u64)
                                as usize)
                                .min(max_page_size);
                            let query_type = ctx.entity_types.first().map(String::as_str).unwrap_or("");
                            let filter_property = input["filter_property"].as_str().unwrap_or("");
                            let filter_value = input.get("filter_value");

                            let matching = serde_json::from_str::<serde_json::Value>(content)
                                .ok()
                                .and_then(|parsed| parsed.as_array().cloned())
                                .unwrap_or_default()
                                .into_iter()
                                .filter(|item| {
                                    (query_type.is_empty()
                                        || archived_item_matches_entity_type(item, query_type))
                                        && archived_item_matches_property(
                                            item,
                                            filter_property,
                                            filter_value,
                                        )
                                })
                                .collect::<Vec<_>>();
                            let total_matches = matching.len();
                            let payload_budget = ctx.preview_size.clamp(512, 8 * 1024);
                            let mut results = Vec::new();
                            let mut payload_bytes = 0usize;
                            let mut consumed = 0usize;
                            for item in matching.iter().skip(offset).take(limit) {
                                let encoded = item.to_string();
                                let needed = encoded.len().saturating_add(1);
                                if payload_bytes.saturating_add(needed) > payload_budget {
                                    if results.is_empty() {
                                        results.push(json!({
                                            "json_preview": crate::utils::text::safe_truncate(
                                                &encoded,
                                                payload_budget.saturating_sub(128).max(64),
                                            ),
                                            "truncated": true,
                                        }));
                                        consumed = 1;
                                    }
                                    break;
                                }
                                results.push(item.clone());
                                payload_bytes += needed;
                                consumed += 1;
                            }
                            let next_offset = offset
                                .saturating_add(consumed)
                                .lt(&total_matches)
                                .then_some(offset.saturating_add(consumed));
                            return Ok(json!({
                                "results": results,
                                "count": results.len(),
                                "total_matches": total_matches,
                                "offset": offset,
                                "next_offset": next_offset,
                                "entity_type": query_type,
                                "call_id": ctx.provider_call_id,
                            }));
                        }
                    } else if tool_name_owned.starts_with("get_entity_details_") {
                        let entity_id = input["entity_id"].as_str().unwrap_or("");
                        if let Some(content) = stored_data.get("content").and_then(|v| v.as_str()) {
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content) {
                                if let Some(arr) = parsed.as_array() {
                                    for item in arr {
                                        if archived_item_matches_id(item, entity_id) {
                                            let encoded = item.to_string();
                                            let char_offset = input["char_offset"]
                                                .as_u64()
                                                .unwrap_or(0) as usize;
                                            let total_chars = encoded.chars().count();
                                            if total_chars > ctx.preview_size {
                                                let page = encoded
                                                    .chars()
                                                    .skip(char_offset)
                                                    .take(ctx.preview_size.max(1))
                                                    .collect::<String>();
                                                let next = char_offset
                                                    .saturating_add(page.chars().count());
                                                return Ok(json!({
                                                    "entity_json_page": page,
                                                    "entity_id": entity_id,
                                                    "total_chars": total_chars,
                                                    "char_offset": char_offset,
                                                    "next_char_offset": (next < total_chars).then_some(next),
                                                    "truncated": true,
                                                    "call_id": ctx.provider_call_id,
                                                }));
                                            }
                                            return Ok(json!({
                                                "entity": item,
                                                "call_id": ctx.provider_call_id,
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                        return Ok(json!({
                            "error": "Entity not found",
                            "entity_id": entity_id,
                            "call_id": ctx.provider_call_id,
                        }));
                    }

                    Err(format!("Unsupported session result tool: {tool_name_owned}"))
                })
            }),
            &[],
        );
    }

    /// Seed duplicate-free reader progress from the exact cursor exposed in
    /// an inline routed preview. A reader can only be invoked on a later model
    /// request, after that preview has crossed the provider boundary, so this
    /// is the authenticated starting prefix for the same Agent/L1 capability.
    /// Arbitrary model-supplied offsets are never accepted as seed material.
    pub(crate) fn seed_archived_reader_progress_from_preview(
        &self,
        reader_name: &str,
        preview: &str,
    ) -> bool {
        let Ok(preview) = serde_json::from_str::<Value>(preview) else {
            return false;
        };
        let Some(context) = self.micro_tool_contexts.read().get(reader_name).cloned() else {
            return false;
        };
        let Some(archive_content) = self
            .micro_tool_data
            .read()
            .get(&context.storage_key)
            .and_then(|stored| stored.get("content"))
            .and_then(Value::as_str)
            .map(str::to_string)
        else {
            return false;
        };
        let reader_view = crate::tools::result_router::micro_tools::ArchivedReaderView::for_result(
            &context.tool_name,
            &archive_content,
        );
        let mut consumption = self.archived_reader_consumption.write();
        let reader = consumption.entry(reader_name.to_string()).or_default();
        match reader_view {
            crate::tools::result_router::micro_tools::ArchivedReaderView::ExecutionStream => {
                let Some(cursors) = preview.get("reader_cursors").and_then(Value::as_object) else {
                    return false;
                };
                let Ok(envelope) = serde_json::from_str::<Value>(&archive_content) else {
                    return false;
                };
                let mut seeded = false;
                for stream in ["stdout", "stderr", "command"] {
                    let Some(cursor_value) = cursors.get(stream) else {
                        continue;
                    };
                    let selected = if stream == "command" {
                        envelope.get("command").or_else(|| envelope.get("code"))
                    } else {
                        envelope.get(stream)
                    }
                    .and_then(Value::as_str);
                    let Some(selected) = selected else {
                        continue;
                    };
                    let total_chars = selected.chars().count();
                    let char_offset = match cursor_value {
                        Value::Null => None,
                        Value::Object(cursor) => {
                            let Some(offset) = cursor
                                .get("char_offset")
                                .and_then(Value::as_u64)
                                .and_then(|offset| usize::try_from(offset).ok())
                            else {
                                continue;
                            };
                            Some(offset)
                        }
                        _ => continue,
                    };
                    if char_offset.is_some_and(|offset| offset > total_chars) {
                        continue;
                    }
                    let view_key = format!("execution_stream:{stream}");
                    let completed_receipt = char_offset.is_none().then(|| {
                        let page = json!({
                            "reader_view": "execution_stream",
                            "stream": stream,
                            "total_chars": total_chars,
                            "exit_code": envelope.get("exit_code").cloned().unwrap_or(Value::Null),
                            "duration_ms": envelope.get("duration_ms").cloned().unwrap_or(Value::Null),
                        });
                        archived_reader_completion_receipt(
                            &context,
                            &archive_content,
                            &page,
                            &view_key,
                        )
                    });
                    reader
                        .views
                        .entry(view_key)
                        .or_insert(ArchivedReaderViewConsumption {
                            progress: ArchivedReaderProgress::Characters {
                                contiguous_end: char_offset.unwrap_or(total_chars),
                                seeded_from_preview: true,
                                preview_replay_used: false,
                            },
                            completed_receipt,
                        });
                    seeded = true;
                }
                reader.retired = archived_reader_can_retire(reader, &context, &archive_content);
                seeded
            }
            crate::tools::result_router::micro_tools::ArchivedReaderView::FileLines => {
                let Some(cursor) = preview.get("reader_cursor").and_then(Value::as_object) else {
                    return false;
                };
                let Some(next_offset) = cursor
                    .get("offset")
                    .and_then(Value::as_u64)
                    .and_then(|offset| usize::try_from(offset).ok())
                else {
                    return false;
                };
                let next_char_offset = cursor
                    .get("char_offset")
                    .and_then(Value::as_u64)
                    .and_then(|offset| usize::try_from(offset).ok())
                    .unwrap_or(0);
                let continuation_limit = (next_char_offset > 0)
                    .then(|| {
                        cursor
                            .get("limit")
                            .and_then(Value::as_u64)
                            .and_then(|limit| usize::try_from(limit).ok())
                    })
                    .flatten();
                if next_char_offset > 0 && continuation_limit.is_none_or(|limit| limit == 0) {
                    return false;
                }
                let Ok(envelope) = serde_json::from_str::<Value>(&archive_content) else {
                    return false;
                };
                let source_offset = envelope
                    .get("offset")
                    .and_then(Value::as_u64)
                    .and_then(|offset| usize::try_from(offset).ok())
                    .unwrap_or(0);
                let archived_end = source_offset.saturating_add(
                    envelope
                        .get("lines")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or(0),
                );
                if next_offset < source_offset || next_offset > archived_end {
                    return false;
                }
                if next_char_offset > 0 {
                    let limit = continuation_limit.unwrap_or(1);
                    let local_offset = next_offset.saturating_sub(source_offset);
                    let selected_text = envelope
                        .get("lines")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .skip(local_offset)
                        .take(limit)
                        .map(|line| {
                            line.as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| line.to_string())
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if next_char_offset > selected_text.chars().count() {
                        return false;
                    }
                }
                reader.views.entry("file_lines".to_string()).or_insert(
                    ArchivedReaderViewConsumption {
                        progress: ArchivedReaderProgress::FileLines {
                            started: true,
                            next_offset,
                            next_char_offset,
                            continuation_limit,
                            seeded_from_preview: true,
                            preview_replay_used: false,
                        },
                        completed_receipt: None,
                    },
                );
                true
            }
            crate::tools::result_router::micro_tools::ArchivedReaderView::RawText => false,
        }
    }

    /// Store micro-tool data
    pub fn store_micro_tool_data(&self, storage_key: &str, data: serde_json::Value) {
        self.micro_tool_data
            .write()
            .insert(storage_key.to_string(), data);
    }

    /// Get list of registered micro-tools
    pub fn get_micro_tool_names(&self) -> Vec<String> {
        self.micro_tool_contexts.read().keys().cloned().collect()
    }

    /// Report whether one exact session-scoped micro-tool was generated from
    /// the named source tool. This consults the composite routing context;
    /// callers must not infer origin from a provider call ID or schema shape.
    pub(crate) fn micro_tool_originates_from(&self, name: &str, source_tool: &str) -> bool {
        self.micro_tool_contexts
            .read()
            .get(name)
            .is_some_and(|context| context.tool_name == source_tool)
    }

    /// Return only dynamic tools generated for one concrete tool call. This
    /// lets AgentRunner keep micro-tool visibility scoped to a BizAgent
    /// execution without deleting globally archived handlers that another
    /// concurrently running agent may still need.
    pub fn get_micro_tool_names_for_routing_key(&self, routing_call_key: &str) -> Vec<String> {
        self.micro_tool_contexts
            .read()
            .iter()
            .filter(|(_, context)| context.routing_call_key == routing_call_key)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Retire only the ephemeral tools owned by one completed L1 execution.
    /// `ToolExecutor` is shared by concurrently running BizAgent children, so
    /// a global clear would revoke live capabilities. Session-scoped routing
    /// identities let us remove the exact handlers, schemas, contexts and
    /// in-memory payloads whose authority has ended while leaving siblings
    /// untouched. Persistent L0 records are intentionally not deleted here.
    pub fn remove_micro_tools_for_session(&mut self, session_id: &str) -> usize {
        let session_scope =
            crate::tools::result_router::ResultRoutingIdentity::new(session_id, "cleanup")
                .session_scope;
        let routing_prefix = format!("{session_scope}_c");
        let retired = self
            .micro_tool_contexts
            .read()
            .iter()
            .filter(|(_, context)| context.routing_call_key.starts_with(&routing_prefix))
            .map(|(name, context)| (name.clone(), context.storage_key.clone()))
            .collect::<Vec<_>>();
        if retired.is_empty() {
            return 0;
        }

        let retired_names = retired
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<HashSet<_>>();
        let candidate_storage_keys = retired
            .iter()
            .map(|(_, storage_key)| storage_key.clone())
            .collect::<HashSet<_>>();

        self.micro_tool_contexts
            .write()
            .retain(|name, _| !retired_names.contains(name));
        self.archived_reader_consumption
            .write()
            .retain(|name, _| !retired_names.contains(name));
        self.tools.retain(|name, _| !retired_names.contains(name));
        self.tool_provenance
            .retain(|name, _| !retired_names.contains(name));
        self.contextual_tools
            .retain(|name, _| !retired_names.contains(name));
        self.tool_descriptions
            .retain(|description| !retired_names.contains(&description.name));

        // Be defensive if a future routing mode intentionally shares one
        // payload across several scoped readers: retain data still referenced
        // by any live context.
        let still_referenced = self
            .micro_tool_contexts
            .read()
            .values()
            .map(|context| context.storage_key.clone())
            .collect::<HashSet<_>>();
        self.micro_tool_data.write().retain(|storage_key, _| {
            !candidate_storage_keys.contains(storage_key) || still_referenced.contains(storage_key)
        });
        retired_names.len()
    }

    /// Reconstruct one active session micro-tool schema even when its catalog
    /// description was evicted by the process-wide prompt-size cap.  Handler
    /// and context lifetime are deliberately longer than catalog visibility;
    /// AgentRunner still decides which owning BizAgent session may advertise
    /// this schema.
    pub fn micro_tool_definition(&self, name: &str) -> Option<Value> {
        let retired = self
            .archived_reader_consumption
            .read()
            .get(name)
            .is_some_and(|state| state.retired);
        if retired {
            return None;
        }
        self.micro_tool_definition_inner(name)
    }

    /// Reconstruct a consumed reader schema only while its native tool call is
    /// still present in the provider history of the owning Agent/L1. Some
    /// OpenAI-compatible providers validate historical tool-call names against
    /// the current `tools` array and return HTTP 400 if the schema disappears
    /// immediately after the terminal reader page. The handler remains
    /// receipt-only after retirement, so retaining this schema grants no
    /// replay capability and ends automatically when context compaction drops
    /// the historical call.
    pub(crate) fn micro_tool_definition_for_history(&self, name: &str) -> Option<Value> {
        self.micro_tool_definition_inner(name)
    }

    fn micro_tool_definition_inner(&self, name: &str) -> Option<Value> {
        let context = self.micro_tool_contexts.read().get(name)?.clone();
        self.try_get_handler(name)?;
        let reader_view = archived_reader_view(
            &context,
            self.micro_tool_data.read().get(&context.storage_key),
        );
        let description = if name.starts_with("read_full_result_") {
            format!(
                "Read a bounded page of archived tool result {}",
                context.provider_call_id
            )
        } else if name.starts_with("query_") {
            format!(
                "Query archived result entities for {}",
                context.provider_call_id
            )
        } else {
            format!("Read archived result data for {}", context.provider_call_id)
        };
        Some(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": micro_tool_parameters(
                    name,
                    reader_view,
                    self.micro_tool_max_page_size,
                    context.preview_size.max(1),
                )
            }
        }))
    }

    pub async fn execute(&self, name: &str, input: Value) -> Result<Value, String> {
        self.execute_internal(
            name,
            input,
            None,
            None,
            None,
            ToolExecutionProfile::Standard,
        )
        .await
    }

    fn nested_llm_scope(
        name: &str,
        security_context: Option<&SecurityContext>,
    ) -> Option<crate::llm::LlmInteractionScope> {
        let context = security_context?;
        let stage = match name {
            "create_skill" => "skill_creation",
            "convert_skill" => "skill_markdown_conversion",
            _ => return None,
        };
        let mut scope = crate::llm::LlmInteractionScope::new(stage)
            .with_agent(context.agent_id.clone(), context.agent_role.clone());
        if let Some(task_iri) = context.task_iri.as_deref() {
            scope = scope.with_task(task_iri.to_string());
        }
        if let Some(invocation) = context.llm_invocation.as_ref() {
            scope = scope
                .with_usage_scope(invocation.usage_scope_iri.clone())
                .with_parent(invocation.parent_interaction_id.clone());
            if !invocation.cycle_id.is_empty() {
                scope = scope.with_cycle(invocation.cycle_id.clone());
            }
        }
        Some(scope)
    }

    fn enforce_workspace_resource_lease(
        name: &str,
        input: &mut Value,
        security_context: Option<&SecurityContext>,
    ) -> Option<Value> {
        let lease = security_context?.workspace_resource_lease.as_ref()?;
        match lease.resolve_authorized_file_tool_path(name, input) {
            Ok(Some(target)) => {
                let Some(target) = target.to_str() else {
                    return Some(json!({
                        "error": "Workspace resource lease resolved to a non-UTF-8 path",
                        "tool": name,
                        "lease_id": lease.lease_id,
                        "reason": "workspace_resource_lease_violation",
                    }));
                };
                if let Some(object) = input.as_object_mut() {
                    object.insert("path".to_string(), Value::String(target.to_string()));
                    None
                } else {
                    Some(json!({
                        "error": "Workspace resource lease requires an object tool input",
                        "tool": name,
                        "lease_id": lease.lease_id,
                        "reason": "workspace_resource_lease_violation",
                    }))
                }
            }
            Ok(None) => None,
            Err(reason) => Some(json!({
                "error": format!("Workspace resource lease rejected tool call: {reason}"),
                "tool": name,
                "lease_id": lease.lease_id,
                "reason": "workspace_resource_lease_violation",
            })),
        }
    }

    fn enforce_agent_turn_read_capability(
        input: &Value,
        security_context: Option<&SecurityContext>,
    ) -> Result<(), Value> {
        let Some(node_iri) = input.get("node_iri").and_then(Value::as_str) else {
            return Err(json!({
                "error": "Security denied: read_agent_output requires an exact node_iri",
                "tool": "read_agent_output",
                "reason": "agent_turn_capability_required",
            }));
        };
        let Some(context) = security_context else {
            return Err(json!({
                "error": "Security denied: read_agent_output requires a trusted runtime context",
                "tool": "read_agent_output",
                "reason": "agent_turn_capability_required",
            }));
        };
        if !context.permits_agent_turn_read(node_iri) {
            return Err(json!({
                "error": "Security denied: AgentTurn is outside the active L1 and typed handoff scope",
                "tool": "read_agent_output",
                "reason": "agent_turn_capability_required",
                "node_iri": node_iri,
            }));
        }
        Ok(())
    }

    fn reject_reserved_internal_fields(input: &Value) -> Result<(), String> {
        if input
            .as_object()
            .is_some_and(|object| object.keys().any(|key| key.starts_with("__gh_")))
        {
            return Err(
                "reserved internal fields prefixed with '__gh_' are not allowed".to_string(),
            );
        }
        Ok(())
    }

    fn normalize_file_baseline_path(
        &self,
        path: &str,
        security_context: Option<&SecurityContext>,
    ) -> String {
        let candidate = std::path::PathBuf::from(path);
        if !candidate.is_absolute() {
            if let Some(root) = security_context
                .and_then(|context| context.workspace_resource_lease.as_ref())
                .map(|lease| lease.workspace_root.as_path())
            {
                let joined = root.join(&candidate);
                return std::fs::canonicalize(&joined)
                    .unwrap_or(joined)
                    .to_string_lossy()
                    .to_string();
            }
        }
        let monitor = self.workspace_monitor.read().clone();
        if let Some(monitor) = monitor {
            return monitor.normalize_path(path);
        }
        let joined = if candidate.is_absolute() {
            candidate
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join(candidate)
        };
        std::fs::canonicalize(&joined)
            .unwrap_or(joined)
            .to_string_lossy()
            .to_string()
    }

    fn expected_file_overwrite_baseline(
        &self,
        security_context: Option<&SecurityContext>,
        path: Option<&str>,
    ) -> Option<String> {
        let context = security_context?;
        let path = path?;
        let target = self.normalize_file_baseline_path(path, security_context);
        let mut expected = None;
        for event in context.file_overwrite_baseline_events() {
            let _source_identity = event.source_call_identity.as_ref();
            match event.path.as_deref() {
                None => expected = None,
                Some(event_path)
                    if self.normalize_file_baseline_path(event_path, Some(context)) == target =>
                {
                    expected = event.content_sha256.clone();
                }
                Some(_) => {}
            }
        }
        expected
    }

    fn is_complete_file_read_request(input: &Value) -> bool {
        let starts_at_beginning = input.get("offset").and_then(Value::as_u64).unwrap_or(0) == 0;
        let has_limit = input.get("limit").is_some_and(|value| !value.is_null());
        let complete_mode = !matches!(
            input.get("mode").and_then(Value::as_str),
            Some("diff" | "changed_only")
        );
        starts_at_beginning && !has_limit && complete_mode
    }

    async fn execute_internal(
        &self,
        name: &str,
        input: Value,
        security_context: Option<&SecurityContext>,
        allowed_tools: Option<&[String]>,
        effect_policy: Option<&crate::core::effect::EffectPolicy>,
        execution_profile: ToolExecutionProfile,
    ) -> Result<Value, String> {
        let mut input = input;
        if let Err(error) = Self::reject_reserved_internal_fields(&input) {
            return Ok(json!({
                "error": format!("Tool arguments rejected: {error}"),
                "tool": name,
                "reason": "reserved_internal_field",
            }));
        }
        if let Some(allowed) = allowed_tools {
            if !self.allowlist_permits(name, allowed) {
                return Ok(json!({"error": format!("Tool not allowed: {}", name), "tool": name}));
            }
        }

        if let Some(rejection) = background_workspace_mutation_rejection(name, &input) {
            return Ok(rejection);
        }

        if effect_policy.is_some_and(|policy| {
            !policy.permits_mutation()
                && crate::core::effect::is_workspace_mutation_candidate(name, &input)
        }) {
            return Ok(json!({
                "error": format!("Effect policy rejected mutating tool call: {name}"),
                "tool": name,
            }));
        }

        if let Some(rejection) =
            Self::enforce_workspace_resource_lease(name, &mut input, security_context)
        {
            return Ok(rejection);
        }

        // Preserve the original fail-closed order: a denied tool cannot use an
        // external hook as a side channel. If a hook later mutates the input,
        // both permission and contextual security are evaluated again.
        if let Some(context) = security_context {
            if let Err(error) = self.enforce_contextual_security(name, context).await {
                return Ok(error);
            }
        }

        // Reject an unauthorized stable archive before external hooks observe
        // its IRI, then repeat this check below against any hook-rewritten
        // input. Argument-aware authorization must bracket that trust boundary.
        if name == "read_agent_output" {
            if input
                .get("node_iri")
                .and_then(Value::as_str)
                .is_some_and(|node_iri| node_iri.starts_with("iri://tool-result/"))
            {
                return Err(
                    "Tool-result IRIs are session-scoped and cannot be read through read_agent_output; use only the exact result reader advertised in the current turn"
                        .to_string(),
                );
            }
            if let Err(rejection) =
                Self::enforce_agent_turn_read_capability(&input, security_context)
            {
                return Ok(rejection);
            }
        }

        let original_input_str = input.to_string();
        if let Some(ref policy) = self.permission_policy {
            if let PermissionOutcome::Deny { reason } =
                policy.authorize(name, &original_input_str, None)
            {
                return Ok(json!({"error": format!("Permission denied: {}", reason)}));
            }
        }

        let mut effective_input = input;
        let mut effective_input_str = original_input_str;
        let mut hook_permission_override = None;
        let mut hook_permission_reason = None;
        let mut hook_updated_input = false;

        if let Some(ref runner) = self.hook_runner {
            let hook_result = runner.run_pre_tool_use(name, &effective_input_str);
            if hook_result.is_denied() {
                return Ok(
                    json!({"error": format!("Pre-tool hook denied: {}", hook_result.messages().join("; "))}),
                );
            }
            if hook_result.is_failed() {
                return Ok(json!({
                    "error": format!("Pre-tool hook failed: {}", hook_result.messages().join("; ")),
                    "hook_timed_out": hook_result.is_timed_out(),
                }));
            }
            if hook_result.is_cancelled() {
                return Ok(json!({"error": "Pre-tool hook was cancelled"}));
            }

            hook_permission_override = hook_result.permission_override();
            hook_permission_reason = hook_result.permission_reason().map(ToOwned::to_owned);
            if let Some(updated_input) = hook_result.updated_input() {
                if self.hook_allow_input_rewrite {
                    effective_input = match self.validate_hook_updated_input(name, updated_input) {
                        Ok(input) => input,
                        Err(error) => {
                            return Ok(json!({
                                "error": format!("Pre-tool hook updatedInput rejected: {error}"),
                                "tool": name,
                            }));
                        }
                    };
                    effective_input_str = effective_input.to_string();
                    hook_updated_input = true;
                } else {
                    warn!(
                        tool = name,
                        "external hook updatedInput ignored because input rewriting is disabled"
                    );
                }
            }
        }

        // A hook may add `run_in_background` or turn a read-only command into
        // a mutating one. Re-evaluate the final arguments before any handler
        // can escape the runner's synchronous delta-settlement window.
        if let Some(rejection) = background_workspace_mutation_rejection(name, &effective_input) {
            return Ok(rejection);
        }

        let permission_context =
            PermissionContext::new(hook_permission_override, hook_permission_reason.clone());
        if let Some(ref policy) = self.permission_policy {
            if let PermissionOutcome::Deny { reason } =
                policy.authorize_with_context(name, &effective_input_str, &permission_context, None)
            {
                return Ok(
                    json!({"error": format!("Permission denied after pre-tool hook: {}", reason)}),
                );
            }
        } else {
            // A hook-level deny/ask must never disappear merely because the
            // optional static permission policy is disabled.
            match hook_permission_override {
                Some(PermissionOverride::Deny) => {
                    return Ok(json!({
                        "error": hook_permission_reason.unwrap_or_else(|| {
                            format!("Permission denied by pre-tool hook for '{name}'")
                        })
                    }));
                }
                Some(PermissionOverride::Ask) => {
                    return Ok(json!({
                        "error": hook_permission_reason.unwrap_or_else(|| {
                            format!("Approval required by pre-tool hook for '{name}'")
                        })
                    }));
                }
                Some(PermissionOverride::Allow) | None => {}
            }
        }

        if hook_updated_input {
            if effect_policy.is_some_and(|policy| {
                !policy.permits_mutation()
                    && (crate::core::effect::is_workspace_mutation_candidate(
                        name,
                        &effective_input,
                    ) || matches!(name, "bash" | "powershell" | "code_execute"))
            }) {
                return Ok(json!({
                    "error": format!(
                        "Effect policy rejected hook-modified input for mutation-capable tool: {name}"
                    ),
                    "tool": name,
                }));
            }
            if let Some(context) = security_context {
                // SecurityEngine currently decides on skill identity and
                // caller context rather than arguments. Re-running it is still
                // intentional: future argument-aware policies cannot be
                // bypassed when this boundary evolves.
                if let Err(error) = self.enforce_contextual_security(name, context).await {
                    return Ok(error);
                }
            }

            // The clean-verification profile was selected from the exact
            // SkillBefore arguments. An external Hook rewrite invalidates
            // that parse; do not silently apply privileged execution
            // semantics to a different command.
            if execution_profile.is_clean_verification() {
                return Ok(json!({
                    "error": "Clean verification profile invalidated by external pre-tool argument rewrite",
                    "tool": name,
                    "reason": "verification_profile_arguments_changed",
                }));
            }
        }

        // External pre-tool hooks may rewrite `path`. Re-evaluate the exact
        // lease against the final handler input so Modify cannot escape the
        // parent's resource partition.
        if let Some(rejection) =
            Self::enforce_workspace_resource_lease(name, &mut effective_input, security_context)
        {
            return Ok(rejection);
        }

        if name == "read_agent_output" {
            if effective_input
                .get("node_iri")
                .and_then(Value::as_str)
                .is_some_and(|node_iri| node_iri.starts_with("iri://tool-result/"))
            {
                return Err(
                    "Tool-result IRIs are session-scoped and cannot be read through read_agent_output; use only the exact result reader advertised in the current turn"
                        .to_string(),
                );
            }
            if let Err(rejection) =
                Self::enforce_agent_turn_read_capability(&effective_input, security_context)
            {
                return Ok(rejection);
            }
        }

        // Tool discovery must query the live executor catalog, including MCP
        // and application tools registered after built-ins. The historical
        // built-in handler contains only a five-item static fallback and made
        // on-demand tool groups impossible to activate in practice.
        if name == "tool_search" {
            let query = effective_input
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let max_results = effective_input
                .get("max_results")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            return Ok(self.search_tools(query, max_results));
        }

        if let Some(ref gate) = self.syscall_gate {
            if let Err(e) = gate.validate_tool_with_5w2h(name, "unknown", None) {
                return Ok(json!({"error": format!("SyscallGate rejected: {}", e)}));
            }
        }

        // Use the same lookup path as the agent loops so dynamically derived
        // micro-tools retain their fallback semantics when execution goes
        // through policy/gate enforcement.
        let handler = match self.try_get_handler(name) {
            Some(h) => h,
            None => return Err(format!("Tool not found: {}", name)),
        };
        debug!(tool = %name, "Executing tool");

        // Execute and capture result for post-hooks
        let final_file_path = effective_input
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_string);
        let complete_file_read =
            name == "file_read" && Self::is_complete_file_read_request(&effective_input);
        let read_exposure_key = security_context.and_then(|context| {
            final_file_path
                .as_ref()
                .map(|path| format!("{}\n{path}", context.file_read_visibility_scope()))
        });
        let expected_file_baseline =
            self.expected_file_overwrite_baseline(security_context, final_file_path.as_deref());
        let mut handler_input = effective_input;
        if name == "file_read" {
            // This internal-only tag is applied after external hooks and schema
            // validation so a hook cannot forge or exfiltrate session identity.
            if let Some(context) = security_context {
                if let Some(object) = handler_input.as_object_mut() {
                    object.insert(
                        "__gh_read_session".to_string(),
                        Value::String(context.file_read_visibility_scope()),
                    );
                }
            }
        } else if matches!(name, "file_write" | "file_edit") {
            // Existing-file changes are authorized only by a complete read
            // returned to this exact Agent/L1. Both values are injected after
            // hooks and lease resolution, so model JSON cannot forge them.
            if let Some(object) = handler_input.as_object_mut() {
                object.insert(
                    "__gh_require_overwrite_baseline".to_string(),
                    Value::Bool(true),
                );
                if let Some(expected) = expected_file_baseline {
                    object.insert(
                        "__gh_expected_current_sha256".to_string(),
                        Value::String(expected),
                    );
                }
            }
        }
        if execution_profile.is_clean_verification() {
            if !matches!(name, "bash" | "powershell") {
                return Ok(json!({
                    "error": "Clean verification profile is supported only for shell verifiers",
                    "tool": name,
                    "reason": "verification_profile_tool_mismatch",
                }));
            }
            let Some(object) = handler_input.as_object_mut() else {
                return Ok(json!({
                    "error": "Clean verification profile requires object tool arguments",
                    "tool": name,
                    "reason": "verification_profile_arguments_invalid",
                }));
            };
            object.insert(
                "__gh_execution_profile".to_string(),
                serde_json::to_value(execution_profile)
                    .expect("ToolExecutionProfile is always JSON serializable"),
            );
        }
        // Dynamic skill tools contain a nested model request. Their built-in
        // contextual handler receives only kernel-issued correlation metadata;
        // model JSON with similar field names cannot affect it. Only an
        // explicitly crate-trusted replacement can remove this privileged
        // handler; external registration fails closed on every collision.
        let result = match self.contextual_tools.get(name).cloned() {
            Some(contextual_handler) => {
                contextual_handler(
                    handler_input,
                    Self::nested_llm_scope(name, security_context),
                )
                .await
            }
            None => handler(handler_input).await,
        };

        // Post-tool-use hook
        if let Some(ref runner) = self.hook_runner {
            match &result {
                Ok(output) => {
                    let output_str = output.to_string();
                    let post_result =
                        runner.run_post_tool_use(name, &effective_input_str, &output_str, false);
                    if post_result.is_denied() {
                        return Ok(json!({
                            "error": "Tool result withheld by post-execution hook policy",
                            "tool": name,
                            "post_hook_denied": true
                        }));
                    }
                }
                Err(e) => {
                    let _ = runner.run_post_tool_use_failure(name, &effective_input_str, e);
                }
            }
        }

        // Cache visibility only after ToolExecutor's own post hook allows the
        // result. Overwrite authorization is intentionally *not* recorded
        // here: AgentRunner first routes the payload and confirms that a later
        // provider request actually consumed it, then supplies the resulting
        // kernel-only baseline events through SecurityContext.
        if let Ok(output) = &result {
            if complete_file_read && output.get("lines").is_some() {
                if let Some(key) = read_exposure_key {
                    self.file_read_exposures.write().insert(key);
                }
            }
        }

        result
    }

    pub async fn execute_with_security_context(
        &self,
        name: &str,
        input: Value,
        context: SecurityContext,
        allowed_tools: Option<&[String]>,
    ) -> Result<Value, String> {
        self.execute_internal(
            name,
            input,
            Some(&context),
            allowed_tools,
            None,
            ToolExecutionProfile::Standard,
        )
        .await
    }

    /// Execute with the complete task policy. This is the AgentRunner entry
    /// point: external Hook `updatedInput` is re-evaluated against the same
    /// effect boundary after schema/permission validation.
    pub async fn execute_with_security_context_and_effect_policy(
        &self,
        name: &str,
        input: Value,
        context: SecurityContext,
        allowed_tools: Option<&[String]>,
        effect_policy: &crate::core::effect::EffectPolicy,
    ) -> Result<Value, String> {
        self.execute_internal(
            name,
            input,
            Some(&context),
            allowed_tools,
            Some(effect_policy),
            ToolExecutionProfile::Standard,
        )
        .await
    }

    /// Execute a kernel-classified verifier with an internal execution
    /// profile. The marker is not part of the model-visible tool schema and is
    /// injected only after Hook, permission, effect and lease validation.
    pub(crate) async fn execute_with_security_context_effect_policy_and_profile(
        &self,
        name: &str,
        input: Value,
        context: SecurityContext,
        allowed_tools: Option<&[String]>,
        effect_policy: &crate::core::effect::EffectPolicy,
        execution_profile: ToolExecutionProfile,
    ) -> Result<Value, String> {
        self.execute_internal(
            name,
            input,
            Some(&context),
            allowed_tools,
            Some(effect_policy),
            execution_profile,
        )
        .await
    }

    /// Execute arguments produced by the internal `SkillBefore` typed patch
    /// contract. Schema and reserved-field checks happen before the canonical
    /// execution path re-evaluates allowlists, effect policy, permission rules,
    /// contextual security and external pre-tool Hooks against the new value.
    pub async fn execute_hook_modified_with_security_context_and_effect_policy(
        &self,
        name: &str,
        input: Value,
        context: SecurityContext,
        allowed_tools: Option<&[String]>,
        effect_policy: &crate::core::effect::EffectPolicy,
    ) -> Result<Value, String> {
        if let Err(error) = self.validate_modified_tool_input(name, &input) {
            return Ok(json!({
                "error": format!("SkillBefore arguments patch rejected: {error}"),
                "tool": name,
            }));
        }
        self.execute_internal(
            name,
            input,
            Some(&context),
            allowed_tools,
            Some(effect_policy),
            ToolExecutionProfile::Standard,
        )
        .await
    }

    /// SkillBefore equivalent of
    /// [`Self::execute_with_security_context_effect_policy_and_profile`].
    pub(crate) async fn execute_hook_modified_with_security_context_effect_policy_and_profile(
        &self,
        name: &str,
        input: Value,
        context: SecurityContext,
        allowed_tools: Option<&[String]>,
        effect_policy: &crate::core::effect::EffectPolicy,
        execution_profile: ToolExecutionProfile,
    ) -> Result<Value, String> {
        if let Err(error) = self.validate_modified_tool_input(name, &input) {
            return Ok(json!({
                "error": format!("SkillBefore arguments patch rejected: {error}"),
                "tool": name,
            }));
        }
        self.execute_internal(
            name,
            input,
            Some(&context),
            allowed_tools,
            Some(effect_policy),
            execution_profile,
        )
        .await
    }

    async fn enforce_contextual_security(
        &self,
        name: &str,
        context: &SecurityContext,
    ) -> Result<(), Value> {
        let security_engine = { self.security_engine.read().clone() };
        if let Some(engine) = security_engine {
            let skill_iri = {
                let registry = self.shared_skill_registry.read();
                registry
                    .as_ref()
                    .and_then(|registry| registry.skill_iri_for_tool_name(name))
            }
            .or_else(|| builtin_security_skill_iri(name).map(str::to_string))
            // Generated result readers expose no independent side effect. They
            // inherit the least-privilege built-in read capability instead of
            // becoming an unregistered security bypass.
            .or_else(|| {
                Self::is_micro_tool_name(name).then(|| "iri://skills/file_read".to_string())
            });
            let Some(skill_iri) = skill_iri else {
                return Err(
                    json!({"error": "Security denied: tool has no registered executable skill", "tool": name}),
                );
            };
            match engine.check_execution(&skill_iri, context).await {
                Ok(SecurityDecision::Allowed) => {}
                Ok(SecurityDecision::Denied { reasons }) => {
                    return Err(
                        json!({"error": "Security denied", "tool": name, "skill_iri": skill_iri, "reasons": reasons}),
                    );
                }
                Ok(SecurityDecision::RequiresApproval { approver, reason }) => {
                    return Err(
                        json!({"error": "Security approval required", "tool": name, "skill_iri": skill_iri, "approver": approver, "reason": reason}),
                    );
                }
                Err(error) => {
                    return Err(
                        json!({"error": format!("Security denied: {error}"), "tool": name, "skill_iri": skill_iri}),
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_hook_updated_input(&self, name: &str, raw_input: &str) -> Result<Value, String> {
        const MAX_HOOK_UPDATED_INPUT_BYTES: usize = 1_048_576;
        if raw_input.len() > MAX_HOOK_UPDATED_INPUT_BYTES {
            return Err(format!(
                "payload exceeds {} byte limit",
                MAX_HOOK_UPDATED_INPUT_BYTES
            ));
        }
        let value: Value =
            serde_json::from_str(raw_input).map_err(|error| format!("invalid JSON: {error}"))?;
        self.validate_modified_tool_input(name, &value)?;
        Ok(value)
    }

    fn validate_modified_tool_input(&self, name: &str, value: &Value) -> Result<(), String> {
        if !value.is_object() {
            return Err("tool input must be a top-level JSON object".to_string());
        }
        Self::reject_reserved_internal_fields(value)?;

        let mut schema = self
            .tool_descriptions
            .iter()
            .find(|description| description.name == name)
            .map(|description| description.parameters.clone())
            .or_else(|| {
                self.micro_tool_definition(name)
                    .and_then(|definition| definition.pointer("/function/parameters").cloned())
            })
            .ok_or_else(|| format!("no input schema is registered for tool '{name}'"))?;
        // Several legacy registrations rely on the OpenAI function wrapper to
        // imply an object schema. Enforce that invariant locally before
        // validating hook-originated data.
        if let Some(schema_object) = schema.as_object_mut() {
            if !schema_object.contains_key("type")
                && (schema_object.contains_key("properties")
                    || schema_object.contains_key("required"))
            {
                schema_object.insert("type".to_string(), Value::String("object".to_string()));
            }
        }
        let compiled = jsonschema::JSONSchema::options()
            .compile(&schema)
            .map_err(|error| format!("registered schema for '{name}' is invalid: {error}"))?;
        if let Err(errors) = compiled.validate(value) {
            let messages = errors
                .take(5)
                .map(|error| format!("{}: {error}", error.instance_path))
                .collect::<Vec<_>>();
            return Err(format!(
                "input does not satisfy schema for '{name}': {}",
                messages.join("; ")
            ));
        }
        Ok(())
    }

    /// Exact allowlist matching plus the read-only micro-tools generated by
    /// result routing.  A caller that grants `file_read` may consume those
    /// archived read results; no write capability is implied.
    pub fn explicit_allowlist_permits(name: &str, allowed: &[String]) -> bool {
        allowed.iter().any(|tool| tool == name)
            || (name == "read_agent_output" && allowed.iter().any(|tool| tool == "file_read"))
    }

    /// A session result tool inherits only the source tool's explicit grant.
    /// This lets a `web_search`-only child page its own routed result without
    /// granting it arbitrary file access or another AgentInstance's reader.
    pub fn allowlist_permits(&self, name: &str, allowed: &[String]) -> bool {
        if Self::explicit_allowlist_permits(name, allowed) {
            return true;
        }
        if !Self::is_micro_tool_name(name) {
            return false;
        }
        self.micro_tool_contexts
            .read()
            .get(name)
            .is_some_and(|context| allowed.iter().any(|tool| tool == &context.tool_name))
    }

    /// Get tool handler (avoid holding lock across await)
    #[cfg(test)]
    pub(crate) fn get_handler(&self, name: &str) -> Option<ToolFn> {
        self.tools.get(name).cloned()
    }

    /// Get tool handler with micro-tool fallback.
    /// When normal lookup fails, dynamically build a handler from micro-tool data storage,
    /// preventing LLM from exhausting turns due to registry/handler inconsistency.
    pub(crate) fn try_get_handler(&self, name: &str) -> Option<ToolFn> {
        // 1. Try registered handler first
        if let Some(handler) = self.tools.get(name) {
            return Some(handler.clone());
        }
        // 2. Fallback: build dynamic handler from stored data for read_full_result_* micro-tools
        if name.starts_with("read_full_result_") {
            return self.make_micro_tool_fallback_handler(name);
        }
        None
    }

    /// Build a dynamic fallback handler for micro-tools (reads from micro_tool_data / micro_tool_contexts)
    fn make_micro_tool_fallback_handler(&self, name: &str) -> Option<ToolFn> {
        let ctx_guard = self.micro_tool_contexts.read();
        let ctx = ctx_guard.get(name)?.clone();
        let storage_key = ctx.storage_key.clone();
        drop(ctx_guard);

        let data_guard = self.micro_tool_data.read();
        let stored_data = data_guard.get(&storage_key)?.clone();
        drop(data_guard);
        let default_page_size = self.micro_tool_page_size;
        let max_page_size = self.micro_tool_max_page_size;
        let reader_name = name.to_string();
        let consumption = Arc::clone(&self.archived_reader_consumption);

        Some(Arc::new(move |input: Value| {
            let _storage_key = storage_key.clone();
            let ctx = ctx.clone();
            let stored_data = stored_data.clone();
            let reader_name = reader_name.clone();
            let consumption = consumption.clone();

            Box::pin(async move {
                let page = execute_archived_result_reader(
                    &input,
                    &ctx,
                    &stored_data,
                    default_page_size,
                    max_page_size,
                )?;
                let archive_content = stored_data
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        format!("Micro-tool data has no text content: {}", ctx.storage_key)
                    })?;
                Ok(record_archived_reader_delivery(
                    &consumption,
                    &reader_name,
                    &ctx,
                    archive_content,
                    page,
                ))
            })
        }))
    }

    /// List tool names visible to the given role (role-filtered definitions).
    pub fn list_tools(&self, role: &str) -> Vec<String> {
        self.tool_definitions_for_role(role)
            .into_iter()
            .filter_map(|td| td["function"]["name"].as_str().map(|s| s.to_string()))
            .collect()
    }

    /// Return all tool definitions (LLM autonomously selects based on role description in agent.md)
    pub fn tool_definitions_for_role(&self, role: &str) -> Vec<Value> {
        if matches!(role, "AA" | "Act") {
            // AA receives CA evidence through BizAgent context and only emits
            // a terminal decision. Never advertise tools on a lower-level
            // Runner path either, so callers cannot bypass the BizAgent rule.
            return Vec::new();
        }
        let role_name = match role {
            "PA" | "Plan" => "Plan",
            "DA" | "Do" => "Do",
            "CA" | "Check" => "Check",
            "AA" | "Act" => "Act",
            _ => role,
        };

        let (default_tools, on_demand_tools) = if self.tool_group_manager.is_some() {
            // This method is the complete role-authorized catalog. Default
            // window filtering is applied by visible_tool_definitions_for_role;
            // keeping the full catalog here allows tool_search to activate
            // on-demand and late-registered MCP tools.
            let all: HashSet<String> = self
                .tool_descriptions
                .iter()
                .map(|td| td.name.clone())
                .collect();
            (all.clone(), all)
        } else {
            let is_pa = role == "Plan" || role == "PA";
            if is_pa {
                let default: HashSet<String> = Self::pa_readonly_tools()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                (default.clone(), default)
            } else {
                let all: HashSet<String> = self
                    .tool_descriptions
                    .iter()
                    .map(|td| td.name.clone())
                    .collect();
                (all.clone(), all)
            }
        };

        let retired_readers = self
            .archived_reader_consumption
            .read()
            .iter()
            .filter(|(_, state)| state.retired)
            .map(|(name, _)| name.clone())
            .collect::<HashSet<_>>();

        let result: Vec<Value> = self
            .tool_descriptions
            .iter()
            .filter(|td| {
                if retired_readers.contains(&td.name) {
                    return false;
                }
                let parsed_role = match role_name {
                    "Plan" => Some(crate::core::agent_instance::AgentRole::Plan),
                    "Do" => Some(crate::core::agent_instance::AgentRole::Do),
                    "Check" => Some(crate::core::agent_instance::AgentRole::Check),
                    "Act" => Some(crate::core::agent_instance::AgentRole::Act),
                    _ => None,
                };
                if parsed_role.map(|role| {
                    crate::core::tool_controller::business_role_allows_tool(role, &td.name)
                }) == Some(false)
                {
                    return false;
                }
                if !td.allowed_roles.is_empty() {
                    return td.allowed_roles.iter().any(|r| r == role || r == role_name);
                }
                default_tools.contains(&td.name) || on_demand_tools.contains(&td.name)
            })
            .map(|td| {
                let mut params = td.parameters.clone();
                if params.get("type").is_none() {
                    params["type"] = json!("object");
                }
                json!({
                    "type": "function",
                    "function": {
                        "name": td.name,
                        "description": td.description,
                        "parameters": params,
                    }
                })
            })
            .collect();

        let tool_names: Vec<&str> = result
            .iter()
            .filter_map(|v| v["function"]["name"].as_str())
            .collect();
        tracing::debug!(
            "[tool_definitions_for_role] role={}, filtered={}/{}, tools={:?}",
            role,
            result.len(),
            self.tool_descriptions.len(),
            tool_names
        );

        result
    }

    /// Return only the default tool window for a role. On-demand tools remain
    /// discoverable through `tool_search` and can be registered dynamically;
    /// they are not sent to the model on every request. The historical
    /// `tool_definitions_for_role` API intentionally remains unchanged for
    /// compatibility with callers that need the complete role-allowed set.
    pub fn visible_tool_definitions_for_role(&self, role: &str) -> Vec<Value> {
        let role_name = match role {
            "PA" | "Plan" => "Plan",
            "DA" | "Do" => "Do",
            "CA" | "Check" => "Check",
            "AA" | "Act" => "Act",
            _ => role,
        };
        let Some(manager) = self.tool_group_manager.as_ref() else {
            // Keep the legacy fallback behavior when no explicit group
            // manager exists; built-in registrations do not carry the
            // default/on-demand distinction needed for safe filtering.
            return self.tool_definitions_for_role(role);
        };
        let (default_tools, _) = manager.get_tool_names_for_role(role_name);
        self.tool_definitions_for_role(role)
            .into_iter()
            .filter(|td| {
                td["function"]["name"]
                    .as_str()
                    .map(|name| default_tools.contains(name) || name == "tool_search")
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Role-filtered tool definitions intersected with an explicit allowlist.
    /// `None` keeps the full role-filtered set. `Some(empty)` is an explicit
    /// deny-all capability set. A non-empty list is intersected with the role
    /// set (SA/task policy may narrow but never broaden role authority).
    pub fn tool_definitions_for_role_with_allowlist(
        &self,
        role: &str,
        allowlist: Option<&[String]>,
    ) -> Vec<Value> {
        let result = self.tool_definitions_for_role(role);
        let Some(allowed) = allowlist else {
            return result;
        };
        let allowed: HashSet<&str> = allowed.iter().map(|s| s.as_str()).collect();
        result
            .into_iter()
            .filter(|td| {
                td["function"]["name"]
                    .as_str()
                    .map(|n| allowed.contains(n))
                    .unwrap_or(false)
            })
            .collect()
    }

    pub fn pa_readonly_tools() -> &'static [&'static str] {
        &[
            "file_read",
            "file_list",
            "glob_search",
            "grep_search",
            "web_search",
            "web_fetch",
            "tool_search",
            "rag_search",
            "knowledge_list",
            "knowledge_search",
            "kg_search",
            "knowledge_extract_code",
            "read_agent_output",
            "bash",
        ]
    }

    pub fn is_pa_readonly_tool(name: &str) -> bool {
        Self::pa_readonly_tools().contains(&name) || Self::is_micro_tool_name(name)
    }

    /// ToolSearch needs access to the tool list
    pub fn search_tools(&self, query: &str, max_results: Option<usize>) -> Value {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower
            .split(|character: char| !character.is_alphanumeric())
            .filter(|term| !term.is_empty())
            .collect();
        let max = max_results.unwrap_or(10);
        let matches: Vec<Value> = self
            .tool_descriptions
            .iter()
            .filter(|t| {
                let searchable = format!(
                    "{} {}",
                    t.name.to_lowercase().replace('_', " "),
                    t.description.to_lowercase()
                );
                query_lower.is_empty()
                    || searchable.contains(&query_lower)
                    || query_terms.iter().all(|term| searchable.contains(term))
            })
            .take(max)
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                })
            })
            .collect();
        json!({
            "matches": matches,
            "count": matches.len(),
            "query": query,
        })
    }
}
