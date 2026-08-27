use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
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
use crate::tools::builtin::permissions::{PermissionMode, PermissionOutcome, PermissionPolicy};
use crate::tools::builtin::rag;
use crate::tools::skill_registry::SkillRegistry;
use crate::tools::tool_groups::ToolGroupManager;
use crate::tools::workspace_monitor::{FileState, WorkspaceMonitor};

mod builtins;

#[cfg(test)]
mod tests;

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
type ToolFn =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> + Send + Sync>;

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

/// Micro-tool context
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MicroToolContext {
    pub call_id: String,
    pub storage_key: String,
    pub tool_name: String,
    pub entity_types: Vec<String>,
    pub preview_size: usize,
}

/// Unified tool executor with built-in tools
#[derive(Clone)]
pub struct ToolExecutor {
    tools: HashMap<String, ToolFn>,
    tool_descriptions: Vec<ToolDescription>,
    kg_store: Arc<std::sync::RwLock<KnowledgeGraphStore>>,
    projection_engine:
        Arc<parking_lot::RwLock<Option<Arc<crate::memory::l3_projection::ProjectionEngine>>>>,
    /// Persistent backing for archived tool-result IRIs.  The handle is
    /// attached by AgentRunner after construction so the generic executor can
    /// keep its standalone default while `read_agent_output` can resolve both
    /// agent-turn and tool-result IRIs through one bounded read contract.
    archived_result_store: Arc<parking_lot::RwLock<Option<Arc<crate::memory::l0_store::L0Store>>>>,
    micro_tool_contexts: Arc<parking_lot::RwLock<HashMap<String, MicroToolContext>>>,
    micro_tool_data: Arc<parking_lot::RwLock<HashMap<String, serde_json::Value>>>,
    syscall_gate: Option<crate::core::syscall_gate::SyscallGate>,
    permission_policy: Option<PermissionPolicy>,
    hook_runner: Option<HookRunner>,
    tool_group_manager: Option<ToolGroupManager>,
    workspace_monitor: Arc<parking_lot::RwLock<Option<Arc<WorkspaceMonitor>>>>,
    shared_skill_graph: Arc<parking_lot::RwLock<Option<Arc<SkillGraphStore>>>>,
    shared_skill_registry: Arc<parking_lot::RwLock<Option<Arc<SkillRegistry>>>>,
    shared_skill_vector_store: Arc<parking_lot::RwLock<Option<Arc<HyperspaceStore>>>>,
    shared_skill_creator_gateway:
        Arc<parking_lot::RwLock<Option<Arc<crate::gateway::unified_gateway::UnifiedGateway>>>>,
    security_engine: Arc<parking_lot::RwLock<Option<Arc<SecurityEngine>>>>,
    /// Whole-file reads whose content has actually been exposed to one
    /// BizAgent execution context.  The WorkspaceMonitor cache is shared by
    /// PA/DA/CA, but their LLM contexts are not; a global cache hit therefore
    /// cannot imply that a different agent still has the content in context.
    file_read_exposures: Arc<parking_lot::RwLock<HashSet<String>>>,
    max_micro_tool_descriptions: usize,
    micro_tool_page_size: usize,
    micro_tool_max_page_size: usize,
}

const MICRO_TOOL_PREFIXES: &[&str] = &[
    "read_full_result_",
    "query_",
    "get_entity_details",
    "expand_relation",
];

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
            tool_descriptions: Vec::new(),
            kg_store,
            projection_engine: Arc::new(parking_lot::RwLock::new(None)),
            archived_result_store: Arc::new(parking_lot::RwLock::new(None)),
            micro_tool_contexts: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            micro_tool_data: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            syscall_gate: None,
            permission_policy: None,
            hook_runner: None,
            tool_group_manager: None,
            workspace_monitor: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_graph: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_registry: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_vector_store: Arc::new(parking_lot::RwLock::new(None)),
            shared_skill_creator_gateway: Arc::new(parking_lot::RwLock::new(None)),
            security_engine: Arc::new(parking_lot::RwLock::new(None)),
            file_read_exposures: Arc::new(parking_lot::RwLock::new(HashSet::new())),
            max_micro_tool_descriptions: 5,
            micro_tool_page_size: 100,
            micro_tool_max_page_size: 200,
        };
        exe.register_builtins();
        exe
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

    pub fn set_archived_result_store(&mut self, store: Arc<crate::memory::l0_store::L0Store>) {
        *self.archived_result_store.write() = Some(store);
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

    /// Inject the application gateway used by dynamic skill creation. This is
    /// executor-local so independent workspaces do not share a process-global
    /// creator gateway.
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
                let read_session = input.get("__gh_read_session")
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let exposure_key = read_session.as_ref()
                    .map(|session| format!("{session}\n{path}"));
                // Calls made through the legacy context-free execute() API
                // retain the historical global-cache behavior. Contextual
                // BizAgent calls require evidence that this exact agent/task
                // context has already received the whole file.
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
                let exposed_whole_file = !has_offset && !has_limit && result.get("lines").is_some();
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
                                                "Cache hit: file unchanged since last read. Content already in your context — skip re-reading.".to_string()
                                            ));
                                        });
                                    }
                                }
                                if exposed_whole_file {
                                    if let Some(key) = exposure_key {
                                        read_exposures.write().insert(key);
                                    }
                                }
                                return Ok(result);
                            }
                        }
                    }
                if exposed_whole_file {
                    if let Some(key) = exposure_key {
                        read_exposures.write().insert(key);
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
        }), Arc::new(|input: Value| Box::pin(async move { knowledge::execute_knowledge_import_file(input).await })), all);

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
        }), Arc::new(|input: Value| Box::pin(async move { knowledge::execute_knowledge_import_directory(input).await })), all);

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
        let gateway_for_create = self.shared_skill_creator_gateway.clone();
        self.register("create_skill", "Create a new Skill definition from natural language using LLM. The definition is registered for review and discovery; it does not create an executable ToolExecutor handler.", json!({
            "properties": {
                "description": {"type":"string","description":"Natural language description of the skill to create"},
                "skill_name_hint": {"type":"string","description":"Suggested skill name (optional, lowercase with underscores)"},
                "category_hint": {"type":"string","description":"Suggested category (optional): file|network|ai|execution|validation|data|meta|system"},
                "security_level_override": {"type":"string","description":"Security level override (optional): low|normal|high|critical"}
            },
            "required": ["description"]
        }), Arc::new(move |input: Value| {
            let sg = sg_for_create.read().clone();
            let registry = registry_for_create.read().clone();
            let vector_store = vector_store_for_create.read().clone();
            let gateway = gateway_for_create.read().clone();
            Box::pin(async move { builtins::execute_create_skill(input, gateway, sg, registry, vector_store).await })
        }), &["DA"]);

        let sg_for_convert = self.shared_skill_graph.clone();
        let registry_for_convert = self.shared_skill_registry.clone();
        let vector_store_for_convert = self.shared_skill_vector_store.clone();
        let gateway_for_convert = self.shared_skill_creator_gateway.clone();
        self.register("convert_skill", "Convert a Markdown-formatted skill description into a JSON-LD Skill definition. Parses the markdown structure and generates proper skill schema.", json!({
            "properties": {
                "markdown_content": {"type":"string","description":"Markdown content describing the skill"},
                "source_path": {"type":"string","description":"Source file path (optional)"}
            },
            "required": ["markdown_content"]
        }), Arc::new(move |input: Value| {
            let sg = sg_for_convert.read().clone();
            let registry = registry_for_convert.read().clone();
            let vector_store = vector_store_for_convert.read().clone();
            let gateway = gateway_for_convert.read().clone();
            Box::pin(async move { builtins::execute_convert_skill(input, gateway, sg, registry, vector_store).await })
        }), &["DA","CA"]);

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
            Box::pin(async move { builtins::execute_knowledge_extract_code(input, kg_store).await })
        }), all);

        // ========== L3 Projection Query Tool ==========
        let proj_for_tool = self.projection_engine.clone();
        let archived_store_for_tool = self.archived_result_store.clone();
        let archived_memory_for_tool = self.micro_tool_data.clone();
        self.register("read_agent_output", "Read archived output by IRI. Agent report IRIs (iri://task/.../turn_N) are projected from L2; tool-result IRIs (iri://tool-result/...) are returned as a bounded page from the result archive. Prefer the exact read_full_result_* session tool when it is advertised.", json!({
            "properties": {
                "node_iri": {"type":"string","description":"Agent report or archived tool-result IRI."},
                "offset": {"type":"integer","description":"Starting line for a tool-result IRI (default 0)."},
                "limit": {"type":"integer","description":"Maximum lines for a tool-result IRI (default 100, maximum 200)."}
            },
            "required": ["node_iri"]
        }), Arc::new(move |input: Value| {
            let proj = proj_for_tool.clone();
            let archived_store = archived_store_for_tool.clone();
            let archived_memory = archived_memory_for_tool.clone();
            Box::pin(async move {
                let node_iri = input
                    .get("node_iri")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "Missing node_iri parameter".to_string())?;
                if node_iri.starts_with("iri://tool-result/") {
                    let archived = archived_memory.read().get(node_iri).cloned().or_else(|| {
                        archived_store
                            .read()
                            .as_ref()
                            .and_then(|store| store.retrieve(node_iri).ok().flatten())
                            .and_then(|entry| serde_json::from_str::<Value>(&entry.content).ok())
                    });
                    let archived = archived
                        .ok_or_else(|| format!("Archived tool result not found: {}", node_iri))?;
                    let content = archived
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or_else(|| archived.as_str().unwrap_or_default());
                    let offset = input.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
                    let limit = input
                        .get("limit")
                        .and_then(Value::as_u64)
                        .unwrap_or(100)
                        .min(200) as usize;
                    let lines = content.lines().collect::<Vec<_>>();
                    let selected = lines
                        .iter()
                        .skip(offset)
                        .take(limit)
                        .copied()
                        .collect::<Vec<_>>();
                    return Ok(json!({
                        "content": selected.join("\n"),
                        "total_lines": lines.len(),
                        "offset": offset,
                        "returned": selected.len(),
                        "iri": node_iri,
                    }));
                }
                let guard = proj.read();
                let engine = guard.as_ref()
                    .ok_or_else(|| "Projection engine not initialized".to_string())?;
                let result = engine.read_node(node_iri)
                    .map_err(|e| format!("Failed to read L2 node: {}", e))?;
                match result {
                    Some(node) => Ok(node),
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

    /// Register a tool with role whitelist. Empty = all roles allowed.
    pub fn register(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        handler: ToolFn,
        allowed_roles: &[&str],
    ) {
        let roles: Vec<String> = allowed_roles.iter().map(|s| s.to_string()).collect();
        if self.tools.contains_key(name) && !Self::is_micro_tool_name(name) {
            warn!(
                tool = name,
                "tool registration overwrites existing handler with same name"
            );
        }
        self.tools.insert(name.to_string(), handler);

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

    pub(crate) fn is_micro_tool_name(name: &str) -> bool {
        MICRO_TOOL_PREFIXES.iter().any(|p| name.starts_with(p))
    }

    /// Register micro-tool (dynamically generated tool for querying large tool results)
    pub fn register_micro_tool(&mut self, tool_name: &str, context: MicroToolContext) {
        let contexts = Arc::clone(&self.micro_tool_contexts);
        let data = Arc::clone(&self.micro_tool_data);
        let tool_name_owned = tool_name.to_string();
        let default_page_size = self.micro_tool_page_size;
        let max_page_size = self.micro_tool_max_page_size;

        contexts
            .write()
            .insert(tool_name.to_string(), context.clone());

        let description = if tool_name.starts_with("read_full_result_") {
            format!("Read full tool result. call_id: {}", context.call_id)
        } else if tool_name.starts_with("query_") {
            format!(
                "Query entity types: {:?}. call_id: {}",
                context.entity_types, context.call_id
            )
        } else if tool_name.starts_with("get_entity_details_") {
            format!("Get entity details. call_id: {}", context.call_id)
        } else {
            format!("Micro-tool: {}", tool_name)
        };

        let params = json!({
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "description": "Starting offset"},
                "limit": {"type": "integer", "description": "Max results to return"}
            }
        });

        self.register(
            tool_name,
            &description,
            params,
            Arc::new(move |input: Value| {
                let contexts = contexts.clone();
                let tool_name_owned = tool_name_owned.clone();
                let data = data.clone();
                Box::pin(async move {
                    let offset = input["offset"].as_u64().unwrap_or(0) as usize;
                    // A result reader is a bounded paging API, not an escape hatch
                    // that can put the complete archived payload back into one
                    // model turn.  Large callers can page explicitly.
                    let limit = (input["limit"].as_u64().unwrap_or(default_page_size as u64)
                        as usize)
                        .min(max_page_size);

                    let ctx_guard = contexts.read();
                    let ctx = ctx_guard.get(&tool_name_owned).ok_or_else(|| {
                        format!("Micro-tool context not found: {}", tool_name_owned)
                    })?;

                    let data_guard = data.read();
                    let stored_data = data_guard
                        .get(&ctx.storage_key)
                        .ok_or_else(|| format!("Micro-tool data not found: {}", ctx.storage_key))?;

                    if tool_name_owned.starts_with("read_full_result_") {
                        if let Some(content) = stored_data.get("content").and_then(|v| v.as_str()) {
                            let lines: Vec<&str> = content.lines().collect();
                            let selected: Vec<String> = lines
                                .iter()
                                .skip(offset)
                                .take(limit)
                                .map(|l| l.to_string())
                                .collect();
                            return Ok(json!({
                                "content": selected.join("\n"),
                                "total_lines": lines.len(),
                                "offset": offset,
                                "returned": selected.len(),
                                "call_id": ctx.call_id,
                            }));
                        }
                    } else if tool_name_owned.starts_with("query_") {
                        if let Some(content) = stored_data.get("content").and_then(|v| v.as_str()) {
                            let query_type = input["entity_type"].as_str().unwrap_or("");
                            let keyword = input["keyword"].as_str().unwrap_or("");

                            let mut results = Vec::new();
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content) {
                                if let Some(arr) = parsed.as_array() {
                                    for item in arr.iter().skip(offset).take(limit) {
                                        let type_match = query_type.is_empty()
                                            || item
                                                .get("type")
                                                .and_then(|v| v.as_str())
                                                .map(|t| t.contains(query_type))
                                                .unwrap_or(false);
                                        let keyword_match = keyword.is_empty()
                                            || item
                                                .to_string()
                                                .to_lowercase()
                                                .contains(&keyword.to_lowercase());
                                        if type_match && keyword_match {
                                            results.push(item.clone());
                                        }
                                    }
                                }
                            }
                            return Ok(json!({
                                "results": results,
                                "count": results.len(),
                                "call_id": ctx.call_id,
                            }));
                        }
                    } else if tool_name_owned.starts_with("get_entity_details_") {
                        let entity_id = input["entity_id"].as_str().unwrap_or("");
                        if let Some(content) = stored_data.get("content").and_then(|v| v.as_str()) {
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content) {
                                if let Some(arr) = parsed.as_array() {
                                    for item in arr {
                                        if item.get("id").and_then(|v| v.as_str())
                                            == Some(entity_id)
                                        {
                                            return Ok(json!({
                                                "entity": item,
                                                "call_id": ctx.call_id,
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                        return Ok(json!({
                            "error": "Entity not found",
                            "entity_id": entity_id,
                            "call_id": ctx.call_id,
                        }));
                    }

                    Ok(json!({
                        "data": stored_data,
                        "call_id": ctx.call_id,
                    }))
                })
            }),
            &[],
        );
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

    /// Return only dynamic tools generated for one concrete tool call. This
    /// lets AgentRunner keep micro-tool visibility scoped to a BizAgent
    /// execution without deleting globally archived handlers that another
    /// concurrently running agent may still need.
    pub fn get_micro_tool_names_for_call(&self, call_id: &str) -> Vec<String> {
        self.micro_tool_contexts
            .read()
            .iter()
            .filter(|(_, context)| context.call_id == call_id)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Reconstruct one active session micro-tool schema even when its catalog
    /// description was evicted by the process-wide prompt-size cap.  Handler
    /// and context lifetime are deliberately longer than catalog visibility;
    /// AgentRunner still decides which owning BizAgent session may advertise
    /// this schema.
    pub fn micro_tool_definition(&self, name: &str) -> Option<Value> {
        let context = self.micro_tool_contexts.read().get(name)?.clone();
        self.try_get_handler(name)?;
        let description = if name.starts_with("read_full_result_") {
            format!(
                "Read a bounded page of archived tool result {}",
                context.call_id
            )
        } else if name.starts_with("query_") {
            format!("Query archived result entities for {}", context.call_id)
        } else {
            format!("Read archived result data for {}", context.call_id)
        };
        Some(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "offset": {"type": "integer", "description": "Starting offset"},
                        "limit": {"type": "integer", "description": "Maximum results to return"}
                    }
                }
            }
        }))
    }

    pub async fn execute(&self, name: &str, input: Value) -> Result<Value, String> {
        let input_str = input.to_string();

        if let Some(ref policy) = self.permission_policy {
            match policy.authorize(name, &input_str, None) {
                PermissionOutcome::Deny { reason } => {
                    return Ok(json!({"error": format!("Permission denied: {}", reason)}));
                }
                PermissionOutcome::Allow => {}
            }
        }

        if let Some(ref runner) = self.hook_runner {
            let hook_result = runner.run_pre_tool_use(name, &input_str);
            if hook_result.is_denied() {
                return Ok(
                    json!({"error": format!("Pre-tool hook denied: {}", hook_result.messages().join("; "))}),
                );
            }
            if hook_result.is_failed() {
                return Ok(
                    json!({"error": format!("Pre-tool hook failed: {}", hook_result.messages().join("; "))}),
                );
            }
            if hook_result.is_cancelled() {
                return Ok(json!({"error": "Pre-tool hook was cancelled"}));
            }
        }

        // Tool discovery must query the live executor catalog, including MCP
        // and application tools registered after built-ins. The historical
        // built-in handler contains only a five-item static fallback and made
        // on-demand tool groups impossible to activate in practice.
        if name == "tool_search" {
            let query = input
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let max_results = input
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
        let result = handler(input).await;

        // Post-tool-use hook
        if let Some(ref runner) = self.hook_runner {
            match &result {
                Ok(output) => {
                    let output_str = output.to_string();
                    let post_result =
                        runner.run_post_tool_use(name, &input_str, &output_str, false);
                    if post_result.is_denied() {
                        return Ok(
                            json!({"error": format!("Post-tool hook denied: {}", post_result.messages().join("; ")), "original_output": output}),
                        );
                    }
                }
                Err(e) => {
                    let _ = runner.run_post_tool_use_failure(name, &input_str, e);
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
        if let Some(ref allowed) = allowed_tools {
            if !Self::explicit_allowlist_permits(name, allowed) {
                return Ok(json!({"error": format!("Tool not allowed: {}", name), "tool": name}));
            }
        }
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
                MICRO_TOOL_PREFIXES
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
                    .then(|| "iri://skills/file_read".to_string())
            });
            let Some(skill_iri) = skill_iri else {
                return Ok(
                    json!({"error": "Security denied: tool has no registered executable skill", "tool": name}),
                );
            };
            match engine.check_execution(&skill_iri, &context).await {
                Ok(SecurityDecision::Allowed) => {}
                Ok(SecurityDecision::Denied { reasons }) => {
                    return Ok(
                        json!({"error": "Security denied", "tool": name, "skill_iri": skill_iri, "reasons": reasons}),
                    );
                }
                Ok(SecurityDecision::RequiresApproval { approver, reason }) => {
                    return Ok(
                        json!({"error": "Security approval required", "tool": name, "skill_iri": skill_iri, "approver": approver, "reason": reason}),
                    );
                }
                Err(error) => {
                    return Ok(
                        json!({"error": format!("Security denied: {error}"), "tool": name, "skill_iri": skill_iri}),
                    )
                }
            }
        }
        let mut input = input;
        if name == "file_read" {
            // Internal-only context tag. serde ignores this extra field in the
            // built-in FileReadInput; it exists solely to isolate cache
            // visibility between PA/DA/CA LLM conversations.
            let task = context.task_iri.as_deref().unwrap_or("no-task");
            if let Some(object) = input.as_object_mut() {
                object.insert(
                    "__gh_read_session".to_string(),
                    Value::String(format!(
                        "{}|{}|{}",
                        task, context.agent_id, context.agent_role
                    )),
                );
            }
        }
        self.execute(name, input).await
    }

    /// Exact allowlist matching plus the read-only micro-tools generated by
    /// result routing.  A caller that grants `file_read` may consume those
    /// archived read results; no write capability is implied.
    pub fn explicit_allowlist_permits(name: &str, allowed: &[String]) -> bool {
        allowed.iter().any(|tool| tool == name)
            || (allowed.iter().any(|tool| tool == "file_read")
                && (name == "read_agent_output" || name.starts_with("read_full_result_")))
    }

    /// Get tool handler (avoid holding lock across await)
    pub fn get_handler(&self, name: &str) -> Option<ToolFn> {
        self.tools.get(name).cloned()
    }

    /// Get tool handler with micro-tool fallback.
    /// When normal lookup fails, dynamically build a handler from micro-tool data storage,
    /// preventing LLM from exhausting turns due to registry/handler inconsistency.
    pub fn try_get_handler(&self, name: &str) -> Option<ToolFn> {
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
        let call_id = ctx.call_id.clone();
        drop(ctx_guard);

        let data_guard = self.micro_tool_data.read();
        let stored_data = data_guard.get(&storage_key)?.clone();
        drop(data_guard);
        let default_page_size = self.micro_tool_page_size;
        let max_page_size = self.micro_tool_max_page_size;

        Some(Arc::new(move |input: Value| {
            let _storage_key = storage_key.clone();
            let call_id = call_id.clone();
            let stored_data = stored_data.clone();

            Box::pin(async move {
                let offset = input["offset"].as_u64().unwrap_or(0) as usize;
                let limit = (input["limit"].as_u64().unwrap_or(default_page_size as u64) as usize)
                    .min(max_page_size);

                if let Some(content) = stored_data.get("content").and_then(|v| v.as_str()) {
                    let lines: Vec<&str> = content.lines().collect();
                    let selected: Vec<String> = lines
                        .iter()
                        .skip(offset)
                        .take(limit)
                        .map(|l| l.to_string())
                        .collect();
                    return Ok(serde_json::json!({
                        "content": selected.join("
                    "),
                        "total_lines": lines.len(),
                        "offset": offset,
                        "returned": selected.len(),
                        "call_id": call_id,
                    }));
                }

                Ok(serde_json::json!({
                    "data": stored_data,
                    "call_id": call_id,
                }))
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

        let result: Vec<Value> = self
            .tool_descriptions
            .iter()
            .filter(|td| {
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
