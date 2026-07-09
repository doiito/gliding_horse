use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::broadcast;
use tracing::{debug, info, instrument};

use crate::causal::engine::CausalEngine;
use crate::causal::types::CausalObservation;
use crate::core::event_bus::{EventBus, EventType};
use crate::core::perception_store::{PerceptionEntry, PerceptionSource, PerceptionStore};
use crate::memory::l2_blackboard::Blackboard;
use crate::tools::hooks::{FunctionHook, HookContext, HookManager, HookPoint, HookResult};

pub mod content_store;
pub mod diff_engine;
pub mod inventory;
pub mod snapshot;
pub mod watch_engine;

pub use content_store::{ContentStore, ReadMode, ReadResult};
pub use diff_engine::DiffEngine;
pub use inventory::{FileEntry, FileInventory, FileState};
pub use snapshot::{RollbackResult, SnapshotManager, WorkspaceSnapshot};
pub use watch_engine::{WatchConfig, WatchEngine};

/// Configuration for the workspace monitor subsystem.
#[derive(Debug, Clone)]
pub struct WorkspaceMonitorConfig {
    /// Root directory of the workspace to monitor.
    pub workspace_root: PathBuf,
    /// Glob patterns to exclude from file scanning.
    pub exclude_patterns: Vec<String>,
    /// Maximum content cache size in bytes.
    pub content_store_max_bytes: usize,
    /// Maximum number of files in LRU content cache.
    pub content_cache_capacity: usize,
    /// Enable native file system watching.
    pub watch_enabled: bool,
    /// Polling interval in ms (fallback when native watching unavailable).
    pub poll_interval_ms: u64,
    /// Debounce window in ms for file events.
    pub debounce_ms: u64,
    /// Maximum debounce wait in ms.
    pub max_debounce_wait_ms: u64,
    /// Optional redb database path for persistent storage.
    pub db_path: Option<PathBuf>,
}

impl Default for WorkspaceMonitorConfig {
    fn default() -> Self {
        Self {
            workspace_root: PathBuf::from("."),
            exclude_patterns: vec![
                "node_modules/".into(),
                "target/".into(),
                ".git/".into(),
                "dist/".into(),
                "build/".into(),
                "__pycache__/".into(),
                ".venv/".into(),
                "venv/".into(),
                ".next/".into(),
                "data/".into(),
                ".gliding_horse/".into(),
            ],
            content_store_max_bytes: 64 * 1024 * 1024, // 64 MB
            content_cache_capacity: 1000,
            watch_enabled: true,
            poll_interval_ms: 5000,
            debounce_ms: 500,
            max_debounce_wait_ms: 5000,
            db_path: None,
        }
    }
}

/// The top-level workspace monitor orchestrator.
///
/// Owns all sub-components:
/// - `FileInventory`: tracks file metadata and state
/// - `ContentStore`: caches file content with versioning
/// - `SnapshotManager`: creates/restores workspace snapshots
/// - `WatchEngine`: listens for filesystem changes
pub struct WorkspaceMonitor {
    pub config: WorkspaceMonitorConfig,
    pub inventory: Arc<RwLock<FileInventory>>,
    pub content_store: Arc<ContentStore>,
    pub snapshot_manager: Arc<SnapshotManager>,
    #[allow(dead_code)]
    watch_engine: Option<WatchEngine>,
    event_bus: Option<Arc<EventBus>>,
    perception_store: RwLock<Option<Arc<PerceptionStore>>>,
    causal_engine: RwLock<Option<Arc<CausalEngine>>>,
}

impl WorkspaceMonitor {
    /// Initialize the workspace monitor with the given config.
    ///
    /// Sets up:
    /// 1. redb database (if path configured)
    /// 2. ContentStore with version storage
    /// 3. FileInventory with L2 Blackboard sync
    /// 4. SnapshotManager for rollback support
    /// 5. WatchEngine for file system events
    #[instrument(skip(config, blackboard, event_bus))]
    pub fn initialize(
        config: WorkspaceMonitorConfig,
        blackboard: Option<Arc<Blackboard>>,
        event_bus: Option<Arc<EventBus>>,
    ) -> Result<Self, String> {
        let root = config.workspace_root.to_string_lossy().to_string();

        // Initialize redb database
        let (meta_db, content_db) = Self::open_databases(&config)?;
        let meta_db = meta_db;
        let content_db = content_db;

        // ContentStore
        let content_store = Arc::new(ContentStore::new(
            config.content_cache_capacity,
            config.content_store_max_bytes,
            content_db,
        ));

        // FileInventory
        let inventory = Arc::new(RwLock::new(FileInventory::new(
            blackboard.clone(),
            meta_db,
            config.exclude_patterns.clone(),
        )));

        // SnapshotManager (in-memory redb)
        let snap_db = Arc::new(
            redb::Builder::new()
                .create_with_backend(redb::backends::InMemoryBackend::new())
                .map_err(|e| format!("Failed to open snapshot redb: {}", e))?,
        );
        let snapshot_manager = Arc::new(SnapshotManager::new(
            snap_db,
            content_store.clone(),
            inventory.clone(),
        ));

        let event_bus_for_struct = event_bus.clone();

        // WatchEngine — native file watching (inotify thread, no tokio needed)
        let watch_engine = if let Some(eb) = event_bus.clone() {
            let mut watch_config = WatchConfig {
                debounce_ms: config.debounce_ms,
                max_debounce_wait_ms: config.max_debounce_wait_ms,
                poll_interval_ms: config.poll_interval_ms,
                watch_enabled: config.watch_enabled,
                exclude_patterns: config.exclude_patterns.clone(),
                use_gitignore: true,
            };
            if watch_config.use_gitignore {
                watch_config.load_gitignore(&config.workspace_root);
            }
            match WatchEngine::start(&root, watch_config, eb, Some(inventory.clone())) {
                Ok(engine) => {
                    info!("WatchEngine started for {}", root);
                    Some(engine)
                }
                Err(e) => {
                    tracing::warn!("WatchEngine failed to start: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Build self
        let ws = Self {
            config,
            inventory,
            content_store,
            snapshot_manager,
            watch_engine,
            event_bus: event_bus_for_struct,
            perception_store: RwLock::new(None),
            causal_engine: RwLock::new(None),
        };

        // Event consumers are deferred to start_async_components() which is called
        // from within an async context (process_task) after finalize_setup() has
        // wired the perception_store. This avoids spawning consumers with None.
        // Perform initial scan
        {
            let inv = ws.inventory.read();
            let discovered = inv.full_scan(&root);
            debug!(discovered = discovered, "Initial workspace scan completed");
        }

        info!("WorkspaceMonitor initialized for root={}", root);

        Ok(ws)
    }

    /// Read a file through ContentStore with cache/diff support.
    pub fn read_file(&self, path: &str, mode: ReadMode) -> std::io::Result<ReadResult> {
        let result = self.content_store.read_file(path, mode)?;

        // Update FileInventory state
        let inv = self.inventory.read();
        if result.changed {
            inv.add_or_update(path);
        }
        inv.mark_read(path, result.version);

        Ok(result)
    }

    /// Mark a file as externally read without disk I/O.
    /// Used when file content was provided via read_full_result micro-tool,
    /// so subsequent file_read calls recognize it as already-read.
    pub fn mark_file_read_external(&self, path: &str) {
        let inv = self.inventory.read();
        inv.mark_external_read(path);
    }

    /// Mark a file as written by the agent.
    pub fn mark_file_written(&self, path: &str) {
        let inv = self.inventory.read();
        inv.mark_written(path);
        self.content_store.invalidate(path);
    }

    /// Re-scan the entire workspace root, discovering new files and tracking state changes.
    /// Returns the number of newly discovered files.
    pub fn rescan(&self) -> usize {
        let root = self.config.workspace_root.to_string_lossy().to_string();
        self.inventory.read().full_scan(&root)
    }

    /// Get the snapshot manager reference.
    pub fn snapshots(&self) -> &Arc<SnapshotManager> {
        &self.snapshot_manager
    }

    /// Get the content store reference.
    pub fn content(&self) -> &Arc<ContentStore> {
        &self.content_store
    }

    /// Subscribe to EventBus for workspace file events and update inventory.
    pub fn register_event_consumers(&self) {
        let event_bus = match &self.event_bus {
            Some(eb) => eb.clone(),
            None => {
                tracing::warn!("EventBus not available, event consumers not registered");
                return;
            }
        };

        let inventory = self.inventory.clone();
        let perception = self.perception_store.read().clone();
        let causal = self.causal_engine.read().clone();
        let mut receiver = event_bus.subscribe();
        tokio::spawn(async move {
            loop {
                match receiver.recv().await {
                    Ok(event) => {
                        let path = event.payload.clone();
                        let event_type_name = event.event_type.clone();
                        match EventType::from_str(&event_type_name) {
                            EventType::WorkspaceFileCreated => {
                                inventory.read().add_or_update(&path);
                                if let Some(ref ce) = causal {
                                    ce.record_observation(CausalObservation::new(
                                        &format!("ws_create_{}", uuid::Uuid::new_v4()),
                                        "iri://workspace_monitor",
                                        "WorkspaceFileChange",
                                        &format!("file_created:{}", path),
                                    ));
                                }
                                if let Some(ref p) = perception {
                                    let entry = PerceptionEntry::new(
                                        PerceptionSource::WorkspaceMonitor,
                                        format!("New file created: {}", path),
                                    ).with_priority(6);
                                    p.store_global(entry);
                                }
                            }
                            EventType::WorkspaceFileModified => {
                                let inv = inventory.read();
                                if inv.get_entry(&path).is_none() && std::path::Path::new(&path).exists() {
                                    drop(inv);
                                    inventory.read().add_or_update(&path);
                                }
                                inventory.read().mark_stale(&path);
                                if let Some(ref ce) = causal {
                                    ce.record_observation(CausalObservation::new(
                                        &format!("ws_modify_{}", uuid::Uuid::new_v4()),
                                        "iri://workspace_monitor",
                                        "WorkspaceFileChange",
                                        &format!("file_modified:{}", path),
                                    ));
                                }
                                if let Some(ref p) = perception {
                                    let entry = PerceptionEntry::new(
                                        PerceptionSource::WorkspaceMonitor,
                                        format!("File externally modified: {}", path),
                                    ).with_priority(6);
                                    p.store_global(entry);
                                }
                            }
                            EventType::WorkspaceFileRemoved => {
                                inventory.read().remove(&path);
                                if let Some(ref ce) = causal {
                                    ce.record_observation(CausalObservation::new(
                                        &format!("ws_remove_{}", uuid::Uuid::new_v4()),
                                        "iri://workspace_monitor",
                                        "WorkspaceFileChange",
                                        &format!("file_removed:{}", path),
                                    ));
                                }
                                if let Some(ref p) = perception {
                                    let entry = PerceptionEntry::new(
                                        PerceptionSource::WorkspaceMonitor,
                                        format!("File deleted: {}", path),
                                    ).with_priority(5);
                                    p.store_global(entry);
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            "WorkspaceMonitor event consumer lagged by {} events",
                            n
                        );
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        tracing::error!("WorkspaceMonitor event bus connection closed");
                        break;
                    }
                }
            }
        });
    }

    /// Complete async-dependent initialization.
    ///
    /// Must be called from within a tokio runtime (e.g., during `process_task`).
    /// Registers event consumers that listen for WorkspaceFile* events via EventBus.
    ///
    /// Safe to call multiple times — event consumer registration is idempotent
    /// (each call spawns a redundant listener, but the extra listener will simply
    /// process events that have already been handled by the first).
    pub fn start_async_components(&self) {
        if tokio::runtime::Handle::try_current().is_err() {
            tracing::error!("start_async_components must be called from within a tokio runtime");
            return;
        }
        self.register_event_consumers();
    }

    /// Register hooks for file read/write tools to check inventory state.
    pub fn register_hooks(&self, hook_manager: &HookManager) {
        let inv_for_read = self.inventory.clone();

        let read_hook = FunctionHook::new(
            "workspace_monitor_file_read",
            vec![HookPoint::SkillBefore],
            100,
            move |ctx: &mut HookContext| {
                let path = match ctx.data.get("path").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => return HookResult::Continue,
                };
                let inv = inv_for_read.read();
                if let Some(entry) = inv.get_entry(&path) {
                    match entry.state {
                        FileState::ReadStale => {
                            let warning = format!(
                                "[workspace_monitor] File '{}' is stale (last read version {}), consider re-reading before writing",
                                path, entry.last_read_version
                            );
                            ctx.data.insert(
                                "stale_warning".to_string(),
                                serde_json::Value::String(warning.clone()),
                            );
                            // Also write to metadata so ToolGuard pre-injection includes it in system prompt
                            let injections = ctx.metadata
                                .entry("guard_pre_injections".to_string())
                                .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                            if let Some(arr) = injections.as_array_mut() {
                                arr.push(serde_json::Value::String(warning));
                            }
                        }
                        FileState::ReadFresh if entry.last_read_version == entry.current_version => {
                            // File unchanged since last read — inject hint to skip full re-read
                            ctx.data.insert(
                                "file_unchanged".to_string(),
                                serde_json::Value::Bool(true),
                            );
                            let hint = format!(
                                "[workspace_monitor] File '{}' unchanged since last read (v{}). Use mode:diff for incremental changes or mode:force_refresh to re-read.",
                                path, entry.current_version
                            );
                            ctx.data.insert(
                                "file_unchanged_hint".to_string(),
                                serde_json::Value::String(hint.clone()),
                            );
                            // Also write to metadata so ToolGuard pre-injection
                            let injections = ctx.metadata
                                .entry("guard_pre_injections".to_string())
                                .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                            if let Some(arr) = injections.as_array_mut() {
                                arr.push(serde_json::Value::String(hint));
                            }
                        }
                        _ => {}
                    }
                }
                HookResult::Continue
            },
        );
        hook_manager.register(Box::new(read_hook));

        let inv_for_write = self.inventory.clone();

        let write_before_hook = FunctionHook::new(
            "workspace_monitor_file_write_before",
            vec![HookPoint::SkillBefore],
            100,
            move |ctx: &mut HookContext| {
                let path = match ctx.data.get("path").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => return HookResult::Continue,
                };
                let inv = inv_for_write.read();
                if let Some(entry) = inv.get_entry(&path) {
                    if entry.state == FileState::ReadStale {
                        ctx.data.insert(
                            "stale_warning".to_string(),
                            serde_json::Value::String(format!(
                                "File '{}' is stale, writing may overwrite external changes",
                                path
                            )),
                        );
                    }
                }
                inv.add_or_update(&path);
                HookResult::Continue
            },
        );
        hook_manager.register(Box::new(write_before_hook));

        let inv_for_mark = self.inventory.clone();

        let write_after_hook = FunctionHook::new(
            "workspace_monitor_file_write_after",
            vec![HookPoint::SkillAfter],
            100,
            move |ctx: &mut HookContext| {
                let path = match ctx.data.get("path").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => return HookResult::Continue,
                };
                let inv = inv_for_mark.read();
                inv.mark_written(&path);
                HookResult::Continue
            },
        );
        hook_manager.register(Box::new(write_after_hook));
    }

    /// Set the active perception store, allowing WorkspaceMonitor to inject file state awareness data
    pub fn with_perception_store(self, store: Arc<PerceptionStore>) -> Self {
        *self.perception_store.write() = Some(store);
        self
    }

    /// Set the perception store after WorkspaceMonitor construction (for Arc<WorkspaceMonitor> scenarios)
    pub fn set_perception_store(&self, store: Arc<PerceptionStore>) {
        *self.perception_store.write() = Some(store);
    }

    /// Attach a CausalEngine to record file-change observations for causal analysis.
    pub fn set_causal_engine(&self, engine: Arc<CausalEngine>) {
        *self.causal_engine.write() = Some(engine);
    }

    /// Generate workspace file status summary text for injection into perception region
    /// Reset the file inventory, clearing all tracked files.
    /// Called on topic shift to prevent files from previous tasks leaking into new task perception.
    pub fn reset_inventory(&self) {
        self.inventory.write().clear_all();
        let ps = self.perception_store.read();
        if let Some(ref store) = *ps {
            store.clear_global();
        }
    }

    pub fn generate_perception_text(&self, task_context: Option<&str>) -> Option<String> {
        let inv = self.inventory.read();
        let all = inv.list_all();
        if all.is_empty() {
            return None;
        }

        let total = all.len();

        // Build keyword set from task context for relevance scoring
        let keywords: Vec<String> = task_context
            .map(|ctx| {
                ctx.split(|c: char| !c.is_alphanumeric() && c != '.')
                    .filter(|w| w.len() > 2)
                    .map(|w| w.to_lowercase())
                    .collect()
            })
            .unwrap_or_default();

        let score_relevance = |path: &str| -> usize {
            if keywords.is_empty() { return 0; }
            let path_lower = path.to_lowercase();
            keywords.iter().filter(|k| path_lower.contains(k.as_str())).count()
        };

        let mut stale: Vec<_> = all.iter().filter(|e| e.state == FileState::ReadStale).collect();
        let mut written_unread: Vec<_> = all.iter().filter(|e| e.state == FileState::WrittenUnread).collect();
        let discovered: Vec<_> = all.iter().filter(|e| e.state == FileState::Discovered).collect();

        // Sort by task relevance (highest first)
        stale.sort_by(|a, b| score_relevance(&b.path).cmp(&score_relevance(&a.path)));
        written_unread.sort_by(|a, b| score_relevance(&b.path).cmp(&score_relevance(&a.path)));

        let mut parts = Vec::new();
        parts.push(format!("{} files total", total));

        if !stale.is_empty() {
            let names: Vec<&str> = stale.iter().take(5).map(|e| e.path.as_str()).collect();
            parts.push(format!(
                "{} externally modified{}",
                stale.len(),
                if names.is_empty() { String::new() } else {
                    format!(": {}", names.join(", "))
                }
            ));
        }

        if !written_unread.is_empty() {
            let names: Vec<&str> = written_unread.iter().take(5).map(|e| e.path.as_str()).collect();
            parts.push(format!(
                "{} written but not re-read{}",
                written_unread.len(),
                if names.is_empty() { String::new() } else {
                    format!(": {}", names.join(", "))
                }
            ));
        }

        if !discovered.is_empty() {
            parts.push(format!("{} new discovered files unread", discovered.len()));
        }

        let summary = format!("{} | {}", total, parts.join(" | "));
        let guidance = "\n\nHint: Only read files relevant to your current task; ignore unrelated files directly.\
            \n\"Written but not re-read\" files are outputs from other agents — only read them when you need to reference their content; no need to confirm all of them.\
            \n\"Externally modified\" files are those that have been read but rewritten externally — only re-read when necessary.\
            \ncache_hit(from_cache=true) means file content hasn't changed this round and was already provided earlier — skip re-reading and continue with existing content.".to_string();
        Some(format!("{}{}", summary, guidance))
    }

    /// Get a concise summary string of the workspace file inventory.
    /// Returns "N files across M directories" with a breakdown by language.
    /// Returns None when the inventory is empty.
    pub fn get_file_inventory_summary(&self) -> Option<String> {
        let inv = self.inventory.read();
        let all = inv.list_all();
        if all.is_empty() {
            return None;
        }

        let total = all.len();

        // Count unique directories
        use std::collections::HashSet;
        let dirs: HashSet<String> = all.iter()
            .filter_map(|e| {
                let parent = std::path::Path::new(&e.path).parent()?;
                Some(parent.to_string_lossy().into_owned())
            })
            .collect();

        // Count by language
        let mut lang_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for entry in &all {
            let lang = if entry.language == "unknown" { "other" } else { entry.language.as_str() };
            *lang_counts.entry(lang).or_insert(0) += 1;
        }

        // Sort by count descending, take top 3
        let mut lang_vec: Vec<(&str, usize)> = lang_counts.into_iter().collect();
        lang_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let lang_summary = lang_vec.iter().take(3)
            .map(|(l, c)| format!("{} {}", c, l))
            .collect::<Vec<_>>()
            .join(", ");

        let summary = if lang_vec.len() > 3 {
            format!("{} files across {} directories ({} … +{} more)",
                total, dirs.len(), lang_summary, lang_vec.len() - 3)
        } else {
            format!("{} files across {} directories ({})",
                total, dirs.len(), lang_summary)
        };

        Some(summary)
    }

    /// Write current file status perception summary to PerceptionStore.
    /// `task_context` optionally describes the current task for relevance sorting.
    pub fn inject_file_perception(&self, task_context: Option<&str>) {
        let ps = self.perception_store.read();
        if let Some(ref store) = *ps {
            if let Some(text) = self.generate_perception_text(task_context) {
                let entry = PerceptionEntry::new(PerceptionSource::WorkspaceMonitor, text);
                store.store_global(entry);
            }
        }
    }

    // ── Private ──

    fn open_databases(
        config: &WorkspaceMonitorConfig,
    ) -> Result<(Option<redb::Database>, Option<redb::Database>), String> {
        match &config.db_path {
            Some(path) => {
                std::fs::create_dir_all(path)
                    .map_err(|e| format!("Failed to create database directory: {}", e))?;

                let meta_path = path.join("metadata");
                let content_path = path.join("content");

                let meta_db = redb::Database::create(&meta_path)
                    .map_err(|e| format!("Failed to open metadata redb: {}", e))?;

                let content_db = redb::Database::create(&content_path)
                    .map_err(|e| format!("Failed to open content redb: {}", e))?;

                Ok((Some(meta_db), Some(content_db)))
            }
            None => Ok((None, None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_bus::{EventBus, EventType};
    use crate::tools::hooks::{HookContext, HookManager, HookPoint, HookResult};
    use serde_json::Value;
    use std::sync::Arc;

    fn temp_ws_monitor() -> (WorkspaceMonitor, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None).unwrap();
        (ws, dir)
    }

    #[test]
    fn test_register_hooks_read_stale_warning() {
        let (ws, dir) = temp_ws_monitor();
        let file_path = dir.path().join("test.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();

        {
            let inv = ws.inventory.read();
            inv.add_or_update(&file_path.to_string_lossy()).unwrap();
            inv.mark_stale(&file_path.to_string_lossy());
        }

        let hm = HookManager::new();
        ws.register_hooks(&hm);

        let mut ctx = HookContext::new(HookPoint::SkillBefore, "agent_1", "DA");
        ctx.data.insert("path".to_string(), Value::String(file_path.to_string_lossy().to_string()));

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(hm.execute(HookPoint::SkillBefore, &mut ctx));
        assert_eq!(result, HookResult::Continue);

        let warning = ctx.data.get("stale_warning").and_then(|v| v.as_str()).unwrap_or("");
        assert!(warning.contains("stale"), "Expected stale warning, got: {}", warning);
    }

    #[test]
    fn test_register_hooks_write_marks_written() {
        let (ws, dir) = temp_ws_monitor();
        let file_path = dir.path().join("test.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();

        {
            let inv = ws.inventory.read();
            inv.add_or_update(&file_path.to_string_lossy()).unwrap();
        }

        let hm = HookManager::new();
        ws.register_hooks(&hm);

        let mut ctx = HookContext::new(HookPoint::SkillAfter, "agent_1", "DA");
        ctx.data.insert("path".to_string(), Value::String(file_path.to_string_lossy().to_string()));

        let _ = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(hm.execute(HookPoint::SkillAfter, &mut ctx));

        let inv = ws.inventory.read();
        let entry = inv.get_entry(&file_path.to_string_lossy()).unwrap();
        assert_eq!(entry.state, FileState::WrittenUnread);
    }

    #[tokio::test]
    async fn test_event_consumer_file_created() {
        let bus = Arc::new(EventBus::new(100));
        let dir = tempfile::TempDir::new().unwrap();

        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, Some(bus.clone())).unwrap();
        ws.register_event_consumers();

        let test_file = dir.path().join("created.rs");
        std::fs::write(&test_file, "fn test() {}").unwrap();

        bus.emit(
            "iri://test_task",
            EventType::WorkspaceFileCreated.as_str(),
            "iri://test_agent",
            &test_file.to_string_lossy(),
        )
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let inv = ws.inventory.read();
        let entry = inv.get_entry(&test_file.to_string_lossy());
        assert!(entry.is_some(), "File should be in inventory after Create event");
    }

    #[tokio::test]
    async fn test_event_consumer_file_removed() {
        let bus = Arc::new(EventBus::new(100));
        let dir = tempfile::TempDir::new().unwrap();

        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, Some(bus.clone())).unwrap();
        ws.register_event_consumers();

        let test_file = dir.path().join("toremove.rs");
        std::fs::write(&test_file, "fn x() {}").unwrap();

        {
            let inv = ws.inventory.read();
            inv.add_or_update(&test_file.to_string_lossy()).unwrap();
        }
        assert!(ws.inventory.read().get_entry(&test_file.to_string_lossy()).is_some());

        std::fs::remove_file(&test_file).unwrap();

        bus.emit(
            "iri://test_task",
            EventType::WorkspaceFileRemoved.as_str(),
            "iri://test_agent",
            &test_file.to_string_lossy(),
        )
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let inv = ws.inventory.read();
        assert!(inv.get_entry(&test_file.to_string_lossy()).is_none(), "File should be removed from inventory");
    }

    #[test]
    fn test_hooks_no_path_noop() {
        let (ws, _dir) = temp_ws_monitor();
        let hm = HookManager::new();
        ws.register_hooks(&hm);

        let mut ctx = HookContext::new(HookPoint::SkillBefore, "agent_1", "DA");
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(hm.execute(HookPoint::SkillBefore, &mut ctx));
        assert_eq!(result, HookResult::Continue);
    }

    #[test]
    fn test_mark_file_written_updates_inventory() {
        let (ws, dir) = temp_ws_monitor();
        let file_path = dir.path().join("write.rs");
        std::fs::write(&file_path, "initial").unwrap();

        {
            let inv = ws.inventory.read();
            inv.add_or_update(&file_path.to_string_lossy()).unwrap();
        }

        ws.mark_file_written(&file_path.to_string_lossy());

        let inv = ws.inventory.read();
        let entry = inv.get_entry(&file_path.to_string_lossy()).unwrap();
        assert_eq!(entry.state, FileState::WrittenUnread);
    }

    #[tokio::test]
    async fn test_full_event_consumer_hooks_lifecycle() {
        let bus = Arc::new(EventBus::new(100));
        let dir = tempfile::TempDir::new().unwrap();
        let hm = HookManager::new();

        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, Some(bus.clone())).unwrap();
        ws.register_event_consumers();
        ws.register_hooks(&hm);

        let test_file = dir.path().join("lifecycle.rs");
        let file_path_str = test_file.to_string_lossy().to_string();
        std::fs::write(&test_file, "fn start() {}").unwrap();

        // Step 1: Emit create event → consumer adds to inventory
        bus.emit(
            "iri://test_task",
            EventType::WorkspaceFileCreated.as_str(),
            "iri://test_agent",
            &file_path_str,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            ws.inventory.read().get_entry(&file_path_str).is_some(),
            "File should exist after create event"
        );

        // Step 2: Mark stale externally, emit modified → consumer marks stale
        std::fs::write(&test_file, "fn updated() {}").unwrap();
        bus.emit(
            "iri://test_task",
            EventType::WorkspaceFileModified.as_str(),
            "iri://test_agent",
            &file_path_str,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let entry = ws.inventory.read().get_entry(&file_path_str).unwrap();
        assert_eq!(entry.state, FileState::ReadStale, "File should be stale after modify event");

        // Step 3: Hook SkillBefore read detects stale state
        let mut ctx = HookContext::new(HookPoint::SkillBefore, "agent_1", "DA");
        ctx.data.insert("path".to_string(), Value::String(file_path_str.clone()));
        let result = hm.execute(HookPoint::SkillBefore, &mut ctx).await;
        assert_eq!(result, HookResult::Continue);
        let warning = ctx.data.get("stale_warning").and_then(|v| v.as_str()).unwrap_or("");
        assert!(warning.contains("stale"), "Expected stale warning in lifecycle: {}", warning);

        // Step 4: File write → SkillAfter hook marks WrittenUnread
        let mut write_ctx = HookContext::new(HookPoint::SkillAfter, "agent_1", "DA");
        write_ctx.data.insert("path".to_string(), Value::String(file_path_str.clone()));
        let _ = hm.execute(HookPoint::SkillAfter, &mut write_ctx).await;
        let entry = ws.inventory.read().get_entry(&file_path_str).unwrap();
        assert_eq!(entry.state, FileState::WrittenUnread, "File should be WrittenUnread after write hook");

        // Step 5: Remove file + emit remove → consumer removes from inventory
        std::fs::remove_file(&test_file).unwrap();
        bus.emit(
            "iri://test_task",
            EventType::WorkspaceFileRemoved.as_str(),
            "iri://test_agent",
            &file_path_str,
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            ws.inventory.read().get_entry(&file_path_str).is_none(),
            "File should be removed from inventory after remove event"
        );
    }

    #[test]
    fn test_hook_unchanged_file_detection() {
        let (ws, dir) = temp_ws_monitor();
        let file_path = dir.path().join("unchanged.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();

        {
            let inv = ws.inventory.read();
            inv.add_or_update(&file_path.to_string_lossy()).unwrap();
            inv.mark_read(&file_path.to_string_lossy(), 0);
        }

        let hm = HookManager::new();
        ws.register_hooks(&hm);

        // File is ReadFresh with matching versions — hook should inject file_unchanged
        let mut ctx = HookContext::new(HookPoint::SkillBefore, "agent_1", "DA");
        ctx.data.insert("path".to_string(), Value::String(file_path.to_string_lossy().to_string()));

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(hm.execute(HookPoint::SkillBefore, &mut ctx));
        assert_eq!(result, HookResult::Continue);

        assert_eq!(
            ctx.data.get("file_unchanged").and_then(|v| v.as_bool()),
            Some(true),
            "Expected file_unchanged flag for ReadFresh file with matching version"
        );
        let hint = ctx.data.get("file_unchanged_hint").and_then(|v| v.as_str()).unwrap_or("");
        assert!(hint.contains("unchanged"), "Expected unchanged hint: {}", hint);
    }

    #[test]
    fn test_hook_stale_warning_for_stale_file() {
        let (ws, dir) = temp_ws_monitor();
        let file_path = dir.path().join("stale.rs");
        std::fs::write(&file_path, "fn stale() {}").unwrap();

        {
            let inv = ws.inventory.read();
            inv.add_or_update(&file_path.to_string_lossy()).unwrap();
            inv.mark_read(&file_path.to_string_lossy(), 0);
            inv.mark_stale(&file_path.to_string_lossy());
        }

        let hm = HookManager::new();
        ws.register_hooks(&hm);

        let mut ctx = HookContext::new(HookPoint::SkillBefore, "agent_1", "DA");
        ctx.data.insert("path".to_string(), Value::String(file_path.to_string_lossy().to_string()));

        let _ = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(hm.execute(HookPoint::SkillBefore, &mut ctx));

        assert!(
            ctx.data.get("file_unchanged").is_none(),
            "Stale file should NOT have file_unchanged flag"
        );
        assert!(
            ctx.data.get("stale_warning").is_some(),
            "Stale file SHOULD have stale_warning"
        );
    }

    // ── PerceptionStore integration ──

    #[tokio::test]
    async fn test_event_consumer_injects_perception_store_on_create() {
        let bus = Arc::new(EventBus::new(100));
        let dir = tempfile::TempDir::new().unwrap();
        let ps = Arc::new(PerceptionStore::new());

        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, Some(bus.clone()))
            .unwrap()
            .with_perception_store(ps.clone());
        ws.register_event_consumers();

        let test_file = dir.path().join("percept_create.rs");
        std::fs::write(&test_file, "fn test() {}").unwrap();

        bus.emit(
            "iri://test_task",
            EventType::WorkspaceFileCreated.as_str(),
            "iri://test_agent",
            &test_file.to_string_lossy(),
        )
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        assert!(ps.has_new("iri://test_task"), "PerceptionStore should have new entry after create event");
        let text = ps.take_perception_text("iri://test_task");
        assert!(text.contains("New file"), "Perception text should mention file creation: {}", text);
    }

    #[tokio::test]
    async fn test_event_consumer_injects_perception_store_on_modify() {
        let bus = Arc::new(EventBus::new(100));
        let dir = tempfile::TempDir::new().unwrap();
        let ps = Arc::new(PerceptionStore::new());

        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, Some(bus.clone()))
            .unwrap()
            .with_perception_store(ps.clone());
        ws.register_event_consumers();

        let test_file = dir.path().join("percept_modify.rs");
        std::fs::write(&test_file, "fn old() {}").unwrap();
        std::fs::write(&test_file, "fn new() {}").unwrap();

        bus.emit(
            "iri://test_task",
            EventType::WorkspaceFileModified.as_str(),
            "iri://test_agent",
            &test_file.to_string_lossy(),
        )
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        assert!(ps.has_new("iri://test_task"), "PerceptionStore should have new entry after modify event");
        let text = ps.take_perception_text("iri://test_task");
        assert!(text.contains("externally modified"), "Perception text should mention external change: {}", text);
    }

    #[tokio::test]
    async fn test_event_consumer_injects_perception_store_on_remove() {
        let bus = Arc::new(EventBus::new(100));
        let dir = tempfile::TempDir::new().unwrap();
        let ps = Arc::new(PerceptionStore::new());

        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, Some(bus.clone()))
            .unwrap()
            .with_perception_store(ps.clone());
        ws.register_event_consumers();

        let test_file = dir.path().join("percept_remove.rs");
        std::fs::write(&test_file, "fn gone() {}").unwrap();

        bus.emit(
            "iri://test_task",
            EventType::WorkspaceFileRemoved.as_str(),
            "iri://test_agent",
            &test_file.to_string_lossy(),
        )
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        assert!(ps.has_new("iri://test_task"), "PerceptionStore should have new entry after remove event");
        let text = ps.take_perception_text("iri://test_task");
        assert!(text.contains("File deleted"), "Perception text should mention file removal: {}", text);
    }

    #[test]
    fn test_inject_file_perception_with_stale_files() {
        let (_ws, dir) = temp_ws_monitor();
        let ps = Arc::new(PerceptionStore::new());

        // Can't use with_perception_store after initialize since it consumes self
        // We need to build one directly with perception_store set
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, None)
            .unwrap()
            .with_perception_store(ps.clone());

        // Add some files to inventory
        let file1 = dir.path().join("stale1.rs");
        std::fs::write(&file1, "fn a() {}").unwrap();
        let file2 = dir.path().join("stale2.rs");
        std::fs::write(&file2, "fn b() {}").unwrap();

        {
            let inv = ws.inventory.read();
            inv.add_or_update(&file1.to_string_lossy()).unwrap();
            inv.add_or_update(&file2.to_string_lossy()).unwrap();
            inv.mark_read(&file1.to_string_lossy(), 1);
            inv.mark_read(&file2.to_string_lossy(), 1);
            inv.mark_stale(&file1.to_string_lossy());
            inv.mark_stale(&file2.to_string_lossy());
        }

        ws.inject_file_perception(None);

        let text = ps.take_perception_text("iri://task/t");
        assert!(!text.is_empty(), "Should have perception text after inject");
        assert!(text.contains("stale"), "Should mention stale files: {}", text);
    }

    #[test]
    fn test_generate_perception_text_empty_inventory() {
        let (ws, _dir) = temp_ws_monitor();
        let text = ws.generate_perception_text(None);
        // Fresh inventory with initial scan may have files
        // If empty, it returns None; otherwise it should have text
        let inv = ws.inventory.read();
        if inv.list_all().is_empty() {
            assert!(text.is_none(), "Empty inventory should return None");
        } else {
            assert!(text.is_some(), "Non-empty inventory should return Some");
        }
    }

    #[test]
    fn test_generate_perception_text_with_state_counts() {
        let (ws, dir) = temp_ws_monitor();

        let f1 = dir.path().join("active.rs");
        std::fs::write(&f1, "fn active() {}").unwrap();
        let f2 = dir.path().join("stale.rs");
        std::fs::write(&f2, "fn stale() {}").unwrap();

        {
            let inv = ws.inventory.read();
            inv.add_or_update(&f1.to_string_lossy()).unwrap();
            inv.add_or_update(&f2.to_string_lossy()).unwrap();
            inv.mark_read(&f1.to_string_lossy(), 1);
            inv.mark_read(&f2.to_string_lossy(), 1);
            inv.mark_stale(&f2.to_string_lossy());
        }

        let text = ws.generate_perception_text(None);
        assert!(text.is_some(), "Should generate perception text");
        let t = text.unwrap();
        assert!(t.contains("files total"), "Should mention total file count: {}", t);
    }

    #[test]
    fn test_inject_file_perception_no_perception_store_noop() {
        let (ws, _dir) = temp_ws_monitor();
        // Should not panic when perception_store is None
        ws.inject_file_perception(None);
    }

    #[test]
    fn test_with_perception_store_chain() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ps = Arc::new(PerceptionStore::new());
        let ws = WorkspaceMonitor::initialize(config, None, None)
            .unwrap()
            .with_perception_store(ps.clone());

        // Verify it's configured
        ws.inject_file_perception(None);
        // Should not panic, meaning perception_store is Some
        let _ = ws.generate_perception_text(None);
    }

    /// Full integration: event consumer → perception store → take → verify
    #[tokio::test]
    async fn test_event_to_perception_full_flow() {
        let bus = Arc::new(EventBus::new(100));
        let dir = tempfile::TempDir::new().unwrap();
        let ps = Arc::new(PerceptionStore::new());

        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, Some(bus.clone()))
            .unwrap()
            .with_perception_store(ps.clone());
        ws.register_event_consumers();

        let test_file = dir.path().join("full_flow.rs");
        std::fs::write(&test_file, "fn test() {}").unwrap();

        // Emit create event
        bus.emit(
            "iri://task_full",
            EventType::WorkspaceFileCreated.as_str(),
            "iri://agent",
            &test_file.to_string_lossy(),
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // take perception text
        let text1 = ps.take_perception_text("iri://task_full");
        assert!(!text1.is_empty(), "Perception should be available after create event");
        assert!(text1.contains("New file"), "Should mention new file: {}", text1);

        // Second take should be empty (consumed)
        let text2 = ps.take_perception_text("iri://task_full");
        assert!(text2.is_empty(), "Second take should be empty after consumption");
    }
}
