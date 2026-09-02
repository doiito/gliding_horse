use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{debug, info, instrument, warn};

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
pub use snapshot::{RollbackPlan, RollbackResult, SnapshotManager, WorkspaceSnapshot};
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
    /// Defer the initial inventory scan until async components start. TUI
    /// applications use this so interface rendering is never blocked by a
    /// large workspace walk; services may retain eager initialization.
    pub defer_initial_scan: bool,
    /// Maximum number of workspace deltas retained for generation queries.
    pub change_history_capacity: usize,
    /// Polling interval in ms (fallback when native watching unavailable).
    pub poll_interval_ms: u64,
    /// Debounce window in ms for file events.
    pub debounce_ms: u64,
    /// Maximum debounce wait in ms.
    pub max_debounce_wait_ms: u64,
    pub initial_scan_wait_ms: u64,
    /// Bounds for semantic before/after snapshots used to confirm shell-like
    /// workspace effects without making full content part of the LLM context.
    pub effect_snapshot_max_files: usize,
    pub effect_snapshot_max_bytes: u64,
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
            defer_initial_scan: false,
            change_history_capacity: 2048,
            poll_interval_ms: 5000,
            debounce_ms: 500,
            max_debounce_wait_ms: 5000,
            initial_scan_wait_ms: 250,
            effect_snapshot_max_files: 10_000,
            effect_snapshot_max_bytes: 64 * 1024 * 1024,
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
    /// Watch installation can require walking every non-excluded directory.
    /// Keep it behind a shared slot so initialization can happen after the TUI
    /// is visible on a blocking worker instead of delaying application startup.
    watch_engine: Arc<RwLock<Option<WatchEngine>>>,
    watch_config: Option<WatchConfig>,
    event_bus: Option<Arc<EventBus>>,
    perception_store: RwLock<Option<Arc<PerceptionStore>>>,
    causal_engine: RwLock<Option<Arc<CausalEngine>>>,
    /// Idempotent guard for start_async_components().
    async_started: AtomicBool,
    /// Monotonic workspace view generation. It changes only when the visible
    /// inventory changes, allowing AgentRunner to inject bounded deltas rather
    /// than repeating a complete manifest every turn.
    generation: Arc<AtomicU64>,
    scan_complete: Arc<AtomicBool>,
    scan_notify: Arc<tokio::sync::Notify>,
    changes: Arc<RwLock<VecDeque<WorkspaceChange>>>,
    recent_agent_writes: Arc<RwLock<std::collections::HashMap<String, std::time::Instant>>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceChangeKind {
    Created,
    Modified,
    Removed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceChangeOrigin {
    AgentTool,
    External,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceChange {
    pub generation: u64,
    pub path: String,
    pub kind: WorkspaceChangeKind,
    pub origin: WorkspaceChangeOrigin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceFileView {
    pub path: String,
    pub file_size: u64,
    pub language: String,
    pub state: String,
    pub version: u64,
    pub content_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceView {
    pub generation: u64,
    pub scan_complete: bool,
    pub total_files: usize,
    pub files: Vec<WorkspaceFileView>,
    pub changes: Vec<WorkspaceChange>,
    pub truncated: bool,
}

fn record_workspace_change(
    generation: &AtomicU64,
    changes: &RwLock<VecDeque<WorkspaceChange>>,
    capacity: usize,
    path: &str,
    kind: WorkspaceChangeKind,
    origin: WorkspaceChangeOrigin,
) {
    let next = generation.fetch_add(1, Ordering::AcqRel).saturating_add(1);
    let mut history = changes.write();
    history.push_back(WorkspaceChange {
        generation: next,
        path: path.to_string(),
        kind,
        origin,
    });
    let capacity = capacity.max(1);
    while history.len() > capacity {
        history.pop_front();
    }
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

        // Workspace manifests must survive a process restart.  Content blobs
        // live in ContentStore's database; this independent manifest database
        // retains their task/reason/path mapping.
        let snap_db = Self::open_snapshot_database(&config)?;
        let snapshot_manager = Arc::new(SnapshotManager::new(
            snap_db,
            content_store.clone(),
            inventory.clone(),
            config.workspace_root.clone(),
        ));

        let event_bus_for_struct = event_bus.clone();

        // Prepare watching now, but defer directory-watch installation. Even
        // metadata-only inventory scans are not the only startup cost: native
        // non-recursive watches must enumerate every included directory.
        let watch_config = if event_bus.is_some() && config.watch_enabled {
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
            // Synchronize gitignore patterns to FileInventory so full_scan also respects them
            let gitignore_patterns = watch_config.exclude_patterns.clone();
            inventory.read().set_exclude_patterns(gitignore_patterns);
            Some(watch_config)
        } else {
            None
        };

        // Build self
        let ws = Self {
            config,
            inventory,
            content_store,
            snapshot_manager,
            watch_engine: Arc::new(RwLock::new(None)),
            watch_config,
            event_bus: event_bus_for_struct,
            perception_store: RwLock::new(None),
            causal_engine: RwLock::new(None),
            async_started: AtomicBool::new(false),
            generation: Arc::new(AtomicU64::new(0)),
            scan_complete: Arc::new(AtomicBool::new(false)),
            scan_notify: Arc::new(tokio::sync::Notify::new()),
            changes: Arc::new(RwLock::new(VecDeque::new())),
            recent_agent_writes: Arc::new(RwLock::new(std::collections::HashMap::new())),
        };

        // Event consumers are deferred to start_async_components() which is called
        // from within an async context (process_task) after finalize_setup() has
        // wired the perception_store. This avoids spawning consumers with None.
        if !ws.config.defer_initial_scan {
            let discovered = ws.inventory.read().full_scan(&root);
            ws.scan_complete.store(true, Ordering::Release);
            ws.scan_notify.notify_waiters();
            ws.generation.store(1, Ordering::Release);
            debug!(discovered, "Initial workspace metadata scan completed");
        }

        info!("WorkspaceMonitor initialized for root={}", root);

        Ok(ws)
    }

    /// Read a file through ContentStore with cache/diff support.
    pub fn read_file(&self, path: &str, mode: ReadMode) -> std::io::Result<ReadResult> {
        let normalized = self.normalize_path(path);
        let result = self.content_store.read_file(&normalized, mode)?;

        // Update FileInventory state
        let inv = self.inventory.read();
        if result.changed {
            inv.add_or_update(&normalized);
        }
        inv.mark_read_with_hash(
            &normalized,
            result.version,
            self.content_store.get_hash(&normalized),
        );

        Ok(result)
    }

    /// Mark a file as externally read without disk I/O.
    /// Used when file content was provided via read_full_result micro-tool,
    /// so subsequent file_read calls recognize it as already-read.
    pub fn mark_file_read_external(&self, path: &str) {
        let path = self.normalize_path(path);
        let inv = self.inventory.read();
        inv.mark_external_read(&path);
    }

    /// Mark a file as written by the agent.
    pub fn mark_file_written(&self, path: &str) {
        let path = self.normalize_path(path);
        let inv = self.inventory.read();
        let existed = inv.get_entry(&path).is_some();
        inv.mark_written(&path);
        self.content_store.invalidate(&path);
        self.recent_agent_writes
            .write()
            .insert(path.clone(), std::time::Instant::now());
        self.record_change(
            &path,
            if existed {
                WorkspaceChangeKind::Modified
            } else {
                WorkspaceChangeKind::Created
            },
            WorkspaceChangeOrigin::AgentTool,
        );
    }

    pub fn normalize_path(&self, path: &str) -> String {
        let candidate = std::path::Path::new(path);
        let joined = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.config.workspace_root.join(candidate)
        };
        std::fs::canonicalize(&joined)
            .unwrap_or(joined)
            .to_string_lossy()
            .to_string()
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Compute a bounded digest of substantive workspace file paths and
    /// contents. This is execution evidence only: the digest is never placed
    /// in conversational history and therefore does not weaken L1's
    /// current-session/history-summary boundary.
    pub fn semantic_effect_fingerprint(&self) -> Result<String, String> {
        self.inventory.read().semantic_fingerprint(
            &self.config.workspace_root,
            self.config.effect_snapshot_max_files,
            self.config.effect_snapshot_max_bytes,
        )
    }

    pub fn scan_complete(&self) -> bool {
        self.scan_complete.load(Ordering::Acquire)
    }

    pub async fn wait_for_initial_scan(&self) -> bool {
        if self.scan_complete() {
            return true;
        }
        let wait = self.scan_notify.notified();
        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.initial_scan_wait_ms),
            wait,
        )
        .await;
        self.scan_complete()
    }

    /// Return a bounded, generation-stamped view. File contents are never
    /// embedded here; callers recover them through the existing micro-tools.
    pub fn workspace_view(
        &self,
        since_generation: Option<u64>,
        objective: Option<&str>,
        max_files: usize,
    ) -> WorkspaceView {
        let terms = objective
            .unwrap_or_default()
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|term| term.chars().count() >= 3)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let mut entries = self.inventory.read().list_all();
        entries.sort_by(|left, right| {
            let relevance = |path: &str| {
                let normalized = path.to_lowercase();
                terms
                    .iter()
                    .filter(|term| normalized.contains(term.as_str()))
                    .count()
            };
            relevance(&right.path)
                .cmp(&relevance(&left.path))
                .then_with(|| left.path.cmp(&right.path))
        });
        let total_files = entries.len();
        let files = entries
            .into_iter()
            .take(max_files)
            .map(|entry| WorkspaceFileView {
                path: entry.path,
                file_size: entry.file_size,
                language: entry.language,
                state: entry.state.as_str().to_string(),
                version: entry.current_version,
                content_hash: entry.content_hash,
            })
            .collect::<Vec<_>>();
        let since = since_generation.unwrap_or(0);
        let changes = self
            .changes
            .read()
            .iter()
            .filter(|change| change.generation > since)
            .cloned()
            .collect();
        WorkspaceView {
            generation: self.generation(),
            scan_complete: self.scan_complete(),
            total_files,
            truncated: files.len() < total_files,
            files,
            changes,
        }
    }

    pub fn format_delta_since(&self, since_generation: u64, max_changes: usize) -> Option<String> {
        let current = self.generation();
        if current <= since_generation {
            return None;
        }
        let changes = self
            .changes
            .read()
            .iter()
            .filter(|change| change.generation > since_generation)
            .take(max_changes)
            .map(|change| {
                format!(
                    "- generation={} kind={:?} origin={:?} path={}",
                    change.generation, change.kind, change.origin, change.path
                )
            })
            .collect::<Vec<_>>();
        Some(if changes.is_empty() {
            format!(
                "Workspace inventory generation advanced from {} to {}; initial scan complete={}",
                since_generation,
                current,
                self.scan_complete()
            )
        } else {
            format!(
                "Workspace generation {} -> {} (scan_complete={}):\n{}",
                since_generation,
                current,
                self.scan_complete(),
                changes.join("\n")
            )
        })
    }

    fn record_change(&self, path: &str, kind: WorkspaceChangeKind, origin: WorkspaceChangeOrigin) {
        record_workspace_change(
            &self.generation,
            &self.changes,
            self.config.change_history_capacity,
            path,
            kind,
            origin,
        );
    }

    /// Re-scan the entire workspace root, discovering new files and tracking state changes.
    /// Returns the number of newly discovered files.
    /// Skips rescan if WatchEngine is active (events handle incremental discovery).
    pub fn rescan(&self) -> usize {
        if self.watch_engine_active() {
            debug!("rescan skipped: WatchEngine active");
            return 0;
        }
        let root = self.config.workspace_root.to_string_lossy().to_string();
        let discovered = self.inventory.read().full_scan(&root);
        if discovered > 0 {
            info!(discovered = discovered, "rescan discovered new files");
            self.inject_file_perception(None);
        }
        discovered
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
        let generation = self.generation.clone();
        let changes = self.changes.clone();
        let recent_agent_writes = self.recent_agent_writes.clone();
        let change_history_capacity = self.config.change_history_capacity;
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
                                record_workspace_change(
                                    &generation,
                                    &changes,
                                    change_history_capacity,
                                    &path,
                                    WorkspaceChangeKind::Created,
                                    WorkspaceChangeOrigin::External,
                                );
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
                                    )
                                    .with_priority(6);
                                    p.store_global(entry);
                                }
                            }
                            EventType::WorkspaceFileModified => {
                                let inv = inventory.read();
                                let newly_discovered = inv.get_entry(&path).is_none()
                                    && std::path::Path::new(&path).exists();
                                if newly_discovered {
                                    drop(inv);
                                    inventory.read().add_or_update(&path);
                                } else {
                                    drop(inv);
                                }
                                let agent_origin = recent_agent_writes
                                    .write()
                                    .remove(&path)
                                    .is_some_and(|instant| {
                                        instant.elapsed() <= std::time::Duration::from_secs(3)
                                    });
                                if !agent_origin {
                                    inventory.read().mark_stale(&path);
                                    record_workspace_change(
                                        &generation,
                                        &changes,
                                        change_history_capacity,
                                        &path,
                                        if newly_discovered {
                                            WorkspaceChangeKind::Created
                                        } else {
                                            WorkspaceChangeKind::Modified
                                        },
                                        WorkspaceChangeOrigin::External,
                                    );
                                }
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
                                    )
                                    .with_priority(6);
                                    p.store_global(entry);
                                }
                            }
                            EventType::WorkspaceFileRemoved => {
                                inventory.read().remove(&path);
                                recent_agent_writes.write().remove(&path);
                                record_workspace_change(
                                    &generation,
                                    &changes,
                                    change_history_capacity,
                                    &path,
                                    WorkspaceChangeKind::Removed,
                                    WorkspaceChangeOrigin::External,
                                );
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
                                    )
                                    .with_priority(5);
                                    p.store_global(entry);
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("WorkspaceMonitor event consumer lagged by {} events", n);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("WorkspaceMonitor event bus closed (shutdown)");
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
    /// Idempotent: only runs once regardless of how many times it's called.
    pub fn start_async_components(&self) {
        if self.async_started.swap(true, Ordering::SeqCst) {
            return;
        }
        if tokio::runtime::Handle::try_current().is_err() {
            tracing::error!("start_async_components must be called from within a tokio runtime");
            self.async_started.store(false, Ordering::SeqCst);
            return;
        }
        self.register_event_consumers();
        if let (Some(watch_config), Some(event_bus)) =
            (self.watch_config.clone(), self.event_bus.clone())
        {
            let watch_engine = self.watch_engine.clone();
            let inventory = self.inventory.clone();
            let root = self.config.workspace_root.to_string_lossy().to_string();
            let runtime = tokio::runtime::Handle::current();
            tokio::task::spawn_blocking(move || {
                // WatchEngine's polling fallback and event callback capture the
                // runtime handle from this entered context.
                let _runtime_guard = runtime.enter();
                match WatchEngine::start(&root, watch_config, event_bus, Some(inventory)) {
                    Ok(engine) => {
                        *watch_engine.write() = Some(engine);
                        info!(root = %root, "Deferred WatchEngine installation completed");
                    }
                    Err(error) => {
                        warn!(root = %root, %error, "Deferred WatchEngine installation failed");
                    }
                }
            });
        }
        if !self.scan_complete() {
            let inventory = self.inventory.clone();
            let root = self.config.workspace_root.to_string_lossy().to_string();
            let scan_complete = self.scan_complete.clone();
            let generation = self.generation.clone();
            let scan_notify = self.scan_notify.clone();
            tokio::task::spawn_blocking(move || {
                let discovered = inventory.read().full_scan(&root);
                scan_complete.store(true, Ordering::Release);
                generation.fetch_add(1, Ordering::AcqRel);
                scan_notify.notify_waiters();
                debug!(discovered, "Deferred workspace metadata scan completed");
            });
        }
    }

    /// Check whether a native or polling WatchEngine is actively monitoring the filesystem.
    /// Returns false when WatchEngine was never started (no EventBus) or failed to start.
    pub fn watch_engine_active(&self) -> bool {
        self.watch_engine.read().is_some()
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
                            let injections = ctx
                                .metadata
                                .entry("guard_pre_injections".to_string())
                                .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                            if let Some(arr) = injections.as_array_mut() {
                                arr.push(serde_json::Value::String(warning));
                            }
                        }
                        FileState::ReadFresh
                            if entry.last_read_version == entry.current_version =>
                        {
                            // File unchanged since last read — inject hint to skip full re-read
                            ctx.data.insert(
                                "file_unchanged".to_string(),
                                serde_json::Value::Bool(true),
                            );
                            let hint = format!(
                                "[workspace_monitor] File '{}' unchanged since last read (v{}). Use mode:diff for changes, or a targeted offset/limit range when prior content is no longer visible.",
                                path, entry.current_version
                            );
                            ctx.data.insert(
                                "file_unchanged_hint".to_string(),
                                serde_json::Value::String(hint.clone()),
                            );
                            // Also write to metadata so ToolGuard pre-injection
                            let injections = ctx
                                .metadata
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
            if keywords.is_empty() {
                return 0;
            }
            let path_lower = path.to_lowercase();
            keywords
                .iter()
                .filter(|k| path_lower.contains(k.as_str()))
                .count()
        };

        let mut stale: Vec<_> = all
            .iter()
            .filter(|e| e.state == FileState::ReadStale)
            .collect();
        let mut written_unread: Vec<_> = all
            .iter()
            .filter(|e| e.state == FileState::WrittenUnread)
            .collect();
        let discovered: Vec<_> = all
            .iter()
            .filter(|e| e.state == FileState::Discovered)
            .collect();

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
                if names.is_empty() {
                    String::new()
                } else {
                    format!(": {}", names.join(", "))
                }
            ));
        }

        if !written_unread.is_empty() {
            let names: Vec<&str> = written_unread
                .iter()
                .take(5)
                .map(|e| e.path.as_str())
                .collect();
            parts.push(format!(
                "{} written but not re-read{}",
                written_unread.len(),
                if names.is_empty() {
                    String::new()
                } else {
                    format!(": {}", names.join(", "))
                }
            ));
        }

        if !discovered.is_empty() {
            let names: Vec<&str> = discovered
                .iter()
                .take(10)
                .map(|e| e.path.as_str())
                .collect();
            parts.push(format!(
                "{} new discovered files unread{}",
                discovered.len(),
                if names.is_empty() {
                    String::new()
                } else {
                    format!(": {}", names.join(", "))
                }
            ));
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
        let dirs: HashSet<String> = all
            .iter()
            .filter_map(|e| {
                let parent = std::path::Path::new(&e.path).parent()?;
                Some(parent.to_string_lossy().into_owned())
            })
            .collect();

        // Count by language
        let mut lang_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for entry in &all {
            let lang = if entry.language == "unknown" {
                "other"
            } else {
                entry.language.as_str()
            };
            *lang_counts.entry(lang).or_insert(0) += 1;
        }

        // Sort by count descending, take top 3
        let mut lang_vec: Vec<(&str, usize)> = lang_counts.into_iter().collect();
        lang_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let lang_summary = lang_vec
            .iter()
            .take(3)
            .map(|(l, c)| format!("{} {}", c, l))
            .collect::<Vec<_>>()
            .join(", ");

        let summary = if lang_vec.len() > 3 {
            format!(
                "{} files across {} directories ({} … +{} more)",
                total,
                dirs.len(),
                lang_summary,
                lang_vec.len() - 3
            )
        } else {
            format!(
                "{} files across {} directories ({})",
                total,
                dirs.len(),
                lang_summary
            )
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

    fn open_snapshot_database(
        config: &WorkspaceMonitorConfig,
    ) -> Result<Arc<redb::Database>, String> {
        match &config.db_path {
            Some(path) => {
                std::fs::create_dir_all(path).map_err(|error| {
                    format!("Failed to create snapshot database directory: {error}")
                })?;
                let snapshot_path = path.join("snapshots");
                redb::Database::create(snapshot_path)
                    .map(Arc::new)
                    .map_err(|error| format!("Failed to open snapshot redb: {error}"))
            }
            None => redb::Builder::new()
                .create_with_backend(redb::backends::InMemoryBackend::new())
                .map(Arc::new)
                .map_err(|error| format!("Failed to open snapshot redb: {error}")),
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
        ctx.data.insert(
            "path".to_string(),
            Value::String(file_path.to_string_lossy().to_string()),
        );

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(hm.execute(HookPoint::SkillBefore, &mut ctx));
        assert_eq!(result, HookResult::Continue);

        let warning = ctx
            .data
            .get("stale_warning")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            warning.contains("stale"),
            "Expected stale warning, got: {}",
            warning
        );
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
        ctx.data.insert(
            "path".to_string(),
            Value::String(file_path.to_string_lossy().to_string()),
        );

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
        assert!(
            entry.is_some(),
            "File should be in inventory after Create event"
        );
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
        assert!(ws
            .inventory
            .read()
            .get_entry(&test_file.to_string_lossy())
            .is_some());

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
        assert!(
            inv.get_entry(&test_file.to_string_lossy()).is_none(),
            "File should be removed from inventory"
        );
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
        assert_eq!(
            entry.state,
            FileState::ReadStale,
            "File should be stale after modify event"
        );

        // Step 3: Hook SkillBefore read detects stale state
        let mut ctx = HookContext::new(HookPoint::SkillBefore, "agent_1", "DA");
        ctx.data
            .insert("path".to_string(), Value::String(file_path_str.clone()));
        let result = hm.execute(HookPoint::SkillBefore, &mut ctx).await;
        assert_eq!(result, HookResult::Continue);
        let warning = ctx
            .data
            .get("stale_warning")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            warning.contains("stale"),
            "Expected stale warning in lifecycle: {}",
            warning
        );

        // Step 4: File write → SkillAfter hook marks WrittenUnread
        let mut write_ctx = HookContext::new(HookPoint::SkillAfter, "agent_1", "DA");
        write_ctx
            .data
            .insert("path".to_string(), Value::String(file_path_str.clone()));
        let _ = hm.execute(HookPoint::SkillAfter, &mut write_ctx).await;
        let entry = ws.inventory.read().get_entry(&file_path_str).unwrap();
        assert_eq!(
            entry.state,
            FileState::WrittenUnread,
            "File should be WrittenUnread after write hook"
        );

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
        ctx.data.insert(
            "path".to_string(),
            Value::String(file_path.to_string_lossy().to_string()),
        );

        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(hm.execute(HookPoint::SkillBefore, &mut ctx));
        assert_eq!(result, HookResult::Continue);

        assert_eq!(
            ctx.data.get("file_unchanged").and_then(|v| v.as_bool()),
            Some(true),
            "Expected file_unchanged flag for ReadFresh file with matching version"
        );
        let hint = ctx
            .data
            .get("file_unchanged_hint")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            hint.contains("unchanged"),
            "Expected unchanged hint: {}",
            hint
        );
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
        ctx.data.insert(
            "path".to_string(),
            Value::String(file_path.to_string_lossy().to_string()),
        );

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

        assert!(
            ps.has_new("iri://test_task"),
            "PerceptionStore should have new entry after create event"
        );
        let text = ps.take_perception_text("iri://test_task");
        assert!(
            text.contains("New file"),
            "Perception text should mention file creation: {}",
            text
        );
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

        assert!(
            ps.has_new("iri://test_task"),
            "PerceptionStore should have new entry after modify event"
        );
        let text = ps.take_perception_text("iri://test_task");
        assert!(
            text.contains("externally modified"),
            "Perception text should mention external change: {}",
            text
        );
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

        assert!(
            ps.has_new("iri://test_task"),
            "PerceptionStore should have new entry after remove event"
        );
        let text = ps.take_perception_text("iri://test_task");
        assert!(
            text.contains("File deleted"),
            "Perception text should mention file removal: {}",
            text
        );
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
        assert!(
            text.contains("stale"),
            "Should mention stale files: {}",
            text
        );
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
        assert!(
            t.contains("files total"),
            "Should mention total file count: {}",
            t
        );
    }

    #[test]
    fn test_generate_perception_text_lists_discovered_paths() {
        let (ws, dir) = temp_ws_monitor();

        let f1 = dir.path().join("new_a.js");
        let f2 = dir.path().join("new_b.js");
        let f3 = dir.path().join("new_c.js");
        std::fs::write(&f1, "// a").unwrap();
        std::fs::write(&f2, "// b").unwrap();
        std::fs::write(&f3, "// c").unwrap();

        // add_or_update leaves entries in Discovered state (never read)
        {
            let inv = ws.inventory.read();
            inv.add_or_update(&f1.to_string_lossy()).unwrap();
            inv.add_or_update(&f2.to_string_lossy()).unwrap();
            inv.add_or_update(&f3.to_string_lossy()).unwrap();
        }

        let text = ws.generate_perception_text(None).expect("perception text");
        assert!(
            text.contains("3 new discovered files unread"),
            "count with names: {}",
            text
        );
        assert!(text.contains("new_a.js"), "lists first discovered path");
        assert!(text.contains("new_c.js"), "lists discovered paths up to 10");
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
        assert!(
            !text1.is_empty(),
            "Perception should be available after create event"
        );
        assert!(
            text1.contains("New file"),
            "Should mention new file: {}",
            text1
        );

        // Second take should be empty (consumed)
        let text2 = ps.take_perception_text("iri://task_full");
        assert!(
            text2.is_empty(),
            "Second take should be empty after consumption"
        );
    }

    // ── New optimization tests ──

    #[tokio::test]
    async fn test_start_async_components_idempotent() {
        let dir = tempfile::TempDir::new().unwrap();
        let bus = Arc::new(EventBus::new(100));

        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };

        let ws = WorkspaceMonitor::initialize(config, None, Some(bus)).unwrap();

        // First call should succeed (inside tokio runtime)
        ws.start_async_components();

        assert!(
            ws.async_started.load(std::sync::atomic::Ordering::SeqCst),
            "async_started should be true after first call"
        );

        // Second call must not spawn another consumer
        let old_flag = ws.async_started.load(std::sync::atomic::Ordering::SeqCst);
        ws.start_async_components();
        assert_eq!(
            ws.async_started.load(std::sync::atomic::Ordering::SeqCst),
            old_flag,
            "async_started unchanged after second call"
        );
    }

    #[test]
    fn test_watch_engine_active() {
        let dir = tempfile::TempDir::new().unwrap();

        // Without EventBus — no watch engine
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None).unwrap();
        assert!(
            !ws.watch_engine_active(),
            "No EventBus → watch_engine should be None"
        );
    }

    #[test]
    fn test_rescan_empty_inventory() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None).unwrap();

        // rescan on empty inventory with no watch engine should do full scan
        let count = ws.rescan();
        // No files in temp dir, so count should be 0 or small
        assert_eq!(count, 0, "rescan on empty dir should discover 0 new files");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_watch_engine_installation_is_deferred_then_rescan_is_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        let bus = Arc::new(EventBus::new(100));
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: true,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, Some(bus)).unwrap();
        assert!(
            !ws.watch_engine_active(),
            "watch enumeration must not delay synchronous/TUI initialization"
        );
        ws.start_async_components();
        for _ in 0..100 {
            if ws.watch_engine_active() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(ws.watch_engine_active());

        // An active watcher owns incremental discovery, so no full rescan runs.
        let count = ws.rescan();
        assert_eq!(
            count, 0,
            "rescan should return 0 when watch engine is active or polling is fallback"
        );
    }

    #[test]
    fn test_rescan_discoveres_new_files_without_watch() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None).unwrap();

        // Add file AFTER initialization
        let new_file = dir.path().join("new_file.rs");
        std::fs::write(&new_file, "fn new() {}").unwrap();

        // rescan should discover it (no watch engine active)
        let count = ws.rescan();
        assert_eq!(count, 1, "rescan should discover the new file");

        // Verify it's in inventory
        let entry = ws.inventory.read().get_entry(&new_file.to_string_lossy());
        assert!(
            entry.is_some(),
            "New file should be in inventory after rescan"
        );
    }

    #[test]
    fn test_rescan_does_not_re_add_tracked_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None).unwrap();

        // Add a file
        let file = dir.path().join("tracked.rs");
        std::fs::write(&file, "fn tracked() {}").unwrap();
        let count1 = ws.rescan();
        assert_eq!(count1, 1, "First rescan should find the file");

        // Second rescan: file already tracked, should be 0 new
        let count2 = ws.rescan();
        assert_eq!(count2, 0, "Second rescan should not re-add tracked file");
    }

    #[test]
    fn test_rescan_injects_perception_on_new_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let ps = Arc::new(PerceptionStore::new());
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None)
            .unwrap()
            .with_perception_store(ps.clone());

        // Add file after init
        let file = dir.path().join("percept.rs");
        std::fs::write(&file, "fn percept() {}").unwrap();

        ws.rescan();

        // PerceptionStore should have an entry from rescan's inject_file_perception
        let text = ps.take_perception_text("iri://task/rescan_test");
        assert!(
            !text.is_empty(),
            "rescan should inject perception on new files"
        );
        assert!(
            text.contains("files total"),
            "Perception text should contain file summary"
        );
    }

    #[test]
    fn test_file_inventory_summary_format() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None).unwrap();

        // Empty → None
        assert!(
            ws.get_file_inventory_summary().is_none(),
            "Empty inventory should return None"
        );

        // Add files
        for name in &["main.rs", "lib.rs", "README.md"] {
            let path = dir.path().join(name);
            std::fs::write(&path, "content").unwrap();
        }
        ws.rescan();

        let summary = ws.get_file_inventory_summary();
        assert!(
            summary.is_some(),
            "Non-empty inventory should return summary"
        );
        let s = summary.unwrap();
        assert!(
            s.contains("files across"),
            "Summary should mention files/dirs: {}",
            s
        );
        assert!(
            s.contains("rust"),
            "Summary should mention rust language: {}",
            s
        );
    }

    #[test]
    fn test_reset_inventory_clears_perception_global() {
        let dir = tempfile::TempDir::new().unwrap();
        let ps = Arc::new(PerceptionStore::new());
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None)
            .unwrap()
            .with_perception_store(ps.clone());

        // Add a file so inject_file_perception has content to work with
        let file = dir.path().join("tracked.rs");
        std::fs::write(&file, "fn tracked() {}").unwrap();
        ws.inventory
            .read()
            .add_or_update(&file.to_string_lossy())
            .unwrap();

        ws.inject_file_perception(Some("test task"));
        assert!(
            ps.has_new("iri://task/check"),
            "Should have perception after inject"
        );

        ws.reset_inventory();
        assert!(
            !ps.has_new("iri://task/check"),
            "Should have no perception after reset"
        );
    }

    #[test]
    fn test_gitignore_patterns_synced_to_inventory() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "*.log\nbuild/\n.env\n").unwrap();
        let files = ["app.log", "src/main.rs", "build/output.o", ".env"];
        for f in &files {
            let path = dir.path().join(f);
            if let Some(parent) = std::path::Path::new(&path).parent() {
                std::fs::create_dir_all(parent).unwrap_or(());
            }
            std::fs::write(&path, "x").unwrap();
        }

        let bus = Arc::new(EventBus::new(100));
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: true,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, Some(bus)).unwrap();

        let inv = ws.inventory.read();
        // *.log → app.log should be excluded
        assert!(inv.is_excluded(std::path::Path::new("app.log")));
        // build/ → build/output.o should be excluded
        assert!(inv.is_excluded(std::path::Path::new("build/output.o")));
        // .env should be excluded
        assert!(inv.is_excluded(std::path::Path::new(".env")));
        // src/main.rs should NOT be excluded
        assert!(!inv.is_excluded(std::path::Path::new("src/main.rs")));
    }

    #[test]
    fn test_force_rescan_after_reset() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = WorkspaceMonitorConfig {
            workspace_root: dir.path().to_path_buf(),
            watch_enabled: false,
            db_path: None,
            ..WorkspaceMonitorConfig::default()
        };
        let ws = WorkspaceMonitor::initialize(config, None, None).unwrap();

        // Add files
        std::fs::write(dir.path().join("file_a.rs"), "fn a() {}").unwrap();
        ws.rescan();
        assert_eq!(ws.inventory.read().total_count(), 1);

        // Reset inventory
        ws.reset_inventory();
        assert_eq!(
            ws.inventory.read().total_count(),
            0,
            "Inventory should be empty after reset"
        );

        // Rescan should rediscover
        let count = ws.rescan();
        assert_eq!(count, 1, "rescan should rediscover files after reset");
        assert_eq!(ws.inventory.read().total_count(), 1);
    }

    #[test]
    fn test_content_store_process_content_equals_read_file() {
        // Verify process_content produces the same result as read_file
        let dir = tempfile::TempDir::new().unwrap();

        // Use separate files for each test path to avoid cross-contamination
        let p1 = dir.path().join("proc_1.rs");
        std::fs::write(&p1, "fn proc1() {}").unwrap();
        let p1s = p1.to_string_lossy().to_string();

        let p2 = dir.path().join("proc_2.rs");
        std::fs::write(&p2, "fn proc2() {}").unwrap();
        let p2s = p2.to_string_lossy().to_string();

        let store = crate::tools::workspace_monitor::ContentStore::new(100, 65536, None);

        // Read file1 via read_file (reads disk)
        let from_disk = store
            .read_file(&p1s, crate::tools::workspace_monitor::ReadMode::Full)
            .unwrap();
        assert!(
            !from_disk.from_cache,
            "First read_file should not be from cache"
        );

        // Read file2 via process_content with pre-read content
        let content = std::fs::read_to_string(&p2s).unwrap();
        let from_content = store.process_content(
            &p2s,
            &content,
            crate::tools::workspace_monitor::ReadMode::Full,
        );

        // Both should have version 1 (first read)
        assert_eq!(from_disk.version, 1);
        assert_eq!(from_content.version, 1);
        assert!(
            !from_content.from_cache,
            "First process_content should not be from cache"
        );
        assert_eq!(from_content.lines.len(), 1, "Should have 1 line of content");

        // Second call via process_content on SAME file should be cache hit
        let cached = store.process_content(
            &p2s,
            &content,
            crate::tools::workspace_monitor::ReadMode::Full,
        );
        assert!(
            cached.from_cache,
            "Second process_content should return from_cache=true"
        );
        assert_eq!(cached.version, 1, "Version should not change on cache hit");
    }

    #[test]
    fn test_content_store_diff_mode_via_process_content() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test_diff.rs");
        std::fs::write(&path, "line1\nline2\nline3").unwrap();
        let path_str = path.to_string_lossy().to_string();

        let store = crate::tools::workspace_monitor::ContentStore::new(100, 65536, None);

        // First read
        let r1_content = std::fs::read_to_string(&path_str).unwrap();
        let r1 = store.process_content(
            &path_str,
            &r1_content,
            crate::tools::workspace_monitor::ReadMode::Full,
        );
        assert_eq!(r1.version, 1);

        // Modify file
        std::fs::write(&path, "line1\nmodified\nline3").unwrap();

        // Read via process_content again with Diff mode
        let r2_content = std::fs::read_to_string(&path_str).unwrap();
        let r2 = store.process_content(
            &path_str,
            &r2_content,
            crate::tools::workspace_monitor::ReadMode::Diff,
        );

        assert_eq!(r2.version, 2, "Version should increment on change");
        assert!(r2.changed, "Changed flag should be true");
        assert!(r2.unified_diff.is_some(), "Diff should be computed");
        assert!(
            r2.unified_diff.as_ref().unwrap().contains("modified"),
            "Diff should contain the modified line"
        );
    }

    #[test]
    fn workspace_view_tracks_agent_change_with_generation() {
        let (ws, dir) = temp_ws_monitor();
        let path = dir.path().join("generated.txt");
        std::fs::write(&path, "generated").unwrap();
        ws.mark_file_written(path.to_string_lossy().as_ref());

        let view = ws.workspace_view(Some(0), Some("generated"), 20);
        assert!(view.generation > 0);
        assert!(view.scan_complete);
        assert!(view
            .files
            .iter()
            .any(|file| file.path == ws.normalize_path(path.to_string_lossy().as_ref())));
        assert!(view.changes.iter().any(|change| {
            change.origin == WorkspaceChangeOrigin::AgentTool
                && change.kind == WorkspaceChangeKind::Created
        }));
    }

    #[test]
    fn initial_scan_is_metadata_only_and_targeted_read_populates_hash() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("source.rs");
        std::fs::write(&path, "fn main() {}\n").unwrap();
        let ws = WorkspaceMonitor::initialize(
            WorkspaceMonitorConfig {
                workspace_root: dir.path().to_path_buf(),
                watch_enabled: false,
                ..Default::default()
            },
            None,
            None,
        )
        .unwrap();
        let normalized = ws.normalize_path(path.to_string_lossy().as_ref());
        assert_eq!(
            ws.inventory
                .read()
                .get_entry(&normalized)
                .unwrap()
                .content_hash,
            ""
        );
        ws.read_file(&normalized, ReadMode::Full).unwrap();
        assert!(!ws
            .inventory
            .read()
            .get_entry(&normalized)
            .unwrap()
            .content_hash
            .is_empty());
    }

    #[tokio::test]
    async fn deferred_initial_scan_completes_without_blocking_initialize() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        let ws = WorkspaceMonitor::initialize(
            WorkspaceMonitorConfig {
                workspace_root: dir.path().to_path_buf(),
                watch_enabled: false,
                defer_initial_scan: true,
                initial_scan_wait_ms: 2_000,
                ..Default::default()
            },
            None,
            Some(Arc::new(EventBus::new(16))),
        )
        .unwrap();
        assert!(!ws.scan_complete());
        ws.start_async_components();
        assert!(ws.wait_for_initial_scan().await);
        assert_eq!(ws.workspace_view(None, None, 10).total_files, 1);
    }
}
