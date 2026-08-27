use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};
use parking_lot::RwLock;
use tracing::{debug, warn};

use crate::core::event_bus::EventBus;
use crate::tools::workspace_monitor::inventory::FileInventory;

/// Minimum time between events for the same path before the watcher re-emits it.
const MIN_EVENT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1000);

/// Configuration for the WatchEngine.
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Debounce time window in milliseconds.
    pub debounce_ms: u64,
    /// Maximum debounce wait in milliseconds.
    pub max_debounce_wait_ms: u64,
    /// Polling interval in milliseconds (fallback mode).
    pub poll_interval_ms: u64,
    /// Enable native file system watching.
    pub watch_enabled: bool,
    /// Glob-like patterns to exclude from file events.
    /// e.g. "node_modules/", "data/", ".git/".
    pub exclude_patterns: Vec<String>,
    /// Whether to load .gitignore patterns from workspace root.
    pub use_gitignore: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 500,
            max_debounce_wait_ms: 5000,
            poll_interval_ms: 5000,
            watch_enabled: true,
            exclude_patterns: vec![],
            use_gitignore: true,
        }
    }
}

impl WatchConfig {
    /// Check if a path should be excluded from file events.
    /// Matches patterns against whole path components only (not substrings within a filename).
    pub fn is_excluded(&self, path: &str) -> bool {
        let normalized = path.replace('\\', "/");
        for pattern in &self.exclude_patterns {
            let pat = pattern.replace('\\', "/");
            if crate::tools::workspace_monitor::inventory::match_glob_pattern(&normalized, &pat) {
                return true;
            }
        }
        false
    }

    /// Load .gitignore patterns from the workspace root and append them to exclude_patterns.
    pub fn load_gitignore(&mut self, root: &Path) {
        let gitignore_path = root.join(".gitignore");
        let content = match std::fs::read_to_string(&gitignore_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut loaded = 0;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Strip leading / (gitignore root-anchored patterns)
            let pat = if let Some(stripped) = line.strip_prefix('/') {
                stripped
            } else {
                line
            };
            // Normalize: ensure trailing / for directories
            let pat = if !pat.contains('.') && !pat.ends_with('/') {
                format!("{}/", pat)
            } else {
                pat.to_string()
            };
            // Only add if not already present
            if !self.exclude_patterns.contains(&pat) {
                self.exclude_patterns.push(pat);
                loaded += 1;
            }
        }
        if loaded > 0 {
            debug!(count = loaded, source = %gitignore_path.display(), "WatchEngine: loaded gitignore patterns");
        }
    }
}

/// WatchEngine wraps `notify` for cross-platform file system monitoring.
///
/// Events are debounced and forwarded to the EventBus as `WorkspaceFile*` events.
/// Falls back to polling when native watching is unavailable.
pub struct WatchEngine {
    debouncer: Option<Arc<Mutex<Option<Debouncer<notify::RecommendedWatcher>>>>>,
    polling_handle: Option<tokio::task::AbortHandle>,
    /// Configuration.
    #[allow(dead_code)]
    config: WatchConfig,
    /// Reference to inventory for direct updates from the watcher callback.
    #[allow(dead_code)]
    inventory: Option<Arc<RwLock<FileInventory>>>,
}

impl WatchEngine {
    /// Start file system watching for a directory.
    ///
    /// Returns a `WatchEngine` that will send events to the EventBus.
    pub fn start(
        root: &str,
        config: WatchConfig,
        event_bus: Arc<EventBus>,
        inventory: Option<Arc<RwLock<FileInventory>>>,
    ) -> Result<Self, String> {
        let root_path = Path::new(root);

        if !root_path.is_dir() {
            return Err(format!("Watch root is not a directory: {}", root));
        }

        if !config.watch_enabled {
            debug!("WatchEngine: file watching disabled by config");
            return Ok(Self {
                debouncer: None,
                polling_handle: None,
                config,
                inventory,
            });
        }

        // Attempt native watching — runs on its own thread (notify), no tokio needed.
        let inv_for_callback = inventory.clone();
        let debouncer = match Self::try_start_native(
            root,
            &config,
            event_bus.clone(),
            inv_for_callback,
        ) {
            Ok(d) => {
                debug!(root = %root, "WatchEngine: native watching started");
                Some(d)
            }
            Err(e) => {
                warn!(root = %root, error = %e, "WatchEngine: native watching failed, falling back to polling");
                return Ok(Self {
                    debouncer: None,
                    polling_handle: Some(Self::start_polling(root, &config, event_bus)),
                    config,
                    inventory,
                });
            }
        };

        Ok(Self {
            debouncer,
            polling_handle: None,
            config,
            inventory,
        })
    }

    /// Check if the engine is using native watching.
    pub fn is_native(&self) -> bool {
        self.debouncer.is_some()
    }

    /// Check if the engine is using polling fallback.
    pub fn is_polling(&self) -> bool {
        self.polling_handle.is_some()
    }

    // ── Native watching ──

    fn try_start_native(
        root: &str,
        config: &WatchConfig,
        event_bus: Arc<EventBus>,
        _inventory: Option<Arc<RwLock<FileInventory>>>,
    ) -> Result<Arc<Mutex<Option<Debouncer<notify::RecommendedWatcher>>>>, String> {
        let debounce = Duration::from_millis(config.debounce_ms);
        let root_owned = root.to_string();
        let exclude = config.exclude_patterns.clone();
        let watch_exclude = config.exclude_patterns.clone();
        let rt_handle = tokio::runtime::Handle::try_current().ok();

        // Per-path cooldown prevents feedback loops: when the app's own output
        // (e.g. a mirrored log file) lives inside the watched workspace, every
        // emitted event produces new log lines that re-trigger the watcher.
        // Silently skipping the same path within the window breaks the cycle.
        let last_emit =
            std::sync::Mutex::new(std::collections::HashMap::<String, std::time::Instant>::new());

        // Shared handle so the event callback can install watches on newly
        // created directories (NonRecursive watches do not auto-follow).
        // The callback holds a Weak ref: the strong ref lives in WatchEngine,
        // so dropping the engine drops the Debouncer (which sends Shutdown).
        // A strong ref here would cycle: thread -> closure -> Arc -> Debouncer.
        let shared_debouncer: Arc<Mutex<Option<Debouncer<notify::RecommendedWatcher>>>> =
            Arc::new(Mutex::new(None));
        let shared_for_callback = Arc::downgrade(&shared_debouncer);

        let mut debouncer = new_debouncer(debounce, move |result: DebounceEventResult| {
            let now = std::time::Instant::now();
            let mut approved: Vec<notify_debouncer_mini::DebouncedEvent> = match &result {
                Ok(events) => {
                    let mut last = last_emit.lock().unwrap();
                    let mut out = Vec::new();
                    for event in events {
                        let Some(path) = event.path.to_str() else {
                            continue;
                        };
                        if Self::is_path_excluded(path, &exclude) {
                            debug!(path = %path, "WatchEngine: excluded event");
                            continue;
                        }
                        if Self::within_cooldown(&mut last, path, now) {
                            continue;
                        }
                        last.insert(path.to_string(), now);
                        out.push(event.clone());
                    }
                    out
                }
                Err(_) => Vec::new(),
            };
            // A directory creation event means future writes inside it would
            // be invisible (NonRecursive watches do not follow new children).
            // Install a watch on the new directory and its existing children
            // so subsequent events keep flowing; deeper nesting is picked up
            // by the same mechanism as more directories appear.
            let mut catch_up: Vec<notify_debouncer_mini::DebouncedEvent> = Vec::new();
            if let Some(shared) = shared_for_callback.upgrade() {
                if let Some(ref mut debouncer) = *shared.lock().unwrap() {
                    for event in &approved {
                        let path = &event.path;
                        if path.is_dir()
                            && !Self::is_path_excluded(&path.to_string_lossy(), &exclude)
                        {
                            // Watch the new directory itself, then any of its
                            // immediate subdirectories, so writes at every level
                            // below the parent watch keep producing events.
                            let _ = debouncer.watcher().watch(path, RecursiveMode::NonRecursive);
                            for dir in walkdir::WalkDir::new(path)
                                .min_depth(1)
                                .max_depth(1)
                                .into_iter()
                                .filter_map(|e| e.ok())
                                .filter(|e| e.file_type().is_dir())
                            {
                                let _ = debouncer
                                    .watcher()
                                    .watch(dir.path(), RecursiveMode::NonRecursive);
                            }
                            // Files already present were written before this
                            // directory was watched, so no event fired for them.
                            // Emit catch-up events so the inventory still learns
                            // about them.
                            for entry in walkdir::WalkDir::new(path)
                                .into_iter()
                                .filter_entry(|e| {
                                    e.depth() == 0
                                        || !Self::is_path_excluded(
                                            &e.path().to_string_lossy(),
                                            &exclude,
                                        )
                                })
                                .filter_map(|e| e.ok())
                            {
                                if entry.file_type().is_file() {
                                    catch_up.push(notify_debouncer_mini::DebouncedEvent {
                                        path: entry.path().to_path_buf(),
                                        kind: notify_debouncer_mini::DebouncedEventKind::Any,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            approved.extend(catch_up);
            // Inventory state and change provenance are updated by the EventBus
            // consumer. Mutating it here made an AgentTool write appear as an
            // external stale change before provenance could be resolved.
            for event in &approved {
                let (event_type_str, path_str) = Self::map_event(event);
                debug!(
                    event_type = %event_type_str,
                    path = %path_str,
                    "WatchEngine: event dispatched"
                );
                if let Some(handle) = &rt_handle {
                    let eb = event_bus.clone();
                    handle.spawn(async move {
                        let _ = eb
                            .emit(
                                "iri://workspace",
                                &event_type_str,
                                "iri://workspace_monitor",
                                &path_str,
                            )
                            .await;
                    });
                } else {
                    debug!(
                        event_type = %event_type_str,
                        path = %path_str,
                        "WatchEngine: no runtime handle, event dropped"
                    );
                }
            }
            if let Err(errors) = result {
                warn!(error = ?errors, "WatchEngine: debouncer error");
            }
        })
        .map_err(|e| format!("Failed to create debouncer: {}", e))?;

        // Watch each directory individually (NonRecursive) instead of one
        // Recursive watch on the root: notify's recursive watch enumerates the
        // whole tree at startup (no filter support), which blocks start() for
        // minutes on repositories with large build directories (e.g. target/).
        // Pre-enumerating with filter_entry keeps excluded dirs out of inotify.
        let watcher = debouncer.watcher();
        for entry in walkdir::WalkDir::new(&root_owned)
            .min_depth(1)
            .into_iter()
            .filter_entry(|entry| {
                entry.depth() == 1
                    || !Self::is_path_excluded(&entry.path().to_string_lossy(), &watch_exclude)
            })
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_dir() {
                continue;
            }
            watcher
                .watch(entry.path(), RecursiveMode::NonRecursive)
                .map_err(|e| {
                    format!(
                        "Failed to watch directory {}: {}",
                        entry.path().display(),
                        e
                    )
                })?;
        }
        // Watch the root itself so creations directly under it (the entry
        // point for the dynamic watch extension) are always observed.
        watcher
            .watch(Path::new(&root_owned), RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to watch directory: {}", e))?;
        // Publish the debouncer so the callback can watch newly created
        // directories that appear after startup. It stays in the shared slot
        // for the callback's lifetime; the returned Arc is held by WatchEngine.
        *shared_debouncer.lock().unwrap() = Some(debouncer);
        Ok(shared_debouncer)
    }

    fn is_path_excluded(path: &str, exclude_patterns: &[String]) -> bool {
        let normalized = path.replace('\\', "/");
        for pattern in exclude_patterns {
            let pat = pattern.replace('\\', "/");
            if crate::tools::workspace_monitor::inventory::match_glob_pattern(&normalized, &pat) {
                return true;
            }
        }
        false
    }

    fn within_cooldown(
        last_emit: &mut std::collections::HashMap<String, std::time::Instant>,
        path: &str,
        now: std::time::Instant,
    ) -> bool {
        match last_emit.get(path) {
            Some(prev) => now.duration_since(*prev) < MIN_EVENT_INTERVAL,
            None => false,
        }
    }

    fn map_event(event: &notify_debouncer_mini::DebouncedEvent) -> (String, String) {
        let event_type = if !event.path.exists() {
            "WORKSPACE_FILE_REMOVED"
        } else {
            "WORKSPACE_FILE_MODIFIED"
        };
        let path = event.path.to_str().unwrap_or("").to_string();
        (event_type.to_string(), path)
    }

    // ── Polling fallback ──

    fn start_polling(
        root: &str,
        config: &WatchConfig,
        event_bus: Arc<EventBus>,
    ) -> tokio::task::AbortHandle {
        let root = root.to_string();
        let interval = Duration::from_millis(config.poll_interval_ms);
        let exclude = config.exclude_patterns.clone();

        let handle = tokio::spawn(async move {
            let mut last_mtimes: std::collections::HashMap<String, i64> =
                std::collections::HashMap::new();
            let mut last_emit: std::collections::HashMap<String, std::time::Instant> =
                std::collections::HashMap::new();
            let mut interval_timer = tokio::time::interval(interval);
            interval_timer.tick().await;

            loop {
                interval_timer.tick().await;

                let mut changed = Vec::new();
                let now = std::time::Instant::now();
                for entry in walkdir::WalkDir::new(&root)
                    .into_iter()
                    .filter_map(|e| e.ok())
                {
                    if !entry.file_type().is_file() {
                        continue;
                    }
                    let path = entry.path().to_string_lossy().to_string();
                    if WatchEngine::is_path_excluded(&path, &exclude) {
                        continue;
                    }
                    if let Ok(meta) = entry.metadata() {
                        if let Ok(modified) = meta.modified() {
                            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                                let mtime = duration.as_millis() as i64;
                                let prev = last_mtimes.get(&path).copied().unwrap_or(0);
                                if mtime > prev {
                                    if let Some(prev_emit) = last_emit.get(&path) {
                                        if now.duration_since(*prev_emit) < MIN_EVENT_INTERVAL {
                                            continue;
                                        }
                                    }
                                    last_mtimes.insert(path.clone(), mtime);
                                    last_emit.insert(path.clone(), now);
                                    changed.push(path);
                                }
                            }
                        }
                    }
                }

                for path in changed {
                    let _ = event_bus
                        .emit(
                            "iri://workspace",
                            "WORKSPACE_FILE_MODIFIED",
                            "iri://workspace_monitor",
                            &path,
                        )
                        .await;
                }
            }
        });

        handle.abort_handle()
    }
}

impl Drop for WatchEngine {
    fn drop(&mut self) {
        // The debouncer will stop on drop
        if let Some(handle) = self.polling_handle.take() {
            handle.abort();
        }
        debug!("WatchEngine: stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watch_config_defaults() {
        let config = WatchConfig::default();
        assert_eq!(config.debounce_ms, 500);
        assert_eq!(config.poll_interval_ms, 5000);
        assert!(config.watch_enabled);
    }

    #[test]
    fn test_start_nonexistent_dir() {
        let config = WatchConfig::default();
        let event_bus = Arc::new(EventBus::new(100));
        let result = WatchEngine::start("/nonexistent_path_xyz", config, event_bus, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_disabled_watch() {
        let config = WatchConfig {
            watch_enabled: false,
            ..Default::default()
        };
        let event_bus = Arc::new(EventBus::new(100));
        let result = WatchEngine::start("/tmp", config, event_bus, None);
        assert!(result.is_ok());
        let engine = result.unwrap();
        assert!(!engine.is_native());
        assert!(!engine.is_polling());
    }

    #[test]
    fn test_is_excluded_matches_directory() {
        let config = WatchConfig {
            exclude_patterns: vec!["node_modules/".into(), "data/".into(), ".git/".into()],
            ..Default::default()
        };
        assert!(config.is_excluded("/home/user/project/node_modules/pkg/index.js"));
        assert!(config.is_excluded("/project/data/rag_index/doc.json"));
        assert!(config.is_excluded("/project/.git/objects/abc123"));
        assert!(!config.is_excluded("/project/src/main.rs"));
        assert!(!config.is_excluded("/project/Cargo.toml"));
    }

    #[test]
    fn test_is_excluded_without_trailing_slash() {
        let config = WatchConfig {
            exclude_patterns: vec!["target".into(), "build".into()],
            ..Default::default()
        };
        assert!(config.is_excluded("/project/target/debug/app"));
        assert!(config.is_excluded("/project/build/output.o"));
        assert!(!config.is_excluded("/project/src/targeting.rs"));
    }

    #[test]
    fn test_load_gitignore_basic() {
        let dir = tempfile::TempDir::new().unwrap();
        let gitignore_path = dir.path().join(".gitignore");
        std::fs::write(&gitignore_path, "node_modules/\n.env\n*.log\n/build\n").unwrap();

        let mut config = WatchConfig::default();
        config.load_gitignore(dir.path());

        assert!(config
            .exclude_patterns
            .contains(&"node_modules/".to_string()));
        assert!(config.exclude_patterns.contains(&".env".to_string()));
        assert!(config.exclude_patterns.contains(&"*.log".to_string()));
        assert!(config.exclude_patterns.contains(&"build/".to_string()));
    }

    #[test]
    fn test_load_gitignore_ignores_comments_and_blanks() {
        let dir = tempfile::TempDir::new().unwrap();
        let gitignore_path = dir.path().join(".gitignore");
        std::fs::write(&gitignore_path, "# this is a comment\n\nnode_modules/\n").unwrap();

        let mut config = WatchConfig::default();
        config.load_gitignore(dir.path());

        assert!(config
            .exclude_patterns
            .contains(&"node_modules/".to_string()));
        assert_eq!(config.exclude_patterns.len(), 1);
    }

    #[test]
    fn test_load_gitignore_missing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = WatchConfig::default();
        config.load_gitignore(dir.path()); // should not panic
        assert!(config.exclude_patterns.is_empty());
    }

    #[test]
    fn test_within_cooldown_suppresses_rapid_repeats() {
        let mut last_emit = std::collections::HashMap::new();
        let t0 = std::time::Instant::now();
        let path = "/tmp/ws/log.txt".to_string();

        assert!(!WatchEngine::within_cooldown(&mut last_emit, &path, t0));
        last_emit.insert(path.clone(), t0);
        assert!(WatchEngine::within_cooldown(&mut last_emit, &path, t0));

        let later = t0 + std::time::Duration::from_millis(2000);
        assert!(!WatchEngine::within_cooldown(&mut last_emit, &path, later));
    }

    #[test]
    fn test_within_cooldown_is_per_path() {
        let mut last_emit = std::collections::HashMap::new();
        let t0 = std::time::Instant::now();
        last_emit.insert("/tmp/ws/a.txt".to_string(), t0);

        assert!(!WatchEngine::within_cooldown(
            &mut last_emit,
            "/tmp/ws/b.txt",
            t0
        ));
    }

    #[test]
    fn test_is_path_excluded_matches_nested_build_dirs() {
        let patterns = vec!["target/".to_string(), "node_modules/".to_string()];
        assert!(WatchEngine::is_path_excluded(
            "/repo/apps/gliding_code/target",
            &patterns
        ));
        assert!(WatchEngine::is_path_excluded(
            "/repo/target/debug/foo.rs",
            &patterns
        ));
        assert!(WatchEngine::is_path_excluded(
            "/repo/node_modules/pkg/index.js",
            &patterns
        ));
        assert!(!WatchEngine::is_path_excluded(
            "/repo/src/main.rs",
            &patterns
        ));
        assert!(!WatchEngine::is_path_excluded(
            "/repo/targeted.rs",
            &patterns
        ));
    }
}
