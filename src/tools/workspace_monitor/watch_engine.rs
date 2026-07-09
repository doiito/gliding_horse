use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};
use parking_lot::RwLock;
use tracing::{debug, error, warn};

use crate::core::event_bus::EventBus;
use crate::tools::workspace_monitor::inventory::FileInventory;

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
            let pat_dir = if pat.ends_with('/') {
                pat.clone()
            } else {
                format!("{}/", pat)
            };
            // Match as full path, directory prefix, path component, or parent dir:
            if normalized == pat
                || normalized.starts_with(&pat_dir)
                || normalized.contains(&format!("/{}", pat_dir))
                || normalized.ends_with(&format!("/{}", pat))
            {
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
    debouncer: Option<Debouncer<notify::RecommendedWatcher>>,
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
        let debouncer = match Self::try_start_native(root, &config, event_bus.clone(), inv_for_callback) {
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
        inventory: Option<Arc<RwLock<FileInventory>>>,
    ) -> Result<Debouncer<notify::RecommendedWatcher>, String> {
        let debounce = Duration::from_millis(config.debounce_ms);
        let eb = event_bus.clone();
        let root_owned = root.to_string();
        let exclude = config.exclude_patterns.clone();

        let mut debouncer = new_debouncer(
            debounce,
            move |result: DebounceEventResult| {
                if let Some(ref inv) = inventory {
                    if let Ok(events) = &result {
                        for event in events {
                            if let Some(path) = event.path.to_str() {
                                if !Self::is_path_excluded(path, &exclude) {
                                    inv.read().mark_stale(path);
                                }
                            }
                        }
                    }
                }
                if let Err(e) = Self::handle_debounced_events(result, &eb, &exclude) {
                    error!(error = %e, "WatchEngine: event handler error");
                }
            },
        )
        .map_err(|e| format!("Failed to create debouncer: {}", e))?;

        debouncer
            .watcher()
            .watch(Path::new(&root_owned), RecursiveMode::Recursive)
            .map_err(|e| format!("Failed to watch directory: {}", e))?;

        Ok(debouncer)
    }

    fn is_path_excluded(path: &str, exclude_patterns: &[String]) -> bool {
        let normalized = path.replace('\\', "/");
        for pattern in exclude_patterns {
            let pat = pattern.replace('\\', "/");
            let pat_dir = if pat.ends_with('/') {
                pat.clone()
            } else {
                format!("{}/", pat)
            };
            if normalized == pat
                || normalized.starts_with(&pat_dir)
                || normalized.contains(&format!("/{}", pat_dir))
                || normalized.ends_with(&format!("/{}", pat))
            {
                return true;
            }
        }
        false
    }

    fn handle_debounced_events(
        result: DebounceEventResult,
        event_bus: &EventBus,
        exclude_patterns: &[String],
    ) -> Result<(), String> {
        match result {
            Ok(events) => {
                for event in events {
                    let path_str = match event.path.to_str() {
                        Some(p) => p.to_string(),
                        None => continue,
                    };
                    if Self::is_path_excluded(&path_str, exclude_patterns) {
                        debug!(path = %path_str, "WatchEngine: excluded event");
                        continue;
                    }
                    let (event_type_str, path_str) = Self::map_event(&event);
                    let _ = event_bus.emit(
                        "iri://workspace",
                        &event_type_str,
                        "iri://workspace_monitor",
                        &path_str,
                    );
                    debug!(
                        event_type = %event_type_str,
                        path = %path_str,
                        "WatchEngine: event emitted"
                    );
                }
            }
            Err(e) => {
                warn!(error = %e, "WatchEngine: debouncer error");
            }
        }
        Ok(())
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

    fn start_polling(root: &str, config: &WatchConfig, event_bus: Arc<EventBus>) -> tokio::task::AbortHandle {
        let root = root.to_string();
        let interval = Duration::from_millis(config.poll_interval_ms);
        let exclude = config.exclude_patterns.clone();

        let handle = tokio::spawn(async move {
            let mut last_mtimes: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
            let mut interval_timer = tokio::time::interval(interval);
            interval_timer.tick().await;

            loop {
                interval_timer.tick().await;

                let mut changed = Vec::new();
                for entry in walkdir::WalkDir::new(&root).into_iter().filter_map(|e| e.ok()) {
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
                                    last_mtimes.insert(path.clone(), mtime);
                                    changed.push(path);
                                }
                            }
                        }
                    }
                }

                for path in changed {
                    let _ = event_bus.emit(
                        "iri://workspace",
                        "WORKSPACE_FILE_MODIFIED",
                        "iri://workspace_monitor",
                        &path,
                    );
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

        assert!(config.exclude_patterns.contains(&"node_modules/".to_string()));
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

        assert!(config.exclude_patterns.contains(&"node_modules/".to_string()));
        assert_eq!(config.exclude_patterns.len(), 1);
    }

    #[test]
    fn test_load_gitignore_missing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = WatchConfig::default();
        config.load_gitignore(dir.path()); // should not panic
        assert!(config.exclude_patterns.is_empty());
    }
}
