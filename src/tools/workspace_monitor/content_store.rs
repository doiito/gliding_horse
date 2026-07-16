use std::collections::HashMap;

use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use redb::{Database, ReadableDatabase, TableDefinition};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use crate::tools::workspace_monitor::DiffEngine;

/// Read mode for file reading operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    /// Full read (first time or cache missed).
    Full,
    /// Diff mode: return unified diff if file changed.
    Diff,
    /// Return only changed lines (with context) instead of the full file.
    /// Falls back to Diff if context can't be computed.
    ChangedOnly,
    /// Force re-read ignoring all caches.
    ForceRefresh,
}

/// Result of a file read operation.
#[derive(Debug, Clone)]
pub struct ReadResult {
    pub path: String,
    pub lines: Vec<String>,
    pub total_lines: usize,
    pub changed: bool,
    pub changed_ranges: Option<Vec<(usize, usize)>>,
    pub unified_diff: Option<String>,
    /// When ReadMode::ChangedOnly, only the changed lines (with 3 context lines).
    pub changed_lines: Option<Vec<String>>,
    pub from_cache: bool,
    pub version: u64,
}

/// Cached content entry in the LRU.
#[derive(Debug, Clone)]
struct CachedContent {
    lines: Vec<String>,
    hash: String,
    mtime: i64,
    version: u64,
}

/// Version store table: key = "version:{path}:v{version}", value = serialized content.
const VERSION_STORE: TableDefinition<&str, &[u8]> = TableDefinition::new("version_store");

/// Content cache with LRU eviction, SHA-256 change detection, and redb version store.
pub struct ContentStore {
    /// In-memory LRU: file path → cached content (mtime-keyed).
    lines_cache: Mutex<LruCache<String, CachedContent>>,
    /// Path → current version number.
    version_index: RwLock<HashMap<String, u64>>,
    /// redb database for storing historical versions (path → versioned content).
    version_store: Option<Database>,
    /// Maximum cache size in bytes (approximate).
    #[allow(dead_code)]
    max_cache_bytes: usize,
}

impl ContentStore {
    /// Create a new ContentStore.
    ///
    /// * `cache_capacity` - Number of files to hold in LRU cache.
    /// * `max_cache_bytes` - Approximate maximum cache memory usage.
    /// * `db` - Optional redb database for historical version storage.
    pub fn new(
        cache_capacity: usize,
        max_cache_bytes: usize,
        db: Option<Database>,
    ) -> Self {
        Self {
            lines_cache: Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(cache_capacity.max(1)).unwrap(),
            )),
            version_index: RwLock::new(HashMap::new()),
            version_store: db,
            max_cache_bytes,
        }
    }

    /// Read a file from disk, applying caching and optional diff.
    ///
    /// Returns a `ReadResult` with content, version info, and optional diff.
    pub fn read_file(&self, path: &str, mode: ReadMode) -> std::io::Result<ReadResult> {
        let disk_content = std::fs::read_to_string(path)?;
        Ok(self.process_content(path, &disk_content, mode))
    }

    /// Process already-read content through the caching and diff pipeline.
    ///
    /// Same logic as `read_file()` but accepts pre-read content to avoid redundant disk I/O.
    /// Callers (tool executor) can read disk once and pass the content here for caching + diff.
    pub fn process_content(&self, path: &str, content: &str, mode: ReadMode) -> ReadResult {
        let disk_lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        let total_lines = disk_lines.len();
        let disk_mtime = std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let disk_hash = hash_content(content);

        let mut version_index = self.version_index.write();
        let current_version = version_index.get(path).copied().unwrap_or(0);

        if mode == ReadMode::ForceRefresh {
            let new_version = current_version + 1;
            version_index.insert(path.to_string(), new_version);
            self.store_version(path, content, new_version);
            drop(version_index);
            self.invalidate_cache(path);

            return ReadResult {
                path: path.to_string(),
                lines: disk_lines,
                total_lines,
                changed: true,
                changed_ranges: None,
                unified_diff: None,
                changed_lines: None,
                from_cache: false,
                version: new_version,
            };
        }

        let cached = {
            let mut cache = self.lines_cache.lock();
            cache.get(path).cloned()
        };

        if let Some(cached) = cached {
            if disk_mtime == cached.mtime && !mode.is_diff() {
                return ReadResult {
                    path: path.to_string(),
                    lines: cached.lines,
                    total_lines,
                    changed: false,
                    changed_ranges: None,
                    unified_diff: None,
                    changed_lines: None,
                    from_cache: true,
                    version: cached.version,
                };
            }

            if disk_hash == cached.hash {
                let mut cache = self.lines_cache.lock();
                cache.put(path.to_string(), CachedContent {
                    lines: cached.lines.clone(),
                    hash: cached.hash.clone(),
                    mtime: disk_mtime,
                    version: cached.version,
                });
                return ReadResult {
                    path: path.to_string(),
                    lines: cached.lines,
                    total_lines,
                    changed: false,
                    changed_ranges: None,
                    unified_diff: None,
                    changed_lines: None,
                    from_cache: true,
                    version: cached.version,
                };
            }

            let new_version = current_version + 1;
            version_index.insert(path.to_string(), new_version);
            self.store_version(path, content, new_version);
            drop(version_index);

            let is_diff_or_changed = mode == ReadMode::Diff || mode == ReadMode::ChangedOnly;
            let (changed_ranges, unified_diff, changed_lines) = if is_diff_or_changed {
                let ranges = DiffEngine::changed_ranges(&cached.lines, &disk_lines);
                let diff = DiffEngine::unified_diff(
                    &cached.lines,
                    &disk_lines,
                    path,
                    cached.version,
                    new_version,
                );
                let changed_lines = if mode == ReadMode::ChangedOnly {
                    Some(Self::extract_changed_lines(&disk_lines, &ranges))
                } else {
                    None
                };
                (Some(ranges), Some(diff), changed_lines)
            } else {
                (None, None, None)
            };

            let mut cache = self.lines_cache.lock();
            cache.put(path.to_string(), CachedContent {
                lines: disk_lines.clone(),
                hash: disk_hash,
                mtime: disk_mtime,
                version: new_version,
            });

            return ReadResult {
                path: path.to_string(),
                lines: disk_lines,
                total_lines,
                changed: true,
                changed_ranges,
                unified_diff,
                changed_lines,
                from_cache: false,
                version: new_version,
            };
        }

        let new_version = current_version + 1;
        version_index.insert(path.to_string(), new_version);
        self.store_version(path, content, new_version);
        drop(version_index);

        let mut cache = self.lines_cache.lock();
        cache.put(path.to_string(), CachedContent {
            lines: disk_lines.clone(),
            hash: disk_hash,
            mtime: disk_mtime,
            version: new_version,
        });

        ReadResult {
            path: path.to_string(),
            lines: disk_lines,
            total_lines,
            changed: true,
            changed_ranges: None,
            unified_diff: None,
            changed_lines: None,
            from_cache: false,
            version: new_version,
        }
    }

    /// Extract only the changed/inserted lines (with 3 lines of context)
    /// from a list of change ranges. Used for ReadMode::ChangedOnly.
    fn extract_changed_lines(
        all_lines: &[String],
        ranges: &[(usize, usize)],
    ) -> Vec<String> {
        if ranges.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut last_end: usize = 0;
        for &(start, end) in ranges {
            // Context lines before change
            let ctx_start = if start >= 3 { start - 3 } else { 0 };
            if ctx_start > last_end {
                result.push("... (snip) ...".to_string());
            }
            for i in ctx_start..start {
                if let Some(line) = all_lines.get(i) {
                    result.push(format!(" {}", line));
                }
            }
            // Changed lines
            for i in start..end.min(all_lines.len()) {
                if let Some(line) = all_lines.get(i) {
                    result.push(format!("+{}", line));
                }
            }
            last_end = end;
        }
        result
    }

    /// Invalidate a specific file from the cache.
    pub fn invalidate(&self, path: &str) {
        let mut cache = self.lines_cache.lock();
        cache.pop(path);
        debug!(path = %path, "ContentStore: cache invalidated");
    }

    /// Try to get cached content for a file (without disk read).
    pub fn try_get_cached(&self, path: &str) -> Option<Vec<String>> {
        let mut cache = self.lines_cache.lock();
        cache.get(path).map(|c| c.lines.clone())
    }

    /// Get the current version number for a file.
    pub fn get_version(&self, path: &str) -> u64 {
        self.version_index.read().get(path).copied().unwrap_or(0)
    }

    /// Get the content hash for a file if cached.
    pub fn get_hash(&self, path: &str) -> Option<String> {
        let mut cache = self.lines_cache.lock();
        cache.get(path).map(|c| c.hash.clone())
    }

    /// Invalidate all cached content.
    pub fn clear(&self) {
        let mut cache = self.lines_cache.lock();
        cache.clear();
        let mut vi = self.version_index.write();
        vi.clear();
        debug!("ContentStore: all caches cleared");
    }

    /// Retrieve a specific version of file content from redb.
    pub fn get_version_content(&self, path: &str, version: u64) -> Option<String> {
        let db = self.version_store.as_ref()?;
        let key = format!("version:{}:v{}", path, version);
        let read_txn = db.begin_read().ok()?;
        let table = read_txn.open_table(VERSION_STORE).ok()?;
        let guard = table.get(key.as_str()).ok()??;
        Some(String::from_utf8_lossy(guard.value()).to_string())
    }

    // ── Private helpers ──

    fn invalidate_cache(&self, path: &str) {
        let mut cache = self.lines_cache.lock();
        cache.pop(path);
    }

    fn store_version(&self, path: &str, content: &str, version: u64) {
        if let Some(ref db) = self.version_store {
            let key = format!("version:{}:v{}", path, version);
            match db.begin_write() {
                Ok(write_txn) => {
                    if let Ok(mut table) = write_txn.open_table(VERSION_STORE) {
                        if let Err(e) = table.insert(key.as_str(), content.as_bytes()) {
                            warn!(
                                path = %path, version = version, error = %e,
                                "ContentStore: failed to store version in redb"
                            );
                        }
                    }
                    let _ = write_txn.commit();
                }
                Err(e) => warn!(
                    path = %path, version = version, error = %e,
                    "ContentStore: failed to begin redb write transaction"
                ),
            }
        }
    }
}

/// Compute SHA-256 hash of content.
fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    format!("sha256:{}", hex::encode(result))
}

impl ReadMode {
    fn is_diff(&self) -> bool {
        matches!(self, ReadMode::Diff | ReadMode::ChangedOnly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_file(dir: &TempDir, name: &str, content: &str) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, content).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn test_basic_read() {
        let dir = TempDir::new().unwrap();
        let path = create_test_file(&dir, "test.txt", "hello\nworld\n");

        let store = ContentStore::new(100, 65536, None);
        let result = store.read_file(&path, ReadMode::Full).unwrap();

        assert_eq!(result.total_lines, 2);
        assert!(!result.from_cache);
        assert_eq!(result.version, 1);
    }

    #[test]
    fn test_cache_hit() {
        let dir = TempDir::new().unwrap();
        let path = create_test_file(&dir, "test.txt", "hello\nworld\n");

        let store = ContentStore::new(100, 65536, None);
        let r1 = store.read_file(&path, ReadMode::Full).unwrap();
        assert!(!r1.from_cache);

        let r2 = store.read_file(&path, ReadMode::Full).unwrap();
        assert!(r2.from_cache);
        assert_eq!(r2.version, r1.version);
    }

    #[test]
    fn test_change_detection() {
        let dir = TempDir::new().unwrap();
        let path = create_test_file(&dir, "test.txt", "hello\nworld\n");

        let store = ContentStore::new(100, 65536, None);
        let r1 = store.read_file(&path, ReadMode::Full).unwrap();
        assert_eq!(r1.version, 1);

        // Modify the file
        create_test_file(&dir, "test.txt", "hello\nmodified\nworld\n");

        let r2 = store.read_file(&path, ReadMode::Diff).unwrap();
        assert_eq!(r2.version, 2);
        assert!(r2.changed);
        assert!(r2.changed_ranges.is_some());
        assert!(r2.unified_diff.is_some());
        assert!(!r2.from_cache);
    }

    #[test]
    fn test_invalidate() {
        let dir = TempDir::new().unwrap();
        let path = create_test_file(&dir, "test.txt", "content");

        let store = ContentStore::new(100, 65536, None);
        let r1 = store.read_file(&path, ReadMode::Full).unwrap();
        assert!(!r1.from_cache);

        store.invalidate(&path);
        assert!(store.try_get_cached(&path).is_none());
    }

    #[test]
    fn test_hash_content() {
        let h1 = hash_content("hello");
        let h2 = hash_content("hello");
        let h3 = hash_content("world");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert!(h1.starts_with("sha256:"));
    }

    #[test]
    fn test_force_refresh() {
        let dir = TempDir::new().unwrap();
        let path = create_test_file(&dir, "test.txt", "content");

        let store = ContentStore::new(100, 65536, None);
        let r1 = store.read_file(&path, ReadMode::Full).unwrap();
        let r2 = store.read_file(&path, ReadMode::ForceRefresh).unwrap();

        assert!(r2.changed);
        assert!(r2.version > r1.version);
        assert!(!r2.from_cache);
    }

    #[test]
    fn test_diff_mode_returns_changed_ranges() {
        let dir = TempDir::new().unwrap();
        let path = create_test_file(&dir, "test.txt", "line1\nline2\nline3\nline4\nline5");

        let store = ContentStore::new(100, 65536, None);
        let r1 = store.read_file(&path, ReadMode::Full).unwrap();
        assert_eq!(r1.version, 1);

        // Modify middle lines
        create_test_file(&dir, "test.txt", "line1\nline2_modified\nline3\nline4_modified\nline5");

        let r2 = store.read_file(&path, ReadMode::Diff).unwrap();
        assert_eq!(r2.version, 2);
        assert!(r2.changed);
        assert!(r2.changed_ranges.is_some());
        assert!(r2.unified_diff.is_some());
        assert!(r2.unified_diff.as_ref().unwrap().contains("line2_modified"));
        assert!(r2.changed_lines.is_none()); // ChangedOnly mode only
    }

    #[test]
    fn test_changed_only_mode() {
        let dir = TempDir::new().unwrap();
        let path = create_test_file(&dir, "test.txt", "keep1\nkeep2\nchange_this\nkeep3\nkeep4");

        let store = ContentStore::new(100, 65536, None);
        let _ = store.read_file(&path, ReadMode::Full).unwrap();

        // Modify one line
        create_test_file(&dir, "test.txt", "keep1\nkeep2\nCHANGED\nkeep3\nkeep4");

        let r2 = store.read_file(&path, ReadMode::ChangedOnly).unwrap();
        assert!(r2.changed);
        assert!(r2.changed_lines.is_some());
        let changed = r2.changed_lines.unwrap();
        // Should contain the changed line (with + prefix) and some context
        let all_text = changed.join(" ");
        assert!(all_text.contains("CHANGED"), "ChangedOnly mode should include the modified line");
    }

    #[test]
    fn test_unchanged_file_cache_hit() {
        let dir = TempDir::new().unwrap();
        let path = create_test_file(&dir, "test.txt", "stable content");

        let store = ContentStore::new(100, 65536, None);
        let r1 = store.read_file(&path, ReadMode::Full).unwrap();
        assert_eq!(r1.version, 1);

        // Read again with no changes — should be cache hit
        let r2 = store.read_file(&path, ReadMode::Full).unwrap();
        assert!(r2.from_cache);
        assert!(!r2.changed);
        assert_eq!(r2.version, r1.version);
    }

    #[test]
    fn test_extract_changed_lines_basic() {
        let lines: Vec<String> = (1..=20).map(|i| format!("line_{}", i)).collect();
        let ranges = vec![(4, 6), (14, 16)];

        let extracted = ContentStore::extract_changed_lines(&lines, &ranges);

        // Should have context before changes and the changes themselves
        let all = extracted.join("\n");
        assert!(all.contains("+line_5"), "Should contain changed line 5");
        assert!(all.contains("+line_15"), "Should contain changed line 15");
        assert!(all.contains("..."), "Should contain snip markers between changes");
    }
}
