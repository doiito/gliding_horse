use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};
use walkdir::WalkDir;

use crate::memory::l2_blackboard::Blackboard;

/// File state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileState {
    /// File exists on disk but not yet tracked.
    Undiscovered,
    /// Discovered via full_scan().
    Discovered,
    /// File has been read and is up-to-date.
    ReadFresh,
    /// File was read but has been modified externally.
    ReadStale,
    /// File was written by the Agent but not yet re-read.
    WrittenUnread,
}

impl FileState {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileState::Undiscovered => "undiscovered",
            FileState::Discovered => "discovered",
            FileState::ReadFresh => "read_fresh",
            FileState::ReadStale => "read_stale",
            FileState::WrittenUnread => "written_unread",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "undiscovered" => FileState::Undiscovered,
            "discovered" => FileState::Discovered,
            "read_fresh" => FileState::ReadFresh,
            "read_stale" => FileState::ReadStale,
            "written_unread" => FileState::WrittenUnread,
            _ => FileState::Undiscovered,
        }
    }
}

/// A single file entry in the inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub file_size: u64,
    pub file_ext: String,
    pub language: String,
    pub mtime: i64,
    pub content_hash: String,
    pub state: FileState,
    pub last_read_at: Option<i64>,
    pub last_read_version: u64,
    pub current_version: u64,
    pub read_count: u64,
}

impl FileEntry {
    /// The IRI used for this file in L2 Named Graph.
    pub fn iri(&self) -> String {
        format!("iri://workspace/file/{}", self.path)
    }

    /// The parent directory IRI.
    pub fn parent_dir_iri(&self) -> String {
        let parent = Path::new(&self.path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        format!("iri://workspace/dir/{}/", parent)
    }
}

/// Classification of a language from a file extension.
fn classify_language(ext: &str) -> &'static str {
    match ext {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" => "python",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "hpp" | "cc" | "cxx" => "cpp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "scala" => "scala",
        "toml" => "toml",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "md" | "mdx" => "markdown",
        "html" | "htm" => "html",
        "css" | "scss" | "less" => "css",
        _ => "unknown",
    }
}

/// Shared workspace RDF constant.
pub const WORKSPACE_GRAPH: &str = "iri://workspace";

/// Inventory cache table: key = file path (str), value = serialized FileEntry.
const INVENTORY_CACHE: TableDefinition<&str, &[u8]> = TableDefinition::new("inventory_cache");

/// Maximum number of entries in the in-memory inventory.
/// Prevents unbounded memory growth in large workspaces.
/// In tests, uses a small value so limit enforcement is testable without 50k files.
#[cfg(not(test))]
pub(crate) const MAX_INVENTORY_ENTRIES: usize = 50_000;
#[cfg(test)]
pub(crate) const MAX_INVENTORY_ENTRIES: usize = 10;

/// FileInventory — thin facade over L2 Blackboard (RDF) with redb hot cache.
///
/// The authority data source is L2 (Oxigraph RDF named graph `iri://workspace`).
/// redb serves as a hot cache for fast metadata lookups.
pub struct FileInventory {
    /// L2 Blackboard (RDF triple store) for authority data.
    blackboard: Option<Arc<Blackboard>>,
    /// redb hot cache: path → serialized FileEntry.
    cache: Option<Database>,
    /// In-memory cache for fastest access (no redb deserialization).
    mem_cache: RwLock<HashMap<String, FileEntry>>,
    /// Exclude patterns for scanning. Uses RwLock for post-construction updates (gitignore sync).
    exclude_patterns: RwLock<Vec<String>>,
}

impl FileInventory {
    /// Create a new FileInventory.
    ///
    /// * `blackboard` - Optional L2 Blackboard for RDF storage.
    /// * `db` - Optional redb database for hot cache.
    /// * `exclude_patterns` - Glob patterns to exclude from scanning (e.g., "node_modules/").
    pub fn new(
        blackboard: Option<Arc<Blackboard>>,
        db: Option<Database>,
        exclude_patterns: Vec<String>,
    ) -> Self {
        let mut mem_cache = HashMap::new();

        // Pre-warm from redb if available
        if let Some(ref database) = db {
            if let Ok(read_txn) = database.begin_read() {
                if let Ok(table) = read_txn.open_table(INVENTORY_CACHE) {
                    if let Ok(iter) = table.iter() {
                        for result in iter {
                            if let Ok((key, value)) = result {
                                let path = key.value().to_string();
                                if let Ok(entry) =
                                    serde_json::from_slice::<FileEntry>(value.value())
                                {
                                    mem_cache.insert(path, entry);
                                }
                            }
                        }
                    }
                }
            }
        }

        Self {
            blackboard,
            cache: db,
            mem_cache: RwLock::new(mem_cache),
            exclude_patterns: RwLock::new(exclude_patterns),
        }
    }

    /// Perform a full directory scan of `root`, discovering all files.
    ///
    /// Returns the number of discovered files.
    #[instrument(skip(self))]
    pub fn full_scan(&self, root: &str) -> usize {
        let mut count = 0;

        // Check if inventory is already full before scanning
        {
            let mem = self.mem_cache.read();
            if mem.len() >= MAX_INVENTORY_ENTRIES {
                enforce_max_entries(mem.len());
                return 0;
            }
        }

        for entry in WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| !self.is_excluded(e.path()))
        {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path().to_string_lossy().to_string();
            // Only add if not already tracked
            if self.get_entry(&path).is_none() {
                self.add_entry(&path);
                count += 1;
            }
        }

        debug!(root = %root, discovered = count, "FileInventory: full scan completed");
        count
    }

    /// Add or update a single file entry by scanning the file on disk.
    pub fn add_or_update(&self, path: &str) -> Option<FileEntry> {
        let path_obj = Path::new(path);
        if !path_obj.is_file() {
            // File doesn't exist — treat as removal
            self.remove_internal(path);
            return None;
        }

        // Enforce maximum entries limit — only reject genuinely NEW entries
        if self.get_entry(path).is_none() {
            let mem = self.mem_cache.read();
            if mem.len() >= MAX_INVENTORY_ENTRIES {
                enforce_max_entries(mem.len());
                return None;
            }
        }

        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => return None,
        };

        let file_size = metadata.len();
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let content = std::fs::read_to_string(path).unwrap_or_default();
        let content_hash = hash_content(&content);

        let ext = path_obj
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_string();

        let language = classify_language(&ext).to_string();
        let state = FileState::Discovered;
        let version = 0;

        let entry = FileEntry {
            path: path.to_string(),
            file_size,
            file_ext: ext,
            language,
            mtime,
            content_hash,
            state,
            last_read_at: None,
            last_read_version: 0,
            current_version: version,
            read_count: 0,
        };

        self.store_entry(&entry);
        self.sync_to_l2(&entry);

        debug!(path = %path, "FileInventory: file added/updated");
        Some(entry)
    }

    /// Mark a file as stale (externally modified).
    pub fn mark_stale(&self, path: &str) {
        let mut mem = self.mem_cache.write();
        if let Some(entry) = mem.get_mut(path) {
            entry.state = FileState::ReadStale;
            // Update content hash & mtime from disk
            if let Ok(content) = std::fs::read_to_string(path) {
                entry.content_hash = hash_content(&content);
            }
            if let Ok(meta) = std::fs::metadata(path) {
                if let Ok(t) = meta.modified() {
                    if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                        entry.mtime = d.as_millis() as i64;
                    }
                }
                entry.file_size = meta.len();
            }
            entry.current_version += 1;
            let cloned = entry.clone();
            drop(mem);
            self.persist_to_cache(&cloned);
            self.sync_to_l2(&cloned);
            debug!(path = %path, version = cloned.current_version, "FileInventory: marked stale");
        }
    }

    /// Mark a file as read (fresh).
    pub fn mark_read(&self, path: &str, version: u64) {
        let mut mem = self.mem_cache.write();
        if let Some(entry) = mem.get_mut(path) {
            entry.state = FileState::ReadFresh;
            entry.last_read_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64,
            );
            entry.last_read_version = version;
            entry.read_count += 1;
            let cloned = entry.clone();
            drop(mem);
            self.persist_to_cache(&cloned);
            self.sync_to_l2(&cloned);
        }
    }

    /// Mark a file as written (but not re-read) by the agent.
    pub fn mark_written(&self, path: &str) {
        let mut mem = self.mem_cache.write();
        if let Some(entry) = mem.get_mut(path) {
            entry.state = FileState::WrittenUnread;
            entry.current_version += 1;
            let cloned = entry.clone();
            drop(mem);
            self.persist_to_cache(&cloned);
            self.sync_to_l2(&cloned);
            debug!(path = %path, "FileInventory: marked written_unread");
        } else {
            // New file written by agent
            drop(mem);
            self.add_or_update(path);
        }
    }

    /// Mark a file as externally read (e.g., via read_full_result micro-tool).
    /// Lightweight: no disk I/O, just updates in-memory state so subsequent file_read calls
    /// recognize the file as already-read and return cached/diff response instead of full content.
    pub fn mark_external_read(&self, path: &str) {
        let mut mem = self.mem_cache.write();
        if let Some(entry) = mem.get_mut(path) {
            entry.state = FileState::ReadFresh;
            entry.read_count += 1;
            entry.last_read_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64,
            );
            let cloned = entry.clone();
            drop(mem);
            self.persist_to_cache(&cloned);
            self.sync_to_l2(&cloned);
        } else {
            // Entry doesn't exist yet — add a minimal placeholder
            drop(mem);
            let minimal = FileEntry {
                path: path.to_string(),
                file_size: 0,
                file_ext: Path::new(path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_string(),
                language: "unknown".to_string(),
                mtime: 0,
                content_hash: String::new(),
                state: FileState::ReadFresh,
                last_read_at: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as i64,
                ),
                last_read_version: 0,
                current_version: 0,
                read_count: 1,
            };
            self.store_entry(&minimal);
        }
    }

    /// Remove a file from the inventory (e.g., on deletion).
    pub fn remove(&self, path: &str) -> bool {
        self.remove_internal(path);
        self.remove_from_l2(path);
        debug!(path = %path, "FileInventory: file removed");
        true
    }

    /// Clear all tracked files from the inventory.
    pub fn clear_all(&self) {
        let mut mem = self.mem_cache.write();
        mem.clear();
    }

    /// Get a file entry by path.
    pub fn get_entry(&self, path: &str) -> Option<FileEntry> {
        let mem = self.mem_cache.read();
        mem.get(path).cloned()
    }

    /// Update exclude patterns after construction (e.g., after WatchEngine loads .gitignore).
    pub fn set_exclude_patterns(&self, patterns: Vec<String>) {
        let mut ep = self.exclude_patterns.write();
        for p in patterns {
            if !ep.contains(&p) {
                ep.push(p);
            }
        }
    }

    /// List all files matching a state filter.
    pub fn list_by_state(&self, state: FileState) -> Vec<FileEntry> {
        let mem = self.mem_cache.read();
        mem.values().filter(|e| e.state == state).cloned().collect()
    }

    /// List all tracked files.
    pub fn list_all(&self) -> Vec<FileEntry> {
        let mem = self.mem_cache.read();
        mem.values().cloned().collect()
    }

    /// List files under a directory prefix.
    pub fn list_dir(&self, dir_prefix: &str) -> Vec<FileEntry> {
        let prefix = if dir_prefix.ends_with('/') {
            dir_prefix.to_string()
        } else {
            format!("{}/", dir_prefix)
        };
        let mem = self.mem_cache.read();
        mem.values()
            .filter(|e| e.path.starts_with(&prefix) || e.path.starts_with(dir_prefix))
            .cloned()
            .collect()
    }

    /// Count files by state.
    pub fn state_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        let mem = self.mem_cache.read();
        for entry in mem.values() {
            *counts.entry(entry.state.as_str().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Total number of tracked files.
    pub fn total_count(&self) -> usize {
        self.mem_cache.read().len()
    }

    /// Get stale files (for prompting re-read).
    pub fn stale_files(&self) -> Vec<FileEntry> {
        self.list_by_state(FileState::ReadStale)
    }

    /// Get files with state ReadStale or WrittenUnread.
    pub fn unread_files(&self) -> Vec<FileEntry> {
        let mut result = self.list_by_state(FileState::ReadStale);
        result.extend(self.list_by_state(FileState::WrittenUnread));
        result
    }

    // ── Private helpers ──

    pub(crate) fn is_excluded(&self, path: &std::path::Path) -> bool {
        let ep = self.exclude_patterns.read();
        if ep.is_empty() {
            return false;
        }
        let path_str = path.to_string_lossy();
        let normalized = path_str.replace('\\', "/");
        for pattern in ep.iter() {
            let pat = pattern.replace('\\', "/");
            if match_glob_pattern(&normalized, &pat) {
                return true;
            }
        }
        false
    }

    fn add_entry(&self, path: &str) -> Option<FileEntry> {
        // Enforce maximum entries limit
        {
            let mem = self.mem_cache.read();
            if mem.len() >= MAX_INVENTORY_ENTRIES {
                enforce_max_entries(mem.len());
                return None;
            }
        }

        let path_obj = Path::new(path);
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let content_hash = hash_content(&content);
        let metadata = std::fs::metadata(path).ok()?;

        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let ext = path_obj
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_string();

        let language = classify_language(&ext).to_string();

        let entry = FileEntry {
            path: path.to_string(),
            file_size: metadata.len(),
            file_ext: ext,
            language,
            mtime,
            content_hash,
            state: FileState::Discovered,
            last_read_at: None,
            last_read_version: 0,
            current_version: 0,
            read_count: 0,
        };

        self.store_entry(&entry);
        self.sync_to_l2(&entry);
        Some(entry)
    }

    fn store_entry(&self, entry: &FileEntry) {
        // In-memory cache
        {
            let mut mem = self.mem_cache.write();
            mem.insert(entry.path.clone(), entry.clone());
        }

        // redb persistence
        if let Some(ref db) = self.cache {
            if let Ok(encoded) = serde_json::to_vec(entry) {
                if let Ok(write_txn) = db.begin_write() {
                    if let Ok(mut table) = write_txn.open_table(INVENTORY_CACHE) {
                        if let Err(e) = table.insert(entry.path.as_str(), encoded.as_slice()) {
                            warn!(path = %entry.path, error = %e, "FileInventory: redb insert failed");
                        }
                    }
                    let _ = write_txn.commit();
                }
            }
        }
    }

    fn remove_internal(&self, path: &str) {
        {
            let mut mem = self.mem_cache.write();
            mem.remove(path);
        }
        if let Some(ref db) = self.cache {
            if let Ok(write_txn) = db.begin_write() {
                if let Ok(mut table) = write_txn.open_table(INVENTORY_CACHE) {
                    let _ = table.remove(path);
                }
                let _ = write_txn.commit();
            }
        }
    }

    fn persist_to_cache(&self, entry: &FileEntry) {
        if let Some(ref db) = self.cache {
            if let Ok(encoded) = serde_json::to_vec(entry) {
                if let Ok(write_txn) = db.begin_write() {
                    if let Ok(mut table) = write_txn.open_table(INVENTORY_CACHE) {
                        if let Err(e) = table.insert(entry.path.as_str(), encoded.as_slice()) {
                            warn!(path = %entry.path, error = %e, "FileInventory: redb persist failed");
                        }
                    }
                    let _ = write_txn.commit();
                }
            }
        }
    }

    fn sync_to_l2(&self, entry: &FileEntry) {
        let blackboard = match self.blackboard.as_ref() {
            Some(b) => b,
            None => return,
        };

        let iri = entry.iri();
        let parent_dir_iri = entry.parent_dir_iri();

        let json_ld = serde_json::json!({
            "@id": &iri,
            "@type": ["ws:File"],
            "ws:filePath": entry.path,
            "ws:fileSize": entry.file_size,
            "ws:fileExt": entry.file_ext,
            "ws:language": entry.language,
            "ws:mtime": entry.mtime,
            "ws:contentHash": entry.content_hash,
            "ws:state": entry.state.as_str(),
            "ws:lastReadAt": entry.last_read_at.unwrap_or(0),
            "ws:lastReadVersion": entry.last_read_version,
            "ws:currentVersion": entry.current_version,
            "ws:readCount": entry.read_count,
            "ws:parentDir": parent_dir_iri,
        });

        let config = crate::CoreConfig {
            max_node_size: 65536,
            ..crate::CoreConfig::default()
        };

        if let Err(e) =
            blackboard.write_node_to_graph(&iri, &json_ld.to_string(), WORKSPACE_GRAPH, &config)
        {
            warn!(path = %entry.path, error = %e, "FileInventory: L2 sync failed");
        }
    }

    fn remove_from_l2(&self, path: &str) {
        let blackboard = match self.blackboard.as_ref() {
            Some(b) => b,
            None => return,
        };

        let iri = format!("iri://workspace/file/{}", path);
        let _ = blackboard.delete_node(&iri);
    }
}

fn hash_content(content: &str) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    format!("sha256:{}", hex::encode(result))
}

/// Match a path against a gitignore-style glob pattern.
///
/// Supports:
/// - `*.ext` (extension glob)
/// - `name/` (directory prefix)
/// - `path/name` (exact path segment)
/// - Substring match as fallback (backward compatibility)
pub fn match_glob_pattern(path: &str, pattern: &str) -> bool {
    // Exact match
    if path == pattern {
        return true;
    }

    // Directory prefix: pattern like "node_modules/" or "target/"
    let pat_dir = if pattern.ends_with('/') {
        pattern.to_string()
    } else if !pattern.contains('.') {
        format!("{}/", pattern)
    } else {
        // Check if it's a glob extension pattern like *.log
        if let Some(ext) = pattern.strip_prefix("*.") {
            if ext.contains('*') || ext.contains('/') {
                // Complex pattern — fall back to substring match
                return path.contains(pattern) || path.ends_with(pattern);
            }
            // Match file extension
            let ext_dot = format!(".{}", ext);
            return path.ends_with(&ext_dot) || path.contains(&format!("/{}", ext_dot));
        }
        return path.contains(pattern) || path.ends_with(pattern);
    };

    // Directory match: the path starts with the dir or contains /<dir> or ends with /<dir>
    path.starts_with(&pat_dir)
        || path.contains(&format!("/{}", pat_dir))
        || path.ends_with(&format!("/{}", pat_dir.trim_end_matches('/')))
}

fn enforce_max_entries(count: usize) {
    if count >= MAX_INVENTORY_ENTRIES {
        warn!(
            "FileInventory has reached {} entries (max {}). Further files will not be tracked.",
            count, MAX_INVENTORY_ENTRIES
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_workspace(dir: &TempDir, files: &[(&str, &str)]) {
        for (name, content) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, content).unwrap();
        }
    }

    #[test]
    fn test_full_scan() {
        let dir = TempDir::new().unwrap();
        create_workspace(
            &dir,
            &[
                ("src/main.rs", "fn main() {}"),
                ("src/lib.rs", "pub fn hello() {}"),
                ("README.md", "# Hello"),
            ],
        );

        let inventory = FileInventory::new(None, None, vec![]);
        let count = inventory.full_scan(&dir.path().to_string_lossy());
        assert_eq!(count, 3);
    }

    #[test]
    fn test_exclude_pattern() {
        let dir = TempDir::new().unwrap();
        create_workspace(
            &dir,
            &[
                ("src/main.rs", "fn main() {}"),
                ("node_modules/pkg/index.js", "module.exports = {};"),
                ("target/debug/app", "binary"),
            ],
        );

        let inventory =
            FileInventory::new(None, None, vec!["node_modules/".into(), "target/".into()]);
        let count = inventory.full_scan(&dir.path().to_string_lossy());
        assert_eq!(count, 1);
    }

    #[test]
    fn test_add_and_get() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        std::fs::write(&path, "fn test() {}").unwrap();

        let inventory = FileInventory::new(None, None, vec![]);
        let entry = inventory.add_or_update(&path.to_string_lossy()).unwrap();

        assert_eq!(entry.file_ext, "rs");
        assert_eq!(entry.language, "rust");
        assert_eq!(entry.state, FileState::Discovered);

        let fetched = inventory.get_entry(&path.to_string_lossy()).unwrap();
        assert_eq!(fetched.path, entry.path);
    }

    #[test]
    fn test_mark_stale_and_read() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        std::fs::write(&path, "v1").unwrap();

        let inventory = FileInventory::new(None, None, vec![]);
        inventory.add_or_update(&path.to_string_lossy());

        inventory.mark_read(&path.to_string_lossy(), 0);
        assert_eq!(
            inventory.get_entry(&path.to_string_lossy()).unwrap().state,
            FileState::ReadFresh
        );

        inventory.mark_stale(&path.to_string_lossy());
        assert_eq!(
            inventory.get_entry(&path.to_string_lossy()).unwrap().state,
            FileState::ReadStale
        );
    }

    #[test]
    fn test_list_by_state() {
        let dir = TempDir::new().unwrap();
        let p1 = dir.path().join("a.rs");
        let p2 = dir.path().join("b.rs");
        std::fs::write(&p1, "a").unwrap();
        std::fs::write(&p2, "b").unwrap();

        let inventory = FileInventory::new(None, None, vec![]);
        inventory.add_or_update(&p1.to_string_lossy());
        inventory.add_or_update(&p2.to_string_lossy());
        inventory.mark_read(&p1.to_string_lossy(), 0);

        assert_eq!(inventory.list_by_state(FileState::ReadFresh).len(), 1);
        assert_eq!(inventory.list_by_state(FileState::Discovered).len(), 1);
    }

    #[test]
    fn test_state_counts() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("f.rs");
        std::fs::write(&p, "fn main() {}").unwrap();

        let inventory = FileInventory::new(None, None, vec![]);
        inventory.add_or_update(&p.to_string_lossy());
        inventory.mark_read(&p.to_string_lossy(), 0);

        let counts = inventory.state_counts();
        assert_eq!(*counts.get("read_fresh").unwrap_or(&0), 1);
    }

    // ── match_glob_pattern tests ──

    #[test]
    fn test_glob_exact_match() {
        assert!(match_glob_pattern(
            "node_modules/pkg/index.js",
            "node_modules/"
        ));
        assert!(match_glob_pattern("target/debug/app", "target/"));
        assert!(match_glob_pattern("src/main.rs", "src/main.rs"));
    }

    #[test]
    fn test_glob_extension_wildcard() {
        // *.log should match .log files
        assert!(match_glob_pattern("server.log", "*.log"));
        assert!(match_glob_pattern("logs/app.log", "*.log"));
        assert!(match_glob_pattern("src/error.log", "*.log"));
        // Should NOT match files without .log ending
        assert!(!match_glob_pattern("server.logger", "*.log"));
        assert!(!match_glob_pattern("log.txt", "*.log"));
        assert!(!match_glob_pattern("logs", "*.log"));
    }

    #[test]
    fn test_glob_extension_wildcard_other() {
        // *.rs should match .rs files
        assert!(match_glob_pattern("src/main.rs", "*.rs"));
        assert!(match_glob_pattern("lib.rs", "*.rs"));
        assert!(!match_glob_pattern("main.rs.bak", "*.rs"));
    }

    #[test]
    fn test_glob_directory_prefix() {
        // Directory patterns - with trailing slash
        assert!(match_glob_pattern(
            "node_modules/pkg/index.js",
            "node_modules/"
        ));
        assert!(match_glob_pattern(
            "project/node_modules/pkg.js",
            "node_modules/"
        ));

        // Without trailing slash, no dot → treated as directory
        assert!(match_glob_pattern(
            "node_modules/pkg/index.js",
            "node_modules"
        ));
        assert!(match_glob_pattern("build/output.o", "build"));

        // Should NOT match partial directory names
        assert!(!match_glob_pattern(
            "src/node_modules_test/helper.js",
            "node_modules/"
        ));
    }

    #[test]
    fn test_glob_path_specific() {
        // Exact path segments in the middle
        assert!(match_glob_pattern(
            "/home/user/project/data/rag/doc.json",
            "data/"
        ));
        assert!(!match_glob_pattern(
            "/home/user/project/database/schema.sql",
            "data/"
        ));
    }

    #[test]
    fn test_glob_gitignore_patterns() {
        // Common .gitignore patterns
        let patterns = vec![
            ".env",            // dotfile
            "*.pyc",           // compiled python
            "__pycache__/",    // cache dir
            ".next/",          // build dir
            "dist/",           // output dir
            ".gliding_horse/", // app data
        ];

        for pat in &patterns {
            assert!(
                match_glob_pattern(&format!("/workspace/{}", pat.trim_end_matches('/')), pat),
                "Pattern '{}' should match itself",
                pat
            );
        }
    }

    #[test]
    fn test_glob_exclude_all_variants() {
        let inventory = FileInventory::new(
            None,
            None,
            vec!["node_modules/".into(), "*.pyc".into(), "build/".into()],
        );

        // Must exclude
        assert!(inventory.is_excluded(Path::new("/project/node_modules/pkg/index.js")));
        assert!(inventory.is_excluded(Path::new("/project/src/__pycache__/cache.pyc")));
        assert!(inventory.is_excluded(Path::new("/project/build/o.app")));

        // Must NOT exclude
        assert!(!inventory.is_excluded(Path::new("/project/src/main.rs")));
        assert!(!inventory.is_excluded(Path::new("/project/Cargo.toml")));
        assert!(!inventory.is_excluded(Path::new("/project/src/pycache/api.py")));
    }

    #[test]
    fn test_glob_no_false_positive_substring() {
        // Ensure "target" doesn't match "targeting.rs"
        let inventory = FileInventory::new(None, None, vec!["target/".into()]);
        assert!(!inventory.is_excluded(Path::new("/project/src/targeting.rs")));
        assert!(inventory.is_excluded(Path::new("/project/target/debug/app")));
    }

    #[test]
    fn test_set_exclude_patterns() {
        let inventory = FileInventory::new(None, None, vec!["node_modules/".into()]);
        assert!(inventory.is_excluded(Path::new("node_modules/pkg/index.js")));
        assert!(!inventory.is_excluded(Path::new("build/output.o")));

        // Add patterns after construction (simulates gitignore sync)
        inventory.set_exclude_patterns(vec!["build/".into(), "*.o".into()]);
        assert!(inventory.is_excluded(Path::new("build/output.o")));
        // Original patterns preserved
        assert!(inventory.is_excluded(Path::new("node_modules/pkg/index.js")));
    }

    #[test]
    fn test_max_entries_limit() {
        let dir = TempDir::new().unwrap();
        // Create files that would exceed limit
        // We use a fresh inventory and lower the effective limit by
        // directly testing add_entry behavior
        let inventory = FileInventory::new(None, None, vec![]);

        // Fill up to MAX_INVENTORY_ENTRIES
        for i in 0..super::MAX_INVENTORY_ENTRIES {
            let path = dir.path().join(format!("file_{}.rs", i));
            std::fs::write(&path, "fn f() {}").unwrap();
            let result = inventory.add_or_update(&path.to_string_lossy());
            assert!(result.is_some(), "Entry {} should be added", i);
        }

        // Next entry should be rejected
        let overflow = dir.path().join("overflow.rs");
        std::fs::write(&overflow, "fn overflow() {}").unwrap();
        let result = inventory.add_or_update(&overflow.to_string_lossy());
        assert!(
            result.is_none(),
            "Entry beyond MAX_INVENTORY_ENTRIES should be rejected"
        );

        assert_eq!(inventory.total_count(), super::MAX_INVENTORY_ENTRIES);
    }

    #[test]
    fn test_full_scan_honors_exclude_patterns() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("vendor")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("vendor/lib.rs"), "pub fn lib() {}").unwrap();

        let inventory = FileInventory::new(None, None, vec!["vendor/".into()]);
        let count = inventory.full_scan(&dir.path().to_string_lossy());
        assert_eq!(count, 1, "full_scan should exclude vendor/");
        assert!(inventory
            .get_entry(&dir.path().join("src/main.rs").to_string_lossy())
            .is_some());
        assert!(inventory
            .get_entry(&dir.path().join("vendor/lib.rs").to_string_lossy())
            .is_none());
    }
}
