use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::tools::workspace_monitor::inventory::FileInventory;

/// A single file entry in a workspace snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotFileEntry {
    pub path: String,
    pub hash: String,
    pub size: u64,
}

/// A workspace snapshot — point-in-time view of all tracked files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSnapshot {
    /// Manifest format version.  Older manifests without this field deserialize
    /// as version 0 and are rejected for destructive restore.
    #[serde(default)]
    pub schema_version: u32,
    pub snapshot_id: String,
    pub created_at: i64,
    pub reason: String,
    pub task_iri: Option<String>,
    /// Canonical workspace root that was in scope when this manifest was made.
    #[serde(default)]
    pub workspace_root: String,
    pub files: Vec<SnapshotFileEntry>,
}

/// Result of a rollback operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackResult {
    pub snapshot_id: String,
    /// Snapshot made immediately before applying the rollback.  It is the
    /// operator's recovery point if a later verification rejects the restore.
    pub safety_snapshot_id: Option<String>,
    pub files_restored: usize,
    pub files_created: usize,
    pub files_deleted: usize,
    pub failed: Vec<String>,
}

/// Non-mutating preview of a rollback.  Callers can show this to a user or
/// policy layer before they opt in to deletion of files absent from a target
/// manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPlan {
    pub snapshot_id: String,
    pub files_to_restore: Vec<String>,
    pub files_to_create: Vec<String>,
    pub files_to_delete: Vec<String>,
}

/// Snapshot metadata table: key = "snapshot:{id}", value = serialized WorkspaceSnapshot.
const SNAPSHOTS: TableDefinition<&str, &[u8]> = TableDefinition::new("snapshots");

/// Manages workspace-level snapshots for rollback operations.
///
/// Snapshots store file path → content hash mappings in redb.
/// Actual file content is stored in the ContentStore's version store.
pub struct SnapshotManager {
    /// redb for snapshot metadata.
    db: Arc<Database>,
    /// Reference to the content store for retrieving file contents by hash.
    content_store: Arc<crate::tools::workspace_monitor::ContentStore>,
    /// Reference to the file inventory.
    inventory: Arc<RwLock<FileInventory>>,
    /// Absolute workspace boundary used to reject a tampered manifest path.
    workspace_root: PathBuf,
    /// Snapshot index: snapshot_id → WorkspaceSnapshot.
    index: RwLock<HashMap<String, WorkspaceSnapshot>>,
}

impl SnapshotManager {
    /// Create a new SnapshotManager.
    pub fn new(
        db: Arc<Database>,
        content_store: Arc<crate::tools::workspace_monitor::ContentStore>,
        inventory: Arc<RwLock<FileInventory>>,
        workspace_root: PathBuf,
    ) -> Self {
        let mut index = HashMap::new();

        // Pre-warm index from redb
        if let Ok(read_txn) = db.begin_read() {
            if let Ok(table) = read_txn.open_table(SNAPSHOTS) {
                if let Ok(iter) = table.iter() {
                    for result in iter {
                        if let Ok((key, value)) = result {
                            let key_str = key.value().to_string();
                            if key_str.starts_with("snapshot:") {
                                if let Ok(snapshot) =
                                    serde_json::from_slice::<WorkspaceSnapshot>(value.value())
                                {
                                    index.insert(snapshot.snapshot_id.clone(), snapshot);
                                }
                            }
                        }
                    }
                }
            }
        }

        Self {
            db,
            content_store,
            inventory,
            workspace_root: std::fs::canonicalize(&workspace_root).unwrap_or(workspace_root),
            index: RwLock::new(index),
        }
    }

    /// Create a snapshot of the current workspace state.
    ///
    /// Iterates over all tracked files in the inventory and records their
    /// current path + hash mappings.
    #[instrument(skip(self))]
    pub fn create_snapshot(&self, reason: &str, task_iri: Option<&str>) -> Result<String, String> {
        let snapshot_id = format!("ws_snap_{}", uuid::Uuid::new_v4().hyphenated());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let all_files = self.inventory.read().list_all();

        // A manifest must name recoverable content.  Inventory discovery is
        // intentionally metadata-only, so its cached hashes cannot be used as
        // a recovery contract here.
        let mut files = Vec::with_capacity(all_files.len());
        for entry in all_files {
            let path = self.resolve_snapshot_path(&entry.path)?;
            if !path.is_file() {
                // The watcher/inventory may lag a deletion. Do not preserve a
                // manifest entry whose content cannot be recovered.
                self.inventory.write().remove(&entry.path);
                continue;
            }
            let path_str = path.to_string_lossy().to_string();
            let (hash, size) = self.content_store.capture_snapshot_file(&path_str)?;
            files.push(SnapshotFileEntry {
                path: path_str,
                hash,
                size,
            });
        }
        files.sort_by(|left, right| left.path.cmp(&right.path));

        let file_count = files.len();

        let snapshot = WorkspaceSnapshot {
            schema_version: 1,
            snapshot_id: snapshot_id.clone(),
            created_at: now,
            reason: reason.to_string(),
            task_iri: task_iri.map(|s| s.to_string()),
            workspace_root: self.workspace_root.to_string_lossy().to_string(),
            files,
        };

        let key = format!("snapshot:{}", snapshot_id);
        let encoded = serde_json::to_vec(&snapshot)
            .map_err(|error| format!("Failed to serialize workspace snapshot: {error}"))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|error| format!("Failed to open snapshot transaction: {error}"))?;
        {
            let mut table = write_txn
                .open_table(SNAPSHOTS)
                .map_err(|error| format!("Failed to open snapshot table: {error}"))?;
            table
                .insert(key.as_str(), encoded.as_slice())
                .map_err(|error| format!("Failed to store snapshot manifest: {error}"))?;
        }
        write_txn
            .commit()
            .map_err(|error| format!("Failed to commit snapshot manifest: {error}"))?;

        self.index.write().insert(snapshot_id.clone(), snapshot);

        debug!(
            snapshot_id = %snapshot_id,
            reason = reason,
            file_count = file_count,
            "SnapshotManager: snapshot created"
        );

        Ok(snapshot_id)
    }

    /// Roll back the entire workspace to a given snapshot state.
    ///
    /// 1. Reads the snapshot record.
    /// 2. For each file in the snapshot, retrieves content from redb version store by hash.
    /// 3. Writes content back to disk.
    /// 4. Files existing on disk but not in the snapshot are optionally deleted.
    #[instrument(skip(self))]
    pub fn plan_rollback(&self, snapshot_id: &str) -> Result<RollbackPlan, String> {
        let snapshot = self
            .index
            .read()
            .get(snapshot_id)
            .cloned()
            .ok_or_else(|| format!("Snapshot not found: {}", snapshot_id))?;

        if snapshot.schema_version != 1 {
            return Err(format!(
                "Unsupported workspace snapshot schema {} for {}",
                snapshot.schema_version, snapshot_id
            ));
        }
        if !snapshot.workspace_root.is_empty()
            && Path::new(&snapshot.workspace_root) != self.workspace_root
        {
            return Err(format!(
                "Snapshot {} belongs to a different workspace root",
                snapshot_id
            ));
        }

        let mut target_paths = HashSet::with_capacity(snapshot.files.len());
        let mut files_to_restore = Vec::new();
        let mut files_to_create = Vec::new();
        for file_entry in &snapshot.files {
            let path = self.resolve_snapshot_path(&file_entry.path)?;
            let path_str = path.to_string_lossy().to_string();
            if self
                .content_store
                .get_snapshot_blob(&file_entry.hash)
                .is_none()
            {
                return Err(format!(
                    "Snapshot {} is missing content blob {} for {}",
                    snapshot_id, file_entry.hash, path_str
                ));
            }
            target_paths.insert(path_str.clone());
            if !path.exists() {
                files_to_create.push(path_str);
            } else if hash_file(&path)? != file_entry.hash {
                files_to_restore.push(path_str);
            }
        }

        let files_to_delete = self
            .inventory
            .read()
            .list_all()
            .into_iter()
            .filter_map(|entry| {
                self.resolve_snapshot_path(&entry.path)
                    .ok()
                    .and_then(|path| {
                        let text = path.to_string_lossy().to_string();
                        (path.exists() && !target_paths.contains(&text)).then_some(text)
                    })
            })
            .collect::<Vec<_>>();

        Ok(RollbackPlan {
            snapshot_id: snapshot_id.to_string(),
            files_to_restore,
            files_to_create,
            files_to_delete,
        })
    }

    /// Restore a snapshot without deleting files that were created later.
    /// Use [`rollback_to_with_options`] with `delete_extras=true` only after a
    /// caller has reviewed [`plan_rollback`].
    #[instrument(skip(self))]
    pub fn rollback_to(&self, snapshot_id: &str) -> Result<RollbackResult, String> {
        self.rollback_to_with_options(snapshot_id, false)
    }

    #[instrument(skip(self))]
    pub fn rollback_to_with_options(
        &self,
        snapshot_id: &str,
        delete_extras: bool,
    ) -> Result<RollbackResult, String> {
        let plan = self.plan_rollback(snapshot_id)?;
        let snapshot = self
            .index
            .read()
            .get(snapshot_id)
            .cloned()
            .ok_or_else(|| format!("Snapshot not found: {}", snapshot_id))?;

        // Capture a durable safety point before the first mutation.  A failed
        // restore can then be manually reversed without relying on partial
        // writes or the original manifest.
        let safety_snapshot_id =
            self.create_snapshot("pre_rollback", snapshot.task_iri.as_deref())?;

        let target_by_path = snapshot
            .files
            .iter()
            .map(|entry| (entry.path.clone(), entry))
            .collect::<HashMap<_, _>>();
        let mut blobs = HashMap::new();
        for path in plan
            .files_to_restore
            .iter()
            .chain(plan.files_to_create.iter())
        {
            let entry = target_by_path
                .get(path)
                .ok_or_else(|| format!("Snapshot entry missing for {path}"))?;
            let bytes = self
                .content_store
                .get_snapshot_blob(&entry.hash)
                .ok_or_else(|| format!("Snapshot content disappeared for {path}"))?;
            blobs.insert(path.clone(), bytes);
        }

        let mut restored = 0usize;
        let mut created = 0usize;
        let mut deleted = 0usize;
        let mut failed: Vec<String> = Vec::new();

        for path in &plan.files_to_restore {
            let result = blobs
                .get(path)
                .ok_or_else(|| format!("Prepared rollback blob missing for {path}"))
                .and_then(|bytes| atomic_write(Path::new(path), bytes));
            match result {
                Ok(()) => {
                    restored += 1;
                    self.inventory.write().add_or_update(path);
                }
                Err(error) => {
                    warn!(path = %path, %error, "Rollback: failed to restore file");
                    failed.push(path.clone());
                }
            }
        }
        for path in &plan.files_to_create {
            let result = blobs
                .get(path)
                .ok_or_else(|| format!("Prepared rollback blob missing for {path}"))
                .and_then(|bytes| atomic_write(Path::new(path), bytes));
            match result {
                Ok(()) => {
                    created += 1;
                    self.inventory.write().add_or_update(path);
                }
                Err(error) => {
                    warn!(path = %path, %error, "Rollback: failed to create file");
                    failed.push(path.clone());
                }
            }
        }
        if delete_extras {
            for path in &plan.files_to_delete {
                match std::fs::remove_file(path) {
                    Ok(()) => {
                        deleted += 1;
                        self.inventory.write().remove(path);
                    }
                    Err(error) => {
                        warn!(path = %path, %error, "Rollback: failed to delete extra file");
                        failed.push(path.clone());
                    }
                }
            }
        }

        debug!(
            snapshot_id = %snapshot_id,
            restored = restored,
            created = created,
            failed = failed.len(),
            "SnapshotManager: rollback completed"
        );

        Ok(RollbackResult {
            snapshot_id: snapshot_id.to_string(),
            safety_snapshot_id: Some(safety_snapshot_id),
            files_restored: restored,
            files_created: created,
            files_deleted: deleted,
            failed,
        })
    }

    /// Restore a single file to a specific version (by hash).
    pub fn restore_file(&self, path: &str, hash: &str) -> Result<(), String> {
        let path = self.resolve_snapshot_path(path)?;
        let content = self
            .content_store
            .get_snapshot_blob(hash)
            .ok_or_else(|| format!("Content not found for hash: {}", hash))?;

        atomic_write(&path, &content)
    }

    /// List available snapshots, newest first.
    pub fn list_snapshots(&self, limit: usize) -> Vec<WorkspaceSnapshot> {
        let mut snapshots: Vec<WorkspaceSnapshot> = self.index.read().values().cloned().collect();
        snapshots.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        snapshots.truncate(limit);
        snapshots
    }

    /// Get a specific snapshot by ID.
    pub fn get_snapshot(&self, snapshot_id: &str) -> Option<WorkspaceSnapshot> {
        self.index.read().get(snapshot_id).cloned()
    }

    /// Delete old snapshots, keeping only the most recent `keep` count.
    pub fn prune_snapshots(&self, keep: usize) -> usize {
        let mut snapshots = self.list_snapshots(usize::MAX);
        if snapshots.len() <= keep {
            return 0;
        }

        let to_remove = snapshots.split_off(keep);
        let count = to_remove.len();

        let mut index = self.index.write();
        for snap in to_remove {
            let key = format!("snapshot:{}", snap.snapshot_id);
            if let Ok(write_txn) = self.db.begin_write() {
                if let Ok(mut table) = write_txn.open_table(SNAPSHOTS) {
                    let _ = table.remove(key.as_str());
                }
                let _ = write_txn.commit();
            }
            index.remove(&snap.snapshot_id);
        }

        debug!(
            removed = count,
            kept = keep,
            "SnapshotManager: snapshots pruned"
        );
        count
    }

    // ── Private ──

    fn resolve_snapshot_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        let path = Path::new(raw_path);
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workspace_root.join(path)
        };
        if !candidate.starts_with(&self.workspace_root) {
            return Err(format!(
                "Snapshot path escapes workspace: {}",
                candidate.display()
            ));
        }
        Ok(candidate)
    }
}

fn hash_file(path: &Path) -> Result<String, String> {
    use sha2::Digest;

    let bytes = std::fs::read(path)
        .map_err(|error| format!("Failed to read workspace file {}: {error}", path.display()))?;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

fn atomic_write(path: &Path, content: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("Snapshot restore path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Failed to create restore directory {}: {error}",
            parent.display()
        )
    })?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid restore file name: {}", path.display()))?;
    let temp_path = parent.join(format!(
        ".{file_name}.gliding-restore-{}.tmp",
        uuid::Uuid::new_v4()
    ));
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&temp_path)
            .map_err(|error| format!("Failed to create restore temp file: {error}"))?;
        file.write_all(content)
            .map_err(|error| format!("Failed to write restore temp file: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("Failed to sync restore temp file: {error}"))?;
    }
    std::fs::rename(&temp_path, path)
        .map_err(|error| format!("Failed to commit restored file {}: {error}", path.display()))?;
    if let Ok(directory) = std::fs::File::open(parent) {
        let _ = directory.sync_all();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use redb::backends::InMemoryBackend;
    use redb::Builder;

    fn manager() -> (
        tempfile::TempDir,
        SnapshotManager,
        Arc<RwLock<crate::tools::workspace_monitor::FileInventory>>,
    ) {
        let workspace = tempfile::TempDir::new().unwrap();
        let db = Arc::new(
            Builder::new()
                .create_with_backend(InMemoryBackend::new())
                .unwrap(),
        );
        let content_store = Arc::new(crate::tools::workspace_monitor::ContentStore::new(
            100,
            65536,
            Some(
                Builder::new()
                    .create_with_backend(InMemoryBackend::new())
                    .unwrap(),
            ),
        ));
        let inventory = Arc::new(RwLock::new(
            crate::tools::workspace_monitor::FileInventory::new(None, None, vec![]),
        ));
        let snapshot_manager = SnapshotManager::new(
            db,
            content_store,
            inventory.clone(),
            workspace.path().to_path_buf(),
        );
        (workspace, snapshot_manager, inventory)
    }

    #[test]
    fn test_snapshot_lifecycle() {
        let (workspace, sm, inventory) = manager();
        let path = workspace.path().join("tracked.txt");
        std::fs::write(&path, "version one").unwrap();
        inventory.write().add_or_update(path.to_str().unwrap());

        let id = sm.create_snapshot("test", None).unwrap();
        assert!(!id.is_empty());

        let snapshots = sm.list_snapshots(10);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].reason, "test");

        let fetched = sm.get_snapshot(&id);
        assert_eq!(fetched.unwrap().files[0].hash.starts_with("sha256:"), true);
    }

    #[test]
    fn test_prune_snapshots() {
        let (_workspace, sm, _inventory) = manager();

        sm.create_snapshot("s1", None).unwrap();
        sm.create_snapshot("s2", None).unwrap();
        sm.create_snapshot("s3", None).unwrap();

        assert_eq!(sm.list_snapshots(10).len(), 3);

        let pruned = sm.prune_snapshots(2);
        assert_eq!(pruned, 1);
        assert_eq!(sm.list_snapshots(10).len(), 2);
    }

    #[test]
    fn test_rollback_nonexistent() {
        let (_workspace, sm, _inventory) = manager();
        let result = sm.rollback_to("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn rollback_restores_exact_snapshot_content_and_can_delete_extras() {
        let (workspace, sm, inventory) = manager();
        let tracked = workspace.path().join("tracked.txt");
        let extra = workspace.path().join("later.txt");
        std::fs::write(&tracked, "before").unwrap();
        inventory.write().add_or_update(tracked.to_str().unwrap());
        let snapshot_id = sm.create_snapshot("baseline", None).unwrap();

        std::fs::write(&tracked, "after").unwrap();
        std::fs::write(&extra, "extra").unwrap();
        inventory.write().add_or_update(tracked.to_str().unwrap());
        inventory.write().add_or_update(extra.to_str().unwrap());

        let plan = sm.plan_rollback(&snapshot_id).unwrap();
        assert_eq!(
            plan.files_to_restore,
            vec![tracked.to_string_lossy().to_string()]
        );
        assert_eq!(
            plan.files_to_delete,
            vec![extra.to_string_lossy().to_string()]
        );

        let result = sm.rollback_to_with_options(&snapshot_id, true).unwrap();
        assert!(result.safety_snapshot_id.is_some());
        assert_eq!(result.files_restored, 1);
        assert_eq!(result.files_deleted, 1);
        assert!(result.failed.is_empty());
        assert_eq!(std::fs::read_to_string(&tracked).unwrap(), "before");
        assert!(!extra.exists());
    }

    #[test]
    fn rollback_recreates_missing_file_from_manifest_blob() {
        let (workspace, sm, inventory) = manager();
        let tracked = workspace.path().join("tracked.txt");
        std::fs::write(&tracked, "before").unwrap();
        inventory.write().add_or_update(tracked.to_str().unwrap());
        let snapshot_id = sm.create_snapshot("baseline", None).unwrap();
        std::fs::remove_file(&tracked).unwrap();

        let plan = sm.plan_rollback(&snapshot_id).unwrap();
        assert_eq!(
            plan.files_to_create,
            vec![tracked.to_string_lossy().to_string()]
        );
        let result = sm.rollback_to(&snapshot_id).unwrap();
        assert_eq!(result.files_created, 1);
        assert_eq!(std::fs::read_to_string(&tracked).unwrap(), "before");
    }
}
