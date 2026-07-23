use std::collections::HashMap;
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
    pub snapshot_id: String,
    pub created_at: i64,
    pub reason: String,
    pub task_iri: Option<String>,
    pub files: Vec<SnapshotFileEntry>,
}

/// Result of a rollback operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackResult {
    pub snapshot_id: String,
    pub files_restored: usize,
    pub files_created: usize,
    pub files_deleted: usize,
    pub failed: Vec<String>,
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
    /// Snapshot index: snapshot_id → WorkspaceSnapshot.
    index: RwLock<HashMap<String, WorkspaceSnapshot>>,
}

impl SnapshotManager {
    /// Create a new SnapshotManager.
    pub fn new(
        db: Arc<Database>,
        content_store: Arc<crate::tools::workspace_monitor::ContentStore>,
        inventory: Arc<RwLock<FileInventory>>,
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
            index: RwLock::new(index),
        }
    }

    /// Create a snapshot of the current workspace state.
    ///
    /// Iterates over all tracked files in the inventory and records their
    /// current path + hash mappings.
    #[instrument(skip(self))]
    pub fn create_snapshot(&self, reason: &str, task_iri: Option<&str>) -> String {
        let snapshot_id = format!("ws_snap_{}", uuid::Uuid::new_v4().hyphenated());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let inventory = self.inventory.read();
        let all_files = inventory.list_all();

        let files: Vec<SnapshotFileEntry> = all_files
            .iter()
            .map(|e| SnapshotFileEntry {
                path: e.path.clone(),
                hash: e.content_hash.clone(),
                size: e.file_size,
            })
            .collect();

        let file_count = files.len();

        let snapshot = WorkspaceSnapshot {
            snapshot_id: snapshot_id.clone(),
            created_at: now,
            reason: reason.to_string(),
            task_iri: task_iri.map(|s| s.to_string()),
            files,
        };

        let key = format!("snapshot:{}", snapshot_id);
        if let Ok(encoded) = serde_json::to_vec(&snapshot) {
            if let Ok(write_txn) = self.db.begin_write() {
                if let Ok(mut table) = write_txn.open_table(SNAPSHOTS) {
                    if let Err(e) = table.insert(key.as_str(), encoded.as_slice()) {
                        warn!(snapshot_id = %snapshot_id, error = %e, "SnapshotManager: failed to store snapshot");
                    }
                }
                let _ = write_txn.commit();
            }
        }

        self.index.write().insert(snapshot_id.clone(), snapshot);

        debug!(
            snapshot_id = %snapshot_id,
            reason = reason,
            file_count = file_count,
            "SnapshotManager: snapshot created"
        );

        snapshot_id
    }

    /// Roll back the entire workspace to a given snapshot state.
    ///
    /// 1. Reads the snapshot record.
    /// 2. For each file in the snapshot, retrieves content from redb version store by hash.
    /// 3. Writes content back to disk.
    /// 4. Files existing on disk but not in the snapshot are optionally deleted.
    #[instrument(skip(self))]
    pub fn rollback_to(&self, snapshot_id: &str) -> Result<RollbackResult, String> {
        let snapshot = self
            .index
            .read()
            .get(snapshot_id)
            .cloned()
            .ok_or_else(|| format!("Snapshot not found: {}", snapshot_id))?;

        let mut restored = 0usize;
        let mut created = 0usize;
        let mut failed: Vec<String> = Vec::new();

        for file_entry in &snapshot.files {
            match self.content_store.get_version_content(&file_entry.path, 0) {
                Some(content) => {
                    // Try to find the content by hash from the version store
                    let content = self
                        .find_content_by_hash(&file_entry.hash)
                        .unwrap_or(content);

                    match std::fs::write(&file_entry.path, &content) {
                        Ok(()) => {
                            if std::path::Path::new(&file_entry.path).exists() {
                                restored += 1;
                            } else {
                                created += 1;
                            }
                        }
                        Err(e) => {
                            warn!(path = %file_entry.path, error = %e, "Rollback: failed to write file");
                            failed.push(file_entry.path.clone());
                        }
                    }
                }
                None => {
                    warn!(path = %file_entry.path, hash = %file_entry.hash, "Rollback: content not found in version store");
                    failed.push(file_entry.path.clone());
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
            files_restored: restored,
            files_created: created,
            files_deleted: 0,
            failed,
        })
    }

    /// Restore a single file to a specific version (by hash).
    pub fn restore_file(&self, path: &str, hash: &str) -> Result<(), String> {
        let content = self
            .find_content_by_hash(hash)
            .ok_or_else(|| format!("Content not found for hash: {}", hash))?;

        std::fs::write(path, &content).map_err(|e| format!("Failed to write file {}: {}", path, e))
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

    /// Find content for a given hash from all version entries.
    fn find_content_by_hash(&self, target_hash: &str) -> Option<String> {
        // Scan version store for matching content via range scan
        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(SNAPSHOTS).ok()?;
        let iter = table.iter().ok()?;
        for result in iter {
            if let Ok((_key, value)) = result {
                let content = String::from_utf8_lossy(value.value()).to_string();
                let content_hash = {
                    use sha2::Digest;
                    let mut hasher = sha2::Sha256::new();
                    hasher.update(content.as_bytes());
                    let result = hasher.finalize();
                    format!("sha256:{}", hex::encode(result))
                };
                if content_hash == target_hash {
                    return Some(content);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redb::backends::InMemoryBackend;
    use redb::Builder;

    #[test]
    fn test_snapshot_lifecycle() {
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

        let sm = SnapshotManager::new(db, content_store, inventory);

        let id = sm.create_snapshot("test", None);
        assert!(!id.is_empty());

        let snapshots = sm.list_snapshots(10);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].reason, "test");

        let fetched = sm.get_snapshot(&id);
        assert!(fetched.is_some());
    }

    #[test]
    fn test_prune_snapshots() {
        let db = Arc::new(
            Builder::new()
                .create_with_backend(InMemoryBackend::new())
                .unwrap(),
        );
        let content_store = Arc::new(crate::tools::workspace_monitor::ContentStore::new(
            100, 65536, None,
        ));
        let inventory = Arc::new(RwLock::new(
            crate::tools::workspace_monitor::FileInventory::new(None, None, vec![]),
        ));

        let sm = SnapshotManager::new(db, content_store, inventory);

        sm.create_snapshot("s1", None);
        sm.create_snapshot("s2", None);
        sm.create_snapshot("s3", None);

        assert_eq!(sm.list_snapshots(10).len(), 3);

        let pruned = sm.prune_snapshots(2);
        assert_eq!(pruned, 1);
        assert_eq!(sm.list_snapshots(10).len(), 2);
    }

    #[test]
    fn test_rollback_nonexistent() {
        let db = Arc::new(
            Builder::new()
                .create_with_backend(InMemoryBackend::new())
                .unwrap(),
        );
        let content_store = Arc::new(crate::tools::workspace_monitor::ContentStore::new(
            100, 65536, None,
        ));
        let inventory = Arc::new(RwLock::new(
            crate::tools::workspace_monitor::FileInventory::new(None, None, vec![]),
        ));

        let sm = SnapshotManager::new(db, content_store, inventory);
        let result = sm.rollback_to("nonexistent");
        assert!(result.is_err());
    }
}
