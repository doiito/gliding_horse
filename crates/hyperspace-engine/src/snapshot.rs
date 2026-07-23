//! Snapshot-based crash recovery using bincode serialization.
//!
//! Provides:
//! - `save_snapshot()` — atomic write via tmp + rename
//! - `load_snapshot()` — deserialize engine state
//!
//! The snapshot captures:
//! - HNSW nodes (vectors + neighbor lists + levels)
//! - Entry point and max layer
//! - Logical clock
//! - IRI registry (id ↔ iri mappings)
//! - Metadata forward index (id → JSON-LD payload)
//!
//! # Crash Safety
//!
//! Writer: write to `.tmp` → `fsync` → rename → `fsync` directory.
//! Reader: only reads the final file (rename is atomic on POSIX).

use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::EngineError;
use crate::hnsw::{HnswConfig, SerializableNode};

/// On-disk snapshot capturing all engine state needed for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSnapshot {
    /// Serialized HNSW nodes.
    pub nodes: Vec<Option<SerializableNode>>,

    /// Current entry point (u32::MAX if empty).
    pub entry_point: u32,

    /// Logical clock value at snapshot time.
    pub clock: u64,

    /// IRI registry: id → IRI string.
    pub iri_registry: Vec<(u32, String)>,

    /// Forward metadata: id → JSON-LD payload (serialized as JSON string).
    pub forward_meta: Vec<(u32, String)>,

    /// Deleted IDs bitmap (serialized as vec).
    pub deleted_ids: Vec<u32>,

    /// Engine dimension.
    pub dimension: usize,

    /// Engine configuration.
    pub config: HnswConfig,
}

/// Save engine state to a snapshot file atomically.
///
/// 1. Serialize to bincode buffer
/// 2. Write to `{path}.tmp`
/// 3. fsync the tmp file
/// 4. Rename tmp → final (atomic on POSIX)
pub fn save_snapshot(path: &Path, snapshot: &EngineSnapshot) -> Result<(), EngineError> {
    let tmp_path = path.with_extension("snapshot.tmp");

    let bytes = bincode::serialize(snapshot).map_err(|e| EngineError::StorageError {
        message: format!("Snapshot serialization: {e}"),
    })?;

    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }

    // Atomic rename (POSIX guarantee)
    fs::rename(&tmp_path, path)?;

    // Sync the directory to persist the rename metadata
    if let Some(parent) = path.parent() {
        let dir_f = File::open(parent)?;
        dir_f.sync_all().ok();
    }

    info!(
        "Snapshot saved: {} bytes -> {}",
        bytes.len(),
        path.display()
    );

    Ok(())
}

/// Load engine state from a snapshot file.
pub fn load_snapshot(path: &Path) -> Result<EngineSnapshot, EngineError> {
    if !path.exists() {
        return Err(EngineError::NotFound(format!(
            "Snapshot file not found: {}",
            path.display()
        )));
    }

    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(64 * 1024, file);

    let snapshot: EngineSnapshot =
        bincode::deserialize_from(&mut reader).map_err(|e| EngineError::StorageError {
            message: format!("Snapshot deserialization: {e}"),
        })?;

    info!(
        "Snapshot loaded: {} bytes, {} nodes, clock={}",
        file_len,
        snapshot.nodes.len(),
        snapshot.clock
    );

    Ok(snapshot)
}

/// Check if a snapshot file exists.
pub fn snapshot_exists(path: &Path) -> bool {
    path.exists()
}

/// Delete a snapshot file.
pub fn delete_snapshot(path: &Path) -> Result<(), EngineError> {
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_roundtrip() {
        let snap = EngineSnapshot {
            nodes: vec![
                None,
                Some(SerializableNode {
                    coords: vec![0.1, 0.2, 0.3, 0.4],
                    metric_tag: 0,
                    alpha: 0.0,
                    neighbors0: vec![0, 2],
                    neighbors_upper: vec![vec![2]],
                    level: 1,
                }),
                Some(SerializableNode {
                    coords: vec![0.5, 0.6, 0.7, 0.8],
                    metric_tag: 0,
                    alpha: 0.0,
                    neighbors0: vec![1],
                    neighbors_upper: vec![],
                    level: 0,
                }),
            ],
            entry_point: 1,
            clock: 42,
            iri_registry: vec![(1, "onto:test".into())],
            forward_meta: vec![],
            deleted_ids: vec![],
            dimension: 4,
            config: HnswConfig::default(),
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.snapshot");

        save_snapshot(&path, &snap).unwrap();
        assert!(path.exists());

        let loaded = load_snapshot(&path).unwrap();
        assert_eq!(loaded.clock, 42);
        assert_eq!(loaded.entry_point, 1);
        assert_eq!(loaded.nodes.len(), 3);
        assert!(loaded.nodes[0].is_none());
        assert_eq!(
            loaded.nodes[1].as_ref().unwrap().coords,
            vec![0.1, 0.2, 0.3, 0.4]
        );
        assert_eq!(loaded.dimension, 4);
    }

    #[test]
    fn test_snapshot_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_such.snapshot");
        let result = load_snapshot(&path);
        assert!(result.is_err());
    }
}
