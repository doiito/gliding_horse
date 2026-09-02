//! Snapshot-based crash recovery using versioned, checksummed bincode payloads.
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
//! Reader: verifies a SHA-256 digest before deserializing. Checkpoints keep a
//! current and previous generation so a torn/corrupt current file can recover
//! from the last known-good generation and its retained WAL segment.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::error::EngineError;
use crate::hnsw::{HnswConfig, SerializableNode};
use crate::hyper_vector::MetricKind;

const SNAPSHOT_MAGIC: [u8; 8] = *b"GHSNPV02";
pub const SNAPSHOT_FORMAT_VERSION: u32 = 2;
const SNAPSHOT_HEADER_LEN: usize = 8 + 4 + 4 + 8 + 32;

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

/// Snapshot plus envelope data required to validate it against the engine
/// opening it. Only the current envelope format is accepted.
#[derive(Debug, Clone)]
pub struct LoadedSnapshot {
    pub snapshot: EngineSnapshot,
    pub format_version: u32,
    pub metric_kind: MetricKind,
    pub source_path: PathBuf,
}

/// Save engine state to a snapshot file atomically.
///
/// 1. Serialize to bincode buffer
/// 2. Write to `{path}.tmp`
/// 3. fsync the tmp file
/// 4. Rename tmp → final (atomic on POSIX)
pub fn save_snapshot(path: &Path, snapshot: &EngineSnapshot) -> Result<(), EngineError> {
    let metric_kind = snapshot
        .nodes
        .iter()
        .flatten()
        .find_map(|node| metric_from_tag(node.metric_tag).ok())
        .unwrap_or(MetricKind::Cosine);
    save_snapshot_with_metric(path, snapshot, metric_kind)
}

/// Save a snapshot with the engine's configured metric in the envelope.
pub fn save_snapshot_with_metric(
    path: &Path,
    snapshot: &EngineSnapshot,
    metric_kind: MetricKind,
) -> Result<(), EngineError> {
    let payload = bincode::serialize(snapshot).map_err(|error| EngineError::StorageError {
        message: format!("Snapshot serialization: {error}"),
    })?;
    let digest = Sha256::digest(&payload);
    let mut bytes = Vec::with_capacity(SNAPSHOT_HEADER_LEN + payload.len());
    bytes.extend_from_slice(&SNAPSHOT_MAGIC);
    bytes.extend_from_slice(&SNAPSHOT_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&metric_kind.tag().to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&digest);
    bytes.extend_from_slice(&payload);
    write_snapshot_atomically(path, &bytes)?;

    info!(
        "Snapshot saved: {} bytes -> {} (format={}, metric={:?})",
        bytes.len(),
        path.display(),
        SNAPSHOT_FORMAT_VERSION,
        metric_kind
    );
    Ok(())
}

/// Save a new checkpoint generation while retaining the prior one at
/// `{path}.previous`. The temporary generation is complete and fsynced before
/// any existing current file is moved, so every crash point retains at least
/// one valid generation.
pub fn save_snapshot_generational(
    path: &Path,
    snapshot: &EngineSnapshot,
    metric_kind: MetricKind,
) -> Result<(), EngineError> {
    let staged_path = snapshot_staged_path(path);
    let previous_path = snapshot_previous_path(path);
    save_snapshot_with_metric(&staged_path, snapshot, metric_kind)?;

    if path.exists() {
        fs::rename(path, &previous_path)?;
        sync_parent(path)?;
    }
    fs::rename(&staged_path, path)?;
    sync_parent(path)?;
    Ok(())
}

/// Load engine state from a snapshot file.
pub fn load_snapshot(path: &Path) -> Result<EngineSnapshot, EngineError> {
    Ok(load_snapshot_with_metadata(path)?.snapshot)
}

/// Load one specified snapshot generation. Pre-envelope files are considered
/// incompatible and are intentionally rejected; callers may delete them and
/// continue from the previous generation or WAL.
pub fn load_snapshot_with_metadata(path: &Path) -> Result<LoadedSnapshot, EngineError> {
    if !path.exists() {
        return Err(EngineError::NotFound(format!(
            "Snapshot file not found: {}",
            path.display()
        )));
    }

    let bytes = fs::read(path)?;
    let file_len = bytes.len();
    if !bytes.starts_with(&SNAPSHOT_MAGIC) {
        return Err(EngineError::StorageError {
            message: "Unsupported legacy snapshot format".to_string(),
        });
    }
    let (snapshot, format_version, metric_kind) = decode_current_snapshot(&bytes)?;

    info!(
        "Snapshot loaded: {} bytes, {} nodes, clock={}",
        file_len,
        snapshot.nodes.len(),
        snapshot.clock
    );

    Ok(LoadedSnapshot {
        snapshot,
        format_version,
        metric_kind,
        source_path: path.to_path_buf(),
    })
}

/// Load the current snapshot, falling back to the previous generation only if
/// the current one is absent or fails integrity/format validation.
pub fn load_snapshot_with_fallback(path: &Path) -> Result<LoadedSnapshot, EngineError> {
    match load_snapshot_with_metadata(path) {
        Ok(snapshot) => Ok(snapshot),
        Err(current_error) => {
            let previous = snapshot_previous_path(path);
            match load_snapshot_with_metadata(&previous) {
                Ok(snapshot) => {
                    info!(
                        current = %path.display(),
                        previous = %previous.display(),
                        error = %current_error,
                        "Current snapshot unavailable; recovered from previous generation"
                    );
                    Ok(snapshot)
                }
                Err(previous_error) => Err(EngineError::StorageError {
                    message: format!(
                        "Unable to load current snapshot {} ({current_error}) or previous generation {} ({previous_error})",
                        path.display(),
                        previous.display()
                    ),
                }),
            }
        }
    }
}

pub fn snapshot_previous_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.previous", path.display()))
}

fn snapshot_staged_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.next", path.display()))
}

fn write_snapshot_atomically(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    let tmp_path = PathBuf::from(format!("{}.tmp", path.display()));
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&tmp_path, path)?;
    sync_parent(path)
}

fn sync_parent(path: &Path) -> Result<(), EngineError> {
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn decode_current_snapshot(bytes: &[u8]) -> Result<(EngineSnapshot, u32, MetricKind), EngineError> {
    if bytes.len() < SNAPSHOT_HEADER_LEN {
        return Err(EngineError::StorageError {
            message: format!(
                "Snapshot envelope is truncated: {} bytes, expected at least {}",
                bytes.len(),
                SNAPSHOT_HEADER_LEN
            ),
        });
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != SNAPSHOT_FORMAT_VERSION {
        return Err(EngineError::StorageError {
            message: format!(
                "Unsupported snapshot format version {version}; supported version is {SNAPSHOT_FORMAT_VERSION}"
            ),
        });
    }
    let metric_tag = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let metric_kind = metric_from_tag(metric_tag)?;
    let payload_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    if payload_len != bytes.len().saturating_sub(SNAPSHOT_HEADER_LEN) {
        return Err(EngineError::StorageError {
            message: format!(
                "Snapshot payload length mismatch: header={payload_len}, actual={}",
                bytes.len().saturating_sub(SNAPSHOT_HEADER_LEN)
            ),
        });
    }
    let expected_digest = &bytes[24..56];
    let payload = &bytes[SNAPSHOT_HEADER_LEN..];
    let actual_digest = Sha256::digest(payload);
    if actual_digest.as_slice() != expected_digest {
        return Err(EngineError::StorageError {
            message: "Snapshot checksum mismatch".to_string(),
        });
    }
    let snapshot = bincode::deserialize(payload).map_err(|error| EngineError::StorageError {
        message: format!("Snapshot deserialization: {error}"),
    })?;
    Ok((snapshot, version, metric_kind))
}

fn metric_from_tag(tag: u32) -> Result<MetricKind, EngineError> {
    MetricKind::from_tag(tag).map_err(|error| EngineError::StorageError {
        message: format!("Snapshot has invalid metric tag {tag}: {error}"),
    })
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

    #[test]
    fn checksum_rejects_a_tampered_snapshot() {
        let snap = EngineSnapshot {
            nodes: vec![],
            entry_point: u32::MAX,
            clock: 7,
            iri_registry: vec![],
            forward_meta: vec![],
            deleted_ids: vec![],
            dimension: 4,
            config: HnswConfig::default(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.snapshot");
        save_snapshot_with_metric(&path, &snap, MetricKind::Cosine).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0x01;
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            load_snapshot_with_metadata(&path),
            Err(EngineError::StorageError { .. })
        ));
    }

    #[test]
    fn legacy_bincode_snapshot_is_rejected_as_incompatible() {
        let snap = EngineSnapshot {
            nodes: vec![],
            entry_point: u32::MAX,
            clock: 9,
            iri_registry: vec![(1, "legacy:one".into())],
            forward_meta: vec![],
            deleted_ids: vec![],
            dimension: 4,
            config: HnswConfig::default(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.snapshot");
        fs::write(&path, bincode::serialize(&snap).unwrap()).unwrap();
        assert!(matches!(
            load_snapshot_with_metadata(&path),
            Err(EngineError::StorageError { .. })
        ));
    }

    #[test]
    fn corrupted_current_generation_falls_back_to_previous() {
        let mut first = EngineSnapshot {
            nodes: vec![],
            entry_point: u32::MAX,
            clock: 1,
            iri_registry: vec![],
            forward_meta: vec![],
            deleted_ids: vec![],
            dimension: 4,
            config: HnswConfig::default(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.snapshot");
        save_snapshot_generational(&path, &first, MetricKind::Cosine).unwrap();
        first.clock = 2;
        save_snapshot_generational(&path, &first, MetricKind::Cosine).unwrap();
        fs::write(&path, b"corrupt current generation").unwrap();

        let loaded = load_snapshot_with_fallback(&path).unwrap();
        assert_eq!(loaded.snapshot.clock, 1);
        assert_eq!(loaded.source_path, snapshot_previous_path(&path));
    }
}
