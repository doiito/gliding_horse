//! Vector storage with optional file-backed persistence.
//!
//! Current implementation uses `Vec<Option<Vec<u8>>>` (simple growable slab).
//! Supports `persist()` and `load()` for file-backed durability.
//!
//! # Future
//!
//! Can be swapped for mmap-based store (hyperspace-db's VectorStore with
//! 64K element segments + ArcSwap) when performance demands it.

use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use bincode;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::EngineError;

/// On-disk format for a persisted VectorStore.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistentStore {
    slots: Vec<Option<Vec<u8>>>,
    next_id: u32,
    element_size: usize,
}

/// A growable slot-based vector store.
///
/// Each slot holds the serialized bytes of an EmbeddingVector.
/// Slots are never physically removed (only tombstoned) to keep IDs stable.
///
/// Supports `persist()` / `load()` for file-backed durability between
/// engine sessions (complementary to WAL replay).
pub struct VectorStore {
    slots: Vec<Option<Vec<u8>>>,
    next_id: AtomicU32,
    element_size: usize,
    /// Optional path for persist/load.
    store_path: Option<PathBuf>,
}

impl VectorStore {
    pub fn new(base_path: &Path, element_size: usize) -> Self {
        let store_path = base_path.join("vector_store.bin");
        Self {
            slots: Vec::new(),
            next_id: AtomicU32::new(0),
            element_size,
            store_path: Some(store_path),
        }
    }

    /// Create in-memory only (no persistence path).
    pub fn new_in_memory(element_size: usize) -> Self {
        Self {
            slots: Vec::new(),
            next_id: AtomicU32::new(0),
            element_size,
            store_path: None,
        }
    }

    /// Append vector bytes, returning the assigned ID.
    /// On success, the ID is guaranteed unique for this store lifetime.
    pub fn append(&mut self, bytes: &[u8]) -> Result<u32, EngineError> {
        if bytes.len() != self.element_size {
            return Err(EngineError::InvalidVector(format!(
                "Vector size mismatch: expected {} got {}",
                self.element_size,
                bytes.len()
            )));
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        if id as usize >= self.slots.len() {
            self.slots.push(Some(bytes.to_vec()));
        } else {
            self.slots[id as usize] = Some(bytes.to_vec());
        }
        Ok(id)
    }

    /// Set a vector at a specific ID slot (for external ID management, e.g. WAL replay).
    pub fn set(&mut self, id: u32, bytes: &[u8]) -> Result<(), EngineError> {
        if bytes.len() != self.element_size {
            return Err(EngineError::InvalidVector(format!(
                "Vector size mismatch: expected {} got {}",
                self.element_size,
                bytes.len()
            )));
        }
        let id_u = id as usize;
        if id_u >= self.slots.len() {
            self.slots.resize(id_u + 1, None);
        }
        self.slots[id_u] = Some(bytes.to_vec());
        let current_next = self.next_id.load(Ordering::Relaxed);
        if id >= current_next {
            self.next_id.store(id + 1, Ordering::Release);
        }
        Ok(())
    }

    /// Get vector bytes by ID.
    pub fn get(&self, id: u32) -> Option<&[u8]> {
        self.slots.get(id as usize).and_then(|slot| slot.as_deref())
    }

    /// Mark slot as deleted (tombstone). Does NOT reclaim space.
    pub fn remove(&mut self, id: u32) {
        if let Some(slot) = self.slots.get_mut(id as usize) {
            *slot = None;
        }
    }

    /// Number of allocated slots (includes tombstones).
    pub fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Number of active (non-tombstone) entries.
    pub fn active_count(&self) -> u32 {
        self.slots.iter().filter(|s| s.is_some()).count() as u32
    }

    /// Iterator over (id, bytes) for active entries.
    pub fn iter_active(&self) -> impl Iterator<Item = (u32, &[u8])> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_ref().map(|b| (id as u32, b.as_slice())))
    }

    pub fn element_size(&self) -> usize {
        self.element_size
    }

    // ── Persistence ─────────────────────────────────────────────────────────

    /// Persist the store to disk atomically.
    pub fn persist(&self) -> Result<(), EngineError> {
        let path = self
            .store_path
            .as_ref()
            .ok_or_else(|| EngineError::StorageError {
                message: "No store path configured".into(),
            })?;

        let data = PersistentStore {
            slots: self.slots.clone(),
            next_id: self.next_id.load(Ordering::Relaxed),
            element_size: self.element_size,
        };

        let bytes = bincode::serialize(&data).map_err(|e| EngineError::StorageError {
            message: format!("Store serialization: {e}"),
        })?;

        // Atomic write via tmp + rename
        let tmp_path = path.with_extension("bin.tmp");
        {
            let mut f = File::create(&tmp_path)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp_path, path)?;

        info!(
            "VectorStore persisted: {} bytes -> {}",
            bytes.len(),
            path.display()
        );
        Ok(())
    }

    /// Load the store from disk.
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        if !path.exists() {
            return Err(EngineError::NotFound(format!(
                "Store file not found: {}",
                path.display()
            )));
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data: PersistentStore =
            bincode::deserialize_from(reader).map_err(|e| EngineError::StorageError {
                message: format!("Store deserialization: {e}"),
            })?;

        info!(
            "VectorStore loaded: {} slots, {} active, {} bytes/elem",
            data.slots.len(),
            data.slots.iter().filter(|s| s.is_some()).count(),
            data.element_size
        );

        Ok(Self {
            slots: data.slots,
            next_id: AtomicU32::new(data.next_id),
            element_size: data.element_size,
            store_path: Some(path.to_owned()),
        })
    }

    /// Reset store (clear all slots).
    pub fn clear(&mut self) {
        self.slots.clear();
        self.next_id.store(0, Ordering::Release);
    }

    /// Return total bytes consumed by active slots.
    pub fn active_bytes(&self) -> u64 {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|b| b.len() as u64)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_get() {
        let mut store = VectorStore::new_in_memory(16);
        let data = vec![0u8; 16];
        let id = store.append(&data).unwrap();
        assert_eq!(id, 0);
        assert_eq!(store.get(id).unwrap(), &data);
    }

    #[test]
    fn test_wrong_size_rejected() {
        let mut store = VectorStore::new_in_memory(16);
        let result = store.append(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_tombstones() {
        let mut store = VectorStore::new_in_memory(8);
        let id = store.append(&[1u8; 8]).unwrap();
        assert!(store.get(id).is_some());
        store.remove(id);
        assert!(store.get(id).is_none());
    }

    #[test]
    fn test_active_count() {
        let mut store = VectorStore::new_in_memory(4);
        assert_eq!(store.active_count(), 0);
        store.append(&[1; 4]).unwrap();
        store.append(&[2; 4]).unwrap();
        assert_eq!(store.active_count(), 2);
        store.remove(0);
        assert_eq!(store.active_count(), 1);
    }

    #[test]
    fn test_persist_roundtrip() {
        let dir = tempfile::tempdir().unwrap();

        // new() takes a base dir and joins "vector_store.bin" internally
        let mut store = VectorStore::new(dir.path(), 8);
        store.append(&[1u8; 8]).unwrap();
        store.append(&[2u8; 8]).unwrap();
        store.persist().unwrap();

        let store_path = dir.path().join("vector_store.bin");
        let loaded = VectorStore::load(&store_path).unwrap();
        assert_eq!(loaded.active_count(), 2);
        assert_eq!(loaded.element_size(), 8);
        assert_eq!(loaded.get(0).unwrap(), &[1u8; 8]);
        assert_eq!(loaded.get(1).unwrap(), &[2u8; 8]);
    }

    #[test]
    fn test_iter_active() {
        let mut store = VectorStore::new_in_memory(4);
        store.append(&[1; 4]).unwrap();
        store.append(&[2; 4]).unwrap();
        store.remove(0);

        let active: Vec<(u32, &[u8])> = store.iter_active().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, 1);
    }
}
