//! HyperspaceEngine — unified orchestration layer.
//!
//! Provides:
//! - `HyperspaceEngine` async trait (design Section 2.1)
//! - `HyperspaceEngineImpl` struct implementing the trait
//! - `IriRegistry` for u32 ↔ String ID mapping
//! - `SearchHit` typed search result
//! - Full lifecycle: open → insert/upsert/delete → search → checkpoint → vacuum
//!
//! # Architecture
//!
//! ```text
//! HyperspaceEngine trait (async)
//!     └── HyperspaceEngineImpl
//!           ├── EngineWal (write-ahead log)
//!           ├── VectorStore (slot-based persistent storage)
//!           ├── IncrementalHNSW (ANN index with multi-layer search)
//!           ├── JsonLdMetadataIndex (JSON-LD metadata + RoaringBitmap filters)
//!           └── IriRegistry (u32 ↔ String IRI mapping)
//! ```

use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

use async_trait::async_trait;
use serde_json::Value;
use tracing::{info, warn};

use crate::error::EngineError;
use crate::filter::{evaluate_filters, FilterEvaluation, JsonLdFilter};
use crate::hnsw::{HnswConfig, IncrementalHNSW};
use crate::hyper_vector::EmbeddingVector;
use crate::jsonld_meta::JsonLdMetadataIndex;
use crate::metric::{metric_from_kind, Metric};
use crate::snapshot::{self, EngineSnapshot};
use crate::storage::VectorStore;
use crate::tangent::{is_poincare, TangentCache};
use crate::wal::{EngineWal, WalOp, WalSyncMode};

// ── SearchHit ────────────────────────────────────────────────────────────────

/// Typed search result (design Section 2.1).
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: u32,
    pub iri: String,
    pub score: f32,
    pub payload: Option<Value>,
}

// ── IriRegistry ──────────────────────────────────────────────────────────────

/// Bi-directional IRI ↔ u32 ID mapping.
///
/// Provides:
/// - `resolve(iri) → Option<u32>`: find existing ID for an IRI
/// - `register(iri) → u32`: get or create ID
/// - `lookup(id) → Option<String>`: get IRI by numeric ID
#[derive(Debug, Clone)]
pub struct IriRegistry {
    iri_to_id: std::collections::HashMap<String, u32>,
    id_to_iri: std::collections::HashMap<u32, String>,
    next_id: u32,
}

impl IriRegistry {
    pub fn new() -> Self {
        Self {
            iri_to_id: std::collections::HashMap::new(),
            id_to_iri: std::collections::HashMap::new(),
            next_id: 1, // Start at 1; 0 is reserved for non-IRI entries
        }
    }

    /// Register an IRI, returning its numeric ID.
    /// If already registered, returns the existing ID.
    pub fn register(&mut self, iri: &str) -> u32 {
        if let Some(&id) = self.iri_to_id.get(iri) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.iri_to_id.insert(iri.to_string(), id);
        self.id_to_iri.insert(id, iri.to_string());
        id
    }

    /// Resolve an IRI to its numeric ID (if registered).
    pub fn resolve(&self, iri: &str) -> Option<u32> {
        self.iri_to_id.get(iri).copied()
    }

    /// Look up the IRI for a numeric ID.
    pub fn lookup(&self, id: u32) -> Option<String> {
        self.id_to_iri.get(&id).cloned()
    }

    /// Number of registered IRIs.
    pub fn len(&self) -> usize {
        self.iri_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.iri_to_id.is_empty()
    }

    /// Export all entries for snapshot serialization.
    pub fn export(&self) -> Vec<(u32, String)> {
        self.id_to_iri
            .iter()
            .map(|(&id, iri)| (id, iri.clone()))
            .collect()
    }

    /// Import entries from a snapshot.
    pub fn import(&mut self, entries: Vec<(u32, String)>) {
        for (id, iri) in entries {
            self.register_with_id(id, iri);
        }
    }

    /// Restore a stable numeric ID from durable state.
    pub fn register_with_id(&mut self, id: u32, iri: String) {
        self.iri_to_id.insert(iri.clone(), id);
        self.id_to_iri.insert(id, iri);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
    }
}

impl Default for IriRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Searcher (read-only snapshot) ────────────────────────────────────────────

/// Searcher carries a snapshot of the engine state for concurrent search.
///
/// Created via `HyperspaceEngineImpl::searcher()`. Clone is O(n) — for production
/// use with large datasets, implement a snapshot-based mechanism.
pub struct Searcher {
    index: IncrementalHNSW,
    metadata: JsonLdMetadataIndex,
    iri_registry: IriRegistry,
    /// Immutable vectors from the same active-store snapshot as `index`.
    /// They are used only to complete selective filtered searches when ANN
    /// candidates are insufficient.
    vectors: HashMap<u32, EmbeddingVector>,
}

impl Searcher {
    fn new(
        index: IncrementalHNSW,
        metadata: JsonLdMetadataIndex,
        iri_registry: IriRegistry,
        vectors: HashMap<u32, EmbeddingVector>,
    ) -> Self {
        Self {
            index,
            metadata,
            iri_registry,
            vectors,
        }
    }

    /// Search without filters.
    pub fn search(&mut self, query: &EmbeddingVector, top_k: usize) -> Vec<SearchHit> {
        self.search_with_filter(query, top_k, &[])
    }

    /// Search with JSON-LD filters.
    pub fn search_with_filter(
        &mut self,
        query: &EmbeddingVector,
        top_k: usize,
        filters: &[JsonLdFilter],
    ) -> Vec<SearchHit> {
        let results = match evaluate_filters(&self.metadata, filters) {
            FilterEvaluation::Unrestricted => self.index.search(query, top_k),
            FilterEvaluation::Matched(allowed) => {
                let ann_results = self.index.search_with_filter(query, top_k, Some(&allowed));
                let expected = top_k.min(allowed.len() as usize);
                if ann_results.len() >= expected {
                    ann_results
                } else {
                    // P6#8: tangent-space pruning for Poincaré — build a tangent
                    // cache over the allowed set, prune candidates, then exact
                    // re-rank within the pruned candidate set.
                    let mut exact = if is_poincare(&self.index.metric().kind())
                        && allowed.len() as usize > top_k
                    {
                        let mut id_vecs: Vec<(u32, EmbeddingVector)> = allowed
                            .iter()
                            .filter_map(|id| {
                                self.vectors
                                    .get(&id)
                                    .map(|v| (id, v.clone()))
                            })
                            .collect();
                        id_vecs.sort_by_key(|(id, _)| *id);
                        let ids: Vec<u32> = id_vecs.iter().map(|(id, _)| *id).collect();
                        let vectors: Vec<EmbeddingVector> =
                            id_vecs.into_iter().map(|(_, v)| v).collect();
                        let cache = TangentCache::build(&vectors, 1.0);
                        cache
                            .search_with_pruning(query, &ids, top_k, 2)
                            .into_iter()
                            .filter_map(|id| {
                                self.vectors.get(&id).map(|vector| {
                                    (id, self.index.metric().distance(query, vector))
                                })
                            })
                            .collect::<Vec<_>>()
                    } else {
                        allowed
                            .iter()
                            .filter_map(|id| {
                                self.vectors
                                    .get(&id)
                                    .map(|vector| (id, self.index.metric().distance(query, vector)))
                            })
                            .collect::<Vec<_>>()
                    };
                    exact.sort_by(|left, right| {
                        left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal)
                    });
                    exact.truncate(top_k);
                    exact
                }
            }
            FilterEvaluation::Empty => Vec::new(),
        };

        results
            .into_iter()
            .map(|(id, dist)| {
                let iri = self.iri_registry.lookup(id).unwrap_or_default();
                let payload = self.metadata.get_payload(id);
                SearchHit {
                    id,
                    iri,
                    score: -(dist as f32),
                    payload,
                }
            })
            .collect()
    }

    pub fn metadata(&self) -> &JsonLdMetadataIndex {
        &self.metadata
    }
}

// ── EngineInner (Mutex-protected mutable state) ──────────────────────────────

struct EngineInner {
    index: IncrementalHNSW,
    store: VectorStore,
    clock: u64,
}

// ── HyperspaceEngine trait ───────────────────────────────────────────────────

/// Core engine trait (design Section 2.1).
#[async_trait]
pub trait HyperspaceEngine: Send + Sync {
    // ── Writing ──
    async fn insert(
        &self,
        iri: &str,
        vector: EmbeddingVector,
        jsonld: Value,
    ) -> Result<u32, EngineError>;
    async fn upsert(
        &self,
        iri: &str,
        vector: EmbeddingVector,
        jsonld: Value,
    ) -> Result<u32, EngineError>;
    async fn delete(&self, iri: &str) -> Result<(), EngineError>;

    /// Resolve an IRI to its numeric ID (if registered).
    async fn resolve_iri(&self, iri: &str) -> Result<Option<u32>, EngineError>;

    /// Look up the IRI for a numeric ID (reverse of resolve_iri).
    async fn lookup_id(&self, id: u32) -> Result<Option<String>, EngineError>;

    // ── Retrieval ──
    async fn search(
        &self,
        query: &EmbeddingVector,
        top_k: usize,
        filters: &[JsonLdFilter],
    ) -> Result<Vec<SearchHit>, EngineError>;

    /// Dual-space hybrid search: text (Cosine) × struct (Poincaré) weighted fusion.
    async fn hybrid_search(
        &self,
        text_query: Option<&EmbeddingVector>,
        struct_query: Option<&EmbeddingVector>,
        top_k: usize,
        alpha: f32,
        filters: &[JsonLdFilter],
    ) -> Result<Vec<SearchHit>, EngineError>;

    // ── Metadata ──
    async fn count(&self) -> Result<u64, EngineError>;
    async fn get_payload(&self, iri: &str) -> Result<Option<Value>, EngineError>;
    async fn get_vector(&self, iri: &str) -> Result<Option<EmbeddingVector>, EngineError>;
    async fn list(&self, offset: usize, limit: usize) -> Result<Vec<SearchHit>, EngineError>;

    // ── Maintenance ──
    async fn checkpoint(&self) -> Result<(), EngineError>;
    async fn vacuum(&self) -> Result<(), EngineError>;
}

// ── HyperspaceEngineImpl ─────────────────────────────────────────────────────

/// Concrete engine implementation.
pub struct HyperspaceEngineImpl {
    /// Serializes mutations with checkpoint creation so a snapshot cannot be
    /// taken between a WAL append and its in-memory application (or vice versa).
    write_barrier: Mutex<()>,
    /// RwLock over the index/store: searches take the read lock (concurrent),
    /// mutations take the write lock.
    inner: RwLock<EngineInner>,
    metadata: JsonLdMetadataIndex,
    iri_registry: Mutex<IriRegistry>,
    wal: EngineWal,
    data_dir: PathBuf,
    config: HnswConfig,
    dim: usize,
}

impl HyperspaceEngineImpl {
    /// Open or create an engine at the given directory.
    pub fn open(
        dir: &Path,
        sync_mode: WalSyncMode,
        dim: usize,
        metric: Box<dyn Metric>,
        config: HnswConfig,
    ) -> Result<Self, EngineError> {
        let wal = EngineWal::open(dir, sync_mode)?;
        let element_size = EmbeddingVector::element_size(dim);
        let store = VectorStore::new(dir, element_size);
        let index = IncrementalHNSW::new(metric, config.clone());
        let metadata = JsonLdMetadataIndex::new();
        let iri_registry = IriRegistry::new();

        let engine = Self {
            write_barrier: Mutex::new(()),
            inner: RwLock::new(EngineInner {
                index,
                store,
                clock: 0,
            }),
            metadata,
            iri_registry: Mutex::new(iri_registry),
            wal,
            data_dir: dir.to_owned(),
            config,
            dim,
        };

        // Load the newest snapshot first, then replay records newer than its
        // clock from both frozen segments and the active WAL.
        let snapshot_path = dir.join("index.snapshot");
        if snapshot_path.exists() {
            match snapshot::load_snapshot(&snapshot_path) {
                Ok(snap) => {
                    info!(
                        "Loading snapshot: {} nodes, clock={}",
                        snap.nodes.len(),
                        snap.clock
                    );
                    let mut inner = engine.inner.write().unwrap();
                    let metric_kind = inner.index.metric().kind();
                    inner.index.import_nodes(snap.nodes.clone());
                    inner.clock = snap.clock;
                    // Populate VectorStore from HNSW node data
                    for (node_id, node_opt) in snap.nodes.iter().enumerate() {
                        if let Some(node) = node_opt {
                            let vec =
                                EmbeddingVector::new_unchecked(node.coords.clone(), metric_kind);
                            let bytes = vec.as_bytes();
                            let _ = inner.store.set(node_id as u32, &bytes);
                        }
                    }
                    if let Ok(mut reg) = engine.iri_registry.lock() {
                        reg.import(snap.iri_registry);
                    }
                    // Restore forward metadata (JSON strings → Values)
                    for (id, payload_str) in snap.forward_meta {
                        if let Ok(payload) = serde_json::from_str(&payload_str) {
                            engine.metadata.index(id, &payload);
                        }
                    }
                    // Restore deleted IDs
                    for id in snap.deleted_ids {
                        engine.metadata.remove(id);
                    }
                    drop(inner);
                }
                Err(e) => {
                    warn!("Failed to load snapshot, falling back to WAL replay: {e}");
                }
            }
        }
        engine.recover()?;

        Ok(engine)
    }

    /// Recover state by replaying frozen segments and the active WAL into all
    /// business-visible indexes.
    fn recover(&self) -> Result<(), EngineError> {
        let mut wal_paths = self.wal.frozen_paths()?;
        wal_paths.push(self.wal.active_path().to_owned());
        let mut inner = self.inner.write().unwrap();
        let mut recovered = 0u64;

        for wal_path in wal_paths {
            recovered += EngineWal::replay(&wal_path, |record| {
                // The snapshot already contains this record. Reapplying it
                // would duplicate HNSW links and stale metadata indexes.
                if record.clock <= inner.clock {
                    return Ok(());
                }
                if record.legacy {
                    return Err(EngineError::StorageError {
                        message: format!(
                            "Cannot safely recover legacy WAL record newer than snapshot clock {} from {}: IRI and metadata were not persisted",
                            inner.clock,
                            wal_path.display()
                        ),
                    });
                }

                inner.clock = record.clock;
                match record.op {
                    WalOp::Insert { id, iri } | WalOp::Upsert { id, iri } => {
                        if iri.is_empty() {
                            return Err(EngineError::StorageError {
                                message: format!("Versioned WAL record {} has an empty IRI", id),
                            });
                        }
                        inner.index.remove(id);
                        self.metadata.remove(id);
                        inner.store.set(id, &record.data)?;
                        let vector =
                            EmbeddingVector::from_bytes(&record.data, self.dim).map_err(|e| {
                                EngineError::StorageError {
                                    message: format!("WAL vector deserialization for {iri}: {e}"),
                                }
                            })?;
                        inner.index.insert(id, vector);
                        self.iri_registry.lock().unwrap().register_with_id(id, iri);
                        if let Some(metadata) = record.metadata {
                            self.metadata.index(id, &metadata);
                        }
                        self.metadata.undelete(id);
                    }
                    WalOp::Delete { id, .. } => {
                        inner.store.remove(id);
                        inner.index.remove(id);
                        self.metadata.remove(id);
                    }
                    WalOp::MetadataUpdate { id, iri } => {
                        if iri.is_empty() {
                            return Err(EngineError::StorageError {
                                message: format!(
                                    "Versioned metadata WAL record {} has an empty IRI",
                                    id
                                ),
                            });
                        }
                        self.iri_registry.lock().unwrap().register_with_id(id, iri);
                        self.metadata.remove(id);
                        if let Some(metadata) = record.metadata {
                            self.metadata.index(id, &metadata);
                        }
                        self.metadata.undelete(id);
                    }
                }
                Ok(())
            })?;
        }

        info!(recovered, "WAL replay complete");
        Ok(())
    }

    /// Create a Searcher (read-only snapshot) for concurrent querying.
    /// Note: O(n) — clones the HNSW index.
    pub fn searcher(&self) -> Searcher {
        let inner = self.inner.read().unwrap();
        let metric_kind = inner.index.metric().kind();
        let mut new_index =
            IncrementalHNSW::new(metric_from_kind(metric_kind), self.config.clone());
        let mut vectors = HashMap::new();
        for (id, _) in inner.store.iter_active() {
            if let Some(bytes) = inner.store.get(id) {
                let dim = (inner.store.element_size().saturating_sub(12)) / 8;
                if let Ok(vec) = EmbeddingVector::from_bytes(bytes, dim) {
                    new_index.insert(id, vec.clone());
                    vectors.insert(id, vec);
                }
            }
        }
        let iri_registry = self.iri_registry.lock().unwrap().clone();
        Searcher::new(new_index, self.metadata.clone(), iri_registry, vectors)
    }
}

#[async_trait]
impl HyperspaceEngine for HyperspaceEngineImpl {
    // ── Insert ──────────────────────────────────────────────────────────────

    async fn insert(
        &self,
        iri: &str,
        vector: EmbeddingVector,
        jsonld: Value,
    ) -> Result<u32, EngineError> {
        let _write_guard = self.write_barrier.lock().unwrap();
        let id = {
            let mut reg = self.iri_registry.lock().unwrap();
            reg.register(iri)
        };
        let bytes = vector.as_bytes();

        // WAL first
        self.wal.append(
            &WalOp::Insert {
                id,
                iri: iri.to_string(),
            },
            {
                let mut inner = self.inner.write().unwrap();
                inner.clock += 1;
                inner.clock
            },
            &bytes,
            Some(&jsonld),
        )?;

        // Apply to store + index + metadata
        let mut inner = self.inner.write().unwrap();
        inner.index.remove(id);
        self.metadata.remove(id);
        inner.store.set(id, &bytes)?;
        inner.index.insert(id, vector);
        self.metadata.index(id, &jsonld);
        // Clear deleted flag in case this is a re-insert of the same IRI
        self.metadata.undelete(id);

        Ok(id)
    }

    // ── Upsert ──────────────────────────────────────────────────────────────

    async fn upsert(
        &self,
        iri: &str,
        vector: EmbeddingVector,
        jsonld: Value,
    ) -> Result<u32, EngineError> {
        let _write_guard = self.write_barrier.lock().unwrap();
        let id = {
            let mut reg = self.iri_registry.lock().unwrap();
            reg.register(iri)
        };
        let bytes = vector.as_bytes();

        self.wal.append(
            &WalOp::Upsert {
                id,
                iri: iri.to_string(),
            },
            {
                let mut inner = self.inner.write().unwrap();
                inner.clock += 1;
                inner.clock
            },
            &bytes,
            Some(&jsonld),
        )?;

        let mut inner = self.inner.write().unwrap();
        // Replace all secondary state for this IRI; a VectorStore overwrite
        // alone cannot repair HNSW edges or metadata bitmap memberships.
        inner.index.remove(id);
        self.metadata.remove(id);
        inner.store.set(id, &bytes)?;
        inner.index.insert(id, vector);
        self.metadata.index(id, &jsonld);
        self.metadata.undelete(id);

        Ok(id)
    }

    // ── Delete ──────────────────────────────────────────────────────────────

    async fn delete(&self, iri: &str) -> Result<(), EngineError> {
        let _write_guard = self.write_barrier.lock().unwrap();
        let id = {
            let reg = self.iri_registry.lock().unwrap();
            reg.resolve(iri)
                .ok_or_else(|| EngineError::NotFound(iri.to_string()))?
        };

        self.wal.append(
            &WalOp::Delete {
                id,
                iri: iri.to_string(),
            },
            {
                let mut inner = self.inner.write().unwrap();
                inner.clock += 1;
                inner.clock
            },
            &[],
            None,
        )?;

        let mut inner = self.inner.write().unwrap();
        inner.store.remove(id);
        inner.index.remove(id);
        self.metadata.remove(id);

        Ok(())
    }

    // ── Search ──────────────────────────────────────────────────────────────

    async fn search(
        &self,
        query: &EmbeddingVector,
        top_k: usize,
        filters: &[JsonLdFilter],
    ) -> Result<Vec<SearchHit>, EngineError> {
        let inner = self.inner.read().unwrap();
        let results = match evaluate_filters(&self.metadata, filters) {
            FilterEvaluation::Unrestricted => inner.index.search(query, top_k),
            FilterEvaluation::Matched(allowed) => {
                let ann_results = inner.index.search_with_filter(query, top_k, Some(&allowed));
                let expected = top_k.min(allowed.len() as usize);
                if ann_results.len() >= expected {
                    ann_results
                } else {
                    // HNSW explores an unfiltered graph and applies the
                    // bitmap afterwards. A selective filter can therefore
                    // leave fewer than the requested number of candidates
                    // even though matching vectors exist. In that case,
                    // complete the finite allowed set exactly: returning a
                    // partial filtered result is a correctness failure, not
                    // an acceptable ANN approximation.
                    let mut exact = inner
                        .store
                        .iter_active()
                        .filter(|(id, _)| allowed.contains(*id))
                        .map(|(id, bytes)| {
                            let vector = EmbeddingVector::from_bytes(bytes, self.dim)?;
                            Ok((id, inner.index.metric().distance(query, &vector)))
                        })
                        .collect::<Result<Vec<(u32, f64)>, EngineError>>()?;
                    exact.sort_by(|left, right| {
                        left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal)
                    });
                    exact.truncate(top_k);
                    exact
                }
            }
            FilterEvaluation::Empty => Vec::new(),
        };

        let reg = self.iri_registry.lock().unwrap();
        Ok(results
            .into_iter()
            .map(|(id, dist)| {
                let iri = reg.lookup(id).unwrap_or_default();
                let payload = self.metadata.get_payload(id);
                SearchHit {
                    id,
                    iri,
                    score: -(dist as f32),
                    payload,
                }
            })
            .collect())
    }

    // ── Hybrid Search ───────────────────────────────────────────────────────

    async fn hybrid_search(
        &self,
        text_query: Option<&EmbeddingVector>,
        struct_query: Option<&EmbeddingVector>,
        top_k: usize,
        alpha: f32,
        filters: &[JsonLdFilter],
    ) -> Result<Vec<SearchHit>, EngineError> {
        let allowed = evaluate_filters(&self.metadata, filters);

        let inner = self.inner.read().unwrap();

        // For hybrid search, text and struct use the same index.
        // In production, use separate indexes: one Cosine, one Poincaré.
        let text_results = text_query.map_or(Vec::new(), |q| match &allowed {
            FilterEvaluation::Unrestricted => inner.index.search(q, top_k * 3),
            FilterEvaluation::Matched(bitmap) => {
                inner.index.search_with_filter(q, top_k * 3, Some(bitmap))
            }
            FilterEvaluation::Empty => Vec::new(),
        });
        let struct_results = struct_query.map_or(Vec::new(), |q| match &allowed {
            FilterEvaluation::Unrestricted => inner.index.search(q, top_k * 3),
            FilterEvaluation::Matched(bitmap) => {
                inner.index.search_with_filter(q, top_k * 3, Some(bitmap))
            }
            FilterEvaluation::Empty => Vec::new(),
        });

        drop(inner);
        let reg = self.iri_registry.lock().unwrap();

        if text_results.is_empty() && struct_results.is_empty() {
            return Ok(Vec::new());
        }

        // RRF-style fusion
        let max_text_dist = text_results.first().map(|r| r.1).unwrap_or(1.0).max(0.001);
        let max_struct_dist = struct_results
            .first()
            .map(|r| r.1)
            .unwrap_or(1.0)
            .max(0.001);

        let mut fused: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for (id, d) in &text_results {
            let score = alpha * (1.0 - (*d as f32) / max_text_dist as f32);
            *fused.entry(*id).or_insert(0.0) += score;
        }
        for (id, d) in &struct_results {
            let score = (1.0 - alpha) * (1.0 - (*d as f32) / max_struct_dist as f32);
            *fused.entry(*id).or_insert(0.0) += score;
        }

        let mut sorted: Vec<(u32, f32)> = fused.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(top_k);

        Ok(sorted
            .into_iter()
            .map(|(id, score)| {
                let iri = reg.lookup(id).unwrap_or_default();
                let payload = self.metadata.get_payload(id);
                SearchHit {
                    id,
                    iri,
                    score,
                    payload,
                }
            })
            .collect())
    }

    // ── Count ───────────────────────────────────────────────────────────────

    async fn count(&self) -> Result<u64, EngineError> {
        Ok(self.metadata.count())
    }

    // ── Get Payload ─────────────────────────────────────────────────────────

    async fn get_payload(&self, iri: &str) -> Result<Option<Value>, EngineError> {
        let reg = self.iri_registry.lock().unwrap();
        let payload = reg
            .resolve(iri)
            .and_then(|id| self.metadata.get_payload(id));
        Ok(payload)
    }

    async fn get_vector(&self, iri: &str) -> Result<Option<EmbeddingVector>, EngineError> {
        let id = {
            let reg = self.iri_registry.lock().unwrap();
            reg.resolve(iri)
        };
        match id {
            None => Ok(None),
            Some(id) => {
                let inner = self.inner.read().unwrap();
                match inner.store.get(id) {
                    None => Ok(None),
                    Some(bytes) => {
                        let dim = (inner.store.element_size().saturating_sub(12)) / 8;
                        let vec = EmbeddingVector::from_bytes(bytes, dim).map_err(|e| {
                            EngineError::StorageError {
                                message: format!("Vector deserialization: {e}"),
                            }
                        })?;
                        Ok(Some(vec))
                    }
                }
            }
        }
    }

    async fn resolve_iri(&self, iri: &str) -> Result<Option<u32>, EngineError> {
        let reg = self.iri_registry.lock().unwrap();
        Ok(reg.resolve(iri))
    }

    async fn lookup_id(&self, id: u32) -> Result<Option<String>, EngineError> {
        let reg = self.iri_registry.lock().unwrap();
        Ok(reg.lookup(id))
    }

    // ── List ────────────────────────────────────────────────────────────────

    async fn list(&self, offset: usize, limit: usize) -> Result<Vec<SearchHit>, EngineError> {
        let reg = self.iri_registry.lock().unwrap();
        let all_ids = self.metadata.all_ids();
        let page: Vec<u32> = all_ids.into_iter().skip(offset).take(limit).collect();
        Ok(page
            .into_iter()
            .map(|id| {
                let iri = reg.lookup(id).unwrap_or_default();
                let payload = self.metadata.get_payload(id);
                SearchHit {
                    id,
                    iri,
                    score: 0.0,
                    payload,
                }
            })
            .collect())
    }

    // ── Checkpoint ──────────────────────────────────────────────────────────

    async fn checkpoint(&self) -> Result<(), EngineError> {
        let snapshot_path = self.data_dir.join("index.snapshot");

        // Keep the WAL and in-memory state at one common clock while the
        // snapshot is made.  The snapshot is committed before rotation: a
        // crash before rotation simply replays already-snapshotted records,
        // which recovery skips by clock; a crash after rotation retains the
        // frozen segment for the same reason.
        let _write_guard = self.write_barrier.lock().unwrap();
        self.wal.sync()?;

        // Phase 1: Build and persist a snapshot of the common clock.
        let (nodes, clock, iri_entries, forward_entries, deleted_ids) = {
            let inner = self.inner.read().unwrap();
            let reg = self.iri_registry.lock().unwrap();

            let nodes = inner.index.export_nodes();
            let clock = inner.clock;
            let iri_entries = reg.export();
            let forward_entries: Vec<(u32, String)> = self
                .metadata
                .forward
                .iter()
                .map(|e| {
                    (
                        *e.key(),
                        serde_json::to_string(e.value()).unwrap_or_default(),
                    )
                })
                .collect();
            let deleted_ids: Vec<u32> = self
                .metadata
                .deleted
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .collect();

            (nodes, clock, iri_entries, forward_entries, deleted_ids)
        };

        let snap = EngineSnapshot {
            nodes,
            entry_point: 0, // will be reconstructed on import
            clock,
            iri_registry: iri_entries,
            forward_meta: forward_entries,
            deleted_ids,
            dimension: self.dim,
            config: self.config.clone(),
        };
        snapshot::save_snapshot(&snapshot_path, &snap)?;

        // Phase 2: move all records covered by the snapshot to a frozen
        // segment, then delete frozen segments only after the snapshot is
        // durable. New mutations are blocked until this completes.
        self.wal.rotate()?;
        self.wal.cleanup_frozen()?;

        info!("Checkpoint complete: snapshot saved, frozen WALs cleaned");
        Ok(())
    }

    // ── Vacuum ──────────────────────────────────────────────────────────────

    async fn vacuum(&self) -> Result<(), EngineError> {
        let cleaned = self.metadata.vacuum();
        info!(
            "Vacuum complete: cleaned {} entries from metadata indexes",
            cleaned
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyper_vector::MetricKind;
    use crate::metric::CosineMetric;
    use serde_json::json;

    fn setup_engine(dir: &Path) -> HyperspaceEngineImpl {
        HyperspaceEngineImpl::open(
            dir,
            WalSyncMode::Strict,
            4,
            Box::new(CosineMetric),
            HnswConfig::default(),
        )
        .unwrap()
    }

    fn v(coords: Vec<f64>) -> EmbeddingVector {
        EmbeddingVector::new_unchecked(coords, MetricKind::Cosine)
    }

    fn setup_async_engine(dir: &Path) -> HyperspaceEngineImpl {
        setup_engine(dir)
    }

    #[tokio::test]
    async fn test_engine_insert_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert(
            "vec:1",
            v(vec![1.0, 0.0, 0.0, 0.0]),
            json!({"@type": ["Test"], "label": "first"}),
        )
        .await
        .unwrap();
        eng.insert(
            "vec:2",
            v(vec![0.0, 1.0, 0.0, 0.0]),
            json!({"@type": ["Test"], "label": "second"}),
        )
        .await
        .unwrap();

        assert_eq!(eng.count().await.unwrap(), 2);

        let results = eng
            .search(&v(vec![1.0, 0.0, 0.0, 0.0]), 5, &[])
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1);
        assert_eq!(results[0].iri, "vec:1");
    }

    #[tokio::test]
    async fn test_engine_delete() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert("a", v(vec![1.0, 0.0, 0.0, 0.0]), json!({"id": "a"}))
            .await
            .unwrap();
        eng.insert("b", v(vec![0.0, 1.0, 0.0, 0.0]), json!({"id": "b"}))
            .await
            .unwrap();
        assert_eq!(eng.count().await.unwrap(), 2);

        eng.delete("a").await.unwrap();
        assert_eq!(eng.count().await.unwrap(), 1);

        let results = eng
            .search(&v(vec![1.0, 0.0, 0.0, 0.0]), 5, &[])
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 2);
    }

    #[tokio::test]
    async fn test_upsert_replaces_vector_and_all_metadata_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert(
            "skill:one",
            v(vec![1.0, 0.0, 0.0, 0.0]),
            json!({
                "@type": ["OldType"],
                "named_graph": "old-graph",
                "tags": ["old"],
                "importance": 0.9
            }),
        )
        .await
        .unwrap();
        eng.upsert(
            "skill:one",
            v(vec![0.0, 1.0, 0.0, 0.0]),
            json!({
                "@type": ["NewType"],
                "named_graph": "new-graph",
                "tags": ["new"],
                "importance": 0.1
            }),
        )
        .await
        .unwrap();

        assert!(eng
            .search(
                &v(vec![1.0, 0.0, 0.0, 0.0]),
                5,
                &[JsonLdFilter::Type("OldType".into())]
            )
            .await
            .unwrap()
            .is_empty());
        assert!(eng
            .search(
                &v(vec![1.0, 0.0, 0.0, 0.0]),
                5,
                &[JsonLdFilter::NamedGraph("old-graph".into())]
            )
            .await
            .unwrap()
            .is_empty());
        let updated = eng
            .search(
                &v(vec![0.0, 1.0, 0.0, 0.0]),
                5,
                &[JsonLdFilter::Type("NewType".into())],
            )
            .await
            .unwrap();
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].iri, "skill:one");
        assert!(eng
            .search(
                &v(vec![0.0, 1.0, 0.0, 0.0]),
                5,
                &[JsonLdFilter::Range {
                    key: "importance".into(),
                    gte: Some(0.8),
                    lte: None,
                }]
            )
            .await
            .unwrap()
            .is_empty());

        eng.delete("skill:one").await.unwrap();
        assert!(eng
            .search(
                &v(vec![0.0, 1.0, 0.0, 0.0]),
                5,
                &[JsonLdFilter::tag("tags", "new")]
            )
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn test_engine_recovery() {
        let dir = tempfile::tempdir().unwrap();
        {
            let eng = setup_async_engine(dir.path());
            eng.insert(
                "p",
                v(vec![1.0, 0.0, 0.0, 0.0]),
                json!({"label": "persist"}),
            )
            .await
            .unwrap();
            eng.checkpoint().await.unwrap();
        }
        // Re-open
        let eng = setup_async_engine(dir.path());
        assert_eq!(eng.count().await.unwrap(), 1);
        let results = eng
            .search(&v(vec![1.0, 0.0, 0.0, 0.0]), 5, &[])
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].iri, "p");
    }

    #[tokio::test]
    async fn test_wal_only_recovery_restores_iri_payload_and_vector() {
        let dir = tempfile::tempdir().unwrap();
        {
            let eng = setup_async_engine(dir.path());
            eng.insert(
                "wal:only",
                v(vec![1.0, 0.0, 0.0, 0.0]),
                json!({"@type": ["Experience"], "label": "durable payload"}),
            )
            .await
            .unwrap();
            // Deliberately no checkpoint: recovery must use WAL alone.
        }

        let eng = setup_async_engine(dir.path());
        assert_eq!(eng.count().await.unwrap(), 1);
        assert_eq!(eng.resolve_iri("wal:only").await.unwrap(), Some(1));
        assert_eq!(
            eng.get_payload("wal:only").await.unwrap().unwrap()["label"],
            "durable payload"
        );
        let hits = eng
            .search(&v(vec![1.0, 0.0, 0.0, 0.0]), 1, &[])
            .await
            .unwrap();
        assert_eq!(hits[0].iri, "wal:only");
    }

    #[tokio::test]
    async fn test_recovery_replays_active_wal_after_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        {
            let eng = setup_async_engine(dir.path());
            eng.insert(
                "snap:base",
                v(vec![1.0, 0.0, 0.0, 0.0]),
                json!({"label": "base"}),
            )
            .await
            .unwrap();
            eng.checkpoint().await.unwrap();
            eng.insert(
                "snap:tail",
                v(vec![0.0, 1.0, 0.0, 0.0]),
                json!({"label": "tail"}),
            )
            .await
            .unwrap();
        }

        let eng = setup_async_engine(dir.path());
        assert_eq!(eng.count().await.unwrap(), 2);
        assert_eq!(
            eng.get_payload("snap:tail").await.unwrap().unwrap()["label"],
            "tail"
        );
    }

    #[tokio::test]
    async fn test_recovery_replays_frozen_wal_after_interrupted_rotation() {
        let dir = tempfile::tempdir().unwrap();
        {
            let eng = setup_async_engine(dir.path());
            eng.insert(
                "frozen:entry",
                v(vec![1.0, 0.0, 0.0, 0.0]),
                json!({"label": "frozen"}),
            )
            .await
            .unwrap();
            // Simulate a crash after WAL rotation but before snapshot/cleanup.
            eng.wal.rotate().unwrap();
        }

        let eng = setup_async_engine(dir.path());
        assert_eq!(eng.count().await.unwrap(), 1);
        assert_eq!(
            eng.get_payload("frozen:entry").await.unwrap().unwrap()["label"],
            "frozen"
        );
    }

    #[tokio::test]
    async fn test_search_with_filters() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert(
            "doc:1",
            v(vec![1.0, 0.0, 0.0, 0.0]),
            json!({"@type": ["Document"], "tags": ["important"], "importance": 0.9}),
        )
        .await
        .unwrap();
        eng.insert(
            "doc:2",
            v(vec![0.0, 1.0, 0.0, 0.0]),
            json!({"@type": ["Document", "Report"], "tags": ["normal"], "importance": 0.5}),
        )
        .await
        .unwrap();
        eng.insert(
            "note:1",
            v(vec![0.0, 0.0, 1.0, 0.0]),
            json!({"@type": ["Note"], "tags": ["important"], "importance": 0.3}),
        )
        .await
        .unwrap();

        // Filter by type
        let results = eng
            .search(
                &v(vec![0.5, 0.5, 0.5, 0.0]),
                10,
                &[JsonLdFilter::Type("Document".into())],
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|h| h.iri == "doc:1"));
        assert!(results.iter().any(|h| h.iri == "doc:2"));

        // Filter by tag
        let results = eng
            .search(
                &v(vec![0.5, 0.5, 0.5, 0.0]),
                10,
                &[JsonLdFilter::tag("tags", "important")],
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|h| h.iri == "doc:1"));
        assert!(results.iter().any(|h| h.iri == "note:1"));
    }

    #[tokio::test]
    async fn test_search_with_non_matching_filter_returns_no_hits() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert(
            "doc:1",
            v(vec![1.0, 0.0, 0.0, 0.0]),
            json!({"@type": ["Document"]}),
        )
        .await
        .unwrap();

        let results = eng
            .search(
                &v(vec![1.0, 0.0, 0.0, 0.0]),
                10,
                &[JsonLdFilter::Type("MissingType".into())],
            )
            .await
            .unwrap();

        assert!(
            results.is_empty(),
            "a filter with no matches must not fall back to unfiltered search"
        );
    }

    #[tokio::test]
    async fn test_get_payload() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert("x", v(vec![1.0, 0.0, 0.0, 0.0]), json!({"text": "hello"}))
            .await
            .unwrap();
        let payload = eng.get_payload("x").await.unwrap();
        assert!(payload.is_some());
        assert_eq!(
            payload.unwrap().get("text").unwrap().as_str().unwrap(),
            "hello"
        );

        let missing = eng.get_payload("nonexistent").await.unwrap();
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_list() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        for i in 0..5u32 {
            let iri = format!("item:{i}");
            eng.insert(&iri, v(vec![1.0, 0.0, 0.0, 0.0]), json!({"idx": i}))
                .await
                .unwrap();
        }

        let all = eng.list(0, 10).await.unwrap();
        assert_eq!(all.len(), 5);

        let page = eng.list(1, 2).await.unwrap();
        assert_eq!(page.len(), 2);
    }

    #[tokio::test]
    async fn test_hybrid_search() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert("h:1", v(vec![1.0, 0.0, 0.0, 0.0]), json!({"text": "first"}))
            .await
            .unwrap();
        eng.insert(
            "h:2",
            v(vec![0.0, 1.0, 0.0, 0.0]),
            json!({"text": "second"}),
        )
        .await
        .unwrap();

        let q = v(vec![1.0, 0.0, 0.0, 0.0]);
        let results = eng
            .hybrid_search(Some(&q), Some(&q), 5, 0.5, &[])
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].iri, "h:1");
    }

    #[tokio::test]
    async fn test_vacuum_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert("v:1", v(vec![1.0, 0.0, 0.0, 0.0]), json!({"x": "y"}))
            .await
            .unwrap();
        eng.delete("v:1").await.unwrap();
        // Vacuum should not panic on empty cleaned state
        eng.vacuum().await.unwrap();
    }

    #[tokio::test]
    async fn test_searcher() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());

        eng.insert("s:1", v(vec![1.0, 0.0, 0.0, 0.0]), json!({"x": "y"}))
            .await
            .unwrap();
        eng.insert("s:2", v(vec![0.0, 1.0, 0.0, 0.0]), json!({"x": "z"}))
            .await
            .unwrap();

        let mut srch = eng.searcher();
        let results = srch.search(&v(vec![1.0, 0.0, 0.0, 0.0]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "s:1");
    }

    #[tokio::test]
    async fn test_concurrent_searches_share_read_lock() {
        let dir = tempfile::tempdir().unwrap();
        let eng = std::sync::Arc::new(setup_async_engine(dir.path()));
        for index in 0..32u32 {
            let mut coords = vec![0.05; 4];
            coords[(index % 4) as usize] = 1.0;
            eng.insert(&format!("c:{}", index), v(coords), json!({"x": index}))
                .await
                .unwrap();
        }

        let query = v(vec![1.0, 0.0, 0.0, 0.0]);
        let mut handles = Vec::new();
        for _ in 0..8 {
            let eng = eng.clone();
            let query = query.clone();
            handles.push(std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                rt.block_on(async move {
                    for _ in 0..20 {
                        let results = eng.search(&query, 5, &[]).await.unwrap();
                        assert!(!results.is_empty(), "concurrent search returned nothing");
                    }
                })
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[tokio::test]
    async fn test_searcher_completes_selective_filtered_results() {
        let dir = tempfile::tempdir().unwrap();
        let eng = setup_async_engine(dir.path());
        for index in 0..64u32 {
            let mut coords = vec![0.05; 4];
            coords[(index % 4) as usize] = 1.0;
            let type_name = if index.is_multiple_of(3) {
                "Other"
            } else {
                "Selected"
            };
            eng.insert(
                &format!("snapshot:{index}"),
                v(coords),
                json!({"@type": [type_name], "index": index}),
            )
            .await
            .unwrap();
        }

        let mut searcher = eng.searcher();
        let hits = searcher.search_with_filter(
            &v(vec![1.0, 0.0, 0.0, 0.0]),
            64,
            &[JsonLdFilter::Type("Selected".to_string())],
        );
        assert_eq!(hits.len(), 42);
        assert!(hits.iter().all(|hit| {
            hit.payload.as_ref().is_some_and(|payload| {
                payload["@type"]
                    .as_array()
                    .is_some_and(|types| types.iter().any(|value| value == "Selected"))
            })
        }));
    }
}
