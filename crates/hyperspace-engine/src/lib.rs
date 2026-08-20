//! HyperspaceEngine — Embedded spatial memory engine for Gliding Horse Agent OS.
//!
//! Provides production-grade vector storage with:
//! - HNSW approximate nearest neighbor search (IncrementalHNSW)
//! - Runtime-switchable metrics (Cosine, Poincaré, Lorentz, Euclidean)
//! - CRC32-verified Write-Ahead Log (WAL) with 3 sync modes
//! - JSON-LD metadata index with RoaringBitmap inverted indexes
//! - Tangent-space pruning for Poincaré ball search
//! - Dual-space hybrid search (text × structural)
//! - rkyv-style snapshot checkpoint via bincode serialization
//!
//! # Architecture
//!
//! ```text
//! HyperspaceEngine trait (async)
//!   └── HyperspaceEngineImpl
//!         ├── EngineWal (WAL)
//!         ├── VectorStore (slot-based storage)
//!         ├── IncrementalHNSW (ANN index)
//!         └── JsonLdMetadataIndex (JSON-LD + RoaringBitmap filters)
//! ```

pub mod engine;
pub mod error;
pub mod filter;
pub mod hnsw;
pub mod hybrid;
pub mod hyper_vector;
pub mod jsonld_meta;
pub mod metric;
pub mod snapshot;
pub mod storage;
pub mod tangent;
pub mod wal;

// Re-exports for convenience
pub use engine::{HyperspaceEngine, HyperspaceEngineImpl, IriRegistry, SearchHit, Searcher};
pub use error::EngineError;
pub use filter::{compile_filter, CompiledFilter, FilterEvaluation, JsonLdFilter};
pub use hnsw::{HnswConfig, IncrementalHNSW, SerializableNode};
pub use hyper_vector::{EmbeddingVector, MetricKind};
pub use jsonld_meta::JsonLdMetadataIndex;
pub use metric::{
    metric_from_kind, CosineMetric, EuclideanMetric, LorentzMetric, Metric, PoincareMetric,
};
pub use snapshot::{load_snapshot, save_snapshot, EngineSnapshot};
pub use storage::VectorStore;
pub use tangent::TangentCache;
pub use wal::{EngineWal, WalOp, WalSyncMode};
