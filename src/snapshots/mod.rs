//! Snapshots — Skill Graph Versioned Snapshots & Temporal Hyperedges.
//!
//! **Experimental** — this module is under active development and its public API
//! may change without notice.
//!
//! Provides:
//! - `TimelineStore`: point-in-time snapshots, rollback, and diff for the skill graph
//! - `TemporalHypergraphStore`: experimental time-aware N-ary hyperedges with
//!   versioning + time-range queries. It is a library component and is not wired
//!   into `SkillGraphStore`, `CausalEngine`, or a production retrieval path.
//!
//! # Architecture
//!
//! ```text
//! SkillGraphStore (via mutation hooks)
//!   ├── TimelineStore
//!   │     ├── Full snapshots (periodic, configurable)
//!   │     ├── Incremental mutations (between snapshots)
//!   │     └── Snapshot index (metadata for list/query)
//!   └── TemporalHypergraphStore
//!         ├── TemporalHyperedge (versioned hyperedges)
//!         └── TemporalIndex (binary-search over sorted time entries)
//! ```
//!
//! # Design Note
//!
//! Named `snapshots` (not `temporal`) because this module is primarily a
//! **versioned snapshot system** for the skill graph. The temporal hyperedge
//! functionality is secondary; it is not a general time-series tensor store.

pub mod timeline;
pub mod types;

pub use timeline::TimelineStore;
pub use types::{
    GraphDiff, GraphMutation, GraphSnapshot, SnapshotMeta, TemporalHyperedge,
    TemporalHypergraphStore, TemporalIndex, TemporalIndexEntry, TemporalVersion, TimeInterval,
};
