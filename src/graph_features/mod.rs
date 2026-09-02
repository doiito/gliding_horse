//! Graph Features — Topological feature extraction + neighborhood aggregation.
//!
//! **Experimental** — this module is under active development and its public API
//! may change without notice.
//!
//! Implements a lightweight **graph feature engine** for the skill graph:
//! - `FeatureExtractor`: converts `SkillGraphNode` + graph topology → numeric feature vectors
//! - `NeighborhoodAggregator`: mean-pooling of neighbor features (simplified GraphSAGE-style aggregation)
//! - `SimilarityEngine`: link prediction via cosine similarity on embeddings
//!
//! # Design Note
//!
//! Named `graph_features` (not `gnn`) because this module performs **graph topology
//! feature extraction and geometric aggregation** — not neural network training
//! or learned-weight backpropagation.

pub mod features;
pub mod similarity;

pub use features::{FeatureExtractor, NeighborhoodAggregator, NodeFeatures};
pub use similarity::{LinkPrediction, SimilarityEngine};
