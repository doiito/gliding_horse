# Advanced Feature Design: Root Cause Analysis, Timeline Versioning, Temporal Hypergraph, GNN

> Status (2026-07-23): this is an architecture/design proposal, not an
> implementation inventory. `TemporalHypergraphStore` exists as an experimental
> library component but is not wired into `SkillGraphStore` or `CausalEngine`.
> No `src/gnn` implementation or production GNN consumer exists in this
> workspace. The code blocks and integration diagrams below therefore describe
> intended work, not current runtime behavior.

> Design document for implementing 4 advanced skill graph features on Gliding Horse Agent OS.
>
> **Status**: Draft for review
> **Target modules**: `src/skill_graph/`, `src/causal/`, `src/temporal/`
> **Existing dependencies**: petgraph 0.6, chrono, serde, uuid, sha2, hyperspace-engine

---

## Overview

This design extends the existing `SkillGraphStore` with 4 integrated features:

```
SkillGraphStore (existing)
  ├── CausalEngine (NEW)    — Root cause analysis
  ├── TimelineStore (NEW)   — Versioned graph snapshots
  ├── TemporalHypergraph (NEW) — Time-aware N-ary edges
  └── GraphNeuralNet (NEW)  — GNN embeddings + inference
```

All 4 features share the same persistence layer (L0Store/redb) and emit/receive events via EventBus.

---

## 1. Root Cause Analysis Engine

### 1.1 Design Goal

Given a set of observed error events, trace backwards through the skill dependency graph to identify the **most probable root cause skill(s)**.

### 1.2 Data Model

```rust
// ── Extending existing CausalChain / SkillCausalModel ──

/// A single error observation with full context
pub struct CausalObservation {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub skill_iri: String,
    pub error_class: String,
    pub error_signature: String,
    pub context: HashMap<String, String>,
    /// Which error propagated into this one (if known)
    pub propagation_from: Option<String>,
    /// Embedding of error_signature for similarity matching
    pub signature_embedding: Option<EmbeddingVector>,
}

/// Inference result from the causal engine
pub struct CausalInference {
    pub root_cause_iri: String,
    pub confidence: f32,
    pub propagation_path: Vec<CausalObservation>,
    pub supporting_evidence: Vec<String>,
    pub alternative_causes: Vec<(String, f32)>,  // (iri, confidence)
}

/// Storage-optimized causal model with persistence
pub struct CausalModelStore {
    // error_signature → [(skill_iri, count)]
    error_index: HashMap<String, Vec<(String, u32)>>,
    // skill_iri → error_signature → count
    error_profiles: HashMap<String, HashMap<String, u32>>,
    // (from, to) → propagation count
    propagation_edges: HashMap<(String, String), u32>,
    // skill_iri → prior probability (based on historical failure rate)
    prior_probability: HashMap<String, f32>,
}
```

### 1.3 Algorithm

```
FUNCTION infer_root_cause(observed_errors: Vec<CausalObservation>)
    → Vec<CausalInference>

1. BUILD subgraph from SkillGraphStore containing all skills reachable
   from observed_errors via prerequisite/extension links (reverse traversal)

2. COMPUTE Bayesian posterior for each node in subgraph:
   P(node_is_root | observed) ∝ P(observed | node_is_root) × P(node)

   Where:
   - P(node) = prior_probability from historical failure rate
   - P(observed | node) = ∏ P(error_i propagates_from node)
     computed by tracing all paths from node to each observed error
     (weighted by propagation_edge counts)

3. RANK nodes by posterior probability → return top-K as root causes

4. For each candidate, reconstruct propagation_path by BFS from
   candidate to each observed error, following most-likely propagation edges
```

### 1.4 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/causal/mod.rs` | CREATE | Module root, re-exports |
| `src/causal/engine.rs` | CREATE | `CausalEngine` — inference algorithm implementation |
| `src/causal/store.rs` | CREATE | `CausalModelStore` — persistence + query |
| `src/causal/types.rs` | CREATE | `CausalObservation`, `CausalInference` types |
| `src/skill_graph/types.rs` | MODIFY | Add `signature_embedding` to existing `CausalEvent` |
| `src/skill_graph/graph_store.rs` | MODIFY | Integrate `CausalEngine` for auto-recording failures |

### 1.5 API

```rust
impl CausalEngine {
    pub fn new(store: Arc<SkillGraphStore>) -> Self;

    /// Record an observed error event
    pub fn record_observation(&self, obs: CausalObservation);

    /// Infer root cause from a batch of observations
    pub fn infer_root_cause(
        &self,
        observations: &[CausalObservation],
        top_k: usize,
    ) -> Vec<CausalInference>;

    /// Get propagation graph as petgraph DiGraph for visualization
    pub fn propagation_graph(&self) -> DiGraph<String, f32>;
}
```

---

## 2. Skill Graph Timeline Versioning

### 2.1 Design Goal

Track every mutation to the skill graph over time, enabling:
- Point-in-time query ("what did the graph look like at timestamp T?")
- Rollback to any historical snapshot
- Diff between two snapshots
- Audit trail for all changes

### 2.2 Data Model

```rust
/// A single change to the graph
#[derive(Serialize, Deserialize)]
pub enum GraphMutation {
    SkillRegistered(SkillGraphNode),
    SkillUpdated { old: SkillGraphNode, new: SkillGraphNode },
    SkillRemoved(SkillGraphNode),
    LinkAdded { source: String, target: String, link_type: SkillLinkType },
    LinkRemoved { source: String, target: String, link_type: SkillLinkType },
    HyperedgeAdded(Hyperedge),
    HyperedgeRemoved(Hyperedge),
    MOCAdded(MOCNode),
    MOCChanged(MOCNode),
}

/// A complete point-in-time graph snapshot
#[derive(Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub snapshot_id: String,
    pub timestamp: DateTime<Utc>,
    pub label: String,
    pub skills: Vec<SkillGraphNode>,
    pub hyperedges: Vec<Hyperedge>,
    pub mocs: Vec<MOCNode>,
    pub fragments: Vec<KnowledgeFragment>,
    pub parent_snapshot_id: Option<String>,
    pub mutation: Option<GraphMutation>,  // what changed since parent
}

/// Diff between two snapshots
pub struct GraphDiff {
    pub from_snapshot: String,
    pub to_snapshot: String,
    pub skills_added: Vec<SkillGraphNode>,
    pub skills_removed: Vec<SkillGraphNode>,
    pub skills_modified: Vec<(SkillGraphNode, SkillGraphNode)>,
    pub hyperedges_added: Vec<Hyperedge>,
    pub hyperedges_removed: Vec<Hyperedge>,
}
```

### 2.3 Storage Strategy

Use **chunked snapshot + write-ahead log** approach:

```
TimelineStore
  ├── Full snapshots: taken every N mutations (configurable, default=100)
  │   Stored as serialized GraphSnapshot in L0 or redb
  ├── Incremental: between full snapshots, store only GraphMutation
  │   Replayed on top of last full snapshot for reconstruction
  └── Index: [snapshot_id → timestamp, parent_id, mutation_count]
```

Snapshots are **copy-on-write**: a snapshot clones the current graph state only when explicitly triggered (e.g., before a risky operation) or periodically. This avoids O(n) per-mutation cost.

### 2.4 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/temporal/mod.rs` | CREATE | Module root |
| `src/temporal/timeline.rs` | CREATE | `TimelineStore` — snapshot creation, rollback, diff |
| `src/temporal/types.rs` | CREATE | `GraphSnapshot`, `GraphMutation`, `GraphDiff` |
| `src/skill_graph/graph_store.rs` | MODIFY | Hook all mutation methods to push to TimelineStore |
| `src/skill_graph/types.rs` | MODIFY | Keep `SnapshotRecord` as lightweight reference |

### 2.5 API

```rust
impl TimelineStore {
    pub fn new(l0: Arc<L0Store>) -> Self;

    /// Record a mutation (called by SkillGraphStore hooks)
    pub fn record_mutation(&self, mutation: GraphMutation);

    /// Create an explicit full snapshot
    pub fn create_snapshot(&self, store: &SkillGraphStore, label: &str) -> String;

    /// List all snapshots
    pub fn list_snapshots(&self) -> Vec<GraphSnapshot>;

    /// Reconstruct graph state at a given snapshot
    pub fn reconstruct(&self, snapshot_id: &str) -> Option<GraphSnapshot>;

    /// Rollback SkillGraphStore to a snapshot
    pub fn rollback(&self, snapshot_id: &str, store: &SkillGraphStore) -> Result<(), Error>;

    /// Diff two snapshots
    pub fn diff(&self, from: &str, to: &str) -> Option<GraphDiff>;

    /// Get all mutations between two snapshots
    pub fn mutation_log(&self, from: &str, to: &str) -> Vec<GraphMutation>;
}
```

---

## 3. Temporal Hypergraph

### 3.1 Design Goal

Extend the existing `Hyperedge` struct to support time-aware N-ary relationships:
- Hyperedges that are only valid during certain time windows
- Time-range queries ("which hyperedges were active on date X?")
- Temporal evolution tracking ("how did this hyperedge change over time")
- Causal constraint inference between temporal hyperedges

### 3.2 Data Model

```rust
/// A hyperedge that exists/evolves over time
#[derive(Serialize, Deserialize)]
pub struct TemporalHyperedge {
    pub hyperedge_id: String,
    pub name: String,
    pub description: String,
    pub components: Vec<String>,
    pub target_composite: Option<String>,
    pub composition_type: CompositionType,
    pub weight: f32,
    pub metadata: HashMap<String, String>,

    // ── Temporal extensions ──
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,  // None = still active
    pub intervals: Vec<TimeInterval>,        // discontinuous active periods
    pub version: u32,
    pub supersedes: Option<String>,          // previous version's hyperedge_id
    pub superseded_by: Option<String>,
}

/// A continuous time interval
#[derive(Serialize, Deserialize)]
pub struct TimeInterval {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub label: Option<String>,
}

/// Temporal index for efficient time-range queries
pub struct TemporalIndex {
    // B-tree-like structure keyed by (timestamp, hyperedge_id)
    // Implemented as sorted Vec<(DateTime<Utc>, String)> with binary search
    // For production: redb table with composite key
    entries: Vec<(DateTime<Utc>, TemporalIndexEntry)>,
}

pub struct TemporalIndexEntry {
    pub hyperedge_id: String,
    pub event_type: TemporalEventType,  // Created, Modified, Activated, Deactivated
}

pub enum TemporalEventType {
    Created,
    Modified,    // metadata/weight changed
    Activated,   // entered a valid_from period
    Deactivated, // passed valid_until
    Superseded,  // replaced by newer version
}
```

### 3.3 Query API

```rust
impl TemporalHypergraphStore {
    pub fn new(l0: Arc<L0Store>) -> Self;

    // ── CRUD ──
    pub fn register_hyperedge(&self, he: TemporalHyperedge);
    pub fn update_hyperedge(&self, he: TemporalHyperedge);  // creates new version
    pub fn deactivate(&self, hyperedge_id: &str, at: DateTime<Utc>);

    // ── Time-range queries ──
    pub fn active_at(&self, instant: DateTime<Utc>) -> Vec<TemporalHyperedge>;
    pub fn active_between(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<TemporalHyperedge>;
    pub fn history_of(&self, hyperedge_id: &str) -> Vec<TemporalHyperedge>;

    // ── Causal constraint inference ──
    /// Find temporally ordered hyperedge pairs where
    /// he1's end ≤ he2's start (potential causal chain)
    pub fn find_causal_candidates(&self) -> Vec<(TemporalHyperedge, TemporalHyperedge)>;
}
```

### 3.4 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/temporal/hypergraph.rs` | CREATE | `TemporalHypergraphStore` + `TemporalIndex` |
| `src/temporal/types.rs` | CREATE | `TemporalHyperedge`, `TimeInterval`, `TemporalEventType` |
| `src/temporal/mod.rs` | MODIFY | Add hypergraph module |
| `src/skill_graph/graph_store.rs` | MODIFY | Hyperedge CRUD delegates to TemporalHypergraphStore |
| `src/skill_graph/types.rs` | MODIFY | `Hyperedge` gets a `version` field |

### 3.5 Integration with CausalEngine

Temporal hyperedges feed into the causal engine:
- `TemporalHypergraphStore::find_causal_candidates()` returns hyperedge pairs with non-overlapping time intervals
- These become prior edges in the `CausalEngine.propagation_graph()`
- If hyperedge A was active before hyperedge B, and both share components, A may be a causal precursor to B

---

## 4. Graph Neural Network (GNN) Integration

### 4.1 Design Goal

Use graph neural networks to compute **learned embeddings** for skill graph nodes, enabling:
- **Node classification**: predict skill category/maturity from graph structure
- **Link prediction**: suggest missing prerequisite/related links
- **Anomaly detection**: flag skills whose embedding diverges from their neighbors
- **Skill recommendation**: find relevant skills for a task via learned similarity

### 4.2 Architecture

```
GraphNeuralNet
  ├── FeatureExtractor — converts SkillGraphNode → feature tensor
  │   Uses existing SkillGraphEmbedder (Poincaré structural embedding)
  │   + skill attributes (tags, success_rate, maturity, node_type)
  │   + neighborhood features (degree, centrality, community)
  ├── GNNModel — graph convolution layers
  │   Implements simplified GCN (Graph Convolutional Network) in pure Rust
  │   2 layers: [Input → Hidden(64) → Output(32)]
  │   Using petgraph adjacency + manual matrix ops
  ├── TrainingEngine — online learning from graph changes
  │   Link prediction objective: maximize P(edge_exists | node_embeddings)
  │   Negative sampling: sample non-edges for contrastive loss
  └── InferenceEngine — embedding + prediction API
```

### 4.3 GCN Implementation (Pure Rust)

Since we cannot depend on PyTorch/TensorFlow in a Rust binary, we implement a **minimal GCN** manually:

```rust
pub struct GraphConvolutionLayer {
    pub weight: Vec<Vec<f32>>,  // [input_dim × output_dim]
    pub bias: Vec<f32>,         // [output_dim]
}

impl GraphConvolutionLayer {
    pub fn forward(
        &self,
        features: &[Vec<f32>],      // [N × input_dim]
        adjacency: &[Vec<f32>],     // [N × N] normalized adjacency
    ) -> Vec<Vec<f32>> {            // [N × output_dim]
        // H = σ(A · X · W + b)
        // where A = D^(-1/2) · (A + I) · D^(-1/2) (symmetric normalization)
        let mut result = mat_mul(adjacency, features);  // [N × input_dim]
        result = mat_mul(&result, &self.weight);         // [N × output_dim]
        result = add_bias(result, &self.bias);           // [N × output_dim]
        result = relu(result);                            // [N × output_dim]
        result
    }
}

pub struct GNNModel {
    pub layer1: GraphConvolutionLayer,  // input_dim → hidden_dim
    pub layer2: GraphConvolutionLayer,  // hidden_dim → output_dim
    pub dropout: f32,
}
```

The GNN is **not trained via SGD at runtime** — instead we use a **geometric embedding alignment** approach:
1. Extract structural features using existing `SkillGraphEmbedder` (Poincaré coordinates)
2. Run 2-3 rounds of **iterative neighborhood aggregation** (mean-pool of neighbor embeddings)
3. The result is a learned embedding that combines structure + attributes

For true training, we support an **offline training mode**:
- Export graph to JSON
- Train in Python with a real GNN framework
- Import learned weights back into the `GraphConvolutionLayer` struct

### 4.4 Feature Extraction

```rust
pub struct NodeFeatures {
    // Structural features (from existing SkillGraphEmbedder)
    pub poincare_coords: [f64; 4],
    // Graph metrics
    pub in_degree: f32,
    pub out_degree: f32,
    pub page_rank_score: f32,
    pub betweenness_score: f32,
    pub community_id: i32,
    // Skill attributes (one-hot or normalized)
    pub node_type: u8,           // Atomic=0, Composite=1, MOC=2, etc.
    pub maturity: f32,           // experimental=0.0, stable=1.0
    pub success_rate: f32,
    pub security_level: u8,
    // Neighborhood features
    pub avg_neighbor_success_rate: f32,
    pub prerequisite_depth: u32,
}
```

### 4.5 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/gnn/mod.rs` | CREATE | Module root |
| `src/gnn/layer.rs` | CREATE | `GraphConvolutionLayer` — forward pass |
| `src/gnn/model.rs` | CREATE | `GNNModel` — 2-layer GCN |
| `src/gnn/features.rs` | CREATE | `FeatureExtractor` — SkillGraphNode → tensor |
| `src/gnn/predict.rs` | CREATE | `LinkPredictor`, `NodeClassifier` |
| `src/gnn/train.rs` | CREATE | `Trainer` — offline training (export/import weights) |
| `src/skill_graph/graph_store.rs` | MODIFY | Call `GNNModel.forward()` after mutations (optional) |

### 4.6 API

```rust
impl GNNModel {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self;

    /// Forward pass: compute embeddings for all nodes
    pub fn forward(&self, features: &NodeFeatures, adj: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Load pre-trained weights (from JSON exported by Python training)
    pub fn load_weights(&mut self, path: &Path) -> Result<(), Error>;

    /// Export current weights for offline training
    pub fn export_weights(&self) -> Value;

    /// Compute link prediction score between two nodes
    pub fn predict_link(&self, emb_i: &[f32], emb_j: &[f32]) -> f32;
}

impl FeatureExtractor {
    pub fn new(store: Arc<SkillGraphStore>, algorithms: Arc<SkillGraphAlgorithms>) -> Self;
    pub fn extract(&self, skill_iri: &str) -> Option<NodeFeatures>;
    pub fn extract_all(&self) -> HashMap<String, NodeFeatures>;
}
```

---

## 5. Cross-Feature Integration

```
                    ┌─────────────────────────────────┐
                    │         EventBus                 │
                    │  emits: CAUSAL_EVENT,            │
                    │  GRAPH_MUTATION, TEMPORAL_EVENT  │
                    └──────┬──────────────┬────────────┘
                           │              │
              ┌────────────▼────┐  ┌──────▼───────────┐
              │  CausalEngine   │  │  TimelineStore   │
              │  subscribes to  │  │  subscribes to   │
              │  error events   │  │  graph mutations │
              └────────▲───────┘  └──────▲────────────┘
                       │                 │
              ┌────────┴─────────────────┴──────────┐
              │         SkillGraphStore              │
              │  (all mutations hook both engines)   │
              └────────┬────────────────────────────┘
                       │
              ┌────────▼──────────┐  ┌───────────────┐
              │TemporalHypergraph │  │    GNNModel    │
              │(time-aware edges) │  │(embeddings    │
              │ feeds CausalEngine│  │ + predictions)│
              └───────────────────┘  └───────────────┘
```

### 5.1 Event Flow

1. **Agent execution fails** → `EventBus` emits error event
2. **CausalEngine** subscribes → records `CausalObservation`
3. **SkillGraphStore mutation** (skill registered/updated) → hooks push to:
   - `TimelineStore.record_mutation()` (versioning)
   - `TemporalHypergraph.update_hyperedge()` if hyperedge changed
4. **On demand**: `CausalEngine.infer_root_cause()` queries `TemporalHypergraphStore.find_causal_candidates()` for temporal priors
5. **GNNModel** consumes `SkillGraphStore` snapshot + `SkillGraphAlgorithms` metrics for embedding computation

---

## 6. Persistence Strategy

| Component | Primary Storage | Backup/Index |
|-----------|----------------|--------------|
| CausalModelStore | L0Store (redb) | In-memory HashMap for hot paths |
| TimelineStore | redb (mutations) + L0 (full snapshots) | — |
| TemporalHypergraphStore | L0Store | In-memory TemporalIndex |
| GNNModel | JSON weights file (occasional) | Computed at startup from graph |

---

## 7. Implementation Order

```
Phase 1: Core Infrastructure
  ├── 1a. causal/types.rs + causal/mod.rs
  ├── 1b. temporal/types.rs + temporal/mod.rs
  └── 1c. temporal/timeline.rs (TimelineStore skeleton)

Phase 2: Causal Engine
  ├── 2a. causal/store.rs (CausalModelStore — persistence)
  ├── 2b. causal/engine.rs (CausalEngine — Bayesian inference)
  └── 2c. SkillGraphStore integration hooks

Phase 3: Temporal Hypergraph
  ├── 3a. temporal/hypergraph.rs (TemporalHypergraphStore)
  ├── 3b. TemporalIndex implementation
  └── 3c. CausalEngine integration (find_causal_candidates)

Phase 4: GNN
  ├── 4a. gnn/features.rs (FeatureExtractor)
  ├── 4b. gnn/layer.rs (GraphConvolutionLayer)
  ├── 4c. gnn/model.rs + gnn/predict.rs
  └── 4d. SkillGraphStore integration
```

---

## 8. Testing Strategy

| Feature | Unit Tests | Integration Tests |
|---------|-----------|-------------------|
| CausalEngine | Inference correctness with synthetic graphs | Full pipeline: observe → infer → verify |
| TimelineStore | Snapshot/create/rollback/diff | Rollback + verify graph state matches |
| TemporalHypergraph | Time-range queries, versioning | Causal candidate generation |
| GNN | Forward pass correctness, feature extraction | Export format roundtrip |
