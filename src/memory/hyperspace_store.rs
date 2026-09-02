use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use hyperspace_engine::engine::{HyperspaceEngine, HyperspaceEngineImpl, SearchHit};
use hyperspace_engine::filter::JsonLdFilter;
use hyperspace_engine::hnsw::HnswConfig;
use hyperspace_engine::hyper_vector::{EmbeddingVector, MetricKind};
use hyperspace_engine::metric::CosineMetric;
use hyperspace_engine::wal::WalSyncMode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{debug, info};

use crate::memory::embedding_service::EmbeddingService;
use crate::CoreError;

/// Search filter combining tag matching, type filtering, and importance range.
///
/// Mirrors the original qdrant-based filter interface for backward compatibility
/// while mapping cleanly to `JsonLdFilter` for HyperspaceEngine.
#[derive(Debug, Clone, Default)]
pub struct HybridSearchFilter {
    pub must_tags: Vec<String>,
    pub should_tags: Vec<String>,
    pub must_not_tags: Vec<String>,
    pub min_importance: Option<f32>,
    pub jsonld_types: Vec<String>,
    pub named_graph: Option<String>,
    /// Only return entries stored after this Unix timestamp (seconds)
    pub created_after: Option<f64>,
    /// Only return entries stored before this Unix timestamp (seconds)
    pub created_before: Option<f64>,
}

impl HybridSearchFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_must_tags(mut self, tags: Vec<String>) -> Self {
        self.must_tags = tags;
        self
    }

    /// Add tags that boost ranking when present (non-exclusive).
    ///
    /// Unlike `with_must_tags`, entries are not excluded when a should-tag is
    /// absent — the tag only contributes to the match score.
    ///
    /// ```
    /// use glidinghorse::memory::hyperspace_store::HybridSearchFilter;
    ///
    /// let filter = HybridSearchFilter::new()
    ///     .with_must_tags(vec!["experience".into()])
    ///     .with_should_tags(vec!["urgent".into(), "critical".into()]);
    /// ```
    pub fn with_should_tags(mut self, tags: Vec<String>) -> Self {
        self.should_tags = tags;
        self
    }

    /// Exclude entries carrying any of these tags.
    ///
    /// ```
    /// use glidinghorse::memory::hyperspace_store::HybridSearchFilter;
    ///
    /// let filter = HybridSearchFilter::new()
    ///     .with_must_not_tags(vec!["archived".into()]);
    /// ```
    pub fn with_must_not_tags(mut self, tags: Vec<String>) -> Self {
        self.must_not_tags = tags;
        self
    }

    pub fn with_min_importance(mut self, min: f32) -> Self {
        self.min_importance = Some(min);
        self
    }

    pub fn with_jsonld_types(mut self, types: Vec<String>) -> Self {
        self.jsonld_types = types;
        self
    }

    /// Restrict the search to a single named graph (knowledge-graph scoping).
    ///
    /// Entries outside `graph` are excluded entirely.
    ///
    /// ```
    /// use glidinghorse::memory::hyperspace_store::HybridSearchFilter;
    ///
    /// let filter = HybridSearchFilter::new().with_named_graph("skill_graph".into());
    /// ```
    pub fn with_named_graph(mut self, graph: String) -> Self {
        self.named_graph = Some(graph);
        self
    }

    /// Filter to only entries stored after this Unix timestamp (seconds)
    pub fn with_created_after(mut self, timestamp: f64) -> Self {
        self.created_after = Some(timestamp);
        self
    }

    /// Filter to only entries stored before this Unix timestamp (seconds)
    pub fn with_created_before(mut self, timestamp: f64) -> Self {
        self.created_before = Some(timestamp);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.must_tags.is_empty()
            && self.should_tags.is_empty()
            && self.must_not_tags.is_empty()
            && self.min_importance.is_none()
            && self.jsonld_types.is_empty()
            && self.named_graph.is_none()
            && self.created_after.is_none()
            && self.created_before.is_none()
    }
}

/// Single search result from the vector store.
#[derive(Debug, Clone)]
pub struct ScoredEntry {
    pub iri: String,
    pub text: String,
    pub score: f32,
    pub tags: Vec<String>,
    pub importance: Option<f32>,
    pub jsonld_types: Vec<String>,
    /// Unix timestamp (seconds) when this entry was stored
    pub stored_at: Option<f64>,
    /// Permanent audit/knowledge evidence must not lose rank merely because
    /// time passes. This flag is set only by explicit recallable upsert APIs.
    pub decay_exempt: bool,
}

/// Bounded, deterministic configuration for an offline/operator ANN quality
/// probe. It does not mutate the index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnnHealthProbeConfig {
    pub sample_size: usize,
    pub top_k: usize,
    pub sample_seed: u64,
    pub max_scan_entries: usize,
    pub min_recall_at_k: f64,
    pub max_candidate_shortfall_rate: f64,
    pub metadata_vacuum_tombstone_ratio: f64,
    pub checkpoint_wal_bytes: u64,
}

impl Default for AnnHealthProbeConfig {
    fn default() -> Self {
        Self {
            sample_size: 32,
            top_k: 10,
            sample_seed: 0x4748_414e_4e50_524f,
            max_scan_entries: 50_000,
            min_recall_at_k: 0.95,
            max_candidate_shortfall_rate: 0.02,
            metadata_vacuum_tombstone_ratio: 0.20,
            checkpoint_wal_bytes: 64 * 1024 * 1024,
        }
    }
}

impl AnnHealthProbeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_size == 0
            || self.sample_size > 1024
            || self.top_k == 0
            || self.top_k > 100
            || self.max_scan_entries == 0
            || self.max_scan_entries > 100_000
            || !self.min_recall_at_k.is_finite()
            || !(0.0..=1.0).contains(&self.min_recall_at_k)
            || !self.max_candidate_shortfall_rate.is_finite()
            || !(0.0..=1.0).contains(&self.max_candidate_shortfall_rate)
            || !self.metadata_vacuum_tombstone_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.metadata_vacuum_tombstone_ratio)
        {
            return Err("ANN health probe configuration is invalid".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnMaintenanceRecommendation {
    Empty,
    Healthy,
    CheckpointRecommended,
    MetadataVacuumRecommended,
    ReindexRequired,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnnHealthReport {
    pub schema_version: u32,
    pub created_at: chrono::DateTime<Utc>,
    pub embedding_provider: String,
    pub sample_seed: u64,
    pub samples_evaluated: u32,
    pub top_k: u32,
    pub mean_recall_at_k: f64,
    pub candidate_shortfall_rate: f64,
    pub ann_p50_us: u64,
    pub ann_p95_us: u64,
    pub exact_p50_us: u64,
    pub exact_p95_us: u64,
    pub active_vectors: u32,
    pub allocated_slots: u32,
    pub tombstone_slots: u32,
    pub active_wal_bytes: u64,
    pub recommendation: AnnMaintenanceRecommendation,
}

pub const ANN_HEALTH_REPORT_SCHEMA_VERSION: u32 = 1;

/// In-memory vector store backed by HyperspaceEngine.
///
/// Replaces the old Qdrant-based VectorStore. Wraps `HyperspaceEngineImpl`
/// for HNSW ANN search + `Arc<dyn EmbeddingService>` for text→vector conversion.
/// All public methods mirror the old API so callers (ProjectionEngine,
/// SkillDiscoveryEngine) work with minimal changes.
pub struct HyperspaceStore {
    engine: Arc<HyperspaceEngineImpl>,
    embed: Arc<dyn EmbeddingService>,
}

impl HyperspaceStore {
    /// Open or create a HyperspaceEngine-backed vector store.
    ///
    /// `data_dir` — persistent storage directory (WAL + snapshots + HNSW index).
    /// `embed` — embedding service that determines the vector dimension.
    pub fn open(data_dir: &Path, embed: Arc<dyn EmbeddingService>) -> Result<Self, CoreError> {
        let dim = embed.dimension();
        let engine = HyperspaceEngineImpl::open(
            data_dir,
            WalSyncMode::Batch { interval_ms: 100 },
            dim,
            Box::new(CosineMetric),
            HnswConfig::default(),
        )
        .map_err(|e| CoreError::Internal {
            message: format!("HyperspaceEngine init: {e}"),
        })?;
        info!(dim = dim, "HyperspaceEngine opened");
        Ok(Self {
            engine: Arc::new(engine),
            embed,
        })
    }

    /// Stable provider identifier of the backing embedding service
    /// ("oneapi" | "ollama" | "fallback" | "unknown").
    ///
    /// Vector recall is only meaningful when the embedding backend is a real
    /// provider; the hash-based fallback produces vectors that cannot be
    /// compared semantically across entries.
    pub fn embedding_provider(&self) -> &'static str {
        self.embed.provider()
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Embed text. A failed embedding must not be indexed as a zero vector,
    /// because that fabricates a semantic point and contaminates retrieval.
    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        self.embed
            .embed(text)
            .await
            .map_err(|error| CoreError::Internal {
                message: format!("Embedding service failed: {}", error),
            })
    }

    /// Convert a `HybridSearchFilter` to a `Vec<JsonLdFilter>` for the engine.
    ///
    /// Semantics matches the original Qdrant filter:
    /// - `must_tags` / `jsonld_types` / `named_graph` / `min_importance` → ANDed together
    /// - `should_tags` → OR (at least one must match)
    /// - `must_not_tags` → NOT (none must match)
    fn to_jsonld_filters(&self, filter: &HybridSearchFilter) -> Vec<JsonLdFilter> {
        if filter.is_empty() {
            return vec![];
        }

        let mut engine_filters: Vec<JsonLdFilter> = Vec::new();

        // Must group (AND of all must conditions)
        let mut must_children: Vec<JsonLdFilter> = Vec::new();
        for tag in &filter.must_tags {
            must_children.push(JsonLdFilter::tag("tags", tag));
        }
        for type_iri in &filter.jsonld_types {
            must_children.push(JsonLdFilter::Type(type_iri.clone()));
        }
        if let Some(ref graph) = filter.named_graph {
            must_children.push(JsonLdFilter::NamedGraph(graph.clone()));
        }
        if let Some(min) = filter.min_importance {
            must_children.push(JsonLdFilter::Range {
                key: "importance".into(),
                gte: Some(min as f64),
                lte: None,
            });
        }
        if let Some(after) = filter.created_after {
            must_children.push(JsonLdFilter::Range {
                key: "stored_at".into(),
                gte: Some(after),
                lte: None,
            });
        }
        if let Some(before) = filter.created_before {
            must_children.push(JsonLdFilter::Range {
                key: "stored_at".into(),
                gte: None,
                lte: Some(before),
            });
        }
        if !must_children.is_empty() {
            engine_filters.push(JsonLdFilter::Must(must_children));
        }

        // Should group (OR — at least one should match)
        if !filter.should_tags.is_empty() {
            let should_children: Vec<JsonLdFilter> = filter
                .should_tags
                .iter()
                .map(|t| JsonLdFilter::tag("tags", t))
                .collect();
            engine_filters.push(JsonLdFilter::Should(should_children));
        }

        // MustNot group (NONE must match)
        if !filter.must_not_tags.is_empty() {
            let must_not_children: Vec<JsonLdFilter> = filter
                .must_not_tags
                .iter()
                .map(|t| JsonLdFilter::tag("tags", t))
                .collect();
            engine_filters.push(JsonLdFilter::MustNot(must_not_children));
        }

        engine_filters
    }

    /// Convert engine `SearchHit`s into `ScoredEntry`s (extracting payload fields).
    fn scored_hits_to_entries(hits: Vec<SearchHit>) -> Vec<ScoredEntry> {
        hits.into_iter()
            .map(|hit| {
                let (text, tags, importance, jsonld_types, stored_at, decay_exempt) = hit
                    .payload
                    .as_ref()
                    .map(|p| {
                        let text = p
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let tags = p
                            .get("tags")
                            .and_then(|v| v.as_array())
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        let importance = p
                            .get("importance")
                            .and_then(|v| v.as_f64().map(|f| f as f32));
                        let jsonld_types = p
                            .get("@type")
                            .and_then(|v| v.as_array())
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        let stored_at = p.get("stored_at").and_then(|v| v.as_f64());
                        let decay_exempt = p
                            .get("_gh_recall_decay_exempt")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        (
                            text,
                            tags,
                            importance,
                            jsonld_types,
                            stored_at,
                            decay_exempt,
                        )
                    })
                    .unwrap_or_default();

                ScoredEntry {
                    iri: hit.iri,
                    text,
                    // The engine exposes negative distance for ranking. Public
                    // store scores are positive relevance, so decay always
                    // penalises older entries instead of reversing their rank.
                    score: 1.0 / (1.0 + (-hit.score).max(0.0)),
                    tags,
                    importance,
                    jsonld_types,
                    stored_at,
                    decay_exempt,
                }
            })
            .collect()
    }

    // ── Public API (mirrors old VectorStore) ─────────────────────────────────

    /// Store a vector entry by IRI, embedding its text content.
    pub async fn upsert(&self, iri: &str, text: &str, tags: &[String]) -> Result<u32, CoreError> {
        self.upsert_with_metadata(iri, text, tags, None, None, None)
            .await
    }

    /// Store a vector entry with full metadata.
    pub async fn upsert_with_metadata(
        &self,
        iri: &str,
        text: &str,
        tags: &[String],
        importance: Option<f32>,
        jsonld_types: Option<&[String]>,
        named_graph: Option<&str>,
    ) -> Result<u32, CoreError> {
        self.upsert_with_recall_metadata(
            iri,
            text,
            tags,
            importance,
            jsonld_types,
            named_graph,
            false,
            false,
        )
        .await
    }

    /// Explicitly index a bounded, already-approved memory record for lexical
    /// recall. Generic vector upserts deliberately do *not* opt in, preventing
    /// raw prompts, tool payloads and telemetry from becoming keyword-searchable.
    pub async fn upsert_recallable_with_metadata(
        &self,
        iri: &str,
        text: &str,
        tags: &[String],
        importance: Option<f32>,
        jsonld_types: Option<&[String]>,
        named_graph: Option<&str>,
        decay_exempt: bool,
    ) -> Result<u32, CoreError> {
        self.upsert_with_recall_metadata(
            iri,
            text,
            tags,
            importance,
            jsonld_types,
            named_graph,
            true,
            decay_exempt,
        )
        .await
    }

    async fn upsert_with_recall_metadata(
        &self,
        iri: &str,
        text: &str,
        tags: &[String],
        importance: Option<f32>,
        jsonld_types: Option<&[String]>,
        named_graph: Option<&str>,
        lexical_recall: bool,
        decay_exempt: bool,
    ) -> Result<u32, CoreError> {
        let vector = self.get_embedding(text).await?;
        let vec = EmbeddingVector::from_f32_slice(&vector, MetricKind::Cosine).map_err(|e| {
            CoreError::Internal {
                message: format!("EmbeddingVector: {e}"),
            }
        })?;

        let mut payload = serde_json::Map::new();
        payload.insert("iri".into(), Value::String(iri.into()));
        // Store current Unix timestamp for time-based filtering
        let now_ts = Utc::now().timestamp() as f64;
        payload.insert(
            "stored_at".into(),
            Value::Number(
                serde_json::Number::from_f64(now_ts).unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
        payload.insert(
            "text".into(),
            Value::String(text.chars().take(500).collect()),
        );
        payload.insert(
            "tags".into(),
            Value::Array(tags.iter().map(|t| Value::String(t.clone())).collect()),
        );

        if let Some(imp) = importance {
            payload.insert(
                "importance".into(),
                Value::Number(
                    serde_json::Number::from_f64(imp as f64)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );
        }
        if let Some(types) = jsonld_types {
            payload.insert(
                "@type".into(),
                Value::Array(types.iter().map(|t| Value::String(t.clone())).collect()),
            );
        }
        if let Some(graph) = named_graph {
            payload.insert("named_graph".into(), Value::String(graph.to_string()));
        }
        if lexical_recall {
            payload.insert("_gh_lexical_recall".into(), Value::Bool(true));
            payload.insert("_gh_recall_decay_exempt".into(), Value::Bool(decay_exempt));
        }

        let point_id = self
            .engine
            .upsert(iri, vec, Value::Object(payload))
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Hyperspace upsert: {e}"),
            })?;

        debug!(iri = %iri, point_id = point_id, "Vector stored via HyperspaceEngine");
        Ok(point_id)
    }

    /// Semantic search by query string.
    pub async fn search(&self, query: &str, limit: u64) -> Result<Vec<ScoredEntry>, CoreError> {
        self.search_with_filter(query, &HybridSearchFilter::new(), limit)
            .await
    }

    /// Semantic search with metadata filters.
    pub async fn search_with_filter(
        &self,
        query: &str,
        filter: &HybridSearchFilter,
        limit: u64,
    ) -> Result<Vec<ScoredEntry>, CoreError> {
        let vector = self.get_embedding(query).await?;
        let vec = EmbeddingVector::from_f32_slice(&vector, MetricKind::Cosine).map_err(|e| {
            CoreError::Internal {
                message: format!("EmbeddingVector: {e}"),
            }
        })?;

        let filters = self.to_jsonld_filters(filter);
        let results = self
            .engine
            .search(&vec, limit as usize, &filters)
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Hyperspace search: {e}"),
            })?;

        Ok(Self::scored_hits_to_entries(results))
    }

    /// Search with exponential time-decay applied to scores.
    ///
    /// After fetching results, each entry's score is multiplied by
    /// `exp(-λ * hours_since_stored)` — older entries are penalised.
    /// The results are then re-sorted by the new score.
    ///
    /// `decay_lambda = 0.0` → no decay (identical to `search_with_filter`).
    pub async fn search_with_time_decay(
        &self,
        query: &str,
        filter: &HybridSearchFilter,
        decay_lambda: f64,
        limit: u64,
    ) -> Result<Vec<ScoredEntry>, CoreError> {
        let mut results = self.search_with_filter(query, filter, limit).await?;
        let now = Utc::now();
        for entry in results.iter_mut() {
            if !entry.decay_exempt {
                if let Some(stored_at) = entry.stored_at {
                    let age_secs = now.timestamp() as f64 - stored_at;
                    let age_hours = age_secs / 3600.0;
                    if age_hours > 0.0 {
                        entry.score *= (-decay_lambda * age_hours).exp() as f32;
                    }
                }
            }
        }
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Deterministically fuse bounded semantic and explicitly whitelisted
    /// lexical candidates using reciprocal-rank fusion. The method is safe to
    /// run in shadow mode because it has no write side effect.
    pub async fn search_fused_recall(
        &self,
        query: &str,
        filter: &HybridSearchFilter,
        decay_lambda: f64,
        limit: u64,
    ) -> Result<Vec<ScoredEntry>, CoreError> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let candidate_limit = limit.saturating_mul(3).clamp(10, 256);
        let semantic = if self.embedding_provider() == "fallback" {
            Vec::new()
        } else {
            self.search_with_filter(query, filter, candidate_limit)
                .await?
        };
        let filters = self.to_jsonld_filters(filter);
        let lexical = self
            .engine
            .lexical_search(query, candidate_limit as usize, &filters)
            .map(Self::scored_hits_to_entries)
            .map_err(|error| CoreError::Internal {
                message: format!("Hyperspace lexical search: {error}"),
            })?;

        const RRF_K: f32 = 60.0;
        let mut fused: HashMap<String, (ScoredEntry, f32)> = HashMap::new();
        for (rank, entry) in semantic.into_iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
            let candidate = fused.entry(entry.iri.clone()).or_insert((entry, 0.0));
            candidate.1 += contribution;
        }
        for (rank, entry) in lexical.into_iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
            let candidate = fused.entry(entry.iri.clone()).or_insert((entry, 0.0));
            candidate.1 += contribution;
        }
        let now = Utc::now();
        let mut results = fused
            .into_values()
            .map(|(mut entry, score)| {
                entry.score = score;
                if !entry.decay_exempt {
                    if let Some(stored_at) = entry.stored_at {
                        let age_hours = (now.timestamp() as f64 - stored_at) / 3600.0;
                        if age_hours > 0.0 {
                            entry.score *= (-decay_lambda * age_hours).exp() as f32;
                        }
                    }
                }
                entry
            })
            .collect::<Vec<_>>();
        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.iri.cmp(&right.iri))
        });
        results.truncate(limit as usize);
        Ok(results)
    }

    /// Run a read-only exact-vs-ANN quality probe. The sampled query text is
    /// used only transiently for embeddings and is never returned or stored by
    /// this method; reports contain aggregate metrics only.
    pub async fn assess_ann_health(
        &self,
        config: &AnnHealthProbeConfig,
    ) -> Result<AnnHealthReport, CoreError> {
        config
            .validate()
            .map_err(|message| CoreError::Internal { message })?;
        if self.embedding_provider() == "fallback" {
            return Err(CoreError::Internal {
                message: "ANN health probe requires a semantic embedding provider".to_string(),
            });
        }
        let stats = self.engine.ann_index_stats();
        let count = self.count().await? as usize;
        if count == 0 {
            return Ok(AnnHealthReport {
                schema_version: ANN_HEALTH_REPORT_SCHEMA_VERSION,
                created_at: Utc::now(),
                embedding_provider: self.embedding_provider().to_string(),
                sample_seed: config.sample_seed,
                samples_evaluated: 0,
                top_k: config.top_k as u32,
                mean_recall_at_k: 1.0,
                candidate_shortfall_rate: 0.0,
                ann_p50_us: 0,
                ann_p95_us: 0,
                exact_p50_us: 0,
                exact_p95_us: 0,
                active_vectors: stats.active_vectors,
                allocated_slots: stats.allocated_slots,
                tombstone_slots: stats.tombstone_slots,
                active_wal_bytes: stats.active_wal_bytes,
                recommendation: AnnMaintenanceRecommendation::Empty,
            });
        }
        let scan_limit = count.min(config.max_scan_entries);
        let entries =
            self.engine
                .list(0, scan_limit)
                .await
                .map_err(|error| CoreError::Internal {
                    message: format!("Hyperspace ANN probe list: {error}"),
                })?;
        let mut candidates = entries
            .into_iter()
            .filter_map(|entry| {
                let text = entry
                    .payload
                    .as_ref()
                    .and_then(|payload| payload.get("text"))
                    .and_then(Value::as_str)?;
                (!text.trim().is_empty()).then(|| {
                    let mut hasher = Sha256::new();
                    hasher.update(config.sample_seed.to_be_bytes());
                    hasher.update(entry.iri.as_bytes());
                    let digest = hasher.finalize();
                    let mut score_bytes = [0u8; 8];
                    score_bytes.copy_from_slice(&digest[..8]);
                    (u64::from_be_bytes(score_bytes), entry.iri, text.to_string())
                })
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        candidates.truncate(config.sample_size);

        let mut recalls = Vec::with_capacity(candidates.len());
        let mut shortfalls = 0usize;
        let mut ann_elapsed = Vec::with_capacity(candidates.len());
        let mut exact_elapsed = Vec::with_capacity(candidates.len());
        for (_, _, text) in candidates {
            let vector = self.get_embedding(&text).await?;
            let query =
                EmbeddingVector::from_f32_slice(&vector, MetricKind::Cosine).map_err(|error| {
                    CoreError::Internal {
                        message: format!("ANN health probe embedding vector: {error}"),
                    }
                })?;
            let probe = self
                .engine
                .probe_ann(&query, config.top_k, &[])
                .map_err(|error| CoreError::Internal {
                    message: format!("Hyperspace ANN probe: {error}"),
                })?;
            let expected = probe.exact_ids.len();
            let overlap = probe
                .ann_ids
                .iter()
                .filter(|id| probe.exact_ids.contains(id))
                .count();
            recalls.push(if expected == 0 {
                1.0
            } else {
                overlap as f64 / expected as f64
            });
            if probe.ann_ids.len() < expected {
                shortfalls = shortfalls.saturating_add(1);
            }
            ann_elapsed.push(probe.ann_elapsed_us);
            exact_elapsed.push(probe.exact_elapsed_us);
        }
        let samples_evaluated = recalls.len();
        let mean_recall_at_k = if samples_evaluated == 0 {
            1.0
        } else {
            recalls.iter().sum::<f64>() / samples_evaluated as f64
        };
        let candidate_shortfall_rate = if samples_evaluated == 0 {
            0.0
        } else {
            shortfalls as f64 / samples_evaluated as f64
        };
        let tombstone_ratio = if stats.allocated_slots == 0 {
            0.0
        } else {
            stats.tombstone_slots as f64 / stats.allocated_slots as f64
        };
        let recommendation = if samples_evaluated == 0 {
            AnnMaintenanceRecommendation::Empty
        } else if mean_recall_at_k < config.min_recall_at_k
            || candidate_shortfall_rate > config.max_candidate_shortfall_rate
        {
            AnnMaintenanceRecommendation::ReindexRequired
        } else if tombstone_ratio >= config.metadata_vacuum_tombstone_ratio {
            AnnMaintenanceRecommendation::MetadataVacuumRecommended
        } else if stats.active_wal_bytes >= config.checkpoint_wal_bytes {
            AnnMaintenanceRecommendation::CheckpointRecommended
        } else {
            AnnMaintenanceRecommendation::Healthy
        };
        Ok(AnnHealthReport {
            schema_version: ANN_HEALTH_REPORT_SCHEMA_VERSION,
            created_at: Utc::now(),
            embedding_provider: self.embedding_provider().to_string(),
            sample_seed: config.sample_seed,
            samples_evaluated: samples_evaluated as u32,
            top_k: config.top_k as u32,
            mean_recall_at_k,
            candidate_shortfall_rate,
            ann_p50_us: percentile_us(&mut ann_elapsed, 0.50),
            ann_p95_us: percentile_us(&mut ann_elapsed, 0.95),
            exact_p50_us: percentile_us(&mut exact_elapsed, 0.50),
            exact_p95_us: percentile_us(&mut exact_elapsed, 0.95),
            active_vectors: stats.active_vectors,
            allocated_slots: stats.allocated_slots,
            tombstone_slots: stats.tombstone_slots,
            active_wal_bytes: stats.active_wal_bytes,
            recommendation,
        })
    }

    /// Search by tag match (uses combined tag string as query).
    ///
    /// ```
    /// use glidinghorse::memory::hyperspace_store::HyperspaceStore;
    ///
    /// # async fn example(store: &HyperspaceStore) {
    /// let entries = store
    ///     .search_by_tags(&["experience".into(), "planning".into()], 5)
    ///     .await
    ///     .expect("tag search failed");
    /// # }
    /// ```
    pub async fn search_by_tags(
        &self,
        tags: &[String],
        limit: u64,
    ) -> Result<Vec<ScoredEntry>, CoreError> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }
        let query = tags.join(" ");
        let filter = HybridSearchFilter::new().with_must_tags(tags.to_vec());
        self.search_with_filter(&query, &filter, limit).await
    }

    /// Hybrid search combining free-text and tag filtering.
    pub async fn hybrid_search(
        &self,
        query: &str,
        must_tags: &[String],
        should_tags: &[String],
        min_importance: Option<f32>,
        limit: u64,
    ) -> Result<Vec<ScoredEntry>, CoreError> {
        let mut filter = HybridSearchFilter::new()
            .with_must_tags(must_tags.to_vec())
            .with_should_tags(should_tags.to_vec());
        if let Some(min) = min_importance {
            filter = filter.with_min_importance(min);
        }
        self.search_with_filter(query, &filter, limit).await
    }

    /// Delete a vector entry by IRI.
    pub async fn delete(&self, iri: &str) -> Result<(), CoreError> {
        self.engine
            .delete(iri)
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Hyperspace delete: {e}"),
            })?;
        Ok(())
    }

    /// Total number of indexed entries.
    ///
    /// ```
    /// use glidinghorse::memory::hyperspace_store::HyperspaceStore;
    ///
    /// # async fn example(store: &HyperspaceStore) {
    /// let n = store.count().await.expect("count failed");
    /// # }
    /// ```
    pub async fn count(&self) -> Result<u64, CoreError> {
        self.engine.count().await.map_err(|e| CoreError::Internal {
            message: format!("Hyperspace count: {e}"),
        })
    }

    /// Resolve an IRI to its numeric point ID (if indexed).
    ///
    /// ```
    /// use glidinghorse::memory::hyperspace_store::HyperspaceStore;
    ///
    /// # async fn example(store: &HyperspaceStore) {
    /// if let Some(id) = store.resolve_iri("task:abc").await.expect("resolve failed") {
    ///     // `id` can be passed to lookup_id for the reverse mapping.
    /// }
    /// # }
    /// ```
    pub async fn resolve_iri(&self, iri: &str) -> Result<Option<u32>, CoreError> {
        self.engine
            .resolve_iri(iri)
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Hyperspace resolve_iri: {e}"),
            })
    }

    /// Look up the IRI for a numeric point ID (reverse of resolve_iri).
    ///
    /// ```
    /// use glidinghorse::memory::hyperspace_store::HyperspaceStore;
    ///
    /// # async fn example(store: &HyperspaceStore) {
    /// let iri = store.lookup_id(42).await.expect("lookup failed");
    /// # }
    /// ```
    pub async fn lookup_id(&self, id: u32) -> Result<Option<String>, CoreError> {
        self.engine
            .lookup_id(id)
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Hyperspace lookup_id: {e}"),
            })
    }

    /// Persist a snapshot and rotate/clean the WAL so it cannot grow unbounded.
    ///
    /// Call periodically (or after tasks) to bound WAL replay time on restart.
    pub async fn checkpoint(&self) -> Result<(), CoreError> {
        self.engine
            .checkpoint()
            .await
            .map_err(|e| CoreError::Internal {
                message: format!("Hyperspace checkpoint: {e}"),
            })
    }

    /// Reclaim metadata index space for tombstoned/deleted entries.
    pub async fn vacuum(&self) -> Result<(), CoreError> {
        self.engine.vacuum().await.map_err(|e| CoreError::Internal {
            message: format!("Hyperspace vacuum: {e}"),
        })
    }
}

fn percentile_us(values: &mut [u64], percentile: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index.min(values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embedding_service::FallbackEmbeddingService;
    use async_trait::async_trait;

    fn setup_store() -> HyperspaceStore {
        let dir = tempfile::tempdir().unwrap();
        let embed = Arc::new(FallbackEmbeddingService::new());
        HyperspaceStore::open(dir.path(), embed).unwrap()
    }

    struct FailingEmbedding;
    #[async_trait]
    impl EmbeddingService for FailingEmbedding {
        async fn embed(&self, _: &str) -> Result<Vec<f32>, String> {
            Err("offline".to_string())
        }
        fn dimension(&self) -> usize {
            4
        }
    }

    struct DeterministicEmbedding;

    #[async_trait]
    impl EmbeddingService for DeterministicEmbedding {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
            let mut values = vec![0.0; 4];
            let dimension = values.len();
            for (index, byte) in text.bytes().enumerate() {
                values[index % dimension] += byte as f32;
            }
            let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in &mut values {
                    *value /= norm;
                }
            }
            Ok(values)
        }

        fn dimension(&self) -> usize {
            4
        }

        fn provider(&self) -> &'static str {
            "deterministic-test"
        }
    }

    #[tokio::test]
    async fn embedding_failure_does_not_insert_zero_vector() {
        let dir = tempfile::tempdir().unwrap();
        let store = HyperspaceStore::open(dir.path(), Arc::new(FailingEmbedding)).unwrap();
        assert!(store.upsert("failed", "text", &[]).await.is_err());
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_upsert_and_count() {
        let store = setup_store();
        store.upsert("v:1", "hello world", &[]).await.unwrap();
        store.upsert("v:2", "foo bar baz", &[]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_search_returns_results() {
        let store = setup_store();
        store
            .upsert("s:1", "rust async programming", &[])
            .await
            .unwrap();
        store
            .upsert("s:2", "python web framework", &[])
            .await
            .unwrap();

        let results = store.search("programming", 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_empty_store() {
        let store = setup_store();
        let results = store.search("nothing", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_delete() {
        let store = setup_store();
        store.upsert("d:1", "delete me", &[]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
        store.delete("d:1").await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_delete_nonexistent_returns_error() {
        let store = setup_store();
        let result = store.delete("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_by_tags() {
        let store = setup_store();
        store
            .upsert("t:1", "rust code", &["lang:rust".into()])
            .await
            .unwrap();
        store
            .upsert("t:2", "python code", &["lang:python".into()])
            .await
            .unwrap();

        let results = store
            .search_by_tags(&["lang:rust".into()], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "t:1");
    }

    #[tokio::test]
    async fn test_search_by_tags_empty() {
        let store = setup_store();
        let results = store.search_by_tags(&[], 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_with_filter_importance() {
        let store = setup_store();
        store
            .upsert_with_metadata("a:1", "important doc", &[], Some(0.9), None, None)
            .await
            .unwrap();
        store
            .upsert_with_metadata("a:2", "low importance doc", &[], Some(0.1), None, None)
            .await
            .unwrap();

        let filter = HybridSearchFilter::new().with_min_importance(0.5);
        let results = store.search_with_filter("doc", &filter, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "a:1");
    }

    #[tokio::test]
    async fn test_search_with_filter_types() {
        let store = setup_store();
        store
            .upsert_with_metadata("c:1", "code", &[], None, Some(&["Code".into()]), None)
            .await
            .unwrap();
        store
            .upsert_with_metadata("d:1", "document", &[], None, Some(&["Doc".into()]), None)
            .await
            .unwrap();

        let filter = HybridSearchFilter::new().with_jsonld_types(vec!["Code".into()]);
        let results = store.search_with_filter("item", &filter, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "c:1");
    }

    #[tokio::test]
    async fn test_hybrid_search() {
        let store = setup_store();
        store
            .upsert("h:1", "urgent bug fix", &["urgent".into()])
            .await
            .unwrap();
        store
            .upsert("h:2", "routine maintenance", &["normal".into()])
            .await
            .unwrap();

        let results = store
            .hybrid_search("task", &["urgent".into()], &[], None, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "h:1");
    }

    #[tokio::test]
    async fn test_upsert_replaces_existing() {
        let store = setup_store();
        store.upsert("u:1", "first version", &[]).await.unwrap();
        store.upsert("u:1", "updated version", &[]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_scored_entry_fields() {
        let store = setup_store();
        store
            .upsert_with_metadata(
                "e:1",
                "test content",
                &["tag1".into(), "tag2".into()],
                Some(0.7),
                Some(&["TypeA".into()]),
                Some("graph1"),
            )
            .await
            .unwrap();

        // Use an importance filter to find it
        let filter = HybridSearchFilter::new().with_min_importance(0.5);
        let results = store.search_with_filter("test", &filter, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "e:1");
        assert_eq!(results[0].text, "test content");
        assert!(results[0].tags.contains(&"tag1".to_string()));
        assert!(results[0].tags.contains(&"tag2".to_string()));
        assert_eq!(results[0].importance, Some(0.7));
        assert!(results[0].jsonld_types.contains(&"TypeA".to_string()));
    }

    #[tokio::test]
    async fn test_search_filter_is_empty() {
        let store = setup_store();
        store.upsert("f:1", "item", &[]).await.unwrap();
        let filter = HybridSearchFilter::new();
        let results = store.search_with_filter("item", &filter, 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn fused_recall_indexes_only_explicitly_approved_documents_and_honors_filters() {
        let store = setup_store();
        store
            .upsert_recallable_with_metadata(
                "recall:rust",
                "Fix parseHTTPResponse error E0425",
                &["lang:rust".into()],
                None,
                Some(&["Code".into()]),
                None,
                true,
            )
            .await
            .unwrap();
        store
            .upsert(
                "raw:telemetry",
                "secret E0425 tool argument",
                &["telemetry".into()],
            )
            .await
            .unwrap();
        let filter = HybridSearchFilter::new().with_must_tags(vec!["lang:rust".into()]);
        let results = store
            .search_fused_recall("parseHTTPResponse E0425", &filter, 0.5, 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "recall:rust");
        assert!(results[0].decay_exempt);
    }

    #[tokio::test]
    async fn lexical_recall_rebuilds_from_durable_payload_after_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let embedding = Arc::new(FallbackEmbeddingService::new());
        {
            let store = HyperspaceStore::open(dir.path(), embedding.clone()).unwrap();
            store
                .upsert_recallable_with_metadata(
                    "recall:durable",
                    "stable ErrorCode E0554",
                    &[],
                    None,
                    None,
                    None,
                    false,
                )
                .await
                .unwrap();
            store.checkpoint().await.unwrap();
        }
        let reopened = HyperspaceStore::open(dir.path(), embedding).unwrap();
        let results = reopened
            .search_fused_recall("E0554", &HybridSearchFilter::new(), 0.5, 10)
            .await
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|entry| entry.iri.as_str())
                .collect::<Vec<_>>(),
            vec!["recall:durable"]
        );
    }

    #[tokio::test]
    async fn ann_health_probe_uses_exact_baseline_and_never_mutates_index() {
        let dir = tempfile::tempdir().unwrap();
        let store = HyperspaceStore::open(dir.path(), Arc::new(DeterministicEmbedding)).unwrap();
        store
            .upsert("ann:1", "rust error handling", &[])
            .await
            .unwrap();
        store
            .upsert("ann:2", "python web service", &[])
            .await
            .unwrap();
        store
            .upsert("ann:3", "typescript parser", &[])
            .await
            .unwrap();
        let before = store.count().await.unwrap();
        let report = store
            .assess_ann_health(&AnnHealthProbeConfig {
                sample_size: 3,
                top_k: 2,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(report.samples_evaluated, 3);
        assert!((0.0..=1.0).contains(&report.mean_recall_at_k));
        assert_eq!(store.count().await.unwrap(), before);
    }

    #[test]
    fn test_hybrid_search_filter_builder() {
        let filter = HybridSearchFilter::new()
            .with_must_tags(vec!["rust".to_string(), "async".to_string()])
            .with_should_tags(vec!["tokio".to_string()])
            .with_min_importance(0.5)
            .with_jsonld_types(vec!["Code".to_string()]);

        assert_eq!(filter.must_tags.len(), 2);
        assert_eq!(filter.should_tags.len(), 1);
        assert_eq!(filter.min_importance, Some(0.5));
        assert_eq!(filter.jsonld_types.len(), 1);
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_empty_filter() {
        let filter = HybridSearchFilter::new();
        assert!(filter.is_empty());
    }

    #[test]
    fn test_to_jsonld_filters_empty() {
        let dir = tempfile::tempdir().unwrap();
        let embed = Arc::new(FallbackEmbeddingService::new());
        let store = HyperspaceStore::open(dir.path(), embed).unwrap();

        let filters = store.to_jsonld_filters(&HybridSearchFilter::new());
        assert!(filters.is_empty());
    }

    #[test]
    fn test_to_jsonld_filters_must_tags() {
        let dir = tempfile::tempdir().unwrap();
        let embed = Arc::new(FallbackEmbeddingService::new());
        let store = HyperspaceStore::open(dir.path(), embed).unwrap();

        let filter = HybridSearchFilter::new().with_must_tags(vec!["a".into(), "b".into()]);
        let filters = store.to_jsonld_filters(&filter);
        assert_eq!(filters.len(), 1);
        match &filters[0] {
            JsonLdFilter::Must(children) => {
                assert_eq!(children.len(), 2);
            }
            _ => panic!("Expected Must filter"),
        }
    }

    #[test]
    fn test_to_jsonld_filters_all_groups() {
        let dir = tempfile::tempdir().unwrap();
        let embed = Arc::new(FallbackEmbeddingService::new());
        let store = HyperspaceStore::open(dir.path(), embed).unwrap();

        let filter = HybridSearchFilter::new()
            .with_must_tags(vec!["must".into()])
            .with_should_tags(vec!["should".into()])
            .with_must_not_tags(vec!["bad".into()]);
        let filters = store.to_jsonld_filters(&filter);
        // Expect 3 top-level filters: Must, Should, MustNot
        assert_eq!(filters.len(), 3);
    }

    #[tokio::test]
    async fn checkpoint_persists_snapshot_recoverable_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let embed = Arc::new(FallbackEmbeddingService::new());
        {
            let store = HyperspaceStore::open(dir.path(), embed.clone()).unwrap();
            store
                .upsert("ck:1", "checkpoint me", &["persist".into()])
                .await
                .unwrap();
            store.checkpoint().await.unwrap();
        }
        // Drop the first store so the WAL flock is released before reopening.
        let reopened = HyperspaceStore::open(dir.path(), embed).unwrap();
        let results = reopened.search("checkpoint", 10).await.unwrap();
        assert!(
            results.iter().any(|r| r.iri == "ck:1"),
            "checkpointed entry must survive reopen, got: {:?}",
            results.iter().map(|r| r.iri.as_str()).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn checkpoint_idempotent_on_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let embed = Arc::new(FallbackEmbeddingService::new());
        let store = HyperspaceStore::open(dir.path(), embed).unwrap();
        store.checkpoint().await.unwrap();
        store.vacuum().await.unwrap();
    }
}
