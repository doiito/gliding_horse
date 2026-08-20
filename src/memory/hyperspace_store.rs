use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use hyperspace_engine::engine::{HyperspaceEngine, HyperspaceEngineImpl, SearchHit};
use hyperspace_engine::filter::JsonLdFilter;
use hyperspace_engine::hnsw::HnswConfig;
use hyperspace_engine::hyper_vector::{EmbeddingVector, MetricKind};
use hyperspace_engine::metric::CosineMetric;
use hyperspace_engine::wal::WalSyncMode;
use serde_json::Value;
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
}

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
                let (text, tags, importance, jsonld_types, stored_at) = hit
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
                        (text, tags, importance, jsonld_types, stored_at)
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
            if let Some(stored_at) = entry.stored_at {
                let age_secs = now.timestamp() as f64 - stored_at;
                let age_hours = age_secs / 3600.0;
                if age_hours > 0.0 {
                    entry.score *= (-decay_lambda * age_hours).exp() as f32;
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
        self.engine.checkpoint().await.map_err(|e| CoreError::Internal {
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
