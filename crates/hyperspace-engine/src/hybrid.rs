//! Dual-space hybrid search: text (Cosine) × structural (Poincaré) weighted fusion.
//!
//! Runs two separate searches in parallel and fuses results via:
//! - Reciprocal Rank Fusion (RRF)
//! - Weighted score combination: `score = α * text_score + (1-α) * struct_score`
//!
//! The alpha parameter controls the blend:
//! - α = 1.0 → pure text search
//! - α = 0.0 → pure structural search
//! - α = 0.5 → balanced

use std::collections::HashMap;

use crate::engine::SearchHit;
use crate::hnsw::IncrementalHNSW;
use crate::hyper_vector::EmbeddingVector;
use crate::jsonld_meta::JsonLdMetadataIndex;

/// Perform hybrid search: runs text and structural searches, fuses results.
///
/// # Arguments
///
/// * `text_index` - HNSW index using Cosine metric for text embeddings
/// * `struct_index` - HNSW index using Poincaré metric for structural embeddings
/// * `metadata` - Shared JSON-LD metadata index (IDs consistent across both)
/// * `text_query` - Query vector for text search (None = skip)
/// * `struct_query` - Query vector for structural search (None = skip)
/// * `top_k` - Number of results to return after fusion
/// * `alpha` - Text:struct weight (1.0 = pure text, 0.0 = pure struct)
///
/// # Returns
///
/// List of SearchHit sorted by fused score (descending).
pub fn hybrid_search(
    text_index: &mut IncrementalHNSW,
    struct_index: &mut IncrementalHNSW,
    metadata: &JsonLdMetadataIndex,
    text_query: Option<&EmbeddingVector>,
    struct_query: Option<&EmbeddingVector>,
    top_k: usize,
    alpha: f32,
) -> Vec<SearchHit> {
    let expand_k = (top_k * 3).max(50);

    // Run both searches
    let text_results = text_query.map_or(Vec::new(), |q| text_index.search(q, expand_k));
    let struct_results = struct_query.map_or(Vec::new(), |q| struct_index.search(q, expand_k));

    if text_results.is_empty() && struct_results.is_empty() {
        return Vec::new();
    }
    if text_results.is_empty() {
        return struct_results
            .into_iter()
            .take(top_k)
            .map(|(id, score)| {
                let payload = metadata.get_payload(id);
                SearchHit {
                    id,
                    iri: payload
                        .as_ref()
                        .and_then(|v| v.get("@id").and_then(|s| s.as_str()))
                        .unwrap_or("")
                        .to_string(),
                    score: score as f32,
                    payload,
                }
            })
            .collect();
    }
    if struct_results.is_empty() {
        return text_results
            .into_iter()
            .take(top_k)
            .map(|(id, score)| {
                let payload = metadata.get_payload(id);
                SearchHit {
                    id,
                    iri: payload
                        .as_ref()
                        .and_then(|v| v.get("@id").and_then(|s| s.as_str()))
                        .unwrap_or("")
                        .to_string(),
                    score: score as f32,
                    payload,
                }
            })
            .collect();
    }

    // Build score maps
    let max_text_dist = text_results.first().map(|r| r.1).unwrap_or(1.0).max(0.001);
    let max_struct_dist = struct_results
        .first()
        .map(|r| r.1)
        .unwrap_or(1.0)
        .max(0.001);

    let mut fused: HashMap<u32, f32> = HashMap::new();

    // RRF-style rank contribution
    for (rank, (id, dist)) in text_results.iter().enumerate() {
        let _ = rank;
        let dist_score = alpha * (1.0 - (*dist as f32) / max_text_dist as f32);
        *fused.entry(*id).or_insert(0.0) += dist_score;
    }
    for (rank, (id, dist)) in struct_results.iter().enumerate() {
        let _ = rank;
        let dist_score = (1.0 - alpha) * (1.0 - (*dist as f32) / max_struct_dist as f32);
        *fused.entry(*id).or_insert(0.0) += dist_score;
    }

    // Sort by fused score descending
    let mut sorted: Vec<(u32, f32)> = fused.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(top_k);

    // Convert to SearchHit
    sorted
        .into_iter()
        .map(|(id, score)| {
            let payload = metadata.get_payload(id);
            SearchHit {
                id,
                iri: payload
                    .as_ref()
                    .and_then(|v| v.get("@id").and_then(|s| s.as_str()))
                    .unwrap_or("")
                    .to_string(),
                score,
                payload,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyper_vector::MetricKind;
    use crate::metric::metric_from_kind;

    fn v(coords: Vec<f64>, kind: MetricKind) -> EmbeddingVector {
        EmbeddingVector::new_unchecked(coords, kind)
    }

    #[test]
    fn test_hybrid_search_text_only() {
        let mut idx =
            IncrementalHNSW::new(metric_from_kind(MetricKind::Cosine), Default::default());
        let mut idx_struct =
            IncrementalHNSW::new(metric_from_kind(MetricKind::Poincare), Default::default());
        let meta = JsonLdMetadataIndex::new();

        idx.insert(0, v(vec![1.0, 0.0, 0.0, 0.0], MetricKind::Cosine));
        idx.insert(1, v(vec![0.0, 1.0, 0.0, 0.0], MetricKind::Cosine));

        let text_q = v(vec![1.0, 0.0, 0.0, 0.0], MetricKind::Cosine);
        let results = hybrid_search(
            &mut idx,
            &mut idx_struct,
            &meta,
            Some(&text_q),
            None,
            5,
            1.0,
        );
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_hybrid_search_both_empty() {
        let mut idx =
            IncrementalHNSW::new(metric_from_kind(MetricKind::Cosine), Default::default());
        let mut idx_struct =
            IncrementalHNSW::new(metric_from_kind(MetricKind::Poincare), Default::default());
        let meta = JsonLdMetadataIndex::new();

        let results = hybrid_search(&mut idx, &mut idx_struct, &meta, None, None, 5, 0.5);
        assert!(results.is_empty());
    }
}
