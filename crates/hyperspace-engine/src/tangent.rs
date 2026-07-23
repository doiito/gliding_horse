//! Tangent-space pruning for Poincaré search.
//!
//! Maps Poincaré ball points to tangent space at the Fréchet mean
//! using the proper exponential/logarithmic maps, performs Euclidean
//! filtering, then provides candidates for exact Poincaré re-ranking.
//!
//! Reference: ruvector's TangentCache pattern, enhanced with proper
//! gyrovector space operations.

use crate::hyper_vector::{l2_squared, EmbeddingVector};
use crate::metric::{exp_map, log_map};

/// Compute the Fréchet mean (Karcher mean) of Poincaré vectors via
/// gradient descent.
pub fn frechet_mean(vectors: &[EmbeddingVector], curvature: f64) -> Vec<f64> {
    if vectors.is_empty() {
        return vec![];
    }
    let dim = vectors[0].coords.len();
    let n = vectors.len() as f64;

    // Start at Euclidean mean (good initial guess for Poincaré ball)
    let mut mean: Vec<f64> = vec![0.0; dim];
    for v in vectors {
        for (i, &c) in v.coords.iter().enumerate() {
            mean[i] += c;
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    // One step of gradient descent in Poincaré geometry:
    // mean_new = exp_mean(learning_rate * sum(log_mean(v_i)))
    // This gives a much better estimate than Euclidean mean alone.
    let lr = 0.5;
    let mut sum_tangent = vec![0.0; dim];
    for v in vectors {
        let tv = log_map(&mean, &v.coords, curvature);
        for (s, t) in sum_tangent.iter_mut().zip(tv.iter()) {
            *s += t;
        }
    }
    // Scale by lr / n
    for s in &mut sum_tangent {
        *s *= lr / n;
    }
    exp_map(&mean, &sum_tangent, curvature)
}

/// Tangent-space cache for pruning Poincaré search.
///
/// Builds tangent vectors at the Fréchet mean once, then uses
/// Euclidean distance in tangent space for fast candidate filtering.
pub struct TangentCache {
    /// Fréchet mean of all vectors (in Poincaré ball).
    pub centroid: Vec<f64>,
    /// Tangent vectors at centroid (one per input vector).
    pub tangent_vectors: Vec<Vec<f64>>,
    /// Curvature parameter.
    pub curvature: f64,
}

impl TangentCache {
    /// Build the cache from vectors.
    ///
    /// O(n * d) for centroid computation + O(n * d) for tangent mapping.
    pub fn build(vectors: &[EmbeddingVector], curvature: f64) -> Self {
        let centroid = frechet_mean(vectors, curvature);
        let tangent_vectors: Vec<Vec<f64>> = vectors
            .iter()
            .map(|v| log_map(&centroid, &v.coords, curvature))
            .collect();
        Self {
            centroid,
            tangent_vectors,
            curvature,
        }
    }

    /// Incrementally update the cache with a single new vector.
    /// Note: This does NOT recompute the centroid (would be O(n*d)).
    /// For batch updates, rebuild the cache.
    pub fn insert(&mut self, vector: &EmbeddingVector) {
        let tv = log_map(&self.centroid, &vector.coords, self.curvature);
        self.tangent_vectors.push(tv);
    }

    /// Remove the tangent vector at the given index.
    pub fn remove(&mut self, idx: usize) {
        if idx < self.tangent_vectors.len() {
            self.tangent_vectors.remove(idx);
        }
    }

    /// Two-step search: Euclidean filter → caller does exact re-ranking.
    ///
    /// Returns up to `top_k * prune_factor` candidate IDs sorted by
    /// Euclidean distance in tangent space.
    pub fn search_with_pruning(
        &self,
        query: &EmbeddingVector,
        all_ids: &[u32],
        top_k: usize,
        prune_factor: usize,
    ) -> Vec<u32> {
        if self.tangent_vectors.is_empty() || all_ids.is_empty() {
            return Vec::new();
        }

        let q_tangent = log_map(&self.centroid, &query.coords, self.curvature);
        let mut candidates: Vec<(u32, f64)> = all_ids
            .iter()
            .zip(self.tangent_vectors.iter())
            .map(|(&id, tv)| (id, l2_squared(&q_tangent, tv)))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k * prune_factor);
        candidates.into_iter().map(|(id, _)| id).collect()
    }

    /// Number of cached tangent vectors.
    pub fn len(&self) -> usize {
        self.tangent_vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tangent_vectors.is_empty()
    }
}

/// Check if the Poincaré ball is the active metric space.
pub fn is_poincare(metric_kind: &crate::hyper_vector::MetricKind) -> bool {
    matches!(metric_kind, crate::hyper_vector::MetricKind::Poincare)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetricKind;

    fn v(coords: Vec<f64>) -> EmbeddingVector {
        EmbeddingVector::new_unchecked(coords, MetricKind::Poincare)
    }

    #[test]
    fn test_frechet_mean_euclidean_start() {
        let v1 = v(vec![0.1, 0.2]);
        let v2 = v(vec![0.3, 0.1]);
        let mean = frechet_mean(&[v1, v2], 1.0);
        assert_eq!(mean.len(), 2);
        // Euclidean mean would be (0.2, 0.15)
        assert!((mean[0] - 0.2).abs() < 0.1);
        assert!((mean[1] - 0.15).abs() < 0.1);
    }

    #[test]
    fn test_tangent_cache_build() {
        let vectors: Vec<EmbeddingVector> = (0..10)
            .map(|i| {
                let x = (i as f64) * 0.05 + 0.1;
                v(vec![x, x * 0.5])
            })
            .collect();
        let cache = TangentCache::build(&vectors, 1.0);
        assert_eq!(cache.len(), 10);
        assert_eq!(cache.tangent_vectors.len(), 10);
    }

    #[test]
    fn test_tangent_cache_pruning() {
        let vectors: Vec<EmbeddingVector> = (0..20)
            .map(|i| {
                let x = (i as f64) * 0.04;
                v(vec![x, x])
            })
            .collect();
        let cache = TangentCache::build(&vectors, 1.0);
        let query = v(vec![0.2, 0.2]);
        let ids: Vec<u32> = (0..20).collect();
        let result = cache.search_with_pruning(&query, &ids, 3, 2);
        assert!(!result.is_empty());
        assert!(result.len() <= 6);
        // The closest in tangent space should include id_5 (x=0.2)
        assert!(
            result.contains(&5),
            "result should contain the point closest to query"
        );
    }

    #[test]
    fn test_insert_after_build() {
        let v1 = v(vec![0.1, 0.1]);
        let v2 = v(vec![0.2, 0.2]);
        let mut cache = TangentCache::build(&[v1], 1.0);
        assert_eq!(cache.len(), 1);
        cache.insert(&v2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_remove() {
        let v1 = v(vec![0.1, 0.1]);
        let v2 = v(vec![0.2, 0.2]);
        let mut cache = TangentCache::build(&[v1, v2], 1.0);
        assert_eq!(cache.len(), 2);
        cache.remove(0);
        assert_eq!(cache.len(), 1);
    }
}
