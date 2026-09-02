//! Optional Poincaré candidate pruning with exact-distance re-ranking.
//!
//! A cache is local to one immutable candidate set.  It is never an index of
//! record: callers must re-rank the returned IDs with exact Poincaré distance
//! before exposing results.  Construction and every mutation validate metric,
//! dimension and identifier consistency.

use std::collections::HashSet;

use crate::error::EngineError;
use crate::hyper_vector::{checked_l2_squared, checked_norm_squared, EmbeddingVector, MetricKind};
use crate::metric::{exp_map, log_map};

const MAX_CENTROID_ITERATIONS: usize = 16;
const CENTROID_CONVERGENCE: f64 = 1e-10;

/// Compute a Fréchet mean in the Poincaré ball using bounded gradient steps.
pub fn frechet_mean(vectors: &[EmbeddingVector], curvature: f64) -> Result<Vec<f64>, EngineError> {
    let dimension = validate_poincare_collection(vectors, curvature)?;
    let count = vectors.len() as f64;
    let mut mean = vec![0.0; dimension];
    for vector in vectors {
        for (target, coordinate) in mean.iter_mut().zip(&vector.coords) {
            *target += coordinate / count;
        }
    }
    // The Poincaré ball is convex; an arithmetic average of interior points
    // stays in the ball. Reconstructing validates finite values and the bound.
    let mut mean = EmbeddingVector::new(mean, MetricKind::Poincare)?.coords;

    for _ in 0..MAX_CENTROID_ITERATIONS {
        let mut average_tangent = vec![0.0; dimension];
        for vector in vectors {
            let tangent = log_map(&mean, &vector.coords, curvature)?;
            for (accumulator, coordinate) in average_tangent.iter_mut().zip(tangent) {
                *accumulator += coordinate / count;
            }
        }
        if checked_norm_squared(&average_tangent)?.sqrt() <= CENTROID_CONVERGENCE {
            break;
        }
        let next = exp_map(&mean, &average_tangent, curvature)?;
        if checked_l2_squared(&mean, &next)?.sqrt() <= CENTROID_CONVERGENCE {
            mean = next;
            break;
        }
        mean = next;
    }
    Ok(mean)
}

/// Candidate-local tangent-space cache.
#[derive(Debug, Clone)]
pub struct TangentCache {
    centroid: Vec<f64>,
    tangent_vectors: Vec<Vec<f64>>,
    ids: Vec<u32>,
    curvature: f64,
    dimension: usize,
}

impl TangentCache {
    /// Build a cache from a single candidate set with stable IDs.
    pub fn build(
        candidates: &[(u32, EmbeddingVector)],
        curvature: f64,
    ) -> Result<Self, EngineError> {
        if candidates.is_empty() {
            return Err(EngineError::InvalidVector(
                "cannot build a tangent cache from an empty candidate set".into(),
            ));
        }
        let ids = candidates.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let unique = ids.iter().copied().collect::<HashSet<_>>();
        if unique.len() != ids.len() {
            return Err(EngineError::InvalidVector(
                "tangent cache candidate IDs must be unique".into(),
            ));
        }
        let vectors = candidates
            .iter()
            .map(|(_, vector)| vector.clone())
            .collect::<Vec<_>>();
        let dimension = validate_poincare_collection(&vectors, curvature)?;
        let centroid = frechet_mean(&vectors, curvature)?;
        let tangent_vectors = vectors
            .iter()
            .map(|vector| log_map(&centroid, &vector.coords, curvature))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            centroid,
            tangent_vectors,
            ids,
            curvature,
            dimension,
        })
    }

    /// Add one validated candidate. Duplicate IDs are rejected rather than
    /// silently replacing a vector and desynchronising the cache.
    pub fn insert(&mut self, id: u32, vector: &EmbeddingVector) -> Result<(), EngineError> {
        self.validate_vector(vector)?;
        if self.ids.contains(&id) {
            return Err(EngineError::InvalidVector(format!(
                "tangent cache already contains candidate ID {id}"
            )));
        }
        self.tangent_vectors
            .push(log_map(&self.centroid, &vector.coords, self.curvature)?);
        self.ids.push(id);
        Ok(())
    }

    /// Remove a candidate and its vector atomically. Returns false when the ID
    /// is absent, preserving an idempotent removal boundary.
    pub fn remove(&mut self, id: u32) -> bool {
        let Some(index) = self.ids.iter().position(|candidate| *candidate == id) else {
            return false;
        };
        self.ids.swap_remove(index);
        self.tangent_vectors.swap_remove(index);
        true
    }

    /// Return candidate IDs ordered by tangent-space distance. The caller must
    /// exact-rerank these IDs in the source metric before using them.
    pub fn search_with_pruning(
        &self,
        query: &EmbeddingVector,
        top_k: usize,
        prune_factor: usize,
    ) -> Result<Vec<u32>, EngineError> {
        self.validate_vector(query)?;
        if top_k == 0 || self.tangent_vectors.is_empty() {
            return Ok(Vec::new());
        }
        let query_tangent = log_map(&self.centroid, &query.coords, self.curvature)?;
        let mut candidates = self
            .ids
            .iter()
            .copied()
            .zip(self.tangent_vectors.iter())
            .map(|(id, tangent)| Ok((id, checked_l2_squared(&query_tangent, tangent)?)))
            .collect::<Result<Vec<_>, EngineError>>()?;
        candidates.sort_by(|left, right| left.1.total_cmp(&right.1));
        let cap = top_k
            .saturating_mul(prune_factor.max(1))
            .min(candidates.len());
        candidates.truncate(cap);
        Ok(candidates.into_iter().map(|(id, _)| id).collect())
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn validate_vector(&self, vector: &EmbeddingVector) -> Result<(), EngineError> {
        if vector.metric != MetricKind::Poincare {
            return Err(EngineError::InvalidVector(
                "tangent cache accepts only Poincare vectors".into(),
            ));
        }
        if vector.coords.len() != self.dimension {
            return Err(EngineError::InvalidVector(format!(
                "tangent cache vector dimension {} does not match {}",
                vector.coords.len(),
                self.dimension
            )));
        }
        vector.validate()
    }
}

/// Whether this optional optimisation applies to an engine metric.
pub fn is_poincare(metric_kind: &MetricKind) -> bool {
    matches!(metric_kind, MetricKind::Poincare)
}

fn validate_poincare_collection(
    vectors: &[EmbeddingVector],
    curvature: f64,
) -> Result<usize, EngineError> {
    let Some(first) = vectors.first() else {
        return Err(EngineError::InvalidVector(
            "Poincare collection must not be empty".into(),
        ));
    };
    if first.metric != MetricKind::Poincare {
        return Err(EngineError::InvalidVector(
            "tangent cache requires Poincare vectors".into(),
        ));
    }
    first.validate()?;
    let dimension = first.coords.len();
    for vector in vectors.iter().skip(1) {
        if vector.metric != MetricKind::Poincare || vector.coords.len() != dimension {
            return Err(EngineError::InvalidVector(
                "Poincare collection mixes metrics or dimensions".into(),
            ));
        }
        vector.validate()?;
    }
    // `log_map` validates the requested curvature. Trigger it here even when
    // the collection holds valid curvature-one vectors.
    crate::hyper_vector::validate_curvature(curvature)?;
    for vector in vectors {
        crate::hyper_vector::validate_poincare_coordinates(&vector.coords, curvature)?;
    }
    Ok(dimension)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(x: f64, y: f64) -> EmbeddingVector {
        EmbeddingVector::new(vec![x, y], MetricKind::Poincare).unwrap()
    }

    #[test]
    fn cache_keeps_ids_and_vectors_in_sync() {
        let mut cache =
            TangentCache::build(&[(10, vector(0.1, 0.1)), (20, vector(0.2, 0.1))], 1.0).unwrap();
        cache.insert(30, &vector(0.3, 0.1)).unwrap();
        assert_eq!(cache.len(), 3);
        assert!(cache.remove(20));
        assert!(!cache.remove(20));
        let result = cache.search_with_pruning(&vector(0.1, 0.1), 2, 2).unwrap();
        assert!(!result.contains(&20));
        assert!(result.contains(&10));
    }

    #[test]
    fn cache_rejects_mixed_metrics_dimensions_and_ids() {
        let cosine = EmbeddingVector::new(vec![0.1, 0.1], MetricKind::Cosine).unwrap();
        assert!(TangentCache::build(&[(1, cosine)], 1.0).is_err());
        assert!(TangentCache::build(&[(1, vector(0.1, 0.1)), (1, vector(0.2, 0.1))], 1.0).is_err());
        let cache = TangentCache::build(&[(1, vector(0.1, 0.1))], 1.0).unwrap();
        let invalid_query = EmbeddingVector::new(vec![0.1, 0.1], MetricKind::Cosine).unwrap();
        assert!(cache.search_with_pruning(&invalid_query, 1, 2).is_err());
    }

    #[test]
    fn frechet_mean_is_valid_and_deterministic() {
        let points = vec![vector(0.1, 0.1), vector(0.2, 0.1), vector(0.3, 0.1)];
        let first = frechet_mean(&points, 1.0).unwrap();
        let second = frechet_mean(&points, 1.0).unwrap();
        EmbeddingVector::new(first.clone(), MetricKind::Poincare).unwrap();
        assert_eq!(first, second);
    }
}
