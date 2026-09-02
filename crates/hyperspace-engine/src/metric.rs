//! Distance metrics and validated Poincaré-ball operations.
//!
//! The mathematical operations here are implemented from their definitions
//! and expose fallible helpers for callers crossing a trust boundary.  The
//! `Metric` trait cannot return an error, so its implementations use positive
//! infinity as a defensive non-neighbour result after validation has failed.

use crate::error::EngineError;
use crate::hyper_vector::{
    checked_dot_product, checked_l2_squared, checked_norm_squared, fast_acosh, validate_curvature,
    validate_finite_non_empty, validate_lorentz_coordinates, validate_poincare_coordinates,
    EmbeddingVector, MetricKind, POINCARE_BOUNDARY_EPSILON,
};

const NUMERIC_EPSILON: f64 = 1e-12;

/// Core metric trait used by the HNSW implementation.
pub trait Metric: Send + Sync {
    fn kind(&self) -> MetricKind;

    fn name(&self) -> &'static str {
        match self.kind() {
            MetricKind::Cosine => "cosine",
            MetricKind::Poincare => "poincare",
            MetricKind::Lorentz => "lorentz",
            MetricKind::Euclidean => "euclidean",
        }
    }

    fn distance(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64;
    fn distance_sq(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64;
    fn distance_upper(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64;
}

/// Angular distance for text embeddings.  Inputs need not be pre-normalized.
pub struct CosineMetric;

impl Metric for CosineMetric {
    fn kind(&self) -> MetricKind {
        MetricKind::Cosine
    }

    fn distance(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        cosine_distance(left, right).unwrap_or(f64::INFINITY)
    }

    fn distance_sq(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        let distance = self.distance(left, right);
        distance * distance
    }

    fn distance_upper(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        self.distance(left, right)
    }
}

/// Exact Poincaré-ball distance at curvature one.
pub struct PoincareMetric;

impl Metric for PoincareMetric {
    fn kind(&self) -> MetricKind {
        MetricKind::Poincare
    }

    fn distance(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        if left.metric != MetricKind::Poincare || right.metric != MetricKind::Poincare {
            return f64::INFINITY;
        }
        poincare_distance_curved(&left.coords, &right.coords, 1.0).unwrap_or(f64::INFINITY)
    }

    fn distance_sq(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        let distance = self.distance(left, right);
        distance * distance
    }

    fn distance_upper(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        self.distance(left, right)
    }
}

/// Lorentz-hyperboloid distance on the future time-oriented sheet.
pub struct LorentzMetric;

impl Metric for LorentzMetric {
    fn kind(&self) -> MetricKind {
        MetricKind::Lorentz
    }

    fn distance(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        if left.metric != MetricKind::Lorentz || right.metric != MetricKind::Lorentz {
            return f64::INFINITY;
        }
        match lorentz_inner(&left.coords, &right.coords) {
            Ok(inner) => {
                let distance = fast_acosh((-inner).max(1.0));
                if distance.is_finite() {
                    distance
                } else {
                    f64::INFINITY
                }
            }
            Err(_) => f64::INFINITY,
        }
    }

    fn distance_sq(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        let distance = self.distance(left, right);
        distance * distance
    }

    fn distance_upper(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        self.distance(left, right)
    }
}

/// Euclidean L2 distance.
pub struct EuclideanMetric;

impl Metric for EuclideanMetric {
    fn kind(&self) -> MetricKind {
        MetricKind::Euclidean
    }

    fn distance(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        if left.metric != MetricKind::Euclidean || right.metric != MetricKind::Euclidean {
            return f64::INFINITY;
        }
        checked_l2_squared(&left.coords, &right.coords)
            .map(f64::sqrt)
            .unwrap_or(f64::INFINITY)
    }

    fn distance_sq(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        if left.metric != MetricKind::Euclidean || right.metric != MetricKind::Euclidean {
            return f64::INFINITY;
        }
        checked_l2_squared(&left.coords, &right.coords).unwrap_or(f64::INFINITY)
    }

    fn distance_upper(&self, left: &EmbeddingVector, right: &EmbeddingVector) -> f64 {
        self.distance(left, right)
    }
}

/// Validated Poincaré distance with arbitrary positive curvature.
pub fn poincare_distance_curved(
    left: &[f64],
    right: &[f64],
    curvature: f64,
) -> Result<f64, EngineError> {
    validate_curvature(curvature)?;
    validate_poincare_coordinates(left, curvature)?;
    validate_poincare_coordinates(right, curvature)?;
    let difference_sq = checked_l2_squared(left, right)?;
    let left_norm_sq = checked_norm_squared(left)?;
    let right_norm_sq = checked_norm_squared(right)?;
    let denominator = (1.0 - curvature * left_norm_sq) * (1.0 - curvature * right_norm_sq);
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(EngineError::InvalidVector(
            "Poincare distance denominator is non-positive".into(),
        ));
    }
    let argument = 1.0 + 2.0 * curvature * difference_sq / denominator;
    let distance = fast_acosh(argument) / curvature.sqrt();
    if distance.is_finite() && distance >= 0.0 {
        Ok(distance)
    } else {
        Err(EngineError::InvalidVector(
            "Poincare distance is not finite".into(),
        ))
    }
}

/// Möbius addition in the Poincaré ball.
pub fn mobius_add(left: &[f64], right: &[f64], curvature: f64) -> Result<Vec<f64>, EngineError> {
    validate_curvature(curvature)?;
    validate_poincare_coordinates(left, curvature)?;
    validate_poincare_coordinates(right, curvature)?;
    let left_norm_sq = checked_norm_squared(left)?;
    let right_norm_sq = checked_norm_squared(right)?;
    let dot = checked_dot_product(left, right)?;
    let numerator_left = 1.0 + 2.0 * curvature * dot + curvature * right_norm_sq;
    let numerator_right = 1.0 - curvature * left_norm_sq;
    let denominator =
        1.0 + 2.0 * curvature * dot + curvature * curvature * left_norm_sq * right_norm_sq;
    if !denominator.is_finite() || denominator.abs() <= NUMERIC_EPSILON {
        return Err(EngineError::InvalidVector(
            "Möbius addition denominator is unstable".into(),
        ));
    }
    let result = left
        .iter()
        .zip(right)
        .map(|(a, b)| (numerator_left * a + numerator_right * b) / denominator)
        .collect::<Vec<_>>();
    project_to_poincare_ball(result, curvature)
}

/// Möbius subtraction: `left ⊕ (-right)`.
pub fn mobius_sub(left: &[f64], right: &[f64], curvature: f64) -> Result<Vec<f64>, EngineError> {
    validate_finite_non_empty(right)?;
    let negated = right.iter().map(|value| -*value).collect::<Vec<_>>();
    mobius_add(left, &negated, curvature)
}

/// Exponential map from a tangent vector at `base` into the Poincaré ball.
pub fn exp_map(base: &[f64], tangent: &[f64], curvature: f64) -> Result<Vec<f64>, EngineError> {
    validate_curvature(curvature)?;
    validate_poincare_coordinates(base, curvature)?;
    validate_finite_non_empty(tangent)?;
    if base.len() != tangent.len() {
        return Err(EngineError::InvalidVector(
            "Poincare base and tangent dimensions differ".into(),
        ));
    }
    let tangent_norm = checked_norm_squared(tangent)?.sqrt();
    if tangent_norm <= NUMERIC_EPSILON {
        return Ok(base.to_vec());
    }
    let base_norm_sq = checked_norm_squared(base)?;
    let conformal = 2.0 / (1.0 - curvature * base_norm_sq);
    let scale = (curvature.sqrt() * conformal * tangent_norm / 2.0).tanh()
        / (curvature.sqrt() * tangent_norm);
    let direction = tangent
        .iter()
        .map(|value| value * scale)
        .collect::<Vec<_>>();
    mobius_add(base, &direction, curvature)
}

/// Logarithmic map from `base` to `target` in the Poincaré ball.
pub fn log_map(base: &[f64], target: &[f64], curvature: f64) -> Result<Vec<f64>, EngineError> {
    validate_curvature(curvature)?;
    validate_poincare_coordinates(base, curvature)?;
    validate_poincare_coordinates(target, curvature)?;
    if base.len() != target.len() {
        return Err(EngineError::InvalidVector(
            "Poincare base and target dimensions differ".into(),
        ));
    }
    let negated_base = base.iter().map(|value| -*value).collect::<Vec<_>>();
    let displacement = mobius_add(&negated_base, target, curvature)?;
    let displacement_norm = checked_norm_squared(&displacement)?.sqrt();
    if displacement_norm <= NUMERIC_EPSILON {
        return Ok(vec![0.0; base.len()]);
    }
    let argument = curvature.sqrt() * displacement_norm;
    if !argument.is_finite() || argument >= 1.0 {
        return Err(EngineError::InvalidVector(
            "Poincare logarithmic map reached the ball boundary".into(),
        ));
    }
    let base_norm_sq = checked_norm_squared(base)?;
    let conformal = 2.0 / (1.0 - curvature * base_norm_sq);
    let scale = 2.0 * argument.atanh() / (curvature.sqrt() * conformal * displacement_norm);
    let tangent = displacement
        .iter()
        .map(|value| value * scale)
        .collect::<Vec<_>>();
    validate_finite_non_empty(&tangent)?;
    Ok(tangent)
}

/// Minkowski inner product using the `(-,+,...,+)` convention.
pub fn lorentz_inner(left: &[f64], right: &[f64]) -> Result<f64, EngineError> {
    validate_lorentz_coordinates(left)?;
    validate_lorentz_coordinates(right)?;
    if left.len() != right.len() {
        return Err(EngineError::InvalidVector(format!(
            "Lorentz vector dimensions differ: {} and {}",
            left.len(),
            right.len()
        )));
    }
    let value = -left[0] * right[0]
        + left
            .iter()
            .skip(1)
            .zip(right.iter().skip(1))
            .map(|(a, b)| a * b)
            .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::InvalidVector(
            "Lorentz inner product is not finite".into(),
        ))
    }
}

/// Validate a Lorentz vector without exposing geometry internals to callers.
pub fn lorentz_validate(coords: &[f64]) -> Result<(), String> {
    validate_lorentz_coordinates(coords).map_err(|error| error.to_string())
}

/// Factory for the engine's configured metric.
pub fn metric_from_kind(kind: MetricKind) -> Box<dyn Metric> {
    match kind {
        MetricKind::Cosine => Box::new(CosineMetric),
        MetricKind::Poincare => Box::new(PoincareMetric),
        MetricKind::Lorentz => Box::new(LorentzMetric),
        MetricKind::Euclidean => Box::new(EuclideanMetric),
    }
}

fn cosine_distance(left: &EmbeddingVector, right: &EmbeddingVector) -> Result<f64, EngineError> {
    if left.metric != MetricKind::Cosine || right.metric != MetricKind::Cosine {
        return Err(EngineError::InvalidVector(
            "cosine distance requires cosine vectors".into(),
        ));
    }
    left.validate()?;
    right.validate()?;
    let dot = checked_dot_product(&left.coords, &right.coords)?;
    let left_norm = checked_norm_squared(&left.coords)?.sqrt();
    let right_norm = checked_norm_squared(&right.coords)?.sqrt();
    if left_norm <= NUMERIC_EPSILON || right_norm <= NUMERIC_EPSILON {
        return Err(EngineError::InvalidVector(
            "cosine distance rejects zero-norm vectors".into(),
        ));
    }
    let similarity = (dot / (left_norm * right_norm)).clamp(-1.0, 1.0);
    Ok(1.0 - similarity)
}

fn project_to_poincare_ball(
    mut coordinates: Vec<f64>,
    curvature: f64,
) -> Result<Vec<f64>, EngineError> {
    validate_finite_non_empty(&coordinates)?;
    let norm_sq = checked_norm_squared(&coordinates)?;
    let max_norm = ((1.0 - POINCARE_BOUNDARY_EPSILON) / curvature).sqrt();
    let norm = norm_sq.sqrt();
    if norm >= max_norm {
        let scale = max_norm / norm;
        for coordinate in &mut coordinates {
            *coordinate *= scale;
        }
    }
    validate_poincare_coordinates(&coordinates, curvature)?;
    Ok(coordinates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(coords: Vec<f64>, kind: MetricKind) -> EmbeddingVector {
        EmbeddingVector::new(coords, kind).unwrap()
    }

    #[test]
    fn cosine_is_scale_invariant_and_rejects_zero_vectors() {
        let metric = CosineMetric;
        let left = vector(vec![2.0, 0.0], MetricKind::Cosine);
        let right = vector(vec![7.0, 0.0], MetricKind::Cosine);
        assert_eq!(metric.distance(&left, &right), 0.0);
        assert!(EmbeddingVector::new(vec![0.0, 0.0], MetricKind::Cosine).is_err());
    }

    #[test]
    fn poincare_distance_is_finite_non_negative_and_symmetric() {
        let left = vector(vec![0.2, 0.1], MetricKind::Poincare);
        let right = vector(vec![-0.3, 0.1], MetricKind::Poincare);
        let metric = PoincareMetric;
        let forward = metric.distance(&left, &right);
        let reverse = metric.distance(&right, &left);
        assert!(forward.is_finite() && forward > 0.0);
        assert!((forward - reverse).abs() < 1e-12);
        assert_eq!(metric.distance(&left, &left), 0.0);
    }

    #[test]
    fn geometry_operations_reject_invalid_dimensions_and_curvature() {
        assert!(poincare_distance_curved(&[0.1], &[0.2, 0.3], 1.0).is_err());
        assert!(poincare_distance_curved(&[0.1], &[0.2], 0.0).is_err());
        assert!(mobius_add(&[0.1], &[0.2, 0.3], 1.0).is_err());
    }

    #[test]
    fn logarithmic_and_exponential_maps_round_trip() {
        let base = vec![0.1, 0.2];
        let target = vec![0.3, 0.4];
        let tangent = log_map(&base, &target, 1.0).unwrap();
        let recovered = exp_map(&base, &tangent, 1.0).unwrap();
        for (expected, actual) in target.iter().zip(recovered) {
            assert!((expected - actual).abs() < 1e-9);
        }
    }

    #[test]
    fn lorentz_vectors_use_the_hyperboloid_not_the_poincare_ball() {
        let left = vector(
            vec![2.0_f64.cosh(), 2.0_f64.sinh(), 0.0],
            MetricKind::Lorentz,
        );
        let right = vector(vec![1.0, 0.0, 0.0], MetricKind::Lorentz);
        let distance = LorentzMetric.distance(&left, &right);
        assert!((distance - 2.0).abs() < 1e-9);
        assert!(EmbeddingVector::new(vec![0.5, 0.1], MetricKind::Lorentz).is_err());
    }

    #[test]
    fn metric_mismatch_cannot_produce_a_near_neighbor() {
        let cosine = vector(vec![1.0, 0.0], MetricKind::Cosine);
        let euclidean = vector(vec![1.0, 0.0], MetricKind::Euclidean);
        assert!(CosineMetric.distance(&cosine, &euclidean).is_infinite());
    }
}
