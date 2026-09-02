//! Validated embedding vectors and their on-disk representation.
//!
//! This module is the trust boundary for every vector entering the embedded
//! index.  A vector is accepted only when its dimensions, coordinates and
//! metric-space invariants are explicit and finite.  Metric implementations
//! may therefore treat an invalid vector as an unreachable defensive failure
//! instead of silently truncating coordinates or propagating NaN values.

use serde::{Deserialize, Serialize};

use crate::error::EngineError;

pub const VECTOR_HEADER_LEN: usize = 12;
pub const POINCARE_BOUNDARY_EPSILON: f64 = 1e-9;
const LORENTZ_TOLERANCE: f64 = 1e-8;

/// Run-time selectable metric space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricKind {
    Cosine,
    Poincare,
    Lorentz,
    Euclidean,
}

impl MetricKind {
    pub const fn tag(self) -> u32 {
        match self {
            Self::Cosine => 0,
            Self::Poincare => 1,
            Self::Lorentz => 2,
            Self::Euclidean => 3,
        }
    }

    pub fn from_tag(tag: u32) -> Result<Self, EngineError> {
        match tag {
            0 => Ok(Self::Cosine),
            1 => Ok(Self::Poincare),
            2 => Ok(Self::Lorentz),
            3 => Ok(Self::Euclidean),
            _ => Err(EngineError::InvalidVector(format!(
                "unknown metric tag {tag}"
            ))),
        }
    }
}

/// An embedding vector together with its metric-space tag.
///
/// `alpha` is a cached, derived Poincaré value retained in the persisted
/// representation for compatibility with existing engine files.  It is never
/// authoritative: constructors and deserializers recompute and validate it.
#[derive(Debug, Clone)]
pub struct EmbeddingVector {
    pub coords: Vec<f64>,
    pub metric: MetricKind,
    pub alpha: f64,
}

impl EmbeddingVector {
    /// Construct a vector that satisfies the selected metric's invariants.
    pub fn new(coords: Vec<f64>, metric: MetricKind) -> Result<Self, EngineError> {
        validate_finite_non_empty(&coords)?;
        let alpha = canonical_alpha(&coords, metric)?;
        Ok(Self {
            coords,
            metric,
            alpha,
        })
    }

    /// Construct from an embedding-provider response.
    pub fn from_f32_slice(coords: &[f32], metric: MetricKind) -> Result<Self, EngineError> {
        Self::new(
            coords.iter().map(|value| f64::from(*value)).collect(),
            metric,
        )
    }

    /// Verify dimensions, finite coordinates and the metric invariant.
    pub fn validate(&self) -> Result<(), EngineError> {
        let canonical = Self::new(self.coords.clone(), self.metric)?;
        if !self.alpha.is_finite()
            || !approximately_equal(self.alpha, canonical.alpha, 1e-10, 1e-12)
        {
            return Err(EngineError::InvalidVector(
                "cached vector alpha does not match the coordinates and metric".into(),
            ));
        }
        Ok(())
    }

    /// Verify this vector against an engine's fixed dimension and metric.
    pub fn validate_for_engine(
        &self,
        expected_dimension: usize,
        expected_metric: MetricKind,
    ) -> Result<(), EngineError> {
        if expected_dimension == 0 {
            return Err(EngineError::InvalidVector(
                "engine dimension must be greater than zero".into(),
            ));
        }
        if self.coords.len() != expected_dimension {
            return Err(EngineError::InvalidVector(format!(
                "vector dimension {} does not match engine dimension {expected_dimension}",
                self.coords.len()
            )));
        }
        if self.metric != expected_metric {
            return Err(EngineError::InvalidVector(format!(
                "vector metric {:?} does not match engine metric {:?}",
                self.metric, expected_metric
            )));
        }
        self.validate()
    }

    /// Legacy constructor for tests and trusted in-memory fixtures only.
    ///
    /// Production callers must use [`Self::new`]. Engine write, query,
    /// recovery and snapshot paths revalidate values before use, so an
    /// unchecked fixture cannot bypass persistence safeguards.
    #[doc(hidden)]
    pub fn new_unchecked(coords: Vec<f64>, metric: MetricKind) -> Self {
        let alpha = canonical_alpha(&coords, metric).unwrap_or(f64::NAN);
        Self {
            coords,
            metric,
            alpha,
        }
    }

    /// Serialize to the fixed-size vector record used by the storage layer.
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::element_size(self.coords.len()));
        bytes.extend_from_slice(&self.metric.tag().to_le_bytes());
        bytes.extend_from_slice(&self.alpha.to_le_bytes());
        for coordinate in &self.coords {
            bytes.extend_from_slice(&coordinate.to_le_bytes());
        }
        bytes
    }

    /// Deserialize a fixed-size vector record and revalidate all invariants.
    pub fn from_bytes(bytes: &[u8], dim: usize) -> Result<Self, EngineError> {
        if dim == 0 {
            return Err(EngineError::InvalidVector(
                "serialized vector dimension must be greater than zero".into(),
            ));
        }
        let expected_len = Self::element_size(dim);
        if bytes.len() != expected_len {
            return Err(EngineError::InvalidVector(format!(
                "vector record length {} does not match expected {expected_len}",
                bytes.len()
            )));
        }

        let mut metric_bytes = [0_u8; 4];
        metric_bytes.copy_from_slice(&bytes[..4]);
        let metric = MetricKind::from_tag(u32::from_le_bytes(metric_bytes))?;

        let mut alpha_bytes = [0_u8; 8];
        alpha_bytes.copy_from_slice(&bytes[4..VECTOR_HEADER_LEN]);
        let stored_alpha = f64::from_le_bytes(alpha_bytes);

        let mut coords = Vec::with_capacity(dim);
        for offset in (VECTOR_HEADER_LEN..expected_len).step_by(std::mem::size_of::<f64>()) {
            let mut coordinate_bytes = [0_u8; 8];
            coordinate_bytes.copy_from_slice(&bytes[offset..offset + 8]);
            coords.push(f64::from_le_bytes(coordinate_bytes));
        }

        let vector = Self::new(coords, metric)?;
        if !stored_alpha.is_finite()
            || !approximately_equal(stored_alpha, vector.alpha, 1e-10, 1e-12)
        {
            return Err(EngineError::InvalidVector(
                "serialized vector alpha does not match the canonical metric value".into(),
            ));
        }
        Ok(vector)
    }

    /// Element size in bytes for fixed-size mmap storage.
    pub const fn element_size(dim: usize) -> usize {
        dim * std::mem::size_of::<f64>() + VECTOR_HEADER_LEN
    }

    /// Squared Euclidean distance. Invalid or mismatched operands are never
    /// treated as near neighbours; callers requiring diagnostics should use
    /// [`checked_l2_squared`].
    pub fn l2_sq(&self, other: &EmbeddingVector) -> f64 {
        checked_l2_squared(&self.coords, &other.coords).unwrap_or(f64::INFINITY)
    }
}

pub fn validate_finite_non_empty(coords: &[f64]) -> Result<(), EngineError> {
    if coords.is_empty() {
        return Err(EngineError::InvalidVector(
            "vector coordinates must not be empty".into(),
        ));
    }
    if coords.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(EngineError::InvalidVector(
            "vector coordinates contain NaN or infinity".into(),
        ));
    }
    Ok(())
}

pub fn validate_poincare_coordinates(coords: &[f64], curvature: f64) -> Result<(), EngineError> {
    validate_finite_non_empty(coords)?;
    validate_curvature(curvature)?;
    let norm_sq = checked_norm_squared(coords)?;
    let boundary_sq = 1.0 / curvature;
    if norm_sq >= boundary_sq * (1.0 - POINCARE_BOUNDARY_EPSILON) {
        return Err(EngineError::InvalidVector(format!(
            "Poincare vector must be strictly inside radius {}, got squared norm {norm_sq}",
            boundary_sq.sqrt()
        )));
    }
    Ok(())
}

pub fn validate_lorentz_coordinates(coords: &[f64]) -> Result<(), EngineError> {
    validate_finite_non_empty(coords)?;
    if coords.len() < 2 {
        return Err(EngineError::InvalidVector(
            "Lorentz vectors require a time coordinate and at least one spatial coordinate".into(),
        ));
    }
    if coords[0] <= 0.0 {
        return Err(EngineError::InvalidVector(
            "Lorentz vector must use the future time-oriented sheet".into(),
        ));
    }
    let minkowski_norm = -coords[0] * coords[0]
        + coords
            .iter()
            .skip(1)
            .map(|value| value * value)
            .sum::<f64>();
    let tolerance = LORENTZ_TOLERANCE * minkowski_norm.abs().max(1.0);
    if (minkowski_norm + 1.0).abs() > tolerance {
        return Err(EngineError::InvalidVector(format!(
            "Lorentz vector violates -t^2 + sum(x_i^2) = -1; got {minkowski_norm}"
        )));
    }
    Ok(())
}

pub fn validate_curvature(curvature: f64) -> Result<(), EngineError> {
    if !curvature.is_finite() || curvature <= 0.0 {
        return Err(EngineError::InvalidVector(
            "Poincare curvature must be finite and greater than zero".into(),
        ));
    }
    Ok(())
}

pub fn checked_l2_squared(left: &[f64], right: &[f64]) -> Result<f64, EngineError> {
    ensure_same_dimension(left, right)?;
    let value = left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::InvalidVector(
            "squared distance is not finite".into(),
        ))
    }
}

pub fn checked_dot_product(left: &[f64], right: &[f64]) -> Result<f64, EngineError> {
    ensure_same_dimension(left, right)?;
    let value = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::InvalidVector(
            "dot product is not finite".into(),
        ))
    }
}

pub fn checked_norm_squared(coords: &[f64]) -> Result<f64, EngineError> {
    validate_finite_non_empty(coords)?;
    let value = coords
        .iter()
        .map(|coordinate| coordinate * coordinate)
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EngineError::InvalidVector(
            "vector norm is not finite".into(),
        ))
    }
}

/// Compatibility helper. Use [`checked_l2_squared`] at a trust boundary.
pub fn l2_squared(left: &[f64], right: &[f64]) -> f64 {
    checked_l2_squared(left, right).unwrap_or(f64::INFINITY)
}

/// Compatibility helper. Use [`checked_dot_product`] at a trust boundary.
pub fn dot_product(left: &[f64], right: &[f64]) -> f64 {
    checked_dot_product(left, right).unwrap_or(f64::NAN)
}

/// Compatibility helper. Use [`checked_norm_squared`] at a trust boundary.
pub fn norm_squared(coords: &[f64]) -> f64 {
    checked_norm_squared(coords).unwrap_or(f64::NAN)
}

/// Numerically stable inverse hyperbolic cosine for finite values >= 1.
pub fn fast_acosh(value: f64) -> f64 {
    if !value.is_finite() {
        return f64::NAN;
    }
    if value <= 1.0 {
        return 0.0;
    }
    let delta = value - 1.0;
    if delta < 1e-8 {
        (2.0 * delta).sqrt()
    } else if value < 1e150 {
        (value + (value * value - 1.0).sqrt()).ln()
    } else {
        value.ln() + std::f64::consts::LN_2
    }
}

fn canonical_alpha(coords: &[f64], metric: MetricKind) -> Result<f64, EngineError> {
    match metric {
        MetricKind::Poincare => {
            validate_poincare_coordinates(coords, 1.0)?;
            let norm_sq = checked_norm_squared(coords)?;
            Ok(1.0 / (1.0 - norm_sq))
        }
        MetricKind::Lorentz => {
            validate_lorentz_coordinates(coords)?;
            Ok(0.0)
        }
        MetricKind::Cosine => {
            validate_finite_non_empty(coords)?;
            if checked_norm_squared(coords)? <= f64::EPSILON {
                return Err(EngineError::InvalidVector(
                    "cosine vectors must have a non-zero norm".into(),
                ));
            }
            Ok(0.0)
        }
        MetricKind::Euclidean => {
            validate_finite_non_empty(coords)?;
            Ok(0.0)
        }
    }
}

fn ensure_same_dimension(left: &[f64], right: &[f64]) -> Result<(), EngineError> {
    validate_finite_non_empty(left)?;
    validate_finite_non_empty(right)?;
    if left.len() != right.len() {
        return Err(EngineError::InvalidVector(format!(
            "vector dimensions differ: {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(())
}

fn approximately_equal(left: f64, right: f64, relative: f64, absolute: f64) -> bool {
    (left - right).abs() <= absolute.max(relative * left.abs().max(right.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_or_non_finite_coordinates() {
        assert!(EmbeddingVector::new(vec![], MetricKind::Cosine).is_err());
        assert!(EmbeddingVector::new(vec![f64::NAN], MetricKind::Cosine).is_err());
        assert!(EmbeddingVector::new(vec![f64::INFINITY], MetricKind::Euclidean).is_err());
    }

    #[test]
    fn validates_metric_specific_geometry() {
        assert!(EmbeddingVector::new(vec![0.5, 0.3], MetricKind::Poincare).is_ok());
        assert!(EmbeddingVector::new(vec![1.0, 0.0], MetricKind::Poincare).is_err());
        assert!(
            EmbeddingVector::new(vec![2.0_f64.cosh(), 2.0_f64.sinh()], MetricKind::Lorentz).is_ok()
        );
        assert!(EmbeddingVector::new(vec![0.0, 1.0], MetricKind::Lorentz).is_err());
    }

    #[test]
    fn serialized_vectors_require_exact_length_and_canonical_alpha() {
        let vector = EmbeddingVector::new(vec![0.1, 0.2], MetricKind::Poincare).unwrap();
        let bytes = vector.as_bytes();
        assert_eq!(
            EmbeddingVector::from_bytes(&bytes, 2).unwrap().coords,
            vector.coords
        );
        assert!(EmbeddingVector::from_bytes(&bytes[..bytes.len() - 1], 2).is_err());

        let mut tampered = bytes;
        tampered[4..12].copy_from_slice(&0.0_f64.to_le_bytes());
        assert!(EmbeddingVector::from_bytes(&tampered, 2).is_err());
    }

    #[test]
    fn dimension_mismatch_is_never_silently_truncated() {
        assert!(checked_l2_squared(&[1.0, 2.0], &[1.0]).is_err());
        assert!(checked_dot_product(&[1.0, 2.0], &[1.0]).is_err());
        assert!(l2_squared(&[1.0, 2.0], &[1.0]).is_infinite());
    }

    #[test]
    fn fast_acosh_stays_finite_across_supported_ranges() {
        assert_eq!(fast_acosh(1.0), 0.0);
        assert!(fast_acosh(1.00000001).is_finite());
        assert!(fast_acosh(1e200).is_finite());
    }
}
