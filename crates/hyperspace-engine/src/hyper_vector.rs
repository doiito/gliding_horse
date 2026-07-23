use serde::{Deserialize, Serialize};

use crate::error::EngineError;

/// Run-time selectable metric space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricKind {
    Cosine,
    Poincare,
    Lorentz,
    Euclidean,
}

/// An embedding vector with precomputed alpha (for Poincaré) and its metric.
#[derive(Debug, Clone)]
pub struct EmbeddingVector {
    pub coords: Vec<f64>,
    pub metric: MetricKind,
    pub alpha: f64,
}

impl EmbeddingVector {
    /// Create a new EmbeddingVector, computing alpha if Poincaré metric.
    pub fn new(coords: Vec<f64>, metric: MetricKind) -> Result<Self, EngineError> {
        for &c in &coords {
            if !c.is_finite() {
                return Err(EngineError::InvalidVector(
                    "Coordinates contain NaN or infinity".into(),
                ));
            }
        }
        let alpha = match metric {
            MetricKind::Poincare | MetricKind::Lorentz => {
                let sq_norm: f64 = coords.iter().map(|&x| x * x).sum();
                if sq_norm >= 1.0 - 1e-9 {
                    return Err(EngineError::InvalidVector(
                        "Poincaré vector must be strictly inside the unit ball".into(),
                    ));
                }
                1.0 / (1.0 - sq_norm)
            }
            _ => 0.0,
        };
        Ok(Self {
            coords,
            metric,
            alpha,
        })
    }

    /// Create from f32 slice (e.g. from ONNX embedding output).
    pub fn from_f32_slice(coords: &[f32], metric: MetricKind) -> Result<Self, EngineError> {
        let f64_coords: Vec<f64> = coords.iter().map(|&x| x as f64).collect();
        Self::new(f64_coords, metric)
    }

    /// Create unchecked (no validation). Alpha computed if Poincaré.
    pub fn new_unchecked(coords: Vec<f64>, metric: MetricKind) -> Self {
        let alpha = match metric {
            MetricKind::Poincare | MetricKind::Lorentz => {
                let sq_norm: f64 = coords.iter().map(|&x| x * x).sum();
                if sq_norm < 1.0 {
                    1.0 / (1.0 - sq_norm)
                } else {
                    1.0
                }
            }
            _ => 0.0,
        };
        Self {
            coords,
            metric,
            alpha,
        }
    }

    /// Serialize to fixed-size bytes for storage (f64 per coord + alpha).
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.coords.len() * 8 + 16);
        // Metric tag
        bytes.extend_from_slice(&(self.metric as u32).to_le_bytes());
        // Alpha
        bytes.extend_from_slice(&self.alpha.to_le_bytes());
        // Coordinates
        for c in &self.coords {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8], dim: usize) -> Result<Self, EngineError> {
        if bytes.len() < 12 {
            return Err(EngineError::InvalidVector("Too short".into()));
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[0..4]);
        let metric_int = u32::from_le_bytes(buf);
        let metric = match metric_int {
            0 => MetricKind::Cosine,
            1 => MetricKind::Poincare,
            2 => MetricKind::Lorentz,
            3 => MetricKind::Euclidean,
            _ => {
                return Err(EngineError::InvalidVector(format!(
                    "Unknown metric kind: {metric_int}"
                )))
            }
        };
        let mut alpha_buf = [0u8; 8];
        alpha_buf.copy_from_slice(&bytes[4..12]);
        let alpha = f64::from_le_bytes(alpha_buf);

        let mut coords = Vec::with_capacity(dim);
        for i in 0..dim {
            let offset = 12 + i * 8;
            if offset + 8 > bytes.len() {
                return Err(EngineError::InvalidVector("Truncated vector data".into()));
            }
            let mut cbuf = [0u8; 8];
            cbuf.copy_from_slice(&bytes[offset..offset + 8]);
            coords.push(f64::from_le_bytes(cbuf));
        }
        Ok(Self {
            coords,
            metric,
            alpha,
        })
    }

    /// Element size in bytes for storage (for fixed-size mmap).
    pub fn element_size(dim: usize) -> usize {
        dim * 8 + 12
    }

    /// Squared L2 distance between two vectors.
    pub fn l2_sq(&self, other: &EmbeddingVector) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
    }
}

// Helper: L2 squared between two f64 slices
pub fn l2_squared(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// Helper: dot product
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Helper: squared norm
pub fn norm_squared(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
}

// Fast acosh with three regimes (borrowed from ruvector)
pub fn fast_acosh(x: f64) -> f64 {
    if x <= 1.0 {
        return 0.0;
    }
    let delta = x - 1.0;
    if delta < 1e-4 {
        (2.0 * delta).sqrt()
    } else if x < 1e6 {
        (x + (x * x - 1.0).sqrt()).ln()
    } else {
        (2.0 * x).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_vector_new_cosine() {
        let v = EmbeddingVector::new(vec![0.1, 0.2, 0.3], MetricKind::Cosine).unwrap();
        assert_eq!(v.metric, MetricKind::Cosine);
        assert_eq!(v.alpha, 0.0);
    }

    #[test]
    fn test_embedding_vector_new_poincare() {
        let v = EmbeddingVector::new(vec![0.5, 0.3], MetricKind::Poincare).unwrap();
        assert_eq!(v.metric, MetricKind::Poincare);
        assert!(v.alpha > 0.0);
    }

    #[test]
    fn test_embedding_vector_rejects_outside_ball() {
        let result = EmbeddingVector::new(vec![1.0, 0.0], MetricKind::Poincare);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let v = EmbeddingVector::new(vec![0.1, 0.2, 0.3, 0.4], MetricKind::Cosine).unwrap();
        let bytes = v.as_bytes();
        let restored = EmbeddingVector::from_bytes(&bytes, 4).unwrap();
        assert_eq!(v.metric, restored.metric);
        assert!((v.alpha - restored.alpha).abs() < 1e-12);
        assert_eq!(v.coords, restored.coords);
    }

    #[test]
    fn test_l2_squared() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = l2_squared(&a, &b);
        assert!((result - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_acosh_near_one() {
        let r = fast_acosh(1.00000001);
        assert!(r > 0.0);
        assert!(r < 0.001);
    }

    #[test]
    fn test_fast_acosh_large() {
        let r = fast_acosh(1e10);
        assert!(r > 0.0);
    }
}
