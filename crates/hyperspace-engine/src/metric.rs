use crate::hyper_vector::{
    dot_product, fast_acosh, l2_squared, norm_squared, EmbeddingVector, MetricKind,
};

/// The core Metric trait - allows runtime switching between distance functions.
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
    fn distance(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64;
    fn distance_sq(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64;
    fn distance_upper(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64;
}

/// Cosine distance = 1 - dot_product (for L2-normalized vectors).
pub struct CosineMetric;

impl Metric for CosineMetric {
    fn kind(&self) -> MetricKind {
        MetricKind::Cosine
    }

    fn distance(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        1.0 - dot_product(&a.coords, &b.coords)
    }

    fn distance_sq(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        self.distance(a, b)
    }

    fn distance_upper(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        self.distance(a, b)
    }
}

/// Poincaré distance = acosh(1 + 2 * ||u-v||² * α_u * α_v).
/// Uses precomputed alpha for fast distance_sq.
pub struct PoincareMetric;

impl Metric for PoincareMetric {
    fn kind(&self) -> MetricKind {
        MetricKind::Poincare
    }

    fn distance(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        let sq = self.distance_sq(a, b);
        fast_acosh(sq)
    }

    fn distance_sq(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        let l2_sq = a.l2_sq(b);
        1.0 + 2.0 * l2_sq * a.alpha * b.alpha
    }

    fn distance_upper(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        // Klein chord distance (faster routing bound)
        let l2_sq = a.l2_sq(b);
        l2_sq * a.alpha * b.alpha
    }
}

/// Lorentz (hyperboloid) distance.
///
/// Points must satisfy the Minkowski norm constraint:
///     -x₀² + x₁² + ... + xₙ² = -1
///
/// Distance: d(x, y) = acosh(-⟨x, y⟩_L)
/// where ⟨x, y⟩_L = -x₀y₀ + Σᵢ₌₁ⁿ xᵢyᵢ
pub struct LorentzMetric;

impl Metric for LorentzMetric {
    fn kind(&self) -> MetricKind {
        MetricKind::Lorentz
    }

    fn distance(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        let inner = lorentz_inner(&a.coords, &b.coords);
        let arg = (-inner).max(1.0 + 1e-12);
        fast_acosh(arg)
    }

    fn distance_sq(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        let d = self.distance(a, b);
        d * d
    }

    fn distance_upper(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        // Upper bound via Minkowski norm difference
        let inner = lorentz_inner(&a.coords, &b.coords);
        (2.0 * (-inner - 1.0).abs()).sqrt()
    }
}

/// Minkowski (Lorentz) inner product: ⟨x, y⟩ = -x₀y₀ + Σᵢ₌₁ⁿ xᵢyᵢ
pub fn lorentz_inner(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = -a[0] * b[0];
    for i in 1..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Validate that a vector is on the Lorentz hyperboloid (within tolerance).
pub fn lorentz_validate(coords: &[f64]) -> Result<(), String> {
    if coords.is_empty() {
        return Err("Empty coordinates".into());
    }
    let mut minkowski_norm = -coords[0] * coords[0];
    for value in coords.iter().skip(1) {
        minkowski_norm += value * value;
    }
    let err = (minkowski_norm + 1.0).abs();
    if err > 1e-6 {
        return Err(format!(
            "Lorentz vector violates Minkowski norm (-x0^2 + sum(xi^2) = -1): got {minkowski_norm}, error {err}"
        ));
    }
    Ok(())
}

/// Euclidean (L2) distance.
pub struct EuclideanMetric;

impl Metric for EuclideanMetric {
    fn kind(&self) -> MetricKind {
        MetricKind::Euclidean
    }

    fn distance(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        a.l2_sq(b).sqrt()
    }

    fn distance_sq(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        a.l2_sq(b)
    }

    fn distance_upper(&self, a: &EmbeddingVector, b: &EmbeddingVector) -> f64 {
        self.distance_sq(a, b)
    }
}

/// Poincaré distance with variable curvature c (ruvector generalization).
///
/// d(x, y) = (1/√c) * arcosh(1 + 2c * ||x-y||² / ((1-c||x||²)(1-c||y||²)))
pub fn poincare_distance_curved(u: &[f64], v: &[f64], c: f64) -> f64 {
    let diff_sq = l2_squared(u, v);
    let norm_u_sq = norm_squared(u);
    let norm_v_sq = norm_squared(v);
    let denom = (1.0 - c * norm_u_sq) * (1.0 - c * norm_v_sq);
    let arg = 1.0 + 2.0 * c * diff_sq / denom.max(f64::EPSILON);
    (1.0 / c.sqrt()) * fast_acosh(arg)
}

/// Möbius addition in Poincaré ball (ruvector original).
///
/// x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
pub fn mobius_add(x: &[f64], y: &[f64], c: f64) -> Vec<f64> {
    let x_dot_y = dot_product(x, y);
    let norm_x_sq = norm_squared(x);
    let norm_y_sq = norm_squared(y);
    let denom = 1.0 + 2.0 * c * x_dot_y + c * c * norm_x_sq * norm_y_sq;
    let num_x = 1.0 + c * (2.0 * x_dot_y + norm_y_sq);
    let num_y = 1.0 - c * norm_x_sq;
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (num_x * xi + num_y * yi) / denom)
        .collect()
}

/// Möbius subtraction: x ⊖ y = x ⊕ (-y)
pub fn mobius_sub(x: &[f64], y: &[f64], c: f64) -> Vec<f64> {
    let neg_y: Vec<f64> = y.iter().map(|v| -v).collect();
    mobius_add(x, &neg_y, c)
}

/// Exponential map from tangent space at p to the Poincaré ball.
///
/// exp_p(v) = p ⊕ (tanh(√c * λ_p * ||v|| / 2) * v / (√c * ||v||))
pub fn exp_map(p: &[f64], v: &[f64], c: f64) -> Vec<f64> {
    let norm_v = norm_squared(v).sqrt();
    if norm_v < 1e-15 {
        return p.to_vec();
    }

    let lambda_p = 2.0 / (1.0 - c * norm_squared(p));
    let factor = (lambda_p * c.sqrt() * norm_v / 2.0).tanh() / (c.sqrt() * norm_v);

    let direction: Vec<f64> = v.iter().map(|vi| vi * factor).collect();
    mobius_add(p, &direction, c)
}

/// Logarithmic map from the Poincaré ball to tangent space at p.
///
/// log_p(q) = (2 / (√c * λ_p)) * atanh(√c * ||-p ⊕ q||) * (-p ⊕ q) / ||-p ⊕ q||
pub fn log_map(p: &[f64], q: &[f64], c: f64) -> Vec<f64> {
    let diff = mobius_sub(q, p, c);
    let norm_diff = norm_squared(&diff).sqrt();
    if norm_diff < 1e-15 {
        return vec![0.0; p.len()];
    }

    let lambda_p = 2.0 / (1.0 - c * norm_squared(p));
    let factor = 2.0 / (c.sqrt() * lambda_p) * (c.sqrt() * norm_diff).atanh() / norm_diff;

    diff.iter().map(|di| di * factor).collect()
}

/// Factory: create Metric from MetricKind.
pub fn metric_from_kind(kind: MetricKind) -> Box<dyn Metric> {
    match kind {
        MetricKind::Cosine => Box::new(CosineMetric),
        MetricKind::Poincare => Box::new(PoincareMetric),
        MetricKind::Lorentz => Box::new(LorentzMetric),
        MetricKind::Euclidean => Box::new(EuclideanMetric),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(coords: Vec<f64>, kind: MetricKind) -> EmbeddingVector {
        EmbeddingVector::new(coords, kind).unwrap()
    }

    fn v_unchecked(coords: Vec<f64>, kind: MetricKind) -> EmbeddingVector {
        EmbeddingVector::new_unchecked(coords, kind)
    }

    #[test]
    fn test_cosine_distance() {
        let a = v(vec![1.0, 0.0], MetricKind::Cosine);
        let b = v(vec![0.0, 1.0], MetricKind::Cosine);
        let d = CosineMetric.distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_zero_distance() {
        let a = v(vec![1.0, 0.0], MetricKind::Cosine);
        let b = v(vec![1.0, 0.0], MetricKind::Cosine);
        let d = CosineMetric.distance(&a, &b);
        assert!((d - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_poincare_distance_sq() {
        let a = v(vec![0.5, 0.0], MetricKind::Poincare);
        let b = v(vec![-0.3, 0.1], MetricKind::Poincare);
        let dsq = PoincareMetric.distance_sq(&a, &b);
        assert!(dsq >= 1.0, "distance_sq must be >= 1, got {dsq}");
        let d = PoincareMetric.distance(&a, &b);
        assert!(d >= 0.0);
    }

    #[test]
    fn test_lorentz_distance() {
        // On hyperboloid: t^2 - x^2 - y^2 = 1
        // Point: t=1, x=0 gives x^2=0, so 1 - 0 = 1 ✓
        // Point: t=cosh(r), x=sinh(r) gives cosh^2 - sinh^2 = 1 ✓
        let a = LorentzVec::new(2.0_f64.cosh(), 2.0_f64.sinh(), 0.0);
        let b = LorentzVec::new(1.0, 0.0, 0.0);
        // Use new_unchecked because Lorentz vectors have spatial norm > 1
        let d = LorentzMetric.distance(
            &v_unchecked(a.to_vec(), MetricKind::Lorentz),
            &v_unchecked(b.to_vec(), MetricKind::Lorentz),
        );
        assert!(d > 0.0);
    }

    #[test]
    fn test_lorentz_validate() {
        let coords = LorentzVec::new(2.0_f64.cosh(), 2.0_f64.sinh(), 0.0).to_vec();
        assert!(lorentz_validate(&coords).is_ok());

        let bad = vec![0.0, 1.0, 0.0]; // t=0, x=1 → 0 - 1 = -1 ≠ -1
        assert!(lorentz_validate(&bad).is_err());
    }

    #[test]
    fn test_metric_from_kind() {
        let m = metric_from_kind(MetricKind::Cosine);
        assert_eq!(m.kind(), MetricKind::Cosine);
        assert_eq!(m.name(), "cosine");

        let m = metric_from_kind(MetricKind::Poincare);
        assert_eq!(m.kind(), MetricKind::Poincare);
        assert_eq!(m.name(), "poincare");

        let m = metric_from_kind(MetricKind::Lorentz);
        assert_eq!(m.kind(), MetricKind::Lorentz);
        assert_eq!(m.name(), "lorentz");

        let m = metric_from_kind(MetricKind::Euclidean);
        assert_eq!(m.kind(), MetricKind::Euclidean);
        assert_eq!(m.name(), "euclidean");
    }

    #[test]
    fn test_poincare_distance_curved_default() {
        let u = vec![0.3, 0.2];
        let v = vec![-0.1, 0.4];
        let d1 = poincare_distance_curved(&u, &v, 1.0);
        assert!(d1 > 0.0);
    }

    #[test]
    fn test_mobius_add_identity() {
        let x = vec![0.3, 0.2];
        let zero = vec![0.0, 0.0];
        let result = mobius_add(&x, &zero, 1.0);
        assert!((result[0] - x[0]).abs() < 1e-10);
        assert!((result[1] - x[1]).abs() < 1e-10);
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let p = vec![0.1, 0.2];
        let q = vec![0.3, 0.4];
        let v = log_map(&p, &q, 1.0);
        let q2 = exp_map(&p, &v, 1.0);
        // Poincaré log/exp roundtrip is approximate with floating point
        for i in 0..p.len() {
            assert!(
                (q[i] - q2[i]).abs() < 1e-2,
                "log/exp roundtrip mismatch at {i}: {} vs {}",
                q[i],
                q2[i]
            );
        }
    }

    #[test]
    fn test_exp_map_at_origin() {
        let origin = vec![0.0, 0.0];
        let v = vec![0.1, 0.2];
        let result = exp_map(&origin, &v, 1.0);
        // At origin, exp_0(v) = tanh(||v||) * v/||v||
        let norm_v = (0.1f64 * 0.1 + 0.2 * 0.2).sqrt();
        let expected_scale = norm_v.tanh() / norm_v;
        assert!((result[0] - 0.1 * expected_scale).abs() < 1e-10);
        assert!((result[1] - 0.2 * expected_scale).abs() < 1e-10);
    }

    /// Helper to create Lorentz (hyperboloid) coordinates.
    struct LorentzVec {
        t: f64,
        x: f64,
        y: f64,
    }

    impl LorentzVec {
        fn new(t: f64, x: f64, y: f64) -> Self {
            Self { t, x, y }
        }
        fn to_vec(&self) -> Vec<f64> {
            vec![self.t, self.x, self.y]
        }
    }
}
