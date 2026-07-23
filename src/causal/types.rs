use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::skill_graph::types::SkillLinkType;

// ─── CausalObservation ──────────────────────────────────────────────────────

/// A single error observation with full context.
///
/// This extends the existing `CausalEvent` with embedding support for
/// error-signature similarity matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalObservation {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub skill_iri: String,
    pub error_class: String,
    pub error_signature: String,
    pub context: HashMap<String, String>,
    /// Which error propagated into this one (if known).
    /// `None` means this is a candidate root (first observed failure).
    pub propagation_from: Option<String>,
}

impl CausalObservation {
    pub fn new(event_id: &str, skill_iri: &str, error_class: &str, error_signature: &str) -> Self {
        Self {
            event_id: event_id.to_string(),
            timestamp: Utc::now(),
            skill_iri: skill_iri.to_string(),
            error_class: error_class.to_string(),
            error_signature: error_signature.to_string(),
            context: HashMap::new(),
            propagation_from: None,
        }
    }

    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_propagation(mut self, from: &str) -> Self {
        self.propagation_from = Some(from.to_string());
        self
    }
}

// ─── CausalInference ─────────────────────────────────────────────────────────

/// The result of a root cause analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalInference {
    /// IRI of the inferred root cause skill
    pub root_cause_iri: String,
    /// Confidence score (0.0–1.0)
    pub confidence: f32,
    /// Full propagation path from root to each observed error
    pub propagation_paths: Vec<PropagationPath>,
    /// Other possible causes with their confidence
    pub alternative_causes: Vec<(String, f32)>,
    /// How many observations were explained by this root cause
    pub observations_explained: usize,
    /// Total observations in the query
    pub total_observations: usize,
}

/// A single propagation chain from root cause to observed error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationPath {
    /// Ordered list of (skill_iri, link_type) from root → observed
    pub hops: Vec<PropagationHop>,
    /// The final observed error
    pub terminal_observation: CausalObservation,
    /// Confidence in this specific path
    pub path_confidence: f32,
}

/// One hop in a propagation chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationHop {
    pub skill_iri: String,
    pub link_type: SkillLinkType,
    pub propagation_probability: f32,
}

// ─── Edge type for the propagation graph ────────────────────────────────────

/// Weighted edge in the propagation graph.
/// `weight = propagation_count / total_failures_from_source`
#[derive(Debug, Clone, Copy)]
pub struct PropagationEdge {
    pub weight: f32,
    pub observation_count: u32,
}
