//! Offline-only candidate-graph reranking experiment.
//!
//! This is deliberately a small, deterministic graph smoother rather than a
//! trainable model.  It operates on one homogeneous, in-memory candidate set
//! and emits only stable IRI rankings.  Raw vectors, queries, labels, prompts,
//! and document bodies never enter L0.  A successful offline result can become
//! an evolution *proposal* but cannot alter online retrieval by itself.

use std::collections::HashSet;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::offline_retrieval_eval::{
    OfflineRanking, OfflineRetrievalCase, OfflineRetrievalEvaluation, OfflineRetrievalEvaluator,
};
use crate::CoreError;

pub const CANDIDATE_GRAPH_RERANK_SCHEMA_VERSION: u32 = 1;
const MAX_CASES: usize = 10_000;
const MAX_CANDIDATES: usize = 256;
const MAX_VECTOR_DIMENSIONS: usize = 8_192;
const MAX_IDENTIFIER_CHARS: usize = 512;
const MAX_TASK_FAMILY_CHARS: usize = 512;
const MIN_CANDIDATES: usize = 2;
const MAX_ROUNDS: u8 = 3;
const MAX_NEIGHBOURS: usize = 64;
const MIN_SELF_WEIGHT: f64 = 0.50;
const MAX_SELF_WEIGHT: f64 = 0.90;
const NORM_EPSILON: f64 = 1e-12;

/// Linear vector spaces supported by the experiment.  Hyperbolic and mixed
/// spaces are intentionally excluded: this smoother assumes pairwise
/// candidate affinity is comparable across the complete candidate set.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateGraphMetric {
    Cosine,
    Euclidean,
}

/// Bounded configuration for one candidate-graph experiment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateGraphRerankConfig {
    pub metric: CandidateGraphMetric,
    /// Maximum neighbours retained per candidate in the transient graph.
    pub neighbour_count: usize,
    /// Weight retained from a candidate's original first-stage score.
    pub self_weight: f64,
    /// Number of synchronous diffusion rounds.  The value is deliberately
    /// small so a local graph cannot erase the first-stage signal.
    pub rounds: u8,
    /// Hard resource boundary for one case.  The graph is O(n² × dimensions).
    pub max_candidates: usize,
}

impl Default for CandidateGraphRerankConfig {
    fn default() -> Self {
        Self {
            metric: CandidateGraphMetric::Cosine,
            neighbour_count: 8,
            self_weight: 0.70,
            rounds: 1,
            max_candidates: 128,
        }
    }
}

impl CandidateGraphRerankConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.neighbour_count == 0
            || self.neighbour_count > MAX_NEIGHBOURS
            || self.rounds == 0
            || self.rounds > MAX_ROUNDS
            || self.max_candidates < MIN_CANDIDATES
            || self.max_candidates > MAX_CANDIDATES
            || !self.self_weight.is_finite()
            || !(MIN_SELF_WEIGHT..=MAX_SELF_WEIGHT).contains(&self.self_weight)
        {
            return Err("candidate graph rerank configuration is invalid".into());
        }
        Ok(())
    }
}

/// One full-precision candidate retrieved by an upstream first stage.
/// `initial_score` must be a finite relevance score in `[0, 1]`, where larger
/// means more relevant.  The restriction avoids silently mixing distances,
/// similarities, or unbounded provider scores during diffusion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateGraphRerankCandidate {
    pub iri: String,
    pub vector: Vec<f32>,
    pub initial_score: f64,
}

/// One independently labelled case.  Labels stay in the caller-supplied input
/// and are reduced to IRI-only rankings before any durable evaluation write.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateGraphRerankCase {
    pub case_id: String,
    pub task_iri: String,
    pub task_family: String,
    pub relevant_iris: Vec<String>,
    pub query_vector: Vec<f32>,
    pub candidates: Vec<CandidateGraphRerankCandidate>,
}

/// A reproducible, caller-supplied offline experiment.  It is accepted from a
/// file or test harness but is never stored in L0 because it contains vectors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateGraphRerankExperiment {
    pub schema_version: u32,
    pub experiment_id: String,
    /// Stable identifier for the experimental algorithm/configuration, not a
    /// user prompt or free-form model description.
    pub candidate_id: String,
    pub config: CandidateGraphRerankConfig,
    pub cases: Vec<CandidateGraphRerankCase>,
}

/// IRI-only rankings generated for one case.  These are safe to pass to the
/// existing durable independent-label evaluator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateGraphRerankOutcome {
    pub case_id: String,
    pub first_stage: OfflineRanking,
    pub exact_reference: OfflineRanking,
    pub graph_diffusion: OfflineRanking,
}

/// Transient execution result.  It contains only IRI rankings and derived
/// scores; source vectors remain in the original experiment object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateGraphRerankExecution {
    pub schema_version: u32,
    pub experiment_id: String,
    pub candidate_id: String,
    pub configuration_digest: String,
    pub outcomes: Vec<CandidateGraphRerankOutcome>,
    #[serde(skip)]
    first_stage_to_exact_cases: Vec<OfflineRetrievalCase>,
    #[serde(skip)]
    exact_to_graph_cases: Vec<OfflineRetrievalCase>,
}

/// Persisted-quality verdicts for the two mandatory comparisons.  An exact
/// rescore is a diagnostic reference; only `graph_vs_exact` may be cited for a
/// future experimental reranker proposal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateGraphRerankAdmission {
    pub exact_reference_vs_first_stage: OfflineRetrievalEvaluation,
    pub graph_diffusion_vs_exact_reference: OfflineRetrievalEvaluation,
}

/// Metadata needed to turn an admitted graph-rerank evaluation into a durable
/// evolution proposal.  It contains no vectors or labels.  The caller must
/// explicitly provide the base and candidate revisions; a proposal remains in
/// the `Proposed` state and cannot alter runtime retrieval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateGraphRerankDeltaProposal {
    pub evaluation_id: String,
    pub candidate_id: String,
    pub source_task_iri: String,
    pub task_family: String,
    pub base_revision: u64,
    pub candidate_revision: u64,
}

impl CandidateGraphRerankDeltaProposal {
    pub fn validate(&self) -> Result<(), String> {
        if !valid_identifier(&self.evaluation_id)
            || !self.evaluation_id.ends_with(":graph-vs-exact")
            || !valid_graph_candidate_id(&self.candidate_id)
            || !valid_iri(&self.source_task_iri)
            || self.task_family.trim().is_empty()
            || self.task_family.chars().count() > MAX_TASK_FAMILY_CHARS
            || self.task_family.chars().any(char::is_control)
            || self.candidate_revision <= self.base_revision
        {
            return Err("candidate graph rerank delta proposal is invalid".into());
        }
        Ok(())
    }
}

impl CandidateGraphRerankExperiment {
    /// Run the bounded experiment without any durable write.
    pub fn execute(&self) -> Result<CandidateGraphRerankExecution, String> {
        self.validate()?;
        let mut outcomes = Vec::with_capacity(self.cases.len());
        let mut first_stage_to_exact_cases = Vec::with_capacity(self.cases.len());
        let mut exact_to_graph_cases = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            case.validate(&self.config)?;
            let result = rerank_case(case, &self.config)?;
            first_stage_to_exact_cases.push(offline_case(
                case,
                result.first_stage.clone(),
                result.exact_reference.clone(),
            ));
            exact_to_graph_cases.push(offline_case(
                case,
                result.exact_reference.clone(),
                result.graph_diffusion.clone(),
            ));
            outcomes.push(result);
        }
        let configuration_digest = configuration_digest(&self.config)?;
        Ok(CandidateGraphRerankExecution {
            schema_version: CANDIDATE_GRAPH_RERANK_SCHEMA_VERSION,
            experiment_id: self.experiment_id.clone(),
            candidate_id: self.candidate_id.clone(),
            configuration_digest,
            outcomes,
            first_stage_to_exact_cases,
            exact_to_graph_cases,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CANDIDATE_GRAPH_RERANK_SCHEMA_VERSION
            || !valid_identifier(&self.experiment_id)
            || !valid_graph_candidate_id(&self.candidate_id)
            || self.experiment_id.chars().count() > MAX_IDENTIFIER_CHARS - 32
            || self.cases.is_empty()
            || self.cases.len() > MAX_CASES
        {
            return Err(
                "candidate graph rerank experiment identity or case count is invalid".into(),
            );
        }
        self.config.validate()?;
        let mut unique_case_ids = HashSet::with_capacity(self.cases.len());
        for case in &self.cases {
            if !unique_case_ids.insert(case.case_id.as_str()) {
                return Err("candidate graph rerank case IDs must be unique".into());
            }
        }
        Ok(())
    }
}

impl CandidateGraphRerankCase {
    fn validate(&self, config: &CandidateGraphRerankConfig) -> Result<(), String> {
        if !valid_identifier(&self.case_id)
            || !valid_iri(&self.task_iri)
            || self.task_family.trim().is_empty()
            || self.task_family.chars().count() > MAX_TASK_FAMILY_CHARS
            || self.task_family.chars().any(char::is_control)
            || self.relevant_iris.is_empty()
            || self.relevant_iris.len() > 128
            || !valid_unique_iris(&self.relevant_iris)
            || self.query_vector.is_empty()
            || self.query_vector.len() > MAX_VECTOR_DIMENSIONS
            || self.query_vector.iter().any(|value| !value.is_finite())
            || self.candidates.len() < MIN_CANDIDATES
            || self.candidates.len() > config.max_candidates
        {
            return Err("candidate graph rerank case is invalid".into());
        }
        if config.metric == CandidateGraphMetric::Cosine && norm(&self.query_vector) <= NORM_EPSILON
        {
            return Err("cosine graph rerank query must have a non-zero norm".into());
        }
        let mut seen_iris = HashSet::with_capacity(self.candidates.len());
        for candidate in &self.candidates {
            if !valid_iri(&candidate.iri)
                || !seen_iris.insert(candidate.iri.as_str())
                || candidate.vector.len() != self.query_vector.len()
                || candidate.vector.iter().any(|value| !value.is_finite())
                || !candidate.initial_score.is_finite()
                || !(0.0..=1.0).contains(&candidate.initial_score)
                || (config.metric == CandidateGraphMetric::Cosine
                    && norm(&candidate.vector) <= NORM_EPSILON)
            {
                return Err("candidate graph rerank candidate is invalid".into());
            }
        }
        Ok(())
    }
}

impl CandidateGraphRerankExecution {
    /// Persist independent-label quality gates.  Both comparisons are retained
    /// for audit, but a graph candidate is eligible for a future evolution
    /// proposal only when the stricter graph-versus-exact verdict is admitted.
    pub fn evaluate_and_persist(
        &self,
        evaluator: &OfflineRetrievalEvaluator,
    ) -> Result<CandidateGraphRerankAdmission, CoreError> {
        let exact_reference_vs_first_stage = evaluator.evaluate_and_persist(
            &format!("{}:exact-reference", self.experiment_id),
            &format!("{}:exact-reference", self.candidate_id),
            &self.first_stage_to_exact_cases,
        )?;
        let graph_diffusion_vs_exact_reference = evaluator.evaluate_and_persist(
            &format!("{}:graph-vs-exact", self.experiment_id),
            &self.candidate_id,
            &self.exact_to_graph_cases,
        )?;
        Ok(CandidateGraphRerankAdmission {
            exact_reference_vs_first_stage,
            graph_diffusion_vs_exact_reference,
        })
    }
}

fn rerank_case(
    case: &CandidateGraphRerankCase,
    config: &CandidateGraphRerankConfig,
) -> Result<CandidateGraphRerankOutcome, String> {
    let first_stage_started = Instant::now();
    let first_stage = ranking(
        &case.candidates,
        case.candidates
            .iter()
            .map(|candidate| candidate.initial_score)
            .collect(),
        elapsed_ms(first_stage_started),
    );

    let exact_started = Instant::now();
    let exact_scores = case
        .candidates
        .iter()
        .map(|candidate| query_affinity(&case.query_vector, &candidate.vector, config.metric))
        .collect::<Vec<_>>();
    let exact_reference = ranking(&case.candidates, exact_scores, elapsed_ms(exact_started));

    let graph_started = Instant::now();
    let graph = build_candidate_graph(&case.candidates, config)?;
    let mut scores = case
        .candidates
        .iter()
        .map(|candidate| candidate.initial_score)
        .collect::<Vec<_>>();
    for _ in 0..config.rounds {
        let previous = scores.clone();
        for (index, neighbours) in graph.iter().enumerate() {
            let mut weighted_score = 0.0;
            let mut total_weight = 0.0;
            for &(neighbour, affinity) in neighbours {
                weighted_score += previous[neighbour] * affinity;
                total_weight += affinity;
            }
            if total_weight > NORM_EPSILON {
                scores[index] = config.self_weight * previous[index]
                    + (1.0 - config.self_weight) * weighted_score / total_weight;
            }
        }
    }
    let graph_diffusion = ranking(&case.candidates, scores, elapsed_ms(graph_started));
    Ok(CandidateGraphRerankOutcome {
        case_id: case.case_id.clone(),
        first_stage,
        exact_reference,
        graph_diffusion,
    })
}

fn build_candidate_graph(
    candidates: &[CandidateGraphRerankCandidate],
    config: &CandidateGraphRerankConfig,
) -> Result<Vec<Vec<(usize, f64)>>, String> {
    let neighbours_per_candidate = config
        .neighbour_count
        .min(candidates.len().saturating_sub(1));
    if neighbours_per_candidate == 0 {
        return Err("candidate graph rerank requires at least two candidates".into());
    }
    let mut graph = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let mut neighbours = candidates
            .iter()
            .enumerate()
            .filter(|(other_index, _)| *other_index != index)
            .map(|(other_index, other)| {
                (
                    other_index,
                    query_affinity(&candidate.vector, &other.vector, config.metric),
                )
            })
            .collect::<Vec<_>>();
        neighbours.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score.total_cmp(left_score).then_with(|| {
                candidates[*left_index]
                    .iri
                    .cmp(&candidates[*right_index].iri)
            })
        });
        neighbours.truncate(neighbours_per_candidate);
        graph.push(neighbours);
    }
    Ok(graph)
}

fn ranking(
    candidates: &[CandidateGraphRerankCandidate],
    scores: Vec<f64>,
    elapsed_ms: u64,
) -> OfflineRanking {
    let mut ranked = candidates
        .iter()
        .zip(scores)
        .map(|(candidate, score)| (candidate.iri.clone(), score))
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_iri, left_score), (right_iri, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_iri.cmp(right_iri))
    });
    OfflineRanking {
        ranked_iris: ranked.into_iter().map(|(iri, _)| iri).collect(),
        elapsed_ms,
    }
}

fn offline_case(
    case: &CandidateGraphRerankCase,
    baseline: OfflineRanking,
    candidate: OfflineRanking,
) -> OfflineRetrievalCase {
    OfflineRetrievalCase {
        case_id: case.case_id.clone(),
        task_iri: case.task_iri.clone(),
        task_family: case.task_family.clone(),
        relevant_iris: case.relevant_iris.clone(),
        baseline,
        candidate,
    }
}

fn query_affinity(query: &[f32], candidate: &[f32], metric: CandidateGraphMetric) -> f64 {
    match metric {
        CandidateGraphMetric::Cosine => {
            let dot = query
                .iter()
                .zip(candidate)
                .map(|(left, right)| *left as f64 * *right as f64)
                .sum::<f64>();
            ((dot / (norm(query) * norm(candidate))) + 1.0).clamp(0.0, 2.0) / 2.0
        }
        CandidateGraphMetric::Euclidean => {
            let distance_squared = query
                .iter()
                .zip(candidate)
                .map(|(left, right)| {
                    let difference = *left as f64 - *right as f64;
                    difference * difference
                })
                .sum::<f64>();
            1.0 / (1.0 + distance_squared.sqrt())
        }
    }
}

fn norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|value| *value as f64 * *value as f64)
        .sum::<f64>()
        .sqrt()
}

fn elapsed_ms(started: Instant) -> u64 {
    let micros = started.elapsed().as_micros().min(u128::from(u64::MAX));
    let rounded = (micros.saturating_add(999) / 1_000) as u64;
    rounded.max(1)
}

fn configuration_digest(config: &CandidateGraphRerankConfig) -> Result<String, String> {
    serde_json::to_vec(config)
        .map(|value| hex::encode(Sha256::digest(value)))
        .map_err(|error| format!("serialize candidate graph rerank configuration: {error}"))
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.chars().count() <= MAX_IDENTIFIER_CHARS
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | ':')
        })
}

fn valid_graph_candidate_id(value: &str) -> bool {
    value.starts_with("candidate-graph-") && valid_identifier(value)
}

fn valid_iri(value: &str) -> bool {
    value.starts_with("iri://")
        && value.chars().count() <= MAX_IDENTIFIER_CHARS
        && !value.chars().any(char::is_control)
}

fn valid_unique_iris(values: &[String]) -> bool {
    values.iter().all(|value| valid_iri(value))
        && values.iter().collect::<HashSet<_>>().len() == values.len()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::offline_retrieval_eval::{
        OfflineRetrievalEvalConfig, OfflineRetrievalEvaluator,
    };
    use crate::memory::l0_store::L0Store;

    use super::*;

    fn candidate(iri: &str, vector: Vec<f32>, initial_score: f64) -> CandidateGraphRerankCandidate {
        CandidateGraphRerankCandidate {
            iri: iri.into(),
            vector,
            initial_score,
        }
    }

    fn case(index: usize, graph_helpful: bool) -> CandidateGraphRerankCase {
        let relevant = format!("iri://evidence/relevant-{index}");
        let cluster_peer = format!("iri://evidence/peer-{index}");
        let distractor = format!("iri://evidence/distractor-{index}");
        CandidateGraphRerankCase {
            case_id: format!("case-{index}"),
            task_iri: format!("iri://task/graph-rerank-{index}"),
            task_family: "planning:v3:intent=inspect;domain=document".into(),
            relevant_iris: vec![relevant.clone()],
            query_vector: vec![1.0, 0.0],
            candidates: vec![
                candidate(
                    &relevant,
                    vec![0.98, 0.02],
                    if graph_helpful { 0.30 } else { 0.90 },
                ),
                candidate(
                    &cluster_peer,
                    vec![0.96, 0.04],
                    if graph_helpful { 0.95 } else { 0.20 },
                ),
                candidate(
                    &distractor,
                    vec![0.0, 1.0],
                    if graph_helpful { 0.80 } else { 0.80 },
                ),
            ],
        }
    }

    fn experiment(graph_helpful: bool) -> CandidateGraphRerankExperiment {
        CandidateGraphRerankExperiment {
            schema_version: CANDIDATE_GRAPH_RERANK_SCHEMA_VERSION,
            experiment_id: if graph_helpful {
                "graph-rerank-helpful-v1"
            } else {
                "graph-rerank-unhelpful-v1"
            }
            .into(),
            candidate_id: "candidate-graph-diffusion-v1".into(),
            config: CandidateGraphRerankConfig {
                neighbour_count: 1,
                self_weight: 0.50,
                max_candidates: 3,
                ..Default::default()
            },
            cases: vec![case(1, graph_helpful), case(2, graph_helpful)],
        }
    }

    fn evaluator(store: Arc<L0Store>) -> OfflineRetrievalEvaluator {
        OfflineRetrievalEvaluator::with_config(
            store,
            OfflineRetrievalEvalConfig {
                cutoff: 2,
                min_cases: 2,
                min_ndcg_improvement: 0.0,
                max_p95_latency_ratio: 10.0,
            },
        )
        .unwrap()
    }

    #[test]
    fn executes_three_deterministic_rankings_without_retaining_vectors() {
        let execution = experiment(true).execute().unwrap();
        assert_eq!(execution.outcomes.len(), 2);
        let outcome = &execution.outcomes[0];
        assert_eq!(outcome.first_stage.ranked_iris.len(), 3);
        assert_eq!(
            outcome.exact_reference.ranked_iris[0],
            "iri://evidence/relevant-1"
        );
        assert_eq!(
            outcome.graph_diffusion.ranked_iris[0],
            "iri://evidence/distractor-1"
        );
        let serialized = serde_json::to_string(&execution).unwrap();
        assert!(!serialized.contains("query_vector"));
        assert!(!serialized.contains("vector"));
    }

    #[test]
    fn rejects_mixed_dimensions_non_finite_scores_and_unsafe_configuration() {
        let mut invalid = experiment(true);
        invalid.cases[0].candidates[1].vector.push(0.0);
        assert!(invalid.execute().is_err());

        let mut invalid = experiment(true);
        invalid.cases[0].candidates[1].initial_score = f64::NAN;
        assert!(invalid.execute().is_err());

        let invalid_config = CandidateGraphRerankConfig {
            self_weight: 0.10,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn exact_reference_and_graph_candidate_are_persisted_as_separate_comparisons() {
        let dir = tempfile::tempdir().unwrap();
        let evaluator = evaluator(Arc::new(
            L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap(),
        ));
        let execution = experiment(true).execute().unwrap();
        let admission = execution.evaluate_and_persist(&evaluator).unwrap();
        assert!(admission.exact_reference_vs_first_stage.admitted);
        assert_eq!(
            admission.graph_diffusion_vs_exact_reference.candidate_id,
            "candidate-graph-diffusion-v1"
        );
        assert!(!admission.graph_diffusion_vs_exact_reference.admitted);
        assert!(admission
            .graph_diffusion_vs_exact_reference
            .rejection_reasons
            .contains(&"insufficient_ndcg_improvement".into()));
    }
}
