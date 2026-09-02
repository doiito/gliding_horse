//! Offline-only admission gate for experimental retrieval re-rankers.
//!
//! This module deliberately accepts rankings, never a runnable re-ranker. It
//! makes a candidate prove quality and latency on an independently labelled
//! corpus before a separate architecture review can consider any runtime use.
//! Inputs contain stable IRIs and measurements only: prompts, embeddings,
//! tool arguments, LLM output and document bodies are not persisted here.

use std::collections::HashSet;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::memory::l0_store::L0Store;
use crate::CoreError;

pub const OFFLINE_RETRIEVAL_EVAL_SCHEMA_VERSION: u32 = 1;
pub const OFFLINE_RETRIEVAL_EVAL_PREFIX: &str = "iri://learning/offline-retrieval-eval/";
const MAX_CASES: usize = 10_000;
const MAX_RANKED_ITEMS: usize = 1_024;
const MAX_RELEVANT_ITEMS: usize = 128;
const MAX_IDENTIFIER_CHARS: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OfflineRanking {
    pub ranked_iris: Vec<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OfflineRetrievalCase {
    pub case_id: String,
    pub task_iri: String,
    pub task_family: String,
    /// Independently labelled relevant identifiers.  A candidate ranking is
    /// compared against this set and cannot supply its own relevance signal.
    pub relevant_iris: Vec<String>,
    pub baseline: OfflineRanking,
    pub candidate: OfflineRanking,
}

impl OfflineRetrievalCase {
    pub fn validate(&self) -> Result<(), String> {
        if self.case_id.trim().is_empty()
            || self.case_id.chars().count() > MAX_IDENTIFIER_CHARS
            || !valid_iri(&self.task_iri)
            || self.task_family.trim().is_empty()
            || self.task_family.chars().count() > MAX_IDENTIFIER_CHARS
        {
            return Err("offline retrieval case identity is invalid".into());
        }
        if self.relevant_iris.is_empty()
            || self.relevant_iris.len() > MAX_RELEVANT_ITEMS
            || !valid_unique_iris(&self.relevant_iris, MAX_RELEVANT_ITEMS)
        {
            return Err("offline retrieval relevance labels are invalid".into());
        }
        for ranking in [&self.baseline, &self.candidate] {
            if ranking.ranked_iris.len() > MAX_RANKED_ITEMS
                || !valid_unique_iris(&ranking.ranked_iris, MAX_RANKED_ITEMS)
            {
                return Err("offline retrieval ranking is invalid".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OfflineRetrievalEvalConfig {
    pub cutoff: usize,
    pub min_cases: usize,
    /// Candidate must improve average NDCG@k by at least this amount and may
    /// not lower Recall@k.  Both requirements avoid a precision-only win that
    /// reduces evidence coverage.
    pub min_ndcg_improvement: f64,
    pub max_p95_latency_ratio: f64,
}

impl Default for OfflineRetrievalEvalConfig {
    fn default() -> Self {
        Self {
            cutoff: 10,
            min_cases: 20,
            min_ndcg_improvement: 0.02,
            max_p95_latency_ratio: 1.10,
        }
    }
}

impl OfflineRetrievalEvalConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.cutoff == 0
            || self.cutoff > MAX_RANKED_ITEMS
            || self.min_cases == 0
            || self.min_cases > MAX_CASES
            || !self.min_ndcg_improvement.is_finite()
            || !(0.0..=1.0).contains(&self.min_ndcg_improvement)
            || !self.max_p95_latency_ratio.is_finite()
            || self.max_p95_latency_ratio < 1.0
            || self.max_p95_latency_ratio > 10.0
        {
            return Err("offline retrieval evaluation configuration is invalid".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalQualityMetrics {
    pub mean_recall_at_k: f64,
    pub mean_mrr_at_k: f64,
    pub mean_ndcg_at_k: f64,
    pub p95_elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OfflineRetrievalEvaluation {
    pub schema_version: u32,
    pub evaluation_id: String,
    pub candidate_id: String,
    /// SHA-256 of the IRI-only labelled corpus and paired rankings.  It makes
    /// retries idempotent while proving that an evaluation ID was not reused
    /// for a different experiment.
    pub corpus_digest: String,
    pub cutoff: usize,
    pub cases_evaluated: u32,
    pub baseline: RetrievalQualityMetrics,
    pub candidate: RetrievalQualityMetrics,
    pub recall_delta: f64,
    pub ndcg_delta: f64,
    pub latency_ratio: f64,
    /// This is an offline quality result only. `admitted` never changes the
    /// online retrieval path and must be reviewed separately.
    pub admitted: bool,
    pub rejection_reasons: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl OfflineRetrievalEvaluation {
    pub fn storage_iri(&self) -> String {
        let digest = Sha256::digest(self.evaluation_id.as_bytes());
        format!("{OFFLINE_RETRIEVAL_EVAL_PREFIX}{}", hex::encode(digest))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != OFFLINE_RETRIEVAL_EVAL_SCHEMA_VERSION
            || self.evaluation_id.trim().is_empty()
            || self.evaluation_id.chars().count() > MAX_IDENTIFIER_CHARS
            || self.candidate_id.trim().is_empty()
            || self.candidate_id.chars().count() > MAX_IDENTIFIER_CHARS
            || self.corpus_digest.len() != 64
            || !self
                .corpus_digest
                .chars()
                .all(|character| character.is_ascii_hexdigit())
            || self.cutoff == 0
            || self.cases_evaluated == 0
            || !self.recall_delta.is_finite()
            || !self.ndcg_delta.is_finite()
            || !self.latency_ratio.is_finite()
        {
            return Err("offline retrieval evaluation is invalid".into());
        }
        for metrics in [&self.baseline, &self.candidate] {
            if !metrics.mean_recall_at_k.is_finite()
                || !metrics.mean_mrr_at_k.is_finite()
                || !metrics.mean_ndcg_at_k.is_finite()
                || !(0.0..=1.0).contains(&metrics.mean_recall_at_k)
                || !(0.0..=1.0).contains(&metrics.mean_mrr_at_k)
                || !(0.0..=1.0).contains(&metrics.mean_ndcg_at_k)
            {
                return Err("offline retrieval metric is invalid".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct OfflineRetrievalEvaluator {
    l0: Arc<L0Store>,
    config: OfflineRetrievalEvalConfig,
}

impl OfflineRetrievalEvaluator {
    pub fn new(l0: Arc<L0Store>) -> Self {
        Self::with_config(l0, OfflineRetrievalEvalConfig::default())
            .expect("default offline retrieval configuration is valid")
    }

    pub fn with_config(
        l0: Arc<L0Store>,
        config: OfflineRetrievalEvalConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { l0, config })
    }

    /// Load one immutable evaluation by its caller-facing identifier.  Corrupt
    /// or unsupported records are rejected rather than being used as evidence
    /// for a later evolution proposal.
    pub fn load(
        &self,
        evaluation_id: &str,
    ) -> Result<Option<OfflineRetrievalEvaluation>, CoreError> {
        if evaluation_id.trim().is_empty() || evaluation_id.chars().count() > MAX_IDENTIFIER_CHARS {
            return Err(invalid_evaluation("evaluation identity is invalid".into()));
        }
        let iri = format!(
            "{OFFLINE_RETRIEVAL_EVAL_PREFIX}{}",
            hex::encode(Sha256::digest(evaluation_id.as_bytes()))
        );
        self.l0
            .retrieve(&iri)?
            .map(|entry| {
                let evaluation = serde_json::from_str::<OfflineRetrievalEvaluation>(&entry.content)
                    .map_err(|error| CoreError::StorageError {
                        message: format!("stored offline retrieval evaluation is corrupt: {error}"),
                    })?;
                evaluation.validate().map_err(invalid_evaluation)?;
                Ok(evaluation)
            })
            .transpose()
    }

    /// List valid immutable verdicts for local audit. Corrupt records are not
    /// guessed into a promotion decision; an operator can inspect storage
    /// separately if recovery is required.
    pub fn list(&self, limit: usize) -> Result<Vec<OfflineRetrievalEvaluation>, CoreError> {
        let mut evaluations = self
            .l0
            .scan_iri_prefix(OFFLINE_RETRIEVAL_EVAL_PREFIX, limit.max(1))?
            .into_iter()
            .filter_map(|entry| {
                let evaluation =
                    serde_json::from_str::<OfflineRetrievalEvaluation>(&entry.content).ok()?;
                evaluation.validate().ok()?;
                Some(evaluation)
            })
            .collect::<Vec<_>>();
        evaluations.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        evaluations.truncate(limit);
        Ok(evaluations)
    }

    /// Evaluate a paired corpus and persist its immutable result.  Reusing an
    /// evaluation ID is only allowed for byte-equivalent output, preventing a
    /// later experiment from silently replacing a prior verdict.
    pub fn evaluate_and_persist(
        &self,
        evaluation_id: &str,
        candidate_id: &str,
        cases: &[OfflineRetrievalCase],
    ) -> Result<OfflineRetrievalEvaluation, CoreError> {
        let report = self.evaluate(evaluation_id, candidate_id, cases)?;
        let iri = report.storage_iri();
        if let Some(existing) = self.l0.retrieve(&iri)? {
            let restored = serde_json::from_str::<OfflineRetrievalEvaluation>(&existing.content)
                .map_err(|error| CoreError::StorageError {
                    message: format!("stored offline retrieval evaluation is corrupt: {error}"),
                })?;
            restored.validate().map_err(invalid_evaluation)?;
            if restored.same_evaluation(&report) {
                return Ok(restored);
            }
            return Err(CoreError::StorageError {
                message: "offline retrieval evaluation ID already exists with different content"
                    .into(),
            });
        }
        self.l0.store(
            &iri,
            &serde_json::to_string(&report).map_err(|error| CoreError::StorageError {
                message: format!("serialize offline retrieval evaluation: {error}"),
            })?,
        )?;
        Ok(report)
    }

    pub fn evaluate(
        &self,
        evaluation_id: &str,
        candidate_id: &str,
        cases: &[OfflineRetrievalCase],
    ) -> Result<OfflineRetrievalEvaluation, CoreError> {
        if evaluation_id.trim().is_empty()
            || candidate_id.trim().is_empty()
            || evaluation_id.chars().count() > MAX_IDENTIFIER_CHARS
            || candidate_id.chars().count() > MAX_IDENTIFIER_CHARS
            || cases.is_empty()
            || cases.len() > MAX_CASES
        {
            return Err(invalid_evaluation(
                "evaluation identity or case count is invalid".into(),
            ));
        }
        let mut unique_cases = HashSet::with_capacity(cases.len());
        let mut baseline_scores = Vec::with_capacity(cases.len());
        let mut candidate_scores = Vec::with_capacity(cases.len());
        for case in cases {
            case.validate().map_err(invalid_evaluation)?;
            if !unique_cases.insert(case.case_id.as_str()) {
                return Err(invalid_evaluation(
                    "offline retrieval case IDs must be unique".into(),
                ));
            }
            baseline_scores.push(score(
                &case.relevant_iris,
                &case.baseline,
                self.config.cutoff,
            ));
            candidate_scores.push(score(
                &case.relevant_iris,
                &case.candidate,
                self.config.cutoff,
            ));
        }
        let corpus_digest = corpus_digest(cases)?;
        let baseline = aggregate(&baseline_scores);
        let candidate = aggregate(&candidate_scores);
        let recall_delta = candidate.mean_recall_at_k - baseline.mean_recall_at_k;
        let ndcg_delta = candidate.mean_ndcg_at_k - baseline.mean_ndcg_at_k;
        let latency_ratio = if baseline.p95_elapsed_ms == 0 {
            if candidate.p95_elapsed_ms == 0 {
                1.0
            } else {
                f64::MAX
            }
        } else {
            candidate.p95_elapsed_ms as f64 / baseline.p95_elapsed_ms as f64
        };
        let mut rejection_reasons = Vec::new();
        if cases.len() < self.config.min_cases {
            rejection_reasons.push("insufficient_labelled_cases".into());
        }
        if recall_delta < 0.0 {
            rejection_reasons.push("recall_regression".into());
        }
        if ndcg_delta < self.config.min_ndcg_improvement {
            rejection_reasons.push("insufficient_ndcg_improvement".into());
        }
        if !latency_ratio.is_finite() || latency_ratio > self.config.max_p95_latency_ratio {
            rejection_reasons.push("p95_latency_budget_exceeded".into());
        }
        let report = OfflineRetrievalEvaluation {
            schema_version: OFFLINE_RETRIEVAL_EVAL_SCHEMA_VERSION,
            evaluation_id: evaluation_id.to_string(),
            candidate_id: candidate_id.to_string(),
            corpus_digest,
            cutoff: self.config.cutoff,
            cases_evaluated: cases.len().min(u32::MAX as usize) as u32,
            baseline,
            candidate,
            recall_delta,
            ndcg_delta,
            latency_ratio,
            admitted: rejection_reasons.is_empty(),
            rejection_reasons,
            created_at: Utc::now(),
        };
        report.validate().map_err(invalid_evaluation)?;
        Ok(report)
    }
}

impl OfflineRetrievalEvaluation {
    fn same_evaluation(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.evaluation_id == other.evaluation_id
            && self.candidate_id == other.candidate_id
            && self.corpus_digest == other.corpus_digest
            && self.cutoff == other.cutoff
            && self.cases_evaluated == other.cases_evaluated
            && self.baseline == other.baseline
            && self.candidate == other.candidate
            && self.recall_delta == other.recall_delta
            && self.ndcg_delta == other.ndcg_delta
            && self.latency_ratio == other.latency_ratio
            && self.admitted == other.admitted
            && self.rejection_reasons == other.rejection_reasons
    }
}

#[derive(Clone, Copy)]
struct CaseScore {
    recall: f64,
    reciprocal_rank: f64,
    ndcg: f64,
    elapsed_ms: u64,
}

fn score(relevant_iris: &[String], ranking: &OfflineRanking, cutoff: usize) -> CaseScore {
    let relevant = relevant_iris
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut hits = 0usize;
    let mut reciprocal_rank = 0.0;
    let mut dcg = 0.0;
    for (index, iri) in ranking.ranked_iris.iter().take(cutoff).enumerate() {
        if relevant.contains(iri.as_str()) {
            hits += 1;
            if reciprocal_rank == 0.0 {
                reciprocal_rank = 1.0 / (index + 1) as f64;
            }
            dcg += 1.0 / ((index + 2) as f64).log2();
        }
    }
    let ideal_hits = relevant.len().min(cutoff);
    let ideal_dcg = (0..ideal_hits)
        .map(|index| 1.0 / ((index + 2) as f64).log2())
        .sum::<f64>();
    CaseScore {
        recall: hits as f64 / relevant.len() as f64,
        reciprocal_rank,
        ndcg: if ideal_dcg == 0.0 {
            0.0
        } else {
            dcg / ideal_dcg
        },
        elapsed_ms: ranking.elapsed_ms,
    }
}

fn aggregate(scores: &[CaseScore]) -> RetrievalQualityMetrics {
    let count = scores.len() as f64;
    let mut latency = scores
        .iter()
        .map(|score| score.elapsed_ms)
        .collect::<Vec<_>>();
    latency.sort_unstable();
    let p95_index = ((latency.len() * 95).saturating_add(99) / 100)
        .saturating_sub(1)
        .min(latency.len().saturating_sub(1));
    RetrievalQualityMetrics {
        mean_recall_at_k: scores.iter().map(|score| score.recall).sum::<f64>() / count,
        mean_mrr_at_k: scores
            .iter()
            .map(|score| score.reciprocal_rank)
            .sum::<f64>()
            / count,
        mean_ndcg_at_k: scores.iter().map(|score| score.ndcg).sum::<f64>() / count,
        p95_elapsed_ms: latency[p95_index],
    }
}

fn valid_iri(value: &str) -> bool {
    value.starts_with("iri://") && value.chars().count() <= MAX_IDENTIFIER_CHARS
}

fn corpus_digest(cases: &[OfflineRetrievalCase]) -> Result<String, CoreError> {
    let canonical = serde_json::to_vec(cases).map_err(|error| CoreError::StorageError {
        message: format!("serialize offline retrieval corpus digest: {error}"),
    })?;
    Ok(hex::encode(Sha256::digest(canonical)))
}

fn valid_unique_iris(values: &[String], maximum: usize) -> bool {
    values.len() <= maximum
        && values.iter().all(|value| valid_iri(value))
        && values.iter().collect::<HashSet<_>>().len() == values.len()
}

fn invalid_evaluation(message: String) -> CoreError {
    CoreError::StorageError {
        message: format!("invalid offline retrieval evaluation: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(index: usize, candidate_first: bool, candidate_latency: u64) -> OfflineRetrievalCase {
        let relevant = format!("iri://evidence/{index}");
        let distractor = format!("iri://distractor/{index}");
        OfflineRetrievalCase {
            case_id: format!("case-{index}"),
            task_iri: format!("iri://task/offline-{index}"),
            task_family: "planning:v3:intent=inspect;domain=document".into(),
            relevant_iris: vec![relevant.clone()],
            baseline: OfflineRanking {
                ranked_iris: vec![distractor.clone(), relevant.clone()],
                elapsed_ms: 10,
            },
            candidate: OfflineRanking {
                ranked_iris: if candidate_first {
                    vec![relevant, distractor]
                } else {
                    vec![distractor, relevant]
                },
                elapsed_ms: candidate_latency,
            },
        }
    }

    fn evaluator(store: Arc<L0Store>) -> OfflineRetrievalEvaluator {
        OfflineRetrievalEvaluator::with_config(
            store,
            OfflineRetrievalEvalConfig {
                cutoff: 2,
                min_cases: 2,
                min_ndcg_improvement: 0.1,
                max_p95_latency_ratio: 1.2,
            },
        )
        .unwrap()
    }

    #[test]
    fn independent_labels_admit_a_better_bounded_candidate() {
        let dir = tempfile::tempdir().unwrap();
        let evaluator = evaluator(Arc::new(
            L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap(),
        ));
        let report = evaluator
            .evaluate_and_persist(
                "eval-good",
                "offline-metadata-rerank-v1",
                &[case(1, true, 11), case(2, true, 11)],
            )
            .unwrap();
        assert!(report.admitted, "{report:?}");
        assert!(report.ndcg_delta > 0.1);
        assert_eq!(
            evaluator
                .evaluate_and_persist(
                    "eval-good",
                    "offline-metadata-rerank-v1",
                    &[case(1, true, 11), case(2, true, 11)]
                )
                .unwrap(),
            report
        );
        assert_eq!(evaluator.load("eval-good").unwrap(), Some(report));
        assert_eq!(evaluator.list(10).unwrap().len(), 1);
    }

    #[test]
    fn regression_or_latency_prevents_admission() {
        let dir = tempfile::tempdir().unwrap();
        let evaluator = evaluator(Arc::new(
            L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap(),
        ));
        let report = evaluator
            .evaluate(
                "eval-bad",
                "offline-gnn-rerank-v1",
                &[case(1, false, 20), case(2, false, 20)],
            )
            .unwrap();
        assert!(!report.admitted);
        assert!(report
            .rejection_reasons
            .contains(&"insufficient_ndcg_improvement".into()));
        assert!(report
            .rejection_reasons
            .contains(&"p95_latency_budget_exceeded".into()));
    }

    #[test]
    fn raw_or_duplicate_identifiers_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let evaluator = evaluator(Arc::new(
            L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap(),
        ));
        let mut invalid = case(1, true, 10);
        invalid.relevant_iris = vec!["raw document text".into()];
        assert!(evaluator
            .evaluate("eval-invalid", "candidate", &[invalid])
            .is_err());
    }
}
