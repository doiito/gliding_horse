//! Auditable, single-node gate for learning and skill evolution deltas.
//!
//! The gate records a constrained state machine around an already-approved
//! candidate. It never applies a graph mutation, modifies an index or expands
//! effects by itself. A caller must provide durable evidence references before
//! a delta can progress from shadow validation to active use.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::retrieval_policy::RetrievalPolicyArm;
use crate::core::{
    offline_retrieval_eval::OfflineRetrievalEvaluation,
    offline_retrieval_eval::OFFLINE_RETRIEVAL_EVAL_PREFIX,
};
use crate::memory::l0_store::L0Store;
use crate::CoreError;

pub const EVOLUTION_DELTA_SCHEMA_VERSION: u32 = 1;
pub const EVOLUTION_DELTA_PREFIX: &str = "iri://learning/evolution-delta/";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionDeltaTarget {
    RetrievalPolicy,
    /// An offline-admitted candidate-stage reranker.  This lifecycle records
    /// evidence only; the gate has no path that changes online ranking.
    RetrievalRerankerCandidate,
    SkillKnowledgeCandidate,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionDeltaState {
    Proposed,
    ShadowValidated,
    Active,
    Frozen,
    RolledBack,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvolutionDeltaTransition {
    pub from: EvolutionDeltaState,
    pub to: EvolutionDeltaState,
    pub reason: String,
    pub at: DateTime<Utc>,
    pub human_approved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvolutionDelta {
    pub schema_version: u32,
    pub delta_id: String,
    pub source_task_iri: String,
    pub task_family: String,
    pub target: EvolutionDeltaTarget,
    pub policy_action: Option<String>,
    pub base_revision: u64,
    pub candidate_revision: u64,
    pub evidence_iris: Vec<String>,
    pub state: EvolutionDeltaState,
    pub transitions: Vec<EvolutionDeltaTransition>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl EvolutionDelta {
    pub fn proposed_policy(
        source_task_iri: &str,
        task_family: &str,
        policy_action: &str,
        base_revision: u64,
        candidate_revision: u64,
        evidence_iris: Vec<String>,
    ) -> Result<Self, String> {
        let delta = Self {
            schema_version: EVOLUTION_DELTA_SCHEMA_VERSION,
            delta_id: format!("delta_{}", uuid::Uuid::new_v4().hyphenated()),
            source_task_iri: source_task_iri.to_string(),
            task_family: task_family.to_string(),
            target: EvolutionDeltaTarget::RetrievalPolicy,
            policy_action: Some(policy_action.to_string()),
            base_revision,
            candidate_revision,
            evidence_iris,
            state: EvolutionDeltaState::Proposed,
            transitions: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        delta.validate()?;
        Ok(delta)
    }

    /// Create an evidence-bound proposal for a candidate-stage reranker.
    /// A gate will additionally verify that at least one referenced offline
    /// evaluation is durable and admitted before accepting this proposal.
    pub fn proposed_retrieval_reranker(
        source_task_iri: &str,
        task_family: &str,
        base_revision: u64,
        candidate_revision: u64,
        evidence_iris: Vec<String>,
    ) -> Result<Self, String> {
        let delta = Self {
            schema_version: EVOLUTION_DELTA_SCHEMA_VERSION,
            delta_id: format!("delta_{}", uuid::Uuid::new_v4().hyphenated()),
            source_task_iri: source_task_iri.to_string(),
            task_family: task_family.to_string(),
            target: EvolutionDeltaTarget::RetrievalRerankerCandidate,
            policy_action: None,
            base_revision,
            candidate_revision,
            evidence_iris,
            state: EvolutionDeltaState::Proposed,
            transitions: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        delta.validate()?;
        Ok(delta)
    }

    pub fn storage_iri(&self) -> String {
        format!("{EVOLUTION_DELTA_PREFIX}{}", self.delta_id)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EVOLUTION_DELTA_SCHEMA_VERSION {
            return Err(format!(
                "unsupported evolution delta schema {}",
                self.schema_version
            ));
        }
        if self.delta_id.trim().is_empty()
            || self.source_task_iri.trim().is_empty()
            || self.task_family.trim().is_empty()
            || self.evidence_iris.is_empty()
            || self.evidence_iris.len() > 32
        {
            return Err("evolution delta identity and evidence are required".into());
        }
        if self
            .evidence_iris
            .iter()
            .any(|iri| !iri.starts_with("iri://"))
        {
            return Err("evolution evidence must use stable IRI references".into());
        }
        if self.candidate_revision <= self.base_revision {
            return Err("candidate revision must advance beyond its base revision".into());
        }
        match self.target {
            EvolutionDeltaTarget::RetrievalPolicy => {
                let Some(action) = self.policy_action.as_deref() else {
                    return Err("policy delta requires a policy action".into());
                };
                if RetrievalPolicyArm::parse(action).is_none() || action == "baseline" {
                    return Err("policy delta action must be a non-baseline whitelist arm".into());
                }
            }
            EvolutionDeltaTarget::RetrievalRerankerCandidate => {
                if self.policy_action.is_some() {
                    return Err("retrieval reranker delta cannot carry a policy action".into());
                }
                if !self
                    .evidence_iris
                    .iter()
                    .any(|iri| iri.starts_with(OFFLINE_RETRIEVAL_EVAL_PREFIX))
                {
                    return Err(
                        "retrieval reranker delta requires an offline evaluation reference".into(),
                    );
                }
            }
            EvolutionDeltaTarget::SkillKnowledgeCandidate => {
                if self.policy_action.is_some() {
                    return Err("skill knowledge delta cannot carry a policy action".into());
                }
            }
        }
        if self
            .transitions
            .iter()
            .any(|transition| transition.reason.trim().is_empty())
        {
            return Err("evolution transitions require a reason".into());
        }
        let mut expected_state = EvolutionDeltaState::Proposed;
        for transition in &self.transitions {
            if transition.from != expected_state
                || !valid_transition(transition.from, transition.to, transition.human_approved)
            {
                return Err("evolution transition history is invalid".into());
            }
            expected_state = transition.to;
        }
        if self.state != expected_state {
            return Err("evolution state does not match its transition history".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaPersistResult {
    Stored { iri: String },
    AlreadyPresent { iri: String },
}

#[derive(Clone)]
pub struct EvolutionDeltaGate {
    l0: Arc<L0Store>,
}

impl EvolutionDeltaGate {
    pub fn new(l0: Arc<L0Store>) -> Self {
        Self { l0 }
    }

    pub fn propose(&self, delta: &EvolutionDelta) -> Result<DeltaPersistResult, CoreError> {
        delta.validate().map_err(invalid_delta)?;
        self.validate_retrieval_reranker_evidence(delta)?;
        let iri = delta.storage_iri();
        if let Some(existing) = self.l0.retrieve(&iri)? {
            let restored = deserialize_delta(&existing.content)?;
            if restored == *delta {
                return Ok(DeltaPersistResult::AlreadyPresent { iri });
            }
            return Err(CoreError::StorageError {
                message: "evolution delta ID already exists with different content".into(),
            });
        }
        self.store(&iri, delta)?;
        Ok(DeltaPersistResult::Stored { iri })
    }

    pub fn load(&self, delta_id: &str) -> Result<Option<EvolutionDelta>, CoreError> {
        self.l0
            .retrieve(&format!("{EVOLUTION_DELTA_PREFIX}{delta_id}"))?
            .map(|entry| deserialize_delta(&entry.content))
            .transpose()
    }

    /// List valid local deltas for operator inspection. Corrupt records are
    /// deliberately excluded rather than being guessed into a lifecycle.
    pub fn list(&self, limit: usize) -> Result<Vec<EvolutionDelta>, CoreError> {
        let mut deltas = self
            .l0
            .scan_iri_prefix(EVOLUTION_DELTA_PREFIX, limit.max(1))?
            .into_iter()
            .filter_map(|entry| deserialize_delta(&entry.content).ok())
            .collect::<Vec<_>>();
        deltas.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        deltas.truncate(limit);
        Ok(deltas)
    }

    /// Advance the durable lifecycle. Active use is permitted only after a
    /// shadow-validation transition; recovery from a freeze always requires a
    /// human approval flag and becomes a rollback, never a silent reactivation.
    pub fn transition(
        &self,
        delta_id: &str,
        next: EvolutionDeltaState,
        reason: &str,
        human_approved: bool,
    ) -> Result<EvolutionDelta, CoreError> {
        let iri = format!("{EVOLUTION_DELTA_PREFIX}{delta_id}");
        let entry = self
            .l0
            .retrieve(&iri)?
            .ok_or_else(|| CoreError::StorageError {
                message: format!("evolution delta not found: {delta_id}"),
            })?;
        let mut delta = deserialize_delta(&entry.content)?;
        if !valid_transition(delta.state, next, human_approved) {
            return Err(CoreError::StorageError {
                message: format!(
                    "invalid evolution transition {:?} -> {:?}",
                    delta.state, next
                ),
            });
        }
        let reason = reason.trim();
        if reason.is_empty() {
            return Err(CoreError::StorageError {
                message: "evolution transition reason is required".into(),
            });
        }
        let previous = delta.state;
        delta.state = next;
        delta.updated_at = Utc::now();
        delta.transitions.push(EvolutionDeltaTransition {
            from: previous,
            to: next,
            reason: reason.chars().take(240).collect(),
            at: delta.updated_at,
            human_approved,
        });
        delta.validate().map_err(invalid_delta)?;
        self.store(&iri, &delta)?;
        Ok(delta)
    }

    /// Freeze every active policy delta for one family. This is idempotent and
    /// is the only operation suitable for automatic health-monitor action.
    pub fn freeze_active_policy_family(
        &self,
        task_family: &str,
        reason: &str,
    ) -> Result<Vec<EvolutionDelta>, CoreError> {
        self.freeze_active_retrieval_family_by_target(task_family, reason, |target| {
            target == EvolutionDeltaTarget::RetrievalPolicy
        })
    }

    /// Freeze every active retrieval-related delta for one family.  This is
    /// the safety boundary used by health monitoring so a future candidate
    /// reranker cannot remain active after the family has been degraded.
    pub fn freeze_active_retrieval_family(
        &self,
        task_family: &str,
        reason: &str,
    ) -> Result<Vec<EvolutionDelta>, CoreError> {
        self.freeze_active_retrieval_family_by_target(task_family, reason, |target| {
            matches!(
                target,
                EvolutionDeltaTarget::RetrievalPolicy
                    | EvolutionDeltaTarget::RetrievalRerankerCandidate
            )
        })
    }

    fn freeze_active_retrieval_family_by_target(
        &self,
        task_family: &str,
        reason: &str,
        target_matches: impl Fn(EvolutionDeltaTarget) -> bool,
    ) -> Result<Vec<EvolutionDelta>, CoreError> {
        let active_ids = self
            .l0
            .scan_iri_prefix(EVOLUTION_DELTA_PREFIX, 1_024)?
            .into_iter()
            .filter_map(|entry| deserialize_delta(&entry.content).ok())
            .filter(|delta| {
                target_matches(delta.target)
                    && delta.task_family == task_family
                    && delta.state == EvolutionDeltaState::Active
            })
            .map(|delta| delta.delta_id)
            .collect::<Vec<_>>();
        active_ids
            .iter()
            .map(|id| self.transition(id, EvolutionDeltaState::Frozen, reason, false))
            .collect()
    }

    /// Record an explicit local-operator approval to close a frozen delta as
    /// rolled back. This does not unfreeze its policy context or activate any
    /// replacement; a fresh controlled evaluation is still required.
    pub fn rollback_frozen_with_approval(
        &self,
        delta_id: &str,
        approver: &str,
        comment: Option<&str>,
    ) -> Result<EvolutionDelta, CoreError> {
        let approver = approver.trim();
        if approver.is_empty()
            || approver.chars().count() > 128
            || approver.chars().any(char::is_control)
        {
            return Err(CoreError::StorageError {
                message: "evolution rollback approver identity is invalid".into(),
            });
        }
        let comment = comment.unwrap_or("").trim();
        if comment.chars().count() > 160 || comment.chars().any(char::is_control) {
            return Err(CoreError::StorageError {
                message: "evolution rollback comment is invalid".into(),
            });
        }
        let reason = if comment.is_empty() {
            format!("human-approved rollback by {approver}")
        } else {
            format!("human-approved rollback by {approver}: {comment}")
        };
        self.transition(delta_id, EvolutionDeltaState::RolledBack, &reason, true)
    }

    fn store(&self, iri: &str, delta: &EvolutionDelta) -> Result<(), CoreError> {
        self.l0.store(
            iri,
            &serde_json::to_string(delta).map_err(|error| CoreError::StorageError {
                message: format!("serialize evolution delta: {error}"),
            })?,
        )
    }

    fn validate_retrieval_reranker_evidence(
        &self,
        delta: &EvolutionDelta,
    ) -> Result<(), CoreError> {
        if delta.target != EvolutionDeltaTarget::RetrievalRerankerCandidate {
            return Ok(());
        }
        let mut admitted = false;
        for evidence_iri in delta
            .evidence_iris
            .iter()
            .filter(|iri| iri.starts_with(OFFLINE_RETRIEVAL_EVAL_PREFIX))
        {
            let entry = self
                .l0
                .retrieve(evidence_iri)?
                .ok_or_else(|| CoreError::StorageError {
                    message: format!("reranker evaluation evidence not found: {evidence_iri}"),
                })?;
            let evaluation = serde_json::from_str::<OfflineRetrievalEvaluation>(&entry.content)
                .map_err(|error| CoreError::StorageError {
                    message: format!("reranker evaluation evidence is corrupt: {error}"),
                })?;
            evaluation
                .validate()
                .map_err(|message| CoreError::StorageError {
                    message: format!("reranker evaluation evidence is invalid: {message}"),
                })?;
            admitted |= evaluation.admitted;
        }
        if admitted {
            Ok(())
        } else {
            Err(CoreError::StorageError {
                message: "retrieval reranker delta requires an admitted offline evaluation".into(),
            })
        }
    }
}

fn valid_transition(
    current: EvolutionDeltaState,
    next: EvolutionDeltaState,
    human_approved: bool,
) -> bool {
    match (current, next) {
        (EvolutionDeltaState::Proposed, EvolutionDeltaState::ShadowValidated)
        | (EvolutionDeltaState::Proposed, EvolutionDeltaState::Rejected)
        | (EvolutionDeltaState::ShadowValidated, EvolutionDeltaState::Rejected)
        | (EvolutionDeltaState::Active, EvolutionDeltaState::Frozen) => true,
        (EvolutionDeltaState::ShadowValidated, EvolutionDeltaState::Active) => true,
        (EvolutionDeltaState::Frozen, EvolutionDeltaState::RolledBack) => human_approved,
        _ => false,
    }
}

fn deserialize_delta(content: &str) -> Result<EvolutionDelta, CoreError> {
    let delta = serde_json::from_str::<EvolutionDelta>(content).map_err(|error| {
        CoreError::StorageError {
            message: format!("deserialize evolution delta: {error}"),
        }
    })?;
    delta.validate().map_err(invalid_delta)?;
    Ok(delta)
}

fn invalid_delta(message: String) -> CoreError {
    CoreError::StorageError {
        message: format!("invalid evolution delta: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use crate::core::offline_retrieval_eval::{
        OfflineRanking, OfflineRetrievalCase, OfflineRetrievalEvalConfig, OfflineRetrievalEvaluator,
    };

    use super::*;

    fn delta() -> EvolutionDelta {
        EvolutionDelta::proposed_policy(
            "iri://task/evolution",
            "planning:v3:intent=inspect;domain=document",
            "knowledge_first",
            2,
            3,
            vec!["iri://learning/evaluations/evolution".into()],
        )
        .unwrap()
    }

    #[test]
    fn delta_requires_shadow_validation_and_approved_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let gate = EvolutionDeltaGate::new(Arc::new(
            L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap(),
        ));
        let delta = delta();
        gate.propose(&delta).unwrap();
        assert!(gate
            .transition(
                &delta.delta_id,
                EvolutionDeltaState::Active,
                "skip shadow",
                false,
            )
            .is_err());
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::ShadowValidated,
            "paired evidence accepted",
            false,
        )
        .unwrap();
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::Active,
            "promotion gate accepted",
            false,
        )
        .unwrap();
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::Frozen,
            "health regression",
            false,
        )
        .unwrap();
        assert!(gate
            .transition(
                &delta.delta_id,
                EvolutionDeltaState::RolledBack,
                "no approval",
                false,
            )
            .is_err());
        let rolled_back = gate
            .transition(
                &delta.delta_id,
                EvolutionDeltaState::RolledBack,
                "approved recovery",
                true,
            )
            .unwrap();
        assert_eq!(rolled_back.state, EvolutionDeltaState::RolledBack);
    }

    #[test]
    fn automatic_freeze_is_family_scoped_and_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let gate = EvolutionDeltaGate::new(Arc::new(
            L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap(),
        ));
        let delta = delta();
        gate.propose(&delta).unwrap();
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::ShadowValidated,
            "shadow",
            false,
        )
        .unwrap();
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::Active,
            "active",
            false,
        )
        .unwrap();
        let frozen = gate
            .freeze_active_policy_family(&delta.task_family, "health")
            .unwrap();
        assert_eq!(frozen.len(), 1);
        assert_eq!(frozen[0].state, EvolutionDeltaState::Frozen);
        assert!(gate
            .freeze_active_policy_family(&delta.task_family, "health")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn delta_rejects_forged_active_state_without_a_lifecycle() {
        let mut forged = delta();
        forged.state = EvolutionDeltaState::Active;
        assert!(forged.validate().is_err());
    }

    #[test]
    fn approved_rollback_is_audited_and_never_unfreezes_a_delta() {
        let dir = tempfile::tempdir().unwrap();
        let gate = EvolutionDeltaGate::new(Arc::new(
            L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap(),
        ));
        let delta = delta();
        gate.propose(&delta).unwrap();
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::ShadowValidated,
            "shadow",
            false,
        )
        .unwrap();
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::Active,
            "active",
            false,
        )
        .unwrap();
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::Frozen,
            "health regression",
            false,
        )
        .unwrap();
        let rolled_back = gate
            .rollback_frozen_with_approval(
                &delta.delta_id,
                "operator@example.test",
                Some("reviewed"),
            )
            .unwrap();
        assert_eq!(rolled_back.state, EvolutionDeltaState::RolledBack);
        assert!(rolled_back
            .transitions
            .last()
            .unwrap()
            .reason
            .contains("operator@example.test"));
        assert_eq!(
            gate.list(10).unwrap()[0].state,
            EvolutionDeltaState::RolledBack
        );
    }

    #[test]
    fn reranker_delta_requires_a_durable_admitted_offline_evaluation() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap());
        let gate = EvolutionDeltaGate::new(l0.clone());
        let missing = EvolutionDelta::proposed_retrieval_reranker(
            "iri://task/reranker",
            "planning:v3:intent=inspect;domain=document",
            2,
            3,
            vec!["iri://learning/offline-retrieval-eval/missing".into()],
        )
        .unwrap();
        assert!(gate.propose(&missing).is_err());

        let evaluator = OfflineRetrievalEvaluator::with_config(
            l0,
            OfflineRetrievalEvalConfig {
                cutoff: 1,
                min_cases: 1,
                min_ndcg_improvement: 0.0,
                max_p95_latency_ratio: 2.0,
            },
        )
        .unwrap();
        let evaluation = evaluator
            .evaluate_and_persist(
                "reranker-admitted-v1",
                "candidate-graph-diffusion-v1",
                &[OfflineRetrievalCase {
                    case_id: "case-1".into(),
                    task_iri: "iri://task/reranker".into(),
                    task_family: "planning:v3:intent=inspect;domain=document".into(),
                    relevant_iris: vec!["iri://evidence/relevant".into()],
                    baseline: OfflineRanking {
                        ranked_iris: vec!["iri://evidence/distractor".into()],
                        elapsed_ms: 10,
                    },
                    candidate: OfflineRanking {
                        ranked_iris: vec!["iri://evidence/relevant".into()],
                        elapsed_ms: 10,
                    },
                }],
            )
            .unwrap();
        assert!(evaluation.admitted);
        let delta = EvolutionDelta::proposed_retrieval_reranker(
            "iri://task/reranker",
            "planning:v3:intent=inspect;domain=document",
            2,
            3,
            vec![evaluation.storage_iri()],
        )
        .unwrap();
        assert!(matches!(
            gate.propose(&delta),
            Ok(DeltaPersistResult::Stored { .. })
        ));
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::ShadowValidated,
            "offline and shadow evidence accepted",
            false,
        )
        .unwrap();
        gate.transition(
            &delta.delta_id,
            EvolutionDeltaState::Active,
            "future runtime binding approved separately",
            false,
        )
        .unwrap();
        let frozen = gate
            .freeze_active_retrieval_family(&delta.task_family, "health regression")
            .unwrap();
        assert_eq!(frozen.len(), 1);
        assert_eq!(frozen[0].state, EvolutionDeltaState::Frozen);
    }
}
