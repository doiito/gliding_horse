//! Durable, privacy-minimised execution trajectories for offline learning.
//!
//! A trajectory is evidence about one completed task, not executable memory.
//! It stores identifiers, bounded tool metadata and independent verification
//! status; prompts, LLM responses, tool arguments and tool results stay in the
//! execution journal's explicit debug-only capture path.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::policy_learning::LearningMode;
use crate::core::retrieval_policy::RetrievalPolicyArm;
use crate::core::tracked_action::{ActionStatus, TrackedAction};
use crate::memory::l0_store::L0Store;
use crate::CoreError;

pub const LEARNING_TRAJECTORY_SCHEMA_VERSION: u32 = 1;
pub const LEARNING_TRAJECTORY_PREFIX: &str = "iri://learning/trajectory/";
const MAX_TRAJECTORY_STEPS: usize = 128;
const MAX_EVIDENCE_REFERENCES: usize = 32;
const MAX_SELECTED_REFERENCES: usize = 64;
const MAX_IDENTIFIER_CHARS: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrajectoryToolStep {
    pub tool_name: String,
    pub agent_role: String,
    pub succeeded: bool,
    pub duration_ms: u64,
    pub substantive_effect: bool,
    pub created_files: u16,
    pub modified_files: u16,
    pub read_files: u16,
}

impl From<&TrackedAction> for TrajectoryToolStep {
    fn from(action: &TrackedAction) -> Self {
        Self {
            tool_name: action.tool_name.clone(),
            agent_role: action.agent_role.clone(),
            succeeded: matches!(action.status, ActionStatus::Success),
            duration_ms: (action.duration_secs.max(0.0) * 1_000.0).min(u64::MAX as f64) as u64,
            substantive_effect: action.substantive_effect,
            created_files: action.files_created.len().min(u16::MAX as usize) as u16,
            modified_files: action.files_modified.len().min(u16::MAX as usize) as u16,
            read_files: action.files_read.len().min(u16::MAX as usize) as u16,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningTrajectoryOutcome {
    pub terminal_status: String,
    pub reward: f32,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub elapsed_ms: u64,
    pub independent_ca_aa_pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningTrajectory {
    pub schema_version: u32,
    pub task_iri: String,
    pub task_family: String,
    pub mode: LearningMode,
    pub policy_action: String,
    pub policy_candidates: Vec<String>,
    pub policy_model_version: u64,
    pub policy_explored: bool,
    pub selected_skill_iris: Vec<String>,
    pub selected_knowledge_fragment_iris: Vec<String>,
    pub evidence_iris: Vec<String>,
    pub tool_steps: Vec<TrajectoryToolStep>,
    pub outcome: LearningTrajectoryOutcome,
    pub created_at: DateTime<Utc>,
}

impl LearningTrajectory {
    pub fn storage_iri(&self) -> String {
        let digest = Sha256::digest(self.task_iri.as_bytes());
        format!("{LEARNING_TRAJECTORY_PREFIX}{}", hex::encode(digest))
    }

    /// A trajectory may become a retrieval/training candidate only after an
    /// independent CA/AA pass. The raw trajectory remains available for
    /// diagnostics even when this returns false.
    pub fn reusable_candidate(&self) -> bool {
        self.mode == LearningMode::Active
            && self.outcome.independent_ca_aa_pass
            && matches!(
                self.outcome.terminal_status.as_str(),
                "success" | "completed"
            )
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != LEARNING_TRAJECTORY_SCHEMA_VERSION {
            return Err(format!(
                "unsupported learning trajectory schema {}",
                self.schema_version
            ));
        }
        if self.task_iri.trim().is_empty() || self.task_family.trim().is_empty() {
            return Err("trajectory task identity and family are required".into());
        }
        if !self.task_iri.starts_with("iri://")
            || self.task_iri.chars().count() > MAX_IDENTIFIER_CHARS
            || self.task_family.chars().count() > MAX_IDENTIFIER_CHARS
        {
            return Err("trajectory identity exceeds its safe bounds".into());
        }
        let Some(action) = RetrievalPolicyArm::parse(&self.policy_action) else {
            return Err("trajectory policy action is not whitelisted".into());
        };
        if self.policy_candidates.is_empty()
            || self
                .policy_candidates
                .iter()
                .any(|candidate| RetrievalPolicyArm::parse(candidate).is_none())
            || !self
                .policy_candidates
                .iter()
                .any(|candidate| candidate == action.as_str())
        {
            return Err("trajectory policy candidates are invalid".into());
        }
        if self.policy_candidates.len()
            != self
                .policy_candidates
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
        {
            return Err("trajectory policy candidates must be unique".into());
        }
        if self.tool_steps.len() > MAX_TRAJECTORY_STEPS
            || self.evidence_iris.len() > MAX_EVIDENCE_REFERENCES
        {
            return Err("trajectory exceeds its bounded retention limits".into());
        }
        if self
            .evidence_iris
            .iter()
            .any(|iri| !valid_iri_reference(iri))
        {
            return Err("trajectory evidence must use stable IRI references".into());
        }
        if !valid_reference_collection(&self.selected_skill_iris)
            || !valid_reference_collection(&self.selected_knowledge_fragment_iris)
        {
            return Err("trajectory selected references must be unique stable IRIs".into());
        }
        if self.tool_steps.iter().any(|step| {
            step.tool_name.trim().is_empty()
                || step.agent_role.trim().is_empty()
                || step.tool_name.chars().count() > 128
                || step.agent_role.chars().count() > 32
        }) {
            return Err("trajectory tool metadata exceeds its safe bounds".into());
        }
        if self.outcome.terminal_status.trim().is_empty()
            || self.outcome.terminal_status.chars().count() > 64
        {
            return Err("trajectory terminal status is invalid".into());
        }
        if !self.outcome.reward.is_finite() {
            return Err("trajectory reward must be finite".into());
        }
        Ok(())
    }
}

fn valid_iri_reference(value: &str) -> bool {
    value.starts_with("iri://") && value.chars().count() <= MAX_IDENTIFIER_CHARS
}

fn valid_reference_collection(values: &[String]) -> bool {
    values.len() <= MAX_SELECTED_REFERENCES
        && values.iter().all(|value| valid_iri_reference(value))
        && values
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == values.len()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryPersistResult {
    Stored { iri: String },
    AlreadyPresent { iri: String },
}

/// Single-process durable trajectory store. L0 supplies the crash-safe write;
/// deterministic task keys make retried finalisation idempotent without
/// allowing a late retry to overwrite a completed task record.
#[derive(Clone)]
pub struct LearningTrajectoryStore {
    l0: Arc<L0Store>,
}

impl LearningTrajectoryStore {
    pub fn new(l0: Arc<L0Store>) -> Self {
        Self { l0 }
    }

    pub fn persist(
        &self,
        trajectory: &LearningTrajectory,
    ) -> Result<TrajectoryPersistResult, CoreError> {
        trajectory.validate().map_err(invalid_trajectory)?;
        let iri = trajectory.storage_iri();
        if let Some(existing) = self.l0.retrieve(&iri)? {
            let restored =
                serde_json::from_str::<LearningTrajectory>(&existing.content).map_err(|error| {
                    CoreError::StorageError {
                        message: format!("stored learning trajectory is corrupt: {error}"),
                    }
                })?;
            restored.validate().map_err(invalid_trajectory)?;
            if restored.task_iri != trajectory.task_iri {
                return Err(CoreError::StorageError {
                    message: "learning trajectory key collision".into(),
                });
            }
            return Ok(TrajectoryPersistResult::AlreadyPresent { iri });
        }
        let content =
            serde_json::to_string(trajectory).map_err(|error| CoreError::StorageError {
                message: format!("serialize learning trajectory: {error}"),
            })?;
        self.l0.store(&iri, &content)?;
        Ok(TrajectoryPersistResult::Stored { iri })
    }

    pub fn load(&self, task_iri: &str) -> Result<Option<LearningTrajectory>, CoreError> {
        let digest = Sha256::digest(task_iri.as_bytes());
        let iri = format!("{LEARNING_TRAJECTORY_PREFIX}{}", hex::encode(digest));
        self.l0
            .retrieve(&iri)?
            .map(|entry| {
                let trajectory = serde_json::from_str::<LearningTrajectory>(&entry.content)
                    .map_err(|error| CoreError::StorageError {
                        message: format!("deserialize learning trajectory: {error}"),
                    })?;
                trajectory.validate().map_err(invalid_trajectory)?;
                Ok(trajectory)
            })
            .transpose()
    }

    pub fn recent_for_family(
        &self,
        task_family: &str,
        limit: usize,
    ) -> Result<Vec<LearningTrajectory>, CoreError> {
        let mut trajectories = self
            .l0
            .scan_iri_prefix(LEARNING_TRAJECTORY_PREFIX, limit.max(1).saturating_mul(8))?
            .into_iter()
            .filter_map(|entry| serde_json::from_str::<LearningTrajectory>(&entry.content).ok())
            .filter(|trajectory| trajectory.task_family == task_family)
            .filter(|trajectory| trajectory.validate().is_ok())
            .collect::<Vec<_>>();
        trajectories.sort_by(|left, right| right.created_at.cmp(&left.created_at));
        trajectories.truncate(limit);
        Ok(trajectories)
    }
}

fn invalid_trajectory(message: String) -> CoreError {
    CoreError::StorageError {
        message: format!("invalid learning trajectory: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trajectory(task_iri: &str) -> LearningTrajectory {
        LearningTrajectory {
            schema_version: LEARNING_TRAJECTORY_SCHEMA_VERSION,
            task_iri: task_iri.into(),
            task_family: "planning:v3:intent=inspect;domain=document".into(),
            mode: LearningMode::Active,
            policy_action: "knowledge_first".into(),
            policy_candidates: vec!["baseline".into(), "knowledge_first".into()],
            policy_model_version: 4,
            policy_explored: false,
            selected_skill_iris: vec!["iri://skills/report".into()],
            selected_knowledge_fragment_iris: vec!["iri://knowledge/report".into()],
            evidence_iris: vec!["iri://learning/ca-audit/example".into()],
            tool_steps: vec![TrajectoryToolStep {
                tool_name: "file_read".into(),
                agent_role: "DA".into(),
                succeeded: true,
                duration_ms: 5,
                substantive_effect: false,
                created_files: 0,
                modified_files: 0,
                read_files: 1,
            }],
            outcome: LearningTrajectoryOutcome {
                terminal_status: "success".into(),
                reward: 0.8,
                prompt_tokens: 20,
                completion_tokens: 10,
                elapsed_ms: 30,
                independent_ca_aa_pass: true,
            },
            created_at: Utc::now(),
        }
    }

    #[test]
    fn trajectory_is_idempotent_and_never_stores_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let store = LearningTrajectoryStore::new(Arc::new(
            L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap(),
        ));
        let record = trajectory("iri://task/trajectory-1");
        let first = store.persist(&record).unwrap();
        assert!(matches!(first, TrajectoryPersistResult::Stored { .. }));
        assert!(matches!(
            store.persist(&record).unwrap(),
            TrajectoryPersistResult::AlreadyPresent { .. }
        ));
        let loaded = store.load("iri://task/trajectory-1").unwrap().unwrap();
        assert!(loaded.reusable_candidate());
        let raw = serde_json::to_string(&loaded).unwrap();
        assert!(!raw.contains("tool_args"));
        assert!(!raw.contains("prompt_response"));
    }

    #[test]
    fn trajectory_rejects_unknown_policy_and_unstable_evidence() {
        let mut record = trajectory("iri://task/trajectory-invalid");
        record.policy_action = "change_hnsw".into();
        assert!(record.validate().is_err());
        record.policy_action = "baseline".into();
        record.policy_candidates = vec!["baseline".into()];
        record.evidence_iris = vec!["not-an-iri".into()];
        assert!(record.validate().is_err());
    }

    #[test]
    fn trajectory_rejects_unbounded_or_non_identifier_metadata() {
        let mut record = trajectory("iri://task/trajectory-bounds");
        record.selected_skill_iris = vec!["sensitive/path".into()];
        assert!(record.validate().is_err());
        record.selected_skill_iris = vec!["iri://skills/valid".into()];
        record.tool_steps[0].tool_name = "x".repeat(129);
        assert!(record.validate().is_err());
    }
}
