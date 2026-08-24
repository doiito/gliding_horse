//! Constrained policy learning for the task-planning loop.
//!
//! A trainable, constrained contextual policy for the task-planning loop.
//!
//! The model is an online policy-gradient contextual bandit: it learns a
//! softmax policy from task outcomes, can be replayed offline, and persists
//! its weights in L0. It is deliberately not allowed to create tools,
//! override CA/AA audits, or select outside the caller-provided safe arms.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::memory::l0_store::L0Store;

const PREFIX: &str = "iri://learning/policy/";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArmStats {
    pub pulls: u32,
    pub reward_sum: f32,
}

impl ArmStats {
    pub fn mean(&self) -> f32 {
        if self.pulls == 0 {
            0.0
        } else {
            self.reward_sum / self.pulls as f32
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyState {
    pub context: String,
    pub arms: HashMap<String, ArmStats>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyChoice {
    pub context: String,
    pub action: String,
    pub used_fallback: bool,
    pub confidence: f32,
    pub explored: bool,
    /// Safe action set used for the gradient update; supplied by SA.
    pub candidates: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyObservation {
    pub context: String,
    pub action: String,
    pub reward: f32,
    pub explored: bool,
    #[serde(default)]
    pub candidates: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Compact trainable policy model. The feature map is deterministic so a
/// persisted model and an offline replay produce the same policy after a
/// restart. Hashing keeps the model dependency-free and bounded in size.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainablePolicyModel {
    pub weights: Vec<f32>,
    pub feature_dim: usize,
    pub learning_rate: f32,
    pub l2: f32,
    pub updates: u64,
    pub reward_mean: f32,
    pub reward_count: u32,
}

/// One decision in a multi-step task trajectory. The candidate list is the
/// safe action boundary supplied by SA for that step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryStep {
    pub context: String,
    pub action: String,
    pub candidates: Vec<String>,
    pub reward: f32,
    pub terminal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingMetrics {
    pub updates: u64,
    pub steps: u32,
    pub discounted_return: f32,
    pub average_reward: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyEvaluation {
    pub samples: u32,
    pub baseline_return: f32,
    pub candidate_return: f32,
    pub improvement: f32,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyVersion {
    pub version: u64,
    pub model: TrainablePolicyModel,
    pub evaluation: Option<PolicyEvaluation>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Deployment gate for learned policies. A model is promoted only when it
/// has enough independent evidence and does not regress against the baseline.
#[derive(Debug, Clone, Copy)]
pub struct PolicyGate {
    pub min_samples: u32,
    pub min_improvement: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyDriftReport {
    pub baseline_mean: f32,
    pub recent_mean: f32,
    pub delta: f32,
    pub drifted: bool,
}

impl Default for PolicyGate {
    fn default() -> Self {
        Self { min_samples: 5, min_improvement: 0.0 }
    }
}

impl PolicyGate {
    pub fn assess(&self, baseline_return: f32, candidate_return: f32, samples: u32) -> PolicyEvaluation {
        let improvement = candidate_return - baseline_return;
        let accepted = samples >= self.min_samples && improvement >= self.min_improvement;
        PolicyEvaluation {
            samples,
            baseline_return,
            candidate_return,
            improvement,
            accepted,
            reason: if samples < self.min_samples {
                "insufficient_evidence".into()
            } else if improvement < self.min_improvement {
                "regression_or_no_improvement".into()
            } else {
                "accepted".into()
            },
        }
    }
}

impl Default for TrainablePolicyModel {
    fn default() -> Self {
        Self {
            weights: vec![0.0; 128],
            feature_dim: 128,
            learning_rate: 0.08,
            l2: 0.0005,
            updates: 0,
            reward_mean: 0.0,
            reward_count: 0,
        }
    }
}

impl TrainablePolicyModel {
    pub fn with_hyperparameters(mut self, learning_rate: f32, l2: f32) -> Self {
        self.learning_rate = learning_rate.clamp(0.001, 0.5);
        self.l2 = l2.clamp(0.0, 0.1);
        self
    }

    fn feature_indices(&self, context: &str, action: &str) -> [usize; 4] {
        let keys = [
            format!("bias:{action}"),
            format!("context:{context}"),
            format!("pair:{context}\x1f{action}"),
            format!("action:{action}"),
        ];
        keys.map(|key| {
            let digest = sha2::Sha256::digest(key.as_bytes());
            let mut bytes = [0_u8; 8];
            bytes.copy_from_slice(&digest[..8]);
            u64::from_le_bytes(bytes) as usize % self.feature_dim
        })
    }

    pub fn score(&self, context: &str, action: &str) -> f32 {
        self.feature_indices(context, action)
            .iter()
            .map(|index| self.weights[*index])
            .sum()
    }

    pub fn probabilities(&self, context: &str, candidates: &[String]) -> Vec<f32> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let scores: Vec<f32> = candidates.iter().map(|a| self.score(context, a)).collect();
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|score| (*score - max_score).exp()).collect();
        let total = exp.iter().sum::<f32>().max(f32::EPSILON);
        exp.into_iter().map(|value| value / total).collect()
    }

    pub fn greedy_action(&self, context: &str, candidates: &[String]) -> Option<String> {
        candidates
            .iter()
            .max_by(|left, right| {
                self.score(context, left)
                    .partial_cmp(&self.score(context, right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    pub fn evaluate(&self, observations: &[PolicyObservation]) -> f32 {
        if observations.is_empty() {
            return 0.0;
        }
        observations
            .iter()
            .map(|observation| {
                let candidates = if observation.candidates.is_empty() {
                    vec![observation.action.clone()]
                } else {
                    observation.candidates.clone()
                };
                if self.greedy_action(&observation.context, &candidates).as_deref()
                    == Some(observation.action.as_str())
                {
                    observation.reward
                } else {
                    0.0
                }
            })
            .sum::<f32>()
            / observations.len() as f32
    }

    pub fn detect_drift(rewards: &[f32], window: usize, threshold: f32) -> PolicyDriftReport {
        let window = window.max(1).min(rewards.len().max(1));
        let baseline_count = rewards.len().saturating_sub(window);
        let baseline = &rewards[..baseline_count];
        let recent = &rewards[baseline_count..];
        let mean = |values: &[f32]| {
            if values.is_empty() { 0.0 } else { values.iter().sum::<f32>() / values.len() as f32 }
        };
        let baseline_mean = mean(baseline);
        let recent_mean = mean(recent);
        let delta = recent_mean - baseline_mean;
        PolicyDriftReport {
            baseline_mean,
            recent_mean,
            delta,
            drifted: !baseline.is_empty() && delta < -threshold.abs(),
        }
    }

    /// One online policy-gradient update using the observed reward as the
    /// return. The moving reward mean is a baseline, reducing update noise.
    pub fn train(&mut self, context: &str, action: &str, candidates: &[String], reward: f32) {
        if candidates.is_empty() || !candidates.iter().any(|candidate| candidate == action) {
            return;
        }
        let reward = reward.clamp(-1.0, 1.0);
        let probabilities = self.probabilities(context, candidates);
        let baseline = self.reward_mean;
        let advantage = (reward - baseline).clamp(-2.0, 2.0);
        for (candidate, probability) in candidates.iter().zip(probabilities) {
            let target = if candidate == action { 1.0 } else { 0.0 };
            let gradient = advantage * (target - probability);
            for index in self.feature_indices(context, candidate) {
                self.weights[index] = (self.weights[index]
                    + self.learning_rate * (gradient - self.l2 * self.weights[index]))
                    .clamp(-8.0, 8.0);
            }
        }
        self.reward_count = self.reward_count.saturating_add(1);
        let count = self.reward_count as f32;
        self.reward_mean += (reward - self.reward_mean) / count;
        self.updates = self.updates.saturating_add(1);
    }

    /// Train on a complete trajectory using discounted returns. This provides
    /// temporal credit assignment instead of treating every terminal task as
    /// a one-step bandit observation.
    pub fn train_trajectory(&mut self, steps: &[TrajectoryStep], gamma: f32) -> TrainingMetrics {
        let gamma = gamma.clamp(0.5, 0.999);
        let mut discounted_return = 0.0;
        let mut discount = 1.0;
        for step in steps {
            discounted_return += discount * step.reward.clamp(-1.0, 1.0);
            discount *= gamma;
        }
        for (index, step) in steps.iter().enumerate() {
            let mut future_return = 0.0;
            let mut future_discount = 1.0;
            for future in steps.iter().skip(index) {
                future_return += future_discount * future.reward.clamp(-1.0, 1.0);
                future_discount *= gamma;
            }
            self.train(&step.context, &step.action, &step.candidates, future_return.clamp(-1.0, 1.0));
        }
        TrainingMetrics {
            updates: self.updates,
            steps: steps.len() as u32,
            discounted_return,
            average_reward: if steps.is_empty() { 0.0 } else { discounted_return / steps.len() as f32 },
        }
    }
}

#[derive(Clone)]
pub struct ConstrainedPolicy {
    states: HashMap<String, PolicyState>,
    store: Option<Arc<L0Store>>,
    model: TrainablePolicyModel,
    model_version: u64,
    rollback_model: Option<TrainablePolicyModel>,
    /// Exploration is bounded and only selects among safe hint-ordering arms.
    exploration_rate: f32,
    min_observations: u32,
}

impl Default for ConstrainedPolicy {
    fn default() -> Self {
        Self {
            states: HashMap::new(),
            store: None,
            model: TrainablePolicyModel::default(),
            model_version: 0,
            rollback_model: None,
            exploration_rate: 0.05,
            min_observations: 2,
        }
    }
}

impl ConstrainedPolicy {
    pub fn with_persistence(mut self, store: Arc<L0Store>) -> Self {
        if let Ok(entries) = store.scan_iri_prefix(PREFIX, usize::MAX) {
            self.states = entries
                .into_iter()
                .filter_map(|entry| {
                    if entry.iri == format!("{PREFIX}model") {
                        self.model = serde_json::from_str::<TrainablePolicyModel>(&entry.content)
                            .unwrap_or_default();
                        None
                    } else if entry.iri == format!("{PREFIX}version") {
                        self.model_version = entry.content.parse::<u64>().unwrap_or(0);
                        None
                    } else {
                        serde_json::from_str::<PolicyState>(&entry.content)
                            .ok()
                            .map(|state| (state.context.clone(), state))
                    }
                })
                .collect();
        }
        self.store = Some(store);
        self
    }

    pub fn with_exploration(mut self, rate: f32) -> Self {
        self.exploration_rate = rate.clamp(0.0, 0.2);
        self
    }

    pub fn choose(&mut self, context: &str, candidates: &[String], fallback: &str) -> PolicyChoice {
        let model_scores: HashMap<String, f32> = candidates
            .iter()
            .map(|candidate| (candidate.clone(), self.model.score(context, candidate)))
            .collect();
        let safe_fallback = if candidates.iter().any(|candidate| candidate == fallback) {
            fallback.to_string()
        } else {
            candidates.first().cloned().unwrap_or_else(|| fallback.to_string())
        };
        let state = self.states.entry(context.to_string()).or_insert_with(|| PolicyState {
            context: context.to_string(),
            arms: HashMap::new(),
            updated_at: chrono::Utc::now(),
        });
        let observed: Vec<&ArmStats> = candidates
            .iter()
            .filter_map(|candidate| state.arms.get(candidate))
            .collect();
        let total_pulls: u32 = observed.iter().map(|stats| stats.pulls).sum();
        let enough_evidence = total_pulls >= self.min_observations;
        let unexplored = candidates.iter().find(|candidate| {
            state.arms.get(*candidate).map(|stats| stats.pulls).unwrap_or(0) == 0
        });
        let interval = (1.0 / self.exploration_rate.max(0.0001)).ceil() as u32;
        let should_explore = enough_evidence
            && unexplored.is_some()
            && total_pulls % interval == 0;
        let action = if !enough_evidence {
            safe_fallback.clone()
        } else if should_explore {
            unexplored.cloned().unwrap_or_else(|| safe_fallback.clone())
        } else {
            candidates
                .iter()
                .max_by(|a, b| {
                    let av = model_scores.get(*a).copied().unwrap_or(0.0);
                    let bv = model_scores.get(*b).copied().unwrap_or(0.0);
                    av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned()
                .unwrap_or(safe_fallback.clone())
        };
        let best = candidates
            .iter()
            .filter_map(|candidate| state.arms.get(candidate).map(ArmStats::mean))
            .fold(0.0_f32, f32::max);
        PolicyChoice {
            context: context.to_string(),
            action,
            used_fallback: !enough_evidence,
            confidence: if enough_evidence { (best.max(0.0) / 1.0).clamp(0.0, 1.0) } else { 0.0 },
            explored: should_explore,
            candidates: candidates.to_vec(),
        }
    }

    pub fn record_reward(&mut self, choice: &PolicyChoice, reward: f32) -> Result<(), String> {
        let state = self.states.entry(choice.context.clone()).or_insert_with(|| PolicyState {
            context: choice.context.clone(),
            arms: HashMap::new(),
            updated_at: chrono::Utc::now(),
        });
        let stats = state.arms.entry(choice.action.clone()).or_insert(ArmStats {
            pulls: 0,
            reward_sum: 0.0,
        });
        stats.pulls = stats.pulls.saturating_add(1);
        stats.reward_sum += reward.clamp(-1.0, 1.0);
        state.updated_at = chrono::Utc::now();
        // The historical arm statistics remain available for auditability;
        // the trainable model is the policy used for future selections.
        self.model.train(
            &choice.context,
            &choice.action,
            &choice.candidates,
            reward,
        );
        if let Some(store) = &self.store {
            let key = format!("{PREFIX}{}", hex::encode(sha2::Sha256::digest(choice.context.as_bytes())));
            let content = serde_json::to_string(state).map_err(|error| error.to_string())?;
            store.store(&key, &content).map_err(|error| error.to_string())?;
            let observation = PolicyObservation {
                context: choice.context.clone(),
                action: choice.action.clone(),
                reward: reward.clamp(-1.0, 1.0),
                explored: choice.explored,
                candidates: choice.candidates.clone(),
                created_at: chrono::Utc::now(),
            };
            let observation_key = format!("{PREFIX}observations/{}", uuid::Uuid::new_v4().hyphenated());
            let observation_content = serde_json::to_string(&observation)
                .map_err(|error| error.to_string())?;
            store.store(&observation_key, &observation_content)
                .map_err(|error| error.to_string())?;
            let model_content = serde_json::to_string(&self.model).map_err(|error| error.to_string())?;
            store.store(&format!("{PREFIX}model"), &model_content)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Train a complete multi-step trajectory and keep a rollback snapshot.
    /// Promotion is intentionally explicit: callers can run PolicyGate on an
    /// evaluation holdout before retaining the new version.
    pub fn train_trajectory(&mut self, steps: &[TrajectoryStep], gamma: f32) -> Result<TrainingMetrics, String> {
        self.rollback_model = Some(self.model.clone());
        let metrics = self.model.train_trajectory(steps, gamma);
        self.model_version = self.model_version.saturating_add(1);
        self.persist_model()?;
        Ok(metrics)
    }

    pub fn train_trajectory_gated(
        &mut self,
        steps: &[TrajectoryStep],
        holdout: &[PolicyObservation],
        gamma: f32,
        gate: PolicyGate,
    ) -> Result<(TrainingMetrics, PolicyEvaluation), String> {
        let mut candidate = self.model.clone();
        let metrics = candidate.train_trajectory(steps, gamma);
        let baseline_return = self.model.evaluate(holdout);
        let candidate_return = candidate.evaluate(holdout);
        let evaluation = gate.assess(baseline_return, candidate_return, holdout.len() as u32);
        if evaluation.accepted {
            self.rollback_model = Some(self.model.clone());
            self.model = candidate;
            self.model_version = self.model_version.saturating_add(1);
            self.persist_model()?;
        }
        Ok((metrics, evaluation))
    }

    pub fn rollback(&mut self) -> Result<bool, String> {
        let Some(previous) = self.rollback_model.take() else { return Ok(false); };
        self.model = previous;
        self.model_version = self.model_version.saturating_add(1);
        self.persist_model()?;
        Ok(true)
    }

    pub fn model_version(&self) -> u64 { self.model_version }

    fn persist_model(&self) -> Result<(), String> {
        if let Some(store) = &self.store {
            let content = serde_json::to_string(&self.model).map_err(|error| error.to_string())?;
            store.store(&format!("{PREFIX}model"), &content).map_err(|error| error.to_string())?;
            store
                .store(&format!("{PREFIX}version"), &self.model_version.to_string())
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    pub fn replay(observations: &[PolicyObservation]) -> Self {
        let mut policy = Self::default();
        for observation in observations {
            let choice = PolicyChoice {
                context: observation.context.clone(),
                action: observation.action.clone(),
                used_fallback: false,
                confidence: 0.0,
                explored: observation.explored,
                candidates: if observation.candidates.is_empty() {
                    vec![observation.action.clone()]
                } else {
                    observation.candidates.clone()
                },
            };
            let _ = policy.record_reward(&choice, observation.reward);
        }
        policy
    }

    pub fn load_observations(store: &L0Store) -> Result<Vec<PolicyObservation>, String> {
        let entries = store
            .scan_iri_prefix(&format!("{PREFIX}observations/"), usize::MAX)
            .map_err(|error| error.to_string())?;
        Ok(entries
            .into_iter()
            .filter_map(|entry| serde_json::from_str(&entry.content).ok())
            .collect())
    }

    pub fn state(&self, context: &str) -> Option<&PolicyState> {
        self.states.get(context)
    }

    pub fn model(&self) -> &TrainablePolicyModel {
        &self.model
    }
}

use sha2::Digest;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_is_used_until_evidence_exists() {
        let mut policy = ConstrainedPolicy::default();
        let choice = policy.choose("task:code", &["baseline".into(), "knowledge_first".into()], "baseline");
        assert_eq!(choice.action, "baseline");
        assert!(choice.used_fallback);
        assert!(!choice.explored);
    }

    #[test]
    fn repeated_rewards_select_the_better_safe_arm_and_survive_restart() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(l0.clone());
        let choice = policy.choose("task:code", &["baseline".into(), "knowledge_first".into()], "baseline");
        policy.record_reward(&choice, 0.1).unwrap();
        let choice = PolicyChoice { action: "knowledge_first".into(), ..choice };
        policy.record_reward(&choice, 0.9).unwrap();
        let selected = policy.choose("task:code", &["baseline".into(), "knowledge_first".into()], "baseline");
        assert_eq!(selected.action, "knowledge_first");

        let restored = ConstrainedPolicy::default().with_persistence(l0);
        assert!(restored.state("task:code").is_some());
        assert_eq!(restored.model().updates, 2);
    }

    #[test]
    fn observations_replay_into_a_fresh_policy() {
        let now = chrono::Utc::now();
        let observations = vec![
            PolicyObservation { context: "task:code".into(), action: "baseline".into(), reward: 0.2, explored: false, candidates: vec!["baseline".into(), "knowledge_first".into()], created_at: now },
            PolicyObservation { context: "task:code".into(), action: "knowledge_first".into(), reward: 0.9, explored: true, candidates: vec!["baseline".into(), "knowledge_first".into()], created_at: now },
        ];
        let policy = ConstrainedPolicy::replay(&observations);
        assert_eq!(policy.state("task:code").unwrap().arms["knowledge_first"].pulls, 1);
        assert_eq!(policy.state("task:code").unwrap().arms["baseline"].pulls, 1);
    }

    #[test]
    fn trainable_model_learns_a_better_action() {
        let mut model = TrainablePolicyModel::default().with_hyperparameters(0.2, 0.0);
        let candidates = vec!["baseline".into(), "knowledge_first".into()];
        for _ in 0..20 {
            model.train("task:code", "knowledge_first", &candidates, 1.0);
        }
        assert!(model.updates >= 20);
        assert!(model.score("task:code", "knowledge_first") > model.score("task:code", "baseline"));
        assert!(model.probabilities("task:code", &candidates)[1] > 0.5);
    }

    #[test]
    fn trajectory_training_assigns_credit_and_can_rollback() {
        let mut policy = ConstrainedPolicy::default();
        let candidates = vec!["baseline".into(), "knowledge_first".into()];
        let before = policy.model().clone();
        let metrics = policy.train_trajectory(&[
            TrajectoryStep { context: "task:code".into(), action: "knowledge_first".into(), candidates: candidates.clone(), reward: 0.2, terminal: false },
            TrajectoryStep { context: "task:code".into(), action: "knowledge_first".into(), candidates, reward: 1.0, terminal: true },
        ], 0.9).unwrap();
        assert_eq!(metrics.steps, 2);
        assert!(policy.model().updates > before.updates);
        assert!(policy.rollback().unwrap());
        assert_eq!(policy.model().updates, before.updates);
    }

    #[test]
    fn promoted_version_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(store.clone());
        let candidates = vec!["pdca".into()];
        policy
            .train_trajectory(&[TrajectoryStep {
                context: "orchestration:code".into(),
                action: "pdca".into(),
                candidates,
                reward: 1.0,
                terminal: true,
            }], 0.9)
            .unwrap();
        let restored = ConstrainedPolicy::default().with_persistence(store);
        assert_eq!(restored.model_version(), 1);
    }

    #[test]
    fn policy_gate_rejects_weak_evidence_or_regression() {
        let gate = PolicyGate::default();
        assert!(!gate.assess(0.5, 0.9, 4).accepted);
        assert!(!gate.assess(0.5, 0.4, 5).accepted);
        assert!(gate.assess(0.5, 0.6, 5).accepted);
    }

    #[test]
    fn gated_training_keeps_old_model_without_holdout_evidence() {
        let mut policy = ConstrainedPolicy::default();
        let before = policy.model().clone();
        let steps = vec![TrajectoryStep {
            context: "task:code".into(),
            action: "knowledge_first".into(),
            candidates: vec!["baseline".into(), "knowledge_first".into()],
            reward: 1.0,
            terminal: true,
        }];
        let (_metrics, evaluation) = policy
            .train_trajectory_gated(&steps, &[], 0.9, PolicyGate::default())
            .unwrap();
        assert!(!evaluation.accepted);
        assert_eq!(policy.model(), &before);
    }

    #[test]
    fn drift_detector_identifies_recent_regression() {
        let report = TrainablePolicyModel::detect_drift(&[1.0, 1.0, 0.0, 0.0], 2, 0.2);
        assert!(report.drifted);
        assert_eq!(report.delta, -1.0);
    }
}
