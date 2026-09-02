//! Direction-aware health monitoring for bounded learning policies.
//!
//! The monitor is intentionally observational. It never tunes an index,
//! retries a task, changes permissions or claims to repair state. Its only
//! actionable result is a family-scoped request to freeze learned retrieval
//! treatment and fall back to the existing baseline.

use std::collections::HashSet;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::retrieval_policy::RetrievalPolicyArm;
use crate::memory::l0_store::L0Store;
use crate::CoreError;

pub const LEARNING_HEALTH_SCHEMA_VERSION: u32 = 1;
pub const LEARNING_HEALTH_OBSERVATION_PREFIX: &str = "iri://learning/health/observation/";
pub const LEARNING_HEALTH_REPORT_PREFIX: &str = "iri://learning/health/report/";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HealthMetricDirection {
    HigherIsBetter,
    LowerIsBetter,
    TargetBand,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthMetricSpec {
    pub name: String,
    pub direction: HealthMetricDirection,
    /// Required relative, direction-normalised change before degradation is
    /// considered. The denominator is at least one, keeping rates stable near
    /// zero and making reward thresholds absolute in the [-1, 1] range.
    pub degradation_threshold: f64,
    #[serde(default)]
    pub target_min: Option<f64>,
    #[serde(default)]
    pub target_max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthMetricValue {
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningHealthObservation {
    pub schema_version: u32,
    pub task_iri: String,
    pub task_family: String,
    pub policy_action: String,
    pub policy_model_version: u64,
    pub metrics: Vec<HealthMetricValue>,
    pub created_at: DateTime<Utc>,
}

impl LearningHealthObservation {
    pub fn storage_iri(&self) -> String {
        let digest = Sha256::digest(self.task_iri.as_bytes());
        format!(
            "{}{}/{}",
            LEARNING_HEALTH_OBSERVATION_PREFIX,
            health_scope_hash(&self.task_family, &self.policy_action),
            hex::encode(digest)
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != LEARNING_HEALTH_SCHEMA_VERSION {
            return Err(format!(
                "unsupported learning health schema {}",
                self.schema_version
            ));
        }
        if self.task_iri.trim().is_empty() || self.task_family.trim().is_empty() {
            return Err("health observation requires task identity and family".into());
        }
        if RetrievalPolicyArm::parse(&self.policy_action).is_none() {
            return Err("health observation policy action is not whitelisted".into());
        }
        if self.metrics.is_empty() || self.metrics.len() > 32 {
            return Err("health observation must contain 1 to 32 metrics".into());
        }
        let mut names = HashSet::new();
        for metric in &self.metrics {
            if metric.name.trim().is_empty() || !metric.value.is_finite() {
                return Err("health metrics require names and finite values".into());
            }
            if !names.insert(metric.name.as_str()) {
                return Err("health metric names must be unique".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LearningHealthState {
    Healthy,
    InsufficientEvidence,
    Degraded,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthMetricAssessment {
    pub name: String,
    pub direction: HealthMetricDirection,
    pub baseline_mean: f64,
    pub recent_mean: f64,
    /// Positive is better after applying the metric direction.
    pub normalized_change: f64,
    pub signal_strength: f64,
    pub degraded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningHealthReport {
    pub schema_version: u32,
    pub task_family: String,
    pub policy_action: String,
    pub state: LearningHealthState,
    pub observed_samples: u32,
    pub assessments: Vec<HealthMetricAssessment>,
    pub generated_at: DateTime<Utc>,
}

impl LearningHealthReport {
    pub fn freeze_reason(&self) -> Option<String> {
        if self.state != LearningHealthState::Degraded {
            return None;
        }
        let metrics = self
            .assessments
            .iter()
            .filter(|assessment| assessment.degraded)
            .map(|assessment| assessment.name.as_str())
            .collect::<Vec<_>>();
        Some(format!("learning_health_degraded:{}", metrics.join(",")))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LearningHealthMonitorConfig {
    pub baseline_window: usize,
    pub recent_window: usize,
    pub min_samples_per_window: usize,
    pub min_signal_strength: f64,
    pub max_family_observations: usize,
    pub metrics: Vec<HealthMetricSpec>,
}

impl Default for LearningHealthMonitorConfig {
    fn default() -> Self {
        Self {
            baseline_window: 12,
            recent_window: 8,
            min_samples_per_window: 5,
            min_signal_strength: 1.96,
            max_family_observations: 128,
            metrics: vec![
                HealthMetricSpec {
                    name: "reward".into(),
                    direction: HealthMetricDirection::HigherIsBetter,
                    degradation_threshold: 0.08,
                    target_min: None,
                    target_max: None,
                },
                HealthMetricSpec {
                    name: "terminal_success".into(),
                    direction: HealthMetricDirection::HigherIsBetter,
                    degradation_threshold: 0.10,
                    target_min: None,
                    target_max: None,
                },
                HealthMetricSpec {
                    name: "verified_evidence".into(),
                    direction: HealthMetricDirection::HigherIsBetter,
                    degradation_threshold: 0.10,
                    target_min: None,
                    target_max: None,
                },
                HealthMetricSpec {
                    name: "tool_failure_rate".into(),
                    direction: HealthMetricDirection::LowerIsBetter,
                    degradation_threshold: 0.10,
                    target_min: None,
                    target_max: None,
                },
                HealthMetricSpec {
                    name: "elapsed_ms".into(),
                    direction: HealthMetricDirection::LowerIsBetter,
                    degradation_threshold: 0.25,
                    target_min: None,
                    target_max: None,
                },
                HealthMetricSpec {
                    name: "total_tokens".into(),
                    direction: HealthMetricDirection::LowerIsBetter,
                    degradation_threshold: 0.25,
                    target_min: None,
                    target_max: None,
                },
            ],
        }
    }
}

impl LearningHealthMonitorConfig {
    fn validate(&self) -> Result<(), String> {
        if self.baseline_window == 0
            || self.recent_window == 0
            || self.min_samples_per_window == 0
            || self.max_family_observations == 0
            || self.metrics.is_empty()
        {
            return Err("health monitor windows, limits and metrics must be positive".into());
        }
        if self.min_samples_per_window > self.baseline_window
            || self.min_samples_per_window > self.recent_window
            || !self.min_signal_strength.is_finite()
            || self.min_signal_strength < 0.0
        {
            return Err("health monitor sample and signal thresholds are invalid".into());
        }
        let mut names = HashSet::new();
        for metric in &self.metrics {
            let target_is_valid = match metric.direction {
                HealthMetricDirection::TargetBand => metric
                    .target_min
                    .zip(metric.target_max)
                    .is_some_and(|(minimum, maximum)| {
                        minimum.is_finite() && maximum.is_finite() && minimum <= maximum
                    }),
                _ => metric.target_min.is_none() && metric.target_max.is_none(),
            };
            if metric.name.trim().is_empty()
                || !metric.degradation_threshold.is_finite()
                || metric.degradation_threshold <= 0.0
                || !target_is_valid
                || !names.insert(metric.name.as_str())
            {
                return Err("health metric specifications are invalid".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthObservationPersistResult {
    Stored { iri: String },
    AlreadyPresent { iri: String },
}

#[derive(Clone)]
pub struct LearningHealthMonitor {
    l0: Arc<L0Store>,
    config: LearningHealthMonitorConfig,
}

impl LearningHealthMonitor {
    pub fn new(l0: Arc<L0Store>) -> Self {
        Self::with_config(l0, LearningHealthMonitorConfig::default())
            .expect("default learning health configuration is valid")
    }

    pub fn with_config(
        l0: Arc<L0Store>,
        config: LearningHealthMonitorConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { l0, config })
    }

    pub fn record_and_assess(
        &self,
        observation: &LearningHealthObservation,
    ) -> Result<(HealthObservationPersistResult, LearningHealthReport), CoreError> {
        observation.validate().map_err(invalid_health)?;
        let persisted = self.persist_observation(observation)?;
        let report = self.assess(&observation.task_family, &observation.policy_action)?;
        let report_iri = report_storage_iri(&report.task_family, &report.policy_action);
        self.l0.store(
            &report_iri,
            &serde_json::to_string(&report).map_err(|error| CoreError::StorageError {
                message: format!("serialize learning health report: {error}"),
            })?,
        )?;
        Ok((persisted, report))
    }

    pub fn assess(
        &self,
        task_family: &str,
        policy_action: &str,
    ) -> Result<LearningHealthReport, CoreError> {
        let mut observations = self
            .l0
            .scan_iri_prefix(
                &format!(
                    "{}{}/",
                    LEARNING_HEALTH_OBSERVATION_PREFIX,
                    health_scope_hash(task_family, policy_action)
                ),
                self.config.max_family_observations,
            )?
            .into_iter()
            .filter_map(|entry| {
                serde_json::from_str::<LearningHealthObservation>(&entry.content).ok()
            })
            .filter(|observation| {
                observation.task_family == task_family
                    && observation.policy_action == policy_action
                    && observation.validate().is_ok()
            })
            .collect::<Vec<_>>();
        observations.sort_by(|left, right| left.created_at.cmp(&right.created_at));
        if observations.len() > self.config.max_family_observations {
            let first = observations.len() - self.config.max_family_observations;
            observations.drain(..first);
        }

        let minimum_total = self.config.baseline_window + self.config.recent_window;
        if observations.len() < minimum_total {
            return Ok(LearningHealthReport {
                schema_version: LEARNING_HEALTH_SCHEMA_VERSION,
                task_family: task_family.to_string(),
                policy_action: policy_action.to_string(),
                state: LearningHealthState::InsufficientEvidence,
                observed_samples: observations.len().min(u32::MAX as usize) as u32,
                assessments: Vec::new(),
                generated_at: Utc::now(),
            });
        }

        let baseline_start = observations.len() - minimum_total;
        let recent_start = observations.len() - self.config.recent_window;
        let baseline = &observations[baseline_start..recent_start];
        let recent = &observations[recent_start..];
        let mut assessments = Vec::new();
        for spec in &self.config.metrics {
            let baseline_values = metric_values(baseline, &spec.name);
            let recent_values = metric_values(recent, &spec.name);
            if baseline_values.len() < self.config.min_samples_per_window
                || recent_values.len() < self.config.min_samples_per_window
            {
                continue;
            }
            let baseline_mean = mean(&baseline_values);
            let recent_mean = mean(&recent_values);
            let raw_change = match spec.direction {
                HealthMetricDirection::HigherIsBetter => recent_mean - baseline_mean,
                HealthMetricDirection::LowerIsBetter => baseline_mean - recent_mean,
                HealthMetricDirection::TargetBand => {
                    let minimum = spec.target_min.expect("validated target band minimum");
                    let maximum = spec.target_max.expect("validated target band maximum");
                    band_deviation(baseline_mean, minimum, maximum)
                        - band_deviation(recent_mean, minimum, maximum)
                }
            };
            let normalized_change = raw_change / baseline_mean.abs().max(1.0);
            let standard_error = (variance(&baseline_values, baseline_mean)
                / baseline_values.len() as f64
                + variance(&recent_values, recent_mean) / recent_values.len() as f64)
                .sqrt();
            let signal_strength = if standard_error <= f64::EPSILON {
                if raw_change.abs() <= f64::EPSILON {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                raw_change.abs() / standard_error
            };
            let degraded = normalized_change <= -spec.degradation_threshold
                && signal_strength >= self.config.min_signal_strength;
            assessments.push(HealthMetricAssessment {
                name: spec.name.clone(),
                direction: spec.direction,
                baseline_mean,
                recent_mean,
                normalized_change,
                signal_strength,
                degraded,
            });
        }
        let state = if assessments.is_empty() {
            LearningHealthState::InsufficientEvidence
        } else if assessments.iter().any(|assessment| assessment.degraded) {
            LearningHealthState::Degraded
        } else {
            LearningHealthState::Healthy
        };
        Ok(LearningHealthReport {
            schema_version: LEARNING_HEALTH_SCHEMA_VERSION,
            task_family: task_family.to_string(),
            policy_action: policy_action.to_string(),
            state,
            observed_samples: observations.len().min(u32::MAX as usize) as u32,
            assessments,
            generated_at: Utc::now(),
        })
    }

    fn persist_observation(
        &self,
        observation: &LearningHealthObservation,
    ) -> Result<HealthObservationPersistResult, CoreError> {
        let iri = observation.storage_iri();
        if let Some(existing) = self.l0.retrieve(&iri)? {
            let restored = serde_json::from_str::<LearningHealthObservation>(&existing.content)
                .map_err(|error| CoreError::StorageError {
                    message: format!("stored learning health observation is corrupt: {error}"),
                })?;
            restored.validate().map_err(invalid_health)?;
            if restored.task_iri != observation.task_iri {
                return Err(CoreError::StorageError {
                    message: "learning health observation key collision".into(),
                });
            }
            return Ok(HealthObservationPersistResult::AlreadyPresent { iri });
        }
        self.l0.store(
            &iri,
            &serde_json::to_string(observation).map_err(|error| CoreError::StorageError {
                message: format!("serialize learning health observation: {error}"),
            })?,
        )?;
        Ok(HealthObservationPersistResult::Stored { iri })
    }
}

fn metric_values(observations: &[LearningHealthObservation], metric_name: &str) -> Vec<f64> {
    observations
        .iter()
        .filter_map(|observation| {
            observation
                .metrics
                .iter()
                .find(|metric| metric.name == metric_name)
                .map(|metric| metric.value)
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64], average: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

fn band_deviation(value: f64, minimum: f64, maximum: f64) -> f64 {
    if value < minimum {
        minimum - value
    } else if value > maximum {
        value - maximum
    } else {
        0.0
    }
}

fn report_storage_iri(task_family: &str, policy_action: &str) -> String {
    format!(
        "{LEARNING_HEALTH_REPORT_PREFIX}{}",
        health_scope_hash(task_family, policy_action)
    )
}

fn health_scope_hash(task_family: &str, policy_action: &str) -> String {
    let digest = Sha256::digest(format!("{task_family}\x1f{policy_action}").as_bytes());
    hex::encode(digest)
}

fn invalid_health(message: String) -> CoreError {
    CoreError::StorageError {
        message: format!("invalid learning health observation: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        task: usize,
        action: &str,
        reward: f64,
        latency: f64,
        created_at: DateTime<Utc>,
    ) -> LearningHealthObservation {
        LearningHealthObservation {
            schema_version: LEARNING_HEALTH_SCHEMA_VERSION,
            task_iri: format!("iri://task/health-{task}"),
            task_family: "planning:v3:intent=inspect;domain=document".into(),
            policy_action: action.into(),
            policy_model_version: 1,
            metrics: vec![
                HealthMetricValue {
                    name: "reward".into(),
                    value: reward,
                },
                HealthMetricValue {
                    name: "elapsed_ms".into(),
                    value: latency,
                },
            ],
            created_at,
        }
    }

    fn monitor(store: Arc<L0Store>) -> LearningHealthMonitor {
        LearningHealthMonitor::with_config(
            store,
            LearningHealthMonitorConfig {
                baseline_window: 2,
                recent_window: 2,
                min_samples_per_window: 2,
                min_signal_strength: 1.0,
                max_family_observations: 16,
                metrics: vec![
                    HealthMetricSpec {
                        name: "reward".into(),
                        direction: HealthMetricDirection::HigherIsBetter,
                        degradation_threshold: 0.1,
                        target_min: None,
                        target_max: None,
                    },
                    HealthMetricSpec {
                        name: "elapsed_ms".into(),
                        direction: HealthMetricDirection::LowerIsBetter,
                        degradation_threshold: 0.1,
                        target_min: None,
                        target_max: None,
                    },
                ],
            },
        )
        .unwrap()
    }

    #[test]
    fn monitor_respects_metric_direction_and_freeze_signal() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap());
        let monitor = monitor(store);
        let now = Utc::now();
        for (index, (reward, latency)) in [(0.9, 10.0), (0.9, 10.0), (0.4, 30.0), (0.4, 30.0)]
            .into_iter()
            .enumerate()
        {
            let (_, report) = monitor
                .record_and_assess(&observation(
                    index,
                    "knowledge_first",
                    reward,
                    latency,
                    now + chrono::Duration::seconds(index as i64),
                ))
                .unwrap();
            if index < 3 {
                assert_eq!(report.state, LearningHealthState::InsufficientEvidence);
            } else {
                assert_eq!(report.state, LearningHealthState::Degraded);
                assert!(report.freeze_reason().unwrap().contains("reward"));
                assert!(report.freeze_reason().unwrap().contains("elapsed_ms"));
            }
        }
    }

    #[test]
    fn health_observations_are_idempotent_and_family_action_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_string_lossy().as_ref()).unwrap());
        let monitor = monitor(store);
        let record = observation(1, "experience_first", 0.8, 10.0, Utc::now());
        assert!(matches!(
            monitor.record_and_assess(&record).unwrap().0,
            HealthObservationPersistResult::Stored { .. }
        ));
        assert!(matches!(
            monitor.record_and_assess(&record).unwrap().0,
            HealthObservationPersistResult::AlreadyPresent { .. }
        ));
        let report = monitor
            .assess(&record.task_family, "knowledge_first")
            .unwrap();
        assert_eq!(report.observed_samples, 0);
        assert_eq!(report.state, LearningHealthState::InsufficientEvidence);
    }
}
