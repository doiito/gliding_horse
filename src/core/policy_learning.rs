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

/// Stable task-family features used by retrieval, treatment comparison and
/// policy learning. `family` is deliberately coarser than `raw_features`:
/// outcomes such as "summarise duration" and "summarise status" belong to the
/// same intervention family, while the raw features remain available for
/// audit and similarity ranking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LearningTaskContext {
    pub family: String,
    pub operations: Vec<String>,
    pub modalities: Vec<String>,
    pub raw_features: Vec<String>,
}

fn canonical_family_from_parts(operations: &[String], modalities: &[String]) -> String {
    let has = |values: &[String], candidates: &[&str]| {
        values
            .iter()
            .any(|value| candidates.contains(&value.as_str()))
    };
    let intent = if has(
        operations,
        &[
            "build",
            "fix",
            "implement",
            "migrate",
            "optimize",
            "refactor",
            "write",
        ],
    ) {
        "change"
    } else if has(operations, &["design"]) {
        "design"
    } else if has(
        operations,
        &["analyze", "audit", "research", "summarize", "verify"],
    ) {
        "inspect"
    } else if has(operations, &["test"]) {
        "test"
    } else if has(operations, &["operate"]) {
        "operate"
    } else {
        "generic"
    };
    let domain = if has(modalities, &["game", "ui", "web"]) {
        "interactive"
    } else if has(modalities, &["api", "code", "network", "service"]) {
        "software"
    } else if has(modalities, &["graph", "skill", "workflow"]) {
        "knowledge_system"
    } else if has(modalities, &["model"]) {
        "model"
    } else if has(modalities, &["data", "storage"]) {
        "data"
    } else if has(modalities, &["document"]) {
        "document"
    } else if has(modalities, &["image"]) {
        "media"
    } else {
        "generic"
    };
    format!("planning:v3:intent={intent};domain={domain}")
}

/// Map persisted v2 family keys to the coarser v3 intervention boundary.
/// Unknown/custom families remain exact, preserving caller-defined scopes.
pub fn canonicalize_learning_family_key(family: &str) -> String {
    if family.starts_with("planning:v3:") || !family.starts_with("planning:v2:") {
        return family.to_string();
    }
    let Some((ops, kinds)) = family
        .strip_prefix("planning:v2:ops=")
        .and_then(|rest| rest.split_once(";kinds="))
    else {
        return family.to_string();
    };
    let operations = ops
        .split('+')
        .filter(|value| *value != "generic")
        .map(str::to_string)
        .collect::<Vec<_>>();
    let modalities = kinds
        .split('+')
        .filter(|value| *value != "generic")
        .map(str::to_string)
        .collect::<Vec<_>>();
    canonical_family_from_parts(&operations, &modalities)
}

pub fn learning_families_compatible(left: &str, right: &str) -> bool {
    canonicalize_learning_family_key(left) == canonicalize_learning_family_key(right)
}

fn contains_any(text: &str, alternatives: &[&str]) -> bool {
    alternatives
        .iter()
        .any(|candidate| text.contains(candidate))
}

/// Build a stable, transport-neutral task family from an objective.
///
/// Numbers, filenames and output-field names are intentionally excluded from
/// the family key. Operations and broad artifact modalities remain, which is
/// enough to share treatment evidence without pooling every implementation or
/// analysis task into one unsafe global arm.
pub fn learning_task_context(objective: &str) -> LearningTaskContext {
    let normalized = objective.to_lowercase();
    let operation_vocabulary: [(&str, &[&str]); 15] = [
        ("analyze", &["analyze", "analyse", "analysis", "分析"]),
        ("audit", &["audit", "审计"]),
        (
            "build",
            &[
                "build", "create", "generate", "develop", "创建", "生成", "搭建", "开发",
            ],
        ),
        ("design", &["design", "设计"]),
        ("fix", &["fix", "repair", "resolve", "修复", "解决", "修正"]),
        (
            "implement",
            &[
                "implement",
                "add",
                "extend",
                "update",
                "modify",
                "support",
                "实现",
                "编写",
                "新增",
                "增加",
                "扩展",
                "修改",
                "支持",
            ],
        ),
        ("migrate", &["migrate", "migration", "迁移"]),
        (
            "optimize",
            &["optimize", "optimise", "optimization", "优化", "完善"],
        ),
        ("refactor", &["refactor", "重构"]),
        ("research", &["research", "search", "调研", "搜索"]),
        (
            "summarize",
            &["summarize", "summarise", "summary", "汇总", "总结"],
        ),
        ("test", &["test", "benchmark", "测试", "基准"]),
        ("verify", &["verify", "validate", "confirm", "验证", "确认"]),
        ("write", &["write", "输出", "写入"]),
        ("operate", &["run", "execute", "执行", "运行"]),
    ];
    let modality_vocabulary: [(&str, &[&str]); 15] = [
        ("api", &[" api", "api ", "接口"]),
        (
            "code",
            &[
                "code",
                "program",
                "python",
                "rust",
                "javascript",
                "typescript",
                "golang",
                "代码",
                "程序",
            ],
        ),
        ("data", &["data", "csv", "json", "jsonl", "数据"]),
        (
            "document",
            &["document", "report", "markdown", "文档", "报告"],
        ),
        ("game", &["game", "游戏"]),
        (
            "graph",
            &["graph", "ontology", "knowledge graph", "图谱", "本体"],
        ),
        ("image", &["image", "picture", "图片", "图像"]),
        ("model", &["model", "policy", "模型", "策略"]),
        ("network", &["network", "websocket", "网络"]),
        ("service", &["service", "server", "daemon", "服务"]),
        ("skill", &["skill", "技能"]),
        (
            "storage",
            &["database", "storage", "memory", "数据库", "存储", "记忆"],
        ),
        ("ui", &[" ui", "ui ", "tui", "界面"]),
        ("web", &["web", "html", "browser", "网页", "浏览器"]),
        ("workflow", &["workflow", "dag", "pdca", "流程", "编排"]),
    ];

    let mut operations = operation_vocabulary
        .iter()
        .filter(|(_, alternatives)| contains_any(&normalized, alternatives))
        .map(|(canonical, _)| (*canonical).to_string())
        .collect::<Vec<_>>();
    let mut modalities = modality_vocabulary
        .iter()
        .filter(|(_, alternatives)| contains_any(&normalized, alternatives))
        .map(|(canonical, _)| (*canonical).to_string())
        .collect::<Vec<_>>();
    // Explicit non-mutation language is part of the task contract, not a
    // mutation keyword. Without this guard, phrases such as "must not modify"
    // and "不得修改" were grouped into intent=change, contaminating both
    // retrieval and controlled promotion evidence for read-only tasks.
    let explicitly_non_mutating = [
        "do not modify",
        "must not modify",
        "do not change",
        "without modifying",
        "read only",
        "read-only",
        "不得修改",
        "禁止修改",
        "不要修改",
        "只读",
    ]
    .iter()
    .any(|marker| normalized.contains(marker));
    if explicitly_non_mutating {
        operations.retain(|operation| {
            !matches!(
                operation.as_str(),
                "build" | "fix" | "implement" | "migrate" | "optimize" | "refactor" | "write"
            )
        });
        if !operations.iter().any(|operation| operation == "verify") {
            operations.push("verify".to_string());
        }
    }
    operations.sort();
    operations.dedup();
    modalities.sort();
    modalities.dedup();

    let mut raw_features = normalized
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.chars().count() >= 2)
        .filter(|term| !term.chars().all(|character| character.is_ascii_digit()))
        .filter(|term| {
            !matches!(
                *term,
                "the"
                    | "and"
                    | "with"
                    | "from"
                    | "for"
                    | "this"
                    | "that"
                    | "into"
                    | "under"
                    | "exact"
                    | "bytes"
            )
        })
        .map(|term| {
            term.trim_end_matches(|character: char| character.is_ascii_digit())
                .to_string()
        })
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    raw_features.sort();
    raw_features.dedup();
    raw_features.truncate(24);

    let family = if operations.is_empty() && modalities.is_empty() {
        let fallback = raw_features.iter().take(6).cloned().collect::<Vec<_>>();
        if fallback.is_empty() {
            "planning:v3:intent=generic;domain=generic".to_string()
        } else {
            format!("planning:v3:terms={}", fallback.join("+"))
        }
    } else {
        canonical_family_from_parts(&operations, &modalities)
    };

    LearningTaskContext {
        family,
        operations,
        modalities,
        raw_features,
    }
}

pub fn learning_policy_context(objective: &str) -> String {
    learning_task_context(objective).family
}

/// Controls whether accumulated experience may influence the current task.
///
/// Baseline is a true ablation arm: it neither retrieves historical context
/// nor updates learned state. Shadow measures what would have been retrieved
/// without changing the prompt or online policy. Active enables the complete
/// continuous-learning loop.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LearningMode {
    Baseline,
    Shadow,
    #[default]
    Active,
}

impl LearningMode {
    pub fn retrieves_history(self) -> bool {
        !matches!(self, Self::Baseline)
    }

    pub fn injects_history(self) -> bool {
        matches!(self, Self::Active)
    }

    pub fn updates_learning(self) -> bool {
        matches!(self, Self::Active)
    }
}

impl std::fmt::Display for LearningMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Baseline => "baseline",
            Self::Shadow => "shadow",
            Self::Active => "active",
        })
    }
}

impl std::str::FromStr for LearningMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "baseline" | "off" | "disabled" => Ok(Self::Baseline),
            "shadow" | "observe" | "observation" => Ok(Self::Shadow),
            "active" | "on" | "enabled" => Ok(Self::Active),
            other => Err(format!(
                "invalid learning mode '{other}'; expected active, baseline, or shadow"
            )),
        }
    }
}

pub const AUDIT_EVIDENCE_PREFIX: &str = "iri://learning/ca-audit/";

/// Durable, domain-neutral evidence emitted by SA after the latest CA audit
/// and AA terminal decision. Applications may nominate a skill IRI to which a
/// compact fragment is attached, but the kernel never invents an application
/// domain or treats an LLM summary as verification by itself.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskAuditKnowledgeEvidence {
    pub task_iri: String,
    pub task_family: String,
    pub raw_features: Vec<String>,
    pub objective: String,
    pub terminal_status: String,
    pub ca_verdict: String,
    #[serde(default)]
    pub failed_dimensions: Vec<String>,
    #[serde(default)]
    pub findings: Vec<String>,
    #[serde(default)]
    pub procedure: Vec<String>,
    #[serde(default)]
    pub successful_checks: Vec<String>,
    #[serde(default)]
    pub failed_checks: Vec<String>,
    #[serde(default)]
    pub attached_skill_iri: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl TaskAuditKnowledgeEvidence {
    pub fn reusable_success(&self) -> bool {
        self.ca_verdict == "pass"
            && matches!(self.terminal_status.as_str(), "success" | "completed")
            && !self.successful_checks.is_empty()
    }

    pub fn storage_iri(&self) -> String {
        let digest = sha2::Sha256::digest(self.task_iri.as_bytes());
        format!("{AUDIT_EVIDENCE_PREFIX}{}", hex::encode(digest))
    }
}

pub fn load_task_audit_evidence(
    store: &L0Store,
    task_family: Option<&str>,
    limit: usize,
) -> Result<Vec<TaskAuditKnowledgeEvidence>, String> {
    let mut evidence = store
        .scan_iri_prefix(AUDIT_EVIDENCE_PREFIX, limit.max(1))
        .map_err(|error| error.to_string())?
        .into_iter()
        .filter_map(|entry| serde_json::from_str(&entry.content).ok())
        .filter(|item: &TaskAuditKnowledgeEvidence| {
            task_family.map_or(true, |family| item.task_family == family)
        })
        .collect::<Vec<_>>();
    evidence.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    evidence.truncate(limit);
    Ok(evidence)
}

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

/// A durable, family-scoped circuit breaker for the executable policy.
/// Freezing never deletes observations or changes a model; it only forces the
/// safe baseline until an explicit operator-approved unfreeze occurs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyFreeze {
    pub context: String,
    pub reason: String,
    pub frozen_at: chrono::DateTime<chrono::Utc>,
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
    /// Causal/audit identity used to deduplicate outcomes and to form strict
    /// controlled-replay pairs. Older observations deserialize as unlabelled
    /// operational evidence for backward compatibility.
    #[serde(default)]
    pub evidence: PolicyObservationEvidence,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyObservationEvidence {
    #[serde(default)]
    pub task_iri: Option<String>,
    #[serde(default)]
    pub experiment_pair_id: Option<String>,
    #[serde(default)]
    pub experiment_seed: Option<String>,
    #[serde(default)]
    pub experiment_model: Option<String>,
    #[serde(default)]
    pub experiment_config_fingerprint: Option<String>,
    #[serde(default)]
    pub workspace_fingerprint: Option<String>,
    #[serde(default)]
    pub objective_fingerprint: Option<String>,
    #[serde(default)]
    pub orchestration_mode: Option<String>,
}

impl PolicyObservationEvidence {
    fn controlled_signature(&self) -> Option<String> {
        let values = [
            self.experiment_seed.as_deref()?,
            self.experiment_model.as_deref()?,
            self.experiment_config_fingerprint.as_deref()?,
            self.workspace_fingerprint.as_deref()?,
            self.objective_fingerprint.as_deref()?,
            self.orchestration_mode.as_deref()?,
        ];
        if values
            .iter()
            .any(|value| value.is_empty() || value.starts_with("unavailable:"))
        {
            return None;
        }
        Some(values.join("\u{1f}"))
    }
}

/// Return one matched outcome per controlled pair. A pair is admissible only
/// when both executions have complete and identical causal controls and came
/// from distinct task runs. Reusing one pair ID therefore cannot manufacture
/// the five independent samples required by the promotion gate.
fn controlled_pair_rewards(
    observations: &[&PolicyObservation],
    candidate_action: &str,
) -> (Vec<f32>, Vec<f32>) {
    let mut baseline_by_pair = std::collections::HashMap::new();
    for observation in observations
        .iter()
        .copied()
        .filter(|item| item.action == "baseline")
    {
        let Some(pair_id) = observation.evidence.experiment_pair_id.as_deref() else {
            continue;
        };
        if observation.evidence.controlled_signature().is_some() {
            baseline_by_pair.insert(pair_id, observation);
        }
    }
    let mut seen_pairs = std::collections::HashSet::new();
    let mut baseline_rewards = Vec::new();
    let mut candidate_rewards = Vec::new();
    for candidate in observations
        .iter()
        .copied()
        .filter(|item| item.action == candidate_action)
    {
        let Some(pair_id) = candidate.evidence.experiment_pair_id.as_deref() else {
            continue;
        };
        if !seen_pairs.insert(pair_id) {
            continue;
        }
        let Some(baseline) = baseline_by_pair.get(pair_id).copied() else {
            continue;
        };
        if baseline.evidence.controlled_signature() != candidate.evidence.controlled_signature()
            || baseline.evidence.task_iri.is_none()
            || candidate.evidence.task_iri.is_none()
            || baseline.evidence.task_iri == candidate.evidence.task_iri
        {
            continue;
        }
        baseline_rewards.push(baseline.reward);
        candidate_rewards.push(candidate.reward);
    }
    (baseline_rewards, candidate_rewards)
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
    #[serde(default)]
    pub baseline_samples: u32,
    #[serde(default)]
    pub candidate_samples: u32,
    #[serde(default)]
    pub candidate_action: Option<String>,
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
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
        Self {
            min_samples: 5,
            min_improvement: 0.01,
        }
    }
}

impl PolicyGate {
    pub fn assess(
        &self,
        baseline_return: f32,
        candidate_return: f32,
        samples: u32,
    ) -> PolicyEvaluation {
        let improvement = candidate_return - baseline_return;
        let accepted = samples >= self.min_samples && improvement >= self.min_improvement;
        PolicyEvaluation {
            samples,
            baseline_samples: samples,
            candidate_samples: samples,
            candidate_action: None,
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
        let exp: Vec<f32> = scores
            .iter()
            .map(|score| (*score - max_score).exp())
            .collect();
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
                if self
                    .greedy_action(&observation.context, &candidates)
                    .as_deref()
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
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f32>() / values.len() as f32
            }
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
            self.train(
                &step.context,
                &step.action,
                &step.candidates,
                future_return.clamp(-1.0, 1.0),
            );
        }
        TrainingMetrics {
            updates: self.updates,
            steps: steps.len() as u32,
            discounted_return,
            average_reward: if steps.is_empty() {
                0.0
            } else {
                discounted_return / steps.len() as f32
            },
        }
    }
}

#[derive(Clone)]
pub struct ConstrainedPolicy {
    states: HashMap<String, PolicyState>,
    frozen_contexts: HashMap<String, PolicyFreeze>,
    store: Option<Arc<L0Store>>,
    model: TrainablePolicyModel,
    model_version: u64,
    rollback_model: Option<TrainablePolicyModel>,
    /// Exploration is bounded and only selects among safe hint-ordering arms.
    exploration_rate: f32,
    min_observations: u32,
    gate: PolicyGate,
}

impl Default for ConstrainedPolicy {
    fn default() -> Self {
        Self {
            states: HashMap::new(),
            frozen_contexts: HashMap::new(),
            store: None,
            model: TrainablePolicyModel::default(),
            model_version: 0,
            rollback_model: None,
            exploration_rate: 0.05,
            // One same-family rule outcome is enough to begin a bounded
            // candidate trial. Promotion still requires PolicyGate's five
            // independent observations on both arms.
            min_observations: 1,
            gate: PolicyGate::default(),
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
                    } else if entry.iri.starts_with(&format!("{PREFIX}freeze/")) {
                        if let Ok(freeze) = serde_json::from_str::<PolicyFreeze>(&entry.content) {
                            self.frozen_contexts.insert(freeze.context.clone(), freeze);
                        }
                        None
                    } else {
                        serde_json::from_str::<PolicyState>(&entry.content)
                            .ok()
                            .map(|state| (state.context.clone(), state))
                    }
                })
                .collect();
        }
        // The persisted model is the only executable model after restart.
        // Replaying every observation here would silently deploy candidate
        // updates that the promotion gate previously rejected. Arm evidence
        // remains separately durable and is pooled by compatible family.
        if self.model_version > 0 && self.model.updates == 0 {
            // Never claim that an absent/corrupt snapshot is promoted.
            self.model_version = 0;
        }
        self.store = Some(store);
        self
    }

    pub fn with_exploration(mut self, rate: f32) -> Self {
        self.exploration_rate = rate.clamp(0.0, 0.2);
        self
    }

    pub fn with_gate(mut self, gate: PolicyGate) -> Self {
        self.gate = PolicyGate {
            min_samples: gate.min_samples.max(1),
            min_improvement: if gate.min_improvement.is_finite() {
                gate.min_improvement.clamp(-2.0, 2.0)
            } else {
                PolicyGate::default().min_improvement
            },
        };
        self
    }

    pub fn with_min_observations(mut self, samples: u32) -> Self {
        self.min_observations = samples.max(1);
        self
    }

    pub fn gate(&self) -> PolicyGate {
        self.gate
    }

    pub fn min_observations(&self) -> u32 {
        self.min_observations
    }

    /// Force one normalized task family to its rule baseline. This is an
    /// idempotent, durable circuit breaker used by LearningHealthMonitor.
    pub fn freeze_context(&mut self, context: &str, reason: &str) -> Result<bool, String> {
        let context = canonicalize_learning_family_key(context);
        if context.trim().is_empty() {
            return Err("cannot freeze an empty policy context".into());
        }
        if self.frozen_contexts.contains_key(&context) {
            return Ok(false);
        }
        let freeze = PolicyFreeze {
            context: context.clone(),
            reason: reason.chars().take(240).collect(),
            frozen_at: chrono::Utc::now(),
        };
        if let Some(store) = &self.store {
            let key = format!(
                "{PREFIX}freeze/{}",
                hex::encode(sha2::Sha256::digest(context.as_bytes()))
            );
            store
                .store(
                    &key,
                    &serde_json::to_string(&freeze).map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
        }
        self.frozen_contexts.insert(context, freeze);
        Ok(true)
    }

    /// Remove a previously recorded freeze after an explicit recovery review.
    /// Callers must keep the approval evidence in their evolution record.
    pub fn unfreeze_context(&mut self, context: &str) -> Result<bool, String> {
        let context = canonicalize_learning_family_key(context);
        if self.frozen_contexts.remove(&context).is_none() {
            return Ok(false);
        }
        if let Some(store) = &self.store {
            let key = format!(
                "{PREFIX}freeze/{}",
                hex::encode(sha2::Sha256::digest(context.as_bytes()))
            );
            store.delete(&key).map_err(|error| error.to_string())?;
        }
        Ok(true)
    }

    pub fn freeze(&self, context: &str) -> Option<&PolicyFreeze> {
        let context = canonicalize_learning_family_key(context);
        self.frozen_contexts.get(&context)
    }

    pub fn choose(&mut self, context: &str, candidates: &[String], fallback: &str) -> PolicyChoice {
        let model_scores: HashMap<String, f32> = candidates
            .iter()
            .map(|candidate| (candidate.clone(), self.model.score(context, candidate)))
            .collect();
        let safe_fallback = if candidates.iter().any(|candidate| candidate == fallback) {
            fallback.to_string()
        } else {
            candidates
                .first()
                .cloned()
                .unwrap_or_else(|| fallback.to_string())
        };
        if self.freeze(context).is_some() {
            return PolicyChoice {
                context: context.to_string(),
                action: safe_fallback,
                used_fallback: true,
                confidence: 0.0,
                explored: false,
                candidates: candidates.to_vec(),
            };
        }
        // Pool legacy v2 and current v3 evidence only when both normalize to
        // the same generic intervention family. The original per-task raw
        // features remain in audit evidence and are not erased by pooling.
        let mut pooled_arms: HashMap<String, ArmStats> = HashMap::new();
        for state in self
            .states
            .values()
            .filter(|state| learning_families_compatible(&state.context, context))
        {
            for (action, stats) in &state.arms {
                let pooled = pooled_arms.entry(action.clone()).or_insert(ArmStats {
                    pulls: 0,
                    reward_sum: 0.0,
                });
                pooled.pulls = pooled.pulls.saturating_add(stats.pulls);
                pooled.reward_sum += stats.reward_sum;
            }
        }
        let pulls_for = |candidate: &str| {
            pooled_arms
                .get(candidate)
                .map(|stats| stats.pulls)
                .unwrap_or(0)
        };
        let baseline_pulls = pulls_for(&safe_fallback);
        let enough_baseline_evidence = baseline_pulls >= self.min_observations;
        let gate_samples = self.gate.min_samples;
        let candidate_under_evaluation = candidates
            .iter()
            .filter(|candidate| candidate.as_str() != safe_fallback)
            .filter(|candidate| pulls_for(candidate) < gate_samples)
            .min_by_key(|candidate| pulls_for(candidate))
            .cloned();
        // Candidate sampling is deterministic and bounded. It starts only
        // after the same family has an independent rule baseline and stops at
        // the promotion gate's evidence threshold.
        let should_explore = enough_baseline_evidence && candidate_under_evaluation.is_some();
        let baseline_mean = pooled_arms
            .get(&safe_fallback)
            .map(ArmStats::mean)
            .unwrap_or(0.0);
        let action = if !enough_baseline_evidence {
            safe_fallback.clone()
        } else if should_explore {
            candidate_under_evaluation.unwrap_or_else(|| safe_fallback.clone())
        } else if self.model_version == 0 {
            // An unpromoted trainable policy remains shadow-only. Empirical
            // means are audit evidence, not authority to bypass the gate.
            safe_fallback.clone()
        } else {
            candidates
                .iter()
                // A promoted model remains bounded by live empirical return.
                // Rejected/late regressions therefore fall back safely even
                // though their observations remain useful training evidence.
                .filter(|candidate| {
                    candidate.as_str() == safe_fallback
                        || pooled_arms.get(*candidate).is_some_and(|stats| {
                            stats.mean() - baseline_mean >= self.gate.min_improvement
                        })
                })
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
            .filter_map(|candidate| pooled_arms.get(candidate).map(ArmStats::mean))
            .fold(0.0_f32, f32::max);
        let used_fallback = action == safe_fallback;
        PolicyChoice {
            context: context.to_string(),
            action,
            used_fallback,
            confidence: if self.model_version > 0 {
                (best.max(0.0) / 1.0).clamp(0.0, 1.0)
            } else {
                0.0
            },
            explored: should_explore,
            candidates: candidates.to_vec(),
        }
    }

    pub fn record_reward(&mut self, choice: &PolicyChoice, reward: f32) -> Result<(), String> {
        self.record_observation(choice, reward, PolicyObservationEvidence::default(), true)?;
        Ok(())
    }

    /// Persist a true ablation outcome without retrieving/injecting history,
    /// training the executable model, or changing its version. This makes a
    /// controlled baseline useful to the promotion gate while preserving the
    /// behavioral meaning of `LearningMode::Baseline`.
    pub fn record_baseline_evidence(
        &mut self,
        choice: &PolicyChoice,
        reward: f32,
        evidence: PolicyObservationEvidence,
    ) -> Result<bool, String> {
        if choice.action != "baseline" {
            return Err("baseline evidence must use the baseline action".to_string());
        }
        self.record_observation(choice, reward, evidence, false)
    }

    fn record_observation(
        &mut self,
        choice: &PolicyChoice,
        reward: f32,
        evidence: PolicyObservationEvidence,
        train_model: bool,
    ) -> Result<bool, String> {
        let observation_key = self.observation_key(choice, &evidence);
        if let (Some(store), Some(key)) = (&self.store, observation_key.as_deref()) {
            if store
                .retrieve(key)
                .map_err(|error| error.to_string())?
                .is_some()
            {
                return Ok(false);
            }
        }
        let state = self
            .states
            .entry(choice.context.clone())
            .or_insert_with(|| PolicyState {
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
        if train_model {
            self.model
                .train(&choice.context, &choice.action, &choice.candidates, reward);
        }
        if let Some(store) = &self.store {
            let key = format!(
                "{PREFIX}{}",
                hex::encode(sha2::Sha256::digest(choice.context.as_bytes()))
            );
            let content = serde_json::to_string(state).map_err(|error| error.to_string())?;
            store
                .store(&key, &content)
                .map_err(|error| error.to_string())?;
            let observation = PolicyObservation {
                context: choice.context.clone(),
                action: choice.action.clone(),
                reward: reward.clamp(-1.0, 1.0),
                explored: choice.explored,
                candidates: choice.candidates.clone(),
                evidence,
                created_at: chrono::Utc::now(),
            };
            let observation_key = observation_key.unwrap_or_else(|| {
                format!("{PREFIX}observations/{}", uuid::Uuid::new_v4().hyphenated())
            });
            let observation_content =
                serde_json::to_string(&observation).map_err(|error| error.to_string())?;
            store
                .store(&observation_key, &observation_content)
                .map_err(|error| error.to_string())?;
            let model_content =
                serde_json::to_string(&self.model).map_err(|error| error.to_string())?;
            store
                .store(&format!("{PREFIX}model"), &model_content)
                .map_err(|error| error.to_string())?;
        }
        Ok(true)
    }

    fn observation_key(
        &self,
        choice: &PolicyChoice,
        evidence: &PolicyObservationEvidence,
    ) -> Option<String> {
        let unit = evidence
            .experiment_pair_id
            .as_deref()
            .map(|value| format!("pair:{value}"))
            .or_else(|| {
                evidence
                    .task_iri
                    .as_deref()
                    .map(|value| format!("task:{value}"))
            })?;
        let identity = format!("{}\n{}\n{}", choice.context, choice.action, unit);
        Some(format!(
            "{PREFIX}observations/identified/{}",
            hex::encode(sha2::Sha256::digest(identity.as_bytes()))
        ))
    }

    /// Record an auditable observation, but promote a replayed model only
    /// after baseline and treatment arms both have independent outcomes in
    /// the same normalized task context. Arm statistics and raw observations
    /// are retained even when promotion is rejected.
    pub fn record_reward_gated(
        &mut self,
        choice: &PolicyChoice,
        reward: f32,
        gate: PolicyGate,
    ) -> Result<PolicyEvaluation, String> {
        self.record_reward_gated_with_evidence(
            choice,
            reward,
            gate,
            PolicyObservationEvidence::default(),
        )
    }

    pub fn record_reward_gated_with_evidence(
        &mut self,
        choice: &PolicyChoice,
        reward: f32,
        gate: PolicyGate,
        evidence: PolicyObservationEvidence,
    ) -> Result<PolicyEvaluation, String> {
        let baseline_model = self.model.clone();
        let baseline_version = self.model_version;
        let inserted = self.record_observation(choice, reward, evidence.clone(), true)?;
        let observations = match &self.store {
            Some(store) => Self::load_observations(store)?,
            None => vec![PolicyObservation {
                context: choice.context.clone(),
                action: choice.action.clone(),
                reward: reward.clamp(-1.0, 1.0),
                explored: choice.explored,
                candidates: choice.candidates.clone(),
                evidence: evidence.clone(),
                created_at: chrono::Utc::now(),
            }],
        };
        if choice.candidates.len() <= 1 {
            // There is no intervention to compare. Keep the baseline outcome
            // as future evidence, but do not promote a no-op model version.
            self.model = baseline_model;
            self.model_version = baseline_version;
            self.persist_model()?;
            let baseline_rewards = observations
                .iter()
                .filter(|item| {
                    learning_families_compatible(&item.context, &choice.context)
                        && item.action == "baseline"
                })
                .map(|item| item.reward)
                .collect::<Vec<_>>();
            let baseline_return = if baseline_rewards.is_empty() {
                0.0
            } else {
                baseline_rewards.iter().sum::<f32>() / baseline_rewards.len() as f32
            };
            return Ok(PolicyEvaluation {
                samples: 0,
                baseline_samples: baseline_rewards.len().min(u32::MAX as usize) as u32,
                candidate_samples: 0,
                candidate_action: None,
                baseline_return,
                candidate_return: 0.0,
                improvement: 0.0,
                accepted: false,
                reason: "no_eligible_alternative".into(),
            });
        }

        // This is a treatment-effect gate, not an action-imitation score.
        // Compare rewards actually observed for baseline and one concrete
        // history treatment in the same normalized task context.  Requiring
        // evidence on both arms prevents an unobserved counterfactual from
        // being reported as a zero return.
        let relevant = observations
            .iter()
            .filter(|item| learning_families_compatible(&item.context, &choice.context))
            .collect::<Vec<_>>();
        let candidate_action = if choice.action != "baseline" {
            Some(choice.action.clone())
        } else {
            choice
                .candidates
                .iter()
                .filter(|action| action.as_str() != "baseline")
                .max_by_key(|action| {
                    relevant
                        .iter()
                        .filter(|item| item.action.as_str() == action.as_str())
                        .count()
                })
                .cloned()
        };
        let rewards_for = |action: &str| {
            relevant
                .iter()
                .filter(|item| item.action == action)
                .map(|item| item.reward)
                .collect::<Vec<_>>()
        };
        let (baseline_rewards, candidate_rewards) = if evidence.experiment_pair_id.is_some() {
            candidate_action
                .as_deref()
                .map(|candidate| controlled_pair_rewards(&relevant, candidate))
                .unwrap_or_default()
        } else {
            (
                rewards_for("baseline"),
                candidate_action
                    .as_deref()
                    .map(rewards_for)
                    .unwrap_or_default(),
            )
        };
        let mean = |values: &[f32]| {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f32>() / values.len() as f32
            }
        };
        let baseline_samples = baseline_rewards.len().min(u32::MAX as usize) as u32;
        let candidate_samples = candidate_rewards.len().min(u32::MAX as usize) as u32;
        let paired_samples = baseline_samples.min(candidate_samples);
        let mut evaluation = gate.assess(
            mean(&baseline_rewards),
            mean(&candidate_rewards),
            paired_samples,
        );
        evaluation.baseline_samples = baseline_samples;
        evaluation.candidate_samples = candidate_samples;
        evaluation.candidate_action = candidate_action;

        // A promoted model must contain the complete accumulated training
        // history.  Replaying durable observations fixes the former behavior
        // where every rejected online update forgot all earlier samples.
        let replayed_model = Self::replay(&observations).model;
        if evaluation.accepted && inserted {
            self.rollback_model = Some(baseline_model);
            self.model = replayed_model;
            self.model_version = baseline_version.saturating_add(1);
            self.persist_policy_version(&evaluation)?;
        } else {
            self.model = baseline_model;
            self.model_version = baseline_version;
        }
        self.persist_model()?;
        Ok(evaluation)
    }

    /// Train a complete multi-step trajectory and keep a rollback snapshot.
    /// Promotion is intentionally explicit: callers can run PolicyGate on an
    /// evaluation holdout before retaining the new version.
    pub fn train_trajectory(
        &mut self,
        steps: &[TrajectoryStep],
        gamma: f32,
    ) -> Result<TrainingMetrics, String> {
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
        let Some(previous) = self.rollback_model.take() else {
            return Ok(false);
        };
        self.model = previous;
        self.model_version = self.model_version.saturating_add(1);
        self.persist_model()?;
        Ok(true)
    }

    pub fn model_version(&self) -> u64 {
        self.model_version
    }

    fn persist_model(&self) -> Result<(), String> {
        if let Some(store) = &self.store {
            let content = serde_json::to_string(&self.model).map_err(|error| error.to_string())?;
            store
                .store(&format!("{PREFIX}model"), &content)
                .map_err(|error| error.to_string())?;
            store
                .store(&format!("{PREFIX}version"), &self.model_version.to_string())
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn persist_policy_version(&self, evaluation: &PolicyEvaluation) -> Result<(), String> {
        let Some(store) = &self.store else {
            return Ok(());
        };
        let version = PolicyVersion {
            version: self.model_version,
            model: self.model.clone(),
            evaluation: Some(evaluation.clone()),
            created_at: chrono::Utc::now(),
        };
        store
            .store(
                &format!("{PREFIX}versions/{}", self.model_version),
                &serde_json::to_string(&version).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())
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
            .filter_map(|entry| {
                let mut observation =
                    serde_json::from_str::<PolicyObservation>(&entry.content).ok()?;
                observation.context = canonicalize_learning_family_key(&observation.context);
                Some(observation)
            })
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
        let choice = policy.choose(
            "task:code",
            &["baseline".into(), "knowledge_first".into()],
            "baseline",
        );
        assert_eq!(choice.action, "baseline");
        assert!(choice.used_fallback);
        assert!(!choice.explored);
    }

    #[test]
    fn frozen_context_is_durable_and_forces_the_safe_baseline() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let candidates = vec!["baseline".into(), "experience_first".into()];
        let context = "planning:v3:intent=inspect;domain=document";
        let mut policy = ConstrainedPolicy::default().with_persistence(store.clone());
        assert!(policy.freeze_context(context, "health_regression").unwrap());
        assert!(!policy.freeze_context(context, "duplicate").unwrap());
        let choice = policy.choose(context, &candidates, "baseline");
        assert_eq!(choice.action, "baseline");
        assert!(choice.used_fallback);
        assert!(!choice.explored);
        drop(policy);

        let mut restored = ConstrainedPolicy::default().with_persistence(store);
        assert_eq!(
            restored
                .freeze(context)
                .map(|freeze| freeze.reason.as_str()),
            Some("health_regression")
        );
        assert!(restored.unfreeze_context(context).unwrap());
        assert!(restored.freeze(context).is_none());
    }

    #[test]
    fn learning_mode_has_a_true_non_injecting_baseline() {
        assert!(!LearningMode::Baseline.retrieves_history());
        assert!(!LearningMode::Baseline.injects_history());
        assert!(!LearningMode::Baseline.updates_learning());
        assert!(LearningMode::Shadow.retrieves_history());
        assert!(!LearningMode::Shadow.injects_history());
        assert!(LearningMode::Active.injects_history());
        assert!(LearningMode::Active.updates_learning());
    }

    #[test]
    fn task_family_groups_field_variants_but_separates_modalities() {
        let duration =
            learning_task_context("Use Python to summarize duration counts from a JSONL test log");
        let status =
            learning_task_context("Summarize status counts in the JSONL test log using Python");
        let web_game = learning_task_context("Build and test a playable web game");
        assert_eq!(duration.family, status.family);
        assert_ne!(duration.family, web_game.family);
        assert_ne!(duration.raw_features, status.raw_features);
    }

    #[test]
    fn task_family_groups_paraphrased_software_feature_changes() {
        let tags = learning_task_context(
            "Add normalized tags to the existing Python task queue and run tests",
        );
        let owner = learning_task_context(
            "Extend the existing Python task queue with owner filtering and run tests",
        );
        let web_game = learning_task_context("Build and test a playable web game");
        assert_eq!(tags.family, owner.family);
        assert_eq!(tags.family, "planning:v3:intent=change;domain=software");
        assert_ne!(tags.family, web_game.family);
    }

    #[test]
    fn explicit_read_only_contract_is_not_grouped_with_mutation_tasks() {
        let chinese =
            learning_task_context("只读 fixture.txt，返回 ANSWER 精确值并引用证据行。不得修改。");
        let english = learning_task_context(
            "Read fixture.txt and return the exact ANSWER; do not modify any files.",
        );
        assert_eq!(chinese.family, "planning:v3:intent=inspect;domain=generic");
        assert_eq!(english.family, chinese.family);
        assert!(!chinese
            .operations
            .iter()
            .any(|operation| { matches!(operation.as_str(), "implement" | "write" | "fix") }));
    }

    #[test]
    fn policy_can_use_compatible_v2_baseline_evidence_without_rewriting_it() {
        let legacy = "planning:v2:ops=build+operate+test+write;kinds=code+data";
        let current = learning_policy_context(
            "Extend the existing Python task queue with owner filtering and run tests",
        );
        assert!(learning_families_compatible(legacy, &current));

        let mut policy = ConstrainedPolicy::default();
        policy.states.insert(
            legacy.into(),
            PolicyState {
                context: legacy.into(),
                arms: HashMap::from([(
                    "baseline".into(),
                    ArmStats {
                        pulls: 1,
                        reward_sum: 0.5,
                    },
                )]),
                updated_at: chrono::Utc::now(),
            },
        );
        let choice = policy.choose(
            &current,
            &["baseline".into(), "knowledge_first".into()],
            "baseline",
        );
        assert_eq!(choice.action, "knowledge_first");
        assert!(choice.explored);
    }

    #[test]
    fn ca_audit_evidence_is_family_filterable_and_requires_direct_checks() {
        let dir = tempfile::tempdir().unwrap();
        let store = L0Store::new(dir.path().to_str().unwrap()).unwrap();
        let mut evidence = TaskAuditKnowledgeEvidence {
            task_iri: "iri://task/verified".into(),
            task_family: "planning:v2:ops=test;kinds=data".into(),
            raw_features: vec!["jsonl".into()],
            objective: "test data".into(),
            terminal_status: "success".into(),
            ca_verdict: "pass".into(),
            failed_dimensions: vec![],
            findings: vec![],
            procedure: vec!["write output".into()],
            successful_checks: vec![],
            failed_checks: vec![],
            attached_skill_iri: None,
            created_at: chrono::Utc::now(),
        };
        assert!(!evidence.reusable_success());
        evidence.successful_checks.push("verify output".into());
        assert!(evidence.reusable_success());
        store
            .store(
                &evidence.storage_iri(),
                &serde_json::to_string(&evidence).unwrap(),
            )
            .unwrap();
        assert_eq!(
            load_task_audit_evidence(&store, Some(&evidence.task_family), 10)
                .unwrap()
                .len(),
            1
        );
        assert!(load_task_audit_evidence(&store, Some("other"), 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn candidate_sampling_requires_same_family_baseline_evidence() {
        let candidates = vec![
            "baseline".into(),
            "knowledge_first".into(),
            "skill_first".into(),
        ];
        let mut policy = ConstrainedPolicy::default();
        for _ in 0..5 {
            let choice = PolicyChoice {
                context: "task:create".into(),
                action: "baseline".into(),
                used_fallback: true,
                confidence: 0.0,
                explored: false,
                candidates: candidates.clone(),
            };
            policy.record_reward(&choice, 0.2).unwrap();
        }

        let unrelated = policy.choose("task:unrelated", &candidates, "baseline");
        assert_eq!(unrelated.action, "baseline");
        assert!(!unrelated.explored);

        let first = policy.choose("task:create", &candidates, "baseline");
        assert_eq!(first.action, "knowledge_first");
        assert!(first.explored);
        policy.record_reward(&first, 0.8).unwrap();

        let second = policy.choose("task:create", &candidates, "baseline");
        assert_eq!(second.action, "skill_first");
        assert!(second.explored);
    }

    #[test]
    fn empirical_best_arm_remains_shadow_while_model_is_still_gated() {
        let candidates = vec!["baseline".into(), "knowledge_first".into()];
        let mut policy = ConstrainedPolicy::default();
        for (action, reward) in std::iter::repeat_n(("baseline", 0.1), 5)
            .chain(std::iter::repeat_n(("knowledge_first", 0.9), 5))
        {
            policy
                .record_reward(
                    &PolicyChoice {
                        context: "task:stable".into(),
                        action: action.into(),
                        used_fallback: false,
                        confidence: 0.0,
                        explored: false,
                        candidates: candidates.clone(),
                    },
                    reward,
                )
                .unwrap();
        }
        // Simulate the production gate retaining observations while rejecting
        // candidate weights.
        policy.model = TrainablePolicyModel::default();
        policy.model_version = 0;
        let choice = policy.choose("task:stable", &candidates, "baseline");
        assert_eq!(choice.action, "baseline");
    }

    #[test]
    fn repeated_rewards_select_the_better_safe_arm_and_survive_restart() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(l0.clone());
        let candidates = vec!["baseline".into(), "knowledge_first".into()];
        let baseline = PolicyChoice {
            context: "task:code".into(),
            action: "baseline".into(),
            used_fallback: true,
            confidence: 0.0,
            explored: false,
            candidates: candidates.clone(),
        };
        let treatment = PolicyChoice {
            action: "knowledge_first".into(),
            used_fallback: false,
            explored: true,
            ..baseline.clone()
        };
        for _ in 0..5 {
            policy
                .record_reward_gated(&baseline, 0.1, PolicyGate::default())
                .unwrap();
        }
        for _ in 0..5 {
            policy
                .record_reward_gated(&treatment, 0.9, PolicyGate::default())
                .unwrap();
        }
        let selected = policy.choose("task:code", &candidates, "baseline");
        assert_eq!(selected.action, "knowledge_first");

        let restored = ConstrainedPolicy::default().with_persistence(l0);
        assert!(restored.state("task:code").is_some());
        assert_eq!(restored.model().updates, 10);
    }

    #[test]
    fn observations_replay_into_a_fresh_policy() {
        let now = chrono::Utc::now();
        let observations = vec![
            PolicyObservation {
                context: "task:code".into(),
                action: "baseline".into(),
                reward: 0.2,
                explored: false,
                candidates: vec!["baseline".into(), "knowledge_first".into()],
                evidence: PolicyObservationEvidence::default(),
                created_at: now,
            },
            PolicyObservation {
                context: "task:code".into(),
                action: "knowledge_first".into(),
                reward: 0.9,
                explored: true,
                candidates: vec!["baseline".into(), "knowledge_first".into()],
                evidence: PolicyObservationEvidence::default(),
                created_at: now,
            },
        ];
        let policy = ConstrainedPolicy::replay(&observations);
        assert_eq!(
            policy.state("task:code").unwrap().arms["knowledge_first"].pulls,
            1
        );
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
        let metrics = policy
            .train_trajectory(
                &[
                    TrajectoryStep {
                        context: "task:code".into(),
                        action: "knowledge_first".into(),
                        candidates: candidates.clone(),
                        reward: 0.2,
                        terminal: false,
                    },
                    TrajectoryStep {
                        context: "task:code".into(),
                        action: "knowledge_first".into(),
                        candidates,
                        reward: 1.0,
                        terminal: true,
                    },
                ],
                0.9,
            )
            .unwrap();
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
            .train_trajectory(
                &[TrajectoryStep {
                    context: "orchestration:code".into(),
                    action: "pdca".into(),
                    candidates,
                    reward: 1.0,
                    terminal: true,
                }],
                0.9,
            )
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
    fn online_update_is_not_promoted_before_gate_has_independent_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(store);
        let choice = PolicyChoice {
            context: "task:code".into(),
            action: "knowledge_first".into(),
            used_fallback: false,
            confidence: 0.0,
            explored: true,
            candidates: vec!["baseline".into(), "knowledge_first".into()],
        };
        let before = policy.model().clone();
        let evaluation = policy
            .record_reward_gated(&choice, 1.0, PolicyGate::default())
            .unwrap();
        assert!(!evaluation.accepted);
        assert_eq!(evaluation.reason, "insufficient_evidence");
        assert_eq!(evaluation.baseline_samples, 0);
        assert_eq!(evaluation.candidate_samples, 1);
        assert_eq!(evaluation.candidate_return, 1.0);
        assert_eq!(policy.model(), &before);
        assert_eq!(policy.model_version(), 0);
    }

    #[test]
    fn gated_online_policy_uses_observed_arm_returns_and_replays_all_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(store.clone());
        let baseline = PolicyChoice {
            context: "task:stable".into(),
            action: "baseline".into(),
            used_fallback: false,
            confidence: 0.0,
            explored: false,
            candidates: vec!["baseline".into(), "knowledge_first".into()],
        };
        let candidate = PolicyChoice {
            action: "knowledge_first".into(),
            explored: true,
            ..baseline.clone()
        };

        for _ in 0..5 {
            let evaluation = policy
                .record_reward_gated(&baseline, 0.6, PolicyGate::default())
                .unwrap();
            assert!(!evaluation.accepted);
        }
        let mut final_evaluation = None;
        for _ in 0..5 {
            final_evaluation = Some(
                policy
                    .record_reward_gated(&candidate, 0.9, PolicyGate::default())
                    .unwrap(),
            );
        }

        let evaluation = final_evaluation.unwrap();
        assert!(evaluation.accepted, "{evaluation:?}");
        assert_eq!(evaluation.samples, 5);
        assert_eq!(evaluation.baseline_samples, 5);
        assert_eq!(evaluation.candidate_samples, 5);
        assert_eq!(
            evaluation.candidate_action.as_deref(),
            Some("knowledge_first")
        );
        assert!((evaluation.baseline_return - 0.6).abs() < f32::EPSILON);
        assert!((evaluation.candidate_return - 0.9).abs() < f32::EPSILON);
        assert_eq!(policy.model_version(), 1);
        assert_eq!(policy.model().updates, 10);
        let restored = ConstrainedPolicy::default().with_persistence(store);
        assert_eq!(restored.model_version(), 1);
        assert_eq!(restored.model().updates, 10);
    }

    fn controlled_evidence(
        pair_id: &str,
        task_iri: &str,
        workspace: &str,
    ) -> PolicyObservationEvidence {
        PolicyObservationEvidence {
            task_iri: Some(task_iri.into()),
            experiment_pair_id: Some(pair_id.into()),
            experiment_seed: Some("seed-42".into()),
            experiment_model: Some("test-model".into()),
            experiment_config_fingerprint: Some("config-v1".into()),
            workspace_fingerprint: Some(workspace.into()),
            objective_fingerprint: Some("objective-1".into()),
            orchestration_mode: Some("pdca".into()),
        }
    }

    #[test]
    fn controlled_replay_promotes_only_five_distinct_comparable_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(store.clone());
        let baseline = PolicyChoice {
            context: "planning:v3:intent=inspect;domain=document".into(),
            action: "baseline".into(),
            used_fallback: true,
            confidence: 0.0,
            explored: false,
            candidates: vec!["baseline".into()],
        };
        let candidate = PolicyChoice {
            action: "experience_first".into(),
            used_fallback: false,
            explored: true,
            candidates: vec!["baseline".into(), "experience_first".into()],
            ..baseline.clone()
        };

        // A mismatched workspace is auditable but not comparable and cannot
        // contribute a synthetic pair.
        policy
            .record_baseline_evidence(
                &baseline,
                0.2,
                controlled_evidence("mismatch", "task:baseline-mismatch", "workspace-a"),
            )
            .unwrap();
        let mismatch = policy
            .record_reward_gated_with_evidence(
                &candidate,
                0.9,
                PolicyGate::default(),
                controlled_evidence("mismatch", "task:candidate-mismatch", "workspace-b"),
            )
            .unwrap();
        assert_eq!(mismatch.samples, 0);
        assert!(!mismatch.accepted);

        let mut final_evaluation = None;
        for index in 0..5 {
            let pair = format!("pair-{index}");
            assert!(policy
                .record_baseline_evidence(
                    &baseline,
                    0.3,
                    controlled_evidence(
                        &pair,
                        &format!("task:baseline-{index}"),
                        "workspace-stable",
                    ),
                )
                .unwrap());
            final_evaluation = Some(
                policy
                    .record_reward_gated_with_evidence(
                        &candidate,
                        0.9,
                        PolicyGate::default(),
                        controlled_evidence(
                            &pair,
                            &format!("task:candidate-{index}"),
                            "workspace-stable",
                        ),
                    )
                    .unwrap(),
            );
        }
        let final_evaluation = final_evaluation.unwrap();
        assert!(final_evaluation.accepted, "{final_evaluation:?}");
        assert_eq!(final_evaluation.samples, 5);
        assert_eq!(policy.model_version(), 1);

        // Replaying an existing pair is idempotent and cannot manufacture a
        // second model version or increase the independent sample count.
        assert!(!policy
            .record_baseline_evidence(
                &baseline,
                1.0,
                controlled_evidence("pair-4", "task:another-baseline", "workspace-stable"),
            )
            .unwrap());
        let duplicate = policy
            .record_reward_gated_with_evidence(
                &candidate,
                1.0,
                PolicyGate::default(),
                controlled_evidence("pair-4", "task:another-candidate", "workspace-stable"),
            )
            .unwrap();
        assert_eq!(duplicate.samples, 5);
        assert_eq!(policy.model_version(), 1);

        let version = store
            .retrieve("iri://learning/policy/versions/1")
            .unwrap()
            .expect("promoted version must have a durable audit snapshot");
        let version: PolicyVersion = serde_json::from_str(&version.content).unwrap();
        assert_eq!(version.version, 1);
        assert!(version.evaluation.unwrap().accepted);

        let similar_family_candidates = vec!["baseline".into(), "experience_first".into()];
        let selected = policy.choose(
            "planning:v3:intent=inspect;domain=document",
            &similar_family_candidates,
            "baseline",
        );
        assert_eq!(selected.action, "experience_first");
        assert!(
            !selected.explored,
            "a promoted model is deployment, not trial"
        );

        let mut restarted = ConstrainedPolicy::default().with_persistence(store);
        let after_restart = restarted.choose(
            "planning:v3:intent=inspect;domain=document",
            &similar_family_candidates,
            "baseline",
        );
        assert_eq!(restarted.model_version(), 1);
        assert_eq!(after_restart.action, "experience_first");
        assert!(!after_restart.used_fallback);
    }

    #[test]
    fn rejected_online_update_does_not_reappear_after_restart() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(store.clone());
        let baseline = PolicyChoice {
            context: "task:restart-gate".into(),
            action: "baseline".into(),
            used_fallback: true,
            confidence: 0.0,
            explored: false,
            candidates: vec!["baseline".into(), "knowledge_first".into()],
        };
        let candidate = PolicyChoice {
            action: "knowledge_first".into(),
            used_fallback: false,
            explored: true,
            ..baseline.clone()
        };
        for _ in 0..5 {
            policy
                .record_reward_gated(&baseline, 0.6, PolicyGate::default())
                .unwrap();
        }
        for _ in 0..5 {
            policy
                .record_reward_gated(&candidate, 0.9, PolicyGate::default())
                .unwrap();
        }
        let promoted = policy.model().clone();
        assert_eq!(policy.model_version(), 1);

        let rejected = policy
            .record_reward_gated(&candidate, -1.0, PolicyGate::default())
            .unwrap();
        assert!(!rejected.accepted);
        assert_eq!(policy.model(), &promoted);
        assert_eq!(policy.model_version(), 1);

        let restored = ConstrainedPolicy::default().with_persistence(store);
        assert_eq!(restored.model_version(), 1);
        assert_eq!(restored.model(), &promoted);
    }

    #[test]
    fn promoted_policy_falls_back_when_live_candidate_return_regresses() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(store);
        let candidates = vec!["baseline".into(), "knowledge_first".into()];
        let baseline = PolicyChoice {
            context: "task:live-guard".into(),
            action: "baseline".into(),
            used_fallback: true,
            confidence: 0.0,
            explored: false,
            candidates: candidates.clone(),
        };
        let candidate = PolicyChoice {
            action: "knowledge_first".into(),
            used_fallback: false,
            explored: true,
            ..baseline.clone()
        };
        for _ in 0..5 {
            policy
                .record_reward_gated(&baseline, 0.6, PolicyGate::default())
                .unwrap();
            policy
                .record_reward_gated(&candidate, 0.9, PolicyGate::default())
                .unwrap();
        }
        assert_eq!(
            policy
                .choose("task:live-guard", &candidates, "baseline")
                .action,
            "knowledge_first"
        );

        for _ in 0..4 {
            policy
                .record_reward_gated(&candidate, -1.0, PolicyGate::default())
                .unwrap();
        }
        let guarded = policy.choose("task:live-guard", &candidates, "baseline");
        assert_eq!(guarded.action, "baseline");
        assert!(guarded.used_fallback);
        assert_eq!(policy.model_version(), 1);
    }

    #[test]
    fn single_arm_observation_never_promotes_a_noop_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mut policy = ConstrainedPolicy::default().with_persistence(store);
        let choice = PolicyChoice {
            context: "task:no-alternative".into(),
            action: "baseline".into(),
            used_fallback: true,
            confidence: 0.0,
            explored: false,
            candidates: vec!["baseline".into()],
        };

        for _ in 0..8 {
            let evaluation = policy
                .record_reward_gated(&choice, 1.0, PolicyGate::default())
                .unwrap();
            assert_eq!(evaluation.reason, "no_eligible_alternative");
            assert!(!evaluation.accepted);
        }
        assert_eq!(policy.model_version(), 0);
    }

    #[test]
    fn drift_detector_identifies_recent_regression() {
        let report = TrainablePolicyModel::detect_drift(&[1.0, 1.0, 0.0, 0.0], 2, 0.2);
        assert!(report.drifted);
        assert_eq!(report.delta, -1.0);
    }
}
