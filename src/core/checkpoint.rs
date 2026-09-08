use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::gateway::unified_gateway::ChatMessage;
use crate::memory::l0_store::L0Store;
use crate::CoreError;

/// Current durable schema for task resumption.
///
/// Schema v4 deliberately has no implicit migration path: a checkpoint must
/// carry one complete, validated [`TaskResumeState`].  Reconstructing state
/// from loosely related legacy fields made an apparently valid resume depend
/// on guesses that were not protected by the execution journal.
// v8/v6 deliberately invalidate pre-isolation, pre-typed-evidence,
// pre-two-level-DAG, and pre-direct-response/research evidence snapshots. A
// restored generated plan must still prove the same downstream Check evidence
// relation and delivery/capability boundary accepted during SA planning.
// snapshots. Older contracts may contain kernel-generic role definitions or
// work packages without exact artifact/verifier requirements; replaying them
// would silently undo the fresh LLM-derived agent.md boundary or admit a
// model-success result without the current kernel evidence contract.
pub const TASK_RESUME_STATE_SCHEMA_VERSION: u32 = 8;
pub const TASK_RESUME_CONTRACT_SCHEMA_VERSION: u32 = 6;
pub const ACTIVE_NODE_CONTINUATION_SCHEMA_VERSION: u32 = 1;

/// Checkpoints belonging to one task share an IRI namespace, but they do not
/// share a recovery protocol.  A ReAct/SA runtime checkpoint can restart the
/// task, while a BizAgent checkpoint can only restart one matching same-role
/// orchestration.  Keeping this discriminator in the serialized record avoids
/// selecting a newer, structurally unrelated BizAgent snapshot by accident.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    TaskRuntime,
    BizOrchestration,
}

impl CheckpointKind {
    fn path_segment(self) -> &'static str {
        match self {
            Self::TaskRuntime => "task_runtime",
            Self::BizOrchestration => "biz_orchestration",
        }
    }
}

/// Immutable task authority and the exact plan revision active at a durable
/// checkpoint boundary.  This is the input to recovery; chat history is only
/// supporting model context and must never be used to reconstruct authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResumeContract {
    pub schema_version: u32,
    pub original_user_task: String,
    pub constraints: BTreeMap<String, String>,
    pub effect_policy: crate::core::effect::EffectPolicy,
    pub delivery_target: Option<String>,
    pub execution_plan: crate::core::sa::ExecutionPlan,
    pub fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedTaskResumeContract {
    task_iri: String,
    contract: TaskResumeContract,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedDagResumeState {
    completed_nodes: BTreeMap<String, crate::core::workflow::NodeResult>,
    skipped_nodes: BTreeSet<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskCumulativeState {
    /// Counts accepted at an SA `step_complete_*` boundary. These never
    /// include an interrupted AgentRunner's local progress.
    pub turn_count: u32,
    pub tool_call_count: u32,
}

/// Receipt binding one transcript to the exact Agent instance and dynamic
/// prompt which produced it.  A task restore may inspect this payload, but it
/// must not replay it into a newly-created Agent instance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ActiveNodeContinuation {
    pub schema_version: u32,
    pub step_id: String,
    pub dispatch_id: String,
    pub agent_id: String,
    pub l1_session_id: String,
    pub role: crate::core::agent_instance::AgentRole,
    pub agent_md_sha256: String,
    pub context_manifest_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_interaction_id: Option<String>,
    pub transcript_sha256: String,
    pub local_turn_count: u32,
    pub local_tool_call_count: u32,
}

/// Stable identity known before an AgentRunner checkpoint is serialized. The
/// writer adds the exact transcript hash and local counters atomically.
#[derive(Debug, Clone)]
pub(crate) struct ActiveNodeIdentity {
    pub step_id: String,
    pub dispatch_id: String,
    pub agent_id: String,
    pub l1_session_id: String,
    pub role: crate::core::agent_instance::AgentRole,
    pub agent_md_sha256: String,
    pub context_manifest_sha256: String,
    pub source_interaction_id: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskResumeState {
    pub schema_version: u32,
    /// Exact durable boundary identity. Names repeat across roles and PDCA
    /// cycles, so the execution journal must correlate by IRI.
    pub checkpoint_iri: String,
    pub checkpoint_name: String,
    /// Task-wide counts committed by SA, deliberately separate from a
    /// possibly interrupted node's local counters.
    pub task_cumulative: TaskCumulativeState,
    /// Present only for an in-flight AgentRunner boundary. `None` denotes a
    /// typed SA commit and therefore requires an empty transcript.
    pub active_continuation: Option<ActiveNodeContinuation>,
    pub current_role: Option<String>,
    pub prev_summary: Option<String>,
    /// Cumulative tool receipts up to this checkpoint. These are restored into
    /// SA's task-wide evidence accumulator; narrative/artifact claims cannot
    /// substitute for them.
    pub tracked_actions: Vec<crate::core::tracked_action::TrackedAction>,
    /// Receipt IDs whose results have crossed an SA `step_complete_*`
    /// boundary. A ReAct `turn_*`/`finish_*` checkpoint may contain newer
    /// receipts, but those are not safe to replay until the parent has
    /// accepted the corresponding typed node result.
    pub committed_action_ids: BTreeSet<String>,
    /// Completed DAG nodes are replayed as data and skipped by exact node ID.
    /// This prevents a role-level heuristic from re-running unrelated nodes.
    pub completed_nodes: BTreeMap<String, crate::core::workflow::NodeResult>,
    /// Exact nodes bypassed by an accepted branch decision. This is execution
    /// control state, not model narrative, and must survive process restart.
    pub skipped_nodes: BTreeSet<String>,
    /// Canonical recovery authority.  Diagnostic checkpoint fields below are
    /// intentionally excluded because the runtime does not replay them.
    pub contract: TaskResumeContract,
}

/// The sole supported checkpoint restoration input for callers.  Parsing and
/// schema validation happen before an application begins a resumed task.
#[derive(Debug, Clone)]
pub struct RestoredTask {
    pub checkpoint: CheckpointData,
    pub messages: Vec<ChatMessage>,
    pub state: TaskResumeState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    pub kind: CheckpointKind,
    pub checkpoint_iri: String,
    pub task_iri: String,
    pub name: String,
    pub node_count: i32,
    pub total_size_bytes: i32,
    pub created_at: DateTime<Utc>,
    pub tags: Vec<String>,
    pub nodes_json: String,
    pub session_messages_json: String,
    pub agent_state_json: String,

    // ── Diagnostic payloads (not recovery authority) ──
    /// The currently executing agent role (PA/DA/CA/AA), used by resume to decide which phases to skip
    pub current_role: Option<String>,

    /// Historical 5W2H diagnostic captured by legacy callers.
    pub five_w2h_json: Option<String>,

    /// prev_summary chain value (summary passed through PA→DA→CA→AA)
    pub prev_summary: Option<String>,

    /// CycleState diagnostic (phase, iteration, phase history, hints).
    pub cycle_state_json: Option<String>,

    /// Raw completed DAG-node diagnostic. Recovery never trusts this string
    /// directly; creation parses it into `resume_state.completed_nodes` and
    /// validates the typed structure before persistence.
    pub completed_nodes_json: Option<String>,

    /// Pending approval diagnostic. Approval is deliberately re-established
    /// through the live approval protocol rather than trusted on recovery.
    pub pending_approvals_json: Option<String>,

    /// Pending supplementary input entries
    pub supplement_json: Option<String>,

    /// Accumulated tool error count + injected recovery tool set in the React loop
    pub tool_error_json: Option<String>,

    /// ActionTracker accumulated tracked actions
    pub action_tracker_json: Option<String>,

    /// Perception engine anomaly history (used for dedup)
    pub perception_anomaly_json: Option<String>,

    /// Present only for `TaskRuntime`. BizAgent orchestration has its own
    /// schema/key validation and is never a generic task-resume candidate.
    pub resume_state: Option<TaskResumeState>,
}

pub struct CheckpointManager {
    l0: Option<Arc<L0Store>>,
    task_checkpoints: RwLock<HashMap<String, Vec<String>>>,
    task_contracts: RwLock<HashMap<String, TaskResumeContract>>,
    counter: AtomicU64,
}

/// Upper bound on checkpoints retained per task. Older entries beyond this
/// limit are evicted on the next create to keep L0 growth bounded for
/// long-running multi-turn tasks.
pub const MAX_CHECKPOINTS_PER_TASK: usize = 20;

impl CheckpointManager {
    pub fn new() -> Self {
        Self {
            l0: None,
            task_checkpoints: RwLock::new(HashMap::new()),
            task_contracts: RwLock::new(HashMap::new()),
            counter: AtomicU64::new(0),
        }
    }

    pub fn with_persistence(l0: Arc<L0Store>) -> Self {
        Self {
            l0: Some(l0),
            task_checkpoints: RwLock::new(HashMap::new()),
            task_contracts: RwLock::new(HashMap::new()),
            counter: AtomicU64::new(0),
        }
    }

    /// Publish the authoritative contract before any role begins execution.
    /// Later CheckpointManager instances load this record from L0, which keeps
    /// AgentRunner's checkpoint writer independent from SA's planning types.
    pub fn register_task_contract(
        &self,
        task_iri: &str,
        contract: TaskResumeContract,
    ) -> Result<(), CoreError> {
        if task_iri.trim().is_empty() {
            return Err(CoreError::Internal {
                message: "Cannot register a resume contract without task IRI".to_string(),
            });
        }
        contract.validate()?;
        if let Some(ref l0) = self.l0 {
            let record = PersistedTaskResumeContract {
                task_iri: task_iri.to_string(),
                contract: contract.clone(),
            };
            let content = serde_json::to_string(&record).map_err(|error| CoreError::Internal {
                message: format!("Failed to serialize task resume contract: {error}"),
            })?;
            l0.store(&task_contract_iri(task_iri), &content)?;
        }
        self.task_contracts
            .write()
            .insert(task_iri.to_string(), contract);
        Ok(())
    }

    pub fn load_task_contract(
        &self,
        task_iri: &str,
    ) -> Result<Option<TaskResumeContract>, CoreError> {
        if let Some(contract) = self.task_contracts.read().get(task_iri).cloned() {
            contract.validate()?;
            return Ok(Some(contract));
        }
        let Some(ref l0) = self.l0 else {
            return Ok(None);
        };
        let Some(entry) = l0.retrieve(&task_contract_iri(task_iri))? else {
            return Ok(None);
        };
        let record = serde_json::from_str::<PersistedTaskResumeContract>(&entry.content).map_err(
            |error| CoreError::Internal {
                message: format!("Invalid persisted task resume contract: {error}"),
            },
        )?;
        if record.task_iri != task_iri {
            return Err(CoreError::Internal {
                message: "Persisted resume contract task identity mismatch".to_string(),
            });
        }
        record.contract.validate()?;
        self.task_contracts
            .write()
            .insert(task_iri.to_string(), record.contract.clone());
        Ok(Some(record.contract))
    }

    /// Load the newest committed SA boundary as the base for a ReAct
    /// checkpoint. Never inherit another in-flight/finished AgentRunner
    /// record: parallel children share a task IRI, and doing so would let one
    /// child's unaccepted receipts leak into a sibling's checkpoint.
    fn latest_committed_runtime_state(&self, task_iri: &str) -> Option<TaskResumeState> {
        let l0 = self.l0.as_ref()?;
        let stripped = task_iri.strip_prefix("iri://").unwrap_or(task_iri);
        let prefix = format!("iri://checkpoint/{}/", stripped);
        let entries = l0.scan_iri_prefix(&prefix, 100_000).ok()?;
        entries
            .into_iter()
            .filter_map(|entry| serde_json::from_str::<CheckpointData>(&entry.content).ok())
            .filter(|checkpoint| {
                checkpoint.kind == CheckpointKind::TaskRuntime && checkpoint.task_iri == task_iri
            })
            .filter_map(|checkpoint| {
                let created_at = checkpoint.created_at;
                let restored = checkpoint.to_restored_task().ok()?;
                restored
                    .state
                    .checkpoint_name
                    .starts_with("step_complete_")
                    .then_some((created_at, restored.state))
            })
            .max_by_key(|(created_at, _)| *created_at)
            .map(|(_, state)| state)
    }

    pub fn create(
        &self,
        task_iri: &str,
        name: &str,
        nodes_json: &str,
        session_messages_json: &str,
        agent_state_json: &str,
        tags: &[String],
    ) -> Result<CheckpointData, CoreError> {
        self.create_ext(
            task_iri,
            name,
            nodes_json,
            session_messages_json,
            agent_state_json,
            tags,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Extended creation method: supports all optional fields. None fields won't appear in serialization (saving L0 space).
    #[allow(clippy::too_many_arguments)]
    pub fn create_ext(
        &self,
        task_iri: &str,
        name: &str,
        nodes_json: &str,
        session_messages_json: &str,
        agent_state_json: &str,
        tags: &[String],
        current_role: Option<&str>,
        five_w2h_json: Option<&str>,
        prev_summary: Option<&str>,
        cycle_state_json: Option<&str>,
        completed_nodes_json: Option<&str>,
        pending_approvals_json: Option<&str>,
        supplement_json: Option<&str>,
        tool_error_json: Option<&str>,
        action_tracker_json: Option<&str>,
        perception_anomaly_json: Option<&str>,
    ) -> Result<CheckpointData, CoreError> {
        self.create_ext_with_kind(
            CheckpointKind::TaskRuntime,
            task_iri,
            name,
            nodes_json,
            session_messages_json,
            agent_state_json,
            tags,
            current_role,
            five_w2h_json,
            prev_summary,
            cycle_state_json,
            completed_nodes_json,
            pending_approvals_json,
            supplement_json,
            tool_error_json,
            action_tracker_json,
            perception_anomaly_json,
        )
    }

    /// Low-level typed checkpoint writer. Most callers create task-runtime
    /// checkpoints through `create_ext`; BizAgent is the sole owner of the
    /// orchestration kind.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_ext_with_kind(
        &self,
        kind: CheckpointKind,
        task_iri: &str,
        name: &str,
        nodes_json: &str,
        session_messages_json: &str,
        agent_state_json: &str,
        tags: &[String],
        current_role: Option<&str>,
        five_w2h_json: Option<&str>,
        prev_summary: Option<&str>,
        cycle_state_json: Option<&str>,
        completed_nodes_json: Option<&str>,
        pending_approvals_json: Option<&str>,
        supplement_json: Option<&str>,
        tool_error_json: Option<&str>,
        action_tracker_json: Option<&str>,
        perception_anomaly_json: Option<&str>,
    ) -> Result<CheckpointData, CoreError> {
        let seq = self.counter.fetch_add(1, Ordering::SeqCst);
        let checkpoint_iri = format!(
            "iri://checkpoint/{}/{}/seq_{}_{}",
            task_iri.strip_prefix("iri://").unwrap_or(task_iri),
            kind.path_segment(),
            seq,
            uuid::Uuid::new_v4().hyphenated(),
        );

        let nodes: Vec<serde_json::Value> = serde_json::from_str(nodes_json).unwrap_or_default();
        let node_count = nodes.len() as i32;

        let mut total = nodes_json.len() as i32
            + session_messages_json.len() as i32
            + agent_state_json.len() as i32;
        if let Some(v) = five_w2h_json {
            total += v.len() as i32;
        }
        if let Some(v) = cycle_state_json {
            total += v.len() as i32;
        }
        if let Some(v) = completed_nodes_json {
            total += v.len() as i32;
        }
        if let Some(v) = pending_approvals_json {
            total += v.len() as i32;
        }
        if let Some(v) = supplement_json {
            total += v.len() as i32;
        }
        if let Some(v) = tool_error_json {
            total += v.len() as i32;
        }
        if let Some(v) = action_tracker_json {
            total += v.len() as i32;
        }
        if let Some(v) = perception_anomaly_json {
            total += v.len() as i32;
        }

        let previous_runtime_state = (kind == CheckpointKind::TaskRuntime)
            .then(|| self.latest_committed_runtime_state(task_iri))
            .flatten();
        let resume_state = match kind {
            CheckpointKind::TaskRuntime => Some(TaskResumeState::from_fields(
                &checkpoint_iri,
                name,
                session_messages_json,
                agent_state_json,
                current_role,
                prev_summary,
                completed_nodes_json,
                action_tracker_json,
                previous_runtime_state.as_ref(),
                self.load_task_contract(task_iri)?.ok_or_else(|| CoreError::Internal {
                    message: format!(
                        "Refusing task-runtime checkpoint for {task_iri}: canonical resume contract is not registered"
                    ),
                })?,
            )?),
            CheckpointKind::BizOrchestration => None,
        };
        if let Some(state) = &resume_state {
            total = total.saturating_add(
                serde_json::to_vec(state)
                    .map(|encoded| encoded.len().min(i32::MAX as usize) as i32)
                    .unwrap_or(0),
            );
        }

        let checkpoint = CheckpointData {
            kind,
            checkpoint_iri: checkpoint_iri.clone(),
            task_iri: task_iri.to_string(),
            name: name.to_string(),
            node_count,
            total_size_bytes: total,
            created_at: Utc::now(),
            tags: tags.to_vec(),
            nodes_json: nodes_json.to_string(),
            session_messages_json: session_messages_json.to_string(),
            agent_state_json: agent_state_json.to_string(),
            current_role: current_role.map(|s| s.to_string()),
            five_w2h_json: five_w2h_json.map(|s| s.to_string()),
            prev_summary: prev_summary.map(|s| s.to_string()),
            cycle_state_json: cycle_state_json.map(|s| s.to_string()),
            completed_nodes_json: completed_nodes_json.map(|s| s.to_string()),
            pending_approvals_json: pending_approvals_json.map(|s| s.to_string()),
            supplement_json: supplement_json.map(|s| s.to_string()),
            tool_error_json: tool_error_json.map(|s| s.to_string()),
            action_tracker_json: action_tracker_json.map(|s| s.to_string()),
            perception_anomaly_json: perception_anomaly_json.map(|s| s.to_string()),
            resume_state,
        };

        let content = serde_json::to_string(&checkpoint).map_err(|e| CoreError::Internal {
            message: format!("Failed to serialize checkpoint: {}", e),
        })?;
        self.store_checkpoint(&checkpoint_iri, &content)?;

        {
            let mut task_cps = self.task_checkpoints.write();
            task_cps
                .entry(task_iri.to_string())
                .or_insert_with(Vec::new)
                .push(checkpoint_iri.clone());
        }
        self.prune_oldest(task_iri, kind);

        Ok(checkpoint)
    }

    fn store_checkpoint(&self, iri: &str, content: &str) -> Result<(), CoreError> {
        if let Some(ref l0) = self.l0 {
            l0.store(iri, content)?;
        }
        Ok(())
    }

    pub fn restore(&self, checkpoint_iri: &str) -> Result<CheckpointData, CoreError> {
        if let Some(ref l0) = self.l0 {
            if let Ok(Some(entry)) = l0.retrieve(checkpoint_iri) {
                return Self::deserialize_checkpoint(&entry.content);
            }
        }
        Err(CoreError::Internal {
            message: format!("Checkpoint not found: {}", checkpoint_iri),
        })
    }

    pub fn restore_latest(&self, task_iri: &str) -> Result<Option<CheckpointData>, CoreError> {
        Ok(self
            .restore_task(task_iri)?
            .map(|restored| restored.checkpoint))
    }

    /// Restore the latest fully valid checkpoint and its parsed, versioned
    /// runtime state.  Invalid/corrupt newest records are skipped in favour of
    /// an older usable checkpoint rather than making recovery all-or-nothing.
    pub fn restore_task(&self, task_iri: &str) -> Result<Option<RestoredTask>, CoreError> {
        for checkpoint in self.list_by_kind(
            task_iri,
            CheckpointKind::TaskRuntime,
            MAX_CHECKPOINTS_PER_TASK as i32,
        ) {
            match checkpoint.to_restored_task() {
                Ok(restored) => return Ok(Some(restored)),
                Err(error) => tracing::warn!(
                    checkpoint_iri = %checkpoint.checkpoint_iri,
                    %error,
                    "Skipping invalid checkpoint during task restore"
                ),
            }
        }
        Ok(None)
    }

    /// Restore the latest checkpoint for a given task, parsing its phase label.
    /// Returns (checkpoint, phase_label) where phase_label is one of:
    ///   "start_<Role>" / "turn_<Role>_N" / "finish_<Role>" / "max_turns_<Role>"
    ///   "force_end_<Role>" / "step_complete_<Role>" / "pre_dispatch_<Role>"
    ///   or "unknown"
    pub fn restore_latest_with_phase(
        &self,
        task_iri: &str,
    ) -> Result<Option<(CheckpointData, String)>, CoreError> {
        let cp = self.restore_latest(task_iri)?;
        Ok(cp.map(|c| {
            let phase = parse_checkpoint_phase(&c.name);
            (c, phase)
        }))
    }

    pub fn list(&self, task_iri: &str, limit: i32) -> Vec<CheckpointData> {
        if limit <= 0 {
            return Vec::new();
        }
        // L0 is the authority. The per-manager index is deliberately not used
        // as a complete view: managers are short-lived, so after creating one
        // new checkpoint their local index may omit older valid boundaries
        // written by another manager/process.
        if let Some(ref l0) = self.l0 {
            let stripped = task_iri.strip_prefix("iri://").unwrap_or(task_iri);
            let prefix = format!("iri://checkpoint/{}/", stripped);
            if let Ok(entries) = l0.scan_iri_prefix(&prefix, 100_000) {
                let mut results: Vec<CheckpointData> = entries
                    .iter()
                    .filter_map(|e| Self::deserialize_checkpoint(&e.content).ok())
                    .filter(|checkpoint| checkpoint.task_iri == task_iri)
                    .collect();
                results.sort_by(|a, b| {
                    b.created_at
                        .cmp(&a.created_at)
                        .then_with(|| b.checkpoint_iri.cmp(&a.checkpoint_iri))
                });
                results.truncate(limit as usize);
                return results;
            }
        }
        Vec::new()
    }

    pub fn list_by_kind(
        &self,
        task_iri: &str,
        kind: CheckpointKind,
        limit: i32,
    ) -> Vec<CheckpointData> {
        if limit <= 0 {
            return Vec::new();
        }
        let mut checkpoints = self
            // Filter by protocol before applying the caller limit. This also
            // finds a runtime checkpoint behind an arbitrary number of newer
            // BizAgent records left by an older deployment.
            .list(task_iri, i32::MAX)
            .into_iter()
            .filter(|checkpoint| checkpoint.kind == kind)
            .collect::<Vec<_>>();
        checkpoints.truncate(limit as usize);
        checkpoints
    }

    pub fn delete(&self, checkpoint_iri: &str) -> Result<(), CoreError> {
        if let Some(ref l0) = self.l0 {
            if l0.retrieve(checkpoint_iri)?.is_none() {
                return Err(CoreError::Internal {
                    message: format!("Checkpoint not found: {}", checkpoint_iri),
                });
            }
            l0.delete(checkpoint_iri)?;
        }
        {
            let mut task_cps = self.task_checkpoints.write();
            for iris in task_cps.values_mut() {
                iris.retain(|iri| iri != checkpoint_iri);
            }
        }
        Ok(())
    }

    pub fn checkpoint_count(&self) -> u64 {
        self.task_checkpoints
            .read()
            .values()
            .map(|v| v.len() as u64)
            .sum()
    }

    fn prune_oldest(&self, task_iri: &str, kind: CheckpointKind) {
        {
            let mut task_cps = self.task_checkpoints.write();
            if let Some(iris) = task_cps.get_mut(task_iri) {
                let mut same_kind_seen = 0usize;
                let mut retained = Vec::with_capacity(iris.len());
                for iri in iris.iter().rev() {
                    let is_same_kind = self
                        .l0
                        .as_ref()
                        .and_then(|l0| l0.retrieve(iri).ok().flatten())
                        .and_then(|entry| Self::deserialize_checkpoint(&entry.content).ok())
                        .is_some_and(|checkpoint| checkpoint.kind == kind);
                    if is_same_kind {
                        same_kind_seen = same_kind_seen.saturating_add(1);
                    }
                    if !is_same_kind || same_kind_seen <= MAX_CHECKPOINTS_PER_TASK {
                        retained.push(iri.clone());
                    }
                }
                retained.reverse();
                *iris = retained;
            }
        }

        // CheckpointManager instances are intentionally short-lived in some
        // execution paths. Enforcing retention only through their in-memory
        // index both leaked old checkpoints and allowed a fresh manager's
        // `seq_0` to overwrite another BizAgent's evidence. UUID-backed IRIs
        // prevent collisions; this persisted scan enforces the task-wide cap
        // across roles and process lifetimes.
        let Some(ref l0) = self.l0 else {
            return;
        };
        let stripped = task_iri.strip_prefix("iri://").unwrap_or(task_iri);
        let prefix = format!("iri://checkpoint/{}/", stripped);
        let Ok(entries) = l0.scan_iri_prefix(&prefix, 100_000) else {
            return;
        };
        let mut checkpoints = entries
            .into_iter()
            .filter_map(|entry| {
                serde_json::from_str::<CheckpointData>(&entry.content)
                    .ok()
                    .filter(|checkpoint| checkpoint.kind == kind)
                    .map(|checkpoint| (entry.iri, checkpoint.created_at))
            })
            .collect::<Vec<_>>();
        checkpoints.sort_by_key(|(_, created_at)| *created_at);
        let remove_count = checkpoints.len().saturating_sub(MAX_CHECKPOINTS_PER_TASK);
        for (iri, _) in checkpoints.into_iter().take(remove_count) {
            let _ = l0.delete(&iri);
            let mut task_cps = self.task_checkpoints.write();
            for iris in task_cps.values_mut() {
                iris.retain(|candidate| candidate != &iri);
            }
        }
    }
}

fn task_contract_iri(task_iri: &str) -> String {
    format!(
        "iri://task-resume-contract/{}",
        hex::encode(Sha256::digest(task_iri.as_bytes()))
    )
}

pub(crate) fn encode_dag_resume_state(
    completed_nodes: &HashMap<String, crate::core::workflow::NodeResult>,
    skipped_nodes: &std::collections::HashSet<String>,
) -> Result<String, CoreError> {
    let state = PersistedDagResumeState {
        completed_nodes: completed_nodes
            .iter()
            .map(|(node_id, result)| (node_id.clone(), result.clone()))
            .collect(),
        skipped_nodes: skipped_nodes.iter().cloned().collect(),
    };
    serde_json::to_string(&state).map_err(|error| CoreError::Internal {
        message: format!("Failed to serialize typed DAG checkpoint state: {error}"),
    })
}

pub(crate) fn sha256_receipt(payload: &str) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(payload.as_bytes())))
}

/// Attach the exact active-node continuation receipt to AgentRunner's local
/// diagnostic state.  Task counters are intentionally not changed here: the
/// checkpoint reader inherits those only from the latest committed SA state.
pub(crate) fn encode_active_node_agent_state(
    agent_state_json: &str,
    session_messages_json: &str,
    identity: &ActiveNodeIdentity,
) -> Result<String, CoreError> {
    let mut state =
        serde_json::from_str::<serde_json::Value>(agent_state_json).map_err(|error| {
            CoreError::Internal {
                message: format!("Invalid AgentRunner checkpoint state: {error}"),
            }
        })?;
    let state_object = state.as_object_mut().ok_or_else(|| CoreError::Internal {
        message: "AgentRunner checkpoint state must be a JSON object".to_string(),
    })?;
    let read_count = |field: &str| {
        state_object
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            .min(u64::from(u32::MAX)) as u32
    };
    let continuation = ActiveNodeContinuation {
        schema_version: ACTIVE_NODE_CONTINUATION_SCHEMA_VERSION,
        step_id: identity.step_id.clone(),
        dispatch_id: identity.dispatch_id.clone(),
        agent_id: identity.agent_id.clone(),
        l1_session_id: identity.l1_session_id.clone(),
        role: identity.role,
        agent_md_sha256: identity.agent_md_sha256.clone(),
        context_manifest_sha256: identity.context_manifest_sha256.clone(),
        source_interaction_id: identity.source_interaction_id.clone(),
        transcript_sha256: sha256_receipt(session_messages_json),
        local_turn_count: read_count("turn"),
        local_tool_call_count: read_count("tc"),
    };
    state_object.insert(
        "active_continuation".to_string(),
        serde_json::to_value(continuation).map_err(|error| CoreError::Internal {
            message: format!("Failed to serialize active-node continuation: {error}"),
        })?,
    );
    serde_json::to_string(&state).map_err(|error| CoreError::Internal {
        message: format!("Failed to serialize AgentRunner checkpoint state: {error}"),
    })
}

fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => serde_json::to_string(value).unwrap_or_default(),
        serde_json::Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        serde_json::Value::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            format!(
                "{{{}}}",
                entries
                    .into_iter()
                    .map(|(key, value)| format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap_or_default(),
                        canonical_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

impl TaskResumeContract {
    pub fn new(
        original_user_task: impl Into<String>,
        constraints: &HashMap<String, String>,
        effect_policy: crate::core::effect::EffectPolicy,
        execution_plan: crate::core::sa::ExecutionPlan,
    ) -> Result<Self, CoreError> {
        let constraints = constraints
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        let delivery_target = constraints
            .get(crate::core::agent_runner::DELIVERY_TARGET_PATH_CONSTRAINT)
            .cloned();
        let mut contract = Self {
            schema_version: TASK_RESUME_CONTRACT_SCHEMA_VERSION,
            original_user_task: original_user_task.into(),
            constraints,
            effect_policy,
            delivery_target,
            execution_plan,
            fingerprint: String::new(),
        };
        contract.fingerprint = contract.compute_fingerprint()?;
        contract.validate()?;
        Ok(contract)
    }

    pub fn constraints_hash_map(&self) -> HashMap<String, String> {
        self.constraints
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    fn compute_fingerprint(&self) -> Result<String, CoreError> {
        let value = serde_json::json!({
            "schema_version": self.schema_version,
            "original_user_task": self.original_user_task,
            "constraints": self.constraints,
            "effect_policy": self.effect_policy,
            "delivery_target": self.delivery_target,
            "execution_plan": self.execution_plan,
        });
        Ok(format!(
            "sha256:{}",
            hex::encode(Sha256::digest(canonical_json(&value).as_bytes()))
        ))
    }

    pub fn validate(&self) -> Result<(), CoreError> {
        if self.schema_version != TASK_RESUME_CONTRACT_SCHEMA_VERSION {
            return Err(CoreError::Internal {
                message: format!(
                    "Unsupported task resume contract schema version {}",
                    self.schema_version
                ),
            });
        }
        if self.original_user_task.trim().is_empty() {
            return Err(CoreError::Internal {
                message: "Task resume contract has no original user task".to_string(),
            });
        }
        if self.execution_plan.plan_id.trim().is_empty() || self.execution_plan.steps.is_empty() {
            return Err(CoreError::Internal {
                message: "Task resume contract has no executable plan identity".to_string(),
            });
        }
        let active_sequence = self
            .execution_plan
            .steps
            .iter()
            .map(|step| step.role)
            .collect::<Vec<_>>();
        if self.execution_plan.agent_sequence != active_sequence {
            return Err(CoreError::Internal {
                message: "Task resume contract agent_sequence disagrees with its active role steps"
                    .to_string(),
            });
        }
        if self.execution_plan.verify_first {
            if active_sequence
                != [
                    crate::core::agent_instance::AgentRole::Check,
                    crate::core::agent_instance::AgentRole::Act,
                ]
                || self.execution_plan.fallback_steps.is_empty()
            {
                return Err(CoreError::Internal {
                    message: "Task resume verify-first contract requires exactly CA→AA plus a non-empty fallback plan".to_string(),
                });
            }
        } else if !self.execution_plan.fallback_steps.is_empty() {
            return Err(CoreError::Internal {
                message:
                    "Task resume non-verify-first contract must not carry dormant fallback steps"
                        .to_string(),
            });
        }
        let provenance = self
            .execution_plan
            .agent_spec_provenance
            .as_ref()
            .ok_or_else(|| CoreError::Internal {
                message: "Task resume contract plan has no provenance".to_string(),
            })?;
        provenance.validate().map_err(|error| CoreError::Internal {
            message: format!("Task resume contract plan provenance is invalid: {error}"),
        })?;
        let validate_step_collection = |label: &str,
                                        steps: &[crate::core::sa::PlanStep]|
         -> Result<BTreeSet<String>, CoreError> {
            let mut ids = BTreeSet::new();
            for step in steps {
                if step.step_id.trim().is_empty() || !ids.insert(step.step_id.clone()) {
                    return Err(CoreError::Internal {
                        message: format!(
                            "Task resume contract {label} has an empty or duplicate step identity"
                        ),
                    });
                }
                if self.execution_plan.dag_jsonld.is_none()
                    && [
                        step.objective.as_str(),
                        step.expected_output.as_str(),
                        step.success_criteria.as_str(),
                    ]
                    .iter()
                    .any(|field| field.trim().is_empty())
                {
                    return Err(CoreError::Internal {
                        message: format!(
                            "Task resume contract {label} step '{}' has an empty role definition field",
                            step.step_id
                        ),
                    });
                }
            }
            if steps.iter().any(|step| {
                step.dependencies
                    .iter()
                    .any(|dependency| !ids.contains(dependency))
            }) {
                return Err(CoreError::Internal {
                    message: format!("Task resume contract {label} has an unknown step dependency"),
                });
            }
            crate::core::sa::validate_generated_step_dag(steps).map_err(|reason| {
                CoreError::Internal {
                    message: format!(
                        "Task resume contract {label} has an invalid parent-step DAG: {reason}"
                    ),
                }
            })?;
            Ok(ids)
        };
        let active_step_ids = validate_step_collection("active plan", &self.execution_plan.steps)?;
        let fallback_step_ids = if self.execution_plan.verify_first {
            validate_step_collection("fallback plan", &self.execution_plan.fallback_steps)?
        } else {
            BTreeSet::new()
        };
        let plan_step_ids = active_step_ids
            .iter()
            .chain(fallback_step_ids.iter())
            .cloned()
            .collect::<BTreeSet<_>>();
        if provenance
            .step_sources
            .keys()
            .any(|step_id| !plan_step_ids.contains(step_id.as_str()))
        {
            return Err(CoreError::Internal {
                message: "Task resume contract provenance references an unknown step".to_string(),
            });
        }
        let workflow_plan = self.execution_plan.dag_jsonld.is_some();
        for step in self
            .execution_plan
            .steps
            .iter()
            .chain(self.execution_plan.fallback_steps.iter())
        {
            crate::core::sa::validate_plan_work_package_dag(&step.work_packages).map_err(
                |error| CoreError::Internal {
                    message: format!(
                        "Task resume contract step '{}' has an invalid work-package DAG: {error}",
                        step.step_id
                    ),
                },
            )?;
            for package in &step.work_packages {
                crate::core::sa::validate_work_package_evidence_requirements(package, true)
                    .map_err(|error| CoreError::Internal {
                        message: format!(
                            "Task resume contract step '{}' has an invalid typed evidence contract: {error}",
                            step.step_id
                        ),
                    })?;
            }
            let source = self
                .execution_plan
                .agent_spec_source_for_step(&step.step_id)
                .map_err(|error| CoreError::Internal {
                    message: format!(
                        "Task resume contract cannot resolve agent.md source for step '{}': {error}",
                        step.step_id
                    ),
                })?
                .ok_or_else(|| CoreError::Internal {
                    message: format!(
                        "Task resume contract step '{}' has no agent.md source",
                        step.step_id
                    ),
                })?;
            let permitted = if workflow_plan {
                source.kind == crate::core::context_model::AgentSpecSourceKind::WorkflowDefinition
            } else {
                matches!(
                    source.kind,
                    crate::core::context_model::AgentSpecSourceKind::LlmGeneratedPlan
                        | crate::core::context_model::AgentSpecSourceKind::AgentHandoffPlan
                )
            };
            if !permitted {
                return Err(CoreError::Internal {
                    message: format!(
                        "Task resume contract refuses {:?} agent.md provenance for step '{}'",
                        source.kind, step.step_id
                    ),
                });
            }
        }
        if !workflow_plan {
            let normalized_steps = if self.execution_plan.verify_first {
                &self.execution_plan.fallback_steps
            } else {
                &self.execution_plan.steps
            };
            crate::core::sa::validate_normalized_generated_plan_work_package_contract(
                normalized_steps,
            )
            .map_err(|error| CoreError::Internal {
                message: format!(
                    "Task resume contract generated plan violates the normalized two-level work-package contract: {error}"
                ),
            })?;
            if crate::core::sa::user_explicitly_requires_order(&self.original_user_task)
                && normalized_steps
                    .iter()
                    .any(|step| step.role == crate::core::agent_instance::AgentRole::Do)
                && !normalized_steps.iter().any(|step| {
                    step.role == crate::core::agent_instance::AgentRole::Do
                        && step
                            .work_packages
                            .iter()
                            .any(|package| !package.dependencies.is_empty())
                })
            {
                return Err(CoreError::Internal {
                    message: "Task resume contract dropped the original user's explicit Do work-package order".to_string(),
                });
            }
        }
        let declared_target = self
            .constraints
            .get(crate::core::agent_runner::DELIVERY_TARGET_PATH_CONSTRAINT);
        if declared_target != self.delivery_target.as_ref() {
            return Err(CoreError::Internal {
                message: "Task resume delivery target disagrees with authoritative constraints"
                    .to_string(),
            });
        }
        if let Some(target) = self.delivery_target.as_deref() {
            let path = std::path::Path::new(target);
            if target.trim().is_empty()
                || path.is_absolute()
                || !path
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_)))
            {
                return Err(CoreError::Internal {
                    message: "Task resume delivery target must be an exact workspace-relative path"
                        .to_string(),
                });
            }
        }
        if self
            .constraints
            .get("required_effect")
            .is_some_and(|effect| effect == "workspace_mutation")
            && !self.effect_policy.requires_workspace_mutation()
        {
            return Err(CoreError::Internal {
                message: "Task resume effect policy weakens a required workspace mutation"
                    .to_string(),
            });
        }
        if let Some(declared_policy) = self.constraints.get("effect_policy") {
            let consistent = match declared_policy.as_str() {
                "evidence_only" => {
                    self.effect_policy == crate::core::effect::EffectPolicy::EvidenceOnly
                }
                "decision_only" => {
                    self.effect_policy == crate::core::effect::EffectPolicy::DecisionOnly
                }
                "conditional_workspace_mutation" => matches!(
                    &self.effect_policy,
                    crate::core::effect::EffectPolicy::Conditional {
                        effect: crate::core::effect::EffectKind::WorkspaceMutation,
                        ..
                    }
                ),
                "required_workspace_mutation" => self.effect_policy.requires_workspace_mutation(),
                _ => false,
            };
            if !consistent {
                return Err(CoreError::Internal {
                    message:
                        "Task resume typed effect policy disagrees with its declared constraint"
                            .to_string(),
                });
            }
        }
        let expected = self.compute_fingerprint()?;
        if self.fingerprint != expected {
            return Err(CoreError::Internal {
                message: "Task resume contract fingerprint mismatch".to_string(),
            });
        }
        Ok(())
    }
}

impl TaskResumeState {
    pub fn observed_turn_count(&self) -> u32 {
        self.task_cumulative.turn_count.saturating_add(
            self.active_continuation
                .as_ref()
                .map_or(0, |continuation| continuation.local_turn_count),
        )
    }

    pub fn observed_tool_call_count(&self) -> u32 {
        self.task_cumulative.tool_call_count.saturating_add(
            self.active_continuation
                .as_ref()
                .map_or(0, |continuation| continuation.local_tool_call_count),
        )
    }

    fn from_fields(
        checkpoint_iri: &str,
        checkpoint_name: &str,
        session_messages_json: &str,
        agent_state_json: &str,
        current_role: Option<&str>,
        prev_summary: Option<&str>,
        completed_nodes_json: Option<&str>,
        action_tracker_json: Option<&str>,
        previous: Option<&TaskResumeState>,
        contract: TaskResumeContract,
    ) -> Result<Self, CoreError> {
        let state = serde_json::from_str::<serde_json::Value>(agent_state_json).ok();
        let local_turn_count = state
            .as_ref()
            .and_then(|value| value.get("turn"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            .min(u64::from(u32::MAX)) as u32;
        let local_tool_call_count = state
            .as_ref()
            .and_then(|value| value.get("tc"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            .min(u64::from(u32::MAX)) as u32;
        let is_sa_commit = checkpoint_name.starts_with("step_complete_");
        let active_continuation = state
            .as_ref()
            .and_then(|value| value.get("active_continuation"))
            .cloned()
            .map(serde_json::from_value::<ActiveNodeContinuation>)
            .transpose()
            .map_err(|error| CoreError::Internal {
                message: format!("Invalid active-node continuation receipt: {error}"),
            })?;
        if is_sa_commit && active_continuation.is_some() {
            return Err(CoreError::Internal {
                message: "SA step-complete checkpoint cannot contain an active-node continuation"
                    .to_string(),
            });
        }
        if !is_sa_commit && active_continuation.is_none() {
            return Err(CoreError::Internal {
                message: "AgentRunner checkpoint is missing its active-node continuation receipt"
                    .to_string(),
            });
        }
        let task_cumulative = if is_sa_commit {
            TaskCumulativeState {
                turn_count: local_turn_count,
                tool_call_count: local_tool_call_count,
            }
        } else {
            previous
                .map(|state| state.task_cumulative.clone())
                .unwrap_or(TaskCumulativeState {
                    turn_count: 0,
                    tool_call_count: 0,
                })
        };
        let (completed_nodes, skipped_nodes) = match completed_nodes_json {
            Some(value) => {
                let dag_state =
                    serde_json::from_str::<PersistedDagResumeState>(value).map_err(|error| {
                        CoreError::Internal {
                            message: format!("Invalid DAG checkpoint state: {error}"),
                        }
                    })?;
                (dag_state.completed_nodes, dag_state.skipped_nodes)
            }
            None if is_sa_commit => {
                return Err(CoreError::Internal {
                    message: "SA step-complete checkpoint is missing typed DAG state".to_string(),
                })
            }
            None => previous
                .map(|state| (state.completed_nodes.clone(), state.skipped_nodes.clone()))
                .unwrap_or_default(),
        };
        // SA supplies the complete task-wide receipt set at its commit
        // boundary. ReAct supplies only the current agent's receipts, which
        // are merged on top of the latest committed SA base.
        let mut tracked_actions = if is_sa_commit {
            Vec::new()
        } else {
            previous
                .map(|state| state.tracked_actions.clone())
                .unwrap_or_default()
        };
        if let Some(value) = action_tracker_json {
            let current =
                serde_json::from_str::<Vec<crate::core::tracked_action::TrackedAction>>(value)
                    .map_err(|error| CoreError::Internal {
                        message: format!("Invalid tracked-action checkpoint state: {error}"),
                    })?;
            for action in current {
                if !tracked_actions
                    .iter()
                    .any(|existing| existing.action_id == action.action_id)
                {
                    tracked_actions.push(action);
                }
            }
        }
        let committed_action_ids = if is_sa_commit {
            tracked_actions
                .iter()
                .map(|action| action.action_id.clone())
                .collect()
        } else {
            previous
                .map(|state| state.committed_action_ids.clone())
                .unwrap_or_default()
        };
        let resume_state = Self {
            schema_version: TASK_RESUME_STATE_SCHEMA_VERSION,
            checkpoint_iri: checkpoint_iri.to_string(),
            checkpoint_name: checkpoint_name.to_string(),
            task_cumulative,
            active_continuation,
            current_role: current_role.map(str::to_string),
            prev_summary: prev_summary
                .map(str::to_string)
                .or_else(|| previous.and_then(|state| state.prev_summary.clone())),
            tracked_actions,
            committed_action_ids,
            completed_nodes,
            skipped_nodes,
            contract,
        };
        resume_state.validated_for_transcript(session_messages_json)
    }

    pub fn validate(&self) -> Result<(), CoreError> {
        if self.schema_version != TASK_RESUME_STATE_SCHEMA_VERSION {
            return Err(CoreError::Internal {
                message: format!(
                    "Unsupported task resume schema version {}",
                    self.schema_version
                ),
            });
        }
        if self.checkpoint_iri.is_empty() || self.checkpoint_name.is_empty() {
            return Err(CoreError::Internal {
                message: "Task resume state has no checkpoint identity".to_string(),
            });
        }
        let is_sa_commit = self.checkpoint_name.starts_with("step_complete_");
        match self.active_continuation.as_ref() {
            Some(_) if is_sa_commit => {
                return Err(CoreError::Internal {
                    message: "SA step-complete resume state contains an active-node continuation"
                        .to_string(),
                });
            }
            None if !is_sa_commit => {
                return Err(CoreError::Internal {
                    message: "AgentRunner resume state has no active-node continuation".to_string(),
                });
            }
            Some(continuation) => continuation.validate(self.current_role.as_deref())?,
            None => {}
        }
        if self.committed_action_ids.iter().any(|committed_id| {
            !self
                .tracked_actions
                .iter()
                .any(|action| &action.action_id == committed_id)
        }) {
            return Err(CoreError::Internal {
                message: "Task resume state commits an unknown action receipt".to_string(),
            });
        }
        if self
            .completed_nodes
            .iter()
            .any(|(node_id, result)| node_id != &result.node_id)
        {
            return Err(CoreError::Internal {
                message: "Task resume state has a mismatched completed-node identity".to_string(),
            });
        }
        let plan = &self.contract.execution_plan;
        let runtime_node_ids = plan
            .steps
            .iter()
            .map(|step| {
                if plan.dag_jsonld.is_some() {
                    step.step_id.clone()
                } else {
                    format!("wf:{}/{}", plan.plan_id, step.step_id)
                }
            })
            .collect::<BTreeSet<_>>();
        if self
            .completed_nodes
            .keys()
            .chain(self.skipped_nodes.iter())
            .any(|node_id| !runtime_node_ids.contains(node_id))
        {
            return Err(CoreError::Internal {
                message: "Task resume state references a node outside the persisted plan"
                    .to_string(),
            });
        }
        self.contract.validate()?;
        Ok(())
    }

    fn validated_for_transcript(self, session_messages_json: &str) -> Result<Self, CoreError> {
        self.validate_for_transcript(session_messages_json)?;
        Ok(self)
    }

    fn validate_for_transcript(&self, session_messages_json: &str) -> Result<(), CoreError> {
        self.validate()?;
        match self.active_continuation.as_ref() {
            Some(continuation) => {
                let actual = sha256_receipt(session_messages_json);
                if continuation.transcript_sha256 != actual {
                    return Err(CoreError::Internal {
                        message: "Active-node continuation transcript hash mismatch".to_string(),
                    });
                }
            }
            None => {
                let messages = serde_json::from_str::<Vec<ChatMessage>>(session_messages_json)
                    .map_err(|error| CoreError::Internal {
                        message: format!("Invalid SA checkpoint session messages: {error}"),
                    })?;
                if !messages.is_empty() {
                    return Err(CoreError::Internal {
                        message:
                            "SA step-complete checkpoint must not carry a mixed Agent transcript"
                                .to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_for_resume(&self) -> Result<(), CoreError> {
        self.validate()?;
        let has_uncommitted_effect = self.tracked_actions.iter().any(|action| {
            action.substantive_effect && !self.committed_action_ids.contains(&action.action_id)
        });
        if has_uncommitted_effect {
            return Err(CoreError::Internal {
                message: "A substantive tool effect has not crossed an SA step-complete boundary; automatic replay is refused"
                    .to_string(),
            });
        }
        Ok(())
    }
}

impl ActiveNodeContinuation {
    pub(crate) fn matches_identity(&self, identity: &ActiveNodeIdentity) -> bool {
        self.step_id == identity.step_id
            && self.dispatch_id == identity.dispatch_id
            && self.agent_id == identity.agent_id
            && self.l1_session_id == identity.l1_session_id
            && self.role == identity.role
            && self.agent_md_sha256 == identity.agent_md_sha256
            && self.context_manifest_sha256 == identity.context_manifest_sha256
            && self.source_interaction_id == identity.source_interaction_id
    }

    fn validate(&self, current_role: Option<&str>) -> Result<(), CoreError> {
        if self.schema_version != ACTIVE_NODE_CONTINUATION_SCHEMA_VERSION {
            return Err(CoreError::Internal {
                message: format!(
                    "Unsupported active-node continuation schema version {}",
                    self.schema_version
                ),
            });
        }
        if [
            self.step_id.as_str(),
            self.dispatch_id.as_str(),
            self.agent_id.as_str(),
            self.l1_session_id.as_str(),
            self.agent_md_sha256.as_str(),
            self.context_manifest_sha256.as_str(),
            self.transcript_sha256.as_str(),
        ]
        .iter()
        .any(|value| value.trim().is_empty())
        {
            return Err(CoreError::Internal {
                message: "Active-node continuation has an incomplete identity receipt".to_string(),
            });
        }
        if ![
            self.agent_md_sha256.as_str(),
            self.context_manifest_sha256.as_str(),
            self.transcript_sha256.as_str(),
        ]
        .iter()
        .all(|hash| {
            hash.strip_prefix("sha256:").is_some_and(|digest| {
                digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        }) {
            return Err(CoreError::Internal {
                message: "Active-node continuation contains an invalid SHA-256 receipt".to_string(),
            });
        }
        if current_role.is_some_and(|role| role != self.role.to_string()) {
            return Err(CoreError::Internal {
                message: "Active-node continuation role does not match checkpoint role".to_string(),
            });
        }
        Ok(())
    }
}

impl CheckpointData {
    pub fn to_restored_task(&self) -> Result<RestoredTask, CoreError> {
        if self.kind != CheckpointKind::TaskRuntime {
            return Err(CoreError::Internal {
                message: format!(
                    "Checkpoint {} is {:?}, not a task-runtime checkpoint",
                    self.checkpoint_iri, self.kind
                ),
            });
        }
        if self.task_iri.is_empty() || self.checkpoint_iri.is_empty() {
            return Err(CoreError::Internal {
                message: "Checkpoint is missing task or checkpoint IRI".to_string(),
            });
        }
        let messages = serde_json::from_str::<Vec<ChatMessage>>(&self.session_messages_json)
            .map_err(|error| CoreError::Internal {
                message: format!("Invalid checkpoint session messages: {error}"),
            })?;
        let state = self
            .resume_state
            .clone()
            .ok_or_else(|| CoreError::Internal {
                message: "Task-runtime checkpoint has no canonical resume state".to_string(),
            })?;
        state.validate_for_transcript(&self.session_messages_json)?;
        state.validate_for_resume()?;
        if state.checkpoint_name != self.name {
            return Err(CoreError::Internal {
                message: "Checkpoint resume state does not match checkpoint name".to_string(),
            });
        }
        if state.checkpoint_iri != self.checkpoint_iri {
            return Err(CoreError::Internal {
                message: "Checkpoint resume state does not match checkpoint IRI".to_string(),
            });
        }
        Ok(RestoredTask {
            checkpoint: self.clone(),
            messages,
            state,
        })
    }
}

impl CheckpointManager {
    fn deserialize_checkpoint(content: &str) -> Result<CheckpointData, CoreError> {
        let checkpoint = serde_json::from_str::<CheckpointData>(content).map_err(|error| {
            CoreError::Internal {
                message: format!("Invalid checkpoint data: {error}"),
            }
        })?;
        // Session message validation happens in restore_task.  Structural and
        // schema validation happens here so unsupported snapshots never enter
        // the candidate set.
        if checkpoint.task_iri.is_empty() || checkpoint.checkpoint_iri.is_empty() {
            return Err(CoreError::Internal {
                message: "Checkpoint is missing task or checkpoint IRI".to_string(),
            });
        }
        match checkpoint.kind {
            CheckpointKind::TaskRuntime => checkpoint
                .resume_state
                .as_ref()
                .ok_or_else(|| CoreError::Internal {
                    message: "Task-runtime checkpoint has no canonical resume state".to_string(),
                })?
                .validate()?,
            CheckpointKind::BizOrchestration if checkpoint.resume_state.is_some() => {
                return Err(CoreError::Internal {
                    message: "BizAgent orchestration checkpoint contains task resume state"
                        .to_string(),
                });
            }
            CheckpointKind::BizOrchestration => {}
        }
        Ok(checkpoint)
    }
}

#[cfg(test)]
pub(crate) fn test_resume_contract(original_user_task: &str) -> TaskResumeContract {
    use crate::core::agent_instance::AgentRole;
    use crate::core::context_model::{
        AgentSpecSourceKind, AgentSpecSourceRecord, ExecutionPlanProvenance,
    };
    use crate::core::sa::{ExecutionPlan, PlanStep, TaskComplexity};

    let plan_id = "test-resume-plan".to_string();
    let roles = [
        AgentRole::Plan,
        AgentRole::Do,
        AgentRole::Check,
        AgentRole::Act,
    ];
    TaskResumeContract::new(
        original_user_task,
        &HashMap::new(),
        crate::core::effect::EffectPolicy::None,
        ExecutionPlan {
            plan_id: plan_id.clone(),
            agent_sequence: vec![
                AgentRole::Plan,
                AgentRole::Do,
                AgentRole::Check,
                AgentRole::Act,
            ],
            parallel_groups: Vec::new(),
            task_complexity: TaskComplexity::Standard,
            description: "test resume plan".to_string(),
            steps: roles
                .iter()
                .enumerate()
                .map(|(index, role)| PlanStep {
                    step_id: format!("test-{}", role),
                    role: *role,
                    objective: original_user_task.to_string(),
                    expected_output: "test output".to_string(),
                    dependencies: (index > 0)
                        .then(|| format!("test-{}", roles[index - 1]))
                        .into_iter()
                        .collect(),
                    tools_allowed: Vec::new(),
                    success_criteria: "test success".to_string(),
                    work_packages: Vec::new(),
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: match role {
                        AgentRole::Plan | AgentRole::Check => {
                            crate::core::effect::EffectPolicy::EvidenceOnly
                        }
                        AgentRole::Act => crate::core::effect::EffectPolicy::DecisionOnly,
                        AgentRole::Do => crate::core::effect::EffectPolicy::None,
                    },
                })
                .collect(),
            agent_spec_provenance: Some(ExecutionPlanProvenance::new(
                AgentSpecSourceRecord::new(AgentSpecSourceKind::LlmGeneratedPlan)
                    .with_source_ref(plan_id)
                    .with_producer("SupervisorAgent.plan_generation")
                    .with_model("checkpoint-test-model")
                    .with_interaction_id("checkpoint-test-plan-interaction"),
            )),
            context_requirements: HashMap::new(),
            success_metrics: vec!["test success".to_string()],
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: Vec::new(),
        },
    )
    .expect("test resume contract is valid")
}

/// Parse the phase label from a checkpoint name.
/// Examples: "start_DA" → "start_DA", "turn_CA_5" → "turn_CA_5", "finish_PA" → "finish_PA"
///       "step_complete_Do" → "step_complete_Do", "unknown_xxx" → "unknown"
pub fn parse_checkpoint_phase(name: &str) -> String {
    let known_prefixes = [
        "start_",
        "turn_",
        "finish_",
        "max_turns_",
        "force_end_",
        "step_complete_",
        "pre_dispatch_",
        "plan_created_",
    ];
    for prefix in &known_prefixes {
        if name.starts_with(prefix) {
            // Extract the role portion: "start_DA" → extract "DA" portion as phase
            // For turn_N_Role format: "turn_DA_5" → extract role between prefix and last _
            let rest = name.strip_prefix(prefix).unwrap_or("");
            if *prefix == "turn_" {
                // "turn_DA_5" → split by _, take first part
                if let Some(role) = rest.split('_').next() {
                    if matches!(
                        role,
                        "PA" | "DA" | "CA" | "AA" | "Plan" | "Do" | "Check" | "Act"
                    ) {
                        return format!("turn_{}", role);
                    }
                }
                return format!("turn_{}", rest);
            }
            return name.to_string();
        }
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn register_test_contract(manager: &CheckpointManager, task_iri: &str) {
        manager
            .register_task_contract(task_iri, test_resume_contract("original test task"))
            .unwrap();
    }

    fn active_state(
        agent_state_json: &str,
        session_messages_json: &str,
        role: crate::core::agent_instance::AgentRole,
    ) -> String {
        encode_active_node_agent_state(
            agent_state_json,
            session_messages_json,
            &ActiveNodeIdentity {
                step_id: format!("test-{role}"),
                dispatch_id: format!("test-dispatch-{role}"),
                agent_id: format!("test-agent-{role}"),
                l1_session_id: format!("test-l1-{role}"),
                role,
                agent_md_sha256: sha256_receipt("# generated test agent"),
                context_manifest_sha256: sha256_receipt("test-context-manifest"),
                source_interaction_id: Some("llm-test-interaction".to_string()),
            },
        )
        .unwrap()
    }

    #[test]
    fn test_create_in_memory() {
        let manager = CheckpointManager::new();
        register_test_contract(&manager, "iri://task/123");
        let checkpoint = manager
            .create(
                "iri://task/123",
                "test",
                r#"[{"@id":"iri://node/1"}]"#,
                r#"[{"role":"user"}]"#,
                &active_state(
                    r#"{"status":"running"}"#,
                    r#"[{"role":"user"}]"#,
                    crate::core::agent_instance::AgentRole::Plan,
                ),
                &["important".to_string()],
            )
            .unwrap();
        assert!(checkpoint.checkpoint_iri.starts_with("iri://checkpoint/"));
        assert_eq!(checkpoint.task_iri, "iri://task/123");
    }

    #[test]
    fn test_list_empty() {
        let manager = CheckpointManager::new();
        let list = manager.list("iri://task/nonexistent", 10);
        assert!(list.is_empty());
    }

    #[test]
    fn test_list_via_l0_scan_cross_process() {
        use crate::memory::l0_store::L0Store;
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mgr = CheckpointManager::with_persistence(l0.clone());
        register_test_contract(&mgr, "iri://task/abc-123");

        // Create checkpoint (simulating running in a previous process)
        mgr.create(
            "iri://task/abc-123",
            "finish_DA",
            "[]",
            r#"[{"role":"user","content":"hello"}]"#,
            &active_state(
                r#"{"turn":3}"#,
                r#"[{"role":"user","content":"hello"}]"#,
                crate::core::agent_instance::AgentRole::Do,
            ),
            &["DA".to_string()],
        )
        .unwrap();

        // New CheckpointManager (simulating cross-process: new instance, empty memory index)
        let mgr2 = CheckpointManager::with_persistence(l0.clone());

        // restore_latest must find the checkpoint (fallback via scan_iri_prefix)
        let cp = mgr2.restore_latest("iri://task/abc-123").unwrap();
        assert!(cp.is_some(), "cross-process recovery must find checkpoint");
        assert_eq!(cp.unwrap().task_iri, "iri://task/abc-123");

        // list must also find it
        let list = mgr2.list("iri://task/abc-123", 10);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "finish_DA");
    }

    #[test]
    fn restore_task_uses_versioned_state_and_skips_a_corrupt_newer_checkpoint() {
        use crate::memory::l0_store::L0Store;
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let manager = CheckpointManager::with_persistence(l0.clone());
        register_test_contract(&manager, "iri://task/resume-fallback");
        let checkpoint = manager
            .create_ext(
                "iri://task/resume-fallback",
                "turn_Do_7",
                "[]",
                r#"[{"role":"system","content":"s"},{"role":"assistant","content":"done"}]"#,
                &active_state(
                    r#"{"turn":7,"tc":3}"#,
                    r#"[{"role":"system","content":"s"},{"role":"assistant","content":"done"}]"#,
                    crate::core::agent_instance::AgentRole::Do,
                ),
                &["Do".to_string()],
                Some("DA"),
                None,
                Some("plan summary"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

        let mut corrupt = checkpoint.clone();
        corrupt.checkpoint_iri = "iri://checkpoint/task/resume-fallback/corrupt".to_string();
        corrupt.created_at += chrono::Duration::seconds(1);
        corrupt.session_messages_json = "{not valid json".to_string();
        l0.store(
            &corrupt.checkpoint_iri,
            &serde_json::to_string(&corrupt).unwrap(),
        )
        .unwrap();

        let restored = CheckpointManager::with_persistence(l0)
            .restore_task("iri://task/resume-fallback")
            .unwrap()
            .unwrap();
        assert_eq!(
            restored.checkpoint.checkpoint_iri,
            checkpoint.checkpoint_iri
        );
        assert_eq!(
            restored.state.schema_version,
            TASK_RESUME_STATE_SCHEMA_VERSION
        );
        assert_eq!(restored.state.task_cumulative.turn_count, 0);
        assert_eq!(restored.state.task_cumulative.tool_call_count, 0);
        assert_eq!(restored.state.observed_turn_count(), 7);
        assert_eq!(restored.state.observed_tool_call_count(), 3);
        assert_eq!(restored.state.current_role.as_deref(), Some("DA"));
        assert_eq!(restored.state.prev_summary.as_deref(), Some("plan summary"));
    }

    #[test]
    fn short_lived_manager_does_not_hide_an_older_valid_checkpoint() {
        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let task_iri = "iri://task/manager-index-fallback";
        let first = CheckpointManager::with_persistence(l0.clone());
        register_test_contract(&first, task_iri);
        let valid = first
            .create(
                task_iri,
                "start_Plan",
                "[]",
                "[]",
                &active_state(
                    r#"{"turn":1}"#,
                    "[]",
                    crate::core::agent_instance::AgentRole::Plan,
                ),
                &[],
            )
            .unwrap();

        // A fresh manager knows only about the checkpoint it creates locally,
        // while L0 also contains the older boundary. Corrupting the newer
        // record simulates an interrupted write discovered during recovery.
        let second = CheckpointManager::with_persistence(l0.clone());
        let mut newer = second
            .create(
                task_iri,
                "turn_Plan_5",
                "[]",
                "[]",
                &active_state(
                    r#"{"turn":5}"#,
                    "[]",
                    crate::core::agent_instance::AgentRole::Plan,
                ),
                &[],
            )
            .unwrap();
        newer.created_at += chrono::Duration::seconds(1);
        newer.session_messages_json = "{invalid".to_string();
        l0.store(
            &newer.checkpoint_iri,
            &serde_json::to_string(&newer).unwrap(),
        )
        .unwrap();

        let restored = second.restore_task(task_iri).unwrap().unwrap();
        assert_eq!(restored.checkpoint.checkpoint_iri, valid.checkpoint_iri);
    }

    #[test]
    fn checkpoint_without_canonical_resume_state_is_rejected() {
        let manager = CheckpointManager::new();
        register_test_contract(&manager, "iri://task/no-legacy-resume");
        let checkpoint = manager
            .create(
                "iri://task/no-legacy-resume",
                "turn_Do_1",
                "[]",
                "[]",
                &active_state(
                    r#"{"turn":1}"#,
                    "[]",
                    crate::core::agent_instance::AgentRole::Do,
                ),
                &[],
            )
            .unwrap();
        let mut encoded = serde_json::to_value(checkpoint).unwrap();
        encoded.as_object_mut().unwrap().remove("resume_state");

        let error = CheckpointManager::deserialize_checkpoint(&encoded.to_string()).unwrap_err();
        assert!(error.to_string().contains("canonical resume state"));
    }

    #[test]
    fn unsupported_resume_schema_is_rejected_without_migration() {
        let manager = CheckpointManager::new();
        register_test_contract(&manager, "iri://task/old-schema");
        let mut checkpoint = manager
            .create(
                "iri://task/old-schema",
                "turn_Do_1",
                "[]",
                "[]",
                &active_state(
                    r#"{"turn":1}"#,
                    "[]",
                    crate::core::agent_instance::AgentRole::Do,
                ),
                &[],
            )
            .unwrap();
        checkpoint.resume_state.as_mut().unwrap().schema_version =
            TASK_RESUME_STATE_SCHEMA_VERSION - 1;

        let error = checkpoint.to_restored_task().unwrap_err();
        assert!(error
            .to_string()
            .contains("Unsupported task resume schema version"));
    }

    #[test]
    fn unsupported_resume_contract_schema_is_rejected_without_migration() {
        let mut contract = test_resume_contract("old contract");
        contract.schema_version = TASK_RESUME_CONTRACT_SCHEMA_VERSION - 1;

        let error = contract.validate().unwrap_err();
        assert!(error
            .to_string()
            .contains("Unsupported task resume contract schema version"));
    }

    #[test]
    fn resume_contract_rejects_role_sequence_and_fallback_drift() {
        let mut sequence_drift = test_resume_contract("sequence drift");
        sequence_drift.execution_plan.agent_sequence.swap(0, 1);
        sequence_drift.fingerprint = sequence_drift.compute_fingerprint().unwrap();
        assert!(sequence_drift
            .validate()
            .unwrap_err()
            .to_string()
            .contains("agent_sequence disagrees"));

        let mut dormant_fallback = test_resume_contract("dormant fallback");
        dormant_fallback.execution_plan.fallback_steps =
            dormant_fallback.execution_plan.steps.clone();
        dormant_fallback.fingerprint = dormant_fallback.compute_fingerprint().unwrap();
        assert!(dormant_fallback
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not carry dormant fallback"));
    }

    #[test]
    fn resume_contract_rejects_pre_typed_work_packages() {
        let mut contract = test_resume_contract("resume typed work packages");
        let do_step = contract
            .execution_plan
            .steps
            .iter_mut()
            .find(|step| step.role == crate::core::agent_instance::AgentRole::Do)
            .expect("test plan contains a Do step");
        do_step
            .work_packages
            .push(crate::core::sa::PlanWorkPackage {
                id: "legacy-delivery".to_string(),
                objective: "deliver a file".to_string(),
                expected_output: "project/output.py".to_string(),
                success_criteria: "the file exists".to_string(),
                evidence_requirements: Vec::new(),
                dependencies: Vec::new(),
            });

        let error = contract.validate().unwrap_err();
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("invalid typed evidence contract"));
        assert!(diagnostic.contains("has no typed evidence requirements"));
    }

    #[test]
    fn resume_contract_revalidates_downstream_check_test_evidence_barrier() {
        let original = "write calculator tests, then run them in the final workspace";
        let mut plan = test_resume_contract(original).execution_plan;
        let do_step = plan
            .steps
            .iter_mut()
            .find(|step| step.role == crate::core::agent_instance::AgentRole::Do)
            .expect("test plan contains a Do step");
        do_step
            .work_packages
            .push(crate::core::sa::PlanWorkPackage {
                id: "write_tests".to_string(),
                objective: "write pytest test cases".to_string(),
                expected_output: "calculator_project/test_calculator.py".to_string(),
                success_criteria: "the pytest source artifact is complete".to_string(),
                evidence_requirements: vec![
                    crate::core::sa::WorkPackageEvidenceRequirement::ArtifactDelivery {
                        paths: vec!["calculator_project/test_calculator.py".to_string()],
                        min_paths: 1,
                    },
                ],
                dependencies: Vec::new(),
            });
        let check_step = plan
            .steps
            .iter_mut()
            .find(|step| step.role == crate::core::agent_instance::AgentRole::Check)
            .expect("test plan contains a Check step");
        check_step
            .work_packages
            .push(crate::core::sa::PlanWorkPackage {
                id: "run_final_tests".to_string(),
                objective: "run the delivered pytest suite in the final workspace".to_string(),
                expected_output: "passing pytest receipt".to_string(),
                success_criteria: "pytest exits successfully".to_string(),
                evidence_requirements: vec![
                    crate::core::sa::WorkPackageEvidenceRequirement::Verification {
                        kind: crate::core::tracked_action::VerificationKind::TestExecution,
                        min_count: 1,
                    },
                    crate::core::sa::WorkPackageEvidenceRequirement::TestArtifactExecutionScope {
                        paths: vec!["calculator_project/test_calculator.py".to_string()],
                    },
                ],
                // Stored plans contain child-local edges only. The existing
                // Check -> Do parent dependency is the cross-role barrier.
                dependencies: Vec::new(),
            });

        TaskResumeContract::new(
            original,
            &HashMap::new(),
            crate::core::effect::EffectPolicy::None,
            plan.clone(),
        )
        .expect("the normalized downstream Check relation must restore");

        plan.steps
            .iter_mut()
            .find(|step| step.role == crate::core::agent_instance::AgentRole::Check)
            .unwrap()
            .dependencies
            .clear();
        let error = TaskResumeContract::new(
            original,
            &HashMap::new(),
            crate::core::effect::EffectPolicy::None,
            plan,
        )
        .expect_err("recovery must not detach the verifier from its mutation epoch");
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("normalized two-level work-package contract"));
        assert!(diagnostic.contains("test execution scope"));
    }

    #[test]
    fn non_workflow_resume_rejects_kernel_generic_agent_definitions() {
        let mut contract = test_resume_contract("resume isolated agents");
        contract.execution_plan.agent_spec_provenance =
            Some(crate::core::context_model::ExecutionPlanProvenance::new(
                crate::core::context_model::AgentSpecSourceRecord::new(
                    crate::core::context_model::AgentSpecSourceKind::KernelGeneratedPlan,
                )
                .with_source_ref("legacy-kernel-plan")
                .with_producer("SupervisorAgent.structural_plan"),
            ));

        let error = contract.validate().unwrap_err();
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("refuses KernelGeneratedPlan"));
        assert!(diagnostic.contains("agent.md provenance"));
    }

    #[test]
    fn test_checkpoint_retention_prunes_oldest() {
        use crate::memory::l0_store::L0Store;
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let mgr = CheckpointManager::with_persistence(l0.clone());
        register_test_contract(&mgr, "iri://task/retention");

        // Create enough checkpoints to exceed the per-task retention cap
        let total = MAX_CHECKPOINTS_PER_TASK + 5;
        for i in 0..total {
            mgr.create(
                "iri://task/retention",
                &format!("turn_Plan_{}", i),
                "[]",
                r#"[{"role":"user","content":"hello"}]"#,
                &active_state(
                    r#"{"turn":1}"#,
                    r#"[{"role":"user","content":"hello"}]"#,
                    crate::core::agent_instance::AgentRole::Plan,
                ),
                &["Plan".to_string()],
            )
            .unwrap();
        }

        // In-memory index is bounded
        assert_eq!(mgr.checkpoint_count() as usize, MAX_CHECKPOINTS_PER_TASK);

        // Oldest entry physically evicted from L0
        let persisted = l0
            .scan_iri_prefix("iri://checkpoint/task/retention/", 100)
            .unwrap();
        assert_eq!(persisted.len(), MAX_CHECKPOINTS_PER_TASK);
        assert!(persisted.iter().all(|entry| !entry.iri.contains("/seq_0_")));

        // Latest entry still present and restorable.
        let cp = mgr.restore_latest("iri://task/retention").unwrap();
        assert_eq!(cp.unwrap().name, format!("turn_Plan_{}", total - 1));
    }

    #[test]
    fn separate_managers_do_not_overwrite_the_same_task_checkpoint() {
        use crate::memory::l0_store::L0Store;
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let first_manager = CheckpointManager::with_persistence(l0.clone());
        register_test_contract(&first_manager, "iri://task/shared");
        let first = first_manager
            .create(
                "iri://task/shared",
                "finish_DA",
                "[]",
                "[]",
                &active_state("{}", "[]", crate::core::agent_instance::AgentRole::Do),
                &[],
            )
            .unwrap();
        let second = CheckpointManager::with_persistence(l0.clone())
            .create(
                "iri://task/shared",
                "finish_CA",
                "[]",
                "[]",
                &active_state("{}", "[]", crate::core::agent_instance::AgentRole::Check),
                &[],
            )
            .unwrap();

        assert_ne!(first.checkpoint_iri, second.checkpoint_iri);
        assert_eq!(
            l0.scan_iri_prefix("iri://checkpoint/task/shared/", 10)
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn cross_process_restore_preserves_authoritative_contract_and_skips_biz_kind() {
        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let task_iri = "iri://task/canonical-contract";
        let original = "在新目录 calculator_project 中实现并测试 Python 计算器";
        let mut constraints = HashMap::new();
        constraints.insert(
            crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT.to_string(),
            crate::core::agent_runner::DELIVERY_MODE_WORKSPACE_ARTIFACT.to_string(),
        );
        constraints.insert(
            crate::core::agent_runner::DELIVERY_TARGET_PATH_CONSTRAINT.to_string(),
            "calculator_project/README.md".to_string(),
        );
        constraints.insert(
            "required_effect".to_string(),
            "workspace_mutation".to_string(),
        );
        constraints.insert(
            "effect_policy".to_string(),
            "required_workspace_mutation".to_string(),
        );
        let mut plan = test_resume_contract(original).execution_plan;
        plan.plan_id = "canonical-plan-revision-7".to_string();
        plan.agent_spec_provenance =
            Some(crate::core::context_model::ExecutionPlanProvenance::new(
                crate::core::context_model::AgentSpecSourceRecord::new(
                    crate::core::context_model::AgentSpecSourceKind::LlmGeneratedPlan,
                )
                .with_source_ref("canonical-plan-revision-7")
                .with_producer("SupervisorAgent.plan_generation")
                .with_model("checkpoint-test-model")
                .with_interaction_id("checkpoint-canonical-plan-interaction"),
            ));
        let contract = TaskResumeContract::new(
            original,
            &constraints,
            crate::core::effect::EffectPolicy::required_workspace_mutation(),
            plan,
        )
        .unwrap();

        let writer = CheckpointManager::with_persistence(l0.clone());
        writer
            .register_task_contract(task_iri, contract.clone())
            .unwrap();
        let runtime = writer
            .create(
                task_iri,
                "finish_Do",
                "[]",
                r#"[{"role":"assistant","content":"implemented"}]"#,
                &active_state(
                    r#"{"turn":8,"tc":4}"#,
                    r#"[{"role":"assistant","content":"implemented"}]"#,
                    crate::core::agent_instance::AgentRole::Do,
                ),
                &["Do".to_string()],
            )
            .unwrap();
        let biz = writer
            .create_ext_with_kind(
                CheckpointKind::BizOrchestration,
                task_iri,
                "biz_orchestration_Do_wave_2_completed",
                "[]",
                "[]",
                r#"{"schema_version":1}"#,
                &["biz-key:test".to_string()],
                Some("Do"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(biz.kind, CheckpointKind::BizOrchestration);
        assert!(biz.resume_state.is_none());

        let reader = CheckpointManager::with_persistence(l0);
        let restored = reader.restore_task(task_iri).unwrap().unwrap();
        assert_eq!(restored.checkpoint.checkpoint_iri, runtime.checkpoint_iri);
        assert_eq!(restored.state.checkpoint_iri, runtime.checkpoint_iri);
        assert_eq!(restored.checkpoint.kind, CheckpointKind::TaskRuntime);
        assert_eq!(restored.state.contract.original_user_task, original);
        assert_eq!(restored.state.contract.constraints, contract.constraints);
        assert_eq!(
            restored.state.contract.effect_policy,
            crate::core::effect::EffectPolicy::required_workspace_mutation()
        );
        assert_eq!(
            restored.state.contract.delivery_target.as_deref(),
            Some("calculator_project/README.md")
        );
        assert_eq!(
            restored.state.contract.execution_plan.plan_id,
            "canonical-plan-revision-7"
        );
        assert!(restored
            .state
            .contract
            .execution_plan
            .agent_spec_provenance
            .as_ref()
            .unwrap()
            .validate()
            .is_ok());
        assert_eq!(
            restored
                .state
                .contract
                .execution_plan
                .steps
                .iter()
                .map(|step| step.role)
                .collect::<Vec<_>>(),
            vec![
                crate::core::agent_instance::AgentRole::Plan,
                crate::core::agent_instance::AgentRole::Do,
                crate::core::agent_instance::AgentRole::Check,
                crate::core::agent_instance::AgentRole::Act,
            ]
        );
        assert!(restored
            .state
            .contract
            .execution_plan
            .steps
            .iter()
            .all(|step| step.objective == original));
        assert_eq!(
            reader
                .list_by_kind(task_iri, CheckpointKind::BizOrchestration, 10)
                .len(),
            1
        );
    }

    #[test]
    fn old_checkpoint_without_kind_is_rejected_instead_of_guessed() {
        let manager = CheckpointManager::new();
        register_test_contract(&manager, "iri://task/no-kind");
        let checkpoint = manager
            .create(
                "iri://task/no-kind",
                "start_Plan",
                "[]",
                "[]",
                &active_state("{}", "[]", crate::core::agent_instance::AgentRole::Plan),
                &[],
            )
            .unwrap();
        let mut encoded = serde_json::to_value(checkpoint).unwrap();
        encoded.as_object_mut().unwrap().remove("kind");
        assert!(CheckpointManager::deserialize_checkpoint(&encoded.to_string()).is_err());
    }

    #[test]
    fn restore_rejects_checkpoint_stored_under_another_tasks_prefix() {
        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let source_task = "iri://task/source-authority";
        let requested_task = "iri://task/requested-authority";
        let writer = CheckpointManager::with_persistence(l0.clone());
        register_test_contract(&writer, source_task);
        let mut forged = writer
            .create(
                source_task,
                "start_Plan",
                "[]",
                "[]",
                &active_state("{}", "[]", crate::core::agent_instance::AgentRole::Plan),
                &[],
            )
            .unwrap();
        forged.checkpoint_iri =
            "iri://checkpoint/task/requested-authority/task_runtime/forged".to_string();
        forged
            .resume_state
            .as_mut()
            .unwrap()
            .checkpoint_iri
            .clone_from(&forged.checkpoint_iri);
        l0.store(
            &forged.checkpoint_iri,
            &serde_json::to_string(&forged).unwrap(),
        )
        .unwrap();

        assert!(CheckpointManager::with_persistence(l0)
            .restore_task(requested_task)
            .unwrap()
            .is_none());
    }

    #[test]
    fn task_runtime_checkpoint_fails_closed_without_registered_contract() {
        let error = CheckpointManager::new()
            .create(
                "iri://task/missing-contract",
                "start_Plan",
                "[]",
                "[]",
                "{}",
                &[],
            )
            .unwrap_err();
        assert!(error.to_string().contains("canonical resume contract"));
    }

    #[test]
    fn task_resume_contract_rejects_split_brain_effect_authority() {
        let constraints =
            HashMap::from([("effect_policy".to_string(), "evidence_only".to_string())]);
        let error = TaskResumeContract::new(
            "write output",
            &constraints,
            crate::core::effect::EffectPolicy::required_workspace_mutation(),
            test_resume_contract("write output").execution_plan,
        )
        .unwrap_err();
        assert!(error.to_string().contains("typed effect policy"));
    }

    #[test]
    fn mutation_receipt_requires_sa_handoff_and_survives_cross_process_restore() {
        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let task_iri = "iri://task/receipt-boundary";
        let original = "create calculator_project/README.md";
        let mut constraints = HashMap::new();
        constraints.insert(
            "required_effect".to_string(),
            "workspace_mutation".to_string(),
        );
        let contract = TaskResumeContract::new(
            original,
            &constraints,
            crate::core::effect::EffectPolicy::required_workspace_mutation(),
            test_resume_contract(original).execution_plan,
        )
        .unwrap();
        let writer = CheckpointManager::with_persistence(l0.clone());
        writer.register_task_contract(task_iri, contract).unwrap();
        let action = crate::core::tracked_action::TrackedAction {
            action_id: "write-receipt-1".to_string(),
            call_identity: None,
            tool_name: "file_write".to_string(),
            agent_role: "Do".to_string(),
            duration_secs: 0.1,
            status: crate::core::tracked_action::ActionStatus::Success,
            files_created: vec![crate::core::tracked_action::FileChange {
                path: "calculator_project/README.md".to_string(),
                size_bytes: Some(100),
                hash: None,
            }],
            files_modified: Vec::new(),
            files_removed: Vec::new(),
            directories_created: Vec::new(),
            directories_removed: Vec::new(),
            workspace_delta_complete: true,
            workspace_delta_sha256: None,
            workspace_delta_contaminated: false,
            files_read: Vec::new(),
            error: None,
            substantive_effect: true,
            verification_attempted: false,
            successful_verification: false,
            tool_args: HashMap::new(),
            disclosure: None,
        };
        let mut unaccepted_parallel_action = action.clone();
        unaccepted_parallel_action.action_id = "parallel-child-unaccepted".to_string();
        let unaccepted_actions =
            serde_json::to_string(&vec![unaccepted_parallel_action.clone()]).unwrap();
        let committed_actions = serde_json::to_string(&vec![action.clone()]).unwrap();

        writer
            .create_ext(
                task_iri,
                "finish_Do",
                "[]",
                "[]",
                &active_state(
                    r#"{"turn":2,"tc":1}"#,
                    "[]",
                    crate::core::agent_instance::AgentRole::Do,
                ),
                &["Do".to_string()],
                Some("DA"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&unaccepted_actions),
                None,
            )
            .unwrap();
        assert!(CheckpointManager::with_persistence(l0.clone())
            .restore_task(task_iri)
            .unwrap()
            .is_none());

        let completed = HashMap::from([(
            "wf:test-resume-plan/test-DA".to_string(),
            crate::core::workflow::NodeResult {
                node_id: "wf:test-resume-plan/test-DA".to_string(),
                status: "success".to_string(),
                summary: "workspace artifact created".to_string(),
                archive_iri: None,
                turn_count: 2,
                tool_call_count: 1,
                error: None,
                output: Some(serde_json::json!("created")),
                artifacts: Vec::new(),
            },
        )]);
        let dag_state =
            encode_dag_resume_state(&completed, &std::collections::HashSet::new()).unwrap();
        writer
            .create_ext(
                task_iri,
                "step_complete_Do",
                "[]",
                "[]",
                r#"{"turn":2,"tc":1}"#,
                &["Do".to_string(), "step_complete".to_string()],
                Some("Do"),
                None,
                Some("workspace artifact created"),
                None,
                Some(&dag_state),
                None,
                None,
                None,
                Some(&committed_actions),
                None,
            )
            .unwrap();
        let restored = CheckpointManager::with_persistence(l0)
            .restore_task(task_iri)
            .unwrap()
            .unwrap();
        assert_eq!(restored.state.tracked_actions.len(), 1);
        assert_eq!(
            restored.state.tracked_actions[0].action_id,
            action.action_id
        );
        assert!(!restored
            .state
            .tracked_actions
            .iter()
            .any(|receipt| receipt.action_id == unaccepted_parallel_action.action_id));
        assert!(restored.state.tracked_actions[0].substantive_effect);
        assert!(restored
            .state
            .committed_action_ids
            .contains(&action.action_id));
        assert!(restored
            .state
            .completed_nodes
            .contains_key("wf:test-resume-plan/test-DA"));
    }

    #[test]
    fn parallel_active_node_transcripts_are_receipt_bound_and_never_crossed() {
        let dir = tempfile::TempDir::new().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let task_iri = "iri://task/parallel-continuations";
        let manager = CheckpointManager::with_persistence(l0);
        register_test_contract(&manager, task_iri);

        let messages_a = r#"[{"role":"assistant","content":"A_ONLY_MARKER"}]"#;
        let messages_b = r#"[{"role":"assistant","content":"B_ONLY_MARKER"}]"#;
        let identity = |marker: &str| ActiveNodeIdentity {
            step_id: format!("parallel-{marker}"),
            dispatch_id: format!("dispatch-{marker}"),
            agent_id: format!("agent-{marker}"),
            l1_session_id: format!("l1-{marker}"),
            role: crate::core::agent_instance::AgentRole::Do,
            agent_md_sha256: sha256_receipt(&format!("agent-md-{marker}")),
            context_manifest_sha256: sha256_receipt(&format!("context-{marker}")),
            source_interaction_id: Some(format!("llm-{marker}")),
        };
        let checkpoint_a = manager
            .create(
                task_iri,
                "turn_DA_2",
                "[]",
                messages_a,
                &encode_active_node_agent_state(r#"{"turn":2,"tc":1}"#, messages_a, &identity("A"))
                    .unwrap(),
                &[],
            )
            .unwrap();
        let checkpoint_b = manager
            .create(
                task_iri,
                "turn_DA_3",
                "[]",
                messages_b,
                &encode_active_node_agent_state(r#"{"turn":3,"tc":2}"#, messages_b, &identity("B"))
                    .unwrap(),
                &[],
            )
            .unwrap();

        let restored_b = checkpoint_b.to_restored_task().unwrap();
        assert_eq!(
            restored_b
                .state
                .active_continuation
                .as_ref()
                .unwrap()
                .agent_id,
            "agent-B"
        );
        assert!(restored_b.messages[0].content.contains("B_ONLY_MARKER"));
        assert!(!restored_b.messages[0].content.contains("A_ONLY_MARKER"));
        assert!(checkpoint_a.to_restored_task().is_ok());

        let mut crossed = checkpoint_b;
        crossed.session_messages_json = messages_a.to_string();
        let error = crossed.to_restored_task().unwrap_err();
        assert!(error.to_string().contains("transcript hash mismatch"));
    }

    #[test]
    fn sa_step_complete_boundary_has_no_agent_transcript() {
        let manager = CheckpointManager::new();
        let task_iri = "iri://task/typed-sa-boundary";
        register_test_contract(&manager, task_iri);
        let dag_state =
            encode_dag_resume_state(&HashMap::new(), &std::collections::HashSet::new()).unwrap();
        let checkpoint = manager
            .create_ext(
                task_iri,
                "step_complete_DA",
                "[]",
                "[]",
                r#"{"turn":5,"tc":2}"#,
                &[],
                Some("DA"),
                None,
                None,
                None,
                Some(&dag_state),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let restored = checkpoint.to_restored_task().unwrap();
        assert!(restored.messages.is_empty());
        assert!(restored.state.active_continuation.is_none());
        assert_eq!(restored.state.task_cumulative.turn_count, 5);
        assert_eq!(restored.state.task_cumulative.tool_call_count, 2);

        let error = manager
            .create_ext(
                task_iri,
                "step_complete_CA",
                "[]",
                r#"[{"role":"assistant","content":"FOREIGN_AGENT_MARKER"}]"#,
                r#"{"turn":6,"tc":2}"#,
                &[],
                Some("CA"),
                None,
                None,
                None,
                Some(&dag_state),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("must not carry a mixed Agent transcript"));
    }

    #[test]
    fn test_parse_checkpoint_phase() {
        assert_eq!(parse_checkpoint_phase("start_DA"), "start_DA");
        assert_eq!(parse_checkpoint_phase("turn_DA_5"), "turn_DA");
        assert_eq!(parse_checkpoint_phase("finish_CA"), "finish_CA");
        assert_eq!(
            parse_checkpoint_phase("step_complete_Do"),
            "step_complete_Do"
        );
        assert_eq!(parse_checkpoint_phase("max_turns_Plan"), "max_turns_Plan");
        assert_eq!(parse_checkpoint_phase("force_end_Act"), "force_end_Act");
        assert_eq!(parse_checkpoint_phase("unknown_xxx"), "unknown");
    }

    #[test]
    fn test_create_ext_roundtrip() {
        let manager = CheckpointManager::new();
        register_test_contract(&manager, "iri://task/roundtrip");
        let completed_nodes = HashMap::from([(
            "wf:test-resume-plan/test-DA".to_string(),
            crate::core::workflow::NodeResult {
                node_id: "wf:test-resume-plan/test-DA".to_string(),
                status: "success".to_string(),
                summary: "completed".to_string(),
                archive_iri: None,
                turn_count: 1,
                tool_call_count: 0,
                error: None,
                output: None,
                artifacts: Vec::new(),
            },
        )]);
        let skipped_nodes =
            std::collections::HashSet::from(["wf:test-resume-plan/test-CA".to_string()]);
        let completed_nodes_json =
            encode_dag_resume_state(&completed_nodes, &skipped_nodes).unwrap();
        let cp = manager
            .create_ext(
                "iri://task/roundtrip",
                "step_complete_DA",
                "[]",
                "[]",
                r#"{"turn":5}"#,
                &["DA".to_string(), "step_complete".to_string()],
                Some("DA"),
                Some(r#"{"what":"test"}"#),
                Some("prev summary here"),
                Some(r#"{"phase":"Executing"}"#),
                Some(&completed_nodes_json),
                Some(r#"{"approval1":true}"#),
                None,
                Some(r#"{"bash":3}"#),
                Some(r#"[]"#),
                None,
            )
            .unwrap();

        assert_eq!(cp.name, "step_complete_DA");
        assert_eq!(cp.current_role.as_deref(), Some("DA"));
        assert_eq!(cp.prev_summary.as_deref(), Some("prev summary here"));
        assert_eq!(cp.tool_error_json.as_deref(), Some(r#"{"bash":3}"#));
        assert!(cp
            .resume_state
            .as_ref()
            .unwrap()
            .skipped_nodes
            .contains("wf:test-resume-plan/test-CA"));
    }
}
