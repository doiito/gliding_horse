//! Durable, privacy-preserving task execution journal.
//!
//! The event bus is intentionally ephemeral and rich enough for local UI
//! rendering.  This journal is its durable counterpart: by default it keeps
//! correlation IDs, timing, sizes and SHA-256 digests, but never the original
//! LLM prompt/response or a tool's arguments/result.  Operators can opt in to
//! payload capture only when a controlled debugging environment requires it.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::memory::l0_store::{
    L0Store, TaskEvidenceAppendOutcome, TaskEvidenceFrameRecord, TaskEvidenceSealOutcome,
    TaskEvidenceSealRecord,
};
use crate::CoreError;

pub const TASK_EXECUTION_JOURNAL_SCHEMA_VERSION: u32 = 4;
pub const TASK_EVIDENCE_SCHEMA_VERSION: u32 = 1;
const MAX_APPEND_RETRIES: usize = 16;
const MAX_EVIDENCE_FRAMES: usize = 100_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PayloadReference {
    pub sha256: String,
    pub bytes: usize,
    /// Present only when an operator explicitly constructs the journal with
    /// payload capture enabled.  The production default is `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub captured: Option<String>,
}

/// Durable identity of one provider-issued tool call.
///
/// Provider call IDs are correlation tokens within one model request, not
/// globally unique operation IDs.  In particular, different isolated agent
/// sessions routinely reuse values such as `call_0`.  The execution journal
/// is task-wide, so every component that establishes the model/agent/session
/// boundary is part of the durable identity used for replay safety.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ToolCallIdentity {
    pub agent_id: String,
    pub l1_session_id: String,
    pub llm_request_id: String,
    /// Raw provider correlation ID. This remains unchanged for protocol
    /// pairing and observability, but must never be used as a task-global key.
    pub provider_call_id: String,
}

impl ToolCallIdentity {
    pub fn new(
        agent_id: impl Into<String>,
        l1_session_id: impl Into<String>,
        llm_request_id: impl Into<String>,
        provider_call_id: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            l1_session_id: l1_session_id.into(),
            llm_request_id: llm_request_id.into(),
            provider_call_id: provider_call_id.into(),
        }
    }
}

/// A tool operation as assessed at a resume boundary. Keeping the tool name
/// bound to the full identity prevents a mismatched finish record from
/// cancelling a different operation even if a provider reused its call ID.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JournalToolCall {
    pub identity: ToolCallIdentity,
    pub tool_name: String,
}

impl JournalToolCall {
    pub fn new(identity: ToolCallIdentity, tool_name: impl Into<String>) -> Self {
        Self {
            identity,
            tool_name: tool_name.into(),
        }
    }
}

impl PayloadReference {
    pub fn metadata_only(payload: &str) -> Self {
        Self {
            sha256: payload_sha256(payload),
            bytes: payload.len(),
            captured: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskExecutionJournalKind {
    LlmRequestPrepared {
        request_id: String,
        role: String,
        turn: u32,
        model: String,
        message_count: usize,
        advertised_tool_names: Vec<String>,
        /// Hash of the metadata-only typed role-context manifest.
        #[serde(default)]
        context_manifest_hash: Option<String>,
        request: PayloadReference,
    },
    LlmResponseReceived {
        request_id: String,
        provider_response_id: Option<String>,
        endpoint: String,
        attempts: u32,
        cache_hit: bool,
        latency_ms: u64,
        http_status: Option<u16>,
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
        response: PayloadReference,
    },
    LlmRequestFailed {
        request_id: String,
        latency_ms: u64,
        error_class: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        http_status: Option<u16>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retryable: Option<bool>,
    },
    ToolExecutionStarted {
        call_identity: ToolCallIdentity,
        tool_name: String,
        turn: u32,
        /// Conservative execution-time classification.
        side_effect_risk: bool,
        arguments: PayloadReference,
    },
    ToolExecutionFinished {
        call_identity: ToolCallIdentity,
        tool_name: String,
        success: bool,
        duration_ms: u64,
        result: PayloadReference,
    },
    CheckpointCommitted {
        checkpoint_iri: String,
        checkpoint_name: String,
    },
    WorkspaceMutationCommitted {
        call_identity: ToolCallIdentity,
        tool_name: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskExecutionJournalEvent {
    pub schema_version: u32,
    pub event_id: String,
    pub sequence: u64,
    pub task_iri: String,
    pub timestamp: DateTime<Utc>,
    pub event: TaskExecutionJournalKind,
}

/// Metadata-only decision used before resuming a ReAct transcript. A risky
/// call after the selected checkpoint may have partially or fully committed
/// even when its response was never checkpointed, so automatic replay must
/// stop for operator review.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResumeSafetyAssessment {
    pub safe_to_resume: bool,
    pub checkpoint_found: bool,
    /// Exact task-journal identities for all effectful calls after the
    /// selected checkpoint. Any entry makes automatic replay unsafe.
    pub risky_calls: Vec<JournalToolCall>,
    /// Exact identities for read-only starts without a matching finish. These
    /// may be retried because they cannot mutate the workspace.
    pub readonly_unfinished_calls: Vec<JournalToolCall>,
}

impl ResumeSafetyAssessment {
    pub fn risky_tool_names(&self) -> Vec<String> {
        let mut names = self
            .risky_calls
            .iter()
            .map(|call| call.tool_name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        names.sort();
        names
    }
}

/// Append-only durable task trace.  Sequence allocation resumes from persisted
/// records, so an interrupted task retains a single increasing timeline after
/// it is resumed in another process.
pub struct TaskExecutionJournal {
    l0: Arc<L0Store>,
    task_iri: String,
    task_key: String,
    capture_payloads: bool,
}

/// Result of a deterministic task-evidence verification. `valid == false`
/// means the durable chain or its L0 projection is incomplete/tampered; it is
/// intentionally data, not a panic, so operators can inspect all findings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskEvidenceVerification {
    pub task_iri: String,
    pub frame_count: u64,
    pub sealed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_status: Option<String>,
    pub root_hash: Option<String>,
    pub valid: bool,
    pub failures: Vec<String>,
}

impl TaskExecutionJournal {
    pub fn new(l0: Arc<L0Store>, task_iri: &str) -> Result<Self, CoreError> {
        Self::with_payload_capture(l0, task_iri, false)
    }

    /// Explicit opt-in constructor for tightly controlled debugging sessions.
    /// It must not be used as a product default because LLM and tool payloads
    /// may contain source code, user data or credentials.
    pub fn with_payload_capture(
        l0: Arc<L0Store>,
        task_iri: &str,
        capture_payloads: bool,
    ) -> Result<Self, CoreError> {
        let task_key = short_hash(task_iri);
        let journal = Self {
            l0,
            task_iri: task_iri.to_string(),
            task_key,
            capture_payloads,
        };
        journal.reconcile_projection()?;
        Ok(journal)
    }

    pub fn payload_reference(&self, payload: &str) -> PayloadReference {
        let mut reference = PayloadReference::metadata_only(payload);
        if self.capture_payloads {
            reference.captured = Some(payload.to_string());
        }
        reference
    }

    pub fn append(
        &self,
        event: TaskExecutionJournalKind,
    ) -> Result<TaskExecutionJournalEvent, CoreError> {
        for _ in 0..MAX_APPEND_RETRIES {
            let head = self.l0.task_evidence_head(&self.task_key)?;
            if head.sealed {
                return Err(CoreError::StorageError {
                    message: "Task evidence is sealed; append rejected".to_string(),
                });
            }
            let journal_event = TaskExecutionJournalEvent {
                schema_version: TASK_EXECUTION_JOURNAL_SCHEMA_VERSION,
                event_id: format!("trace_{}_{}", self.task_key, head.next_sequence),
                sequence: head.next_sequence,
                task_iri: self.task_iri.clone(),
                timestamp: Utc::now(),
                event: event.clone(),
            };
            let event_json =
                serde_json::to_string(&journal_event).map_err(|error| CoreError::StorageError {
                    message: format!("Failed to serialize task execution journal event: {error}"),
                })?;
            let event_iri = format!(
                "iri://task-journal/{}/seq_{:020}",
                self.task_key, journal_event.sequence
            );
            let event_hash = payload_sha256(&event_json);
            let created_at = Utc::now();
            let frame_hash = evidence_frame_hash(
                &self.task_key,
                &self.task_iri,
                journal_event.sequence,
                &event_iri,
                &event_hash,
                head.last_frame_hash.as_deref(),
                &event_json,
                created_at,
            )?;
            let frame = TaskEvidenceFrameRecord {
                schema_version: TASK_EVIDENCE_SCHEMA_VERSION,
                task_key: self.task_key.clone(),
                task_iri: self.task_iri.clone(),
                sequence: journal_event.sequence,
                event_iri: event_iri.clone(),
                event_hash,
                event_json: event_json.clone(),
                previous_frame_hash: head.last_frame_hash,
                frame_hash,
                created_at,
            };
            match self.l0.try_append_task_evidence(&frame)? {
                TaskEvidenceAppendOutcome::Appended => {
                    self.persist_projection(&frame)?;
                    return Ok(journal_event);
                }
                TaskEvidenceAppendOutcome::Conflict => continue,
                TaskEvidenceAppendOutcome::Sealed => {
                    return Err(CoreError::StorageError {
                        message: "Task evidence is sealed; append rejected".to_string(),
                    });
                }
            }
        }
        Err(CoreError::StorageError {
            message: "Task evidence append conflicted repeatedly; retry task operation".to_string(),
        })
    }

    pub fn events(&self, limit: usize) -> Result<Vec<TaskExecutionJournalEvent>, CoreError> {
        self.l0
            .task_evidence_frames(&self.task_key, limit.max(1))?
            .into_iter()
            .map(|frame| {
                let event = serde_json::from_str::<TaskExecutionJournalEvent>(&frame.event_json)
                    .map_err(|error| CoreError::StorageError {
                        message: format!("Failed to decode task journal evidence event: {error}"),
                    })?;
                if event.schema_version != TASK_EXECUTION_JOURNAL_SCHEMA_VERSION {
                    return Err(CoreError::StorageError {
                        message: format!(
                            "Task journal schema {} is incompatible with required schema {}",
                            event.schema_version, TASK_EXECUTION_JOURNAL_SCHEMA_VERSION
                        ),
                    });
                }
                Ok(event)
            })
            .collect()
    }

    /// Determine whether replay from the exact `checkpoint_iri` is
    /// side-effect safe. Human-readable names repeat across roles and PDCA
    /// cycles and therefore cannot identify a replay boundary.
    /// Only unfinished read-only operations may be retried automatically.
    /// Tool payloads are neither required nor returned by this audit.
    pub fn assess_resume_safety(
        &self,
        checkpoint_iri: Option<&str>,
    ) -> Result<ResumeSafetyAssessment, CoreError> {
        use std::collections::HashSet;

        let events = self.events(MAX_EVIDENCE_FRAMES)?;
        let checkpoint_index = checkpoint_iri
            .and_then(|expected| {
                events.iter().rposition(|event| {
                    matches!(
                        &event.event,
                        TaskExecutionJournalKind::CheckpointCommitted {
                            checkpoint_iri,
                            ..
                        } if checkpoint_iri == expected
                    )
                })
            })
            .or_else(|| {
                checkpoint_iri.is_none().then(|| {
                    events.iter().rposition(|event| {
                        matches!(
                            event.event,
                            TaskExecutionJournalKind::CheckpointCommitted { .. }
                        )
                    })
                })?
            });
        let checkpoint_found = checkpoint_index.is_some();
        let tail = checkpoint_index
            .map(|index| &events[index.saturating_add(1)..])
            .unwrap_or(events.as_slice());

        let mut risky_calls = HashSet::<JournalToolCall>::new();
        let mut readonly_started = HashSet::<JournalToolCall>::new();
        let mut finished = HashSet::<JournalToolCall>::new();
        for event in tail {
            match &event.event {
                TaskExecutionJournalKind::ToolExecutionStarted {
                    call_identity,
                    tool_name,
                    side_effect_risk,
                    ..
                } => {
                    let call = JournalToolCall::new(call_identity.clone(), tool_name.clone());
                    if *side_effect_risk {
                        risky_calls.insert(call);
                    } else {
                        readonly_started.insert(call);
                    }
                }
                TaskExecutionJournalKind::ToolExecutionFinished {
                    call_identity,
                    tool_name,
                    ..
                } => {
                    finished.insert(JournalToolCall::new(
                        call_identity.clone(),
                        tool_name.clone(),
                    ));
                }
                TaskExecutionJournalKind::WorkspaceMutationCommitted {
                    call_identity,
                    tool_name,
                } => {
                    risky_calls.insert(JournalToolCall::new(
                        call_identity.clone(),
                        tool_name.clone(),
                    ));
                }
                _ => {}
            }
        }

        let mut risky_calls = risky_calls.into_iter().collect::<Vec<_>>();
        let mut readonly_unfinished_calls = readonly_started
            .difference(&finished)
            .cloned()
            .collect::<Vec<_>>();
        risky_calls.sort();
        readonly_unfinished_calls.sort();

        // Any missing checkpoint receipt means we cannot establish the replay
        // boundary. Old snapshots without a matching v4 execution journal are
        // intentionally incompatible and must not be auto-resumed.
        let unknown_boundary = !checkpoint_found;
        Ok(ResumeSafetyAssessment {
            safe_to_resume: !unknown_boundary && risky_calls.is_empty(),
            checkpoint_found,
            risky_calls,
            readonly_unfinished_calls,
        })
    }

    /// Seal the task only after its terminal result is known. A seal is
    /// idempotent for a completed task and permanently rejects later appends.
    pub fn seal(&self, terminal_status: &str) -> Result<(), CoreError> {
        for _ in 0..MAX_APPEND_RETRIES {
            let head = self.l0.task_evidence_head(&self.task_key)?;
            if head.sealed {
                return Ok(());
            }
            let seal = TaskEvidenceSealRecord {
                schema_version: TASK_EVIDENCE_SCHEMA_VERSION,
                task_key: self.task_key.clone(),
                task_iri: self.task_iri.clone(),
                frame_count: head.next_sequence,
                root_hash: head.last_frame_hash,
                terminal_status: truncate_terminal_status(terminal_status),
                sealed_at: Utc::now(),
            };
            match self.l0.try_seal_task_evidence(&seal)? {
                TaskEvidenceSealOutcome::Sealed | TaskEvidenceSealOutcome::AlreadySealed => {
                    return Ok(())
                }
                TaskEvidenceSealOutcome::Conflict => continue,
            }
        }
        Err(CoreError::StorageError {
            message: "Task evidence seal conflicted repeatedly; retry finalization".to_string(),
        })
    }

    pub fn verify(&self) -> Result<TaskEvidenceVerification, CoreError> {
        let frames = self
            .l0
            .task_evidence_frames(&self.task_key, MAX_EVIDENCE_FRAMES)?;
        let seal = self.l0.task_evidence_seal(&self.task_key)?;
        let mut failures = Vec::new();
        let mut previous_hash = None;
        for (expected_sequence, frame) in frames.iter().enumerate() {
            if frame.schema_version != TASK_EVIDENCE_SCHEMA_VERSION
                || frame.task_key != self.task_key
                || frame.task_iri != self.task_iri
                || frame.sequence != expected_sequence as u64
            {
                failures.push(format!(
                    "invalid frame identity at sequence {expected_sequence}"
                ));
            }
            if frame.previous_frame_hash != previous_hash {
                failures.push(format!(
                    "previous hash mismatch at sequence {}",
                    frame.sequence
                ));
            }
            if payload_sha256(&frame.event_json) != frame.event_hash {
                failures.push(format!(
                    "event hash mismatch at sequence {}",
                    frame.sequence
                ));
            }
            let expected_hash = evidence_frame_hash(
                &frame.task_key,
                &frame.task_iri,
                frame.sequence,
                &frame.event_iri,
                &frame.event_hash,
                frame.previous_frame_hash.as_deref(),
                &frame.event_json,
                frame.created_at,
            )?;
            if expected_hash != frame.frame_hash {
                failures.push(format!(
                    "frame hash mismatch at sequence {}",
                    frame.sequence
                ));
            }
            match self.l0.retrieve(&frame.event_iri)? {
                Some(entry) if entry.content == frame.event_json => {}
                Some(_) => failures.push(format!(
                    "L0 projection differs at sequence {}",
                    frame.sequence
                )),
                None => failures.push(format!(
                    "L0 projection missing at sequence {}",
                    frame.sequence
                )),
            }
            previous_hash = Some(frame.frame_hash.clone());
        }
        let sealed = seal.is_some();
        if let Some(seal) = &seal {
            if seal.schema_version != TASK_EVIDENCE_SCHEMA_VERSION
                || seal.task_key != self.task_key
                || seal.task_iri != self.task_iri
                || seal.frame_count != frames.len() as u64
                || seal.root_hash != previous_hash
            {
                failures.push("terminal seal does not match evidence chain".to_string());
            }
        }
        Ok(TaskEvidenceVerification {
            task_iri: self.task_iri.clone(),
            frame_count: frames.len() as u64,
            sealed,
            terminal_status: seal.as_ref().map(|seal| seal.terminal_status.clone()),
            root_hash: previous_hash,
            valid: failures.is_empty(),
            failures,
        })
    }

    /// Open existing task evidence for explicit verification/finalization.
    pub fn open(l0: Arc<L0Store>, task_iri: &str) -> Result<Self, CoreError> {
        Self::new(l0, task_iri)
    }

    fn reconcile_projection(&self) -> Result<(), CoreError> {
        for frame in self
            .l0
            .task_evidence_frames(&self.task_key, MAX_EVIDENCE_FRAMES)?
        {
            self.persist_projection(&frame)?;
        }
        Ok(())
    }

    fn persist_projection(&self, frame: &TaskEvidenceFrameRecord) -> Result<(), CoreError> {
        match self.l0.retrieve(&frame.event_iri)? {
            Some(entry) if entry.content == frame.event_json => Ok(()),
            Some(_) => Err(CoreError::StorageError {
                message: format!(
                    "Task evidence projection collision at {}; refusing overwrite",
                    frame.event_iri
                ),
            }),
            None => self.l0.store(&frame.event_iri, &frame.event_json),
        }
    }
}

pub fn payload_sha256(payload: &str) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(payload.as_bytes())))
}

fn short_hash(input: &str) -> String {
    hex::encode(&Sha256::digest(input.as_bytes())[..12])
}

fn evidence_frame_hash(
    task_key: &str,
    task_iri: &str,
    sequence: u64,
    event_iri: &str,
    event_hash: &str,
    previous_frame_hash: Option<&str>,
    event_json: &str,
    created_at: DateTime<Utc>,
) -> Result<String, CoreError> {
    let material = serde_json::json!({
        "schema_version": TASK_EVIDENCE_SCHEMA_VERSION,
        "task_key": task_key,
        "task_iri": task_iri,
        "sequence": sequence,
        "event_iri": event_iri,
        "event_hash": event_hash,
        "previous_frame_hash": previous_frame_hash,
        "event_json": event_json,
        "created_at": created_at,
    });
    serde_json::to_string(&material)
        .map(|json| payload_sha256(&json))
        .map_err(|error| CoreError::StorageError {
            message: format!("Failed to serialize task evidence hash material: {error}"),
        })
}

fn truncate_terminal_status(status: &str) -> String {
    status.chars().take(128).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn journal() -> (tempfile::TempDir, Arc<L0Store>, TaskExecutionJournal) {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let journal = TaskExecutionJournal::new(l0.clone(), "iri://task/trace-test").unwrap();
        (dir, l0, journal)
    }

    #[test]
    fn default_journal_is_durable_ordered_and_does_not_store_payload() {
        let (_dir, l0, journal) = journal();
        let secret = "authorization=very-secret-token";
        let reference = journal.payload_reference(secret);
        assert_eq!(reference.bytes, secret.len());
        assert!(reference.captured.is_none());
        journal
            .append(TaskExecutionJournalKind::LlmRequestPrepared {
                request_id: "req-1".into(),
                role: "Do".into(),
                turn: 1,
                model: "model-a".into(),
                message_count: 2,
                advertised_tool_names: vec!["file_write".into()],
                context_manifest_hash: Some("sha256:context".into()),
                request: reference,
            })
            .unwrap();
        journal
            .append(TaskExecutionJournalKind::CheckpointCommitted {
                checkpoint_iri: "iri://checkpoint/task/1".into(),
                checkpoint_name: "turn_Do_1".into(),
            })
            .unwrap();

        let serialized = l0
            .scan_iri_prefix("iri://task-journal/", 10)
            .unwrap()
            .into_iter()
            .map(|entry| entry.content)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!serialized.contains(secret));
        assert!(serialized.contains("sha256:"));

        let reopened = TaskExecutionJournal::new(l0, "iri://task/trace-test").unwrap();
        let events = reopened.events(10).unwrap();
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let verification = reopened.verify().unwrap();
        assert!(verification.valid, "{:?}", verification.failures);
        assert!(!verification.sealed);
        assert_eq!(verification.terminal_status, None);
        reopened.seal("success").unwrap();
        let verification = reopened.verify().unwrap();
        assert!(verification.valid, "{:?}", verification.failures);
        assert!(verification.sealed);
        assert_eq!(verification.terminal_status.as_deref(), Some("success"));
        assert!(reopened
            .append(TaskExecutionJournalKind::CheckpointCommitted {
                checkpoint_iri: "iri://checkpoint/task/after-seal".into(),
                checkpoint_name: "after-seal".into(),
            })
            .is_err());
    }

    #[test]
    fn payload_capture_is_an_explicit_opt_in() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let journal =
            TaskExecutionJournal::with_payload_capture(l0, "iri://task/debug", true).unwrap();
        assert_eq!(
            journal
                .payload_reference("debug payload")
                .captured
                .as_deref(),
            Some("debug payload")
        );
    }

    fn append_checkpoint(journal: &TaskExecutionJournal, name: &str) {
        journal
            .append(TaskExecutionJournalKind::CheckpointCommitted {
                checkpoint_iri: format!("iri://checkpoint/{name}"),
                checkpoint_name: name.to_string(),
            })
            .unwrap();
    }

    fn append_tool_start(
        journal: &TaskExecutionJournal,
        call_identity: ToolCallIdentity,
        tool_name: &str,
        side_effect_risk: bool,
    ) {
        journal
            .append(TaskExecutionJournalKind::ToolExecutionStarted {
                call_identity,
                tool_name: tool_name.to_string(),
                turn: 1,
                side_effect_risk,
                arguments: PayloadReference::metadata_only("{}"),
            })
            .unwrap();
    }

    fn call_identity(
        agent_id: &str,
        l1_session_id: &str,
        llm_request_id: &str,
        provider_call_id: &str,
    ) -> ToolCallIdentity {
        ToolCallIdentity::new(agent_id, l1_session_id, llm_request_id, provider_call_id)
    }

    #[test]
    fn resume_safety_allows_only_readonly_tail_and_uses_named_boundary() {
        let (_dir, _l0, journal) = journal();
        let legacy_without_receipt = journal
            .assess_resume_safety(Some("iri://checkpoint/legacy-checkpoint"))
            .unwrap();
        assert!(!legacy_without_receipt.safe_to_resume);
        assert!(!legacy_without_receipt.checkpoint_found);

        append_checkpoint(&journal, "before-read");
        let read = call_identity("agent-da", "l1-da", "req-da-1", "read-1");
        append_tool_start(&journal, read.clone(), "file_read", false);

        let safe = journal
            .assess_resume_safety(Some("iri://checkpoint/before-read"))
            .unwrap();
        assert!(safe.safe_to_resume);
        assert!(safe.checkpoint_found);
        assert_eq!(
            safe.readonly_unfinished_calls,
            vec![JournalToolCall::new(read, "file_read")]
        );

        let unknown = journal
            .assess_resume_safety(Some("iri://checkpoint/missing-checkpoint"))
            .unwrap();
        assert!(!unknown.safe_to_resume);
        assert!(!unknown.checkpoint_found);
    }

    #[test]
    fn resume_safety_blocks_started_or_committed_effect_until_later_checkpoint() {
        let (_dir, _l0, journal) = journal();
        append_checkpoint(&journal, "before-write");
        let write = call_identity("agent-da", "l1-da", "req-da-1", "write-1");
        append_tool_start(&journal, write.clone(), "file_write", true);

        let started = journal
            .assess_resume_safety(Some("iri://checkpoint/before-write"))
            .unwrap();
        assert!(!started.safe_to_resume);
        assert_eq!(
            started.risky_calls,
            vec![JournalToolCall::new(write.clone(), "file_write")]
        );

        journal
            .append(TaskExecutionJournalKind::WorkspaceMutationCommitted {
                call_identity: write,
                tool_name: "file_write".to_string(),
            })
            .unwrap();
        let committed = journal
            .assess_resume_safety(Some("iri://checkpoint/before-write"))
            .unwrap();
        assert!(!committed.safe_to_resume);
        assert_eq!(committed.risky_tool_names(), vec!["file_write"]);

        append_checkpoint(&journal, "after-write");
        let checkpointed = journal
            .assess_resume_safety(Some("iri://checkpoint/after-write"))
            .unwrap();
        assert!(checkpointed.safe_to_resume);
    }

    #[test]
    fn resume_safety_distinguishes_repeated_checkpoint_names_by_exact_iri() {
        let (_dir, _l0, journal) = journal();
        journal
            .append(TaskExecutionJournalKind::CheckpointCommitted {
                checkpoint_iri: "iri://checkpoint/old-step".to_string(),
                checkpoint_name: "step_complete_Do".to_string(),
            })
            .unwrap();
        let write = call_identity("agent-da", "l1-da", "req-da-2", "write-between-boundaries");
        append_tool_start(&journal, write.clone(), "file_write", true);
        journal
            .append(TaskExecutionJournalKind::WorkspaceMutationCommitted {
                call_identity: write.clone(),
                tool_name: "file_write".to_string(),
            })
            .unwrap();
        journal
            .append(TaskExecutionJournalKind::CheckpointCommitted {
                checkpoint_iri: "iri://checkpoint/new-step".to_string(),
                checkpoint_name: "step_complete_Do".to_string(),
            })
            .unwrap();

        let old = journal
            .assess_resume_safety(Some("iri://checkpoint/old-step"))
            .unwrap();
        assert!(!old.safe_to_resume);
        assert_eq!(
            old.risky_calls,
            vec![JournalToolCall::new(write, "file_write")]
        );

        let new = journal
            .assess_resume_safety(Some("iri://checkpoint/new-step"))
            .unwrap();
        assert!(new.safe_to_resume);
    }

    #[test]
    fn finished_same_provider_call_id_in_another_session_cannot_cancel_unfinished_read() {
        let (_dir, _l0, journal) = journal();
        append_checkpoint(&journal, "before-parallel-reads");
        let pa_call = call_identity("agent-pa", "l1-pa", "req-pa-1", "call_0");
        let ca_call = call_identity("agent-ca", "l1-ca", "req-ca-1", "call_0");
        append_tool_start(&journal, pa_call.clone(), "file_read", false);
        append_tool_start(&journal, ca_call.clone(), "file_read", false);
        journal
            .append(TaskExecutionJournalKind::ToolExecutionFinished {
                call_identity: ca_call.clone(),
                tool_name: "file_read".to_string(),
                success: true,
                duration_ms: 2,
                result: PayloadReference::metadata_only("ok"),
            })
            .unwrap();

        let assessment = journal
            .assess_resume_safety(Some("iri://checkpoint/before-parallel-reads"))
            .unwrap();
        assert!(assessment.safe_to_resume);
        assert_eq!(
            assessment.readonly_unfinished_calls,
            vec![JournalToolCall::new(pa_call, "file_read")]
        );
        assert!(!assessment
            .readonly_unfinished_calls
            .iter()
            .any(|call| call.identity == ca_call));
    }

    #[test]
    fn risky_same_provider_call_id_is_bound_to_exact_agent_session_and_request() {
        let (_dir, _l0, journal) = journal();
        append_checkpoint(&journal, "before-colliding-calls");
        let da_write = call_identity("agent-da", "l1-da", "req-da-7", "call_0");
        let ca_read = call_identity("agent-ca", "l1-ca", "req-ca-4", "call_0");
        append_tool_start(&journal, da_write.clone(), "file_write", true);
        append_tool_start(&journal, ca_read.clone(), "file_read", false);
        journal
            .append(TaskExecutionJournalKind::ToolExecutionFinished {
                call_identity: ca_read.clone(),
                tool_name: "file_read".to_string(),
                success: true,
                duration_ms: 1,
                result: PayloadReference::metadata_only("ok"),
            })
            .unwrap();

        let assessment = journal
            .assess_resume_safety(Some("iri://checkpoint/before-colliding-calls"))
            .unwrap();
        assert!(!assessment.safe_to_resume);
        assert_eq!(
            assessment.risky_calls,
            vec![JournalToolCall::new(da_write, "file_write")]
        );
        assert!(assessment.readonly_unfinished_calls.is_empty());
        assert_ne!(
            assessment.risky_calls[0].identity.l1_session_id,
            ca_read.l1_session_id
        );
    }

    #[test]
    fn concurrent_journal_instances_allocate_one_unbroken_chain() {
        let dir = tempfile::tempdir().unwrap();
        let l0 = Arc::new(L0Store::new(dir.path().to_str().unwrap()).unwrap());
        let task_iri = "iri://task/concurrent-journal";
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let mut workers = Vec::new();
        for worker in 0..8 {
            let l0 = l0.clone();
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                let journal = TaskExecutionJournal::new(l0, task_iri).unwrap();
                barrier.wait();
                journal
                    .append(TaskExecutionJournalKind::CheckpointCommitted {
                        checkpoint_iri: format!("iri://checkpoint/concurrent/{worker}"),
                        checkpoint_name: format!("worker-{worker}"),
                    })
                    .unwrap()
                    .sequence
            }));
        }
        let mut sequences = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        sequences.sort_unstable();
        assert_eq!(sequences, (0..8).collect::<Vec<_>>());
        let journal = TaskExecutionJournal::new(l0.clone(), task_iri).unwrap();
        let verification = journal.verify().unwrap();
        assert!(verification.valid, "{:?}", verification.failures);
        journal.seal("completed").unwrap();
        assert!(journal.verify().unwrap().sealed);
    }

    #[test]
    fn verification_detects_l0_projection_replacement() {
        let (_dir, l0, journal) = journal();
        let event = journal
            .append(TaskExecutionJournalKind::CheckpointCommitted {
                checkpoint_iri: "iri://checkpoint/task/1".into(),
                checkpoint_name: "turn".into(),
            })
            .unwrap();
        let iri = format!(
            "iri://task-journal/{}/seq_{:020}",
            short_hash("iri://task/trace-test"),
            event.sequence
        );
        l0.store(&iri, "tampered projection").unwrap();
        let verification = journal.verify().unwrap();
        assert!(!verification.valid);
        assert!(verification
            .failures
            .iter()
            .any(|failure| failure.contains("L0 projection differs")));
    }
}
