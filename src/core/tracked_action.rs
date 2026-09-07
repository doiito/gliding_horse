use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::execution_journal::ToolCallIdentity;

/// Normalize the heterogeneous failure signals returned by built-in and MCP
/// tools.  Command tools report a non-zero `exit_code` without an `error`
/// field, while other tools may use `success=false` or `timed_out=true`.
pub fn tool_result_failed(result: &Value) -> bool {
    result.get("error").is_some()
        || result
            .get("exit_code")
            .and_then(Value::as_i64)
            .is_some_and(|code| code != 0)
        || result.get("success").and_then(Value::as_bool) == Some(false)
        || result.get("timed_out").and_then(Value::as_bool) == Some(true)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub path: String,
    pub size_bytes: Option<u64>,
    pub hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActionStatus {
    Success,
    Failed,
    Retried,
}

/// Stable semantic class assigned by the kernel to an executable verifier.
/// This is intentionally independent of the process exit status: a test
/// runner that exits zero after executing no tests is still a successful
/// process, but it is not successful verification evidence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerificationKind {
    TestExecution,
    Build,
    Lint,
    Type,
    Syntax,
    Artifact,
    Smoke,
}

/// Evidence outcome derived from both process status and verifier-specific
/// output. `Inconclusive` is fail-closed: it can record a real invocation,
/// but can never authorize a positive receipt.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerificationOutcome {
    Passed,
    Failed,
    Inconclusive,
}

/// Versioned, serializable kernel assessment of one verifier invocation.
/// `count` is the number of checks that actually ran (not merely collected or
/// skipped). Non-test verifiers represent their one deterministic invocation
/// as `Some(1)`; test execution requires an output-derived cardinality.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerificationAssessment {
    pub parser_version: String,
    pub kind: VerificationKind,
    pub outcome: VerificationOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    #[serde(default)]
    pub skipped_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Bounded, non-authoritative failure data selected by the kernel from
    /// the verifier's routed stdout/stderr. It exists so a fresh corrective
    /// Agent can target an observed failure without inheriting the previous
    /// Agent's transcript. Consumers must still bind it to this action's
    /// composite call identity and must never treat its text as instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<String>,
}

pub const VERIFICATION_ASSESSMENT_PARSER_VERSION: &str = "glidinghorse.verification-assessment/v2";

const VERIFICATION_ASSESSMENT_KEY: &str = "verification_assessment";
const VERIFICATION_INVALIDATED_KEY: &str = "verification_receipt_invalidated";
const RUNTIME_EPOCH_ID_KEY: &str = "__gh_runtime_epoch_id";
const COMPLETION_SEQUENCE_KEY: &str = "__gh_completion_sequence";
const WORKSPACE_COORDINATOR_ID_KEY: &str = "__gh_workspace_coordinator_id";
const WORKSPACE_SETTLEMENT_SEQUENCE_KEY: &str = "__gh_workspace_settlement_sequence";
const WORKSPACE_MUTATION_EPOCH_KEY: &str = "__gh_workspace_mutation_epoch";
const WORKSPACE_MANIFEST_SHA256_KEY: &str = "__gh_workspace_manifest_sha256";
const WORKSPACE_MANIFEST_DRIFT_KEY: &str = "__gh_workspace_manifest_drift";
const VERIFICATION_INVOCATION_SHA256_KEY: &str = "__gh_verification_invocation_sha256";

static PROCESS_RUNTIME_EPOCH_ID: OnceLock<String> = OnceLock::new();
static ACTION_COMPLETION_SEQUENCE: AtomicU64 = AtomicU64::new(1);

pub(crate) fn current_runtime_epoch_id() -> &'static str {
    PROCESS_RUNTIME_EPOCH_ID.get_or_init(|| {
        let started_unix_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        format!(
            "runtime:{started_unix_nanos}:{}",
            uuid::Uuid::new_v4().hyphenated()
        )
    })
}

fn next_action_completion_sequence() -> u64 {
    ACTION_COMPLETION_SEQUENCE
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
            current.checked_add(1)
        })
        .expect("process action completion sequence exhausted")
}

fn normalized_shell_invocation(command: &str) -> Vec<String> {
    let characters = command.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote = None::<char>;
    let mut escaped = false;
    let mut word_started = false;
    let mut index = 0usize;
    let finish_word = |current: &mut String, word_started: &mut bool, tokens: &mut Vec<String>| {
        if *word_started {
            tokens.push(std::mem::take(current));
            *word_started = false;
        }
    };

    while index < characters.len() {
        let character = characters[index];
        if escaped {
            current.push(character);
            word_started = true;
            escaped = false;
            index = index.saturating_add(1);
            continue;
        }
        match quote {
            Some(active) => {
                current.push(character);
                word_started = true;
                if character == active {
                    quote = None;
                } else if active == '"' && character == '\\' {
                    escaped = true;
                }
            }
            None if matches!(character, '\'' | '"') => {
                current.push(character);
                word_started = true;
                quote = Some(character);
            }
            None if character == '\\' => {
                current.push(character);
                word_started = true;
                escaped = true;
            }
            None if character.is_whitespace() => {
                finish_word(&mut current, &mut word_started, &mut tokens);
            }
            None if matches!(
                character,
                '&' | '|' | ';' | '<' | '>' | '(' | ')' | '{' | '}'
            ) =>
            {
                finish_word(&mut current, &mut word_started, &mut tokens);
                let doubled = matches!(character, '&' | '|' | '<' | '>')
                    && characters.get(index + 1) == Some(&character);
                tokens.push(if doubled {
                    index = index.saturating_add(1);
                    format!("{character}{character}")
                } else {
                    character.to_string()
                });
            }
            None => {
                current.push(character);
                word_started = true;
            }
        }
        index = index.saturating_add(1);
    }
    finish_word(&mut current, &mut word_started, &mut tokens);
    tokens
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        Value::Object(object) => {
            let mut keys = object.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut canonical = serde_json::Map::new();
            for key in keys {
                canonical.insert(key.clone(), canonical_json(&object[key]));
            }
            Value::Object(canonical)
        }
        value => value.clone(),
    }
}

/// Hash the hook-final effective invocation without including Agent/L1/call
/// identity. A later isolated Agent can therefore refresh the same verifier,
/// while its successful receipt remains distinct because that receipt also
/// binds the full provider call identity.
pub(crate) fn verification_invocation_sha256(tool_name: &str, args: &Value) -> String {
    let arguments = if matches!(tool_name, "bash" | "powershell") {
        let mut normalized = canonical_json(args);
        if let Some(object) = normalized.as_object_mut() {
            if let Some((key, command)) = ["command", "script", "code"].iter().find_map(|key| {
                object
                    .get(*key)
                    .and_then(Value::as_str)
                    .map(|command| ((*key).to_string(), command.to_string()))
            }) {
                object.insert(
                    key,
                    Value::Array(
                        normalized_shell_invocation(&command)
                            .into_iter()
                            .map(Value::String)
                            .collect(),
                    ),
                );
            }
        }
        normalized
    } else {
        canonical_json(args)
    };
    let material = serde_json::json!({
        "schema_version": "glidinghorse.verification-invocation/v1",
        "tool_name": tool_name,
        "arguments": arguments,
    });
    format!(
        "sha256:{}",
        crate::utils::CryptoUtils::sha256_hex(
            &serde_json::to_string(&material).unwrap_or_default()
        )
    )
}

/// The exact bounded portion of a `file_read` result that was placed in the
/// next model message.  Execution success alone is not evidence disclosure:
/// a post-execution hook may withhold the result, and the result router may
/// expose only a prefix of a large file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileReadDisclosure {
    pub path: String,
    pub offset: u64,
    pub returned: u64,
    pub total_lines: u64,
    /// Number of concrete line values present in the routed payload. A result
    /// router may retain original range metadata while exposing only a
    /// preview, so this must match `returned` before the disclosure can serve
    /// as an overwrite baseline.
    #[serde(default)]
    pub delivered_line_count: u64,
    /// Hash of the complete file revision from which this range was read.
    /// Coverage from different revisions must never be combined.
    pub content_sha256: String,
    #[serde(default)]
    pub partial_line_preview: bool,
    #[serde(default)]
    pub archived: bool,
}

/// Exact fields from a routed built-in `file_write` result.  This is kept
/// separately from the raw execution record so an ownership attestation
/// cannot be minted when routing replaced or summarized away the decisive
/// no-op/content-hash fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileWriteDisclosure {
    pub path: String,
    pub changed: bool,
    pub content_sha256: String,
}

/// Privacy-preserving proof of what the model actually received after both
/// SkillAfter policy and result routing.  The routed payload itself is never
/// retained in the action ledger.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDisclosureReceipt {
    pub disclosed_to_model: bool,
    /// The model received a policy placeholder rather than the actual tool
    /// result. Such a message is visible, but it cannot prove file contents
    /// or a verifier outcome.
    #[serde(default)]
    pub result_withheld: bool,
    pub routed_payload_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_read: Option<FileReadDisclosure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_write: Option<FileWriteDisclosure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedAction {
    pub action_id: String,
    /// Full internal identity for this one provider-issued call.  The raw
    /// provider call id remains unchanged inside the composite identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_identity: Option<ToolCallIdentity>,
    pub tool_name: String,
    pub agent_role: String,
    pub duration_secs: f64,
    pub status: ActionStatus,
    pub files_created: Vec<FileChange>,
    pub files_modified: Vec<FileChange>,
    /// Exact removals observed in the same bounded mutation window. Removed
    /// paths are never delivery artifacts, but retaining them prevents a
    /// failed or shell-based call from hiding cross-package side effects.
    #[serde(default)]
    pub files_removed: Vec<FileChange>,
    #[serde(default)]
    pub directories_created: Vec<String>,
    #[serde(default)]
    pub directories_removed: Vec<String>,
    /// True only when the kernel captured a complete per-path delta (or the
    /// reserved single-file builtin itself supplies an exact effect). A
    /// generation/fingerprint fallback may inform progress but cannot grant
    /// artifact ownership.
    #[serde(default)]
    pub workspace_delta_complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_delta_sha256: Option<String>,
    /// A captured path escaped this child's resource lease or the mutation
    /// window otherwise lost exclusive attribution. Such an action remains
    /// visible for diagnostics/conflict handling but is never delivery proof.
    #[serde(default)]
    pub workspace_delta_contaminated: bool,
    pub files_read: Vec<String>,
    pub error: Option<String>,
    /// Confirmed material workspace effect, independent of command syntax.
    /// This is set directly for changed file tools and after before/after
    /// confirmation for shell-like tools.  The historical field name is kept
    /// for durable-record readability; a structure-only effect can therefore
    /// be true even when both file-change lists are empty.
    #[serde(default)]
    pub substantive_effect: bool,
    /// Kernel classification that the executed call was a deterministic
    /// verifier, regardless of whether the check passed. This distinguishes
    /// an observed test failure from an arbitrary failed shell command.
    #[serde(default)]
    pub verification_attempted: bool,
    /// Kernel classification of an actually executed, successful,
    /// deterministic verifier. Model prose and artifact metadata cannot set
    /// this bit; evidence consumers must additionally require an admitted,
    /// non-withheld disclosure receipt.
    #[serde(default)]
    pub successful_verification: bool,
    #[serde(default)]
    pub tool_args: HashMap<String, Value>,
    /// Filled only after the post-execution hook and result router have run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disclosure: Option<ToolDisclosureReceipt>,
}

#[derive(Serialize)]
struct SuccessfulVerificationReceiptMaterial<'a> {
    schema_version: &'static str,
    action_id: &'a str,
    call_identity: &'a ToolCallIdentity,
    tool_name: &'a str,
    assessment_parser_version: &'a str,
    verification_kind: VerificationKind,
    verification_outcome: VerificationOutcome,
    verification_count: Option<u64>,
    verification_skipped_count: u64,
    runtime_epoch_id: &'a str,
    completion_sequence: u64,
    workspace_coordinator_id: &'a str,
    workspace_settlement_sequence: u64,
    workspace_mutation_epoch: u64,
    workspace_manifest_sha256: Option<&'a str>,
    invocation_sha256: &'a str,
    routed_payload_sha256: &'a str,
}

/// Typed, content-addressed evidence for a verifier that still describes the
/// current workspace epoch. Aggregators can carry this value without
/// re-parsing command text or trusting a detached receipt hash.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SuccessfulVerificationEvidence {
    pub receipt_sha256: String,
    pub assessment: VerificationAssessment,
    pub action_id: String,
    pub call_identity: ToolCallIdentity,
    pub tool_name: String,
    pub runtime_epoch_id: String,
    pub completion_sequence: u64,
    pub workspace_coordinator_id: String,
    pub workspace_settlement_sequence: u64,
    pub workspace_mutation_epoch: u64,
    pub workspace_manifest_sha256: Option<String>,
    pub invocation_sha256: String,
}

/// Latest trustworthy typed attempt for one normalized verifier invocation in
/// the current coordinated workspace epoch. Failed/Inconclusive attempts are
/// intentionally retained so recovery and persistence cannot fall back to an
/// older green result for the same invocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct CurrentVerificationAttemptEvidence {
    pub assessment: VerificationAssessment,
    pub action_id: String,
    pub call_identity: ToolCallIdentity,
    pub tool_name: String,
    pub runtime_epoch_id: String,
    pub completion_sequence: u64,
    pub workspace_coordinator_id: String,
    pub workspace_settlement_sequence: u64,
    pub workspace_mutation_epoch: u64,
    pub workspace_manifest_sha256: Option<String>,
    pub invocation_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub successful_receipt_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActionWorkspaceOrdering {
    runtime_epoch_id: String,
    completion_sequence: u64,
    coordinator_id: String,
    settlement_sequence: u64,
    mutation_epoch: u64,
    manifest_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArtifactAttestationReceipt {
    pub path: String,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct ArtifactAttestationReceiptMaterial<'a> {
    schema_version: &'static str,
    action_id: &'a str,
    call_identity: &'a ToolCallIdentity,
    tool_name: &'a str,
    path: &'a str,
    content_sha256: &'a str,
    routed_payload_sha256: &'a str,
}

impl TrackedAction {
    pub(crate) fn verification_assessment(&self) -> Option<VerificationAssessment> {
        serde_json::from_value(self.tool_args.get(VERIFICATION_ASSESSMENT_KEY)?.clone()).ok()
    }

    fn workspace_ordering(&self) -> Option<ActionWorkspaceOrdering> {
        let runtime_epoch_id = self
            .tool_args
            .get(RUNTIME_EPOCH_ID_KEY)?
            .as_str()?
            .trim()
            .to_string();
        let completion_sequence = self.tool_args.get(COMPLETION_SEQUENCE_KEY)?.as_u64()?;
        let coordinator_id = self
            .tool_args
            .get(WORKSPACE_COORDINATOR_ID_KEY)?
            .as_str()?
            .trim()
            .to_string();
        let settlement_sequence = self
            .tool_args
            .get(WORKSPACE_SETTLEMENT_SEQUENCE_KEY)?
            .as_u64()?;
        let mutation_epoch = self.tool_args.get(WORKSPACE_MUTATION_EPOCH_KEY)?.as_u64()?;
        if runtime_epoch_id.is_empty()
            || completion_sequence == 0
            || coordinator_id.is_empty()
            || settlement_sequence == 0
        {
            return None;
        }
        let manifest_sha256 = self
            .tool_args
            .get(WORKSPACE_MANIFEST_SHA256_KEY)
            .and_then(Value::as_str)
            .filter(|digest| {
                digest.len() == 71
                    && digest.starts_with("sha256:")
                    && digest[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            .map(str::to_ascii_lowercase);
        Some(ActionWorkspaceOrdering {
            runtime_epoch_id,
            completion_sequence,
            coordinator_id,
            settlement_sequence,
            mutation_epoch,
            manifest_sha256,
        })
    }

    pub(crate) fn verification_invocation_sha256(&self) -> Option<&str> {
        self.tool_args
            .get(VERIFICATION_INVOCATION_SHA256_KEY)?
            .as_str()
            .filter(|digest| {
                digest.len() == 71
                    && digest.starts_with("sha256:")
                    && digest[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
            })
    }

    /// Return a content-addressed kernel receipt only when a deterministic
    /// verifier actually succeeded and its unredacted routed result was
    /// consumed by the next provider request. The full composite identity is
    /// hashed into the receipt while the provider's raw call id itself stays
    /// unchanged in `call_identity`.
    pub(crate) fn successful_verification_receipt_sha256(&self) -> Option<String> {
        let identity = self.call_identity.as_ref()?;
        let disclosure = self.disclosure.as_ref()?;
        let assessment = self.verification_assessment()?;
        let ordering = self.workspace_ordering()?;
        let invocation_sha256 = self.verification_invocation_sha256()?;
        if self.status != ActionStatus::Success
            || self.error.is_some()
            || self.substantive_effect
            || !self.verification_attempted
            || !self.successful_verification
            || assessment.parser_version != VERIFICATION_ASSESSMENT_PARSER_VERSION
            || assessment.outcome != VerificationOutcome::Passed
            || assessment.count.is_none_or(|count| count == 0)
            || self
                .tool_args
                .get(VERIFICATION_INVALIDATED_KEY)
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || !disclosure.disclosed_to_model
            || disclosure.result_withheld
            || ordering.runtime_epoch_id != current_runtime_epoch_id()
        {
            return None;
        }
        let material = SuccessfulVerificationReceiptMaterial {
            schema_version: "glidinghorse.successful-verification-receipt/v3",
            action_id: &self.action_id,
            call_identity: identity,
            tool_name: &self.tool_name,
            assessment_parser_version: &assessment.parser_version,
            verification_kind: assessment.kind,
            verification_outcome: assessment.outcome,
            verification_count: assessment.count,
            verification_skipped_count: assessment.skipped_count,
            runtime_epoch_id: &ordering.runtime_epoch_id,
            completion_sequence: ordering.completion_sequence,
            workspace_coordinator_id: &ordering.coordinator_id,
            workspace_settlement_sequence: ordering.settlement_sequence,
            workspace_mutation_epoch: ordering.mutation_epoch,
            workspace_manifest_sha256: ordering.manifest_sha256.as_deref(),
            invocation_sha256,
            routed_payload_sha256: &disclosure.routed_payload_sha256,
        };
        let encoded = serde_json::to_string(&material).ok()?;
        Some(format!(
            "sha256:{}",
            crate::utils::CryptoUtils::sha256_hex(&encoded)
        ))
    }

    /// Produce a content-addressed ownership attestation when a child writes
    /// the complete intended contents of an already-identical file.  A retry
    /// must not mutate correct artifacts merely to recreate a delivery path,
    /// but model prose alone must not be allowed to claim the existing file.
    /// The built-in `file_write` comparison, full isolated-call identity and
    /// the subsequently consumed unredacted tool result are all bound here.
    pub(crate) fn successful_artifact_attestation(&self) -> Option<ArtifactAttestationReceipt> {
        self.artifact_attestation(true)
    }

    /// Candidate safe for closing the *next* provider dispatch. This mirrors
    /// `current_verification_close_candidates`: the exact routed file_write
    /// result is queued in that request but is committed as consumed only
    /// after the provider accepts it. Durable/result-boundary validation still
    /// calls `successful_artifact_attestation` and therefore requires that
    /// confirmation.
    pub(crate) fn artifact_attestation_close_candidate(
        &self,
    ) -> Option<ArtifactAttestationReceipt> {
        self.artifact_attestation(false)
    }

    fn artifact_attestation(
        &self,
        require_confirmed_disclosure: bool,
    ) -> Option<ArtifactAttestationReceipt> {
        let identity = self.call_identity.as_ref()?;
        let disclosure = self.disclosure.as_ref()?;
        let path = self.tool_args.get("path")?.as_str()?.trim();
        let content_sha256 = self.tool_args.get("write_content_sha256")?.as_str()?.trim();
        let routed_write = disclosure.file_write.as_ref()?;
        if self.tool_name != "file_write"
            || self.status != ActionStatus::Success
            || self.error.is_some()
            || self.substantive_effect
            || !self.workspace_delta_complete
            || self.workspace_delta_contaminated
            || self.tool_args.get("write_changed").and_then(Value::as_bool) != Some(false)
            || path.is_empty()
            || path.chars().any(char::is_control)
            || content_sha256.len() != 64
            || !content_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || (require_confirmed_disclosure && !disclosure.disclosed_to_model)
            || disclosure.result_withheld
            || routed_write.changed
            || routed_write.path.trim() != path
            || routed_write.content_sha256.trim() != content_sha256
        {
            return None;
        }
        let material = ArtifactAttestationReceiptMaterial {
            schema_version: "glidinghorse.artifact-attestation-receipt/v1",
            action_id: &self.action_id,
            call_identity: identity,
            tool_name: &self.tool_name,
            path,
            content_sha256,
            routed_payload_sha256: &disclosure.routed_payload_sha256,
        };
        let encoded = serde_json::to_string(&material).ok()?;
        Some(ArtifactAttestationReceipt {
            path: path.to_string(),
            receipt_sha256: format!("sha256:{}", crate::utils::CryptoUtils::sha256_hex(&encoded)),
        })
    }
}

/// Extract typed verifier evidence without trusting the slice order. Parallel
/// child results are appended in completion order, which is not necessarily
/// the order in which their workspace calls settled. Kernel coordinator
/// stamps are therefore the sole freshness clock.
fn current_typed_verification_attempts_with_disclosure(
    actions: &[TrackedAction],
    require_provider_disclosure: bool,
) -> Vec<CurrentVerificationAttemptEvidence> {
    let runtime_epoch_id = current_runtime_epoch_id();
    let relevant = actions
        .iter()
        .filter_map(|action| Some((action, action.workspace_ordering()?)))
        .filter(|(action, ordering)| {
            ordering.runtime_epoch_id == runtime_epoch_id
                && (action.substantive_effect
                    || action.workspace_delta_contaminated
                    || action.verification_attempted
                    || action
                        .tool_args
                        .get(WORKSPACE_MANIFEST_DRIFT_KEY)
                        .and_then(Value::as_bool)
                        .unwrap_or(false))
        })
        .collect::<Vec<_>>();
    let coordinator_ids = relevant
        .iter()
        .map(|(_, ordering)| ordering.coordinator_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    if coordinator_ids.len() != 1 {
        // No current-runtime evidence, or multiple independent executors that
        // did not share one workspace gate. Neither case has a total order.
        return Vec::new();
    }
    let coordinator_id = *coordinator_ids
        .iter()
        .next()
        .expect("one coordinator id was checked above");
    let current_mutation_epoch = relevant
        .iter()
        .map(|(_, ordering)| ordering.mutation_epoch)
        .max()
        .unwrap_or(0);
    let latest_invalidating_completion = relevant
        .iter()
        .filter(|(action, _)| action.substantive_effect || action.workspace_delta_contaminated)
        .map(|(_, ordering)| ordering.completion_sequence)
        .max()
        .unwrap_or(0);
    let latest_invalidating_settlement = relevant
        .iter()
        .filter(|(action, _)| action.substantive_effect || action.workspace_delta_contaminated)
        .map(|(_, ordering)| ordering.settlement_sequence)
        .max()
        .unwrap_or(0);
    let current_manifest_sha256 = relevant
        .iter()
        .filter(|(_, ordering)| ordering.mutation_epoch == current_mutation_epoch)
        .filter_map(|(_, ordering)| {
            Some((
                ordering.settlement_sequence,
                ordering.manifest_sha256.clone()?,
            ))
        })
        .max_by_key(|(settlement_sequence, _)| *settlement_sequence)
        .map(|(_, digest)| digest);
    let mut latest_attempt_by_invocation = HashMap::<String, u64>::new();
    for (action, ordering) in &relevant {
        if ordering.mutation_epoch != current_mutation_epoch || !action.verification_attempted {
            continue;
        }
        let Some(invocation_sha256) = action.verification_invocation_sha256() else {
            continue;
        };
        latest_attempt_by_invocation
            .entry(invocation_sha256.to_string())
            .and_modify(|latest| *latest = (*latest).max(ordering.settlement_sequence))
            .or_insert(ordering.settlement_sequence);
    }

    let mut evidence = relevant
        .iter()
        .filter(|(action, ordering)| {
            ordering.coordinator_id == coordinator_id
                && ordering.mutation_epoch == current_mutation_epoch
                && ordering.completion_sequence > latest_invalidating_completion
                && ordering.settlement_sequence > latest_invalidating_settlement
                && current_manifest_sha256
                    .as_ref()
                    .is_none_or(|current| ordering.manifest_sha256.as_ref() == Some(current))
                && action
                    .verification_invocation_sha256()
                    .is_some_and(|invocation| {
                        latest_attempt_by_invocation.get(invocation).copied()
                            == Some(ordering.settlement_sequence)
                    })
        })
        .filter_map(|(action, ordering)| {
            let assessment = action.verification_assessment()?;
            let call_identity = action.call_identity.clone()?;
            let disclosure = action.disclosure.as_ref()?;
            let invocation_sha256 = action.verification_invocation_sha256()?.to_string();
            if !action.verification_attempted
                || action.substantive_effect
                || action.workspace_delta_contaminated
                || assessment.parser_version != VERIFICATION_ASSESSMENT_PARSER_VERSION
                || (require_provider_disclosure && !disclosure.disclosed_to_model)
                || disclosure.result_withheld
                || [
                    call_identity.agent_id.as_str(),
                    call_identity.l1_session_id.as_str(),
                    call_identity.llm_request_id.as_str(),
                    call_identity.provider_call_id.as_str(),
                ]
                .iter()
                .any(|component| component.trim().is_empty())
            {
                return None;
            }
            Some(CurrentVerificationAttemptEvidence {
                successful_receipt_sha256: action.successful_verification_receipt_sha256(),
                assessment,
                action_id: action.action_id.clone(),
                call_identity,
                tool_name: action.tool_name.clone(),
                runtime_epoch_id: ordering.runtime_epoch_id.clone(),
                completion_sequence: ordering.completion_sequence,
                workspace_coordinator_id: ordering.coordinator_id.clone(),
                workspace_settlement_sequence: ordering.settlement_sequence,
                workspace_mutation_epoch: ordering.mutation_epoch,
                workspace_manifest_sha256: ordering.manifest_sha256.clone(),
                invocation_sha256,
            })
        })
        .collect::<Vec<_>>();
    evidence.sort_by_key(|item| (item.workspace_settlement_sequence, item.completion_sequence));

    // A later executable failure is current counter-evidence for the whole
    // verifier kind, even when the model chose a differently normalized
    // command line. Invocation hashes prevent stale retries of one command
    // from winning, but they cannot prove that two test/build invocations
    // cover disjoint acceptance criteria. Fail closed until a still-later
    // pass of that kind establishes a new green boundary. Inconclusive probes
    // do not invalidate an independent pass; for the same invocation they
    // already supersede it through `latest_attempt_by_invocation` above.
    let failed_boundaries = evidence
        .iter()
        .filter(|item| item.assessment.outcome == VerificationOutcome::Failed)
        .map(|item| {
            (
                item.assessment.kind,
                (item.workspace_settlement_sequence, item.completion_sequence),
            )
        })
        .collect::<Vec<_>>();
    evidence.retain(|item| {
        item.assessment.outcome != VerificationOutcome::Passed
            || !failed_boundaries.iter().any(|(kind, failed_order)| {
                *kind == item.assessment.kind
                    && *failed_order
                        > (item.workspace_settlement_sequence, item.completion_sequence)
            })
    });
    evidence
}

pub(crate) fn current_typed_verification_attempts(
    actions: &[TrackedAction],
) -> Vec<CurrentVerificationAttemptEvidence> {
    current_typed_verification_attempts_with_disclosure(actions, true)
}

/// Verification candidates safe for narrowing the *next* provider dispatch
/// to a terminal response. The routed tool result must already exist and must
/// not be withheld, but it may still be awaiting confirmation that this very
/// next provider request consumed it. No durable success receipt is minted
/// until `current_typed_verification_attempts` observes that confirmation.
pub(crate) fn current_verification_close_candidates(
    actions: &[TrackedAction],
) -> Vec<CurrentVerificationAttemptEvidence> {
    current_typed_verification_attempts_with_disclosure(actions, false)
}

pub(crate) fn current_successful_verification_evidence(
    actions: &[TrackedAction],
) -> Vec<SuccessfulVerificationEvidence> {
    current_typed_verification_attempts(actions)
        .into_iter()
        .filter_map(|attempt| {
            if attempt.assessment.outcome != VerificationOutcome::Passed {
                return None;
            }
            Some(SuccessfulVerificationEvidence {
                receipt_sha256: attempt.successful_receipt_sha256?,
                assessment: attempt.assessment,
                action_id: attempt.action_id,
                call_identity: attempt.call_identity,
                tool_name: attempt.tool_name,
                runtime_epoch_id: attempt.runtime_epoch_id,
                completion_sequence: attempt.completion_sequence,
                workspace_coordinator_id: attempt.workspace_coordinator_id,
                workspace_settlement_sequence: attempt.workspace_settlement_sequence,
                workspace_mutation_epoch: attempt.workspace_mutation_epoch,
                workspace_manifest_sha256: attempt.workspace_manifest_sha256,
                invocation_sha256: attempt.invocation_sha256,
            })
        })
        .collect()
}

pub struct ActionTracker {
    pub actions: Vec<TrackedAction>,
    pub task_iri: String,
    pub agent_role: String,
    pub started_at: DateTime<Utc>,
    local_workspace_coordinator_id: String,
    local_workspace_settlement_sequence: u64,
    local_workspace_mutation_epoch: u64,
}

impl ActionTracker {
    pub fn new(task_iri: &str, agent_role: &str) -> Self {
        Self {
            actions: Vec::new(),
            task_iri: task_iri.to_string(),
            agent_role: agent_role.to_string(),
            started_at: Utc::now(),
            local_workspace_coordinator_id: format!(
                "action-tracker:{}",
                uuid::Uuid::new_v4().hyphenated()
            ),
            local_workspace_settlement_sequence: 0,
            local_workspace_mutation_epoch: 0,
        }
    }

    pub fn record(&mut self, tool_name: &str, args: &Value, result: &Value, duration_secs: f64) {
        self.record_with_identity(tool_name, args, result, duration_secs, None);
    }

    pub fn record_with_identity(
        &mut self,
        tool_name: &str,
        args: &Value,
        result: &Value,
        duration_secs: f64,
        call_identity: Option<ToolCallIdentity>,
    ) {
        let completion_sequence = next_action_completion_sequence();
        self.local_workspace_settlement_sequence = self
            .local_workspace_settlement_sequence
            .checked_add(1)
            .expect("local workspace settlement sequence exhausted");
        let mut action = TrackedAction {
            action_id: format!("act_{}", uuid::Uuid::new_v4().hyphenated()),
            call_identity,
            tool_name: tool_name.to_string(),
            agent_role: self.agent_role.clone(),
            duration_secs,
            status: if tool_result_failed(result) {
                ActionStatus::Failed
            } else {
                ActionStatus::Success
            },
            files_created: vec![],
            files_modified: vec![],
            files_removed: vec![],
            directories_created: vec![],
            directories_removed: vec![],
            workspace_delta_complete: false,
            workspace_delta_sha256: None,
            workspace_delta_contaminated: false,
            files_read: vec![],
            error: result
                .get("error")
                .and_then(|e| e.as_str())
                .map(String::from),
            substantive_effect: false,
            verification_attempted: false,
            successful_verification: false,
            tool_args: HashMap::new(),
            disclosure: None,
        };

        match tool_name {
            "file_write" => {
                if let Some(path) = result
                    .get("path")
                    .and_then(Value::as_str)
                    .or_else(|| args.get("path").and_then(Value::as_str))
                {
                    action
                        .tool_args
                        .insert("path".to_string(), Value::String(path.to_string()));
                    if let Some(requested_path) = args.get("path").and_then(Value::as_str) {
                        action.tool_args.insert(
                            "requested_path".to_string(),
                            Value::String(requested_path.to_string()),
                        );
                    }
                    if let Some(changed) = result.get("changed").and_then(Value::as_bool) {
                        action
                            .tool_args
                            .insert("write_changed".to_string(), Value::Bool(changed));
                    }
                    let content_hash =
                        result
                            .get("content_sha256")
                            .and_then(Value::as_str)
                            .filter(|hash| {
                                hash.len() == 64
                                    && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                            });
                    if let Some(hash) = content_hash {
                        action.tool_args.insert(
                            "write_content_sha256".to_string(),
                            Value::String(hash.to_string()),
                        );
                    }
                    if let Some(requested_content) = args.get("content").and_then(Value::as_str) {
                        action.tool_args.insert(
                            "requested_content_sha256".to_string(),
                            Value::String(crate::utils::CryptoUtils::sha256_hex(requested_content)),
                        );
                    }
                    if action.status == ActionStatus::Success
                        && result.get("changed").and_then(Value::as_bool) == Some(true)
                    {
                        let change = FileChange {
                            path: path.to_string(),
                            size_bytes: result.get("bytes_written").and_then(Value::as_u64),
                            hash: content_hash.map(str::to_string),
                        };
                        if result.get("created").and_then(Value::as_bool) == Some(true) {
                            action.files_created.push(change);
                        } else {
                            action.files_modified.push(change);
                        }
                        action.substantive_effect = true;
                    }
                    if action.status == ActionStatus::Success {
                        // Reserved file_write has one exact secure target, so
                        // its changed/no-op effect is complete without a
                        // whole-workspace manifest.
                        action.workspace_delta_complete = true;
                    }
                }
            }
            "file_edit" => {
                if let Some(path) = result
                    .get("path")
                    .and_then(Value::as_str)
                    .or_else(|| args.get("path").and_then(Value::as_str))
                {
                    action
                        .tool_args
                        .insert("path".to_string(), Value::String(path.to_string()));
                    if action.status == ActionStatus::Success
                        && result.get("changed").and_then(Value::as_bool) == Some(true)
                    {
                        action.files_modified.push(FileChange {
                            path: path.to_string(),
                            size_bytes: result.get("size_bytes").and_then(Value::as_u64),
                            hash: result
                                .get("content_sha256")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                        });
                        action.substantive_effect = true;
                    }
                    if action.status == ActionStatus::Success {
                        action.workspace_delta_complete = true;
                    }
                }
            }
            "file_read" => {
                if action.status == ActionStatus::Success {
                    if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                        action.files_read.push(path.to_string());
                        action
                            .tool_args
                            .insert("path".to_string(), Value::String(path.to_string()));
                    }
                    // Preserve execution metadata for diagnostics only. CA
                    // conformance gates use `disclosure`, which is recorded
                    // after hooks and routing and therefore reflects what the
                    // model actually saw.
                    for (source, receipt) in [
                        ("offset", "read_offset"),
                        ("returned", "read_returned"),
                        ("total_lines", "read_total_lines"),
                    ] {
                        if let Some(value) = result.get(source).and_then(Value::as_u64) {
                            action
                                .tool_args
                                .insert(receipt.to_string(), Value::from(value));
                        }
                    }
                    if let Some(hash) = result.get("content_sha256").and_then(Value::as_str) {
                        action.tool_args.insert(
                            "read_content_sha256".to_string(),
                            Value::String(hash.to_string()),
                        );
                    }
                }
            }
            "bash" | "powershell" => {
                action.tool_args.insert(
                    "command".to_string(),
                    args.get("command").cloned().unwrap_or_default(),
                );
            }
            "code_execute" => {
                action.tool_args.insert(
                    "code".to_string(),
                    args.get("code").cloned().unwrap_or_default(),
                );
            }
            _ => {}
        }

        let invalidates_prior_verification =
            action.substantive_effect || action.workspace_delta_contaminated;
        if invalidates_prior_verification {
            self.local_workspace_mutation_epoch = self
                .local_workspace_mutation_epoch
                .checked_add(1)
                .expect("local workspace mutation epoch exhausted");
        }
        action.tool_args.insert(
            RUNTIME_EPOCH_ID_KEY.to_string(),
            Value::String(current_runtime_epoch_id().to_string()),
        );
        action.tool_args.insert(
            COMPLETION_SEQUENCE_KEY.to_string(),
            Value::from(completion_sequence),
        );
        action.tool_args.insert(
            WORKSPACE_COORDINATOR_ID_KEY.to_string(),
            Value::String(self.local_workspace_coordinator_id.clone()),
        );
        action.tool_args.insert(
            WORKSPACE_SETTLEMENT_SEQUENCE_KEY.to_string(),
            Value::from(self.local_workspace_settlement_sequence),
        );
        action.tool_args.insert(
            WORKSPACE_MUTATION_EPOCH_KEY.to_string(),
            Value::from(self.local_workspace_mutation_epoch),
        );
        action.tool_args.insert(
            VERIFICATION_INVOCATION_SHA256_KEY.to_string(),
            Value::String(verification_invocation_sha256(tool_name, args)),
        );
        self.actions.push(action);
        if invalidates_prior_verification {
            self.invalidate_prior_verification_receipts();
        }
    }

    /// Replace the tracker-local provisional ordering with the stamp emitted
    /// by the shared ToolExecutor coordinator while its guard is still held.
    pub(crate) fn record_last_workspace_settlement(
        &mut self,
        stamp: &crate::tools::tool_executor::WorkspaceSettlementStamp,
    ) {
        let Some(action) = self.actions.last_mut() else {
            return;
        };
        action.tool_args.insert(
            WORKSPACE_COORDINATOR_ID_KEY.to_string(),
            Value::String(stamp.coordinator_id.clone()),
        );
        action.tool_args.insert(
            WORKSPACE_SETTLEMENT_SEQUENCE_KEY.to_string(),
            Value::from(stamp.settlement_sequence),
        );
        action.tool_args.insert(
            WORKSPACE_MUTATION_EPOCH_KEY.to_string(),
            Value::from(stamp.mutation_epoch),
        );
        if let Some(digest) = stamp.manifest_sha256.as_ref() {
            action.tool_args.insert(
                WORKSPACE_MANIFEST_SHA256_KEY.to_string(),
                Value::String(digest.clone()),
            );
        } else {
            action.tool_args.remove(WORKSPACE_MANIFEST_SHA256_KEY);
        }
        if stamp.manifest_drift_observed {
            action
                .tool_args
                .insert(WORKSPACE_MANIFEST_DRIFT_KEY.to_string(), Value::Bool(true));
        } else {
            action.tool_args.remove(WORKSPACE_MANIFEST_DRIFT_KEY);
        }
    }

    pub fn last_action_invalidates_verification(&self) -> bool {
        self.actions
            .last()
            .is_some_and(|action| action.substantive_effect || action.workspace_delta_contaminated)
    }

    fn advance_local_workspace_mutation_epoch(&mut self) {
        self.local_workspace_mutation_epoch = self
            .local_workspace_mutation_epoch
            .checked_add(1)
            .expect("local workspace mutation epoch exhausted");
        if let Some(action) = self.actions.last_mut() {
            action.tool_args.insert(
                WORKSPACE_MUTATION_EPOCH_KEY.to_string(),
                Value::from(self.local_workspace_mutation_epoch),
            );
        }
    }

    /// Bind the post-policy, post-routing model message to the exact action.
    /// A lookup by provider call id alone would alias fresh isolated sessions,
    /// because providers may legitimately reuse ids such as `call_0`.
    pub fn record_disclosure(
        &mut self,
        call_identity: &ToolCallIdentity,
        tool_name: &str,
        post_hook_denied: bool,
        routed_payload: &str,
    ) -> bool {
        let Some(action) = self.actions.iter_mut().rev().find(|action| {
            action.tool_name == tool_name && action.call_identity.as_ref() == Some(call_identity)
        }) else {
            return false;
        };

        let file_read = if !post_hook_denied
            && action.status == ActionStatus::Success
            && (tool_name == "file_read"
                || crate::tools::tool_executor::ToolExecutor::is_micro_tool_name(tool_name))
        {
            first_json_value(routed_payload).and_then(parse_file_read_disclosure)
        } else {
            None
        };
        let file_write = if !post_hook_denied
            && action.status == ActionStatus::Success
            && tool_name == "file_write"
        {
            first_json_value(routed_payload).and_then(parse_file_write_disclosure)
        } else {
            None
        };
        action.disclosure = Some(ToolDisclosureReceipt {
            // Queuing a tool message is not proof that the provider consumed
            // it. This flag is committed only after the following provider
            // request succeeds with the exact payload still present after
            // context assembly/compression.
            disclosed_to_model: false,
            result_withheld: post_hook_denied,
            routed_payload_sha256: crate::utils::CryptoUtils::sha256_hex(routed_payload),
            file_read,
            file_write,
        });
        true
    }

    /// Commit queued disclosure receipts after a provider request containing
    /// the exact tool-result messages succeeds. Counts are consumed so two
    /// isolated calls that reuse the same raw provider id and payload cannot
    /// be over-credited when only one occurrence survived context assembly.
    pub fn confirm_disclosures_for_provider_request(
        &mut self,
        visible_tool_result_hashes: &[(String, String)],
    ) {
        let mut available = HashMap::<(String, String), usize>::new();
        for key in visible_tool_result_hashes {
            *available.entry(key.clone()).or_insert(0) += 1;
        }
        for action in self.actions.iter_mut().rev() {
            let (Some(identity), Some(disclosure)) =
                (action.call_identity.as_ref(), action.disclosure.as_mut())
            else {
                continue;
            };
            if disclosure.disclosed_to_model {
                continue;
            }
            let key = (
                identity.provider_call_id.clone(),
                disclosure.routed_payload_sha256.clone(),
            );
            let Some(remaining) = available.get_mut(&key) else {
                continue;
            };
            if *remaining == 0 {
                continue;
            }
            disclosure.disclosed_to_model = true;
            *remaining -= 1;
        }
    }

    /// Build ordered, kernel-only overwrite baseline events for the next
    /// provider-issued tool batch. Only disclosures that actually survived
    /// routing/context assembly and were consumed by a successful provider
    /// request can establish a complete-file baseline. Workspace effects
    /// invalidate older observations before an exact built-in write/edit may
    /// establish its resulting revision.
    pub(crate) fn confirmed_file_overwrite_baseline_events(
        &self,
    ) -> Vec<crate::skill_graph::security::FileOverwriteBaselineEvent> {
        use crate::skill_graph::security::FileOverwriteBaselineEvent;

        let mut events = Vec::new();
        for action in &self.actions {
            let identity = action.call_identity.clone();
            if action.workspace_delta_contaminated
                || (action.substantive_effect && !action.workspace_delta_complete)
            {
                events.push(FileOverwriteBaselineEvent {
                    path: None,
                    content_sha256: None,
                    source_call_identity: identity.clone(),
                });
            } else if action.substantive_effect {
                for change in action
                    .files_created
                    .iter()
                    .chain(action.files_modified.iter())
                    .chain(action.files_removed.iter())
                {
                    events.push(FileOverwriteBaselineEvent {
                        path: Some(change.path.clone()),
                        content_sha256: None,
                        source_call_identity: identity.clone(),
                    });
                }
            }

            let Some(disclosure) = action
                .disclosure
                .as_ref()
                .filter(|receipt| receipt.disclosed_to_model && !receipt.result_withheld)
            else {
                continue;
            };

            if let Some(read) = disclosure.file_read.as_ref().filter(|read| {
                read.offset == 0
                    && read.returned == read.total_lines
                    && read.delivered_line_count == read.returned
                    && !read.partial_line_preview
                    && !read.archived
                    && read.content_sha256.len() == 64
                    && read
                        .content_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit())
            }) {
                events.push(FileOverwriteBaselineEvent {
                    path: Some(read.path.clone()),
                    content_sha256: Some(read.content_sha256.clone()),
                    source_call_identity: identity.clone(),
                });
                continue;
            }

            if action.status == ActionStatus::Success
                && action.substantive_effect
                && matches!(action.tool_name.as_str(), "file_write" | "file_edit")
            {
                if let Some(change) = action
                    .files_created
                    .iter()
                    .chain(action.files_modified.iter())
                    .find(|change| {
                        change.hash.as_ref().is_some_and(|hash| {
                            hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                        })
                    })
                {
                    events.push(FileOverwriteBaselineEvent {
                        path: Some(change.path.clone()),
                        content_sha256: change.hash.clone(),
                        source_call_identity: identity,
                    });
                }
            }
        }
        events
    }

    pub fn mark_last_substantive_effect(&mut self) {
        let was_invalidating = self.last_action_invalidates_verification();
        if let Some(action) = self.actions.last_mut() {
            action.substantive_effect = true;
        }
        if !was_invalidating && self.last_action_invalidates_verification() {
            self.advance_local_workspace_mutation_epoch();
        }
        self.invalidate_prior_verification_receipts();
    }

    /// Bind a direct-filesystem before/after delta to the most recently
    /// recorded shell-like action. The manifest layer has already normalized
    /// every path relative to one stable workspace root. Partial captures are
    /// retained for diagnostics, but `workspace_delta_complete=false` keeps
    /// every downstream ownership gate fail-closed.
    pub fn record_last_workspace_delta(
        &mut self,
        delta: &crate::tools::workspace_monitor::WorkspaceEffectDelta,
        contaminated: bool,
    ) {
        let was_invalidating = self.last_action_invalidates_verification();
        let Some(action) = self.actions.last_mut() else {
            return;
        };
        action.files_created = delta
            .files_created
            .iter()
            .map(|file| FileChange {
                path: file.path.clone(),
                size_bytes: Some(file.size_bytes),
                hash: Some(file.content_sha256.clone()),
            })
            .collect();
        action.files_modified = delta
            .files_modified
            .iter()
            .map(|file| FileChange {
                path: file.path.clone(),
                size_bytes: Some(file.after_size_bytes),
                hash: Some(file.after_content_sha256.clone()),
            })
            .collect();
        action.files_removed = delta
            .files_removed
            .iter()
            .map(|file| FileChange {
                path: file.path.clone(),
                size_bytes: Some(file.size_bytes),
                hash: Some(file.content_sha256.clone()),
            })
            .collect();
        action.directories_created = delta.directories_created.clone();
        action.directories_removed = delta.directories_removed.clone();
        action.workspace_delta_complete = delta.complete;
        action.workspace_delta_contaminated = contaminated;
        action.workspace_delta_sha256 = serde_json::to_string(delta)
            .ok()
            .map(|encoded| format!("sha256:{}", crate::utils::CryptoUtils::sha256_hex(&encoded)));
        if !action.files_created.is_empty()
            || !action.files_modified.is_empty()
            || !action.files_removed.is_empty()
            || !action.directories_created.is_empty()
            || !action.directories_removed.is_empty()
        {
            action.substantive_effect = true;
        }
        let is_invalidating = action.substantive_effect || action.workspace_delta_contaminated;
        if !was_invalidating && is_invalidating {
            self.advance_local_workspace_mutation_epoch();
        }
        if is_invalidating {
            self.invalidate_prior_verification_receipts();
        }
    }

    /// Persist the sole authoritative verifier assessment for the most
    /// recently recorded action. The typed value is stored inside the
    /// existing durable `tool_args` map to preserve compatibility with older
    /// `TrackedAction` struct literals while still serializing the assessment
    /// in checkpoints and handoff evidence.
    pub fn record_last_verification_assessment(&mut self, assessment: VerificationAssessment) {
        let Some(action) = self.actions.last_mut() else {
            return;
        };
        action.verification_attempted = true;
        action.successful_verification = assessment.outcome == VerificationOutcome::Passed;
        if let Ok(value) = serde_json::to_value(assessment) {
            action
                .tool_args
                .insert(VERIFICATION_ASSESSMENT_KEY.to_string(), value);
        } else {
            // Serialization is currently infallible, but failure must never
            // leave a success bit that could outlive its typed evidence.
            action.successful_verification = false;
        }
    }

    /// Return only receipts that still describe the latest known workspace
    /// state. Any later substantive or uncertain effect starts a new
    /// verification epoch and permanently invalidates prior success receipts.
    pub(crate) fn current_successful_verification_receipt_sha256s(&self) -> Vec<String> {
        current_successful_verification_evidence(&self.actions)
            .into_iter()
            .map(|evidence| evidence.receipt_sha256)
            .collect()
    }

    fn invalidate_prior_verification_receipts(&mut self) {
        let Some(effect_index) = self.actions.len().checked_sub(1) else {
            return;
        };
        for action in &mut self.actions[..effect_index] {
            if action.verification_attempted {
                action
                    .tool_args
                    .insert(VERIFICATION_INVALIDATED_KEY.to_string(), Value::Bool(true));
            }
        }
    }

    pub fn mark_last_successful_verification(&mut self) {
        if let Some(action) = self.actions.last_mut() {
            action.successful_verification = true;
        }
    }

    pub fn mark_last_verification_attempt(&mut self) {
        if let Some(action) = self.actions.last_mut() {
            action.verification_attempted = true;
        }
    }

    pub fn success_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| a.status == ActionStatus::Success)
            .count()
    }

    pub fn failure_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| a.status == ActionStatus::Failed)
            .count()
    }

    pub fn files_created_all(&self) -> Vec<&FileChange> {
        self.actions.iter().flat_map(|a| &a.files_created).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

fn first_json_value(payload: &str) -> Option<Value> {
    serde_json::Deserializer::from_str(payload.trim_start())
        .into_iter::<Value>()
        .next()?
        .ok()
}

fn parse_file_read_disclosure(value: Value) -> Option<FileReadDisclosure> {
    let object = value.as_object()?;
    let path = object.get("path")?.as_str()?.trim();
    let content_sha256 = object.get("content_sha256")?.as_str()?.trim();
    // A session-scoped archived file reader may disclose one file in several
    // bounded pages. Its terminal page is a kernel receipt for the contiguous
    // delivery ledger, not a claim that this final message alone contained
    // every line. Convert only an exact whole-file completion into the same
    // overwrite-baseline shape used by an inline file_read.
    if object.get("reader_view").and_then(Value::as_str) == Some("file_lines") {
        let total_lines = object.get("total_lines")?.as_u64()?;
        let archived_start = object.get("archived_source_offset")?.as_u64()?;
        let archived_end = object.get("archived_source_end")?.as_u64()?;
        let complete = object.get("complete").and_then(Value::as_bool) == Some(true);
        let source_has_more = object.get("source_has_more").and_then(Value::as_bool) == Some(true);
        if path.is_empty()
            || content_sha256.len() != 64
            || !content_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || !complete
            || source_has_more
            || archived_start != 0
            || archived_end != total_lines
        {
            return None;
        }
        return Some(FileReadDisclosure {
            path: path.to_string(),
            offset: 0,
            returned: total_lines,
            total_lines,
            delivered_line_count: total_lines,
            content_sha256: content_sha256.to_string(),
            partial_line_preview: false,
            archived: false,
        });
    }
    let offset = object.get("offset")?.as_u64()?;
    let returned = object.get("returned")?.as_u64()?;
    let total_lines = object.get("total_lines")?.as_u64()?;
    let delivered_lines = object.get("lines")?.as_array()?.len() as u64;
    if path.is_empty()
        || content_sha256.is_empty()
        || offset > total_lines
        || returned > total_lines.saturating_sub(offset)
    {
        return None;
    }
    Some(FileReadDisclosure {
        path: path.to_string(),
        offset,
        returned,
        total_lines,
        delivered_line_count: delivered_lines,
        content_sha256: content_sha256.to_string(),
        partial_line_preview: object
            .get("partial_line_preview")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        archived: object
            .get("archived")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn parse_file_write_disclosure(value: Value) -> Option<FileWriteDisclosure> {
    let object = value.as_object()?;
    let path = object.get("path")?.as_str()?.trim();
    let changed = object.get("changed")?.as_bool()?;
    let success = object.get("success")?.as_bool()?;
    let content_sha256 = object.get("content_sha256")?.as_str()?.trim();
    if !success
        || path.is_empty()
        || path.chars().any(char::is_control)
        || content_sha256.len() != 64
        || !content_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return None;
    }
    Some(FileWriteDisclosure {
        path: path.to_string(),
        changed,
        content_sha256: content_sha256.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity(provider_call_id: &str) -> ToolCallIdentity {
        ToolCallIdentity::new("agent-ca-1", "l1-ca-1", "llm-request-1", provider_call_id)
    }

    fn record_consumed_passed_verifier(
        tracker: &mut ActionTracker,
        provider_call_id: &str,
        command: &str,
    ) {
        let args = json!({"command":command});
        let result = json!({"exit_code":0,"stdout":"1 passed in 0.01s"});
        let call = identity(provider_call_id);
        tracker.record_with_identity("bash", &args, &result, 0.01, Some(call.clone()));
        tracker.record_last_verification_assessment(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind: VerificationKind::TestExecution,
            outcome: VerificationOutcome::Passed,
            count: Some(1),
            skipped_count: 0,
            reason: None,
            diagnostic: None,
        });
        assert!(tracker.record_disclosure(&call, "bash", false, &result.to_string()));
        let routed_payload_sha256 = tracker
            .actions
            .last()
            .unwrap()
            .disclosure
            .as_ref()
            .unwrap()
            .routed_payload_sha256
            .clone();
        tracker.confirm_disclosures_for_provider_request(&[(
            call.provider_call_id.clone(),
            routed_payload_sha256,
        )]);
    }

    #[test]
    fn file_edit_uses_the_builtin_path_argument() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "file_edit",
            &json!({"path": "/workspace/app.rs"}),
            &json!({"success": true, "changed": true}),
            0.01,
        );

        assert_eq!(tracker.actions[0].files_modified.len(), 1);
        assert_eq!(
            tracker.actions[0].files_modified[0].path,
            "/workspace/app.rs"
        );
        assert_eq!(
            tracker.actions[0].tool_args.get("path"),
            Some(&Value::String("/workspace/app.rs".to_string()))
        );
    }

    #[test]
    fn failed_writes_are_not_recorded_as_file_changes() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "file_write",
            &json!({"path": "/workspace/app.rs", "content": "invalid"}),
            &json!({"error": "permission denied"}),
            0.01,
        );

        assert_eq!(tracker.actions[0].status, ActionStatus::Failed);
        assert!(tracker.actions[0].files_created.is_empty());
    }

    #[test]
    fn nonzero_command_exit_is_a_failed_action() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "bash",
            &json!({"command": "run acceptance checks"}),
            &json!({"exit_code": 1, "stderr": "failed"}),
            0.01,
        );

        assert_eq!(tracker.actions[0].status, ActionStatus::Failed);
    }

    #[test]
    fn no_op_file_tool_is_not_a_substantive_action() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "file_write",
            &json!({"path": "/workspace/app.rs", "content": "same"}),
            &json!({"success": true, "changed": false, "created": false}),
            0.01,
        );

        assert!(!tracker.actions[0].substantive_effect);
        assert!(tracker.actions[0].files_created.is_empty());
        assert!(tracker.actions[0].files_modified.is_empty());
    }

    #[test]
    fn no_op_full_file_write_can_attest_existing_artifact_after_consumption() {
        let content_sha256 = crate::utils::CryptoUtils::sha256_hex("same");
        let call = identity("call_attest");
        let result = json!({
            "path": "calculator/app.py",
            "success": true,
            "changed": false,
            "created": false,
            "bytes_written": 0,
            "content_sha256": content_sha256,
        });
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record_with_identity(
            "file_write",
            &json!({"path": "calculator/app.py", "content": "same"}),
            &result,
            0.01,
            Some(call.clone()),
        );
        assert!(tracker.actions[0]
            .successful_artifact_attestation()
            .is_none());
        assert!(tracker.record_disclosure(&call, "file_write", false, &result.to_string()));
        assert!(tracker.actions[0]
            .artifact_attestation_close_candidate()
            .is_some());
        assert!(tracker.actions[0]
            .successful_artifact_attestation()
            .is_none());
        let routed_hash = tracker.actions[0]
            .disclosure
            .as_ref()
            .unwrap()
            .routed_payload_sha256
            .clone();
        tracker.confirm_disclosures_for_provider_request(&[(
            call.provider_call_id.clone(),
            routed_hash,
        )]);

        let attestation = tracker.actions[0]
            .successful_artifact_attestation()
            .expect("confirmed no-op full write should attest exact existing contents");
        assert_eq!(attestation.path, "calculator/app.py");
        assert!(attestation.receipt_sha256.starts_with("sha256:"));
        assert_eq!(attestation.receipt_sha256.len(), 71);

        tracker.actions[0]
            .disclosure
            .as_mut()
            .unwrap()
            .result_withheld = true;
        assert!(tracker.actions[0]
            .successful_artifact_attestation()
            .is_none());
    }

    #[test]
    fn confirmed_shell_effect_can_be_recorded_after_execution() {
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record(
            "bash",
            &json!({"command": "generator"}),
            &json!({"exit_code": 0}),
            0.01,
        );
        tracker.mark_last_substantive_effect();
        assert!(tracker.actions[0].substantive_effect);
    }

    #[test]
    fn expected_negative_result_requires_a_real_assertion_wrapper() {
        let raw_child_failure = json!({
            "exit_code": 2,
            "stdout": "",
            "stderr": "invalid expression"
        });
        assert!(tool_result_failed(&raw_child_failure));

        // The shell/test framework owns expectation matching. The tracker
        // deliberately sees only its honest aggregate status: zero after all
        // expected conditions matched, non-zero for every mismatch or setup
        // failure. No model-controlled "expected_failure" flag can launder an
        // arbitrary failed process into a successful kernel receipt.
        let exact_assertion_passed = json!({
            "exit_code": 0,
            "stdout": "expected rejection and diagnostic observed",
            "stderr": ""
        });
        assert!(!tool_result_failed(&exact_assertion_passed));

        let wrapper_setup_failed = json!({
            "exit_code": 0,
            "error": "assertion helper unavailable"
        });
        assert!(tool_result_failed(&wrapper_setup_failed));
    }

    #[test]
    fn file_read_receipt_is_bound_to_post_routing_disclosure() {
        let mut tracker = ActionTracker::new("iri://task/test", "CA");
        let call = identity("call_0");
        tracker.record_with_identity(
            "file_read",
            &json!({"path": "calculator/design.md", "offset": 0, "limit": 20}),
            &json!({
                "path": "calculator/design.md",
                "offset": 0,
                "returned": 20,
                "total_lines": 40,
                "content_sha256": "revision-a",
                "lines": ["not retained"]
            }),
            0.01,
            Some(call.clone()),
        );

        assert!(tracker.record_disclosure(
            &call,
            "file_read",
            false,
            &json!({
                "path": "calculator/design.md",
                "offset": 0,
                "returned": 12,
                "total_lines": 40,
                "content_sha256": "revision-a",
                "partial_line_preview": true,
                "archived": true,
                "lines": ["bounded preview"]
            })
            .to_string(),
        ));

        let queued = tracker.actions[0].disclosure.as_ref().unwrap();
        assert!(!queued.disclosed_to_model);
        tracker.confirm_disclosures_for_provider_request(&[(
            call.provider_call_id.clone(),
            queued.routed_payload_sha256.clone(),
        )]);
        let receipt = tracker.actions[0].disclosure.as_ref().unwrap();
        assert!(receipt.disclosed_to_model);
        let read = receipt.file_read.as_ref().unwrap();
        assert_eq!(read.returned, 12);
        assert!(read.partial_line_preview);
        assert_eq!(read.content_sha256, "revision-a");
        assert_eq!(tracker.actions[0].call_identity, Some(call));
    }

    #[test]
    fn overwrite_baseline_events_require_confirmed_complete_delivery_and_preserve_call_identity() {
        let original_hash = crate::utils::CryptoUtils::sha256_hex("original\n");
        let replacement_hash = crate::utils::CryptoUtils::sha256_hex("replacement\n");
        let read_call = identity("raw-read-call");
        let write_call =
            ToolCallIdentity::new("agent-ca-1", "l1-ca-1", "llm-request-2", "raw-write-call");
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        let read_result = json!({
            "path": "calculator/tests.py",
            "offset": 0,
            "returned": 2,
            "total_lines": 2,
            "content_sha256": original_hash,
            "lines": ["original", ""],
        });
        tracker.record_with_identity(
            "file_read",
            &json!({"path":"calculator/tests.py"}),
            &read_result,
            0.01,
            Some(read_call.clone()),
        );
        assert!(tracker.record_disclosure(
            &read_call,
            "file_read",
            false,
            &read_result.to_string(),
        ));
        assert!(tracker
            .confirmed_file_overwrite_baseline_events()
            .is_empty());
        let read_payload_hash = tracker.actions[0]
            .disclosure
            .as_ref()
            .unwrap()
            .routed_payload_sha256
            .clone();
        tracker.confirm_disclosures_for_provider_request(&[(
            read_call.provider_call_id.clone(),
            read_payload_hash,
        )]);
        let delivered = tracker.confirmed_file_overwrite_baseline_events();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].path.as_deref(), Some("calculator/tests.py"));
        assert_eq!(delivered[0].source_call_identity.as_ref(), Some(&read_call));
        assert_eq!(
            delivered[0]
                .source_call_identity
                .as_ref()
                .unwrap()
                .provider_call_id,
            "raw-read-call"
        );

        let write_result = json!({
            "path": "calculator/tests.py",
            "success": true,
            "changed": true,
            "created": false,
            "bytes_written": 12,
            "content_sha256": replacement_hash,
        });
        tracker.record_with_identity(
            "file_write",
            &json!({"path":"calculator/tests.py","content":"replacement\n"}),
            &write_result,
            0.01,
            Some(write_call.clone()),
        );
        assert!(tracker.record_disclosure(
            &write_call,
            "file_write",
            false,
            &write_result.to_string(),
        ));
        let before_write_delivery = tracker.confirmed_file_overwrite_baseline_events();
        assert!(before_write_delivery
            .iter()
            .any(|event| event.path.as_deref() == Some("calculator/tests.py")
                && event.content_sha256.is_none()));
        assert!(!before_write_delivery.iter().any(|event| {
            event.source_call_identity.as_ref() == Some(&write_call)
                && event.content_sha256.is_some()
        }));

        let write_payload_hash = tracker.actions[1]
            .disclosure
            .as_ref()
            .unwrap()
            .routed_payload_sha256
            .clone();
        tracker.confirm_disclosures_for_provider_request(&[(
            write_call.provider_call_id.clone(),
            write_payload_hash,
        )]);
        let after_write_delivery = tracker.confirmed_file_overwrite_baseline_events();
        let latest = after_write_delivery.last().unwrap();
        assert_eq!(latest.path.as_deref(), Some("calculator/tests.py"));
        assert_eq!(
            latest.content_sha256.as_deref(),
            Some(replacement_hash.as_str())
        );
        assert_eq!(latest.source_call_identity.as_ref(), Some(&write_call));
    }

    #[test]
    fn completed_session_file_reader_establishes_whole_file_overwrite_baseline() {
        let content_sha256 = crate::utils::CryptoUtils::sha256_hex("line one\nline two\n");
        let call = identity("raw-reader-page-call");
        let terminal_page = json!({
            "reader_view": "file_lines",
            "path": "calculator/calculator.py",
            "content_sha256": content_sha256,
            "content": "line two",
            "total_lines": 2,
            "archived_source_offset": 0,
            "archived_source_end": 2,
            "offset": 1,
            "limit": 1,
            "returned": 1,
            "selected_lines": 1,
            "char_offset": 0,
            "next_cursor": null,
            "source_has_more": false,
            "truncated": false,
            "complete": true
        });
        let mut tracker = ActionTracker::new("iri://task/reader-baseline", "DA");
        let reader_name = crate::tools::result_router::ResultRoutingIdentity::new(
            "l1-reader-baseline",
            "raw-source-file-read",
        )
        .reader_name;
        tracker.record_with_identity(
            &reader_name,
            &json!({"offset": 1, "limit": 1}),
            &terminal_page,
            0.01,
            Some(call.clone()),
        );
        assert!(tracker.record_disclosure(&call, &reader_name, false, &terminal_page.to_string(),));
        assert!(tracker
            .confirmed_file_overwrite_baseline_events()
            .is_empty());
        let routed_hash = tracker.actions[0]
            .disclosure
            .as_ref()
            .unwrap()
            .routed_payload_sha256
            .clone();
        tracker.confirm_disclosures_for_provider_request(&[(
            call.provider_call_id.clone(),
            routed_hash,
        )]);
        let events = tracker.confirmed_file_overwrite_baseline_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].path.as_deref(), Some("calculator/calculator.py"));
        assert_eq!(
            events[0].content_sha256.as_deref(),
            Some(content_sha256.as_str())
        );
        assert_eq!(events[0].source_call_identity.as_ref(), Some(&call));
    }

    #[test]
    fn partial_or_archived_file_delivery_never_authorizes_overwrite() {
        let hash = crate::utils::CryptoUtils::sha256_hex("large file");
        for (returned, total_lines, archived, partial_line_preview) in [
            (2, 10, false, false),
            (10, 10, true, false),
            (10, 10, false, true),
        ] {
            let call = identity(&format!("raw-{returned}-{archived}-{partial_line_preview}"));
            let result = json!({
                "path": "calculator/large.py",
                "offset": 0,
                "returned": returned,
                "total_lines": total_lines,
                "content_sha256": hash,
                "archived": archived,
                "partial_line_preview": partial_line_preview,
                "lines": ["bounded"],
            });
            let mut tracker = ActionTracker::new("iri://task/test", "DA");
            tracker.record_with_identity(
                "file_read",
                &json!({"path":"calculator/large.py"}),
                &result,
                0.01,
                Some(call.clone()),
            );
            assert!(tracker.record_disclosure(&call, "file_read", false, &result.to_string(),));
            let routed_hash = tracker.actions[0]
                .disclosure
                .as_ref()
                .unwrap()
                .routed_payload_sha256
                .clone();
            tracker.confirm_disclosures_for_provider_request(&[(
                call.provider_call_id.clone(),
                routed_hash,
            )]);
            assert!(tracker
                .confirmed_file_overwrite_baseline_events()
                .is_empty());
        }
    }

    #[test]
    fn withheld_result_never_becomes_read_evidence() {
        let mut tracker = ActionTracker::new("iri://task/test", "CA");
        let call = identity("call_0");
        tracker.record_with_identity(
            "file_read",
            &json!({"path": "calculator/design.md"}),
            &json!({
                "path": "calculator/design.md",
                "offset": 0,
                "returned": 40,
                "total_lines": 40,
                "content_sha256": "revision-a"
            }),
            0.01,
            Some(call.clone()),
        );
        assert!(tracker.record_disclosure(
            &call,
            "file_read",
            true,
            r#"{"error":"result withheld"}"#,
        ));

        let queued = tracker.actions[0].disclosure.as_ref().unwrap();
        tracker.confirm_disclosures_for_provider_request(&[(
            call.provider_call_id.clone(),
            queued.routed_payload_sha256.clone(),
        )]);
        let receipt = tracker.actions[0].disclosure.as_ref().unwrap();
        assert!(receipt.disclosed_to_model);
        assert!(receipt.result_withheld);
        assert!(receipt.file_read.is_none());
    }

    #[test]
    fn successful_verification_receipt_requires_full_identity_and_consumed_unredacted_result() {
        let args = json!({"command":"python3 -m unittest -q"});
        let result = json!({"exit_code":0,"stdout":"OK"});
        let call = identity("call_0");
        let mut tracker = ActionTracker::new("iri://task/test", "DA");
        tracker.record_with_identity("bash", &args, &result, 0.01, Some(call.clone()));
        tracker.record_last_verification_assessment(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind: VerificationKind::TestExecution,
            outcome: VerificationOutcome::Passed,
            count: Some(42),
            skipped_count: 0,
            reason: None,
            diagnostic: None,
        });
        assert!(tracker.actions[0]
            .successful_verification_receipt_sha256()
            .is_none());

        let visible = result.to_string();
        assert!(tracker.record_disclosure(&call, "bash", false, &visible));
        let payload_hash = tracker.actions[0]
            .disclosure
            .as_ref()
            .unwrap()
            .routed_payload_sha256
            .clone();
        tracker.confirm_disclosures_for_provider_request(&[(
            call.provider_call_id.clone(),
            payload_hash,
        )]);
        let receipt = tracker.actions[0]
            .successful_verification_receipt_sha256()
            .expect("confirmed verifier receipt");
        assert!(receipt.starts_with("sha256:"));
        assert_eq!(receipt.len(), 71);

        tracker.record_last_verification_assessment(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind: VerificationKind::TestExecution,
            outcome: VerificationOutcome::Passed,
            count: Some(43),
            skipped_count: 0,
            reason: None,
            diagnostic: None,
        });
        let changed_count_receipt = tracker.actions[0]
            .successful_verification_receipt_sha256()
            .expect("updated typed receipt");
        assert_ne!(receipt, changed_count_receipt);

        tracker.record_last_verification_assessment(VerificationAssessment {
            parser_version: "foreign-parser/v9".to_string(),
            kind: VerificationKind::Build,
            outcome: VerificationOutcome::Passed,
            count: Some(1),
            skipped_count: 0,
            reason: None,
            diagnostic: None,
        });
        assert!(tracker.actions[0]
            .successful_verification_receipt_sha256()
            .is_none());
        tracker.record_last_verification_assessment(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind: VerificationKind::TestExecution,
            outcome: VerificationOutcome::Inconclusive,
            count: Some(0),
            skipped_count: 0,
            reason: Some("zero_tests_executed".to_string()),
            diagnostic: None,
        });
        assert!(tracker.actions[0]
            .successful_verification_receipt_sha256()
            .is_none());
        tracker.record_last_verification_assessment(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind: VerificationKind::TestExecution,
            outcome: VerificationOutcome::Passed,
            count: Some(43),
            skipped_count: 0,
            reason: None,
            diagnostic: None,
        });

        tracker.actions[0]
            .disclosure
            .as_mut()
            .unwrap()
            .result_withheld = true;
        assert!(tracker.actions[0]
            .successful_verification_receipt_sha256()
            .is_none());
        tracker.actions[0]
            .disclosure
            .as_mut()
            .unwrap()
            .result_withheld = false;
        tracker.actions[0].substantive_effect = true;
        assert!(tracker.actions[0]
            .successful_verification_receipt_sha256()
            .is_none());
    }

    #[test]
    fn verification_freshness_uses_coordinator_sequence_not_child_result_order() {
        use crate::tools::tool_executor::WorkspaceSettlementStamp;

        let mut mutation = ActionTracker::new("iri://task/mutation-child", "DA");
        mutation.record(
            "file_write",
            &json!({"path":"calculator/app.py","content":"fixed"}),
            &json!({"success":true,"changed":true,"created":false,"path":"calculator/app.py"}),
            0.01,
        );
        mutation.record_last_workspace_settlement(&WorkspaceSettlementStamp {
            coordinator_id: "shared-test-coordinator".to_string(),
            settlement_sequence: 1,
            mutation_epoch: 1,
            manifest_sha256: None,
            manifest_drift_observed: false,
        });

        let mut verifier = ActionTracker::new("iri://task/verifier-child", "DA");
        record_consumed_passed_verifier(&mut verifier, "call-fresh", "python -m pytest -q");
        verifier.record_last_workspace_settlement(&WorkspaceSettlementStamp {
            coordinator_id: "shared-test-coordinator".to_string(),
            settlement_sequence: 2,
            mutation_epoch: 1,
            manifest_sha256: None,
            manifest_drift_observed: false,
        });

        // Parallel child TaskResults may arrive in this reverse order even
        // though the mutation settled before the verifier.
        let reversed = vec![verifier.actions[0].clone(), mutation.actions[0].clone()];
        let evidence = current_successful_verification_evidence(&reversed);
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].workspace_settlement_sequence, 2);
        assert_eq!(
            evidence[0].invocation_sha256,
            verification_invocation_sha256("bash", &json!({"command":"python -m pytest -q"}))
        );

        let mut stale_verifier = ActionTracker::new("iri://task/stale-child", "DA");
        record_consumed_passed_verifier(&mut stale_verifier, "call-stale", "python -m pytest -q");
        stale_verifier.record_last_workspace_settlement(&WorkspaceSettlementStamp {
            coordinator_id: "another-shared-coordinator".to_string(),
            settlement_sequence: 1,
            mutation_epoch: 0,
            manifest_sha256: None,
            manifest_drift_observed: false,
        });
        let mut later_mutation = ActionTracker::new("iri://task/later-mutation", "DA");
        later_mutation.record(
            "file_write",
            &json!({"path":"calculator/app.py","content":"changed-again"}),
            &json!({"success":true,"changed":true,"created":false,"path":"calculator/app.py"}),
            0.01,
        );
        later_mutation.record_last_workspace_settlement(&WorkspaceSettlementStamp {
            coordinator_id: "another-shared-coordinator".to_string(),
            settlement_sequence: 2,
            mutation_epoch: 1,
            manifest_sha256: None,
            manifest_drift_observed: false,
        });
        assert!(current_successful_verification_evidence(&[
            later_mutation.actions[0].clone(),
            stale_verifier.actions[0].clone(),
        ])
        .is_empty());
    }

    #[test]
    fn cross_runtime_receipt_is_rejected_but_current_runtime_reverification_refreshes_it() {
        use crate::tools::tool_executor::WorkspaceSettlementStamp;

        let mut old_verifier = ActionTracker::new("iri://task/old-verifier", "DA");
        record_consumed_passed_verifier(&mut old_verifier, "call-old", "python -m pytest -q");
        old_verifier.actions[0].tool_args.insert(
            RUNTIME_EPOCH_ID_KEY.to_string(),
            Value::String("runtime:previous-process:untrusted-order".to_string()),
        );

        let mut current_mutation = ActionTracker::new("iri://task/current-mutation", "DA");
        current_mutation.record(
            "file_write",
            &json!({"path":"calculator/app.py","content":"current"}),
            &json!({"success":true,"changed":true,"created":false,"path":"calculator/app.py"}),
            0.01,
        );
        current_mutation.record_last_workspace_settlement(&WorkspaceSettlementStamp {
            coordinator_id: "current-coordinator".to_string(),
            settlement_sequence: 1,
            mutation_epoch: 1,
            manifest_sha256: None,
            manifest_drift_observed: false,
        });
        assert!(current_successful_verification_evidence(&[
            old_verifier.actions[0].clone(),
            current_mutation.actions[0].clone(),
        ])
        .is_empty());

        let mut refreshed = ActionTracker::new("iri://task/current-verifier", "CA");
        record_consumed_passed_verifier(&mut refreshed, "call-new", "python -m pytest -q");
        refreshed.record_last_workspace_settlement(&WorkspaceSettlementStamp {
            coordinator_id: "current-coordinator".to_string(),
            settlement_sequence: 2,
            mutation_epoch: 1,
            manifest_sha256: None,
            manifest_drift_observed: false,
        });
        let actions = vec![
            refreshed.actions[0].clone(),
            old_verifier.actions[0].clone(),
            current_mutation.actions[0].clone(),
        ];
        assert_eq!(current_successful_verification_evidence(&actions).len(), 1);
    }

    #[test]
    fn later_same_kind_failure_supersedes_an_earlier_pass() {
        use crate::tools::tool_executor::WorkspaceSettlementStamp;

        let command = "python -m pytest -q";
        let mut passed = ActionTracker::new("iri://task/pass-child", "DA");
        record_consumed_passed_verifier(&mut passed, "call-pass", command);
        passed.record_last_workspace_settlement(&WorkspaceSettlementStamp {
            coordinator_id: "flaky-coordinator".to_string(),
            settlement_sequence: 1,
            mutation_epoch: 0,
            manifest_sha256: None,
            manifest_drift_observed: false,
        });

        let mut failed = ActionTracker::new("iri://task/fail-child", "CA");
        let failed_args = json!({"command":command});
        let failed_result = json!({"exit_code":1,"stdout":"1 failed in 0.01s"});
        let failed_call = identity("call-fail");
        failed.record_with_identity(
            "bash",
            &failed_args,
            &failed_result,
            0.01,
            Some(failed_call.clone()),
        );
        failed.record_last_verification_assessment(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind: VerificationKind::TestExecution,
            outcome: VerificationOutcome::Failed,
            count: Some(1),
            skipped_count: 0,
            reason: None,
            diagnostic: None,
        });
        assert!(failed.record_disclosure(&failed_call, "bash", false, &failed_result.to_string(),));
        let failed_payload_sha256 = failed.actions[0]
            .disclosure
            .as_ref()
            .unwrap()
            .routed_payload_sha256
            .clone();
        failed.confirm_disclosures_for_provider_request(&[(
            failed_call.provider_call_id.clone(),
            failed_payload_sha256,
        )]);
        failed.record_last_workspace_settlement(&WorkspaceSettlementStamp {
            coordinator_id: "flaky-coordinator".to_string(),
            settlement_sequence: 2,
            mutation_epoch: 0,
            manifest_sha256: None,
            manifest_drift_observed: false,
        });
        assert!(current_successful_verification_evidence(&[
            passed.actions[0].clone(),
            failed.actions[0].clone(),
        ])
        .is_empty());
        let attempts = current_typed_verification_attempts(&[
            passed.actions[0].clone(),
            failed.actions[0].clone(),
        ]);
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].assessment.outcome, VerificationOutcome::Failed);

        failed.record_last_verification_assessment(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind: VerificationKind::TestExecution,
            outcome: VerificationOutcome::Inconclusive,
            count: Some(0),
            skipped_count: 1,
            reason: Some("all_tests_skipped".to_string()),
            diagnostic: None,
        });
        assert!(current_successful_verification_evidence(&[
            failed.actions[0].clone(),
            passed.actions[0].clone(),
        ])
        .is_empty());
        let attempts = current_typed_verification_attempts(&[
            failed.actions[0].clone(),
            passed.actions[0].clone(),
        ]);
        assert_eq!(attempts.len(), 1);
        assert_eq!(
            attempts[0].assessment.outcome,
            VerificationOutcome::Inconclusive
        );

        // A later failure of the same verifier kind is counter-evidence even
        // when its normalized invocation differs. The kernel cannot assume
        // that two test commands cover disjoint acceptance criteria.
        failed.record_last_verification_assessment(VerificationAssessment {
            parser_version: VERIFICATION_ASSESSMENT_PARSER_VERSION.to_string(),
            kind: VerificationKind::TestExecution,
            outcome: VerificationOutcome::Failed,
            count: Some(1),
            skipped_count: 0,
            reason: None,
            diagnostic: None,
        });
        failed.actions[0].tool_args.insert(
            VERIFICATION_INVOCATION_SHA256_KEY.to_string(),
            Value::String(verification_invocation_sha256(
                "bash",
                &json!({"command":"python -m pytest -q integration"}),
            )),
        );
        assert!(current_successful_verification_evidence(&[
            failed.actions[0].clone(),
            passed.actions[0].clone(),
        ])
        .is_empty());
    }

    #[test]
    fn disclosure_lookup_uses_full_identity_not_raw_provider_id() {
        let mut tracker = ActionTracker::new("iri://task/test", "CA");
        let first = identity("call_0");
        let second = ToolCallIdentity::new("agent-ca-2", "l1-ca-2", "llm-request-2", "call_0");
        for call in [&first, &second] {
            tracker.record_with_identity(
                "file_read",
                &json!({"path": "calculator/design.md"}),
                &json!({"path": "calculator/design.md", "offset": 0, "returned": 1, "total_lines": 1, "content_sha256": "revision-a"}),
                0.01,
                Some(call.clone()),
            );
        }

        assert!(tracker.record_disclosure(
            &first,
            "file_read",
            false,
            r#"{"path":"calculator/design.md","offset":0,"returned":1,"total_lines":1,"content_sha256":"revision-a"}
IRI: iri://result/not-part-of-json"#,
        ));
        let first_hash = tracker.actions[0]
            .disclosure
            .as_ref()
            .unwrap()
            .routed_payload_sha256
            .clone();
        tracker.confirm_disclosures_for_provider_request(&[(
            first.provider_call_id.clone(),
            first_hash,
        )]);
        assert!(
            tracker.actions[0]
                .disclosure
                .as_ref()
                .unwrap()
                .disclosed_to_model
        );
        assert!(tracker.actions[1].disclosure.is_none());
        assert_eq!(first.provider_call_id, "call_0");
        assert_eq!(second.provider_call_id, "call_0");
    }
}
