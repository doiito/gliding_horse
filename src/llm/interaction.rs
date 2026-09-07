//! One observable lifecycle for every model interaction.
//!
//! The provider gateway remains responsible for transport, retries and API
//! adaptation.  This facade adds stable correlation and privacy-preserving
//! telemetry for callers in the agent, supervisor and background subsystems.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::broadcast;
use tracing::{debug, warn};
use uuid::Uuid;

use once_cell::sync::Lazy;

use crate::core::context_model::{
    AgentSpecSourceRecord, ContextDisposition, ContextFragmentKind, EffectiveContextManifest,
    GeneratedAgentSpec,
};
use crate::gateway::unified_gateway::{
    ChatCompletionResponse, ChatMessage, GatewayCallMetadata, LlmRequestOptions, UnifiedGateway,
};
use crate::llm::sse::{SseError, SseErrorKind};
use crate::llm::stream_types::{StreamEvent, StreamResponse, Usage as StreamUsage};
use crate::tools::hooks::{HookContext, HookControl, HookManager, HookPoint};
use crate::CoreError;

pub const LLM_INTERACTION_SCHEMA_VERSION: u16 = 6;

/// One interaction plane per gateway instance. Business components that
/// receive the same `Arc<UnifiedGateway>` therefore share hooks, subscribers
/// and EventBus forwarding instead of creating isolated observability islands.
static SHARED_INTERACTION_SERVICES: Lazy<
    parking_lot::Mutex<std::collections::HashMap<usize, Weak<LlmInteractionService>>>,
> = Lazy::new(|| parking_lot::Mutex::new(std::collections::HashMap::new()));

/// Idle callers that do not own a task-lifetime lease are retained only up to
/// this diagnostic bound. Active SA task scopes are never evicted.
const MAX_IDLE_ACCOUNTING_SCOPES: usize = 4_096;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LlmInteractionPhase {
    Assembled,
    BeforeDispatch,
    FirstToken,
    Completed,
    Failed,
    Cancelled,
}

/// Payload-free proof of the compiled Agent specification that owned an LLM
/// interaction. It deliberately records hashes and provenance only: the
/// model-authored `agent.md`, objective, criteria and context fragment text
/// remain in their access-controlled stores rather than routine telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentSpecMaterializationReceipt {
    pub schema_version: u16,
    pub spec_schema_version: String,
    pub role: crate::core::agent_instance::AgentRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_id: Option<String>,
    pub source: AgentSpecSourceRecord,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_md_sha256: Option<String>,
    pub agent_md_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_manifest_hash: Option<String>,
}

impl From<&GeneratedAgentSpec> for AgentSpecMaterializationReceipt {
    fn from(spec: &GeneratedAgentSpec) -> Self {
        Self {
            schema_version: 1,
            spec_schema_version: spec.schema_version.clone(),
            role: spec.role,
            step_id: spec.step_id.clone(),
            source: spec.source.clone(),
            agent_md_sha256: spec.agent_md_sha256.clone(),
            agent_md_chars: spec.agent_md_chars,
            context_manifest_hash: spec
                .context_manifest
                .as_ref()
                .map(|manifest| manifest.effective_sha256.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmInteractionScope {
    pub interaction_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_interaction_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_iri: Option<String>,
    /// Stable accounting scope for a complete business task. A BizAgent child
    /// has its own `task_iri` for isolation while retaining the parent's task
    /// here so budgets include decomposition, all children and aggregation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_scope_iri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycle_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_manifest_hash: Option<String>,
    /// Metadata-only provenance of the concrete compiled `agent.md` instance
    /// that owns this interaction. This is distinct from the interaction that
    /// originally authored the plan and never contains the markdown payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_spec_receipt: Option<AgentSpecMaterializationReceipt>,
    /// The complete role-context receipt is carried only in process and is
    /// copied onto the `Assembled` event. Keeping it out of the scope avoids
    /// repeating a potentially large metadata structure on every lifecycle
    /// event while preserving the stable hash for correlation.
    #[serde(skip)]
    context_manifest_receipt: Option<EffectiveContextManifest>,
}

impl LlmInteractionScope {
    pub fn new(stage: impl Into<String>) -> Self {
        Self {
            interaction_id: format!("llm_{}", Uuid::new_v4().hyphenated()),
            parent_interaction_id: None,
            task_iri: None,
            usage_scope_iri: None,
            cycle_id: None,
            agent_id: None,
            role: None,
            stage: stage.into(),
            context_manifest_hash: None,
            agent_spec_receipt: None,
            context_manifest_receipt: None,
        }
    }

    pub fn with_interaction_id(mut self, interaction_id: impl Into<String>) -> Self {
        self.interaction_id = interaction_id.into();
        self
    }

    pub fn with_parent(mut self, interaction_id: impl Into<String>) -> Self {
        self.parent_interaction_id = Some(interaction_id.into());
        self
    }

    pub fn with_task(mut self, task_iri: impl Into<String>) -> Self {
        let task_iri = task_iri.into();
        if self.usage_scope_iri.is_none() {
            self.usage_scope_iri = Some(task_iri.clone());
        }
        self.task_iri = Some(task_iri);
        self
    }

    pub fn with_usage_scope(mut self, usage_scope_iri: impl Into<String>) -> Self {
        self.usage_scope_iri = Some(usage_scope_iri.into());
        self
    }

    pub fn with_cycle(mut self, cycle_id: impl Into<String>) -> Self {
        self.cycle_id = Some(cycle_id.into());
        self
    }

    pub fn with_agent(mut self, agent_id: impl Into<String>, role: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self.role = Some(role.into());
        self
    }

    pub fn with_context_manifest_hash(mut self, hash: impl Into<String>) -> Self {
        self.context_manifest_hash = Some(hash.into());
        self
    }

    /// Attach the metadata-only receipt for the complete typed dispatch
    /// context (initial role context, runtime fragments, and provider protocol
    /// receipts). Fragment payloads are intentionally absent from
    /// `EffectiveContextManifest`.
    pub fn with_context_manifest(mut self, manifest: &EffectiveContextManifest) -> Self {
        self.context_manifest_hash = Some(manifest.effective_sha256.clone());
        self.context_manifest_receipt = Some(manifest.clone());
        self
    }

    /// Correlate a provider call with its already-materialized Agent spec.
    /// `GeneratedAgentSpec` contains business fields, so only the compact
    /// receipt above is copied into the observable lifecycle.
    pub fn with_agent_spec(mut self, spec: &GeneratedAgentSpec) -> Self {
        self.agent_spec_receipt = Some(AgentSpecMaterializationReceipt::from(spec));
        self
    }
}

/// Metadata for one message that was actually dispatched to the provider.
/// It complements the typed dispatch manifest with the exact provider order
/// and wire-level message hashes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmMessageReceipt {
    pub index: usize,
    pub role: String,
    pub kind: ContextFragmentKind,
    pub content_chars: usize,
    pub message_sha256: String,
    pub tool_call_count: usize,
    pub has_reasoning: bool,
}

/// Message counts and character volume for one of the six context trust
/// classes. Character counts use Unicode scalar values, matching context
/// assembly budgets. This compact aggregate retains neither request text nor
/// hashes; per-message SHA-256 receipts remain restricted to the detailed
/// `Assembled` event and are never rendered by the normal TUI.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmContextClassReceipt {
    pub messages: u64,
    pub chars: u64,
}

impl LlmContextClassReceipt {
    fn saturating_add_assign(&mut self, other: Self) {
        self.messages = self.messages.saturating_add(other.messages);
        self.chars = self.chars.saturating_add(other.chars);
    }

    fn saturating_sub(self, baseline: Self) -> Self {
        Self {
            messages: self.messages.saturating_sub(baseline.messages),
            chars: self.chars.saturating_sub(baseline.chars),
        }
    }
}

/// Fixed, exhaustive breakdown of the six supported context trust classes.
/// Named fields keep the telemetry schema stable and prevent an omitted class
/// from being mistaken for a classifier implementation gap.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmContextKindReceipt {
    pub authoritative_instruction: LlmContextClassReceipt,
    pub user_input: LlmContextClassReceipt,
    pub verified_evidence: LlmContextClassReceipt,
    pub unverified_retrieval: LlmContextClassReceipt,
    pub tool_output: LlmContextClassReceipt,
    pub model_history: LlmContextClassReceipt,
}

impl LlmContextKindReceipt {
    fn get_mut(&mut self, kind: ContextFragmentKind) -> &mut LlmContextClassReceipt {
        match kind {
            ContextFragmentKind::AuthoritativeInstruction => &mut self.authoritative_instruction,
            ContextFragmentKind::UserInput => &mut self.user_input,
            ContextFragmentKind::VerifiedEvidence => &mut self.verified_evidence,
            ContextFragmentKind::UnverifiedRetrieval => &mut self.unverified_retrieval,
            ContextFragmentKind::ToolOutput => &mut self.tool_output,
            ContextFragmentKind::ModelHistory => &mut self.model_history,
        }
    }

    fn saturating_add_assign(&mut self, other: Self) {
        self.authoritative_instruction
            .saturating_add_assign(other.authoritative_instruction);
        self.user_input.saturating_add_assign(other.user_input);
        self.verified_evidence
            .saturating_add_assign(other.verified_evidence);
        self.unverified_retrieval
            .saturating_add_assign(other.unverified_retrieval);
        self.tool_output.saturating_add_assign(other.tool_output);
        self.model_history
            .saturating_add_assign(other.model_history);
    }

    fn saturating_sub(self, baseline: Self) -> Self {
        Self {
            authoritative_instruction: self
                .authoritative_instruction
                .saturating_sub(baseline.authoritative_instruction),
            user_input: self.user_input.saturating_sub(baseline.user_input),
            verified_evidence: self
                .verified_evidence
                .saturating_sub(baseline.verified_evidence),
            unverified_retrieval: self
                .unverified_retrieval
                .saturating_sub(baseline.unverified_retrieval),
            tool_output: self.tool_output.saturating_sub(baseline.tool_output),
            model_history: self.model_history.saturating_sub(baseline.model_history),
        }
    }

    fn totals(self) -> LlmContextClassReceipt {
        let classes = [
            self.authoritative_instruction,
            self.user_input,
            self.verified_evidence,
            self.unverified_retrieval,
            self.tool_output,
            self.model_history,
        ];
        classes
            .into_iter()
            .fold(LlmContextClassReceipt::default(), |mut total, class| {
                total.saturating_add_assign(class);
                total
            })
    }
}

/// Counts every possible role-context admission outcome. A manifest entry has
/// exactly one disposition, so these counters can be audited against
/// `manifest_fragments` without retaining fragment identifiers or hashes.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmContextDispositionReceipt {
    pub included: u64,
    pub truncated: u64,
    pub dropped_expired: u64,
    pub dropped_scope: u64,
    pub dropped_role_policy: u64,
    pub dropped_budget: u64,
}

impl LlmContextDispositionReceipt {
    pub fn dropped(self) -> u64 {
        self.dropped_expired
            .saturating_add(self.dropped_scope)
            .saturating_add(self.dropped_role_policy)
            .saturating_add(self.dropped_budget)
    }

    pub fn total(self) -> u64 {
        self.included
            .saturating_add(self.truncated)
            .saturating_add(self.dropped())
    }

    fn saturating_add_assign(&mut self, other: Self) {
        self.included = self.included.saturating_add(other.included);
        self.truncated = self.truncated.saturating_add(other.truncated);
        self.dropped_expired = self.dropped_expired.saturating_add(other.dropped_expired);
        self.dropped_scope = self.dropped_scope.saturating_add(other.dropped_scope);
        self.dropped_role_policy = self
            .dropped_role_policy
            .saturating_add(other.dropped_role_policy);
        self.dropped_budget = self.dropped_budget.saturating_add(other.dropped_budget);
    }

    fn saturating_sub(self, baseline: Self) -> Self {
        Self {
            included: self.included.saturating_sub(baseline.included),
            truncated: self.truncated.saturating_sub(baseline.truncated),
            dropped_expired: self
                .dropped_expired
                .saturating_sub(baseline.dropped_expired),
            dropped_scope: self.dropped_scope.saturating_sub(baseline.dropped_scope),
            dropped_role_policy: self
                .dropped_role_policy
                .saturating_sub(baseline.dropped_role_policy),
            dropped_budget: self.dropped_budget.saturating_sub(baseline.dropped_budget),
        }
    }
}

/// Compact metadata-only description of the exact request context. This is
/// repeated on every lifecycle event so consumers can render STARTED without
/// joining it to an ASSEMBLED event that may have been intentionally hidden or
/// lost under broadcast backpressure.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmContextReceipt {
    pub request_messages: u64,
    pub request_chars: u64,
    pub by_kind: LlmContextKindReceipt,
    pub manifest_fragments: u64,
    pub dispositions: LlmContextDispositionReceipt,
    pub required_budget_exceeded: bool,
}

impl LlmContextReceipt {
    fn from_request(
        messages: &[LlmMessageReceipt],
        manifest: Option<&EffectiveContextManifest>,
    ) -> Self {
        let mut by_kind = LlmContextKindReceipt::default();
        for message in messages {
            let class = by_kind.get_mut(message.kind);
            class.messages = class.messages.saturating_add(1);
            class.chars = class
                .chars
                .saturating_add(u64::try_from(message.content_chars).unwrap_or(u64::MAX));
        }
        let mut dispositions = LlmContextDispositionReceipt::default();
        if let Some(manifest) = manifest {
            for entry in &manifest.entries {
                let target = match entry.disposition {
                    ContextDisposition::Included => &mut dispositions.included,
                    ContextDisposition::Truncated => &mut dispositions.truncated,
                    ContextDisposition::DroppedExpired => &mut dispositions.dropped_expired,
                    ContextDisposition::DroppedScope => &mut dispositions.dropped_scope,
                    ContextDisposition::DroppedByRolePolicy => {
                        &mut dispositions.dropped_role_policy
                    }
                    ContextDisposition::DroppedByBudget => &mut dispositions.dropped_budget,
                };
                *target = target.saturating_add(1);
            }
        }
        let totals = by_kind.totals();
        Self {
            request_messages: totals.messages,
            request_chars: totals.chars,
            by_kind,
            manifest_fragments: manifest
                .map(|manifest| u64::try_from(manifest.entries.len()).unwrap_or(u64::MAX))
                .unwrap_or_default(),
            dispositions,
            required_budget_exceeded: manifest
                .is_some_and(|manifest| manifest.required_budget_exceeded),
        }
    }
}

/// Monotonic, task-scope aggregate of context that reached a real provider
/// dispatch. Snapshot differences are safe for concurrent child agents and
/// contain only counts, never task text, message hashes, or agent identity.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmContextQualitySnapshot {
    pub dispatch_count: u64,
    pub request_messages: u64,
    pub request_chars: u64,
    pub by_kind: LlmContextKindReceipt,
    pub manifest_fragments: u64,
    pub dispositions: LlmContextDispositionReceipt,
    pub required_budget_overflow_dispatches: u64,
}

impl LlmContextQualitySnapshot {
    fn record_dispatch(&mut self, receipt: LlmContextReceipt) {
        self.dispatch_count = self.dispatch_count.saturating_add(1);
        self.request_messages = self
            .request_messages
            .saturating_add(receipt.request_messages);
        self.request_chars = self.request_chars.saturating_add(receipt.request_chars);
        self.by_kind.saturating_add_assign(receipt.by_kind);
        self.manifest_fragments = self
            .manifest_fragments
            .saturating_add(receipt.manifest_fragments);
        self.dispositions
            .saturating_add_assign(receipt.dispositions);
        if receipt.required_budget_exceeded {
            self.required_budget_overflow_dispatches =
                self.required_budget_overflow_dispatches.saturating_add(1);
        }
    }

    pub fn saturating_sub(self, baseline: Self) -> Self {
        Self {
            dispatch_count: self.dispatch_count.saturating_sub(baseline.dispatch_count),
            request_messages: self
                .request_messages
                .saturating_sub(baseline.request_messages),
            request_chars: self.request_chars.saturating_sub(baseline.request_chars),
            by_kind: self.by_kind.saturating_sub(baseline.by_kind),
            manifest_fragments: self
                .manifest_fragments
                .saturating_sub(baseline.manifest_fragments),
            dispositions: self.dispositions.saturating_sub(baseline.dispositions),
            required_budget_overflow_dispatches: self
                .required_budget_overflow_dispatches
                .saturating_sub(baseline.required_budget_overflow_dispatches),
        }
    }

    pub fn is_consistent(self) -> bool {
        let message_totals = self.by_kind.totals();
        message_totals.messages == self.request_messages
            && message_totals.chars == self.request_chars
            && self.dispositions.total() == self.manifest_fragments
            && self.required_budget_overflow_dispatches <= self.dispatch_count
    }

    pub fn drop_rate(self) -> f64 {
        ratio(self.dispositions.dropped(), self.manifest_fragments)
    }

    pub fn truncate_rate(self) -> f64 {
        ratio(self.dispositions.truncated, self.manifest_fragments)
    }

    pub fn expired_rate(self) -> f64 {
        ratio(self.dispositions.dropped_expired, self.manifest_fragments)
    }

    pub fn required_budget_overflow_rate(self) -> f64 {
        ratio(
            self.required_budget_overflow_dispatches,
            self.dispatch_count,
        )
    }

    pub fn chars_per_dispatch(self) -> f64 {
        ratio(self.request_chars, self.dispatch_count)
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Metadata-only event. Request messages, model responses, reasoning and tool
/// arguments are deliberately excluded. Payload capture remains the explicit,
/// access-controlled responsibility of the task execution journal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmInteractionEvent {
    pub schema_version: u16,
    pub phase: LlmInteractionPhase,
    pub emitted_at_ms: i64,
    pub scope: LlmInteractionScope,
    pub model: String,
    pub streaming: bool,
    /// Validated provider reasoning policy. Payload text and provider-private
    /// chain-of-thought are never included in this metadata-only event.
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Gateway-normalized effective reasoning policy. An unsupported request
    /// is omitted here exactly as it is omitted from the provider wire body.
    pub reasoning_effort: Option<String>,
    pub message_count: usize,
    pub advertised_tool_names: Vec<String>,
    /// Compact metadata available on every phase, including STARTED.
    #[serde(default)]
    pub context_receipt: LlmContextReceipt,
    /// Present only on `Assembled`; contains hashes/sizes/classes, never text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub request_message_receipts: Vec<LlmMessageReceipt>,
    /// Present only on `Assembled`; fragment contents are not part of this
    /// structure, so normal diagnostics remain metadata-first.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role_context_manifest: Option<EffectiveContextManifest>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    /// Distinct tool names are an inventory, not a call counter. Repeated
    /// invocations of the same tool deliberately appear only once here.
    pub response_tool_names: Vec<String>,
    /// Exact number of structured tool calls in the selected model response.
    /// This is separate from `response_tool_names.len()` because one tool can
    /// be invoked more than once in a single response.
    #[serde(default)]
    pub response_tool_call_count: usize,
    pub request_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_schema_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    /// Provider-reported usage accumulated across every actual dispatch in
    /// this logical interaction (including response-hook retries).
    pub billed_prompt_tokens: u64,
    pub billed_completion_tokens: u64,
    pub model_dispatch_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choices_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_empty: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_class: Option<String>,
    /// Safe transport metadata only; provider response bodies and request
    /// payloads remain excluded from this normal observability event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gateway: Option<GatewayCallMetadata>,
}

#[derive(Debug, Clone)]
struct RequestMetadata {
    scope: LlmInteractionScope,
    model: String,
    streaming: bool,
    reasoning_effort: Option<String>,
    message_count: usize,
    advertised_tool_names: Vec<String>,
    request_message_receipts: Vec<LlmMessageReceipt>,
    role_context_manifest: Option<EffectiveContextManifest>,
    context_receipt: LlmContextReceipt,
    request_hash: String,
    tool_schema_hash: Option<String>,
}

/// Stable model-call boundary shared by all business paths.
pub struct LlmInteractionService {
    gateway: Arc<UnifiedGateway>,
    events: broadcast::Sender<LlmInteractionEvent>,
    event_bus_forwarder_attached: AtomicBool,
    hook_manager: parking_lot::RwLock<Option<Arc<HookManager>>>,
    total_prompt_tokens: Arc<AtomicU64>,
    total_completion_tokens: Arc<AtomicU64>,
    last_prompt_tokens: Arc<AtomicU64>,
    last_completion_tokens: Arc<AtomicU64>,
    usage_by_scope: parking_lot::Mutex<std::collections::HashMap<String, LlmUsageSnapshot>>,
    context_quality_by_scope:
        parking_lot::Mutex<std::collections::HashMap<String, LlmContextQualitySnapshot>>,
    scope_accounting_leases: parking_lot::Mutex<std::collections::HashMap<String, u32>>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmUsageSnapshot {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl LlmUsageSnapshot {
    pub fn total_tokens(self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    pub fn saturating_sub(self, baseline: Self) -> Self {
        Self {
            prompt_tokens: self.prompt_tokens.saturating_sub(baseline.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_sub(baseline.completion_tokens),
        }
    }
}

impl LlmInteractionService {
    pub fn new(gateway: Arc<UnifiedGateway>) -> Self {
        Self::with_event_capacity(gateway, 512)
    }

    pub fn with_event_capacity(gateway: Arc<UnifiedGateway>, capacity: usize) -> Self {
        let (events, _) = broadcast::channel(capacity.max(16));
        Self {
            gateway,
            events,
            event_bus_forwarder_attached: AtomicBool::new(false),
            hook_manager: parking_lot::RwLock::new(None),
            total_prompt_tokens: Arc::new(AtomicU64::new(0)),
            total_completion_tokens: Arc::new(AtomicU64::new(0)),
            last_prompt_tokens: Arc::new(AtomicU64::new(0)),
            last_completion_tokens: Arc::new(AtomicU64::new(0)),
            usage_by_scope: parking_lot::Mutex::new(std::collections::HashMap::new()),
            context_quality_by_scope: parking_lot::Mutex::new(std::collections::HashMap::new()),
            scope_accounting_leases: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn shared(gateway: Arc<UnifiedGateway>) -> Arc<Self> {
        Self::shared_with_event_capacity(gateway, 512)
    }

    pub fn shared_with_event_capacity(gateway: Arc<UnifiedGateway>, capacity: usize) -> Arc<Self> {
        let key = Arc::as_ptr(&gateway) as usize;
        let mut registry = SHARED_INTERACTION_SERVICES.lock();
        registry.retain(|_, service| service.strong_count() > 0);
        if let Some(service) = registry.get(&key).and_then(Weak::upgrade) {
            if Arc::ptr_eq(service.gateway(), &gateway) {
                return service;
            }
        }
        let service = Arc::new(Self::with_event_capacity(gateway, capacity));
        registry.insert(key, Arc::downgrade(&service));
        service
    }

    pub fn gateway(&self) -> &Arc<UnifiedGateway> {
        &self.gateway
    }

    pub fn subscribe(&self) -> broadcast::Receiver<LlmInteractionEvent> {
        self.events.subscribe()
    }

    /// Compatibility bridge for existing UI and SA statistics consumers.
    /// The counters are owned by the common interaction plane, so every model
    /// call routed through it is covered exactly once.
    pub fn token_usage_arcs(
        &self,
    ) -> (
        Arc<AtomicU64>,
        Arc<AtomicU64>,
        Arc<AtomicU64>,
        Arc<AtomicU64>,
    ) {
        (
            self.total_prompt_tokens.clone(),
            self.total_completion_tokens.clone(),
            self.last_prompt_tokens.clone(),
            self.last_completion_tokens.clone(),
        )
    }

    pub fn usage_snapshot(&self) -> LlmUsageSnapshot {
        LlmUsageSnapshot {
            prompt_tokens: self.total_prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.total_completion_tokens.load(Ordering::Relaxed),
        }
    }

    pub fn usage_snapshot_for_scope(&self, usage_scope_iri: &str) -> LlmUsageSnapshot {
        self.usage_by_scope
            .lock()
            .get(usage_scope_iri)
            .copied()
            .unwrap_or_default()
    }

    pub fn context_quality_snapshot_for_scope(
        &self,
        usage_scope_iri: &str,
    ) -> LlmContextQualitySnapshot {
        self.context_quality_by_scope
            .lock()
            .get(usage_scope_iri)
            .copied()
            .unwrap_or_default()
    }

    /// Protect one root task's accounting while its concurrent children run.
    /// Dropping the last guard atomically retires both usage and context
    /// aggregates, including cancellation/error paths in the SA future.
    pub fn begin_scope_accounting(
        self: &Arc<Self>,
        usage_scope_iri: impl Into<String>,
    ) -> LlmScopeAccountingGuard {
        let scope = usage_scope_iri.into();
        let mut leases = self.scope_accounting_leases.lock();
        let count = leases.entry(scope.clone()).or_default();
        *count = count.saturating_add(1);
        LlmScopeAccountingGuard {
            service: Arc::downgrade(self),
            scope: Some(scope),
        }
    }

    fn release_scope_accounting(&self, scope: &str) {
        let should_remove = {
            let mut leases = self.scope_accounting_leases.lock();
            match leases.get_mut(scope) {
                Some(count) if *count > 1 => {
                    *count -= 1;
                    false
                }
                Some(_) => {
                    leases.remove(scope);
                    true
                }
                None => false,
            }
        };
        if should_remove {
            self.usage_by_scope.lock().remove(scope);
            self.context_quality_by_scope.lock().remove(scope);
        }
    }

    fn prune_idle_accounting(&self, current_scope: &str) {
        let leases = self.scope_accounting_leases.lock();
        let mut usage = self.usage_by_scope.lock();
        let mut quality = self.context_quality_by_scope.lock();
        let mut scopes = usage
            .keys()
            .chain(quality.keys())
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        if scopes.len() <= MAX_IDLE_ACCOUNTING_SCOPES.saturating_add(leases.len()) {
            return;
        }
        let remove_count = scopes
            .len()
            .saturating_sub(MAX_IDLE_ACCOUNTING_SCOPES.saturating_add(leases.len()));
        let removable = scopes
            .drain()
            .filter(|scope| scope != current_scope && !leases.contains_key(scope))
            .take(remove_count)
            .collect::<Vec<_>>();
        for scope in removable {
            usage.remove(&scope);
            quality.remove(&scope);
        }
    }

    fn record_context_dispatch(&self, metadata: &RequestMetadata) {
        let Some(scope) = metadata.scope.usage_scope_iri.as_deref() else {
            return;
        };
        self.context_quality_by_scope
            .lock()
            .entry(scope.to_string())
            .or_default()
            .record_dispatch(metadata.context_receipt);
        self.prune_idle_accounting(scope);
    }

    fn record_usage(&self, metadata: &RequestMetadata, usage: (u32, u32)) {
        let prompt = u64::from(usage.0);
        let completion = u64::from(usage.1);
        self.total_prompt_tokens
            .fetch_add(prompt, Ordering::Relaxed);
        self.total_completion_tokens
            .fetch_add(completion, Ordering::Relaxed);
        self.last_prompt_tokens.store(prompt, Ordering::Relaxed);
        self.last_completion_tokens
            .store(completion, Ordering::Relaxed);
        if let Some(scope) = metadata.scope.usage_scope_iri.as_deref() {
            let mut by_scope = self.usage_by_scope.lock();
            let entry = by_scope.entry(scope.to_string()).or_default();
            entry.prompt_tokens = entry.prompt_tokens.saturating_add(prompt);
            entry.completion_tokens = entry.completion_tokens.saturating_add(completion);
        }
        if let Some(scope) = metadata.scope.usage_scope_iri.as_deref() {
            self.prune_idle_accounting(scope);
        }
    }

    /// Attach the agent Hook plane to the common model-call boundary. This is
    /// optional so standalone clients retain a lightweight telemetry facade.
    pub fn set_hook_manager(&self, hook_manager: Arc<HookManager>) {
        *self.hook_manager.write() = Some(hook_manager);
    }

    /// Forward metadata-only lifecycle receipts to the task event bus used by
    /// TUI/API clients. At most one forwarder is installed per service.
    pub fn attach_event_bus(self: &Arc<Self>, event_bus: Arc<crate::core::event_bus::EventBus>) {
        if self
            .event_bus_forwarder_attached
            .swap(true, Ordering::AcqRel)
        {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            self.event_bus_forwarder_attached
                .store(false, Ordering::Release);
            warn!("cannot attach LLM interaction EventBus forwarder outside a Tokio runtime");
            return;
        };
        let mut receiver = self.subscribe();
        runtime.spawn(async move {
            loop {
                let event = match receiver.recv().await {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!(skipped, "LLM interaction EventBus forwarder lagged");
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };
                let Some(task_iri) = event.scope.task_iri.as_deref() else {
                    continue;
                };
                let source = event.scope.agent_id.as_deref().unwrap_or("LLM");
                let event_type = match event.phase {
                    LlmInteractionPhase::Assembled => "LLM_INTERACTION_ASSEMBLED",
                    LlmInteractionPhase::BeforeDispatch => "LLM_INTERACTION_STARTED",
                    LlmInteractionPhase::FirstToken => "LLM_INTERACTION_FIRST_TOKEN",
                    LlmInteractionPhase::Completed => "LLM_INTERACTION_COMPLETED",
                    LlmInteractionPhase::Failed => "LLM_INTERACTION_FAILED",
                    LlmInteractionPhase::Cancelled => "LLM_INTERACTION_CANCELLED",
                };
                let payload = serde_json::to_string(&event).unwrap_or_else(|_| {
                    serde_json::json!({
                        "interaction_id": event.scope.interaction_id,
                        "stage": event.scope.stage,
                        "serialization_failed": true,
                    })
                    .to_string()
                });
                event_bus.emit(task_iri, event_type, source, &payload).await;
            }
        });
    }

    pub async fn chat(
        self: &Arc<Self>,
        scope: LlmInteractionScope,
        model: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<ChatCompletionResponse, CoreError> {
        self.chat_with_params(scope, model, messages, None, None, None, None)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn chat_with_params(
        self: &Arc<Self>,
        scope: LlmInteractionScope,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
    ) -> Result<ChatCompletionResponse, CoreError> {
        self.chat_with_params_and_options(
            scope,
            model,
            messages,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            LlmRequestOptions::default(),
        )
        .await
    }

    /// Parameterized interaction with explicit, typed provider options. The
    /// legacy method delegates with an empty option set, preserving both its
    /// wire representation and request hash.
    #[allow(clippy::too_many_arguments)]
    pub async fn chat_with_params_and_options(
        self: &Arc<Self>,
        scope: LlmInteractionScope,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
        options: LlmRequestOptions,
    ) -> Result<ChatCompletionResponse, CoreError> {
        self.chat_with_params_traced_and_options(
            scope,
            model,
            messages,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            options,
        )
        .await
        .map(|(response, _)| response)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn chat_with_params_traced(
        self: &Arc<Self>,
        scope: LlmInteractionScope,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
    ) -> Result<(ChatCompletionResponse, GatewayCallMetadata), CoreError> {
        self.chat_with_params_traced_and_options(
            scope,
            model,
            messages,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            LlmRequestOptions::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn chat_with_params_traced_and_options(
        self: &Arc<Self>,
        scope: LlmInteractionScope,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
        options: LlmRequestOptions,
    ) -> Result<(ChatCompletionResponse, GatewayCallMetadata), CoreError> {
        let options = self.gateway.effective_request_options(model, options);
        let metadata = RequestMetadata::new_with_options(
            scope,
            model,
            false,
            &messages,
            temperature,
            max_tokens,
            tools.as_deref(),
            tool_choice,
            options,
        );
        self.emit(&metadata, LlmInteractionPhase::Assembled, None);
        let started = Instant::now();
        let mut guard = InteractionGuard::new(self.clone(), metadata, started);
        if let Err(error) = self.run_request_hooks(&guard.metadata).await {
            guard.finish(
                LlmInteractionPhase::Failed,
                TerminalMetadata {
                    error_class: Some("request_hook_rejected".to_string()),
                    ..TerminalMetadata::default()
                },
            );
            return Err(error);
        }
        self.emit(&guard.metadata, LlmInteractionPhase::BeforeDispatch, None);
        if let Err(error) =
            self.gateway
                .validate_request_capabilities(model, tools.is_some(), tool_choice, options)
        {
            guard.finish(
                LlmInteractionPhase::Failed,
                TerminalMetadata::from_error(&error),
            );
            return Err(error);
        }

        const RESPONSE_HOOK_RETRIES: usize = 2;
        for response_hook_attempt in 0..=RESPONSE_HOOK_RETRIES {
            guard.record_dispatch();
            match self
                .gateway
                .chat_with_params_traced_and_options(
                    model,
                    messages.clone(),
                    temperature,
                    max_tokens,
                    tools.clone(),
                    tool_choice,
                    options,
                )
                .await
            {
                Ok((response, gateway)) => {
                    let terminal = TerminalMetadata::from_response(&response, gateway.clone());
                    guard.record_provider_usage(terminal.usage);
                    match self.run_response_hooks(&guard.metadata, &terminal).await {
                        HookControl::Continue => {
                            guard.finish(LlmInteractionPhase::Completed, terminal);
                            return Ok((response, gateway));
                        }
                        HookControl::Retry if response_hook_attempt < RESPONSE_HOOK_RETRIES => {
                            continue;
                        }
                        HookControl::Retry => {
                            guard.finish(
                                LlmInteractionPhase::Failed,
                                terminal.with_error_class("response_hook_retry_exhausted"),
                            );
                            return Err(CoreError::Internal {
                                message: "LLM response hook retry budget exhausted".to_string(),
                            });
                        }
                        HookControl::Abort | HookControl::SkipOperation => {
                            guard.finish(
                                LlmInteractionPhase::Failed,
                                terminal.with_error_class("response_hook_rejected"),
                            );
                            return Err(CoreError::InteractionRejected {
                                stage: guard.metadata.scope.stage.clone(),
                                reason: "LLM response rejected by hook policy".to_string(),
                            });
                        }
                    }
                }
                Err(error) => {
                    guard.finish(
                        LlmInteractionPhase::Failed,
                        TerminalMetadata::from_error(&error),
                    );
                    return Err(error);
                }
            }
        }
        unreachable!("bounded response-hook loop always returns")
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn stream_chat_with_params(
        self: &Arc<Self>,
        scope: LlmInteractionScope,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
    ) -> Result<TrackedMessageStream, CoreError> {
        self.stream_chat_with_params_and_options(
            scope,
            model,
            messages,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            LlmRequestOptions::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn stream_chat_with_params_and_options(
        self: &Arc<Self>,
        scope: LlmInteractionScope,
        model: &str,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<Vec<Value>>,
        tool_choice: Option<&str>,
        options: LlmRequestOptions,
    ) -> Result<TrackedMessageStream, CoreError> {
        // Normalize invalid/orphaned tool-message roles before both metadata
        // hashing and wire serialization. The synchronous path applies the
        // same defense, and direct gateway callers repeat it idempotently.
        let messages = UnifiedGateway::sanitize_tool_messages(messages);
        let options = self.gateway.effective_request_options(model, options);
        let metadata = RequestMetadata::new_with_options(
            scope,
            model,
            true,
            &messages,
            temperature,
            max_tokens,
            tools.as_deref(),
            tool_choice,
            options,
        );
        self.emit(&metadata, LlmInteractionPhase::Assembled, None);
        let started = Instant::now();
        let mut guard = InteractionGuard::new(self.clone(), metadata, started);
        if let Err(error) = self.run_request_hooks(&guard.metadata).await {
            guard.finish(
                LlmInteractionPhase::Failed,
                TerminalMetadata {
                    error_class: Some("request_hook_rejected".to_string()),
                    ..TerminalMetadata::default()
                },
            );
            return Err(error);
        }
        self.emit(&guard.metadata, LlmInteractionPhase::BeforeDispatch, None);
        if let Err(error) =
            self.gateway
                .validate_request_capabilities(model, tools.is_some(), tool_choice, options)
        {
            guard.finish(
                LlmInteractionPhase::Failed,
                TerminalMetadata::from_error(&error),
            );
            return Err(error);
        }
        guard.record_dispatch();

        match self
            .gateway
            .stream_chat_with_params_traced_and_options(
                model,
                messages,
                temperature,
                max_tokens,
                tools,
                tool_choice,
                options,
            )
            .await
        {
            Ok((stream, gateway)) => Ok(TrackedMessageStream::new(stream, guard, gateway)),
            Err(error) => {
                guard.finish(
                    LlmInteractionPhase::Failed,
                    TerminalMetadata::from_error(&error),
                );
                Err(error)
            }
        }
    }

    async fn run_request_hooks(&self, metadata: &RequestMetadata) -> Result<(), CoreError> {
        let Some(manager) = ({ self.hook_manager.read().clone() }) else {
            return Ok(());
        };
        const REQUEST_HOOK_RETRIES: usize = 2;
        for attempt in 0..=REQUEST_HOOK_RETRIES {
            let mut context = hook_context(metadata, HookPoint::LlmRequest)
                .with_data("hook_attempt", Value::Number((attempt as u64).into()));
            let decision = manager
                .execute_decision(HookPoint::LlmRequest, &mut context)
                .await;
            match decision.control {
                HookControl::Continue => return Ok(()),
                HookControl::Retry if attempt < REQUEST_HOOK_RETRIES => continue,
                HookControl::Retry => {
                    return Err(CoreError::Internal {
                        message: "LLM request hook retry budget exhausted".to_string(),
                    });
                }
                HookControl::Abort => {
                    return Err(CoreError::InteractionRejected {
                        stage: metadata.scope.stage.clone(),
                        reason: format!(
                            "LLM request aborted by hook {}",
                            decision.terminal_hook.as_deref().unwrap_or("unknown")
                        ),
                    });
                }
                HookControl::SkipOperation => {
                    return Err(CoreError::InteractionRejected {
                        stage: metadata.scope.stage.clone(),
                        reason: format!(
                            "LLM request skipped by hook {}",
                            decision.terminal_hook.as_deref().unwrap_or("unknown")
                        ),
                    });
                }
            }
        }
        unreachable!("bounded request-hook loop always returns")
    }

    async fn run_response_hooks(
        &self,
        metadata: &RequestMetadata,
        terminal: &TerminalMetadata,
    ) -> HookControl {
        let Some(manager) = ({ self.hook_manager.read().clone() }) else {
            return HookControl::Continue;
        };
        let mut context = hook_context(metadata, HookPoint::LlmResponse)
            .with_data(
                "response_hash",
                terminal
                    .response_hash
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            )
            .with_data(
                "prompt_tokens",
                terminal
                    .usage
                    .map(|usage| Value::Number(u64::from(usage.0).into()))
                    .unwrap_or(Value::Null),
            )
            .with_data(
                "completion_tokens",
                terminal
                    .usage
                    .map(|usage| Value::Number(u64::from(usage.1).into()))
                    .unwrap_or(Value::Null),
            )
            .with_data(
                "finish_reason",
                terminal
                    .finish_reason
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            )
            .with_data(
                "choices_count",
                Value::Number((terminal.choices_count as u64).into()),
            )
            .with_data("content_empty", Value::Bool(terminal.content_empty))
            .with_data(
                "response_tool_names",
                serde_json::to_value(&terminal.response_tool_names).unwrap_or(Value::Null),
            )
            .with_data(
                "response_tool_call_count",
                Value::Number((terminal.response_tool_call_count as u64).into()),
            );
        manager
            .execute_decision(HookPoint::LlmResponse, &mut context)
            .await
            .control
    }

    fn emit(
        &self,
        metadata: &RequestMetadata,
        phase: LlmInteractionPhase,
        terminal: Option<TerminalEvent<'_>>,
    ) {
        let terminal = terminal.unwrap_or_default();
        let assembled = phase == LlmInteractionPhase::Assembled;
        let event = LlmInteractionEvent {
            schema_version: LLM_INTERACTION_SCHEMA_VERSION,
            phase,
            emitted_at_ms: chrono::Utc::now().timestamp_millis(),
            scope: metadata.scope.clone(),
            model: metadata.model.clone(),
            streaming: metadata.streaming,
            reasoning_effort: metadata.reasoning_effort.clone(),
            message_count: metadata.message_count,
            advertised_tool_names: metadata.advertised_tool_names.clone(),
            context_receipt: metadata.context_receipt,
            request_message_receipts: assembled
                .then(|| metadata.request_message_receipts.clone())
                .unwrap_or_default(),
            role_context_manifest: assembled
                .then(|| metadata.role_context_manifest.clone())
                .flatten(),
            response_tool_names: terminal.response_tool_names.to_vec(),
            response_tool_call_count: terminal.response_tool_call_count,
            request_hash: metadata.request_hash.clone(),
            tool_schema_hash: metadata.tool_schema_hash.clone(),
            response_hash: terminal.response_hash.map(ToOwned::to_owned),
            elapsed_ms: terminal.elapsed_ms,
            prompt_tokens: terminal.usage.map(|usage| usage.0),
            completion_tokens: terminal.usage.map(|usage| usage.1),
            billed_prompt_tokens: terminal.billed_usage.0,
            billed_completion_tokens: terminal.billed_usage.1,
            model_dispatch_count: terminal.model_dispatch_count,
            finish_reason: terminal.finish_reason.map(ToOwned::to_owned),
            choices_count: terminal.choices_count,
            content_empty: terminal.content_empty,
            error_class: terminal.error_class.map(ToOwned::to_owned),
            http_status: terminal.http_status,
            retryable: terminal.retryable,
            gateway: terminal.gateway.cloned(),
        };

        match phase {
            LlmInteractionPhase::Failed | LlmInteractionPhase::Cancelled => warn!(
                interaction_id = %event.scope.interaction_id,
                task_iri = event.scope.task_iri.as_deref().unwrap_or(""),
                stage = %event.scope.stage,
                model = %event.model,
                phase = ?event.phase,
                elapsed_ms = event.elapsed_ms.unwrap_or_default(),
                error_class = event.error_class.as_deref().unwrap_or(""),
                http_status = event.http_status,
                retryable = event.retryable,
                "LLM interaction terminal event"
            ),
            _ => debug!(
                interaction_id = %event.scope.interaction_id,
                task_iri = event.scope.task_iri.as_deref().unwrap_or(""),
                stage = %event.scope.stage,
                model = %event.model,
                phase = ?event.phase,
                elapsed_ms = event.elapsed_ms.unwrap_or_default(),
                "LLM interaction lifecycle event"
            ),
        }
        let _ = self.events.send(event);
    }
}

/// RAII lifetime for root-task accounting. It intentionally contains no task
/// payload beyond the opaque scope key and can be moved across await points.
pub struct LlmScopeAccountingGuard {
    service: Weak<LlmInteractionService>,
    scope: Option<String>,
}

impl Drop for LlmScopeAccountingGuard {
    fn drop(&mut self) {
        let Some(scope) = self.scope.take() else {
            return;
        };
        if let Some(service) = self.service.upgrade() {
            service.release_scope_accounting(&scope);
        }
    }
}

impl RequestMetadata {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        scope: LlmInteractionScope,
        model: &str,
        streaming: bool,
        messages: &[ChatMessage],
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<&[Value]>,
        tool_choice: Option<&str>,
    ) -> Self {
        Self::new_with_options(
            scope,
            model,
            streaming,
            messages,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            LlmRequestOptions::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_options(
        scope: LlmInteractionScope,
        model: &str,
        streaming: bool,
        messages: &[ChatMessage],
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        tools: Option<&[Value]>,
        tool_choice: Option<&str>,
        options: LlmRequestOptions,
    ) -> Self {
        let advertised_tool_names = advertised_tool_names(tools);
        let request_message_receipts = message_receipts(messages);
        let role_context_manifest = scope.context_manifest_receipt.clone();
        let context_receipt = LlmContextReceipt::from_request(
            &request_message_receipts,
            role_context_manifest.as_ref(),
        );
        let reasoning_effort = options
            .reasoning_effort
            .map(|effort| effort.as_str().to_string());
        let mut request_descriptor = serde_json::json!({
            "model": model,
            "streaming": streaming,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        });
        // Preserve historical hashes for legacy callers. An explicit effort is
        // part of request identity because it can change latency and output.
        if let Some(effort) = reasoning_effort.as_deref() {
            request_descriptor["reasoning_effort"] = Value::String(effort.to_string());
        }
        let request_hash = hash_serializable(&request_descriptor);
        let tool_schema_hash = tools.map(hash_serializable);
        Self {
            scope,
            model: model.to_string(),
            streaming,
            reasoning_effort,
            message_count: messages.len(),
            advertised_tool_names,
            request_message_receipts,
            role_context_manifest,
            context_receipt,
            request_hash,
            tool_schema_hash,
        }
    }
}

fn hook_context(metadata: &RequestMetadata, point: HookPoint) -> HookContext {
    let agent_id = metadata.scope.agent_id.as_deref().unwrap_or("LLM");
    let role = metadata.scope.role.as_deref().unwrap_or("LLM");
    let mut context = HookContext::new(point, agent_id, role)
        .with_trace_id(metadata.scope.interaction_id.clone())
        .with_data("stage", Value::String(metadata.scope.stage.clone()))
        .with_data("model", Value::String(metadata.model.clone()))
        .with_data(
            "message_count",
            Value::Number((metadata.message_count as u64).into()),
        )
        .with_data("request_hash", Value::String(metadata.request_hash.clone()))
        .with_data(
            "advertised_tool_names",
            serde_json::to_value(&metadata.advertised_tool_names).unwrap_or(Value::Null),
        );
    if let Some(task_iri) = metadata.scope.task_iri.as_deref() {
        context = context.with_task(task_iri, task_iri);
    }
    if let Some(cycle_id) = metadata.scope.cycle_id.as_deref() {
        context = context.with_data("cycle_id", Value::String(cycle_id.to_string()));
    }
    if let Some(hash) = metadata.scope.context_manifest_hash.as_deref() {
        context = context.with_data("context_manifest_hash", Value::String(hash.to_string()));
    }
    if let Some(effort) = metadata.reasoning_effort.as_deref() {
        context = context.with_data("reasoning_effort", Value::String(effort.to_string()));
    }
    context
}

#[derive(Default)]
struct TerminalMetadata {
    response_hash: Option<String>,
    usage: Option<(u32, u32)>,
    finish_reason: Option<String>,
    error_class: Option<String>,
    http_status: Option<u16>,
    retryable: Option<bool>,
    gateway: Option<GatewayCallMetadata>,
    choices_count: usize,
    content_empty: bool,
    response_tool_names: Vec<String>,
    response_tool_call_count: usize,
}

impl TerminalMetadata {
    fn from_error(error: &CoreError) -> Self {
        Self {
            error_class: Some(core_error_class(error).to_string()),
            http_status: crate::gateway::unified_gateway::gateway_error_http_status(error),
            retryable: crate::gateway::unified_gateway::gateway_error_retryable(error)
                .or_else(|| crate::llm::sse::stream_core_error_retryable(error)),
            ..Self::default()
        }
    }

    fn from_response(response: &ChatCompletionResponse, gateway: GatewayCallMetadata) -> Self {
        let first_choice = response.choices.first();
        let response_tool_calls = first_choice
            .and_then(|choice| choice.message.tool_calls.as_ref())
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let response_tool_names = normalized_names(
            &response_tool_calls
                .iter()
                .map(|call| call.function.name.clone())
                .collect::<Vec<_>>(),
        );
        Self {
            response_hash: Some(hash_serializable(response)),
            usage: response
                .usage
                .as_ref()
                .map(|usage| (usage.prompt_tokens, usage.completion_tokens)),
            finish_reason: response
                .choices
                .first()
                .and_then(|choice| choice.finish_reason.clone()),
            error_class: None,
            http_status: gateway.http_status,
            retryable: None,
            gateway: Some(gateway),
            choices_count: response.choices.len(),
            content_empty: first_choice
                .and_then(|choice| choice.message.content.as_deref())
                .is_none_or(|content| content.trim().is_empty()),
            response_tool_names,
            response_tool_call_count: response_tool_calls.len(),
        }
    }

    fn with_error_class(mut self, error_class: &str) -> Self {
        self.error_class = Some(error_class.to_string());
        self
    }
}

#[derive(Default)]
struct TerminalEvent<'a> {
    response_hash: Option<&'a str>,
    elapsed_ms: Option<u64>,
    usage: Option<(u32, u32)>,
    billed_usage: (u64, u64),
    model_dispatch_count: u32,
    finish_reason: Option<&'a str>,
    error_class: Option<&'a str>,
    http_status: Option<u16>,
    retryable: Option<bool>,
    gateway: Option<&'a GatewayCallMetadata>,
    response_tool_names: &'a [String],
    response_tool_call_count: usize,
    choices_count: Option<usize>,
    content_empty: Option<bool>,
}

struct InteractionGuard {
    service: Arc<LlmInteractionService>,
    metadata: RequestMetadata,
    started: Instant,
    finished: bool,
    billed_usage: (u64, u64),
    model_dispatch_count: u32,
}

impl InteractionGuard {
    fn new(
        service: Arc<LlmInteractionService>,
        metadata: RequestMetadata,
        started: Instant,
    ) -> Self {
        Self {
            service,
            metadata,
            started,
            finished: false,
            billed_usage: (0, 0),
            model_dispatch_count: 0,
        }
    }

    fn record_dispatch(&mut self) {
        self.model_dispatch_count = self.model_dispatch_count.saturating_add(1);
        self.service.record_context_dispatch(&self.metadata);
    }

    fn record_provider_usage(&mut self, usage: Option<(u32, u32)>) {
        let Some(usage) = usage else {
            return;
        };
        self.service.record_usage(&self.metadata, usage);
        self.billed_usage.0 = self.billed_usage.0.saturating_add(u64::from(usage.0));
        self.billed_usage.1 = self.billed_usage.1.saturating_add(u64::from(usage.1));
    }

    fn emit(&self, phase: LlmInteractionPhase, terminal: &TerminalMetadata) {
        self.service.emit(
            &self.metadata,
            phase,
            Some(TerminalEvent {
                response_hash: terminal.response_hash.as_deref(),
                elapsed_ms: Some(elapsed_ms(self.started)),
                usage: terminal.usage,
                billed_usage: self.billed_usage,
                model_dispatch_count: self.model_dispatch_count,
                finish_reason: terminal.finish_reason.as_deref(),
                error_class: terminal.error_class.as_deref(),
                http_status: terminal.http_status,
                retryable: terminal.retryable,
                gateway: terminal.gateway.as_ref(),
                response_tool_names: &terminal.response_tool_names,
                response_tool_call_count: terminal.response_tool_call_count,
                choices_count: Some(terminal.choices_count),
                content_empty: Some(terminal.content_empty),
            }),
        );
    }

    fn finish(&mut self, phase: LlmInteractionPhase, terminal: TerminalMetadata) {
        if !self.finished {
            self.emit(phase, &terminal);
            self.finished = true;
        }
    }
}

impl Drop for InteractionGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.emit(
                LlmInteractionPhase::Cancelled,
                &TerminalMetadata {
                    error_class: Some("cancelled".to_string()),
                    ..TerminalMetadata::default()
                },
            );
            self.finished = true;
        }
    }
}

/// Streaming response wrapper that guarantees exactly one terminal lifecycle
/// event, including when a caller abandons the stream before completion.
pub struct TrackedMessageStream {
    inner: crate::llm::stream_processor::MessageStream,
    guard: InteractionGuard,
    gateway: GatewayCallMetadata,
    first_token_seen: bool,
    response_hasher: Sha256,
    usage: Option<StreamUsage>,
    finish_reason: Option<String>,
    saw_visible_content: bool,
    /// Tool calls keyed by their provider response index. Both a block-start
    /// and later deltas may describe the same call, so counting raw events
    /// would over-count. Distinct indexes preserve repeated same-name calls.
    response_tool_calls: BTreeMap<u32, String>,
    usage_recorded: bool,
}

impl TrackedMessageStream {
    fn new(
        inner: crate::llm::stream_processor::MessageStream,
        guard: InteractionGuard,
        gateway: GatewayCallMetadata,
    ) -> Self {
        Self {
            inner,
            guard,
            gateway,
            first_token_seen: false,
            response_hasher: Sha256::new(),
            usage: None,
            finish_reason: None,
            saw_visible_content: false,
            response_tool_calls: BTreeMap::new(),
            usage_recorded: false,
        }
    }

    pub fn interaction_id(&self) -> &str {
        &self.guard.metadata.scope.interaction_id
    }

    pub fn gateway_metadata(&self) -> &GatewayCallMetadata {
        &self.gateway
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, SseError> {
        match self.inner.next_event().await {
            Ok(Some(event)) => {
                if let Ok(encoded) = serde_json::to_vec(&event) {
                    self.response_hasher.update(encoded);
                }
                if let StreamEvent::MessageStart(start) = &event {
                    if let Some(provider_response_id) =
                        start.id.as_ref().filter(|id| !id.is_empty())
                    {
                        // Provider response IDs are observability metadata,
                        // not tool correlation IDs. Preserve the exact value.
                        self.gateway.provider_response_id = Some(provider_response_id.clone());
                    }
                }
                if is_first_token_event(&event) && !self.first_token_seen {
                    self.guard.service.emit(
                        &self.guard.metadata,
                        LlmInteractionPhase::FirstToken,
                        Some(TerminalEvent {
                            elapsed_ms: Some(elapsed_ms(self.guard.started)),
                            billed_usage: self.guard.billed_usage,
                            model_dispatch_count: self.guard.model_dispatch_count,
                            ..TerminalEvent::default()
                        }),
                    );
                    self.first_token_seen = true;
                }
                match &event {
                    StreamEvent::ContentBlockStart(start) => {
                        if let crate::llm::stream_types::ContentBlock::ToolUse { name, .. } =
                            &start.content_block
                        {
                            self.response_tool_calls.insert(start.index, name.clone());
                        }
                    }
                    StreamEvent::ContentBlockDelta(delta) => {
                        use crate::llm::stream_types::ContentBlockDelta;
                        match &delta.delta {
                            ContentBlockDelta::TextDelta { text } => {
                                self.saw_visible_content |= !text.is_empty();
                            }
                            // Reasoning is deliberately tracked separately
                            // from visible assistant content.  A thinking-only
                            // completion is unusable to schema consumers and
                            // must remain `content_empty=true` in telemetry.
                            ContentBlockDelta::ThinkingDelta { .. } => {}
                            ContentBlockDelta::ToolCallDelta { id, name, .. } => {
                                if let Some(name) = name {
                                    self.response_tool_calls.insert(delta.index, name.clone());
                                } else if id.is_some() {
                                    self.response_tool_calls.entry(delta.index).or_default();
                                }
                            }
                            ContentBlockDelta::InputJsonDelta { .. } => {}
                        }
                    }
                    _ => {}
                }
                if let StreamEvent::MessageDelta(delta) = &event {
                    if let Some(usage) = &delta.usage {
                        self.usage = Some(usage.clone());
                    }
                    if delta.finish_reason.is_some() {
                        self.finish_reason = delta.finish_reason.clone();
                    }
                }
                if matches!(event, StreamEvent::MessageStop(_)) {
                    self.finish_completed().await?;
                }
                Ok(Some(event))
            }
            Ok(None) => {
                self.finish_completed().await?;
                Ok(None)
            }
            Err(error) => {
                self.record_usage_once();
                let error_class = error.error_class().to_string();
                self.guard.finish(
                    LlmInteractionPhase::Failed,
                    TerminalMetadata {
                        response_hash: Some(self.response_hash()),
                        usage: self.usage.as_ref().map(stream_usage_pair),
                        finish_reason: self.finish_reason.clone(),
                        error_class: Some(error_class),
                        http_status: self.gateway.http_status,
                        retryable: error.retryable(),
                        gateway: Some(self.gateway.clone()),
                        choices_count: usize::from(self.first_token_seen),
                        content_empty: !self.saw_visible_content,
                        response_tool_names: self.response_tool_names(),
                        response_tool_call_count: self.response_tool_calls.len(),
                    },
                );
                Err(error)
            }
        }
    }

    pub async fn collect_all(&mut self) -> Result<StreamResponse, SseError> {
        let mut accumulator = crate::llm::stream_types::StreamAccumulator::new();
        while let Some(event) = self.next_event().await? {
            accumulator.process_event(&event);
        }
        Ok(accumulator.into())
    }

    async fn finish_completed(&mut self) -> Result<(), SseError> {
        if self.guard.finished {
            return Ok(());
        }
        self.record_usage_once();
        let terminal = TerminalMetadata {
            response_hash: Some(self.response_hash()),
            usage: self.usage.as_ref().map(stream_usage_pair),
            finish_reason: self.finish_reason.clone(),
            error_class: None,
            http_status: self.gateway.http_status,
            retryable: None,
            gateway: Some(self.gateway.clone()),
            choices_count: usize::from(self.first_token_seen),
            content_empty: !self.saw_visible_content,
            response_tool_names: self.response_tool_names(),
            response_tool_call_count: self.response_tool_calls.len(),
        };
        match self
            .guard
            .service
            .run_response_hooks(&self.guard.metadata, &terminal)
            .await
        {
            HookControl::Continue => {
                self.guard.finish(LlmInteractionPhase::Completed, terminal);
                Ok(())
            }
            HookControl::Retry => {
                self.guard.finish(
                    LlmInteractionPhase::Failed,
                    terminal.with_error_class("stream_response_hook_retry_unsupported"),
                );
                Err(SseError::new(SseErrorKind::ResponseHookRetryUnsupported))
            }
            HookControl::Abort | HookControl::SkipOperation => {
                self.guard.finish(
                    LlmInteractionPhase::Failed,
                    terminal.with_error_class("stream_response_hook_rejected"),
                );
                Err(SseError::new(SseErrorKind::ResponseHookRejected))
            }
        }
    }

    fn response_tool_names(&self) -> Vec<String> {
        normalized_names(
            &self
                .response_tool_calls
                .values()
                .cloned()
                .collect::<Vec<_>>(),
        )
    }

    fn response_hash(&self) -> String {
        hex::encode(self.response_hasher.clone().finalize())
    }

    fn record_usage_once(&mut self) {
        if self.usage_recorded {
            return;
        }
        let usage = self.usage.as_ref().map(stream_usage_pair);
        if usage.is_some() {
            self.guard.record_provider_usage(usage);
            self.usage_recorded = true;
        }
    }
}

impl Drop for TrackedMessageStream {
    fn drop(&mut self) {
        // Some providers report usage before their final stop event. Preserve
        // that billed usage even when the consumer cancels immediately after.
        self.record_usage_once();
    }
}

fn stream_usage_pair(usage: &StreamUsage) -> (u32, u32) {
    (usage.prompt_tokens, usage.completion_tokens)
}

fn is_first_token_event(event: &StreamEvent) -> bool {
    use crate::llm::stream_types::{ContentBlock, ContentBlockDelta};
    match event {
        StreamEvent::ContentBlockStart(start) => {
            matches!(start.content_block, ContentBlock::ToolUse { .. })
        }
        StreamEvent::ContentBlockDelta(delta) => match &delta.delta {
            ContentBlockDelta::TextDelta { text } => !text.is_empty(),
            ContentBlockDelta::ThinkingDelta { thinking } => !thinking.is_empty(),
            ContentBlockDelta::InputJsonDelta { partial_json } => !partial_json.is_empty(),
            ContentBlockDelta::ToolCallDelta {
                id,
                name,
                arguments,
            } => {
                id.is_some()
                    || name.is_some()
                    || arguments.as_deref().is_some_and(|v| !v.is_empty())
            }
        },
        _ => false,
    }
}

fn advertised_tool_names(tools: Option<&[Value]>) -> Vec<String> {
    let mut names = tools
        .unwrap_or_default()
        .iter()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| tool.get("name"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn message_receipts(messages: &[ChatMessage]) -> Vec<LlmMessageReceipt> {
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| LlmMessageReceipt {
            index,
            role: message.role.clone(),
            kind: message_receipt_kind(message),
            content_chars: message.content.chars().count(),
            message_sha256: hash_serializable(message),
            tool_call_count: message
                .tool_calls
                .as_ref()
                .map(Vec::len)
                .unwrap_or_default(),
            has_reasoning: message
                .reasoning_content
                .as_deref()
                .is_some_and(|reasoning| !reasoning.is_empty()),
        })
        .collect()
}

fn message_receipt_kind(message: &ChatMessage) -> ContextFragmentKind {
    match message.name.as_deref() {
        Some("context_model_generated_plan" | "context_model_history") => {
            return ContextFragmentKind::ModelHistory;
        }
        Some("context_unverified_retrieval") => {
            return ContextFragmentKind::UnverifiedRetrieval;
        }
        Some("context_verified_evidence") => return ContextFragmentKind::VerifiedEvidence,
        Some("context_tool_output") => return ContextFragmentKind::ToolOutput,
        Some("context_user_input") => return ContextFragmentKind::UserInput,
        Some("context_authoritative_instruction") => {
            return ContextFragmentKind::AuthoritativeInstruction;
        }
        Some(name) if name.starts_with("runtime_control_") => {
            return ContextFragmentKind::AuthoritativeInstruction;
        }
        _ => {}
    }
    match message.role.as_str() {
        "system" | "developer" => ContextFragmentKind::AuthoritativeInstruction,
        "user" => ContextFragmentKind::UserInput,
        "tool" | "function" => ContextFragmentKind::ToolOutput,
        // Assistant messages and provider-specific roles are model history
        // unless an upstream typed name says otherwise.
        _ => ContextFragmentKind::ModelHistory,
    }
}

fn normalized_names(names: &[String]) -> Vec<String> {
    let mut names = names.to_vec();
    names.retain(|name| !name.trim().is_empty());
    names.sort();
    names.dedup();
    names
}

fn hash_serializable<T: Serialize + ?Sized>(value: &T) -> String {
    let encoded = serde_json::to_vec(value).unwrap_or_default();
    hex::encode(Sha256::digest(encoded))
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn core_error_class(error: &CoreError) -> &'static str {
    if let Some(class) = crate::gateway::unified_gateway::gateway_error_class(error) {
        return class;
    }
    if let Some(class) = crate::llm::sse::stream_core_error_class(error) {
        return class;
    }
    match error {
        CoreError::NodeTooLarge { .. } => "node_too_large",
        CoreError::ProjectionTooLarge { .. } => "projection_too_large",
        CoreError::InvalidJsonLd { .. } => "invalid_jsonld",
        CoreError::NodeNotFound { .. } => "node_not_found",
        CoreError::TaskNotFound { .. } => "task_not_found",
        CoreError::SkillNotFound { .. } => "skill_not_found",
        CoreError::FrameNotFound { .. } => "frame_not_found",
        CoreError::ValidationFailed { .. } => "validation_failed",
        CoreError::SparqlError { .. } => "sparql",
        CoreError::StorageError { .. } => "storage",
        CoreError::OxigraphSyncFailed { .. } => "oxigraph_sync",
        CoreError::Internal { message } if message.contains("max_output_tokens") => {
            "output_token_limit"
        }
        CoreError::Internal { .. } => "internal",
        CoreError::InteractionRejected { .. } => "interaction_rejected",
        CoreError::PermissionDenied { .. } => "permission_denied",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::settings::GatewaySettings;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn service_at(base_url: &str) -> Arc<LlmInteractionService> {
        service_at_with_responses_api(base_url, false)
    }

    fn service_at_with_responses_api(
        base_url: &str,
        use_responses_api: bool,
    ) -> Arc<LlmInteractionService> {
        let gateway = UnifiedGateway::new(&GatewaySettings {
            base_url: base_url.to_string(),
            api_key: "secret-that-must-not-appear".to_string(),
            default_model: "test-model".to_string(),
            timeout_seconds: 1,
            max_retries: 0,
            retry_base_ms: 1,
            model_mapping: Default::default(),
            use_responses_api,
        })
        .expect("gateway");
        Arc::new(LlmInteractionService::new(Arc::new(gateway)))
    }

    fn service() -> Arc<LlmInteractionService> {
        service_at("http://127.0.0.1:1")
    }

    async fn response_server(
        body: String,
        content_type: &'static str,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("local address");
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut request = vec![0_u8; 16 * 1024];
            let _ = socket.read(&mut request).await;
            let header = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(header.as_bytes()).await.expect("header");
            socket.write_all(body.as_bytes()).await.expect("body");
            let _ = socket.shutdown().await;
        });
        (format!("http://{address}"), handle)
    }

    async fn repeated_response_server(
        body: String,
        content_type: &'static str,
        requests: usize,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("local address");
        let handle = tokio::spawn(async move {
            for _ in 0..requests {
                let (mut socket, _) = listener.accept().await.expect("accept request");
                let mut request = vec![0_u8; 16 * 1024];
                let _ = socket.read(&mut request).await;
                let header = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(header.as_bytes()).await.expect("header");
                socket.write_all(body.as_bytes()).await.expect("body");
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{address}"), handle)
    }

    async fn truncated_stream_server(
        body: String,
        missing_bytes: usize,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("local address");
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut request = vec![0_u8; 16 * 1024];
            let _ = socket.read(&mut request).await;
            let declared_len = body.len().saturating_add(missing_bytes);
            let header = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {declared_len}\r\nconnection: close\r\n\r\n"
            );
            socket.write_all(header.as_bytes()).await.expect("header");
            socket
                .write_all(body.as_bytes())
                .await
                .expect("partial body");
            let _ = socket.shutdown().await;
        });
        (format!("http://{address}"), handle)
    }

    fn message(content: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn context_manifest(dispositions: &[ContextDisposition]) -> EffectiveContextManifest {
        use crate::core::agent_instance::AgentRole;
        use crate::core::context_model::{
            ContextFreshnessPolicy, ContextScope, ContextSlot, ContextSourceKind,
            ContextSourceRecord, ContextTrustClass, EffectiveContextManifestEntry,
        };

        EffectiveContextManifest {
            schema_version: "test".into(),
            policy_version: "test".into(),
            role: AgentRole::Do,
            scope: ContextScope::Task {
                task_iri: "iri://task/context".into(),
            },
            entries: dispositions
                .iter()
                .enumerate()
                .map(|(index, disposition)| EffectiveContextManifestEntry {
                    fragment_id: format!("fragment-{index}"),
                    slot: ContextSlot::OriginalTask,
                    kind: ContextFragmentKind::UserInput,
                    trust: ContextTrustClass::UserProvided,
                    source: ContextSourceRecord::new(ContextSourceKind::UserRequest),
                    scope: ContextScope::Task {
                        task_iri: "iri://task/context".into(),
                    },
                    created_at: chrono::Utc::now(),
                    expires_at: None,
                    freshness: ContextFreshnessPolicy::Immutable,
                    required: false,
                    priority: 1,
                    max_chars: 10,
                    provider_payload_preserved: false,
                    original_chars: 10,
                    effective_chars: 10,
                    content_sha256: "metadata-hash".into(),
                    disposition: *disposition,
                    policy_rejection: None,
                })
                .collect(),
            effective_chars: 10,
            required_budget_exceeded: true,
            base_context_sha256: None,
            effective_sha256: "manifest-hash".into(),
        }
    }

    #[test]
    fn events_are_metadata_only_and_extract_tool_names() {
        let service = service();
        let mut receiver = service.subscribe();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "sensitive prompt".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "file_read", "parameters": {}}
        })];
        let metadata = RequestMetadata::new(
            LlmInteractionScope::new("test"),
            "test-model",
            false,
            &messages,
            None,
            None,
            Some(&tools),
            None,
        );
        service.emit(&metadata, LlmInteractionPhase::Assembled, None);
        let event = receiver.try_recv().expect("event");
        let encoded = serde_json::to_string(&event).expect("serialize event");
        assert_eq!(event.advertised_tool_names, vec!["file_read"]);
        assert_eq!(event.request_message_receipts.len(), 1);
        assert_eq!(
            event.request_message_receipts[0].kind,
            ContextFragmentKind::UserInput
        );
        assert_eq!(event.request_message_receipts[0].content_chars, 16);
        assert_eq!(event.context_receipt.request_messages, 1);
        assert_eq!(event.context_receipt.request_chars, 16);
        assert_eq!(event.context_receipt.by_kind.user_input.messages, 1);
        assert_eq!(event.context_receipt.by_kind.user_input.chars, 16);
        assert!(!encoded.contains("sensitive prompt"));
        assert!(!encoded.contains("secret-that-must-not-appear"));
        assert_eq!(event.request_hash.len(), 64);
    }

    #[test]
    fn none_reasoning_policy_is_reported_as_the_unchanged_provider_default() {
        let service = service();
        let mut receiver = service.subscribe();
        let messages = vec![message("sensitive decomposition prompt")];
        let scope = LlmInteractionScope::new("bizagent_decompose");
        let legacy = RequestMetadata::new(
            scope.clone(),
            "test-model",
            false,
            &messages,
            Some(0.1),
            Some(4_096),
            None,
            None,
        );
        let explicit = RequestMetadata::new_with_options(
            scope,
            "test-model",
            false,
            &messages,
            Some(0.1),
            Some(4_096),
            None,
            None,
            LlmRequestOptions::default()
                .with_reasoning_effort(crate::config::settings::ReasoningEffort::None),
        );

        assert_eq!(legacy.reasoning_effort, None);
        assert_eq!(explicit.reasoning_effort, None);
        assert_eq!(legacy.request_hash, explicit.request_hash);

        service.emit(&explicit, LlmInteractionPhase::Assembled, None);
        let event = receiver.try_recv().expect("interaction event");
        assert_eq!(event.reasoning_effort, None);
        let encoded = serde_json::to_string(&event).expect("serialize event");
        assert!(!encoded.contains("\"reasoning_effort\""));
        assert!(!encoded.contains("sensitive decomposition prompt"));
    }

    #[test]
    fn compact_receipt_covers_every_manifest_disposition_without_identifiers() {
        let manifest = context_manifest(&[
            ContextDisposition::Included,
            ContextDisposition::Truncated,
            ContextDisposition::DroppedExpired,
            ContextDisposition::DroppedScope,
            ContextDisposition::DroppedByRolePolicy,
            ContextDisposition::DroppedByBudget,
        ]);
        let messages = message_receipts(&[message("private-context")]);
        let receipt = LlmContextReceipt::from_request(&messages, Some(&manifest));
        assert_eq!(receipt.manifest_fragments, 6);
        assert_eq!(receipt.dispositions.included, 1);
        assert_eq!(receipt.dispositions.truncated, 1);
        assert_eq!(receipt.dispositions.dropped(), 4);
        assert!(receipt.required_budget_exceeded);
        let encoded = serde_json::to_string(&receipt).unwrap();
        assert!(!encoded.contains("private-context"));
        assert!(!encoded.contains("fragment-"));
        assert!(!encoded.contains("metadata-hash"));
        assert!(!encoded.contains("manifest-hash"));
    }

    #[test]
    fn scope_preserves_explicit_correlation() {
        let scope = LlmInteractionScope::new("plan")
            .with_interaction_id("request-42")
            .with_parent("request-1")
            .with_task("iri://task/42")
            .with_cycle("cycle-2")
            .with_agent("PA-1", "PA")
            .with_context_manifest_hash("abc");
        assert_eq!(scope.interaction_id, "request-42");
        assert_eq!(scope.parent_interaction_id.as_deref(), Some("request-1"));
        assert_eq!(scope.usage_scope_iri.as_deref(), Some("iri://task/42"));
        assert_eq!(scope.role.as_deref(), Some("PA"));
        assert_eq!(scope.context_manifest_hash.as_deref(), Some("abc"));
    }

    #[test]
    fn scope_records_materialized_agent_spec_without_business_payload() {
        use crate::core::agent_instance::AgentRole;
        use crate::core::context_model::{AgentSpecSourceKind, GeneratedAgentSpec};

        let manifest = context_manifest(&[ContextDisposition::Included]);
        let spec = GeneratedAgentSpec::runtime_fallback(
            AgentRole::Do,
            "secret objective that telemetry must not contain",
            "test-model",
        )
        .record_materialization("# secret model-authored agent.md", manifest.clone());
        let scope = LlmInteractionScope::new("bizagent_decompose")
            .with_agent("DA-parent", "DA")
            .with_context_manifest(&manifest)
            .with_agent_spec(&spec);

        let receipt = scope
            .agent_spec_receipt
            .as_ref()
            .expect("compiled parent Agent must expose a metadata receipt");
        assert_eq!(receipt.role, AgentRole::Do);
        assert_eq!(receipt.source.kind, AgentSpecSourceKind::RuntimeFallback);
        assert_eq!(receipt.source.producer.as_deref(), Some("AgentRunner"));
        assert_eq!(receipt.source.model.as_deref(), Some("test-model"));
        assert_eq!(receipt.agent_md_sha256, spec.agent_md_sha256);
        assert_eq!(receipt.agent_md_chars, spec.agent_md_chars);
        assert_eq!(
            receipt.context_manifest_hash.as_deref(),
            Some(manifest.effective_sha256.as_str())
        );
        assert_eq!(
            scope.context_manifest_hash.as_deref(),
            receipt.context_manifest_hash.as_deref()
        );

        let encoded = serde_json::to_string(&scope).unwrap();
        assert!(!encoded.contains("secret objective"));
        assert!(!encoded.contains("secret model-authored agent.md"));
        assert!(encoded.contains("agent_md_sha256"));
        assert!(encoded.contains("runtime_fallback"));
    }

    #[tokio::test]
    async fn synchronous_lifecycle_has_one_correlated_terminal_event() {
        let body = serde_json::json!({
            "id": "provider-1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })
        .to_string();
        let (base_url, server) = response_server(body, "application/json").await;
        let service = service_at(&base_url);
        let mut events = service.subscribe();

        let response = service
            .chat_with_params(
                LlmInteractionScope::new("unit_sync").with_interaction_id("sync-42"),
                "test-model",
                vec![message("private request")],
                None,
                None,
                None,
                None,
            )
            .await
            .expect("model response");
        assert_eq!(response.id.as_deref(), Some("provider-1"));
        server.await.expect("server task");

        let received = [
            events.recv().await.expect("assembled"),
            events.recv().await.expect("dispatch"),
            events.recv().await.expect("completed"),
        ];
        assert_eq!(received[0].phase, LlmInteractionPhase::Assembled);
        assert_eq!(received[1].phase, LlmInteractionPhase::BeforeDispatch);
        assert_eq!(received[2].phase, LlmInteractionPhase::Completed);
        assert!(received
            .iter()
            .all(|event| event.scope.interaction_id == "sync-42"));
        assert!(received
            .iter()
            .all(|event| event.context_receipt == received[0].context_receipt));
        assert_eq!(received[2].prompt_tokens, Some(3));
        assert_eq!(received[2].completion_tokens, Some(2));
        assert_eq!(received[2].billed_prompt_tokens, 3);
        assert_eq!(received[2].billed_completion_tokens, 2);
        assert_eq!(received[2].model_dispatch_count, 1);
        assert_eq!(received[2].choices_count, Some(1));
        assert_eq!(received[2].content_empty, Some(false));
        assert!(received[2].response_hash.is_some());
        assert!(!serde_json::to_string(&received)
            .unwrap()
            .contains("private request"));
        assert!(
            events.try_recv().is_err(),
            "exactly one terminal event expected"
        );
    }

    #[tokio::test]
    async fn synchronous_lifecycle_counts_repeated_same_name_tool_calls_exactly() {
        let body = serde_json::json!({
            "id": "provider-tools",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "provider-call-a",
                            "type": "function",
                            "function": {"name": "file_read", "arguments": "{\"path\":\"a\"}"}
                        },
                        {
                            "id": "provider-call-b",
                            "type": "function",
                            "function": {"name": "file_read", "arguments": "{\"path\":\"b\"}"}
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
        })
        .to_string();
        let (base_url, server) = response_server(body, "application/json").await;
        let service = service_at(&base_url);
        let mut events = service.subscribe();

        let response = service
            .chat(
                LlmInteractionScope::new("unit_sync_tools"),
                "test-model",
                vec![message("use tools")],
            )
            .await
            .expect("model response");
        server.await.expect("server task");

        let raw_call_ids = response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls")
            .iter()
            .map(|call| call.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(raw_call_ids, vec!["provider-call-a", "provider-call-b"]);

        let mut terminal = None;
        while let Ok(event) = events.try_recv() {
            if event.phase == LlmInteractionPhase::Completed {
                terminal = Some(event);
            }
        }
        let terminal = terminal.expect("completed event");
        assert_eq!(terminal.schema_version, LLM_INTERACTION_SCHEMA_VERSION);
        assert_eq!(terminal.response_tool_call_count, 2);
        assert_eq!(terminal.response_tool_names, vec!["file_read"]);
    }

    #[tokio::test]
    async fn streaming_lifecycle_reports_first_token_usage_and_completion() {
        let body = concat!(
            "data: {\"id\":\"stream-1\",\"model\":\"test-model\"}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1,\"total_tokens\":5}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let (base_url, server) = response_server(body, "text/event-stream").await;
        let service = service_at(&base_url);
        let mut events = service.subscribe();
        let mut stream = service
            .stream_chat_with_params(
                LlmInteractionScope::new("unit_stream").with_interaction_id("stream-42"),
                "test-model",
                vec![message("stream request")],
                None,
                None,
                None,
                None,
            )
            .await
            .expect("stream response");
        let response = stream.collect_all().await.expect("collect stream");
        assert_eq!(response.content, "hello");
        server.await.expect("server task");

        let mut received = Vec::new();
        while let Ok(event) = events.try_recv() {
            received.push(event);
        }
        assert_eq!(
            received.iter().map(|event| event.phase).collect::<Vec<_>>(),
            vec![
                LlmInteractionPhase::Assembled,
                LlmInteractionPhase::BeforeDispatch,
                LlmInteractionPhase::FirstToken,
                LlmInteractionPhase::Completed,
            ]
        );
        let terminal = received.last().expect("terminal event");
        assert_eq!(terminal.prompt_tokens, Some(4));
        assert_eq!(terminal.completion_tokens, Some(1));
        assert_eq!(terminal.billed_prompt_tokens, 4);
        assert_eq!(terminal.billed_completion_tokens, 1);
        assert_eq!(terminal.model_dispatch_count, 1);
        assert_eq!(terminal.finish_reason.as_deref(), Some("stop"));
        assert_eq!(terminal.choices_count, Some(1));
        assert_eq!(terminal.content_empty, Some(false));
        assert_eq!(
            terminal
                .gateway
                .as_ref()
                .and_then(|gateway| gateway.provider_response_id.as_deref()),
            Some("stream-1")
        );
    }

    #[tokio::test]
    async fn streaming_lifecycle_counts_tool_indexes_without_counting_deltas_twice() {
        let body = concat!(
            "data: {\"id\":\"stream-tools\",\"model\":\"test-model\"}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"provider-stream-a\",\"function\":{\"name\":\"file_read\",\"arguments\":\"{\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"} \"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"provider-stream-b\",\"function\":{\"name\":\"file_read\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":5,\"total_tokens\":14}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let (base_url, server) = response_server(body, "text/event-stream").await;
        let service = service_at(&base_url);
        let mut events = service.subscribe();
        let mut stream = service
            .stream_chat_with_params(
                LlmInteractionScope::new("unit_stream_tools"),
                "test-model",
                vec![message("stream tools")],
                None,
                None,
                None,
                None,
            )
            .await
            .expect("stream response");
        let response = stream.collect_all().await.expect("collect stream");
        server.await.expect("server task");

        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].id, "provider-stream-a");
        assert_eq!(response.tool_calls[1].id, "provider-stream-b");
        let mut terminal = None;
        while let Ok(event) = events.try_recv() {
            if event.phase == LlmInteractionPhase::Completed {
                terminal = Some(event);
            }
        }
        let terminal = terminal.expect("completed event");
        assert_eq!(terminal.response_tool_call_count, 2);
        assert_eq!(terminal.response_tool_names, vec!["file_read"]);
        assert_eq!(terminal.content_empty, Some(true));
    }

    #[tokio::test]
    async fn thinking_only_length_completion_stays_visible_empty_after_usage_delta() {
        let body = concat!(
            "data: {\"id\":\"thinking-only-1\",\"model\":\"test-model\"}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hidden planning\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":8,\"total_tokens\":12}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let (base_url, server) = response_server(body, "text/event-stream").await;
        let service = service_at(&base_url);
        let mut events = service.subscribe();
        let mut stream = service
            .stream_chat_with_params(
                LlmInteractionScope::new("thinking_only").with_interaction_id("thinking-only-42"),
                "test-model",
                vec![message("private plan request")],
                None,
                None,
                None,
                None,
            )
            .await
            .expect("stream response");
        let response = stream.collect_all().await.expect("collect stream");
        server.await.expect("server task");

        assert_eq!(response.content, "");
        assert_eq!(response.thought.as_deref(), Some("hidden planning"));
        assert_eq!(response.finish_reason, "length");
        let received = std::iter::from_fn(|| events.try_recv().ok()).collect::<Vec<_>>();
        let terminal = received.last().expect("terminal event");
        assert_eq!(terminal.phase, LlmInteractionPhase::Completed);
        assert_eq!(terminal.finish_reason.as_deref(), Some("length"));
        assert_eq!(terminal.content_empty, Some(true));
        assert_eq!(terminal.prompt_tokens, Some(4));
        assert_eq!(terminal.completion_tokens, Some(8));
    }

    #[tokio::test]
    async fn truncated_stream_body_has_specific_safe_failure_class() {
        let secret = "TOP_SECRET_STREAM_FRAGMENT";
        let body = format!(
            "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{secret}\"}}}}]}}\n\n"
        );
        let (base_url, server) = truncated_stream_server(body, 64).await;
        let service = service_at(&base_url);
        let mut events = service.subscribe();
        let mut stream = service
            .stream_chat_with_params(
                LlmInteractionScope::new("truncated_stream")
                    .with_interaction_id("truncated-stream-42"),
                "test-model",
                vec![message("private truncated stream request")],
                None,
                None,
                None,
                None,
            )
            .await
            .expect("HTTP headers are valid");
        let error = stream
            .collect_all()
            .await
            .expect_err("premature EOF must fail the stream");
        server.await.expect("server task");

        assert_eq!(error.error_class(), "stream_transport_body");
        let mut received = Vec::new();
        while let Ok(event) = events.try_recv() {
            received.push(event);
        }
        let terminal = received.last().expect("terminal interaction event");
        assert_eq!(terminal.phase, LlmInteractionPhase::Failed);
        assert_eq!(
            terminal.error_class.as_deref(),
            Some("stream_transport_body")
        );
        let encoded = serde_json::to_string(&received).expect("serialize events");
        assert!(!encoded.contains(secret));
        assert!(!format!("{error:?} {error}").contains(secret));
    }

    #[tokio::test]
    async fn clean_eof_without_chat_done_fails_closed_with_gateway_metadata() {
        let secret = "TOP_SECRET_CLEAN_EOF_FRAGMENT";
        let body = format!(
            "data: {{\"id\":\"provider-clean-eof-id\",\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{secret}\"}},\"finish_reason\":\"stop\"}}]}}\n\n"
        );
        let (base_url, server) = response_server(body, "text/event-stream").await;
        let service = service_at(&base_url);
        let mut events = service.subscribe();
        let mut stream = service
            .stream_chat_with_params(
                LlmInteractionScope::new("clean_eof").with_interaction_id("clean-eof-42"),
                "test-model",
                vec![message("private clean EOF request")],
                None,
                None,
                None,
                None,
            )
            .await
            .expect("HTTP headers are valid");
        let error = stream
            .collect_all()
            .await
            .expect_err("Chat EOF before [DONE] must fail");
        server.await.expect("server task");

        assert_eq!(error.error_class(), "stream_protocol");
        assert_eq!(error.retryable(), Some(false));
        let received = std::iter::from_fn(|| events.try_recv().ok()).collect::<Vec<_>>();
        let terminal = received.last().expect("terminal interaction event");
        assert_eq!(terminal.phase, LlmInteractionPhase::Failed);
        assert_eq!(terminal.error_class.as_deref(), Some("stream_protocol"));
        assert_eq!(terminal.http_status, Some(200));
        assert_eq!(terminal.retryable, Some(false));
        assert_eq!(
            terminal
                .gateway
                .as_ref()
                .and_then(|gateway| gateway.provider_response_id.as_deref()),
            Some("provider-clean-eof-id")
        );
        assert!(!serde_json::to_string(&received).unwrap().contains(secret));
    }

    #[tokio::test]
    async fn responses_failed_and_incomplete_are_failed_interaction_terminals() {
        for (terminal_frame, expected_class, retryable) in [
            (
                "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp-failed\",\"status\":\"failed\",\"error\":{\"code\":\"PRIVATE_PROVIDER_ERROR\"},\"output\":[],\"usage\":null}}\n\n",
                "stream_provider_failed",
                Some(true),
            ),
            (
                "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp-incomplete\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":9,\"total_tokens\":14}}}\n\n",
                "output_token_limit",
                Some(false),
            ),
        ] {
            let (base_url, server) =
                response_server(terminal_frame.to_string(), "text/event-stream").await;
            let service = service_at_with_responses_api(&base_url, true);
            let mut events = service.subscribe();
            let mut stream = service
                .stream_chat_with_params(
                    LlmInteractionScope::new("responses_failure"),
                    "test-model",
                    vec![message("private Responses request")],
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .expect("HTTP headers are valid");
            let error = stream
                .collect_all()
                .await
                .expect_err("non-completed Responses terminal must fail");
            server.await.expect("server task");
            assert_eq!(error.error_class(), expected_class);
            assert_eq!(error.retryable(), retryable);

            let received = std::iter::from_fn(|| events.try_recv().ok()).collect::<Vec<_>>();
            let terminal = received.last().expect("terminal interaction event");
            assert_eq!(terminal.phase, LlmInteractionPhase::Failed);
            assert_eq!(terminal.error_class.as_deref(), Some(expected_class));
            assert_eq!(terminal.http_status, Some(200));
            assert_eq!(terminal.retryable, retryable);
            let encoded = serde_json::to_string(&received).unwrap();
            assert!(!encoded.contains("PRIVATE_PROVIDER_ERROR"));
            assert!(!encoded.contains("private Responses request"));
        }
    }

    #[tokio::test]
    async fn configured_stream_dialect_rejects_cross_protocol_terminals() {
        for (body, use_responses_api) in [
            (
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"must-not-complete-chat\",\"status\":\"completed\",\"output\":[]}}\n\n",
                false,
            ),
            ("data: [DONE]\n\n", true),
        ] {
            let (base_url, server) =
                response_server(body.to_string(), "text/event-stream").await;
            let service = service_at_with_responses_api(&base_url, use_responses_api);
            let mut events = service.subscribe();
            let mut stream = service
                .stream_chat_with_params(
                    LlmInteractionScope::new("cross_dialect_terminal"),
                    "test-model",
                    vec![message("private cross-dialect request")],
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .expect("HTTP headers are valid");
            let error = stream
                .collect_all()
                .await
                .expect_err("the other endpoint's terminal must fail closed");
            server.await.expect("server task");

            assert_eq!(error.error_class(), "stream_protocol");
            assert_eq!(error.retryable(), Some(false));
            let received = std::iter::from_fn(|| events.try_recv().ok()).collect::<Vec<_>>();
            let terminal = received.last().expect("terminal interaction event");
            assert_eq!(terminal.phase, LlmInteractionPhase::Failed);
            assert_eq!(terminal.error_class.as_deref(), Some("stream_protocol"));
            assert_eq!(terminal.retryable, Some(false));
            let encoded = serde_json::to_string(&received).unwrap();
            assert!(!encoded.contains("private cross-dialect request"));
            assert!(!encoded.contains("must-not-complete-chat"));
        }
    }

    #[test]
    fn shared_service_reuses_one_interaction_plane_per_gateway() {
        let gateway = Arc::new(
            UnifiedGateway::new(&GatewaySettings {
                base_url: "http://127.0.0.1:1".to_string(),
                api_key: "test".to_string(),
                default_model: "test-model".to_string(),
                timeout_seconds: 1,
                max_retries: 0,
                retry_base_ms: 1,
                model_mapping: Default::default(),
                use_responses_api: false,
            })
            .expect("gateway"),
        );
        let first = LlmInteractionService::shared(gateway.clone());
        let second = LlmInteractionService::shared_with_event_capacity(gateway, 32);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn request_hook_abort_never_reaches_provider_or_emits_started() {
        let service = service();
        let manager = Arc::new(HookManager::new());
        manager.register(Box::new(crate::tools::hooks::FunctionHook::new(
            "deny-request",
            vec![HookPoint::LlmRequest],
            1,
            |_| crate::tools::hooks::HookResult::Abort,
        )));
        service.set_hook_manager(manager);
        let mut events = service.subscribe();

        let error = service
            .chat(
                LlmInteractionScope::new("blocked")
                    .with_task("iri://task/blocked")
                    .with_interaction_id("blocked-42"),
                "test-model",
                vec![message("must stay local")],
            )
            .await
            .expect_err("request hook must abort");
        assert!(matches!(error, CoreError::InteractionRejected { .. }));
        let phases = [
            events.recv().await.expect("assembled").phase,
            events.recv().await.expect("failed").phase,
        ];
        assert_eq!(
            phases,
            [LlmInteractionPhase::Assembled, LlmInteractionPhase::Failed]
        );
        assert!(events.try_recv().is_err());
        assert_eq!(service.usage_snapshot(), LlmUsageSnapshot::default());
        assert_eq!(
            service.context_quality_snapshot_for_scope("iri://task/blocked"),
            LlmContextQualitySnapshot::default(),
            "a request rejected before provider dispatch must not enter context accounting"
        );
    }

    #[tokio::test]
    async fn reasoning_capability_preflight_rejection_records_no_dispatch_or_http_status() {
        let service = service();
        let mut events = service.subscribe();
        let task_iri = "iri://task/reasoning-preflight";
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "read one file",
                "parameters": {"type": "object", "properties": {}}
            }
        })];

        let error = service
            .chat_with_params_and_options(
                LlmInteractionScope::new("reasoning_preflight")
                    .with_task(task_iri)
                    .with_interaction_id("reasoning-preflight-42"),
                "deepseek-v4-flash",
                vec![message("must stay local")],
                None,
                None,
                Some(tools),
                Some("required"),
                LlmRequestOptions::default()
                    .with_reasoning_effort(crate::config::settings::ReasoningEffort::Low),
            )
            .await
            .expect_err("unsupported reasoning/tool-choice combination must fail locally");
        assert!(matches!(
            error,
            CoreError::InteractionRejected { ref stage, .. } if stage == "provider_capability"
        ));

        let received = std::iter::from_fn(|| events.try_recv().ok()).collect::<Vec<_>>();
        assert_eq!(
            received.first().map(|event| event.phase),
            Some(LlmInteractionPhase::Assembled)
        );
        let terminal = received
            .iter()
            .find(|event| event.phase == LlmInteractionPhase::Failed)
            .expect("failed terminal event");
        assert_eq!(terminal.phase, LlmInteractionPhase::Failed);
        assert_eq!(
            terminal.error_class.as_deref(),
            Some("interaction_rejected")
        );
        assert_eq!(terminal.model_dispatch_count, 0);
        assert_eq!(terminal.http_status, None);
        assert_eq!(terminal.retryable, None);
        assert!(terminal.gateway.is_none());
        assert_eq!(service.usage_snapshot(), LlmUsageSnapshot::default());
        assert_eq!(
            service.context_quality_snapshot_for_scope(task_iri),
            LlmContextQualitySnapshot::default(),
            "a capability rejection before transport must not count as a model dispatch"
        );
        let encoded = serde_json::to_string(&terminal).expect("serialize terminal event");
        assert!(!encoded.contains("http_status"));
        assert!(!encoded.contains("retryable"));
        assert!(!encoded.contains("must stay local"));
        assert!(!received
            .iter()
            .any(|event| event.phase == LlmInteractionPhase::Completed));
    }

    #[tokio::test]
    async fn response_hook_retry_accounts_every_provider_dispatch_once() {
        let body = serde_json::json!({
            "id": "provider-retry",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })
        .to_string();
        let (base_url, server) = repeated_response_server(body, "application/json", 2).await;
        let service = service_at(&base_url);
        let attempts = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let attempts_for_hook = attempts.clone();
        let manager = Arc::new(HookManager::new());
        manager.register(Box::new(crate::tools::hooks::FunctionHook::new(
            "retry-once",
            vec![HookPoint::LlmResponse],
            1,
            move |_| {
                if attempts_for_hook.fetch_add(1, Ordering::SeqCst) == 0 {
                    crate::tools::hooks::HookResult::Retry
                } else {
                    crate::tools::hooks::HookResult::Continue
                }
            },
        )));
        service.set_hook_manager(manager);
        let mut events = service.subscribe();
        service
            .chat(
                LlmInteractionScope::new("retry")
                    .with_task("iri://task/child")
                    .with_usage_scope("iri://task/root"),
                "test-model",
                vec![message("retry")],
            )
            .await
            .expect("second response accepted");
        server.await.expect("server task");

        let mut terminal = None;
        while let Ok(event) = events.try_recv() {
            if event.phase == LlmInteractionPhase::Completed {
                terminal = Some(event);
            }
        }
        let terminal = terminal.expect("completed event");
        assert_eq!(terminal.model_dispatch_count, 2);
        assert_eq!(terminal.billed_prompt_tokens, 6);
        assert_eq!(terminal.billed_completion_tokens, 4);
        assert_eq!(
            service.usage_snapshot_for_scope("iri://task/root"),
            LlmUsageSnapshot {
                prompt_tokens: 6,
                completion_tokens: 4,
            }
        );
        assert_eq!(
            service.usage_snapshot_for_scope("iri://task/child"),
            LlmUsageSnapshot::default(),
            "child must be billed to its root task scope"
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(
            service.context_quality_snapshot_for_scope("iri://task/root"),
            LlmContextQualitySnapshot {
                dispatch_count: 2,
                request_messages: 2,
                request_chars: 10,
                by_kind: LlmContextKindReceipt {
                    user_input: LlmContextClassReceipt {
                        messages: 2,
                        chars: 10,
                    },
                    ..LlmContextKindReceipt::default()
                },
                ..LlmContextQualitySnapshot::default()
            },
            "the same request context is counted for every real response-hook redispatch"
        );
        assert_eq!(
            service.context_quality_snapshot_for_scope("iri://task/child"),
            LlmContextQualitySnapshot::default(),
            "child context must aggregate only under the root usage scope"
        );
    }

    #[test]
    fn scope_accounting_lease_keeps_concurrent_users_and_retires_the_last_snapshot() {
        let service = service();
        let first = service.begin_scope_accounting("iri://task/root");
        let second = service.begin_scope_accounting("iri://task/root");
        let metadata = RequestMetadata::new(
            LlmInteractionScope::new("lease")
                .with_task("iri://task/child")
                .with_usage_scope("iri://task/root"),
            "test-model",
            false,
            &[message("abc")],
            None,
            None,
            None,
            None,
        );
        service.record_context_dispatch(&metadata);
        service.record_usage(&metadata, (2, 1));
        drop(first);
        assert_eq!(
            service
                .context_quality_snapshot_for_scope("iri://task/root")
                .dispatch_count,
            1,
            "one active owner must protect concurrent child aggregation"
        );
        drop(second);
        assert_eq!(
            service.context_quality_snapshot_for_scope("iri://task/root"),
            LlmContextQualitySnapshot::default()
        );
        assert_eq!(
            service.usage_snapshot_for_scope("iri://task/root"),
            LlmUsageSnapshot::default()
        );
    }

    #[test]
    fn idle_scope_accounting_is_bounded_without_evicting_a_leased_scope() {
        let service = service();
        let active = service.begin_scope_accounting("iri://task/active");
        for index in 0..=MAX_IDLE_ACCOUNTING_SCOPES {
            let scope = format!("iri://task/idle-{index}");
            let metadata = RequestMetadata::new(
                LlmInteractionScope::new("bounded").with_task(scope),
                "test-model",
                false,
                &[message("x")],
                None,
                None,
                None,
                None,
            );
            service.record_context_dispatch(&metadata);
        }
        let active_metadata = RequestMetadata::new(
            LlmInteractionScope::new("bounded")
                .with_task("iri://task/active-child")
                .with_usage_scope("iri://task/active"),
            "test-model",
            false,
            &[message("active")],
            None,
            None,
            None,
            None,
        );
        service.record_context_dispatch(&active_metadata);
        assert!(service.context_quality_by_scope.lock().len() <= MAX_IDLE_ACCOUNTING_SCOPES + 1);
        assert_eq!(
            service
                .context_quality_snapshot_for_scope("iri://task/active")
                .dispatch_count,
            1
        );
        drop(active);
        assert_eq!(
            service.context_quality_snapshot_for_scope("iri://task/active"),
            LlmContextQualitySnapshot::default()
        );
    }

    #[test]
    fn actual_request_receipts_classify_history_and_tool_output_without_payloads() {
        let messages = vec![
            message("user secret"),
            ChatMessage {
                role: "assistant".to_string(),
                content: "model secret".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: Some("private reasoning".to_string()),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "tool secret".to_string(),
                name: Some("file_read".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-secret".to_string()),
                reasoning_content: None,
            },
        ];
        let receipts = message_receipts(&messages);
        assert_eq!(
            receipts
                .iter()
                .map(|receipt| receipt.kind)
                .collect::<Vec<_>>(),
            vec![
                ContextFragmentKind::UserInput,
                ContextFragmentKind::ModelHistory,
                ContextFragmentKind::ToolOutput,
            ]
        );
        assert!(receipts[1].has_reasoning);
        let encoded = serde_json::to_string(&receipts).unwrap();
        for secret in [
            "user secret",
            "model secret",
            "private reasoning",
            "tool secret",
            "call-secret",
        ] {
            assert!(!encoded.contains(secret));
        }
    }

    #[tokio::test]
    async fn cancelling_an_inflight_call_emits_cancelled_once() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("local address");
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.expect("accept request");
            std::future::pending::<()>().await;
        });
        let service = service_at(&format!("http://{address}"));
        let mut events = service.subscribe();
        let caller_service = service.clone();
        let call = tokio::spawn(async move {
            caller_service
                .chat(
                    LlmInteractionScope::new("unit_cancel").with_interaction_id("cancel-42"),
                    "test-model",
                    vec![message("cancel me")],
                )
                .await
        });

        assert_eq!(
            events.recv().await.unwrap().phase,
            LlmInteractionPhase::Assembled
        );
        assert_eq!(
            events.recv().await.unwrap().phase,
            LlmInteractionPhase::BeforeDispatch
        );
        call.abort();
        let _ = call.await;
        let terminal = tokio::time::timeout(std::time::Duration::from_secs(1), events.recv())
            .await
            .expect("cancel event timeout")
            .expect("cancel event");
        assert_eq!(terminal.phase, LlmInteractionPhase::Cancelled);
        assert_eq!(terminal.scope.interaction_id, "cancel-42");
        assert!(events.try_recv().is_err());
        server.abort();
    }

    #[tokio::test]
    async fn transport_failure_emits_failed_without_error_payload() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("reserve port");
        let address = listener.local_addr().expect("local address");
        drop(listener);
        let service = service_at(&format!("http://{address}"));
        let mut events = service.subscribe();
        let error = service
            .chat(
                LlmInteractionScope::new("unit_failure").with_interaction_id("failure-42"),
                "test-model",
                vec![message("secret failure payload")],
            )
            .await
            .expect_err("closed port must fail");
        assert!(!error.to_string().is_empty());
        assert_eq!(
            events.recv().await.unwrap().phase,
            LlmInteractionPhase::Assembled
        );
        assert_eq!(
            events.recv().await.unwrap().phase,
            LlmInteractionPhase::BeforeDispatch
        );
        let terminal = events.recv().await.unwrap();
        assert_eq!(terminal.phase, LlmInteractionPhase::Failed);
        assert_eq!(terminal.error_class.as_deref(), Some("transport_connect"));
        assert!(!serde_json::to_string(&terminal)
            .unwrap()
            .contains("secret failure payload"));
        assert!(events.try_recv().is_err());
    }

    #[tokio::test]
    async fn provider_http_failure_emits_safe_status_and_retryability() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind provider-status fixture");
        let address = listener.local_addr().expect("provider-status address");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept provider request");
            let mut request = vec![0_u8; 8192];
            let _ = stream
                .read(&mut request)
                .await
                .expect("read provider request");
            stream
                .write_all(
                    b"HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                )
                .await
                .expect("write provider status");
        });
        let service = service_at(&format!("http://{address}"));
        let mut events = service.subscribe();
        let error = service
            .chat(
                LlmInteractionScope::new("unit_provider_status")
                    .with_interaction_id("provider-status-400"),
                "test-model",
                vec![message("private request content")],
            )
            .await
            .expect_err("HTTP 400 must fail");
        server.await.unwrap();
        assert_eq!(core_error_class(&error), "provider_http_client");
        assert_eq!(
            events.recv().await.unwrap().phase,
            LlmInteractionPhase::Assembled
        );
        assert_eq!(
            events.recv().await.unwrap().phase,
            LlmInteractionPhase::BeforeDispatch
        );
        let terminal = events.recv().await.unwrap();
        assert_eq!(terminal.phase, LlmInteractionPhase::Failed);
        assert_eq!(
            terminal.error_class.as_deref(),
            Some("provider_http_client")
        );
        assert_eq!(terminal.http_status, Some(400));
        assert_eq!(terminal.retryable, Some(false));
        let encoded = serde_json::to_string(&terminal).unwrap();
        assert!(!encoded.contains("private request content"));
        assert!(events.try_recv().is_err());
    }

    #[test]
    fn output_budget_exhaustion_has_a_specific_safe_error_class() {
        let error = CoreError::Internal {
            message: "Responses API response incomplete: max_output_tokens reached".to_string(),
        };
        assert_eq!(core_error_class(&error), "output_token_limit");

        let generic = CoreError::Internal {
            message: "connection reset".to_string(),
        };
        assert_eq!(core_error_class(&generic), "internal");
    }

    #[test]
    fn business_modules_cannot_bypass_the_interaction_facade() {
        let sources = [
            (
                "agent execution",
                include_str!("../core/agent_runner/execution.rs"),
            ),
            (
                "agent streaming",
                include_str!("../core/agent_runner/utils.rs"),
            ),
            ("supervisor", include_str!("../core/sa/intervention.rs")),
            ("planning", include_str!("../core/sa/planning.rs")),
            ("bizagent", include_str!("../core/biz_agent.rs")),
            ("batch", include_str!("../batch/extractor.rs")),
            (
                "skill creator",
                include_str!("../skill_graph/skill_creator.rs"),
            ),
            ("llm client", include_str!("client.rs")),
        ];
        for (name, source) in sources {
            let compact = source
                .chars()
                .filter(|ch| !ch.is_whitespace())
                .collect::<String>();
            for forbidden in [
                ".gateway.chat(",
                ".gateway.chat_with_model(",
                ".gateway.chat_with_params(",
                ".gateway.chat_with_params_and_options(",
                ".gateway.chat_with_params_traced(",
                ".gateway.chat_with_params_traced_and_options(",
                ".gateway.stream_chat_with_params(",
                ".gateway.stream_chat_with_params_and_options(",
            ] {
                assert!(
                    !compact.contains(forbidden),
                    "{name} bypasses LlmInteractionService via {forbidden}"
                );
            }
        }
    }
}
