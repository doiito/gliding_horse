use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::core::event_bus::EventBus;
use crate::core::execution_journal::ToolCallIdentity;
use crate::tools::result_router::ResultRoutingIdentity;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionEventType {
    PhaseChange,
    AgentStatus,
    LlmContent,
    ToolCall,
    ToolResult,
    Thought,
    TokenUsage,
    Error,
    Completion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseChange {
    pub from_phase: String,
    pub to_phase: String,
    pub agent_role: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub agent_id: String,
    pub role: String,
    pub status: String,
    pub turn: u32,
    pub iteration: u32,
    /// When this status was reported
    #[serde(default)]
    pub timestamp: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmContent {
    pub agent_id: String,
    pub role: String,
    pub content_delta: String,
    pub is_reasoning: bool,
    pub token_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Raw provider-issued protocol correlation ID.  This value is preserved
    /// verbatim and must never be replaced with an internal composite key.
    pub call_id: String,
    pub tool_name: String,
    pub arguments_json: String,
    pub agent_id: String,
    /// Isolated L1 execution session which received the provider response.
    #[serde(default)]
    pub l1_session_id: String,
    /// Unique model request which produced this call. Provider call IDs may be
    /// reused by later requests, including within the same L1 session.
    #[serde(default)]
    pub llm_request_id: String,
    /// Collision-resistant internal correlation key. This is observability
    /// metadata only and is never sent back through the provider protocol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_call_key: Option<String>,
    pub sequence: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Raw provider-issued protocol correlation ID, byte-for-byte equivalent
    /// to the corresponding [`ToolCall::call_id`].
    pub call_id: String,
    pub tool_name: String,
    pub result: String,
    pub success: bool,
    /// Whether the tool handler actually ran. Policy and lifecycle rejections
    /// are terminal results with `executed = false` and `success = false`.
    #[serde(default = "legacy_tool_result_was_executed")]
    pub executed: bool,
    /// Stable machine-readable explanation for a rejected or exceptional
    /// terminal path. Human-readable detail remains in `result`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub result_size_bytes: u32,
    pub duration_ms: u32,
    pub agent_id: String,
    #[serde(default)]
    pub l1_session_id: String,
    #[serde(default)]
    pub llm_request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_call_key: Option<String>,
}

fn legacy_tool_result_was_executed() -> bool {
    true
}

/// Stable reason codes for non-standard tool terminal events.
pub mod tool_terminal_reason {
    pub const UNADVERTISED_TOOL: &str = "unadvertised_tool";
    pub const SKILL_BEFORE_SKIPPED: &str = "skill_before_skipped";
    /// A trusted pre-execution hook declined the operation but supplied
    /// recoverable coaching. The handler did not run, so this is policy
    /// guidance rather than a tool execution failure.
    pub const RECOVERABLE_POLICY_GUIDANCE: &str = "recoverable_policy_guidance";
    pub const SKILL_BEFORE_ABORTED: &str = "skill_before_aborted";
    pub const SKILL_BEFORE_RETRY: &str = "skill_before_retry";
    pub const EFFECT_POLICY_DENIED: &str = "effect_policy_denied";
    pub const RECOVERY_GUARD_DENIED: &str = "recovery_guard_denied";
    /// A CA emitted verifier-shaped shell syntax whose status/output cannot be
    /// attributed to exactly one foreground process. The handler never ran.
    pub const VERIFICATION_COMMAND_NOT_ATTRIBUTABLE: &str = "verification_command_not_attributable";
    pub const SOFT_LIMIT_FORCE_FINISH: &str = "soft_limit_force_finish";
    pub const ROLE_POLICY_FORCE_FINISH: &str = "role_policy_force_finish";
    pub const JOURNAL_START_FAILED: &str = "journal_start_failed";
    pub const BATCH_CANCELLED: &str = "batch_cancelled";
    pub const INTERNAL_TERMINATION: &str = "internal_termination";
    pub const RESULT_DISCLOSURE_DENIED: &str = "result_disclosure_denied";
    pub const EXECUTION_FAILED: &str = "execution_failed";
}

fn routing_call_key(identity: &ToolCallIdentity) -> String {
    ResultRoutingIdentity::for_tool_call(
        &identity.agent_id,
        &identity.l1_session_id,
        &identity.llm_request_id,
        &identity.provider_call_id,
    )
    .routing_call_key
}

impl ToolCall {
    pub fn from_identity(
        identity: &ToolCallIdentity,
        tool_name: impl Into<String>,
        arguments_json: impl Into<String>,
        sequence: u32,
    ) -> Self {
        Self {
            call_id: identity.provider_call_id.clone(),
            tool_name: tool_name.into(),
            arguments_json: arguments_json.into(),
            agent_id: identity.agent_id.clone(),
            l1_session_id: identity.l1_session_id.clone(),
            llm_request_id: identity.llm_request_id.clone(),
            routing_call_key: Some(routing_call_key(identity)),
            sequence,
        }
    }
}

impl ToolResult {
    #[allow(clippy::too_many_arguments)]
    pub fn from_identity(
        identity: &ToolCallIdentity,
        tool_name: impl Into<String>,
        result: impl Into<String>,
        success: bool,
        executed: bool,
        reason: Option<&str>,
        result_size_bytes: u32,
        duration_ms: u32,
    ) -> Self {
        Self {
            call_id: identity.provider_call_id.clone(),
            tool_name: tool_name.into(),
            result: result.into(),
            success,
            executed,
            reason: reason.map(str::to_string),
            result_size_bytes,
            duration_ms,
            agent_id: identity.agent_id.clone(),
            l1_session_id: identity.l1_session_id.clone(),
            llm_request_id: identity.llm_request_id.clone(),
            routing_call_key: Some(routing_call_key(identity)),
        }
    }

    /// True only for a kernel-classified, recoverable pre-execution denial.
    /// Requiring all three fields prevents an executed failure, a post-hook
    /// disclosure denial, or a malformed success event from being presented
    /// as harmless guidance merely because its text sounds recoverable.
    pub fn is_recoverable_policy_guidance(&self) -> bool {
        !self.success
            && !self.executed
            && matches!(
                self.reason.as_deref(),
                Some(
                    tool_terminal_reason::RECOVERABLE_POLICY_GUIDANCE
                        | tool_terminal_reason::VERIFICATION_COMMAND_NOT_ATTRIBUTABLE
                )
            )
    }
}

/// Per-AgentRunner publication ledger enforcing a one-call/one-terminal-event
/// contract. It is deliberately keyed by the full durable identity rather
/// than the provider-local `call_id`.
#[derive(Debug, Default)]
pub(crate) struct ToolExecutionEventLedger {
    calls: HashMap<ToolCallIdentity, String>,
    results: HashSet<ToolCallIdentity>,
}

impl ToolExecutionEventLedger {
    pub(crate) fn register_call(
        &mut self,
        identity: &ToolCallIdentity,
        tool_name: &str,
    ) -> Result<(), String> {
        if self.results.contains(identity) {
            return Err("tool result was registered before its call".to_string());
        }
        match self.calls.get(identity) {
            Some(existing) if existing == tool_name => {
                Err("duplicate tool-call execution event".to_string())
            }
            Some(_) => Err("tool-call identity was reused for another tool".to_string()),
            None => {
                self.calls.insert(identity.clone(), tool_name.to_string());
                Ok(())
            }
        }
    }

    pub(crate) fn register_result(
        &mut self,
        identity: &ToolCallIdentity,
        tool_name: &str,
    ) -> Result<(), String> {
        match self.calls.get(identity) {
            None => return Err("orphan tool-result execution event".to_string()),
            Some(existing) if existing != tool_name => {
                return Err("tool-result name does not match its tool call".to_string())
            }
            Some(_) => {}
        }
        if !self.results.insert(identity.clone()) {
            return Err("duplicate terminal tool-result execution event".to_string());
        }
        Ok(())
    }

    pub(crate) fn pending(&self) -> Vec<(ToolCallIdentity, String)> {
        self.calls
            .iter()
            .filter(|(identity, _)| !self.results.contains(*identity))
            .map(|(identity, tool_name)| (identity.clone(), tool_name.clone()))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    pub agent_id: String,
    pub thought: String,
    pub action: String,
    pub emphasis: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub model: String,
    pub turn: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error {
    pub error_type: String,
    pub message: String,
    pub agent_id: String,
    pub recoverable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Completion {
    pub status: String,
    pub summary: String,
    pub total_turns: u32,
    pub total_tool_calls: u32,
    pub total_tokens: u32,
    pub output_json: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvent {
    pub event_id: String,
    pub task_iri: String,
    pub timestamp: i64,
    pub event: ExecutionEventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionEventKind {
    PhaseChange(PhaseChange),
    AgentStatus(AgentStatus),
    LlmContent(LlmContent),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Thought(Thought),
    TokenUsage(TokenUsage),
    Error(Error),
    Completion(Completion),
}

static EVENT_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct ExecutionEventEmitter {
    task_iri: String,
    sender: Option<mpsc::Sender<ExecutionEvent>>,
    event_bus: Option<Arc<EventBus>>,
    include_thought: bool,
    include_tool_calls: bool,
    current_agent_id: String,
    current_phase: String,
    token_total: AtomicU64,
    tool_call_total: AtomicU64,
    turn_total: AtomicU64,
    tool_event_ledger: std::sync::Mutex<ToolExecutionEventLedger>,
}

impl ExecutionEventEmitter {
    pub fn new(
        task_iri: &str,
        sender: Option<mpsc::Sender<ExecutionEvent>>,
        event_bus: Option<Arc<EventBus>>,
    ) -> Self {
        Self {
            task_iri: task_iri.to_string(),
            sender,
            event_bus,
            include_thought: true,
            include_tool_calls: true,
            current_agent_id: String::new(),
            current_phase: "idle".to_string(),
            token_total: AtomicU64::new(0),
            tool_call_total: AtomicU64::new(0),
            turn_total: AtomicU64::new(0),
            tool_event_ledger: std::sync::Mutex::new(ToolExecutionEventLedger::default()),
        }
    }

    pub fn with_options(
        task_iri: &str,
        sender: Option<mpsc::Sender<ExecutionEvent>>,
        event_bus: Option<Arc<EventBus>>,
        include_thought: bool,
        include_tool_calls: bool,
    ) -> Self {
        Self {
            task_iri: task_iri.to_string(),
            sender,
            event_bus,
            include_thought,
            include_tool_calls,
            current_agent_id: String::new(),
            current_phase: "idle".to_string(),
            token_total: AtomicU64::new(0),
            tool_call_total: AtomicU64::new(0),
            turn_total: AtomicU64::new(0),
            tool_event_ledger: std::sync::Mutex::new(ToolExecutionEventLedger::default()),
        }
    }

    pub fn set_current_agent(&mut self, agent_id: &str) {
        self.current_agent_id = agent_id.to_string();
    }

    pub fn set_current_phase(&mut self, phase: &str) {
        self.current_phase = phase.to_string();
    }

    fn generate_event_id() -> String {
        let seq = EVENT_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("evt_{}_{}", chrono::Utc::now().timestamp_millis(), seq)
    }

    fn emit(&self, event: ExecutionEvent) {
        if let Some(ref sender) = self.sender {
            let sender = sender.clone();
            let event = event;
            tokio::spawn(async move {
                if let Err(e) = sender.send(event).await {
                    debug!("Failed to send execution event: {}", e);
                }
            });
        }
    }

    fn emit_to_event_bus(&self, event_type: &str, payload: &str) {
        if let Some(ref event_bus) = self.event_bus {
            let event_bus = event_bus.clone();
            let task_iri = self.task_iri.clone();
            let event_type = event_type.to_string();
            let payload = payload.to_string();
            tokio::spawn(async move {
                event_bus
                    .emit(&task_iri, &event_type, "ExecutionEventEmitter", &payload)
                    .await;
            });
        }
    }

    pub fn emit_phase_change(&self, from: &str, to: &str, role: &str, reason: &str) {
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::PhaseChange(PhaseChange {
                from_phase: from.to_string(),
                to_phase: to.to_string(),
                agent_role: role.to_string(),
                reason: reason.to_string(),
            }),
        };
        self.emit(event.clone());
        self.emit_to_event_bus(
            "LLM_CONTENT",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
        self.emit_to_event_bus(
            "PHASE_CHANGE",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
    }

    pub fn emit_agent_status(
        &self,
        agent_id: &str,
        role: &str,
        status: &str,
        turn: u32,
        iteration: u32,
    ) {
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::AgentStatus(AgentStatus {
                agent_id: agent_id.to_string(),
                role: role.to_string(),
                status: status.to_string(),
                turn,
                iteration,
                timestamp: Some(Utc::now()),
            }),
        };
        self.emit(event.clone());
        self.emit_to_event_bus(
            "TOKEN_USAGE",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
        self.emit_to_event_bus(
            "AGENT_STATUS",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
    }

    pub fn emit_llm_content(
        &self,
        agent_id: &str,
        role: &str,
        delta: &str,
        is_reasoning: bool,
        token_count: u32,
    ) {
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::LlmContent(LlmContent {
                agent_id: agent_id.to_string(),
                role: role.to_string(),
                content_delta: delta.to_string(),
                is_reasoning,
                token_count,
            }),
        };
        self.emit(event.clone());
    }

    pub fn emit_tool_call(
        &self,
        identity: &ToolCallIdentity,
        tool_name: &str,
        args: &Value,
        sequence: u32,
    ) {
        if !self.include_tool_calls {
            return;
        }
        if let Err(error) = self
            .tool_event_ledger
            .lock()
            .expect("execution-event tool ledger poisoned")
            .register_call(identity, tool_name)
        {
            warn!(%error, "ExecutionEventEmitter suppressed invalid tool-call event");
            return;
        }
        self.tool_call_total.fetch_add(1, Ordering::Relaxed);
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::ToolCall(ToolCall::from_identity(
                identity,
                tool_name,
                serde_json::to_string(args).unwrap_or_default(),
                sequence,
            )),
        };
        self.emit(event.clone());
        self.emit_to_event_bus(
            "TOOL_CALL",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
    }

    pub fn emit_tool_result(
        &self,
        identity: &ToolCallIdentity,
        tool_name: &str,
        result: &str,
        success: bool,
        executed: bool,
        reason: Option<&str>,
        size_bytes: u32,
        duration_ms: u32,
    ) {
        if !self.include_tool_calls {
            return;
        }
        if let Err(error) = self
            .tool_event_ledger
            .lock()
            .expect("execution-event tool ledger poisoned")
            .register_result(identity, tool_name)
        {
            warn!(%error, "ExecutionEventEmitter suppressed invalid tool-result event");
            return;
        }
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::ToolResult(ToolResult::from_identity(
                identity,
                tool_name,
                result,
                success,
                executed,
                reason,
                size_bytes,
                duration_ms,
            )),
        };
        self.emit(event.clone());
        self.emit_to_event_bus(
            "TOOL_RESULT",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
    }

    pub fn emit_thought(&self, agent_id: &str, thought: &str, action: &str, emphasis: &[String]) {
        if !self.include_thought {
            return;
        }
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::Thought(Thought {
                agent_id: agent_id.to_string(),
                thought: thought.to_string(),
                action: action.to_string(),
                emphasis: emphasis.to_vec(),
            }),
        };
        self.emit(event.clone());
        self.emit_to_event_bus(
            "THOUGHT",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
    }

    pub fn emit_token_usage(&self, prompt: u32, completion: u32, model: &str, turn: u32) {
        self.token_total
            .fetch_add((prompt + completion) as u64, Ordering::Relaxed);
        self.turn_total.fetch_add(1, Ordering::Relaxed);
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::TokenUsage(TokenUsage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
                model: model.to_string(),
                turn,
            }),
        };
        self.emit(event.clone());
    }

    pub fn emit_error(&self, error_type: &str, message: &str, agent_id: &str, recoverable: bool) {
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::Error(Error {
                error_type: error_type.to_string(),
                message: message.to_string(),
                agent_id: agent_id.to_string(),
                recoverable,
            }),
        };
        self.emit(event.clone());
        self.emit_to_event_bus(
            "EXECUTION_ERROR",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
    }

    pub fn emit_completion(&self, status: &str, summary: &str, output: Option<Value>) {
        let total_tokens = self.token_total.load(Ordering::Relaxed) as u32;
        let total_tool_calls = self.tool_call_total.load(Ordering::Relaxed) as u32;
        let total_turns = self.turn_total.load(Ordering::Relaxed) as u32;

        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::Completion(Completion {
                status: status.to_string(),
                summary: summary.to_string(),
                total_turns,
                total_tool_calls,
                total_tokens,
                output_json: output,
            }),
        };
        self.emit(event.clone());
        self.emit_to_event_bus(
            "EXECUTION_COMPLETE",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
    }

    pub fn get_stats(&self) -> (u32, u32, u32) {
        (
            self.turn_total.load(Ordering::Relaxed) as u32,
            self.tool_call_total.load(Ordering::Relaxed) as u32,
            self.token_total.load(Ordering::Relaxed) as u32,
        )
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionState {
    pub current_phase: String,
    pub current_agent_id: String,
    pub current_agent_role: String,
    pub current_turn: u32,
    pub current_tool: Option<String>,
    pub current_thought_preview: String,
    pub completed_steps: u32,
    pub total_steps: u32,
    pub phase_history: Vec<PhaseHistoryRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseHistoryRecord {
    pub phase: String,
    pub agent_id: String,
    pub started_at: i64,
    pub completed_at: Option<i64>,
    pub status: String,
}

impl ExecutionState {
    pub fn new() -> Self {
        Self {
            current_phase: "idle".to_string(),
            current_agent_id: String::new(),
            current_agent_role: String::new(),
            current_turn: 0,
            current_tool: None,
            current_thought_preview: String::new(),
            completed_steps: 0,
            total_steps: 0,
            phase_history: Vec::new(),
        }
    }

    pub fn update_from_event(&mut self, event: &ExecutionEvent) {
        match &event.event {
            ExecutionEventKind::PhaseChange(pc) => {
                if let Some(last) = self.phase_history.last_mut() {
                    if last.completed_at.is_none() {
                        last.completed_at = Some(event.timestamp);
                        last.status = "completed".to_string();
                    }
                }
                self.phase_history.push(PhaseHistoryRecord {
                    phase: pc.to_phase.clone(),
                    agent_id: self.current_agent_id.clone(),
                    started_at: event.timestamp,
                    completed_at: None,
                    status: "running".to_string(),
                });
                self.current_phase = pc.to_phase.clone();
            }
            ExecutionEventKind::AgentStatus(as_) => {
                self.current_agent_id = as_.agent_id.clone();
                self.current_agent_role = as_.role.clone();
                self.current_turn = as_.turn;
            }
            ExecutionEventKind::LlmContent(lc) => {
                if lc.is_reasoning && lc.content_delta.len() < 100 {
                    self.current_thought_preview = lc.content_delta.clone();
                }
            }
            ExecutionEventKind::ToolCall(tc) => {
                self.current_tool = Some(tc.tool_name.clone());
            }
            ExecutionEventKind::ToolResult(_) => {
                self.current_tool = None;
            }
            ExecutionEventKind::Thought(t) => {
                if t.thought.len() < 100 {
                    self.current_thought_preview = t.thought.clone();
                }
            }
            ExecutionEventKind::Completion(c) => {
                if let Some(last) = self.phase_history.last_mut() {
                    last.completed_at = Some(event.timestamp);
                    last.status = c.status.clone();
                }
                self.completed_steps = self.total_steps;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_state_phase_change() {
        let mut state = ExecutionState::new();
        assert_eq!(state.current_phase, "idle");

        let event = ExecutionEvent {
            event_id: "evt_1".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 1000,
            event: ExecutionEventKind::PhaseChange(PhaseChange {
                from_phase: "idle".to_string(),
                to_phase: "plan".to_string(),
                agent_role: "PA".to_string(),
                reason: "Task started".to_string(),
            }),
        };
        state.update_from_event(&event);
        assert_eq!(state.current_phase, "plan");
        assert_eq!(state.phase_history.len(), 1);
        assert_eq!(state.phase_history[0].phase, "plan");
    }

    #[test]
    fn test_execution_state_agent_status() {
        let mut state = ExecutionState::new();
        let event = ExecutionEvent {
            event_id: "evt_2".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 2000,
            event: ExecutionEventKind::AgentStatus(AgentStatus {
                agent_id: "pa_001".to_string(),
                role: "PA".to_string(),
                status: "running".to_string(),
                turn: 3,
                iteration: 1,
                timestamp: None,
            }),
        };
        state.update_from_event(&event);
        assert_eq!(state.current_agent_id, "pa_001");
        assert_eq!(state.current_agent_role, "PA");
        assert_eq!(state.current_turn, 3);
    }

    #[test]
    fn test_execution_state_tool_call_and_result() {
        let mut state = ExecutionState::new();
        let identity = ToolCallIdentity::new("da_001", "l1-da-1", "request-1", "tc_1");
        let tool_call_event = ExecutionEvent {
            event_id: "evt_3".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 3000,
            event: ExecutionEventKind::ToolCall(ToolCall::from_identity(
                &identity,
                "write_file",
                "{}",
                1,
            )),
        };
        state.update_from_event(&tool_call_event);
        assert_eq!(state.current_tool, Some("write_file".to_string()));

        let tool_result_event = ExecutionEvent {
            event_id: "evt_4".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 3100,
            event: ExecutionEventKind::ToolResult(ToolResult::from_identity(
                &identity,
                "write_file",
                "OK",
                true,
                true,
                None,
                100,
                50,
            )),
        };
        state.update_from_event(&tool_result_event);
        assert_eq!(state.current_tool, None);
    }

    #[test]
    fn test_execution_state_thought() {
        let mut state = ExecutionState::new();
        let event = ExecutionEvent {
            event_id: "evt_5".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 4000,
            event: ExecutionEventKind::Thought(Thought {
                agent_id: "pa_001".to_string(),
                thought: "Need to analyze user requirements".to_string(),
                action: "continue".to_string(),
                emphasis: vec!["Must complete".to_string()],
            }),
        };
        state.update_from_event(&event);
        assert_eq!(
            state.current_thought_preview,
            "Need to analyze user requirements"
        );
    }

    #[test]
    fn test_execution_state_completion() {
        let mut state = ExecutionState::new();
        state.total_steps = 4;
        let event = ExecutionEvent {
            event_id: "evt_6".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 5000,
            event: ExecutionEventKind::Completion(Completion {
                status: "success".to_string(),
                summary: "Task completed".to_string(),
                total_turns: 5,
                total_tool_calls: 3,
                total_tokens: 1500,
                output_json: None,
            }),
        };
        state.update_from_event(&event);
        assert_eq!(state.completed_steps, 4);
    }

    #[test]
    fn test_execution_state_llm_content_reasoning() {
        let mut state = ExecutionState::new();
        let event = ExecutionEvent {
            event_id: "evt_7".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 6000,
            event: ExecutionEventKind::LlmContent(LlmContent {
                agent_id: "da_001".to_string(),
                role: "DA".to_string(),
                content_delta: "Planning solution".to_string(),
                is_reasoning: true,
                token_count: 10,
            }),
        };
        state.update_from_event(&event);
        assert_eq!(state.current_thought_preview, "Planning solution");
    }

    #[tokio::test]
    async fn test_execution_event_emitter_emit() {
        let (tx, mut rx) = mpsc::channel::<ExecutionEvent>(64);
        let emitter = ExecutionEventEmitter::new("iri://task/test", Some(tx), None);

        emitter.emit_phase_change("idle", "plan", "PA", "Test started");
        emitter.emit_agent_status("pa_001", "PA", "running", 1, 1);
        emitter.emit_completion("success", "Done", None);

        let event1 = rx.recv().await.unwrap();
        assert!(matches!(event1.event, ExecutionEventKind::PhaseChange(_)));

        let event2 = rx.recv().await.unwrap();
        assert!(matches!(event2.event, ExecutionEventKind::AgentStatus(_)));

        let event3 = rx.recv().await.unwrap();
        assert!(matches!(event3.event, ExecutionEventKind::Completion(_)));
    }

    #[tokio::test]
    async fn test_execution_event_emitter_with_options() {
        let (tx, mut rx) = mpsc::channel::<ExecutionEvent>(64);
        let emitter =
            ExecutionEventEmitter::with_options("iri://task/test2", Some(tx), None, false, false);

        emitter.emit_thought("pa_001", "thinking...", "continue", &[]);
        let identity = ToolCallIdentity::new("da_001", "l1-da-1", "request-1", "tc_1");
        emitter.emit_tool_call(&identity, "write_file", &serde_json::json!({}), 1);

        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_execution_event_emitter_stats() {
        let (tx, _rx) = mpsc::channel::<ExecutionEvent>(64);
        let emitter = ExecutionEventEmitter::new("iri://task/test3", Some(tx), None);

        emitter.emit_token_usage(100, 50, "deepseek-v4-flash", 1);
        emitter.emit_token_usage(200, 100, "deepseek-v4-flash", 2);

        let (turns, _tool_calls, tokens) = emitter.get_stats();
        assert_eq!(turns, 2);
        assert_eq!(tokens, 450);
    }

    #[test]
    fn tool_events_preserve_raw_provider_id_and_expose_unique_request_identity() {
        let raw = " provider/CALL:Raw#09 ";
        let first = ToolCallIdentity::new("agent", "l1", "request-1", raw);
        let second = ToolCallIdentity::new("agent", "l1", "request-2", raw);
        let call = ToolCall::from_identity(&first, "bash", r#"{"command":"true"}"#, 1);
        let result =
            ToolResult::from_identity(&first, "bash", r#"{"ok":true}"#, true, true, None, 11, 2);
        let later = ToolCall::from_identity(&second, "bash", "{}", 2);

        assert_eq!(call.call_id, raw);
        assert_eq!(result.call_id, raw);
        assert_eq!(call.l1_session_id, "l1");
        assert_eq!(call.llm_request_id, "request-1");
        assert_eq!(call.routing_call_key, result.routing_call_key);
        assert_ne!(call.routing_call_key, later.routing_call_key);

        let encoded = serde_json::to_string(&call).unwrap();
        let decoded: ToolCall = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.call_id.as_bytes(), raw.as_bytes());
    }

    #[test]
    fn recoverable_policy_guidance_requires_unexecuted_failed_terminal_identity() {
        let identity = ToolCallIdentity::new("agent", "l1", "request", "call_0");
        let result = |success, executed, reason| {
            ToolResult::from_identity(
                &identity,
                "bash",
                r#"{"classification":"recoverable_methodology_constraint"}"#,
                success,
                executed,
                reason,
                64,
                0,
            )
        };

        assert!(result(
            false,
            false,
            Some(tool_terminal_reason::RECOVERABLE_POLICY_GUIDANCE),
        )
        .is_recoverable_policy_guidance());
        assert!(result(
            false,
            false,
            Some(tool_terminal_reason::VERIFICATION_COMMAND_NOT_ATTRIBUTABLE),
        )
        .is_recoverable_policy_guidance());
        assert!(!result(
            false,
            true,
            Some(tool_terminal_reason::VERIFICATION_COMMAND_NOT_ATTRIBUTABLE),
        )
        .is_recoverable_policy_guidance());
        assert!(!result(
            false,
            true,
            Some(tool_terminal_reason::RECOVERABLE_POLICY_GUIDANCE),
        )
        .is_recoverable_policy_guidance());
        assert!(!result(
            false,
            false,
            Some(tool_terminal_reason::SKILL_BEFORE_SKIPPED),
        )
        .is_recoverable_policy_guidance());
        assert!(!result(
            false,
            true,
            Some(tool_terminal_reason::RESULT_DISCLOSURE_DENIED),
        )
        .is_recoverable_policy_guidance());
        assert!(
            !result(false, true, Some(tool_terminal_reason::EXECUTION_FAILED),)
                .is_recoverable_policy_guidance()
        );
    }

    #[test]
    fn legacy_tool_result_deserialization_marks_execution_as_legacy_completed() {
        let result: ToolResult = serde_json::from_str(
            r#"{"call_id":"call_0","tool_name":"bash","result":"{}","success":true,"result_size_bytes":2,"duration_ms":1,"agent_id":"da"}"#,
        )
        .unwrap();
        assert!(result.executed);
        assert!(result.l1_session_id.is_empty());
        assert!(result.llm_request_id.is_empty());
        assert!(result.routing_call_key.is_none());
    }

    #[test]
    fn event_ledger_rejects_duplicate_and_orphan_terminal_events() {
        let identity = ToolCallIdentity::new("agent", "l1", "request", "call_0");
        let orphan = ToolCallIdentity::new("agent", "l1", "request", "call_1");
        let mut ledger = ToolExecutionEventLedger::default();

        assert!(ledger.register_result(&orphan, "bash").is_err());
        assert!(ledger.register_call(&identity, "bash").is_ok());
        assert!(ledger.register_call(&identity, "bash").is_err());
        assert_eq!(ledger.pending().len(), 1);
        assert!(ledger.register_result(&identity, "bash").is_ok());
        assert!(ledger.register_result(&identity, "bash").is_err());
        assert!(ledger.pending().is_empty());
    }
}
