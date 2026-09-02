use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::debug;

use crate::core::event_bus::EventBus;

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
    pub call_id: String,
    pub tool_name: String,
    pub arguments_json: String,
    pub agent_id: String,
    pub sequence: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub tool_name: String,
    pub result: String,
    pub success: bool,
    pub result_size_bytes: u32,
    pub duration_ms: u32,
    pub agent_id: String,
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
        call_id: &str,
        tool_name: &str,
        args: &Value,
        agent_id: &str,
        sequence: u32,
    ) {
        if !self.include_tool_calls {
            return;
        }
        self.tool_call_total.fetch_add(1, Ordering::Relaxed);
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::ToolCall(ToolCall {
                call_id: call_id.to_string(),
                tool_name: tool_name.to_string(),
                arguments_json: serde_json::to_string(args).unwrap_or_default(),
                agent_id: agent_id.to_string(),
                sequence,
            }),
        };
        self.emit(event.clone());
        self.emit_to_event_bus(
            "TOOL_CALL",
            &serde_json::to_string(&event).unwrap_or_default(),
        );
    }

    pub fn emit_tool_result(
        &self,
        call_id: &str,
        tool_name: &str,
        result: &str,
        success: bool,
        size_bytes: u32,
        duration_ms: u32,
        agent_id: &str,
    ) {
        if !self.include_tool_calls {
            return;
        }
        let event = ExecutionEvent {
            event_id: Self::generate_event_id(),
            task_iri: self.task_iri.clone(),
            timestamp: Utc::now().timestamp_millis(),
            event: ExecutionEventKind::ToolResult(ToolResult {
                call_id: call_id.to_string(),
                tool_name: tool_name.to_string(),
                result: result.to_string(),
                success,
                result_size_bytes: size_bytes,
                duration_ms,
                agent_id: agent_id.to_string(),
            }),
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
        let tool_call_event = ExecutionEvent {
            event_id: "evt_3".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 3000,
            event: ExecutionEventKind::ToolCall(ToolCall {
                call_id: "tc_1".to_string(),
                tool_name: "write_file".to_string(),
                arguments_json: "{}".to_string(),
                agent_id: "da_001".to_string(),
                sequence: 1,
            }),
        };
        state.update_from_event(&tool_call_event);
        assert_eq!(state.current_tool, Some("write_file".to_string()));

        let tool_result_event = ExecutionEvent {
            event_id: "evt_4".to_string(),
            task_iri: "iri://task/1".to_string(),
            timestamp: 3100,
            event: ExecutionEventKind::ToolResult(ToolResult {
                call_id: "tc_1".to_string(),
                tool_name: "write_file".to_string(),
                result: "OK".to_string(),
                success: true,
                result_size_bytes: 100,
                duration_ms: 50,
                agent_id: "da_001".to_string(),
            }),
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
        emitter.emit_tool_call("tc_1", "write_file", &serde_json::json!({}), "da_001", 1);

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
}
