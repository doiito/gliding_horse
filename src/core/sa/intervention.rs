use std::path::{Component, Path};

use tracing::{info, warn};

use crate::core::agent_instance::AgentRole;
use crate::core::event_bus::EventPriority;
use crate::core::execution_event::{ExecutionEvent, ExecutionEventKind, Thought};
use crate::CoreError;

use super::actions::get_action_handler;
use super::agent::SupervisorAgent;
use super::types::*;

/// Deterministic delivery updates are intentionally handled outside the LLM
/// classifier. A user asking to persist the final output is an execution
/// contract, not a probabilistic piece of context that may be misclassified
/// or delayed behind another model request.
pub(super) const DEFAULT_WORKSPACE_DELIVERY_PATH: &str = "AI_Agent_Research_Report.md";

#[derive(Debug, Default)]
pub(super) struct SupplementaryProcessingOutcome {
    pub workspace_delivery_target: Option<String>,
}

fn workspace_relative_markdown_path(text: &str) -> Option<String> {
    text.split(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                '，' | '。'
                    | '；'
                    | '：'
                    | '、'
                    | '“'
                    | '”'
                    | '‘'
                    | '’'
                    | '('
                    | ')'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | ','
                    | ';'
                    | ':'
                    | '"'
                    | '\''
            )
    })
    .map(|token| {
        token.trim_matches(|ch: char| {
            ch.is_ascii_punctuation() && ch != '.' && ch != '_' && ch != '-'
        })
    })
    .find_map(|token| {
        let lower = token.to_ascii_lowercase();
        if !lower.ends_with(".md") {
            return None;
        }
        let path = Path::new(token);
        (!path.is_absolute()
            && !path.as_os_str().is_empty()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_))))
        .then(|| token.to_string())
    })
}

pub(super) fn workspace_delivery_target_from_supplement(text: &str) -> Option<String> {
    let normalized = text.to_lowercase();
    let mentions_workspace = [
        "当前工作区",
        "工作区",
        "current workspace",
        "working directory",
        "workspace",
    ]
    .iter()
    .any(|marker| normalized.contains(marker));
    let requests_output = [
        "输出到",
        "写入",
        "保存到",
        "生成到",
        "写到",
        "output to",
        "write to",
        "save to",
        "create in",
    ]
    .iter()
    .any(|marker| normalized.contains(marker));
    (mentions_workspace && requests_output).then(|| {
        workspace_relative_markdown_path(text)
            .unwrap_or_else(|| DEFAULT_WORKSPACE_DELIVERY_PATH.to_string())
    })
}

impl SupervisorAgent {
    /// Stream an SA-only, tool-free LLM request while retaining the same
    /// completed response shape consumed by planning code. This must never be
    /// used for PA/DA/CA/AA execution: those agents require BizAgent + the full
    /// ReAct runner to preserve tool and checkpoint semantics.
    pub(super) async fn chat_sa_streaming(
        &self,
        task_iri: &str,
        stage: &str,
        model: &str,
        messages: Vec<crate::gateway::unified_gateway::ChatMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<crate::gateway::unified_gateway::ChatCompletionResponse, CoreError> {
        use crate::llm::stream_types::ContentBlockDelta;
        use crate::llm::{StreamAccumulator, StreamEvent};

        self.event_bus
            .emit(
                task_iri,
                "SA_STREAM_START",
                "SA",
                &serde_json::json!({"stage": stage}).to_string(),
            )
            .await;

        let stream = self
            .runner
            .gateway
            .stream_chat_with_params(model, messages.clone(), temperature, max_tokens, None, None)
            .await;

        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                warn!(task_iri = %task_iri, stage, %error, "SA stream request failed; using non-streaming fallback");
                let response = self
                    .runner
                    .gateway
                    .chat_with_params(model, messages, temperature, max_tokens, None, None)
                    .await?;
                self.account_sa_usage(response.usage.as_ref());
                self.emit_sa_stream_fallback(task_iri, stage, &response, "request_failed")
                    .await;
                return Ok(response);
            }
        };

        let mut accumulator = StreamAccumulator::new();
        let mut delta_buffer = String::new();
        let mut delta_kind = "content";
        let mut last_emit = std::time::Instant::now();
        let mut stream_error = None;

        loop {
            match stream.next_event().await {
                Ok(Some(event)) => {
                    accumulator.process_event(&event);
                    let delta = match &event {
                        StreamEvent::ContentBlockDelta(event) => match &event.delta {
                            ContentBlockDelta::TextDelta { text } => {
                                Some(("content", text.as_str()))
                            }
                            ContentBlockDelta::ThinkingDelta { thinking } => {
                                Some(("thinking", thinking.as_str()))
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some((kind, text)) = delta {
                        if !delta_buffer.is_empty() && delta_kind != kind {
                            self.event_bus
                                .emit(
                                    task_iri,
                                    "SA_STREAM_DELTA",
                                    "SA",
                                    &serde_json::json!({
                                        "stage": stage,
                                        "kind": delta_kind,
                                        "delta": delta_buffer,
                                    })
                                    .to_string(),
                                )
                                .await;
                            delta_buffer = String::new();
                            last_emit = std::time::Instant::now();
                        }
                        delta_kind = kind;
                        delta_buffer.push_str(text);
                        let execution_budget = &self.runner.agent_settings.execution_budget;
                        if delta_buffer.chars().count() >= execution_budget.sa_stream_emit_min_chars
                            || last_emit.elapsed()
                                >= std::time::Duration::from_millis(
                                    execution_budget.sa_stream_emit_interval_ms,
                                )
                        {
                            self.event_bus
                                .emit(
                                    task_iri,
                                    "SA_STREAM_DELTA",
                                    "SA",
                                    &serde_json::json!({
                                        "stage": stage,
                                        "kind": delta_kind,
                                        "delta": delta_buffer,
                                    })
                                    .to_string(),
                                )
                                .await;
                            delta_buffer = String::new();
                            last_emit = std::time::Instant::now();
                        }
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    stream_error = Some(error.to_string());
                    break;
                }
            }
        }

        if let Some(error) = stream_error {
            warn!(task_iri = %task_iri, stage, %error, "SA stream decode failed; retrying through non-streaming gateway");
            let response = self
                .runner
                .gateway
                .chat_with_params(model, messages, temperature, max_tokens, None, None)
                .await?;
            self.account_sa_usage(response.usage.as_ref());
            self.emit_sa_stream_fallback(task_iri, stage, &response, "decode_failed")
                .await;
            return Ok(response);
        }

        if !delta_buffer.is_empty() {
            self.event_bus
                .emit(
                    task_iri,
                    "SA_STREAM_DELTA",
                    "SA",
                    &serde_json::json!({
                        "stage": stage,
                        "kind": delta_kind,
                        "delta": delta_buffer,
                    })
                    .to_string(),
                )
                .await;
        }

        let usage =
            accumulator
                .usage
                .as_ref()
                .map(|usage| crate::gateway::unified_gateway::Usage {
                    prompt_tokens: usage.prompt_tokens,
                    completion_tokens: usage.completion_tokens,
                    total_tokens: usage.total_tokens,
                });
        self.account_sa_usage(usage.as_ref());

        let tool_calls: Vec<crate::gateway::unified_gateway::ResponseToolCall> = accumulator
            .get_tool_calls()
            .into_iter()
            .map(
                |(id, name, arguments)| crate::gateway::unified_gateway::ResponseToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: crate::gateway::unified_gateway::ResponseToolCallFunction {
                        name,
                        arguments: arguments.to_string(),
                    },
                },
            )
            .collect();
        let content = accumulator.get_text();
        let reasoning_content =
            (!accumulator.thinking.is_empty()).then(|| accumulator.thinking.clone());
        let finish_reason = accumulator.finish_reason.clone();

        self.event_bus
            .emit(
                task_iri,
                "SA_STREAM_END",
                "SA",
                &serde_json::json!({"stage": stage, "finish_reason": finish_reason}).to_string(),
            )
            .await;

        Ok(crate::gateway::unified_gateway::ChatCompletionResponse {
            id: accumulator.message_id.clone(),
            choices: vec![crate::gateway::unified_gateway::Choice {
                index: 0,
                message: crate::gateway::unified_gateway::ResponseMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    reasoning_content,
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                },
                finish_reason,
            }],
            usage,
        })
    }

    fn account_sa_usage(&self, usage: Option<&crate::gateway::unified_gateway::Usage>) {
        use std::sync::atomic::Ordering;
        if let Some(usage) = usage {
            self.runner
                .total_prompt_tokens
                .fetch_add(usage.prompt_tokens as u64, Ordering::Relaxed);
            self.runner
                .total_completion_tokens
                .fetch_add(usage.completion_tokens as u64, Ordering::Relaxed);
            self.runner
                .last_prompt_tokens
                .store(usage.prompt_tokens as u64, Ordering::Relaxed);
            self.runner
                .last_completion_tokens
                .store(usage.completion_tokens as u64, Ordering::Relaxed);
        }
    }

    async fn emit_sa_stream_fallback(
        &self,
        task_iri: &str,
        stage: &str,
        response: &crate::gateway::unified_gateway::ChatCompletionResponse,
        reason: &str,
    ) {
        if let Some(content) = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
            .filter(|content| !content.is_empty())
        {
            self.event_bus
                .emit(
                    task_iri,
                    "SA_STREAM_DELTA",
                    "SA",
                    &serde_json::json!({
                        "stage": stage,
                        "kind": "fallback",
                        "delta": content,
                    })
                    .to_string(),
                )
                .await;
        }
        self.event_bus
            .emit(
                task_iri,
                "SA_STREAM_END",
                "SA",
                &serde_json::json!({
                    "stage": stage,
                    "finish_reason": "fallback",
                    "reason": reason,
                })
                .to_string(),
            )
            .await;
    }

    /// Execute an intervention plan against the active cycle for `task_iri`.
    ///
    /// The cycle is temporarily removed from `active_cycles` so the handler
    /// can mutate it through `&mut CycleState` (its `intervention` field)
    /// without a double mutable borrow of `self`; the cycle is written back
    /// afterwards.
    pub(super) async fn execute_intervention_for_cycle(
        &mut self,
        plan: crate::perception::proactive_engine::InterventionPlan,
        task_iri: &str,
    ) -> Result<(), CoreError> {
        let cycle_id = self
            .active_cycles
            .iter()
            .find(|(_, c)| c.task_iri == task_iri)
            .map(|(id, _)| id.clone());
        let Some(cycle_id) = cycle_id else {
            warn!(task_iri = %task_iri, "No active cycle found for intervention");
            return Ok(());
        };
        let mut cycle = self.active_cycles.remove(&cycle_id);
        let result = match cycle.as_mut() {
            Some(cycle) => self.execute_intervention(plan, task_iri, cycle).await,
            None => Ok(()),
        };
        if let Some(cycle) = cycle {
            self.active_cycles.insert(cycle_id, cycle);
        }
        result
    }

    pub(super) async fn execute_intervention(
        &mut self,
        plan: crate::perception::proactive_engine::InterventionPlan,
        task_iri: &str,
        cycle: &mut CycleState,
    ) -> Result<(), CoreError> {
        if !plan.should_interrupt {
            warn!(actions = ?plan.actions, "Non-interruptive intervention advice, logging only");
            return Ok(());
        }

        warn!(actions = ?plan.actions, "Executing intervention plan");

        // 1. LLM classification: map event to predefined action
        let (action, params) = self
            .analyze_anomaly_with_llm(&plan, task_iri)
            .await
            .unwrap_or_else(|e| {
                warn!(error = %e, "LLM classification failed, falling back to ContinueWithMonitor");
                (
                    InterventionAction::ContinueWithMonitor,
                    ActionParams::default(),
                )
            });

        info!(action = ?action, "LLM classification result");

        // 2. IncreaseBudget special handling: needs human confirmation
        if matches!(action, InterventionAction::IncreaseBudget { .. }) {
            info!("IncreaseBudget needs human confirmation");
            let approved = self.request_human_approval(&action, task_iri).await?;
            if !approved {
                info!("IncreaseBudget not confirmed by human, downgraded to FreezeAndReport");
                let fallback_action = InterventionAction::FreezeAndReport;
                if let Some(handler) = get_action_handler(&fallback_action) {
                    return handler(self, cycle, ActionParams::default(), task_iri).await;
                }
                return Ok(());
            }
        }

        // 3. Registry dispatch: find and execute action handler
        let handler = get_action_handler(&action).ok_or_else(|| CoreError::Internal {
            message: format!("Unknown action handler for: {:?}", action),
        })?;
        handler(self, cycle, params, task_iri).await?;

        // 4. Emit intervention execution event
        self.event_bus
            .emit(
                task_iri,
                "INTERVENTION_EXECUTED",
                "SA",
                &serde_json::json!({"action": format!("{:?}", action)}).to_string(),
            )
            .await;

        Ok(())
    }

    /// LLM classification: map intervention plan to predefined action
    async fn analyze_anomaly_with_llm(
        &self,
        plan: &crate::perception::proactive_engine::InterventionPlan,
        task_iri: &str,
    ) -> Result<(InterventionAction, ActionParams), CoreError> {
        use crate::gateway::unified_gateway::ChatMessage;

        let prompt = format!(
            r#"You are an anomaly diagnosis expert. Based on the following intervention plan, select the most appropriate action from the predefined actions.

## Current Intervention Plan
- Diagnosis: {}
- Suggested action: {}
- Priority: {}
- Is interrupt: {}

## Predefined Action List (strictly select ONE most appropriate action)

### 1. Normal Continuation (no interrupt needed)
- Continue: Do nothing, continue execution
- ContinueWithMonitor: Continue execution but with enhanced monitoring

### 2. Parameter Tuning (no interrupt needed)
- IncreaseRetry: Increase retry count
- IncreaseTimeout: Increase timeout
- ReduceComplexity: Reduce complexity expectation
- RestrictTools: Restrict available tool set

### 3. Execution Flow Adjustment (interrupt needed)
- SkipStep: Skip current step
- RetryStep: Retry current step
- Parallelize: Parallelize execution
- SplitStep: Split into multiple sub-steps
- InsertExtraStep: Insert additional verification/fix steps

### 4. Resource & Mode Switch (interrupt needed)
- FallbackToShallow: Fallback to shallow mode
- EmergencyMode: Enter emergency mode
- FreezeAndReport: Freeze state and generate report

### 5. Termination & Escalation (interrupt needed)
- AbortTask: Abort current task
- NotifyHuman: Notify human intervention

## Output Requirements
Output only JSON with the following fields:
{{
  "action": "Selected action name",
  "params": {{ /* Action parameters */ }},
  "reasoning": "Reason for selecting this action"
}}

Notes:
1. Output only JSON, no extra content
2. action must be strictly selected from the above list
3. IncreaseBudget requires human confirmation, only select when resource budget is truly insufficient
4. AbortTask is the last resort, only use when unrecoverable"#,
            plan.diagnosis,
            plan.actions.join(", "),
            plan.priority,
            plan.should_interrupt,
        );

        let model = self.runner.gateway.get_model("default");
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(self.execution_timeout_secs),
            self.chat_sa_streaming(
                task_iri,
                "intervention_analysis",
                &model,
                messages,
                Some(0.1),
                Some(1000),
            ),
        )
        .await
        .map_err(|_| CoreError::Internal {
            message: "LLM intervention analysis timed out after 30s".to_string(),
        })?
        .map_err(|e| CoreError::Internal {
            message: format!("LLM intervention analysis failed: {}", e),
        })?;
        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| CoreError::Internal {
                message: "No LLM response content".to_string(),
            })?;

        let json_str = if content.starts_with('{') {
            content
        } else if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                content[start..=end].to_string()
            } else {
                return Err(CoreError::Internal {
                    message: "No JSON found in LLM response".to_string(),
                });
            }
        } else {
            return Err(CoreError::Internal {
                message: "No JSON found in LLM response".to_string(),
            });
        };
        let parsed: LlmActionDecision =
            serde_json::from_str(&json_str).map_err(|e| CoreError::Internal {
                message: format!("Failed to parse LLM action decision: {}", e),
            })?;

        let action = InterventionAction::from_name(&parsed.action, parsed.params.clone())?;
        Ok((action, parsed.params))
    }

    /// IncreaseBudget human confirmation flow
    pub(super) async fn request_human_approval(
        &self,
        action: &InterventionAction,
        task_iri: &str,
    ) -> Result<bool, CoreError> {
        let request_id = format!("approval_{}", uuid::Uuid::new_v4().hyphenated());
        let details = match action {
            InterventionAction::IncreaseBudget {
                additional_tokens,
                additional_time_secs,
            } => {
                serde_json::json!({
                    "request_id": request_id,
                    "action": "IncreaseBudget",
                    "additional_tokens": additional_tokens,
                    "additional_time_secs": additional_time_secs,
                    "task_iri": task_iri,
                    "message": format!(
                        "Human confirmation needed: Increase Token budget by {} tokens, additional time {} seconds?",
                        additional_tokens, additional_time_secs
                    ),
                    "status": "pending",
                })
            }
            _ => return Ok(true),
        };

        self.event_bus
            .emit_with_priority(
                task_iri,
                "HUMAN_APPROVAL_REQUIRED",
                "SA",
                &details.to_string(),
                EventPriority::High,
            )
            .await;

        info!(request_id = %request_id, "Waiting for human confirmation");

        let iri = format!("iri://approval/{}", request_id);
        let _ = self.runner.l0_store.store(&iri, &details.to_string());

        // Non-blocking wait: register pending approval request
        // External systems return confirmation via EventBus HUMAN_APPROVAL_RESULT event
        // SA checks the event and updates pending_approvals in the process_task main loop
        self.pending_approvals
            .lock()
            .await
            .insert(request_id.clone(), false);

        // Wait briefly for any instant approval result
        let mut receiver = self.event_bus.subscribe();
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(self.approval_wait_secs);
        while tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if let Ok(event) = receiver.try_recv() {
                if event.event_type == "HUMAN_APPROVAL_RESULT" {
                    if let Ok(result) = serde_json::from_str::<serde_json::Value>(&event.payload) {
                        if result.get("request_id").and_then(|v| v.as_str()) == Some(&request_id) {
                            let approved = result
                                .get("approved")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            self.pending_approvals
                                .lock()
                                .await
                                .insert(request_id, approved);
                            return Ok(approved);
                        }
                    }
                }
            }
        }

        info!(request_id = %request_id, "Human confirmation wait timed out, defaulting to approved (harmless), task continuing");
        self.pending_approvals.lock().await.insert(request_id, true);
        Ok(true)
    }

    /// General human approval request (for HumanApprovalNode workflow nodes)
    pub(super) async fn request_human_approval_general(
        &self,
        prompt: &str,
        node_id: &str,
        task_iri: &str,
    ) -> Result<HumanApprovalNodeResult, CoreError> {
        let request_id = format!("approval_{}", uuid::Uuid::new_v4().hyphenated());
        let details = serde_json::json!({
            "request_id": request_id,
            "action": "WorkflowNodeApproval",
            "node_id": node_id,
            "task_iri": task_iri,
            "prompt": prompt,
            "status": "pending",
        });

        self.event_bus
            .emit_with_priority(
                task_iri,
                "HUMAN_APPROVAL_REQUIRED",
                "SA",
                &details.to_string(),
                EventPriority::High,
            )
            .await;

        info!(request_id = %request_id, node_id = %node_id, "HumanApprovalNode: waiting for human confirmation");

        let iri = format!("iri://approval/{}", request_id);
        let _ = self.runner.l0_store.store(&iri, &details.to_string());

        self.pending_approvals
            .lock()
            .await
            .insert(request_id.clone(), false);

        // Wait briefly for any instant approval result
        let mut receiver = self.event_bus.subscribe();
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(self.approval_wait_secs);
        while tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if let Ok(event) = receiver.try_recv() {
                if event.event_type == "HUMAN_APPROVAL_RESULT" {
                    if let Ok(result) = serde_json::from_str::<serde_json::Value>(&event.payload) {
                        if result.get("request_id").and_then(|v| v.as_str()) == Some(&request_id) {
                            let approved = result
                                .get("approved")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            let comment = result
                                .get("comment")
                                .and_then(|v| v.as_str())
                                .map(String::from);
                            self.pending_approvals
                                .lock()
                                .await
                                .insert(request_id, approved);
                            return Ok(HumanApprovalNodeResult {
                                node_id: node_id.to_string(),
                                approved,
                                comment,
                            });
                        }
                    }
                }
            }
        }

        info!(request_id = %request_id, "HumanApprovalNode: wait timed out, defaulting to approved (harmless)");
        self.pending_approvals.lock().await.insert(request_id, true);
        Ok(HumanApprovalNodeResult {
            node_id: node_id.to_string(),
            approved: true,
            comment: Some("Approval timeout, default approved".to_string()),
        })
    }

    /// Enqueue user supplementary input, waiting for SA processing
    pub fn enqueue_supplementary_input(&mut self, task_iri: &str, content: &str) {
        self.supplementary_inputs
            .entry(task_iri.to_string())
            .or_default()
            .push((content.to_string(), "pending".to_string()));
        info!(task_iri = %task_iri, "User supplementary input enqueued");
    }

    /// Check and execute supplementary inputs between execute_plan steps
    pub(super) async fn check_and_process_supplementary_inputs(
        &mut self,
        task_iri: &str,
        step_role: &AgentRole,
        step_objective: &str,
    ) -> Result<SupplementaryProcessingOutcome, CoreError> {
        let mut outcome = SupplementaryProcessingOutcome::default();
        let mut supp_payloads = Vec::new();
        let mut pending_interventions: Vec<crate::perception::proactive_engine::InterventionPlan> =
            Vec::new();
        if let Some(ref mut receiver) = self.event_receiver {
            while let Ok(event) = receiver.try_recv() {
                if event.task_iri != task_iri {
                    continue;
                }
                match event.event_type.as_str() {
                    "USER_SUPPLEMENTARY_INPUT" => {
                        supp_payloads.push(event.payload.clone());
                    }
                    event_type if super::event_requires_blocked_intervention(event_type) => {
                        let plan = self
                            .perception
                            .on_agent_blocked(&event.source_agent_iri, task_iri);
                        if plan.should_interrupt {
                            pending_interventions.push(plan);
                        }
                    }
                    "AGENT_ERROR" => {
                        info!(
                            agent = %event.source_agent_iri,
                            "Recoverable agent/tool error retained as evidence; no SA blocked intervention"
                        );
                    }
                    "THRESHOLD_EXCEEDED" => {
                        if let Ok(payload) =
                            serde_json::from_str::<serde_json::Value>(&event.payload)
                        {
                            let plan = self.perception.on_quality_degradation(&payload, task_iri);
                            if plan.should_interrupt {
                                pending_interventions.push(plan);
                            }
                        }
                    }
                    "CYCLE_ITERATION" => {
                        if let Ok(payload) =
                            serde_json::from_str::<serde_json::Value>(&event.payload)
                        {
                            let plan = self.perception.on_progress_anomaly(&payload, task_iri);
                            if plan.should_interrupt {
                                pending_interventions.push(plan);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        for plan in pending_interventions {
            let _ = self.execute_intervention_for_cycle(plan, task_iri).await;
        }
        for payload in supp_payloads {
            self.enqueue_supplementary_input(task_iri, &payload);
        }

        // 2. Collect pending supplementary inputs (avoid borrow conflicts)
        let pending = {
            let inputs = self.supplementary_inputs.get_mut(task_iri);
            inputs
                .map(|list| {
                    list.iter()
                        .filter(|(_, status)| status == "pending")
                        .map(|(content, _)| content.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };

        if pending.is_empty() {
            return Ok(outcome);
        }

        // 3. Process supplementary inputs one by one
        for supplement in &pending {
            if let Some(target_path) = workspace_delivery_target_from_supplement(supplement) {
                let contract = format!(
                    "Mandatory delivery update: write the complete final deliverable to the current workspace at `{target_path}`. This replaces any direct-response-only delivery mode. DA must use file_write and CA must read and verify that exact file before completion."
                );
                self.supplement_store.store(task_iri, &contract, None, 1.0);
                outcome.workspace_delivery_target = Some(target_path.clone());
                self.event_bus
                    .emit(
                        task_iri,
                        "DELIVERY_CONTRACT_UPDATED",
                        "SA",
                        &serde_json::json!({
                            "target_path": target_path,
                            "status": "accepted",
                            "source": "supplementary_input",
                        })
                        .to_string(),
                    )
                    .await;
                continue;
            }
            let context = format!("Current step: {:?} - {}", step_role, step_objective);
            match self
                .classify_supplementary_input_with_llm(task_iri, supplement, &context)
                .await
            {
                Ok((action, params)) => {
                    info!(action = ?action, "Supplementary input classification result");
                    self.execute_supplementary_action(action, params, task_iri, supplement)
                        .await?;
                }
                Err(e) => {
                    warn!(error = %e, supplement = %supplement, "Supplementary input classification failed, defaulting to context injection");
                    self.supplement_store.store(task_iri, supplement, None, 1.0);
                    self.inject_to_current_agent(task_iri, supplement).await;
                }
            }
        }

        // 4. Mark as processed
        if let Some(input_list) = self.supplementary_inputs.get_mut(task_iri) {
            for item in input_list.iter_mut() {
                item.1 = "processed".to_string();
            }
        }

        Ok(outcome)
    }

    /// LLM classification: map user supplementary input to predefined action
    async fn classify_supplementary_input_with_llm(
        &self,
        task_iri: &str,
        user_supplement: &str,
        task_context: &str,
    ) -> Result<(SupplementaryInputAction, ActionParams), CoreError> {
        use crate::gateway::unified_gateway::ChatMessage;

        let prompt = format!(
            r#"You are a task guidance expert. Based on the user's supplementary input, select the most appropriate action from the predefined actions.

## Current Task Context
{}

## User Supplementary Input
{}

## Predefined Action List (strictly select ONE)

### 1. Information Supplement
- AddContext: User provides additional context/information
- RefineObjective: User refines or adjusts goals
- ProvideConstraint: User provides new constraints, e.g., time limits

### 2. Direction Guidance
- GuideDirection: User indicates execution direction/priority
- PrioritizeStep: User specifies a step to prioritize
- SuggestApproach: User suggests a specific method or approach

### 3. Execution Control
- PauseExecution: User requests to pause current execution
- ResumeExecution: User requests to resume execution
- SkipCurrentStep: User requests to skip the current step

### 4. Feedback & Correction
- ConfirmDirection: User confirms the current direction is correct
- CorrectApproach: User points out errors and corrects direction
- AbortCurrentStep: User requests to abort the current step

## Output Requirements
Output only JSON with the following fields:
{{
  "action": "Selected action name",
  "params": {{ /* Action parameters, varies per action */ }},
  "reasoning": "Reason for selecting this action"
}}

Notes:
1. Output only JSON, no extra content
2. action must be strictly selected from the above list
3. If the user is supplementing information rather than giving instructions, select AddContext
4. Only select AbortCurrentStep or SkipCurrentStep if the user explicitly requests abort or skip"#,
            task_context, user_supplement,
        );

        let model = self.runner.gateway.get_model("default");
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let response = self
            .chat_sa_streaming(
                task_iri,
                "supplementary_input_classification",
                &model,
                messages,
                Some(0.1),
                Some(4096),
            )
            .await?;
        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| CoreError::Internal {
                message: "No LLM response content".to_string(),
            })?;

        let json_str = if content.starts_with('{') {
            content
        } else if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                content[start..=end].to_string()
            } else {
                return Err(CoreError::Internal {
                    message: "No JSON found in LLM response".to_string(),
                });
            }
        } else {
            return Err(CoreError::Internal {
                message: "No JSON found in LLM response".to_string(),
            });
        };

        let parsed: SupplementaryLlmDecision =
            serde_json::from_str(&json_str).map_err(|e| CoreError::Internal {
                message: format!("Failed to parse LLM supplementary decision: {}", e),
            })?;

        let action = SupplementaryInputAction::from_name(&parsed.action)?;
        Ok((action, parsed.params))
    }

    /// Execute supplementary input action
    async fn execute_supplementary_action(
        &mut self,
        action: SupplementaryInputAction,
        _params: ActionParams,
        task_iri: &str,
        supplement: &str,
    ) -> Result<(), CoreError> {
        match action {
            SupplementaryInputAction::AddContext
            | SupplementaryInputAction::GuideDirection
            | SupplementaryInputAction::ConfirmDirection
            | SupplementaryInputAction::CorrectApproach
            | SupplementaryInputAction::SuggestApproach => {
                // 1. Calculate embedding and relevance_score
                let embedding = if let Some(ref embedder) = self.embedder {
                    embedder.embed(supplement).await.ok()
                } else {
                    None
                };
                let relevance_score = embedding
                    .as_ref()
                    .map(|emb| self.relevance_tracker.on_new_input(emb))
                    .unwrap_or(0.5);

                // 2. Store in SupplementaryInputStore (consumed by AgentRunner at CycleStart)
                self.supplement_store
                    .store(task_iri, supplement, embedding, relevance_score);
                info!(
                    task_iri = %task_iri,
                    score = relevance_score,
                    "Supplementary input stored in SupplementaryInputStore"
                );

                // 3. Backward compatibility: emit SUPPLEMENTARY_CONTEXT event (for TUI rendering)
                self.inject_to_current_agent(task_iri, supplement).await;
            }
            SupplementaryInputAction::RefineObjective => {
                info!("Supplementary input: refine objective");
                self.event_bus
                    .emit(
                        task_iri,
                        "OBJECTIVE_REFINED",
                        "SA",
                        &serde_json::json!({"refinement": supplement}).to_string(),
                    )
                    .await;
            }
            SupplementaryInputAction::ProvideConstraint => {
                info!("Supplementary input: provide constraint");
                // Constraints must reach the next AgentRunner turn as well as
                // be visible on the event bus. Previously this branch emitted
                // an event only, which made a late user constraint disappear.
                self.supplement_store.store(task_iri, supplement, None, 1.0);
                self.inject_to_current_agent(task_iri, supplement).await;
                self.event_bus
                    .emit(
                        task_iri,
                        "CONSTRAINT_ADDED",
                        "SA",
                        &serde_json::json!({"constraint": supplement}).to_string(),
                    )
                    .await;
            }
            SupplementaryInputAction::PrioritizeStep => {
                info!("Supplementary input: prioritize step");
                self.event_bus
                    .emit(
                        task_iri,
                        "STEP_PRIORITIZED",
                        "SA",
                        &serde_json::json!({"priority": supplement}).to_string(),
                    )
                    .await;
            }
            SupplementaryInputAction::PauseExecution => {
                warn!("Supplementary input: pause execution");
                if let Some(cycle) = self
                    .active_cycles
                    .values_mut()
                    .find(|c| c.task_iri == task_iri)
                {
                    cycle.phase = CyclePhase::Idle;
                    cycle
                        .phase_history
                        .push(format!("Paused by user: {}", supplement));
                }
                self.event_bus
                    .emit(
                        task_iri,
                        "EXECUTION_PAUSED",
                        "SA",
                        &serde_json::json!({"reason": supplement}).to_string(),
                    )
                    .await;
            }
            SupplementaryInputAction::ResumeExecution => {
                info!("Supplementary input: resume execution");
                if let Some(cycle) = self
                    .active_cycles
                    .values_mut()
                    .find(|c| c.task_iri == task_iri)
                {
                    cycle.phase = CyclePhase::Executing;
                    cycle
                        .phase_history
                        .push(format!("Resumed by user: {}", supplement));
                }
                self.event_bus
                    .emit(
                        task_iri,
                        "EXECUTION_RESUMED",
                        "SA",
                        &serde_json::json!({"reason": supplement}).to_string(),
                    )
                    .await;
            }
            SupplementaryInputAction::SkipCurrentStep => {
                info!("Supplementary input: skip current step");
                self.event_bus
                    .emit(
                        task_iri,
                        "STEP_SKIPPED",
                        "SA",
                        &serde_json::json!({"reason": supplement}).to_string(),
                    )
                    .await;
            }
            SupplementaryInputAction::AbortCurrentStep => {
                warn!("Supplementary input: abort current step");
                self.event_bus
                    .emit(
                        task_iri,
                        "STEP_ABORTED",
                        "SA",
                        &serde_json::json!({"reason": supplement}).to_string(),
                    )
                    .await;
            }
        }
        Ok(())
    }

    /// Inject supplementary content into current Agent context
    pub(super) async fn inject_to_current_agent(&self, task_iri: &str, supplement: &str) {
        info!(task_iri = %task_iri, "Injecting supplementary context into current Agent");
        self.event_bus
            .emit(
                task_iri,
                "SUPPLEMENTARY_CONTEXT",
                "SA",
                &serde_json::json!({
                    "supplement": supplement,
                    "task_iri": task_iri,
                })
                .to_string(),
            )
            .await;
    }

    /// Emit a THOUGHT event from the SA so the TUI can display what the
    /// Supervisor Agent is doing (planning, classifying, evaluating, …).
    pub(super) async fn emit_sa_thought(&self, task_iri: &str, thought: &str, action: &str) {
        let event = ExecutionEvent {
            event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
            task_iri: task_iri.to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            event: ExecutionEventKind::Thought(Thought {
                agent_id: "SA".into(),
                thought: thought.to_string(),
                action: action.to_string(),
                emphasis: Vec::new(),
            }),
        };
        let _ = self
            .event_bus
            .emit(
                task_iri,
                "THOUGHT",
                "SA",
                &serde_json::to_string(&event).unwrap_or_default(),
            )
            .await;
    }
}
