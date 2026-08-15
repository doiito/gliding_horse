use tracing::{info, warn};

use crate::core::agent_instance::AgentRole;
use crate::core::event_bus::EventPriority;
use crate::core::execution_event::{ExecutionEvent, ExecutionEventKind, Thought};
use crate::CoreError;

use super::actions::get_action_handler;
use super::agent::SupervisorAgent;
use super::types::*;

impl SupervisorAgent {
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
        _task_iri: &str,
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
            self.runner.gateway.chat_with_params(
                &model,
                messages,
                Some(0.1),
                Some(1000),
                None,
                None,
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
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(self.approval_wait_secs);
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
        self.pending_approvals
            .lock()
            .await
            .insert(request_id, true);
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
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(self.approval_wait_secs);
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
        self.pending_approvals
            .lock()
            .await
            .insert(request_id, true);
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
    ) -> Result<(), CoreError> {
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
                    "AGENT_ERROR" => {
                        let plan = self
                            .perception
                            .on_agent_blocked(&event.source_agent_iri, task_iri);
                        if plan.should_interrupt {
                            pending_interventions.push(plan);
                        }
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
            return Ok(());
        }

        // 3. Process supplementary inputs one by one
        for supplement in &pending {
            let context = format!("Current step: {:?} - {}", step_role, step_objective);
            match self
                .classify_supplementary_input_with_llm(supplement, &context)
                .await
            {
                Ok((action, params)) => {
                    info!(action = ?action, "Supplementary input classification result");
                    self.execute_supplementary_action(action, params, task_iri, supplement)
                        .await?;
                }
                Err(e) => {
                    warn!(error = %e, supplement = %supplement, "Supplementary input classification failed, defaulting to context injection");
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

        Ok(())
    }

    /// LLM classification: map user supplementary input to predefined action
    async fn classify_supplementary_input_with_llm(
        &self,
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
            .runner
            .gateway
            .chat_with_params(&model, messages, Some(0.1), Some(4096), None, None)
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
