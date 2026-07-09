
use petgraph::prelude::NodeIndex;
use petgraph::Incoming;
use serde::Deserialize;
use tracing::{info, warn};

use crate::core::agent_instance::{AgentInstance, AgentRole};
use crate::core::agent_runner::{TaskContext, TaskResult};
use crate::memory::l2_blackboard::QueryFilter;
use crate::CoreError;

use super::agent::SupervisorAgent;
use super::types::*;

impl SupervisorAgent {
    fn create_agent(&self, role: AgentRole, cycle_id: &str) -> AgentInstance {
        let agent_id = format!("{}_{}_{}", cycle_id, role, uuid::Uuid::new_v4().hyphenated());
        AgentInstance::new(agent_id, role)
    }

    async fn dispatch_agent(
        &self,
        role: AgentRole,
        context: TaskContext,
        cycle_id: &str,
        plan_step: Option<PlanStep>,
        timeout_secs: u64,
    ) -> Result<TaskResult, CoreError> {
        let agent = self.create_agent(role, cycle_id);

        // Query context from L2 blackboard (replaces prev_summary)
        // Use query_nodes_filtered for role/cycle-aware context (AA uses prev_summary)
        let prev_agent_summary = context.prev_agent_summary.clone();
        let prev_summary = if role == AgentRole::Act {
            prev_agent_summary.clone()
        } else if let Some(blackboard) = &self.blackboard {
            let prev_role = match role {
                AgentRole::Do => Some(AgentRole::Plan),
                AgentRole::Check => Some(AgentRole::Do),
                _ => None,
            };
            // Only apply cycle_id filter when we have a specific role filter
            // (PA sees all context nodes; DA/CA see only the previous agent's output from this cycle)
            let filter = QueryFilter {
                role: prev_role.clone(),
                cycle_id: prev_role.map(|_| cycle_id.to_string()),
                node_type: None,
            };
            let nodes = blackboard.query_nodes_filtered(&context.task_iri, &filter).unwrap_or_default();
            if !nodes.is_empty() {
                let mut contents: Vec<String> = Vec::new();
                let mut summaries: Vec<String> = Vec::new();
                for n in nodes.iter() {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&n.json_ld) {
                        let role = parsed.get("role").and_then(|v| v.as_str()).unwrap_or("");
                        let prefix = if !role.is_empty() { format!("[{}] ", role) } else { String::new() };
                        // Prefer content field (full LLM output)
                        if let Some(content) = parsed.get("content").and_then(|s| s.as_str()) {
                            let trimmed = content.trim();
                            if !trimmed.is_empty() && trimmed.len() > 20 {
                                contents.push(format!("{}{}", prefix, trimmed));
                                continue;
                            }
                        }
                        // Fallback to summary field
                        if let Some(summary) = parsed.get("summary").and_then(|s| s.as_str()) {
                            let trimmed = summary.trim();
                            if !trimmed.is_empty() {
                                summaries.push(format!("{}{}", prefix, trimmed));
                            }
                        }
                    }
                }
                // Prefer content with substance
                if !contents.is_empty() {
                    Some(contents.join("\n\n---\n\n"))
                } else if !summaries.is_empty() {
                    Some(summaries.join("\n"))
                } else {
                    prev_agent_summary.clone()
                }
            } else {
                prev_agent_summary.clone()
            }
        } else {
            prev_agent_summary.clone()
        };

        let context = if let Some(ref summary) = prev_summary {
            context.with_prev_summary(summary)
        } else {
            context
        };

        info!(agent_id = %agent.agent_id, role = ?role, task = %context.task_iri, "Dispatching agent with isolation");

        self.event_bus
            .emit(&context.task_iri, &format!("{:?}_STARTED", role), &agent.agent_id,
                &serde_json::json!({"cycle_id": cycle_id}).to_string())
            .await;

        // Execute with independent BizAgent instance (agent isolation), with optional per-node timeout
        let iri = context.task_iri.clone();
        let exec_fut = self.runner.execute_with_biz_agent(&agent, context, plan_step);
        let result = if timeout_secs > 0 {
            match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), exec_fut).await {
                Ok(r) => r?,
                Err(_) => {
                    warn!(role = ?role, timeout = timeout_secs, "Agent dispatch timed out");
                    return Ok(TaskResult {
                        task_iri: iri,
                        status: "timeout".to_string(),
                        summary: format!("Agent {:?} timed out after {} seconds", role, timeout_secs),
                        output: None,
                        jsonld_output: None,
                        artifacts: vec![],
                        errors: vec![],
                        turn_count: 0,
                        tool_call_count: 0,
                        five_w2h_updates: None,
                        tracked_actions: vec![],
                        archive_iri: None,
                    });
                }
            }
        } else {
            exec_fut.await?
        };

        match result.status.as_str() {
            "success" => {
                let task_result = serde_json::json!({"status": "success", "summary": &result.summary});
                self.perception.on_task_end(&task_result, &result.task_iri).await;
            }
            "failed" => {
                let task_result = serde_json::json!({"status": "failed", "summary": &result.summary});
                self.perception.on_task_end(&task_result, &result.task_iri).await;
            }
            _ => {}
        }

        self.event_bus
            .emit(&result.task_iri, &format!("{:?}_COMPLETED", role), &agent.agent_id,
                &serde_json::json!({"status": &result.status, "summary": &result.summary}).to_string())
            .await;

        Ok(result)
    }

    async fn dispatch_agents_parallel(
        &self,
        role: AgentRole,
        count: usize,
        base_objective: &str,
        task_iri: &str,
        cycle_id: &str,
        max_iterations: u32,
        timeout_secs: u64,
    ) -> Result<Vec<TaskResult>, CoreError> {
        let _ = self.event_bus.emit(
            task_iri,
            "PARALLEL_START",
            "system:sa",
            &serde_json::json!({
                "role": format!("{:?}", role),
                "count": count,
                "cycle_id": cycle_id,
            }).to_string(),
        ).await;

        let runner = self.runner.clone();
        let mut handles = Vec::new();

        for i in 0..count {
            let objective = format!("[{}-{}] {}", role, i + 1, base_objective);
            let ctx = TaskContext::new(task_iri, &objective, max_iterations);
            let tid = cycle_id.to_string();
            let runner_clone = runner.clone();

            handles.push(tokio::spawn(async move {
                let agent_id = format!("{}_{}_{}", tid, role, uuid::Uuid::new_v4().hyphenated());
                let mut agent = AgentInstance::new(agent_id, role);
                let iri = ctx.task_iri.clone();
                if timeout_secs > 0 {
                    match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), runner_clone.execute(&mut agent, ctx)).await {
                        Ok(r) => r,
                        Err(_) => Ok(TaskResult {
                            task_iri: iri,
                            status: "timeout".to_string(),
                            summary: format!("Parallel agent {:?} timed out after {}s", role, timeout_secs),
                            output: None,
                            jsonld_output: None,
                            artifacts: vec![],
                            errors: vec![],
                            turn_count: 0,
                            tool_call_count: 0,
                            five_w2h_updates: None,
                            tracked_actions: vec![],
                            archive_iri: None,
                        }),
                    }
                } else {
                    runner_clone.execute(&mut agent, ctx).await
                }
            }));
        }

        let mut results = Vec::new();
        for h in handles {
            match h.await {
                Ok(Ok(res)) => results.push(res),
                Ok(Err(e)) => warn!("Parallel agent failed: {}", e),
                Err(e) => warn!("Parallel agent panicked: {}", e),
            }
        }

        let _ = self.event_bus.emit(
            task_iri,
            "PARALLEL_COMPLETE",
            "system:sa",
            &serde_json::json!({
                "role": format!("{:?}", role),
                "success_count": results.len(),
                "total_count": count,
            }).to_string(),
        ).await;

        info!(count = results.len(), "Parallel agents completed");
        Ok(results)
    }

    pub async fn execute_plan(
        &mut self,
        plan: ExecutionPlan,
        task_iri: &str,
        user_input: &str,
        mut five_w2h: crate::core::five_w2h::Task5W2H,
        five_w2h_iri: &str,
        resumed_messages: Option<Vec<crate::gateway::unified_gateway::ChatMessage>>,
        initial_prev_summary: Option<String>,
    ) -> Result<TaskResult, CoreError> {
        
        let cycle_id = self
            .active_cycles
            .iter()
            .find(|(_, c)| c.task_iri == task_iri)
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| format!("cycle_{}", uuid::Uuid::new_v4().hyphenated()));
        
        let _task_id = task_iri.strip_prefix("iri://task/")
            .unwrap_or_else(|| task_iri.strip_prefix("iri://").unwrap_or(task_iri));

        if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
            cycle.phase = CyclePhase::Dispatching;
            cycle.phase_history.push("Dispatching".to_string());
        }

        info!(plan_id = %plan.plan_id, steps = plan.steps.len(), "Executing plan with detailed steps");

        if let Some(prefetch) = &self.prefetch_engine {
            let entities: Vec<String> = plan.steps.iter()
                .filter_map(|s| {
                    if s.expected_output.starts_with("iri://") {
                        Some(s.expected_output.clone())
                    } else {
                        None
                    }
                })
                .collect();
            prefetch.on_intent_change(&plan.description, &entities).await;
        }

        let mut last_result: Option<TaskResult> = None;
        let mut prev_summary: Option<String> = initial_prev_summary;
        // Track the Do agent's output separately so AA can access it alongside CA's evaluation.
        let mut da_output: Option<String> = None;

        // Resume mode: determine which phase to start from
        // Load latest checkpoint from L0 to resolve phase tags
        let resume_skip_phases: Vec<AgentRole> = if resumed_messages.is_some() {
            let cm = crate::core::checkpoint::CheckpointManager::with_persistence(self.runner.l0_store.clone());
            let skip_roles = cm.restore_latest_with_skip_roles(task_iri)
                .ok()
                .flatten()
                .map(|(_, roles)| roles)
                .unwrap_or_else(|| vec!["Plan".to_string()]);
            skip_roles.iter().filter_map(|r| {
                match r.as_str() {
                    "Plan" => Some(AgentRole::Plan),
                    "Do" => Some(AgentRole::Do),
                    "Check" => Some(AgentRole::Check),
                    "Act" => Some(AgentRole::Act),
                    _ => None,
                }
            }).collect()
        } else {
            vec![]
        };
        info!("[resume] skip phases: {:?}", resume_skip_phases);

        // Resume mode: prefer restoring from checkpoint's prev_summary field
        // If no prev_summary in checkpoint, extract PA output from history messages
        let resume_prev_summary: Option<String> = if resumed_messages.is_some() {
            // Try to read saved prev_summary from L0 checkpoint
            let cm = crate::core::checkpoint::CheckpointManager::with_persistence(self.runner.l0_store.clone());
            let from_cp: Option<String> = cm.restore_latest(task_iri).ok().flatten()
                .and_then(|cp| cp.prev_summary);
            if from_cp.is_some() {
                from_cp
            } else {
                // Fallback: extract PA phase output from history messages as prev_summary
                resumed_messages.as_ref().and_then(|msgs| {
                    let mut found_first_user = false;
                    for msg in msgs.iter() {
                        if msg.role == "user" && !found_first_user {
                            found_first_user = true;
                            continue;
                        }
                        if msg.role == "assistant" && found_first_user {
                            return Some(msg.content.clone());
                        }
                    }
                    msgs.iter().rev()
                        .find(|m| m.role == "assistant")
                        .map(|m| m.content.clone())
                })
            }
        } else {
            None
        };

        let _task_level = match plan.task_complexity {
            TaskComplexity::Instant => "Instant",
            TaskComplexity::Simple => "Simple",
            TaskComplexity::Standard => "Standard",
            TaskComplexity::Complex => "Complex",
            TaskComplexity::Exploratory => "Complex",
            TaskComplexity::Emergency => "Standard",
            TaskComplexity::Recursive => "Recursive",
        };

        // --- Unified DAG execution path ---
        // Convert ExecutionPlan to DAG (LLM path adapter) or use external JSON-LD DAG directly (--workflow path)
        let dag = if let Some(ref dag_jsonld) = plan.dag_jsonld {
            let def = crate::core::workflow::loader::load_workflow_jsonld(dag_jsonld)
                .map_err(|e| CoreError::Internal { message: format!("Workflow parsing failed: {}", e) })?;
            crate::core::workflow::loader::build_dag(&def)
                .map_err(|e| CoreError::Internal { message: format!("DAG build failed: {}", e) })?
        } else {
            let wf = crate::core::workflow::adapter::plan_to_workflow(&plan, task_iri);
            crate::core::workflow::loader::build_dag(&wf)
                .map_err(|e| CoreError::Internal { message: format!("DAG build failed: {}", e) })?
        };
        let order = crate::core::workflow::loader::topological_order(&dag)
            .map_err(|e| CoreError::Internal { message: format!("Topological sort failed: {}", e) })?;

        let mut completed_node_results: std::collections::HashMap<String, crate::core::workflow::NodeResult> = std::collections::HashMap::new();
        let mut skip_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();

        // ── Compute topological depth for wave-based parallel dispatch ──
        // Depth = longest path from entry node (all predecessors must complete before this depth)
        let mut node_depth: std::collections::HashMap<NodeIndex, usize> = std::collections::HashMap::new();
        for &nidx in &order {
            let depth = dag.graph.neighbors_directed(nidx, Incoming)
                .filter_map(|p| node_depth.get(&p))
                .max()
                .map(|d| d + 1)
                .unwrap_or(0);
            node_depth.insert(nidx, depth);
        }
        // Group consecutive order indices with the same depth into waves
        let mut waves: Vec<Vec<usize>> = Vec::new();
        {
            let mut pos = 0;
            while pos < order.len() {
                let d = node_depth[&order[pos]];
                let mut wave = vec![pos];
                pos += 1;
                while pos < order.len() && node_depth[&order[pos]] == d {
                    wave.push(pos);
                    pos += 1;
                }
                waves.push(wave);
            }
        }

        // Execute DAG wave by wave — nodes at the same topological depth have all deps met
        // and can run concurrently via join_all
        for wave in &waves {
            // ═══════════════════════════════════════════════════════════
            // Phase 1: Pre-process each node in the wave
            // (skip checks, HumanApprovalNode, objective building, context)
            // ═══════════════════════════════════════════════════════════
            struct WaveTask {
                wi: usize,
                ni: NodeIndex,
                step: PlanStep,
                ctx: TaskContext,
                timeout_secs: u64,
            }
            let mut agent_tasks: Vec<WaveTask> = Vec::new();

            for &wi in wave {
                let ni = order[wi];
                let nd = &dag.graph[ni].def;
                let step = crate::core::workflow::adapter::node_to_planstep(nd);

                // Check skip set (branch jump from HumanApprovalNode)
                if skip_nodes.contains(&nd.id) {
                    info!(node_id = %nd.id, "HumanApprovalNode branch jump: skipping this node");
                    continue;
                }

                // Resume mode: skip completed phases
                if resume_skip_phases.contains(&step.role) {
                    info!(role = ?step.role, "[resume] skipping completed phase");
                    if prev_summary.is_none() {
                        prev_summary = resume_prev_summary.clone().or_else(|| Some("Restored from checkpoint, preceding phase completed.".to_string()));
                    }
                    continue;
                }

                // HumanApprovalNode: blocking, runs inline in the wave's pre-phase
                if nd.node_type == "HumanApprovalNode" {
                    let approval = self.request_human_approval_general(
                        &nd.approval_prompt, &nd.id, task_iri
                    ).await?;

                    let status = if approval.approved { "approved" } else { "rejected" };
                    let summary = format!("[HumanApproval] {}: {}",
                        if approval.approved { "Approved" } else { "Rejected" },
                        approval.comment.as_deref().unwrap_or(""));

                    completed_node_results.insert(nd.id.clone(), crate::core::workflow::NodeResult {
                        node_id: nd.id.clone(),
                        status: status.to_string(),
                        summary: summary.clone(),
                        archive_iri: None,
                        turn_count: 0,
                        tool_call_count: 0,
                        error: if approval.approved { None } else { Some("User rejected".to_string()) },
                        output: None,
                        artifacts: vec![],
                    });

                    let ha_result = TaskResult {
                        task_iri: task_iri.to_string(),
                        status: status.to_string(),
                        summary: summary.clone(),
                        output: None,
                        jsonld_output: None,
                        artifacts: vec![],
                        errors: vec![],
                        turn_count: 0,
                        tool_call_count: 0,
                        five_w2h_updates: None,
                        tracked_actions: vec![],
                        archive_iri: None,
                    };
                    prev_summary = Some(format!("## Human Approval Result\n{}", summary));
                    last_result = Some(ha_result);

                    // Branch jump handling (rejected → skip to reject target)
                    if !approval.approved {
                        if let Some(ref reject_target) = nd.approval_next_on_reject {
                            let mut found = false;
                            for skip_idx in (wi + 1)..order.len() {
                                let sid = dag.graph[order[skip_idx]].def.id.clone();
                                if sid == *reject_target { found = true; break; }
                                skip_nodes.insert(sid);
                            }
                            if !found {
                                for skip_idx in (wi + 1)..order.len() {
                                    skip_nodes.insert(dag.graph[order[skip_idx]].def.id.clone());
                                }
                            }
                        }
                    }
                    // Approved → skip to approve target
                    if approval.approved {
                        if let Some(ref approve_target) = nd.approval_next_on_approve {
                            let mut found = false;
                            for skip_idx in (wi + 1)..order.len() {
                                let sid = dag.graph[order[skip_idx]].def.id.clone();
                                if sid == *approve_target { found = true; break; }
                                skip_nodes.insert(sid);
                            }
                            if !found {
                                for skip_idx in (wi + 1)..order.len() {
                                    skip_nodes.insert(dag.graph[order[skip_idx]].def.id.clone());
                                }
                            }
                        }
                    }

                    info!(node_id = %nd.id, status = %status, "HumanApprovalNode processing complete");
                    continue;
                }

                // ── Supplementary input processing & pause check ──
                self.check_and_process_supplementary_inputs(
                    task_iri, &step.role, &step.objective,
                ).await?;
                // Cycle timeout check
                {
                    let cycle_start = self.active_cycles.get(&cycle_id).map(|c| c.started_at);
                    if let Some(started_at) = cycle_start {
                        let elapsed = chrono::Utc::now().signed_duration_since(started_at);
                        if elapsed.num_seconds() > self.perception.cycle_timeout_secs() {
                            let intervention = self.perception.on_cycle_timeout(&cycle_id, task_iri, elapsed.num_seconds() as f64);
                            if intervention.should_interrupt {
                                let _ = tokio::time::timeout(
                                    std::time::Duration::from_secs(self.execution_timeout_secs),
                                    self.execute_intervention(intervention, task_iri),
                                ).await;
                            }
                        }
                    }
                }
                // Pause check
                let paused = self.active_cycles.get(&cycle_id)
                    .map(|c| c.phase == CyclePhase::Idle)
                    .unwrap_or(false);
                if paused {
                    info!(step_id = %step.step_id, role = ?step.role, "Execution paused, waiting for resume");
                    loop {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        let mut payloads = Vec::new();
                        if let Some(ref mut receiver) = self.event_receiver {
                            while let Ok(event) = receiver.try_recv() {
                                if event.task_iri != task_iri { continue; }
                                if event.event_type == "USER_SUPPLEMENTARY_INPUT" {
                                    payloads.push(event.payload.clone());
                                }
                            }
                        }
                        for payload in payloads {
                            self.enqueue_supplementary_input(task_iri, &payload);
                        }
                        let resumed = self.active_cycles.get(&cycle_id)
                            .map(|c| c.phase == CyclePhase::Executing)
                            .unwrap_or(false);
                        if resumed { break; }
                    }
                }

                // ── Check for parallel_groups (ExecutionPlan same-role parallelism) ──
                if plan.parallel_groups.iter().any(|g| g.len() > 1 && g.contains(&step.role)) {
                    let matching_groups: Vec<_> = plan.parallel_groups.iter()
                        .filter(|g| g.contains(&step.role))
                        .collect();
                    let parallel_group = match matching_groups.first() {
                        Some(g) => (*g).clone(),
                        None => {
                            warn!(role = ?step.role, "No parallel group found despite any() check");
                            continue;
                        }
                    };
                    let count = parallel_group.len();
                    let results = self.dispatch_agents_parallel(
                        step.role, count, &step.objective, task_iri, &cycle_id, self.max_iterations, nd.timeout_secs,
                    ).await?;

                    let failed = results.iter().find(|r| r.status == "failed");
                    if let Some(f) = failed {
                        warn!(role = ?step.role, step_id = %step.step_id, "Parallel agent failed");
                        return Ok(TaskResult {
                            task_iri: task_iri.to_string(),
                            status: "partial_failure".to_string(),
                            summary: format!("Some parallel {:?} agents failed", step.role),
                            output: None,
                            jsonld_output: None,
                            artifacts: Vec::new(),
                            errors: f.errors.clone(),
                            turn_count: results.iter().map(|r| r.turn_count).sum(),
                            tool_call_count: results.iter().map(|r| r.tool_call_count).sum(),
                            five_w2h_updates: None,
                            tracked_actions: Vec::new(),
                            archive_iri: None,
                        });
                    }

                    let combined_summary: String = results.iter()
                        .map(|r| {
                            let iri_part = r.archive_iri.as_ref()
                                .map(|iri| format!(" | read_agent_output query: {}", iri))
                                .unwrap_or_default();
                            format!("[{}] {}{}", r.task_iri, r.summary, iri_part)
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    prev_summary = Some(combined_summary);
                    last_result = results.into_iter().last();
                    continue;
                }

                // ── Build objective (PDCA role-specific templates) ──
                let cycle_hints = self.active_cycles.values()
                    .find(|c| c.task_iri == task_iri)
                    .map(|c| c.experience_hints.clone())
                    .unwrap_or_default();
                let hints_block = if cycle_hints.is_empty() {
                    String::new()
                } else {
                    format!("\n\n## Historical Experience\n{}", cycle_hints.iter().map(|h| format!("- {}", h)).collect::<Vec<_>>().join("\n"))
                };
                let objective = match (&prev_summary, step.role) {
                    (Some(summary), AgentRole::Plan) => {
                        format!("{}\n\n## User Task\n{}{}\n\n## Feedback from Previous PDCA Cycle\n{}\n\nPlease create a detailed execution plan for the above user task, addressing all feedback from the previous cycle.", step.objective, user_input, hints_block, summary)
                    }
                    (Some(summary), AgentRole::Do) => {
                        format!("{}\n\nUpper PA's Plan:\n{}{}\n\nPlease execute the task according to the plan.", step.objective, summary, hints_block)
                    }
                    (Some(summary), AgentRole::Check) => {
                        format!("{}\n\nExecution Results:\n{}{}\n\nPlease verify whether the execution results are correct and complete.", step.objective, summary, hints_block)
                    }
                    (Some(summary), AgentRole::Act) => {
                        let da_context = da_output.as_ref()
                            .filter(|_| da_output.as_ref().map_or(false, |da| da != summary))
                            .map(|da| format!("\n\n## Execution Results\n{}", da))
                            .unwrap_or_default();
                        format!("{}\n\n## Original Task\n{}{}\n\n## Check Conclusions\n{}{}\n\nPlease make final decisions and summarize.", step.objective, user_input, da_context, summary, hints_block)
                    }
                    (None, AgentRole::Plan) => {
                        format!("{}\n\n## User Task\n{}{}\n\nPlease create a detailed execution plan for the above user task.", step.objective, user_input, hints_block)
                    }
                    _ => step.objective.clone(),
                };

                // ── Build context ──
                let mut context = TaskContext::new(task_iri, &objective, self.max_iterations)
                    .with_original_task(user_input)
                    .with_step_info(&step.expected_output, &step.success_criteria)
                    .with_cycle_id(&cycle_id);
                context = context.with_five_w2h(five_w2h_iri, five_w2h.clone());

                // Resume mode: history messages on first executed step
                let is_first_executed_step = if resume_skip_phases.is_empty() {
                    wi == 0
                } else {
                    !resume_skip_phases.contains(&step.role) 
                        && plan.steps[..wi].iter().all(|s| resume_skip_phases.contains(&s.role))
                };
                if is_first_executed_step {
                    if let Some(ref msgs) = resumed_messages {
                        let turn_count = msgs.iter().filter(|m| m.role == "assistant").count() as u32;
                        let tool_count = msgs.iter().filter(|m| m.role == "tool" || m.tool_call_id.is_some()).count() as u32;
                        context = context.with_resumed_messages(msgs.clone(), turn_count, tool_count);
                    }
                }
                if let Some(ref pv) = prev_summary {
                    context = context.with_prev_summary(pv);
                }

                // ── Thought emission ──
                let role_name = format!("{:?}", step.role);
                self.emit_sa_thought(task_iri,
                    &format!("Wave step {}/{}: dispatching {} — {}",
                        wi + 1, plan.steps.len(), role_name, step.objective),
                    &format!("dispatch_{}", role_name.to_lowercase())).await;

                agent_tasks.push(WaveTask {
                    wi,
                    ni,
                    step,
                    ctx: context,
                    timeout_secs: nd.timeout_secs,
                });
            }

            // ═══════════════════════════════════════════════════════════
            // Phase 2: Dispatch all agent nodes in this wave
            // ═══════════════════════════════════════════════════════════
            let num_tasks = agent_tasks.len();

            if num_tasks > 1 {
                // Multi-node wave: concurrent dispatch (all futures share the same type)
                let self_ref: &Self = &*self;
                let mut futs = Vec::new();
                for wt in &agent_tasks {
                    let role = wt.step.role;
                    let ctx = wt.ctx.clone();
                    let step = wt.step.clone();
                    let cid = cycle_id.to_string();
                    let wi = wt.wi;
                    let to = wt.timeout_secs;
                    futs.push(async move {
                        (wi, self_ref.dispatch_agent(role, ctx, &cid, Some(step), to).await)
                    });
                }

                let dispatch_results = futures::future::join_all(futs).await;

                for (result_wi, result_res) in dispatch_results {
                    let task_ni = order[result_wi];
                    let task_nd = &dag.graph[task_ni].def;
                    let task_step = crate::core::workflow::adapter::node_to_planstep(task_nd);
                    let result = match result_res {
                        Ok(r) => r,
                        Err(e) => {
                            warn!(error = %e, "Wave node dispatch error");
                            return Ok(TaskResult {
                                task_iri: task_iri.to_string(),
                                status: "failed".to_string(),
                                summary: format!("Wave node dispatch failed: {}", e),
                                output: None,
                                jsonld_output: None,
                                artifacts: vec![],
                                errors: vec![],
                                turn_count: 0,
                                tool_call_count: 0,
                                five_w2h_updates: None,
                                tracked_actions: vec![],
                                archive_iri: None,
                            });
                        }
                    };
                    if let Some(failed_task) = self.handle_step_result(
                        result, task_step, task_ni, result_wi,
                        &mut prev_summary, &mut da_output, &mut last_result,
                        &mut completed_node_results, &mut skip_nodes,
                        &mut five_w2h, task_iri, &cycle_id, &plan,
                        &dag, &order, five_w2h_iri,
                    ).await? {
                        return Ok(failed_task);
                    }
                }
            } else if num_tasks == 1 {
                let wt = agent_tasks.into_iter().next().unwrap();
                let result = self.dispatch_agent(wt.step.role, wt.ctx, &cycle_id, Some(wt.step.clone()), wt.timeout_secs).await?;
                if let Some(failed_task) = self.handle_step_result(
                    result, wt.step, wt.ni, wt.wi,
                    &mut prev_summary, &mut da_output, &mut last_result,
                    &mut completed_node_results, &mut skip_nodes,
                    &mut five_w2h, task_iri, &cycle_id, &plan,
                    &dag, &order, five_w2h_iri,
                ).await? {
                    return Ok(failed_task);
                }
            }

        }

        // ── CA→DA correction loop ──
        // When CA's dimension audit detects failures in DA's work, re-dispatch DA
        // with corrective feedback instead of immediately proceeding to AA.
        const MAX_CA_DA_CORRECTIONS: usize = 3;
        let mut correction_count = 0;
        loop {
            let ca_summary = prev_summary.clone();
            let has_ca_failures = ca_summary.as_ref().map_or(false, |ps| {
                ps.contains("Dimension Audit")
                    && (ps.contains("[what]") || ps.contains("[why]")
                        || ps.contains("[how]") || ps.contains("[how_much]"))
                    && ps.len() > 50
            });

            if !has_ca_failures || correction_count >= MAX_CA_DA_CORRECTIONS {
                break;
            }

            correction_count += 1;
            let ca_text = ca_summary.unwrap_or_default();

            info!(
                task_iri = %task_iri,
                correction = correction_count,
                "CA dimension audit found failures — re-dispatching DA with corrective context"
            );

            let da_corrective_objective = format!(
                "## Corrective Re-Execution (iteration {})\n\n\
                 Previous audit found gaps:\n\n{}\n\n\
                 Fix ALL identified issues. Output what was fixed.",
                correction_count,
                &ca_text[..ca_text.len().min(4000)]
            );

            let da_ctx = TaskContext::new(task_iri, &da_corrective_objective, self.max_iterations)
                .with_original_task(user_input)
                .with_cycle_id(&cycle_id);

            match self.dispatch_agent(AgentRole::Do, da_ctx, &cycle_id, None, 0).await {
                Ok(da_result) => {
                    let ca_objective = format!(
                        "Re-evaluate corrected execution:\n\n\
                         Corrected Output (iteration {}):\n{}\n\n\
                         Verify ALL previous audit issues are resolved.",
                        correction_count, da_result.summary
                    );

                    let ca_ctx = TaskContext::new(task_iri, &ca_objective, self.max_iterations)
                        .with_original_task(user_input)
                        .with_cycle_id(&cycle_id);

                    match self.dispatch_agent(AgentRole::Check, ca_ctx, &cycle_id, None, 0).await {
                        Ok(ca_result) => {
                            let archive_ref = ca_result.archive_iri.as_ref()
                                .map(|iri| format!("\n\nFor detailed report, use read_agent_output: {}", iri))
                                .unwrap_or_default();
                            prev_summary = Some(format!(
                                "## Corrected Execution (iter {})\n{}\n\n## CA Re-Evaluation\n{}{}",
                                correction_count, da_result.summary, ca_result.summary, archive_ref
                            ));
                            last_result = Some(ca_result);

                            self.emit_sa_thought(task_iri,
                                &format!("CA→DA correction #{} completed", correction_count),
                                "ca_da_correction").await;
                        }
                        Err(e) => {
                            warn!(error = %e, "CA re-dispatch after DA correction failed");
                            break;
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "DA corrective re-dispatch failed");
                    break;
                }
            }
        }

        if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
            cycle.phase = CyclePhase::Completed;
            cycle.task_completed = true;
            cycle.phase_history.push("Completed".to_string());
        }

        self.event_bus
            .emit(task_iri, "CYCLE_COMPLETED", "SA",
                &serde_json::json!({"cycle_id": &cycle_id}).to_string())
            .await;

        Ok(last_result.unwrap_or(TaskResult {
            task_iri: task_iri.to_string(),
            status: "completed".to_string(),
            summary: "No agents executed".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 0,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        }))
    }

    /// Process a single DAG node's execution result — handles failure, 5W2H, perception,
    /// AA early exit, recursive sub-cycles, prev_summary tracking, and checkpoint.
    /// Returns `Ok(Some(TaskResult))` if the caller should terminate (node failure),
    /// `Ok(None)` to continue normally.
    #[allow(clippy::too_many_arguments)]
    async fn handle_step_result(
        &self,
        mut result: TaskResult,
        step: PlanStep,
        _node_idx: NodeIndex,
        i: usize,
        prev_summary: &mut Option<String>,
        da_output: &mut Option<String>,
        last_result: &mut Option<TaskResult>,
        completed_node_results: &mut std::collections::HashMap<String, crate::core::workflow::NodeResult>,
        skip_nodes: &mut std::collections::HashSet<String>,
        five_w2h: &mut crate::core::five_w2h::Task5W2H,
        task_iri: &str,
        cycle_id: &str,
        plan: &ExecutionPlan,
        dag: &crate::core::workflow::loader::WorkflowDag,
        order: &[NodeIndex],
        five_w2h_iri: &str,
    ) -> Result<Option<TaskResult>, CoreError> {
        let task_id = task_iri.strip_prefix("iri://task/")
            .unwrap_or_else(|| task_iri.strip_prefix("iri://").unwrap_or(task_iri));

        // Early return on agent failure
        if result.status == "failed" {
            warn!(role = ?step.role, step_id = %step.step_id, "Agent failed, aborting plan");
            let error_detail = result.errors.first()
                .map(|e| format!("\n\n**Error details**: {}", e))
                .unwrap_or_default();
            return Ok(Some(TaskResult {
                task_iri: task_iri.to_string(),
                status: "failed".to_string(),
                summary: format!("Agent {:?} failed at step {}{}", step.role, step.step_id, error_detail),
                output: None,
                jsonld_output: None,
                artifacts: Vec::new(),
                errors: result.errors,
                turn_count: result.turn_count,
                tool_call_count: result.tool_call_count,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
            }));
        }

        // Propagate 5W2H updates
        if let Some(ref updates) = result.five_w2h_updates {
            five_w2h.merge_updates(updates);
            if let Ok(updated_json_ld) = five_w2h.to_json_ld(task_iri) {
                let _ = self.runner.l0_store.store(&five_w2h_iri, &updated_json_ld.to_string());
                let cfg = crate::CoreConfig::default();
                if let Some(ref bb) = self.blackboard {
                    if bb.write_node(&five_w2h_iri, &updated_json_ld.to_string(), &cfg).is_ok() {
                        tracing::debug!(five_w2h_iri = %five_w2h_iri, "5W2H update synced to blackboard");
                    }
                }
            }
        }

        // AA freeze
        if step.role == AgentRole::Act && result.status == "success" {
            let is_last_aa = plan.steps.iter().rposition(|s| s.role == AgentRole::Act)
                .map(|last_act| plan.steps.iter().position(|s| s.step_id == step.step_id)
                    .map(|idx| idx >= last_act)
                    .unwrap_or(true))
                .unwrap_or(true);
            if is_last_aa {
                five_w2h.freeze();
                if let Ok(frozen_json_ld) = five_w2h.to_json_ld(task_iri) {
                    let snapshot_iri = format!("iri://task/{}/snapshot", task_id);
                    let _ = self.runner.l0_store.store(&snapshot_iri, &frozen_json_ld.to_string());
                    let _ = self.runner.l0_store.store(&five_w2h_iri, &frozen_json_ld.to_string());
                    let cfg = crate::CoreConfig::default();
                    if let Some(ref bb) = self.blackboard {
                        let _ = bb.write_node(&snapshot_iri, &frozen_json_ld.to_string(), &cfg);
                        let _ = bb.write_node(&five_w2h_iri, &frozen_json_ld.to_string(), &cfg);
                    }
                    info!(task_iri = %task_iri, "5W2H frozen and archived");
                }
            } else {
                info!(task_iri = %task_iri, step_id = %step.step_id, "Intermediate AA step: 5W2H not frozen yet");
            }
        }

        // Sharing
        self.sharing.create_share(
            &format!("iri://agent/{}", step.role),
            "iri://agent/next",
            &[format!("iri://task/{}/result", task_iri)],
            crate::tools::sharing::ShareType::Projection,
            crate::tools::sharing::Permission::Read,
            Some(3600),
            None,
        );

        // PA perception
        if step.role == AgentRole::Plan && result.status == "success" {
            let plan_data = serde_json::json!({
                "summary": &result.summary,
                "objective": &step.objective,
            });
            let advisories = self.perception.on_plan_completed(&plan_data, task_iri);
            if !advisories.is_empty() {
                info!(count = advisories.len(), "PA perception advisories generated");
            }
        }

        // CA perception + dimension audit
        if step.role == AgentRole::Check {
            let check_data = serde_json::json!({
                "summary": &result.summary,
                "objective": &step.objective,
            });
            if let Some(advisory) = self.perception.on_check_completed(&check_data, task_iri) {
                info!(advisory = ?advisory, "CA perception advisories generated");
            }

            // Run dimension-level audit against the 5W2H specification
            let audit_results = crate::core::five_w2h::audit_dimensions(
                five_w2h,
                &result,
                task_iri,
                self.runner.causal_engine.as_ref().map(|ce| ce.as_ref()),
            );
            let failures: Vec<&crate::core::five_w2h::DimensionAuditResult> = audit_results.iter()
                .filter(|r| matches!(r.status, crate::core::five_w2h::AuditStatus::Fail(_)))
                .collect();
            if !failures.is_empty() {
                let fail_summary: Vec<String> = failures.iter()
                    .map(|r| format!("[{}] {}: {}", r.dimension, match &r.status {
                        crate::core::five_w2h::AuditStatus::Fail(msg) => msg.as_str(),
                        _ => "failed",
                    }, r.evidence))
                    .collect();
                warn!(
                    task_iri = %task_iri,
                    dimensions = ?failures.iter().map(|r| &r.dimension).collect::<Vec<_>>(),
                    "Dimension audit: {} dimension(s) failed — causal observations recorded",
                    failures.len()
                );
                // Append dimension audit findings to the CA agent's result summary
                if result.summary.len() < 4000 {
                    let audit_note = format!("\n\n--- Dimension Audit ---\n{}", fail_summary.join("\n"));
                    if !result.summary.contains("Dimension Audit") {
                        result.summary.push_str(&audit_note);
                    }
                }
            }
        }

        // AA early exit — skip remaining PDCA cycles after AA evaluates
        if step.role == AgentRole::Act {
            let has_remaining = (i + 1) < order.len();
            if has_remaining {
                let reason = match result.status.as_str() {
                    "success" => "AA passed, task completed",
                    "failed" | "partial_success" => "AA did not pass",
                    _ => "AA evaluated",
                };
                info!(step_id = %step.step_id, status = %result.status, "{}, skipping remaining PDCA cycles", reason);
                for skip_idx in (i + 1)..order.len() {
                    skip_nodes.insert(dag.graph[order[skip_idx]].def.id.clone());
                }
            }
        }

        // Recursive sub-cycle for Do agents
        if step.role == AgentRole::Do
            && (result.status == "success" || result.status == "partial_success")
            && plan.max_recursion_depth > 0
            && (plan.task_complexity == crate::core::sa::types::TaskComplexity::Recursive || plan.task_complexity == crate::core::sa::types::TaskComplexity::Complex)
        {
            let sub_results = self.execute_recursive_sub_cycle(
                &result.summary,
                task_iri,
                cycle_id,
                &step.step_id,
                plan.max_recursion_depth,
                1,
                five_w2h,
                five_w2h_iri,
            ).await;

            match sub_results {
                Ok(sub_summary) => {
                    *prev_summary = Some(format!(
                        "{}\n\n## Sub-task Execution Results\n{}",
                        result.summary, sub_summary
                    ));
                }
                Err(e) => {
                    warn!(error = %e, "Recursive sub-cycle execution failed, using DA original result");
                    *prev_summary = Some(match result.archive_iri {
                        Some(ref iri) => format!("{}\n\nFor the full report, use read_agent_output tool to query: {}", result.summary, iri),
                        None => result.summary.clone(),
                    });
                }
            }
        } else {
            *prev_summary = Some(match result.archive_iri {
                Some(ref iri) => format!("{}\n\nFor the full report, use read_agent_output tool to query: {}", result.summary, iri),
                None => result.summary.clone(),
            });
        }

        *last_result = Some(result);

        // Track Do agent output separately
        if step.role == AgentRole::Do {
            if let Some(ref s) = *prev_summary {
                *da_output = Some(s.clone());
            }
        }

        // 5W2H constraint check
        if let Some(alert) = self.perception.check_5w2h_constraints(five_w2h_iri) {
            tracing::warn!(alert = %alert, "5W2H constraint alert");
            self.event_bus.emit(task_iri, &alert, "SA", &serde_json::json!({"task_iri": task_iri}).to_string()).await;
        }

        info!(step_id = %step.step_id, role = ?step.role, status = ?last_result.as_ref().map(|r| &r.status), "Step completed");

        // ── Checkpoint ──
        {
            let cm = crate::core::checkpoint::CheckpointManager::with_persistence(self.runner.l0_store.clone());
            let role_name = format!("{:?}", step.role);
            let state_json = serde_json::json!({
                "turn": last_result.as_ref().map(|r| r.turn_count).unwrap_or(0),
                "tc": last_result.as_ref().map(|r| r.tool_call_count).unwrap_or(0),
                "prompt_tokens": self.runner.total_prompt_tokens.load(std::sync::atomic::Ordering::Relaxed),
                "completion_tokens": self.runner.total_completion_tokens.load(std::sync::atomic::Ordering::Relaxed),
            }).to_string();

            let cycle_state = self.active_cycles.get(cycle_id).map(|c| serde_json::json!({
                "phase": format!("{:?}", c.phase),
                "iteration": c.iteration,
                "phase_history": c.phase_history,
                "task_completed": c.task_completed,
                "experience_hints": c.experience_hints,
            }).to_string());

            let completed_nodes = if completed_node_results.is_empty() {
                None
            } else {
                Some(serde_json::to_string(completed_node_results).unwrap_or_default())
            };

            let pending_approvals = {
                let map = self.pending_approvals.lock().await;
                if map.is_empty() { None }
                else { Some(serde_json::to_string(&*map).unwrap_or_default()) }
            };

            let supplement_data = {
                let pending = self.supplement_store.take_pending(task_iri);
                if pending.is_empty() { None }
                else {
                    let entries: Vec<serde_json::Value> = pending.iter().map(|e| serde_json::json!({
                        "content": e.content,
                        "relevance_score": e.relevance_score,
                        "timestamp": e.timestamp,
                    })).collect();
                    Some(serde_json::to_string(&entries).unwrap_or_default())
                }
            };

            let cp_name = format!("step_complete_{}", role_name);
            let tags = vec![role_name.clone(), "step_complete".to_string()];

            let session_msgs_json: String = if let Some(ref bb) = self.blackboard {
                let filter = crate::memory::l2_blackboard::QueryFilter {
                    role: None,
                    cycle_id: Some(cycle_id.to_string()),
                    node_type: Some("AgentTurn".to_string()),
                };
                let nodes = bb.query_nodes_filtered(task_iri, &filter).unwrap_or_default();
                let msgs: Vec<serde_json::Value> = nodes.iter().filter_map(|n| {
                    let parsed: serde_json::Value = serde_json::from_str(&n.json_ld).ok()?;
                    Some(serde_json::json!({
                        "role": parsed.get("role").and_then(|r| r.as_str()).unwrap_or("assistant"),
                        "content": parsed.get("content").and_then(|c| c.as_str()).unwrap_or(""),
                        "summary": parsed.get("summary").and_then(|s| s.as_str()),
                    }))
                }).collect();
                serde_json::to_string(&msgs).unwrap_or_else(|_| "[]".to_string())
            } else {
                "[]".to_string()
            };

            if let Err(e) = cm.create_ext(
                task_iri,
                &cp_name,
                "[]",
                &session_msgs_json,
                &state_json,
                &tags,
                Some(&role_name),
                None,
                prev_summary.as_deref(),
                cycle_state.as_deref(),
                completed_nodes.as_deref(),
                pending_approvals.as_deref(),
                supplement_data.as_deref(),
                None,
                None,
                None,
            ) {
                warn!("[checkpoint] step_complete save failed: {}", e);
            } else {
                info!("[checkpoint] step_complete_{} saved", role_name);
            }
        }

        Ok(None)
    }

    fn execute_recursive_sub_cycle<'a>(
        &'a self,
        da_summary: &'a str,
        task_iri: &'a str,
        cycle_id: &'a str,
        parent_step_id: &'a str,
        max_depth: u32,
        current_depth: u32,
        five_w2h: &'a crate::core::five_w2h::Task5W2H,
        five_w2h_iri: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, CoreError>> + Send + 'a>> {
        Box::pin(async move {
        if current_depth > max_depth {
            info!(depth = current_depth, max_depth, "Recursive depth limit reached, stopping sub-cycle");
            return Ok("Recursive depth limit reached".to_string());
        }

        self.emit_sa_thought(task_iri,
            &format!("▶ Recursive sub-cycle (depth {}/{})", current_depth, max_depth),
            "recursive_sub_cycle_start").await;

        let sub_task = SubTask::new(
            &format!("Decomposing sub-tasks from DA result (depth={})", current_depth),
            parent_step_id,
            current_depth,
        );

        info!(
            sub_task_id = %sub_task.sub_task_id,
            depth = current_depth,
            max_depth,
            "Starting recursive sub-cycle"
        );

        self.emit_sa_thought(task_iri,
            &format!("Decomposing DA result, identifying sub-tasks... (depth {}/{})", current_depth, max_depth),
            "recursive_decompose").await;

        let decompose_prompt = format!(
            r#"You are a task decomposition expert. Below is an execution result summary of a DA (Do Agent). Analyze whether there are sub-tasks that need further execution.

## DA Execution Result
{}

## Task Context
- Original goal: {}
- Current recursion depth: {}/{}

## Output Requirements
Output the list of sub-tasks that need further execution in JSON format. If no further sub-tasks are needed, return an empty array.

```json
{{
  "has_sub_tasks": true/false,
  "sub_tasks": [
    {{
      "objective": "Sub-task objective description",
      "role": "Do",
      "success_criteria": "Success criteria"
    }}
  ]
}}
```

## Evaluation Criteria
1. If the DA result explicitly mentions "still needs...", "next step needs...", etc., there are sub-tasks
2. If the DA result has fully completed the goal with no remaining work, there are no sub-tasks
3. Sub-tasks should be concrete and executable, not abstract
4. Maximum of 3 sub-tasks

Output only JSON."#,
            da_summary,
            five_w2h.what,
            current_depth,
            max_depth,
        );

        let model = self.runner.gateway.get_model("default");
        let messages = vec![crate::gateway::unified_gateway::ChatMessage {
            role: "user".to_string(),
            content: decompose_prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let response = self.runner.gateway.chat_with_params(
            &model,
            messages,
            Some(0.3),
            Some(1000),
            None,
            None,
        ).await.map_err(|e| CoreError::Internal { message: format!("Recursive decomposition LLM call failed: {}", e) })?;

        let content = response.choices.first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let json_str = if content.starts_with('{') {
            content.clone()
        } else if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                content[start..=end].to_string()
            } else {
                content.clone()
            }
        } else {
            return Ok("LLM did not return valid decomposition result".to_string());
        };

        #[derive(Deserialize)]
        struct DecomposeResult {
            has_sub_tasks: bool,
            sub_tasks: Vec<SubTaskDef>,
        }

        #[derive(Deserialize)]
        struct SubTaskDef {
            objective: String,
            #[serde(default = "default_role")]
            #[allow(dead_code)]
            role: String,
            success_criteria: String,
        }

        fn default_role() -> String { "Do".to_string() }

        let parsed: DecomposeResult = serde_json::from_str(&json_str)
            .map_err(|e| CoreError::Internal { message: format!("Recursive decomposition JSON parse failed: {}", e) })?;

        if !parsed.has_sub_tasks || parsed.sub_tasks.is_empty() {
            info!(depth = current_depth, "No further decomposition needed for DA result");
            self.emit_sa_thought(task_iri,
                &format!("Sub-task decomposition complete: no further decomposition needed (depth {}/{})", current_depth, max_depth),
                "recursive_no_tasks").await;
            return Ok("No further decomposition needed".to_string());
        }

        self.emit_sa_thought(task_iri,
            &format!("Identified {} sub-tasks (depth {}/{})", parsed.sub_tasks.len(), current_depth, max_depth),
            "recursive_tasks_found").await;

        let mut sub_summaries = Vec::new();

        for (idx, sub_def) in parsed.sub_tasks.iter().enumerate() {
            let sub_objective = format!("[recursive depth={}] {}", current_depth, sub_def.objective);
            info!(sub_idx = idx, objective = %sub_def.objective, "Executing recursive sub-task");

            let mut sub_ctx = TaskContext::new(
                task_iri,
                &sub_objective,
                self.max_iterations.min(8),
            ).with_original_task(&sub_def.objective);

            sub_ctx = sub_ctx.with_five_w2h(five_w2h_iri, five_w2h.clone());

            if let Some(ref bb) = self.blackboard {
                let nodes = bb.query_nodes(task_iri).unwrap_or_default();
                if !nodes.is_empty() {
                    let summaries: Vec<String> = nodes.iter()
                        .filter_map(|n| {
                            let parsed: serde_json::Value = serde_json::from_str(&n.json_ld).ok()?;
                            parsed.get("summary").and_then(|s| s.as_str()).map(String::from)
                        })
                        .collect();
                    if !summaries.is_empty() {
                        sub_ctx = sub_ctx.with_prev_summary(&summaries.join("\n"));
                    }
                }
            }

            let sub_step = PlanStep {
                step_id: format!("{}_sub_{}", parent_step_id, idx),
                role: AgentRole::Do,
                objective: sub_def.objective.clone(),
                expected_output: sub_def.success_criteria.clone(),
                dependencies: vec![parent_step_id.to_string()],
                tools_allowed: vec![],
                success_criteria: sub_def.success_criteria.clone(),
            };

            let total = parsed.sub_tasks.len();
            self.emit_sa_thought(task_iri,
                &format!("▶ Executing sub-task {}/{}: {} (depth {})", idx + 1, total, sub_def.objective, current_depth),
                "recursive_sub_task_start").await;

            let sub_result = self.dispatch_agent(AgentRole::Do, sub_ctx, cycle_id, Some(sub_step), 0).await?;

            self.emit_sa_thought(task_iri,
                &format!("{}/{} sub-task complete [{}]: {}", idx + 1, total,
                    sub_result.status, sub_def.objective),
                "recursive_sub_task_end").await;

            if sub_result.status == "success" || sub_result.status == "partial_success" {
                let icon = if sub_result.status == "success" { "✅" } else { "⚠️" };
                sub_summaries.push(format!("### Sub-task {} {}\n{}", idx + 1, icon, sub_result.summary));

                if current_depth < max_depth && sub_result.status == "success" {
                    // Only fully successful sub-tasks continue deeper recursion; partial_success continues in upper recursion
                    self.emit_sa_thought(task_iri,
                        &format!("Entering deeper recursion (depth {}/{})", current_depth + 1, max_depth),
                        "recursive_deeper").await;
                    match self.execute_recursive_sub_cycle(
                        &sub_result.summary,
                        task_iri,
                        cycle_id,
                        &format!("{}_sub_{}", parent_step_id, idx),
                        max_depth,
                        current_depth + 1,
                        five_w2h,
                        five_w2h_iri,
                    ).await {
                        Ok(deeper_summary) => {
                            sub_summaries.push(format!("#### Deep sub-task (depth={})\n{}", current_depth + 1, deeper_summary));
                        }
                        Err(e) => {
                            warn!(error = %e, "Deep recursive sub-cycle failed");
                        }
                    }
                }
            } else {
                sub_summaries.push(format!("### Sub-task {} ❌\nExecution failed: {}", idx + 1, sub_result.summary));
            }
        }

        self.emit_sa_thought(task_iri,
            &format!("Recursive sub-cycle complete (depth {}/{})", current_depth, max_depth),
            "recursive_sub_cycle_end").await;
        Ok(sub_summaries.join("\n\n"))
        })
    }
}
