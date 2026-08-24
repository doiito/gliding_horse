use tracing::{info, instrument, warn};

use crate::core::agent_instance::AgentRole;
use crate::core::agent_runner::{TaskContext, TaskResult};
use crate::CoreError;

use super::agent::SupervisorAgent;
use super::types::*;

/// Upper bound on distinct experience hints injected into the LLM prompt.
/// Perception events plus skill discovery results can overlap; the cap keeps
/// the prompt lean while retaining the first-seen (highest priority) hints.
const MAX_ALL_HINTS: usize = 20;

/// Deduplicate hints preserving first-seen order, then truncate to `cap`.
fn dedup_hints(hints: Vec<String>, cap: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::with_capacity(cap.min(hints.len()));
    let mut out = Vec::with_capacity(cap.min(hints.len()));
    for h in hints {
        if out.len() >= cap {
            break;
        }
        if seen.insert(h.clone()) {
            out.push(h);
        }
    }
    out
}

fn policy_reward(result: &TaskResult) -> f32 {
    match result.status.as_str() {
        "success" | "completed" if !result.summary.contains("[Recovery] scope=Task") => 1.0,
        "partial_success" => 0.25,
        "failed" | "timeout" => -1.0,
        _ => -0.5,
    }
}

impl SupervisorAgent {
    #[instrument(skip(self, user_input), fields(task_iri = %task_iri))]
    pub async fn process_task(
        &mut self,
        user_input: &str,
        task_iri: &str,
    ) -> Result<TaskResult, CoreError> {
        self.process_task_with_context(
            user_input,
            task_iri,
            TaskContext::new(task_iri, user_input, self.max_iterations),
        )
        .await
    }

    /// Process task with custom TaskContext, supports resume mode
    #[instrument(skip(self, user_input, ctx), fields(task_iri = %task_iri))]
    pub async fn process_task_with_context(
        &mut self,
        user_input: &str,
        task_iri: &str,
        ctx: TaskContext,
    ) -> Result<TaskResult, CoreError> {
        let cycle_id = self.start_cycle(user_input, task_iri).await?;

        let mut five_w2h = self.extract_5w2h_from_input(user_input).await;
        let task_id = task_iri
            .strip_prefix("iri://task/")
            .unwrap_or_else(|| task_iri.strip_prefix("iri://").unwrap_or(task_iri));
        let five_w2h_iri = format!("iri://task/{}/5w2h", task_id);

        // A3: Calculate task_embedding from 5W2H → set to relevance_tracker
        if let Some(ref embedder) = self.embedder {
            let task_text = format!("{}\n{}", five_w2h.what, five_w2h.why.description);
            if let Ok(task_emb) = embedder.embed(&task_text).await {
                self.relevance_tracker.set_task_context(task_emb);
            }
        }

        // Inject current working directory as execution environment, so LLM knows where to create files
        if five_w2h
            .where_
            .as_ref()
            .and_then(|w| w.execution_environment.as_ref())
            .is_none()
        {
            if let Ok(cwd) = std::env::current_dir() {
                let cwd_str = cwd.to_string_lossy().to_string();
                five_w2h = five_w2h.with_where(crate::core::five_w2h::WhereDetail {
                    data_sources: vec![],
                    execution_environment: Some(cwd_str),
                    target_repository: None,
                    target_branch: None,
                });
            }
        }

        // Fill missing 5W2H dimensions before PA dispatch (SA phase), not at CA stage.
        five_w2h.derive_defaults(self.max_iterations, self.max_pdca_cycles);

        if let Ok(json_ld) = five_w2h.to_json_ld(task_iri) {
            let _ = self
                .runner
                .l0_store
                .store(&five_w2h_iri, &json_ld.to_string());
            let cfg = crate::CoreConfig::default();
            if let Some(ref bb) = self.blackboard {
                if bb
                    .write_node(&five_w2h_iri, &json_ld.to_string(), &cfg)
                    .is_ok()
                {
                    tracing::debug!(five_w2h_iri = %five_w2h_iri, "5W2H written to blackboard");
                    let route = self.type_router.get_route("task:5W2H");
                    if let Some(route) = route {
                        for event in &route.events {
                            let _ = self
                                .event_bus
                                .emit(task_iri, event, "system:sa", &five_w2h_iri)
                                .await;
                        }
                    }
                }
            }
            tracing::info!(task_iri = %task_iri, what = %five_w2h.what, "5W2H initialization complete");
        }

        let perception_hints = self
            .perception
            .on_task_start(user_input, task_iri)
            .await
            .map(|a| a.relevant_experience_hints)
            .unwrap_or_default();

        // Enrich with skill discovery results if a discovery engine is available
        let mut all_hints: Vec<String> = perception_hints;
        let policy_context = format!(
            "planning:{}",
            five_w2h.what.chars().take(80).collect::<String>().to_lowercase()
        );
        if let Some(ref de) = self.discovery_engine {
            let disc_task = crate::skill_graph::discovery::Task5W2H {
                what: five_w2h.what.clone(),
                why: five_w2h.why.description.clone(),
                who: five_w2h.who.as_ref().and_then(|w| w.required_role.clone()),
                when_phase: five_w2h.when.as_ref().map(|w| format!("{:?}", w)),
                where_context: five_w2h.where_.as_ref().map(|w| format!("{:?}", w)),
                how_approach: five_w2h.how.as_ref().and_then(|h| h.required_steps.clone()),
                constraints: five_w2h.why.success_criteria.clone(),
            };
            let matches = de.discover_for_task(&disc_task).await;
            let skill_hints: Vec<String> = matches
                .iter()
                .filter_map(|m| {
                    let name = if !m.skill.name.is_empty() {
                        m.skill.name.clone()
                    } else {
                        m.skill.skill_iri.rsplit('/').next()?.to_string()
                    };
                    Some(name)
                })
                .take(5)
                .collect();
            let knowledge_hints: Vec<String> = if let Some(graph) = self.runner.skill_graph_store.as_ref() {
                matches
                    .iter()
                    .flat_map(|m| graph.get_fragments_for_skill(&m.skill.skill_iri))
                    .map(|fragment| {
                        format!(
                            "Knowledge fragment for {}: problem={} mitigation={}",
                            fragment.attached_to, fragment.problem, fragment.recommendation
                        )
                    })
                    .take(8)
                    .collect()
            } else {
                Vec::new()
            };
            if !skill_hints.is_empty() {
                all_hints.extend(skill_hints);
                tracing::info!(
                    task_iri = %task_iri,
                    "Skill discovery enriched planning with relevant skills"
                );
            }
            if !knowledge_hints.is_empty() {
                all_hints.extend(knowledge_hints);
                tracing::info!(
                    task_iri = %task_iri,
                    "Committed skill knowledge fragments enriched planning"
                );
            }
        }
        all_hints = dedup_hints(all_hints, MAX_ALL_HINTS);
        let policy_choice = self.policy_learning.choose(
            &policy_context,
            &[
                "baseline".to_string(),
                "knowledge_first".to_string(),
                "skill_first".to_string(),
            ],
            "baseline",
        );
        if policy_choice.action == "knowledge_first" {
            all_hints.sort_by_key(|hint| if hint.starts_with("Knowledge fragment") { 0 } else { 1 });
        } else if policy_choice.action == "skill_first" {
            all_hints.sort_by_key(|hint| {
                if hint.contains("skill") || hint.contains("Skill") { 0 } else { 1 }
            });
        }
        all_hints.push(format!(
            "Constrained policy: {} (confidence {:.2}; fallback={})",
            policy_choice.action, policy_choice.confidence, policy_choice.used_fallback
        ));
        all_hints = dedup_hints(all_hints, MAX_ALL_HINTS);

        // Unified execution path: build ExecutionPlan from JSON-LD workflow or LLM
        let mut plan = if let Some(ref wf_jsonld) = ctx.workflow_jsonld {
            info!(task_iri = %task_iri, "Using JSON-LD workflow mode — converting through adapter to ExecutionPlan");
            let def =
                crate::core::workflow::loader::load_workflow_jsonld(wf_jsonld).map_err(|e| {
                    CoreError::Internal {
                        message: format!("Workflow parsing failed: {}", e),
                    }
                })?;
            let dag = crate::core::workflow::loader::build_dag(&def).map_err(|e| {
                CoreError::Internal {
                    message: format!("DAG build failed: {}", e),
                }
            })?;
            let mut plan =
                crate::core::workflow::adapter::dag_to_execution_plan(&dag, &def, task_iri);
            plan.dag_jsonld = Some(wf_jsonld.clone());
            plan
        } else if ctx.resumed_messages.is_some() {
            self.build_resume_plan()
        } else {
            self.analyze_task_with_llm(user_input, &five_w2h, &all_hints)
                .await
        };

        // ── Verify-first optimization ──
        // When workspace has existing files and plan is non-trivial:
        // prepend CA→AA to check existing code first, store original as fallback_steps.
        // execute_plan returns "failed" if verify CA fails → retry loop uses fallback_steps.
        if !plan.verify_first
            && ctx.workspace_file_summary.is_some()
            && ctx.resumed_messages.is_none()
            && ctx.workflow_jsonld.is_none()
            && plan.steps.len() >= 2
            && plan
                .steps
                .iter()
                .any(|s| matches!(s.role, AgentRole::Plan | AgentRole::Do))
        {
            let ws_summary = ctx
                .workspace_file_summary
                .as_deref()
                .unwrap_or("workspace has files");
            plan.fallback_steps = plan.steps.clone();
            plan.verify_first = true;

            let verify_ca = PlanStep {
                step_id: "verify_ca".to_string(),
                role: AgentRole::Check,
                objective: format!(
                    "Check if existing workspace files already satisfy the task requirement.\n\
                     Workspace inventory: {}\n\
                     If existing code meets requirements, report VERIFIED-PASS with evidence.\n\
                     If not, report what is missing or needs modification.",
                    ws_summary
                ),
                expected_output:
                    "Verification result: PASS (existing code sufficient) or FAIL (list gaps)"
                        .to_string(),
                dependencies: vec![],
                tools_allowed: vec![],
                success_criteria: "Clear pass/fail verdict with evidence from workspace"
                    .to_string(),
                branch_on_failure: false,
                branch_fallback: None,
                retry_count: 0,
                retry_delay_secs: 0,
            };
            let verify_aa = PlanStep {
                step_id: "verify_aa".to_string(),
                role: AgentRole::Act,
                objective: "Evaluate verification results. If existing code already satisfies requirements, confirm task complete. Otherwise indicate full execution is needed.".to_string(),
                expected_output: "Final verdict: task already done vs needs full execution".to_string(),
                dependencies: vec!["verify_ca".to_string()],
                tools_allowed: vec![],
                success_criteria: "Decision clear with justification".to_string(),
                branch_on_failure: false,
                branch_fallback: None,
                retry_count: 0,
                retry_delay_secs: 0,
            };
            // Store original description, prepend verify steps
            let original_desc = plan.description.clone();
            plan.steps = vec![verify_ca, verify_aa];
            plan.agent_sequence = vec![AgentRole::Check, AgentRole::Act];
            plan.description = format!(
                "[Verify-first] Check existing workspace code before full PDCA. Fallback: {}",
                original_desc
            );

            info!(task_iri = %task_iri, ws = %ws_summary, "Verify-first: CA→AA prepended, fallback_steps={}", plan.fallback_steps.len());
        }

        // Adapt relevance tracker decay λ to task complexity
        self.relevance_tracker
            .adapt_to_complexity(&plan.task_complexity);

        let step_roles: Vec<String> = plan.steps.iter().map(|s| format!("{:?}", s.role)).collect();
        self.emit_sa_thought(
            task_iri,
            &format!(
                "Task classified. Plan: {} ({} steps: {})",
                plan.description,
                plan.steps.len(),
                step_roles.join(" → ")
            ),
            "plan_created",
        )
        .await;

        if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
            cycle.phase = CyclePhase::Executing;
            cycle
                .phase_history
                .push(format!("Plan: {}", plan.description));
        }

        let mut pending_interventions: Vec<crate::perception::proactive_engine::InterventionPlan> =
            Vec::new();
        if let Some(ref mut receiver) = self.event_receiver {
            while let Ok(event) = receiver.try_recv() {
                if event.task_iri != task_iri {
                    continue;
                }
                match event.event_type.as_str() {
                    "INTERVENTION_REQUIRED" => {
                        if let Ok(plan) = serde_json::from_str::<
                            crate::perception::proactive_engine::InterventionPlan,
                        >(&event.payload)
                        {
                            pending_interventions.push(plan);
                        }
                    }
                    "DEADLINE_APPROACHING" => {
                        warn!("Deadline approaching, marking task as urgent");
                    }
                    "HUMAN_APPROVAL_RESULT" => {
                        if let Ok(result) =
                            serde_json::from_str::<serde_json::Value>(&event.payload)
                        {
                            let request_id = result
                                .get("request_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let approved = result
                                .get("approved")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            if !request_id.is_empty() {
                                self.pending_approvals
                                    .lock()
                                    .await
                                    .insert(request_id.to_string(), approved);
                                info!(request_id = %request_id, approved = %approved, "Received human approval result");
                            }
                        }
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
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(self.execution_timeout_secs),
                self.execute_intervention_for_cycle(plan, task_iri),
            )
            .await;
        }

        // ── Outer SA-level PDCA retry loop ──
        // Ensure at least 2 cycles when verify_first is active
        // (cycle 0 = verify-first CA→AA, cycle 1+ = fallback PDCA)
        let max_cycles = if plan.verify_first {
            (self.max_pdca_cycles.max(1)).max(2)
        } else {
            self.max_pdca_cycles.max(1)
        };
        let mut cycle_feedback: Option<String> = None;
        let mut final_result: Option<TaskResult> = None;
        let mut plan_revision = 1u32;
        let mut orchestration_trajectory = Vec::new();
        let token_budget = five_w2h.how_much.as_ref().and_then(|h| h.token_budget);

        for cycle_num in 0..max_cycles {
            if let Some(limit) = token_budget {
                let used = self
                    .runner
                    .total_prompt_tokens
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .saturating_add(
                        self.runner
                            .total_completion_tokens
                            .load(std::sync::atomic::Ordering::Relaxed),
                    );
                if used >= limit {
                    let summary = format!(
                        "Recovery blocked: token budget exhausted before PDCA cycle {} (used {}, limit {}).",
                        cycle_num + 1,
                        used,
                        limit
                    );
                    tracing::warn!(task_iri = %task_iri, used, limit, "Task token budget exhausted");
                    self.event_bus
                        .emit(task_iri, "RECOVERY_BLOCKED", "SA", &summary)
                        .await;
                    // Preserve the last completed cycle's evidence. Replacing
                    // it with a zeroed result made a budget stop look like a
                    // task that had executed no turns or tools at all.
                    let mut blocked_result = final_result.take().unwrap_or_else(|| TaskResult {
                        task_iri: task_iri.to_string(),
                        status: "failed".to_string(),
                        summary: String::new(),
                        output: None,
                        jsonld_output: None,
                        artifacts: Vec::new(),
                        errors: Vec::new(),
                        turn_count: 0,
                        tool_call_count: 0,
                        five_w2h_updates: None,
                        tracked_actions: Vec::new(),
                        verdict: None,
                        archive_iri: None,
                    });
                    blocked_result.status = "failed".to_string();
                    blocked_result.summary = if blocked_result.summary.is_empty() {
                        summary.clone()
                    } else {
                        format!("{}\n\n{}", blocked_result.summary, summary)
                    };
                    blocked_result.errors.push("token budget exhausted".to_string());
                    final_result = Some(blocked_result);
                    break;
                }
            }
            let resumed = if cycle_num == 0 {
                ctx.resumed_messages.clone()
            } else {
                None
            };

            // On retry after verify-first failed, switch to fallback_steps (full PDCA)
            let current_plan =
                if cycle_num >= 1 && plan.verify_first && !plan.fallback_steps.is_empty() {
                    let mut fb = plan.clone();
                    plan_revision = cycle_num as u32 + 1;
                    fb.plan_id = format!("{}_rev_{}", plan.plan_id, plan_revision);
                    fb.steps = plan.fallback_steps.clone();
                    fb.verify_first = false;
                    fb.agent_sequence = fb.steps.iter().map(|s| s.role).collect();
                    fb.description = format!(
                    "Fallback PDCA (verify-first CA did not pass): {}",
                    plan.description.trim_start_matches(
                        "[Verify-first] Check existing workspace code before full PDCA. Fallback: "
                    )
                );
                    fb
                } else {
                    let mut current = plan.clone();
                    if cycle_num > 0 {
                        plan_revision = cycle_num as u32 + 1;
                        current.plan_id = format!("{}_rev_{}", plan.plan_id, plan_revision);
                    }
                    current
                };

            info!(
                task_iri = %task_iri,
                cycle_num = cycle_num + 1,
                max_cycles = max_cycles,
                has_feedback = cycle_feedback.is_some(),
                "Starting SA-level PDCA cycle"
            );

            if let Some(ref _feedback) = cycle_feedback {
                self.emit_sa_thought(
                    task_iri,
                    &format!(
                        "⚠️ AA did not pass cycle #{} — restarting PDCA with feedback",
                        cycle_num + 1
                    ),
                    "pdca_retry_start",
                )
                .await;
            } else {
                self.emit_sa_thought(
                    task_iri,
                    &format!("Starting PDCA cycle {}/{}", cycle_num + 1, max_cycles),
                    "pdca_cycle_start",
                )
                .await;
            }

            let mode = if current_plan.dag_jsonld.is_some() {
                crate::core::recovery::OrchestrationMode::Dag
            } else {
                crate::core::recovery::OrchestrationMode::Pdca
            };
            let is_dag_plan = current_plan.dag_jsonld.is_some();
            let result = self
                .execute_plan(
                    current_plan,
                    task_iri,
                    user_input,
                    five_w2h.clone(),
                    &five_w2h_iri,
                    resumed,
                    cycle_feedback.clone(),
                )
                .await?;

            // Verify-first: the AA's finish action hardcodes status "success" even when it
            // concluded full execution is needed, so that status is unusable here. Only treat
            // a verify-first cycle as complete when the AA verdict explicitly confirms the
            // task is already done; otherwise fall through to the retry loop which runs the
            // stored fallback_steps (full PDCA) on the next cycle.
            let needs_execution_after_verify = cycle_num == 0
                && plan.verify_first
                && verify_aa_needs_execution(&result);

            let decision_report = if result.status == "success" && !needs_execution_after_verify {
                crate::core::recovery::DecisionReport {
                    mode,
                    directive: crate::core::recovery::RecoveryDirective::Accept,
                    reason: crate::core::recovery::RecoveryReason::Accepted,
                    scope: crate::core::recovery::RepairScope::Task,
                    plan_revision,
                }
            } else {
                crate::core::recovery::DecisionReport {
                    mode,
                    directive: crate::core::recovery::RecoveryDirective::ReplanPa,
                    reason: if result.summary.contains("Dimension Audit") {
                        crate::core::recovery::RecoveryReason::PlanInvalid
                    } else {
                        crate::core::recovery::RecoveryReason::LocalExecutionGap
                    },
                    scope: crate::core::recovery::RepairScope::Task,
                    plan_revision,
                }
            };
            self.event_bus
                .emit(
                    task_iri,
                    "AA_DECISION",
                    "SA",
                    &serde_json::to_string(&decision_report).unwrap_or_else(|_| "{}".to_string()),
                )
                .await;

            let mode_action = match mode {
                crate::core::recovery::OrchestrationMode::Dag => "dag",
                crate::core::recovery::OrchestrationMode::Pdca => "pdca",
            };
            orchestration_trajectory.push(crate::core::policy_learning::TrajectoryStep {
                context: format!("orchestration:{}", five_w2h.what),
                action: mode_action.to_string(),
                candidates: if is_dag_plan {
                    vec!["pdca".to_string(), "dag".to_string()]
                } else {
                    vec!["pdca".to_string()]
                },
                reward: policy_reward(&result),
                terminal: result.status == "success" && !needs_execution_after_verify,
            });

            if result.status == "success" && !needs_execution_after_verify {
                info!(task_iri = %task_iri, cycle_num = cycle_num + 1, "PDCA cycle passed");
                self.emit_sa_thought(
                    task_iri,
                    &format!("✅ PDCA cycle #{} passed — task complete", cycle_num + 1),
                    "pdca_cycle_passed",
                )
                .await;

                if let Some(scheduler) = &self.scheduler {
                    let _ = scheduler.on_task_complete(task_iri).await;
                }
                let _ = self
                    .policy_learning
                    .record_reward(&policy_choice, policy_reward(&result));
                if let Ok(holdout) = crate::core::policy_learning::ConstrainedPolicy::load_observations(
                    self.runner.l0_store.as_ref(),
                ) {
                    if let Ok((metrics, evaluation)) = self.policy_learning.train_trajectory_gated(
                        &orchestration_trajectory,
                        &holdout,
                        0.9,
                        crate::core::policy_learning::PolicyGate::default(),
                    ) {
                        self.event_bus
                            .emit(
                                task_iri,
                                "POLICY_TRAINING",
                                "SA",
                                &serde_json::json!({"metrics": metrics, "evaluation": evaluation}).to_string(),
                            )
                            .await;
                    }
                }
                return Ok(result);
            }

            let last_cycle = cycle_num + 1 >= max_cycles;
            if last_cycle {
                info!(task_iri = %task_iri, cycle_num = cycle_num + 1, "All PDCA cycles exhausted");
                self.emit_sa_thought(
                    task_iri,
                    &format!(
                        "⚠️ All {} PDCA cycles completed without full pass — returning last result",
                        max_cycles
                    ),
                    "pdca_cycles_exhausted",
                )
                .await;
                final_result = Some(result);
                break;
            }

            // Build targeted feedback — when AA identifies execution gaps, tell PA
            // to preserve the plan structure and focus on specific DA improvements.
            // The CA→DA correction loop (in execute_plan) already handles in-cycle fixes;
            // this SA-level feedback addresses persistent failures requiring plan adjustment.
            let task_level_audit = result.summary.contains("[Recovery] scope=Task");
            let da_fixes = !task_level_audit
                && (result.summary.contains("Dimension Audit")
                    || result.summary.contains("execution failed")
                    || result.summary.contains("not completed"));

            cycle_feedback = Some(if da_fixes {
                format!(
                    "PDCA Cycle #{} (plan revision {}): AA identified execution-level issues.\n\
                     Status: {}\n\n\
                     AA Evaluation:\n{}\n\n\
                     ---\n\
                     PRESERVE your previous plan structure. Focus ONLY on:\n\
                     1. Which specific execution steps failed or were incomplete\n\
                     2. What DA needs to do differently (more detail, different approach)\n\
                     3. Do NOT create a brand new plan — refine the existing one",
                    cycle_num + 1,
                    plan_revision,
                    result.status,
                    result.summary
                )
            } else {
                format!(
                    "PDCA Cycle #{} (plan revision {}): result\nStatus: {}\nRecovery directive: replan_pa\n\n\
                     AA Evaluation:\n{}\n\n\
                     ---\n\
                     The previous plan is not sufficient. Re-plan the task at PA,\
                     preserve verified evidence where possible, and create an improved approach.",
                    cycle_num + 1,
                    plan_revision,
                    result.status,
                    result.summary
                )
            });
            final_result = Some(result);
        }

        if let Some(scheduler) = &self.scheduler {
            let _ = scheduler.on_task_complete(task_iri).await;
        }
        let final_result = final_result.unwrap_or_else(|| TaskResult {
            task_iri: task_iri.to_string(),
            status: "failed".to_string(),
            summary: "All PDCA cycles exhausted without success".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: Vec::new(),
            turn_count: 0,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            verdict: None,
            archive_iri: None,
        });
        let _ = self
            .policy_learning
            .record_reward(&policy_choice, policy_reward(&final_result));
        if let Ok(holdout) = crate::core::policy_learning::ConstrainedPolicy::load_observations(
            self.runner.l0_store.as_ref(),
        ) {
            if let Ok((metrics, evaluation)) = self.policy_learning.train_trajectory_gated(
                &orchestration_trajectory,
                &holdout,
                0.9,
                crate::core::policy_learning::PolicyGate::default(),
            ) {
                self.event_bus
                    .emit(
                        task_iri,
                        "POLICY_TRAINING",
                        "SA",
                        &serde_json::json!({"metrics": metrics, "evaluation": evaluation}).to_string(),
                    )
                    .await;
            }
        }
        Ok(final_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_hints_removes_duplicates_preserving_order() {
        // Given: hints with interleaved duplicates from perception + skill discovery
        let hints = vec![
            "skill:calculator".to_string(),
            "workspace_event:main.rs modified".to_string(),
            "skill:calculator".to_string(),
            "scenario:debug loop".to_string(),
        ];
        // When: deduplicated with generous cap
        let deduped = dedup_hints(hints, 10);
        // Then: first occurrences kept, order preserved, count reduced
        assert_eq!(
            deduped,
            vec![
                "skill:calculator".to_string(),
                "workspace_event:main.rs modified".to_string(),
                "scenario:debug loop".to_string(),
            ]
        );
    }

    #[test]
    fn test_dedup_hints_respects_cap() {
        // Given: 5 distinct hints and a cap of 3
        let hints = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        // When: deduplicated with cap 3
        let deduped = dedup_hints(hints, 3);
        // Then: only the first 3 distinct hints remain
        assert_eq!(deduped, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_dedup_hints_empty_input() {
        // Given: empty hints
        let hints: Vec<String> = Vec::new();
        // When: deduplicated
        let deduped = dedup_hints(hints, 5);
        // Then: empty output
        assert!(deduped.is_empty());
    }
}
