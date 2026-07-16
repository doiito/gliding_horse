use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::{json, Value};
use tracing::{debug, info, warn};

        use crate::core::agent_instance::{AgentInstance, AgentRole, AgentStatus};
use crate::core::execution_event::{ExecutionEvent, ExecutionEventKind};
use crate::core::system_prompt::{SystemPromptBuilder, SystemPromptRegion, build_constitution_prompt, build_time_awareness_text};
use crate::gateway::unified_gateway::ChatMessage;
use crate::jsonld::{JsonLdContext, JsonLdNode};
use crate::memory::l1_session::L1Session;
use crate::methodology::integration::MethodologyPromptInjector;
use crate::tools::hooks::{HookContext, HookPoint, HookResult};
use crate::tools::tool_executor::ToolExecutor;
use crate::CoreError;

use super::{TaskContext, TaskResult, LLM_RESPONSE_FORMAT_NO_THOUGHT, LLM_RESPONSE_FORMAT_WITH_THOUGHT};

impl super::AgentRunner {
    pub async fn execute(
        &self,
        agent: &mut AgentInstance,
        ctx: TaskContext,
    ) -> Result<TaskResult, CoreError> {
        // AgentInit hook
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::AgentInit,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            self.hook_manager
                .execute(HookPoint::AgentInit, &mut hook_ctx)
                .await;
        }

        agent.status = AgentStatus::Running;

        // TaskStart hook
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::TaskStart,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            let hook_result = self
                .hook_manager
                .execute(HookPoint::TaskStart, &mut hook_ctx)
                .await;
            if hook_result == HookResult::Abort {
                agent.status = AgentStatus::Failed;
                return Ok(TaskResult {
                    task_iri: ctx.task_iri,
                    status: "aborted".to_string(),
                    summary: "Task aborted by hook".to_string(),
                    output: None,
                    jsonld_output: None,
                    artifacts: Vec::new(),
                    errors: vec!["Task aborted by hook".to_string()],
                    turn_count: 0,
                    tool_call_count: 0,
                    five_w2h_updates: None,
                    tracked_actions: Vec::new(),
                    archive_iri: None,
                });
            }
        }

        // AgentStart hook
        let mut hook_ctx = HookContext::new(
            HookPoint::AgentStart,
            &agent.agent_id,
            &agent.role.to_string(),
        )
        .with_task(&ctx.task_iri, &ctx.task_iri);
        let hook_result = self
            .hook_manager
            .execute(HookPoint::AgentStart, &mut hook_ctx)
            .await;
        if hook_result == HookResult::Abort {
            agent.status = AgentStatus::Failed;
            return Ok(TaskResult {
                task_iri: ctx.task_iri,
                status: "aborted".to_string(),
                summary: "Agent aborted by hook".to_string(),
                output: None,
                jsonld_output: None,
                artifacts: Vec::new(),
                errors: vec!["Agent aborted by hook".to_string()],
                turn_count: 0,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
            });
        }

        let mut session = self.memory_manager.lock().await.create_session(
            &agent.agent_id,
            &agent.role.to_string(),
            &ctx.task_iri,
        );

        // Compute task embedding for semantic relevance pruning
        if let Some(ref embedder) = self.embedder {
            if let Ok(task_emb) = embedder.embed(&ctx.objective).await {
                session.set_task_embedding(task_emb.clone());
                if let Some(ref tracker_lock) = self.relevance_tracker {
                    let mut tracker = tracker_lock.lock().unwrap();
                    tracker.reset();
                    tracker.set_task_context(task_emb);
                }
            }
        }

        // MemoryWrite hook for session creation
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::MemoryWrite,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            self.hook_manager
                .execute(HookPoint::MemoryWrite, &mut hook_ctx)
                .await;
        }

        let result = self.exec(agent, ctx.clone(), &mut session).await;

        {
            let mut mm = self.memory_manager.lock().await;
            if !result.as_ref().map(|r| r.tracked_actions.is_empty()).unwrap_or(true) {
                if let Ok(ref r) = result {
                    let _ = mm.archive_session_actions(&r.task_iri, &r.tracked_actions, &r.summary);
                    if !r.tracked_actions.is_empty() {
                        let success_rate = r.tracked_actions.iter()
                            .filter(|a| a.status == crate::core::tracked_action::ActionStatus::Success).count() as f32
                            / r.tracked_actions.len().max(1) as f32;
                        let _ = mm.archive_experience(&r.task_iri, &agent.role.to_string(), &r.summary, success_rate);
                    }
                }
            }
            let _ = mm.finalize_session(session, &ctx.task_iri);
        }

        // TaskEnd hook
        {
            let mut hook_ctx =
                HookContext::new(HookPoint::TaskEnd, &agent.agent_id, &agent.role.to_string())
                    .with_task(&ctx.task_iri, &ctx.task_iri);
            self.hook_manager
                .execute(HookPoint::TaskEnd, &mut hook_ctx)
                .await;
        }

        // AgentEnd hook
        let mut hook_ctx = HookContext::new(
            HookPoint::AgentEnd,
            &agent.agent_id,
            &agent.role.to_string(),
        );
        self.hook_manager
            .execute(HookPoint::AgentEnd, &mut hook_ctx)
            .await;

        // Handle errors
        if let Ok(ref r) = result {
            if r.status == "failed" {
                let mut hook_ctx = HookContext::new(
                    HookPoint::AgentError,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri);
                hook_ctx.error = Some(r.summary.clone());
                self.hook_manager
                    .execute(HookPoint::AgentError, &mut hook_ctx)
                    .await;

                let mut hook_ctx = HookContext::new(
                    HookPoint::TaskError,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri);
                hook_ctx.error = Some(r.summary.clone());
                self.hook_manager
                    .execute(HookPoint::TaskError, &mut hook_ctx)
                    .await;
            }
        }

        result
    }

    /// Execute task using an independent BizAgent instance (Agent isolation)
    pub async fn execute_with_biz_agent(
        &self,
        agent: &AgentInstance,
        ctx: TaskContext,
        plan_step: Option<crate::core::sa::PlanStep>,
    ) -> Result<TaskResult, CoreError> {
        use crate::core::biz_agent::{AgentConfig, BizAgent};

        // AgentInit hook
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::AgentInit,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            self.hook_manager
                .execute(HookPoint::AgentInit, &mut hook_ctx)
                .await;
        }

        // Build independent agent.md
        let context_data = self.gather_context_data_async(agent.role, &ctx).await;
        let agent_md = if let Some(ref step) = plan_step {
            self.build_agent_md_from_step(agent.role, step, &context_data)
        } else {
            let model = self
                .gateway
                .get_model(&agent.role.to_string().to_lowercase());
            self.build_agent_md(agent.role, &ctx.objective, &context_data, &model)
        };

        // Create BizAgent configuration
        let config = AgentConfig {
            orchestrator_mode: false,
            max_sub_agents: 5,
            max_iterations: ctx.max_iterations,
            parallel_sub_agents: true,
        };

        // Create independent BizAgent instance
        let mut biz_agent = BizAgent::new(
            agent.agent_id.clone(),
            agent.role,
            &agent_md,
            Arc::new((*self).clone()),
            config,
        );

        // TaskStart hook
        {
            let mut hook_ctx = HookContext::new(
                HookPoint::TaskStart,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            let hook_result = self
                .hook_manager
                .execute(HookPoint::TaskStart, &mut hook_ctx)
                .await;
            if hook_result == HookResult::Abort {
            return Ok(TaskResult {
                task_iri: ctx.task_iri,
                status: "aborted".to_string(),
                summary: "Agent aborted by hook".to_string(),
                output: None,
                jsonld_output: None,
                artifacts: Vec::new(),
                errors: vec!["Agent aborted by hook".to_string()],
                turn_count: 0,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
            });
            }
        }

        // AgentStart hook
        let mut hook_ctx = HookContext::new(
            HookPoint::AgentStart,
            &agent.agent_id,
            &agent.role.to_string(),
        )
        .with_task(&ctx.task_iri, &ctx.task_iri);
        let hook_result = self
            .hook_manager
            .execute(HookPoint::AgentStart, &mut hook_ctx)
            .await;
        if hook_result == HookResult::Abort {
            return Ok(TaskResult {
                task_iri: ctx.task_iri,
                status: "aborted".to_string(),
                summary: "Agent aborted by hook".to_string(),
                output: None,
                jsonld_output: None,
                artifacts: Vec::new(),
                errors: vec!["Agent aborted by hook".to_string()],
                turn_count: 0,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
            });
        }

        // Execute BizAgent (isolated environment)
        let result = biz_agent.execute(ctx.clone()).await;

        // TaskEnd hook
        {
            let mut hook_ctx =
                HookContext::new(HookPoint::TaskEnd, &agent.agent_id, &agent.role.to_string())
                    .with_task(&ctx.task_iri, &ctx.task_iri);
            self.hook_manager
                .execute(HookPoint::TaskEnd, &mut hook_ctx)
                .await;
        }

        // AgentEnd hook
        let mut hook_ctx = HookContext::new(
            HookPoint::AgentEnd,
            &agent.agent_id,
            &agent.role.to_string(),
        );
        self.hook_manager
            .execute(HookPoint::AgentEnd, &mut hook_ctx)
            .await;

        // Handle errors
        if result.status == "failed" {
            let mut hook_ctx = HookContext::new(
                HookPoint::AgentError,
                &agent.agent_id,
                &agent.role.to_string(),
            )
            .with_task(&ctx.task_iri, &ctx.task_iri);
            hook_ctx.error = Some(result.summary.clone());
            self.hook_manager
                .execute(HookPoint::AgentError, &mut hook_ctx)
                .await;
        }

        Ok(result)
    }

    /// In force-finish scenarios, extract tool results from messages and call LLM for final aggregated summary.
    /// Returns (summary, full_content), or None if no tool results are aggregatable or LLM fails.
    async fn aggregate_tool_results(
        &self,
        messages: &[ChatMessage],
        agent: &AgentInstance,
        ctx: &TaskContext,
    ) -> Option<(String, String)> {
        // Extract assistant messages with tool_calls and corresponding tool results
        let tool_entries: Vec<(String, String)> = messages
            .windows(2)
            .filter_map(|w| {
                if w[0].role == "assistant" && w[0].tool_calls.is_some() && w[1].role == "tool" {
                    let tool_names: Vec<&str> = w[0]
                        .tool_calls
                        .as_ref()
                        .map(|calls| calls.iter().map(|tc| tc.function.name.as_str()).collect())
                        .unwrap_or_default();
                    Some((tool_names.join(", "), w[1].content.clone()))
                } else {
                    None
                }
            })
            .collect();

        if tool_entries.is_empty() {
            return None;
        }

        let prompt_parts: Vec<String> = tool_entries
            .iter()
            .map(|(name, result)| {
                let truncated = if result.len() > 2000 {
                    format!("{}...\n[truncated, original {} chars]", &result[..2000], result.len())
                } else {
                    result.clone()
                };
                format!("## Tool: {}\n{}", name, truncated)
            })
            .collect();

        let prompt = format!(
            r#"You are an AI assistant. Below are all tool call results from your task execution. Please generate a complete summary report based on these results.

## Original Task Objective
{}

## Tool Call Records and Results
{}

## Output Requirements
1. Summarize task completion status
2. List key findings and results
3. Provide final conclusions
4. If the above results are insufficient for a complete report, produce the best summary possible based on available information

Output the summary report directly, not in JSON format."#,
            ctx.objective,
            prompt_parts.join("\n\n"),
        );

        let model = self.gateway.get_model(&agent.role.to_string().to_lowercase());
        let req_messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        match self.gateway.chat_with_params(&model, req_messages, None, None, None, None).await {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    if let Some(content) = &choice.message.content {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() {
                            let summary = Self::generate_auto_summary(trimmed);
                            return Some((summary, trimmed.to_string()));
                        }
                    }
                }
                warn!("[force-finish] LLM aggregation returned empty content");
                None
            }
            Err(e) => {
                warn!("[force-finish] LLM aggregation call failed: {}", e);
                None
            }
        }
    }


    async fn exec(
        &self,
        agent: &AgentInstance,
        ctx: TaskContext,
        sess: &mut L1Session,
    ) -> Result<TaskResult, CoreError> {
        let model = self
            .gateway
            .get_model(&agent.role.to_string().to_lowercase());
        let supports_reasoning = self.gateway.supports_native_reasoning(&model);

        let context_data = self.gather_context_data_async(agent.role, &ctx).await;
        let agent_md = self.build_agent_md(agent.role, &ctx.objective, &context_data, &model);

        // Build system prompt using SystemPromptBuilder
        let mut prompt_builder = SystemPromptBuilder::new();

        // Region 1: Role definition area
        prompt_builder.set_region(SystemPromptRegion::RoleDefinition, agent_md.clone());

        // Region 2: Time awareness — current time and session context
        {
            let session_start = sess.created_at().format("%Y-%m-%d %H:%M:%S UTC").to_string();
            let time_text = build_time_awareness_text(Some(&session_start));
            prompt_builder.set_region(SystemPromptRegion::TimeAwareness, time_text);
        }

        // Region 3: Workspace environment info area (let Agent know its workspace boundaries)
        if let Some(ref ws_root) = self.workspace_root {
            let env_info = format!(
                "## Workspace\n\n- Workspace path: {}\n\
                 - All file operations (read, write, search, command execution) must stay within the workspace\n\
                 - Files outside the workspace are unrelated to the current task and must not be accessed\n\
                 - The workspace root may contain other directories and files unrelated to the current task — distinguish carefully",
                ws_root.display()
            );
            prompt_builder.set_region(SystemPromptRegion::EnvironmentInfo, env_info);
        }

        // Region 2: Behavioral policy area (constitution layer + methodology layer)
        {
            let mut policy_text = build_constitution_prompt(agent.role);

            policy_text.push_str("\n\n### 🔴 Task Focus Principles (Mandatory)\n");
            policy_text.push_str("- Your only task is the designated 'Current Task'. All other directories/files in the workspace are unrelated to your task\n");
            policy_text.push_str("- Irrelevant files or directories (e.g. other projects, test artifacts, unrelated codebases) must be directly ignored — do not explore or process them\n");
            policy_text.push_str("- When using glob_search, file_list or similar tools, if results contain irrelevant content, automatically filter it out — do not get distracted\n");
            policy_text.push_str("- If you encounter files/directories not belonging to the current task, skip them and continue executing the current task — do not change direction due to irrelevant content\n");
            policy_text.push_str("- Check Agent (CA) special note: your audit report may only contain content related to the current task. Irrelevant files found must be ignored and not written into the report\n");
            policy_text.push_str("- Decision Agent (AA) special note: do NOT proactively explore files. Your decisions must be based solely on CA audit results, ignoring any irrelevant content in the audit\n");
            policy_text.push_str("\n### 📖 File Reading Efficiency Principles (Mandatory)\n");
            policy_text.push_str("- Only read files relevant to the current task. Files that have been 'written but not re-read' are output from other agents — only read them when you need to reference their content\n");
            policy_text.push_str("- Do not re-read the same file. If file_read returns from_cache=true, the content is unchanged and was already provided — skip re-reading and continue with what you have\n");
            policy_text.push_str("- Do NOT try mode:force_refresh just because file_read returns from_cache=true — this only wastes tokens reading unchanged content\n");
            policy_text.push_str("- For files already read, their content is already in your context. No need to re-confirm or re-verify\n");

            // Inject methodology discipline (PA/CA/AA specific)
            if let Some(methodology_addendum) = MethodologyPromptInjector::build_for_role(agent.role) {
                policy_text.push_str(&methodology_addendum);
            }
            // Inject active methodology persuasive directives
            if let Some(ref gate) = self.methodology_gate {
                let directives = gate.inner().read().persuasive_directives();
                if !directives.is_empty() {
                    policy_text.push_str("\n\n### Methodology Execution Requirements\n");
                    for d in &directives {
                        policy_text.push_str(&format!("- {}\n", d));
                    }
                }
            }
            // AA specific: inject methodology evolution briefing
            if agent.role == AgentRole::Act {
                if let Some(ref gate) = self.methodology_gate {
                    if let Some(ref evo) = gate.evolution_handle() {
                        let briefing = evo.inner().read().aa_evolution_briefing();
                        if !briefing.is_empty() {
                            policy_text.push_str("\n\n");
                            policy_text.push_str(&briefing);
                        }
                    }
                }
            }
            prompt_builder.set_region(SystemPromptRegion::BehavioralPolicy, policy_text);
        }

        // Region 3: Emphasized constraints area (loaded from L0)
        let emphasis_items = self.load_emphasis_from_l0(&ctx.task_iri).await;
        if !emphasis_items.is_empty() {
            let emphasis_content = emphasis_items
                .iter()
                .map(|e| format!("- {}", e))
                .collect::<Vec<_>>()
                .join("\n");
            prompt_builder.set_region(SystemPromptRegion::EmphasizedConstraints, emphasis_content);
        }

        // Region 3: Output format area
        let format_constraint = if supports_reasoning {
            LLM_RESPONSE_FORMAT_NO_THOUGHT.to_string()
        } else {
            LLM_RESPONSE_FORMAT_WITH_THOUGHT.to_string()
        };
        prompt_builder.set_region(SystemPromptRegion::OutputFormat, format_constraint);

        // Region 4: Output management area
        prompt_builder.set_region(
            SystemPromptRegion::OutputManagement,
            crate::core::system_prompt::OUTPUT_MANAGEMENT.to_string(),
        );

        // Region 5: Tools area (built-in tools + dynamic tools)
        let tool_menu = self.build_readable_tool_menu(&agent.role);
        if !tool_menu.is_empty() {
            prompt_builder.set_region(SystemPromptRegion::Tools, tool_menu);
        }

        // Region 5: Extraction prompt area (loaded from config)
        if let Some(ref config) = self.emphasis_config {
            if config.enabled {
                prompt_builder.set_region(
                    SystemPromptRegion::ExtractionPrompt,
                    config.extraction_prompt.clone(),
                );
            }
        }

        // Build system prompt (relatively static, placed in system role)
        let system_content = prompt_builder.build();

        // Build context message (dynamic, placed in the final user role)
        let summary_iris = sess.get_summary_chain_with_iris(20, 100);
        let summary_text = summary_iris.join("\n");

        let mut task_parts = vec![format!("## Current Task\n{}", ctx.objective)];
        if !ctx.expected_output.is_empty() {
            task_parts.push(format!("## Expected Output\n{}", ctx.expected_output));
        }
        if !ctx.success_criteria.is_empty() {
            task_parts.push(format!("## Success Criteria\n{}", ctx.success_criteria));
        }
        let task_section = task_parts.join("\n\n");

        let context_msg = if summary_text.is_empty() {
            format!(
                "{}\n\n## Available Tools\nUse tools as needed to complete the task.",
                task_section,
            )
        } else {
            format!(
                "{}\n\n## History Summary\n{}\n\nTo view the full report of a specific turn, use the read_agent_output tool with the corresponding IRI.\n\n## Available Tools\nUse tools as needed to complete the task.",
                task_section, summary_text
            )
            .to_string()
        };

        let mut messages: Vec<ChatMessage> = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_content,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        {
            let executor = self.tool_executor.read();
            if let Some(wm) = executor.get_workspace_monitor() {
                wm.snapshots().create_snapshot("pre_task", Some(&ctx.task_iri));
                wm.inject_file_perception(Some(&ctx.objective));
            }
        }

        // Agent active perception area: environment-level perception data from system components (file changes, batch analysis, alerts, etc.)
        // Placed after system and before history messages so LLM sees global environment state first
        let perception_text = self.perception_store.take_perception_text(&ctx.task_iri);
        if !perception_text.is_empty() {
            info!(
                "[perception] injecting {} bytes of perception content",
                perception_text.len()
            );
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: format!("# Agent Perception\n\n{}", perception_text),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }

        // Proactive KG context injection: query the knowledge graph for entities relevant to the task
        // and inject them as environment context before the agent begins reasoning.
        if let Some(ref kg_store) = self.unified_graph_store {
            let kg_context = Self::build_kg_context(kg_store, &ctx.objective);
            if !kg_context.is_empty() {
                info!(
                    "[kg_context] injecting {} bytes of knowledge graph context",
                    kg_context.len()
                );
                messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: format!("# Knowledge Graph Context\n\n{}", kg_context),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
        }

        // Resume mode: restore history messages from checkpoint, placed after system and before new user message
        // So LLM sees historical context first, then the continue instruction
        if let Some(ref resumed) = ctx.resumed_messages {
            // Skip original system message (replaced with new one), append remaining history
            for msg in resumed.iter().skip(1) {
                messages.push(msg.clone());
            }
            info!("[resume] restored {} history messages from checkpoint", resumed.len().saturating_sub(1));
        }

        // New user message placed after history as continue instruction
        let resume_task_parts = if ctx.expected_output.is_empty() && ctx.success_criteria.is_empty() {
            format!("Current Task: {}", ctx.objective)
        } else {
            let mut parts = vec![format!("Current Task: {}", ctx.objective)];
            if !ctx.expected_output.is_empty() {
                parts.push(format!("Expected Output: {}", ctx.expected_output));
            }
            if !ctx.success_criteria.is_empty() {
                parts.push(format!("Success Criteria: {}", ctx.success_criteria));
            }
            parts.join("\n")
        };
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: if ctx.resumed_messages.is_some() {
                format!(
                    "[Continue] Please continue the task from where you left off.\n\n{}",
                    resume_task_parts
                )
            } else {
                context_msg
            },
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });

        let tools = self
            .tool_executor
            .read()
            .tool_definitions_for_role(&agent.role.to_string());

        info!(
            "AgentRunner start: role={}, model={}, tools={}, supports_reasoning={}",
            agent.role,
            model,
            tools.len(),
            supports_reasoning
        );

        let mut tc = ctx.resumed_tool_count;
        let mut errs = Vec::new();
        let mut turn = ctx.resumed_turn_count;
        let mut consecutive_failures = 0u32;
        let mut recovery_mode_active = false;
        let mut guard_pending_pre_injections: Vec<String> = Vec::new();
        // Track error count per tool, early terminate if same tool fails repeatedly
        let mut tool_error_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut tool_recovery_injected: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut action_tracker = crate::core::tracked_action::ActionTracker::new(
            &ctx.task_iri,
            &agent.role.to_string(),
        );
        let checkpoint_manager = crate::core::checkpoint::CheckpointManager::with_persistence(self.l0_store.clone());

        // Track the richest content turn (used for passing archive_iri across agents, pointing to substantive content rather than final turn summary)
        let mut best_content_len: usize = 0;
        let mut best_content_str: String = String::new();
        let mut best_content_iri: String = String::new();

        let effective_max_turns = match agent.role {
            AgentRole::Plan => ctx.max_iterations.min(200),
            AgentRole::Do => ctx.max_iterations,
            AgentRole::Check => ctx.max_iterations.min(200),
            AgentRole::Act => ctx.max_iterations.min(150),
        };

        // Initial checkpoint: record task start state
        let start_role_str = agent.role.to_string();
        if let Err(e) = checkpoint_manager.create_ext(
            &ctx.task_iri,
            &format!("start_{}", agent.role),
            "[]",
            &serde_json::to_string(&messages).unwrap_or_default(),
            &serde_json::json!({
                "turn": ctx.resumed_turn_count,
                "tc": ctx.resumed_tool_count,
                "prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
                "completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
            }).to_string(),
            &[start_role_str.clone()],
            Some(&start_role_str),
            None, None, None, None, None, None,
            None, None, None,
        ) {
            warn!("[checkpoint] initial save failed: {}", e);
        }

        // Soft limit state: progressive prompts, no hard truncation (DA and AA use 3-stage degradation)
        let mut soft_limit_early_warning_sent = false;
        let mut soft_limit_final_warning_sent = false;
        let mut soft_limit_force_finish = false;

        loop {
            // --- Soft limit phase 1: Early warning (~8 turns remaining) ---
            if !soft_limit_early_warning_sent && turn >= effective_max_turns.saturating_sub(8) {
                soft_limit_early_warning_sent = true;
                warn!("[turn {}] soft limit warning: ~8 turns remaining (max={})", turn, effective_max_turns);
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "【Turn Limit Notice】Please control execution turns. Limited turns remain. Focus on the core task, avoid unnecessary tool calls, and finish as soon as possible.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            // --- Soft limit phase 2: Final warning (~3 turns remaining) ---
            if !soft_limit_final_warning_sent && turn >= effective_max_turns.saturating_sub(3) {
                soft_limit_final_warning_sent = true;
                warn!("[turn {}] soft limit final warning: ~3 turns remaining (max={})", turn, effective_max_turns);
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: "【Turn Limit Urgent】Only 3 turns remaining. Please finish your current work and output the final result immediately. Do not initiate new tool calls.".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            // --- Soft limit phase 3: Force finish (inject directive at limit, let LLM respond, no truncation) ---
            if turn >= effective_max_turns {
                if !soft_limit_force_finish {
                    soft_limit_force_finish = true;
                    warn!("[turn {}] max turns {} reached, injecting force-finish directive (no truncation)", turn, effective_max_turns);
                    errs.push("max turns reached".to_string());
                    if let Some(ref event_bus) = self.event_bus {
                        let _ = event_bus.emit(&ctx.task_iri, "AGENT_BLOCKED", &agent.agent_id, &serde_json::json!({"iterations": turn}).to_string()).await;
                    }
                    let max_role_str = agent.role.to_string();
                    let tool_error_str = serde_json::json!({
                        "error_counts": tool_error_counts,
                        "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                    }).to_string();
                    let action_str = serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                    if let Err(e) = checkpoint_manager.create_ext(
                        &ctx.task_iri,
                        &format!("max_turns_{}", agent.role),
                        "[]",
                        &serde_json::to_string(&messages).unwrap_or_default(),
                        &serde_json::json!({
                            "turn": turn,
                            "tc": tc,
                            "prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
                            "completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
                        }).to_string(),
                        &[max_role_str.clone()],
                        Some(&max_role_str),
                        None, None, None, None, None, None,
                        Some(&tool_error_str),
                        Some(&action_str),
                        None,
                    ) {
                        warn!("[checkpoint] max_turns save failed: {}", e);
                    }
                    messages.push(ChatMessage {
                        role: "user".to_string(),
                        content: "【System Force-Finish】Maximum execution turns reached. Please output your final summary and results immediately. Do not call any more tools. If there are incomplete tool executions, base your summary on the results already available.".to_string(),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                    // Don't break, let this turn's LLM respond to the force-finish directive
                } else {
                    // Force-finish already injected, LLM still hasn't completed -> hard stop, take last assistant reply
                    warn!("[turn {}] LLM still not completed after force-finish, hard stopping", turn);
                    let force_role_str = agent.role.to_string();
                    let tool_error_str = serde_json::json!({
                        "error_counts": tool_error_counts,
                        "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                    }).to_string();
                    let action_str = serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                    let _ = checkpoint_manager.create_ext(
                        &ctx.task_iri,
                        &format!("force_end_{}", agent.role),
                        "[]",
                        &serde_json::to_string(&messages).unwrap_or_default(),
                        &serde_json::json!({
                            "turn": turn,
                            "tc": tc,
                        }).to_string(),
                        &[force_role_str.clone()],
                        Some(&force_role_str),
                        None, None, None, None, None, None,
                        Some(&tool_error_str),
                        Some(&action_str),
                        None,
                    );
                    // Fallback: if no turn has substantive content, aggregate tool results via LLM
                    let (force_summary, force_output, force_archive) =
                        if !best_content_str.is_empty() {
                            let s = Self::generate_auto_summary(&best_content_str);
                            (s, Some(Value::String(best_content_str.clone())),
                             if !best_content_iri.is_empty() { Some(best_content_iri.clone()) } else { None })
                        } else if let Some((agg_summary, agg_content)) =
                            self.aggregate_tool_results(&messages, agent, &ctx).await
                        {
                            (agg_summary, Some(Value::String(agg_content)),
                             if !best_content_iri.is_empty() { Some(best_content_iri.clone()) } else { None })
                        } else if let Some(last) = messages.iter().rev().find(|m| m.role == "assistant") {
                            (Self::generate_auto_summary(&last.content),
                             Some(Value::String(last.content.clone())), None)
                        } else {
                            ("Task not completed".to_string(), None, None)
                        };
                    return Ok(TaskResult {
                        task_iri: ctx.task_iri,
                        status: "partial_success".to_string(),
                        summary: force_summary,
                        output: force_output,
                        jsonld_output: None,
                        artifacts: vec![],
                        errors: errs,
                        turn_count: turn,
                        tool_call_count: tc,
                        five_w2h_updates: None,
                        tracked_actions: action_tracker.actions,
                        archive_iri: force_archive,
                    });
                }
            }
            turn += 1;

            // Save periodic checkpoint every 5 turns (including tool error state)
            if turn % 5 == 0 {
                let turn_role_str = agent.role.to_string();
                let tool_error_str = serde_json::json!({
                    "error_counts": tool_error_counts,
                    "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                }).to_string();
                let action_str = serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                if let Err(e) = checkpoint_manager.create_ext(
                    &ctx.task_iri,
                    &format!("turn_{}_{}", agent.role, turn),
                    "[]",
                    &serde_json::to_string(&messages).unwrap_or_default(),
                    &serde_json::json!({
                        "turn": turn,
                        "tc": tc,
                        "prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
                        "completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
                    }).to_string(),
                    &[turn_role_str.clone()],
                    Some(&turn_role_str),
                    None, None, None, None, None, None,
                    Some(&tool_error_str),
                    Some(&action_str),
                    None,
                ) {
                    warn!("[checkpoint] periodic save failed (turn={}): {}", turn, e);
                }
            }

            // Failure mode detection and recovery mode
            if consecutive_failures >= 3 && !recovery_mode_active {
                recovery_mode_active = true;
                let recovery_msg = format!(
                    "[System Diagnostic] Detected {} consecutive operation failures. Pause execution, analyze the cause, and propose an alternative approach.\
                     \n\nFailure record: {}\n\nPlease re-evaluate the current method and consider alternatives before continuing.",
                    consecutive_failures,
                    errs.last().map(|e| e.as_str()).unwrap_or("multiple failures")
                );
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: recovery_msg,
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                info!("[consecutive_failures] triggered recovery mode: {} consecutive failures", consecutive_failures);
                consecutive_failures = 0;
                continue;
            }

            // ===== Thought Phase =====
            info!("[ReAct Turn {}] ===== Thought =====", turn);

            // CycleStart: inject supplementary input (SA writes -> AgentRunner consumes)
            {
                let pending = self.supplement_store.take_pending(&ctx.task_iri);
                if !pending.is_empty() {
                    info!(
                        task_iri = %ctx.task_iri,
                        count = pending.len(),
                        "injecting {} supplementary inputs into AgentRunner context",
                        pending.len()
                    );
                    for entry in &pending {
                        messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: entry.content.clone(),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        });
                        sess.add_supplement("user", &entry.content, entry.embedding.clone(), Some(entry.relevance_score));
                    }
                }
            }

            // CycleStart hook
            {
                let mut hook_ctx = HookContext::new(
                    HookPoint::CycleStart,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri)
                .with_data("turn", Value::Number(turn.into()));
                self.hook_manager
                    .execute(HookPoint::CycleStart, &mut hook_ctx)
                    .await;
            }

            {
                let mut hook_ctx = HookContext::new(
                    HookPoint::LlmRequest,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri);
                let hook_result = self
                    .hook_manager
                    .execute(HookPoint::LlmRequest, &mut hook_ctx)
                    .await;
                if hook_result == HookResult::Abort {
                    errs.push("LLM request aborted by hook".to_string());
                    break;
                }
            }

            // Use ContextWindowManager for dual-dimension compression based on message count and tokens
            let context_window_compressed = if let Some(ref cwm_lock) = self.context_window_manager {
                let cwm = cwm_lock.lock().expect("cwm_lock Mutex poisoned");
                if cwm.should_compress(messages.len(), &messages) {
                    let (compressed, summary_text) = cwm.compress_messages(&messages);
                    if !summary_text.is_empty() {
                        sess.add_summary("system", &summary_text, None);
                    }
                    info!(
                        "[turn {}] ContextWindowManager compressed: {} -> {} messages",
                        turn,
                        messages.len(),
                        compressed.len()
                    );
                    Some(compressed)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(compressed) = context_window_compressed {
                messages = compressed;
            } else if messages.len() > 30 {
                // Fallback: hard truncation (only triggered when CWM is unavailable or misconfigured)
                let system_msg = messages.first().cloned();
                let kept_recent = messages.len().saturating_sub(15);

                let mut recent: Vec<_> = messages.drain(kept_recent..).collect();

                while !recent.is_empty() {
                    let first = &recent[0];
                    if first.role == "tool" {
                        recent.remove(0);
                        continue;
                    }
                    if first.role == "assistant" {
                        if let Some(ref tool_calls) = first.tool_calls {
                            let expected_tool_results = tool_calls.len();
                            let actual_tool_results = recent
                                .iter()
                                .skip(1)
                                .take_while(|m| m.role == "tool")
                                .count();
                            if actual_tool_results < expected_tool_results {
                                recent.remove(0);
                                continue;
                            }
                        }
                    }
                    break;
                }

                messages.clear();
                if let Some(sys) = system_msg {
                    messages.push(sys);
                }

                let summary_iris = sess.get_summary_chain_with_iris(10, 100);
                let summary_text = if summary_iris.is_empty() {
                    format!(
                        "[History Summary] Previously executed {} turns with {} tool calls. Here is the recent conversation:",
                        turn - 1, tc
                    )
                } else {
                    format!(
                        "[History Summary] Executed {} turns. Key records:\n{}\n\nFor details, use kg_search / knowledge_query with the IRI.",
                        turn - 1,
                        summary_iris.join("\n")
                    )
                };

                let summary_note = ChatMessage {
                    role: "user".to_string(),
                    content: summary_text,
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                };
                messages.push(summary_note);
                messages.extend(recent);

                info!(
                    "[turn {}] message history truncated: kept {} (original {} messages)",
                    turn,
                    messages.len(),
                    kept_recent + 17
                );
            }

            if !guard_pending_pre_injections.is_empty() {
                let prompt = format!(
                    "\n\n[ToolGuard Constraint Directive]\n{}\nNote: The above constraints only apply to the same-named tool calls you make next. Strictly comply.",
                    guard_pending_pre_injections.join("\n")
                );
                if let Some(sys_msg) = messages.first_mut() {
                    if sys_msg.role == "system" {
                        // Replace rather than append: remove old ToolGuard block to prevent cumulative bloat per turn
                        if let Some(pos) = sys_msg.content.find("\n\n[ToolGuard Constraint Directive]") {
                            sys_msg.content.truncate(pos);
                        }
                        sys_msg.content.push_str(&prompt);
                    }
                }
                guard_pending_pre_injections.clear();
            }

            debug!(
                "[turn {}] calling LLM (history: {} msgs, tools: {})",
                turn,
                messages.len(),
                tools.len()
            );

            let response = self
                .gateway
                .chat_with_params(
                    &model,
                    messages.clone(),
                    None,
                    None,
                    {
                        // Refresh tools list before each call to ensure newly registered micro-tools (e.g. read_full_result_*) are included
                        let current_tools = self
                            .tool_executor
                            .read()
                            .tool_definitions_for_role(&agent.role.to_string());
                        if current_tools.is_empty() { None } else { Some(current_tools) }
                    },
                    None,
                )
                .await
                .map_err(|e| CoreError::Internal {
                    message: e.to_string(),
                })?;

            // Accumulate token usage
            if let Some(ref usage) = response.usage {
                self.total_prompt_tokens.fetch_add(usage.prompt_tokens as u64, Ordering::Relaxed);
                self.total_completion_tokens.fetch_add(usage.completion_tokens as u64, Ordering::Relaxed);
                self.last_prompt_tokens.store(usage.prompt_tokens as u64, Ordering::Relaxed);
                self.last_completion_tokens.store(usage.completion_tokens as u64, Ordering::Relaxed);
            }

            {
                let mut hook_ctx = HookContext::new(
                    HookPoint::LlmResponse,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri);
                self.hook_manager
                    .execute(HookPoint::LlmResponse, &mut hook_ctx)
                    .await;
            }

            let choice = response
                .choices
                .first()
                .ok_or_else(|| CoreError::Internal {
                    message: "No choices in response".to_string(),
                })?;
            let raw_content = choice.message.content.clone().unwrap_or_default();
            let reasoning_content = choice.message.reasoning_content.clone();
            let finish = choice.finish_reason.as_deref().unwrap_or("");

            debug!(
                "[turn {}] LLM response: finish={}, content_len={}, has_reasoning={}",
                turn,
                finish,
                raw_content.len(),
                reasoning_content.is_some()
            );

            let parsed = self.parse_llm_response(
                &raw_content,
                reasoning_content.as_deref(),
                supports_reasoning,
            );

            if !parsed.is_valid_json && finish != "tool_calls" {
                warn!("[turn {}] LLM response is not valid JSON, using fallback", turn);
                consecutive_failures += 1;
                debug!("[consecutive_failures] JSON parse failed: {}/3", consecutive_failures);
            }

            let mut action = parsed
                .action
                .clone()
                .unwrap_or_else(|| "continue".to_string());

            if finish == "tool_calls" && choice.message.tool_calls.is_some() {
                action = "tool_call".to_string();
                debug!(
                    "[turn {}] finish=tool_calls with tool_calls present, forcing action=tool_call",
                    turn
                );
            }

            if (finish == "stop" || finish == "end_turn") && action != "tool_call" {
                if action != "finish" {
                    debug!(
                        "[turn {}] finish={} with no tool calls, correcting action from {} to finish",
                        turn, finish, action
                    );
                }
                action = "finish".to_string();
            }

            info!(
                "[ReAct Turn {}] Thought: action={}, summary={}",
                turn,
                action,
                parsed.summary.as_deref().unwrap_or("")
            );

            // Emit thought event to event bus for TUI display
            if let Some(ref event_bus) = self.event_bus {
                let thought_content = parsed.thought.clone().unwrap_or_default();
                let thought_event = ExecutionEvent {
                    event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                    task_iri: ctx.task_iri.clone(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                    event: ExecutionEventKind::Thought(crate::core::execution_event::Thought {
                        agent_id: agent.agent_id.clone(),
                        thought: if thought_content.is_empty() { parsed.content.clone() } else { thought_content },
                        action: action.clone(),
                        emphasis: parsed.emphasis.clone(),
                    }),
                };
                let _ = event_bus.emit(
                    &ctx.task_iri,
                    "THOUGHT",
                    &agent.agent_id,
                    &serde_json::to_string(&thought_event).unwrap_or_default(),
                ).await;
            }

            // Save emphasis content to L0 persistent memory
            if !parsed.emphasis.is_empty() {
                let dedup_threshold = self
                    .emphasis_config
                    .as_ref()
                    .map(|c| c.dedup_threshold)
                    .unwrap_or(0.85);
                self.save_emphasis_to_l0(
                    &parsed.emphasis,
                    &ctx.task_iri,
                    &agent.agent_id,
                    dedup_threshold,
                )
                .await;
            }

            // Archive to L0: save full response + thought content
            let l0_iri = sess
                .archive_full_to_l0(
                    &self.l0_store,
                    &agent.role.to_string(),
                    &parsed.thought.clone().unwrap_or_default(),
                    &parsed.content,
                )
                .ok();
            debug!(
                "[L0] archived: {:?}, has_reasoning={}, is_valid_json={}",
                l0_iri, parsed.has_native_reasoning, parsed.is_valid_json
            );

            // MemoryWrite hook for L0 archive
            {
                let mut hook_ctx = HookContext::new(
                    HookPoint::MemoryWrite,
                    &agent.agent_id,
                    &agent.role.to_string(),
                )
                .with_task(&ctx.task_iri, &ctx.task_iri)
                .with_data("storage", Value::String("L0".to_string()));
                if let Some(ref iri) = l0_iri {
                    hook_ctx
                        .data
                        .insert("iri".to_string(), Value::String(iri.clone()));
                }
                self.hook_manager
                    .execute(HookPoint::MemoryWrite, &mut hook_ctx)
                    .await;
            }

            let task_id = ctx.task_iri
                .strip_prefix("iri://task/")
                .unwrap_or_else(|| ctx.task_iri.strip_prefix("iri://").unwrap_or(&ctx.task_iri));
            let node_iri = format!(
                "iri://task/{}/turn_{}",
                task_id,
                turn
            );
            if parsed.content.len() > best_content_len {
                best_content_len = parsed.content.len();
                best_content_str = parsed.content.clone();
                best_content_iri.clone_from(&node_iri);
            }
            let mut node_json = json!({
                "@id": &node_iri,
                "@type": "AgentTurn",
                "role": agent.role.to_string(),
                "cycle_id": ctx.cycle_id,
                "content": parsed.content,
                "content_len": parsed.content.len(),
                "is_valid_json": parsed.is_valid_json,
                "has_native_reasoning": parsed.has_native_reasoning
            });
            if let Some(ref thought) = parsed.thought {
                node_json["has_thought"] = Value::Bool(true);
                node_json["thought_len"] = Value::Number(thought.len().into());
            }
            if let Some(ref act) = parsed.action {
                node_json["action"] = Value::String(act.clone());
            }
            if let Some(ref s) = parsed.summary {
                node_json["summary"] = Value::String(s.clone());
            }
            JsonLdContext::inject(&mut node_json);
            let cfg = crate::CoreConfig::default();
            match self
                .blackboard
                .write_node(&node_iri, &node_json.to_string(), &cfg)
            {
                Ok(_) => {
                    debug!("[L2] writing node: {}", node_iri);

                    // BlackboardWrite hook
                    let mut hook_ctx = HookContext::new(
                        HookPoint::BlackboardWrite,
                        &agent.agent_id,
                        &agent.role.to_string(),
                    )
                    .with_task(&ctx.task_iri, &ctx.task_iri)
                    .with_data("node_iri", Value::String(node_iri.clone()));
                    self.hook_manager
                        .execute(HookPoint::BlackboardWrite, &mut hook_ctx)
                        .await;
                }
                Err(e) => {
                    warn!("[L2] failed to write node {}: {:?}", node_iri, e);
                }
            }

            // Use parsed summary or generate fallback
            let summary_text = parsed
                .summary
                .clone()
                .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
            let l1_turn = sess.add_summary(&agent.role.to_string(), &summary_text, l0_iri.clone());
            // Compute turn embedding and relevance_score
            if let (Some(ref embedder), Some(ref tracker_lock)) = (&self.embedder, &self.relevance_tracker) {
                if let Ok(emb) = embedder.embed(&summary_text).await {
                    let mut tracker = tracker_lock.lock().unwrap();
                    let score = tracker.on_new_input(&emb);
                    l1_turn.embedding = Some(emb);
                    l1_turn.relevance_score = Some(score);
                }
            }

            // ===== Action Phase =====
            info!("[ReAct Turn {}] ===== Action =====", turn);

            match action.as_str() {
                "finish" => {
                    info!("[ReAct] Agent decided to complete task");

                    // CycleEnd hook
                    {
                        let mut hook_ctx = HookContext::new(
                            HookPoint::CycleEnd,
                            &agent.agent_id,
                            &agent.role.to_string(),
                        )
                        .with_task(&ctx.task_iri, &ctx.task_iri)
                        .with_data("turn", Value::Number(turn.into()))
                        .with_data("had_tool_calls", Value::Bool(false));
                        self.hook_manager
                            .execute(HookPoint::CycleEnd, &mut hook_ctx)
                            .await;
                    }

                    info!("AgentRunner completed: {} turns, {} tools", turn, tc);
                    debug!("[L0] L0 entries: {}", self.l0_store.count().unwrap_or(0));

                    // When parsed.content is empty (LLM returned content=null + tool_calls),
                    // aggregate from tool results in messages to ensure subsequent agent can read a valid plan
                    let (final_summary, output_value) =
                        if !parsed.content.trim().is_empty() {
                            (parsed.summary.clone()
                                .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content)),
                             Value::String(parsed.content.clone()))
                        } else if let Some((agg_summary, agg_content)) =
                            self.aggregate_tool_results(&messages, agent, &ctx).await
                        {
                            (agg_summary, Value::String(agg_content))
                        } else {
                            (parsed.summary.clone()
                                .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content)),
                             Value::String(parsed.content.clone()))
                        };
                    let jsonld_output =
                        self.apply_output_mapping(&output_value, &agent.role, &ctx.task_iri);

                    if let Some(ref jsonld) = jsonld_output {
                        if let Ok(node) = JsonLdNode::from_json(jsonld) {
                            let emphasis_items = self.extract_emphasis(&node);
                            if !emphasis_items.is_empty() {
                                let dedup_threshold = self
                                    .emphasis_config
                                    .as_ref()
                                    .map(|c| c.dedup_threshold)
                                    .unwrap_or(0.85);
                                self.save_emphasis_to_l0(
                                    &emphasis_items,
                                    &ctx.task_iri,
                                    &agent.agent_id,
                                    dedup_threshold,
                                )
                                .await;
                            }

                            if let Ok(node_iri) =
                                self.store_jsonld_to_l2(&node, &ctx.task_iri).await
                            {
                                debug!("[L2] JSON-LD output stored: {}", node_iri);
                            }
                        }
                    }

                    let nodes_str = jsonld_output
                        .as_ref()
                        .map(|j| j.to_string())
                        .unwrap_or_else(|| "[]".to_string());
                    let finish_role_str = agent.role.to_string();
                    let tool_error_str = serde_json::json!({
                        "error_counts": tool_error_counts,
                        "recovery_injected": tool_recovery_injected.iter().cloned().collect::<Vec<_>>(),
                    }).to_string();
                    let action_str = serde_json::to_string(&action_tracker.actions).unwrap_or_default();
                    if let Err(e) = checkpoint_manager.create_ext(
                        &ctx.task_iri,
                        &format!("finish_{}", agent.role),
                        &nodes_str,
                        &serde_json::to_string(&messages).unwrap_or_default(),
                        &serde_json::json!({
                            "turn": turn,
                            "tc": tc,
                            "prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
                            "completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
                        }).to_string(),
                        &[finish_role_str.clone()],
                        Some(&finish_role_str),
                        None, None, None, None, None, None,
                        Some(&tool_error_str),
                        Some(&action_str),
                        None,
                    ) {
                        warn!("[checkpoint] finish save failed: {}", e);
                    }

                    // Point to the turn with the longest content (not the last summary),
                    // so dispatch_agent can get substantive content when reading from L2.
                    let archive_iri = if !best_content_iri.is_empty() {
                        Some(best_content_iri.clone())
                    } else {
                        Some(node_iri.clone())
                    };
                    return Ok(TaskResult {
                        task_iri: ctx.task_iri,
                        status: "success".to_string(),
                        summary: final_summary,
                        output: Some(output_value),
                        jsonld_output,
                        artifacts: vec![],
                        errors: errs,
                        turn_count: turn,
                        tool_call_count: tc,
                        five_w2h_updates: None,
                        tracked_actions: action_tracker.actions,
                        archive_iri,
                    });
                }
                "tool_call" => {
                    // After soft limit phase 3: intercept tool calls, force current output as final result
                    if soft_limit_force_finish {
                        warn!("[force-finish] intercepted tool_call={:?}, forcing final output", 
                            choice.message.tool_calls.as_ref().map(|c| {
                                c.iter().map(|t| t.function.name.as_str()).collect::<Vec<_>>()
                            }));
                        // If parsed.content is empty (tool_calls-only response), try LLM aggregation of existing tool results
                        let (final_summary, output_value) =
                            if !parsed.content.trim().is_empty() {
                                (parsed.summary.clone()
                                    .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content)),
                                 Value::String(parsed.content.clone()))
                            } else if let Some((agg_summary, agg_content)) =
                                self.aggregate_tool_results(&messages, agent, &ctx).await
                            {
                                (agg_summary, Value::String(agg_content))
                            } else {
                                (parsed.summary.clone()
                                    .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content)),
                                 Value::String(parsed.content.clone()))
                            };
                        let jsonld_output =
                            self.apply_output_mapping(&output_value, &agent.role, &ctx.task_iri);
                        let intercept_archive = if !best_content_iri.is_empty() {
                            Some(best_content_iri.clone())
                        } else {
                            None
                        };
                        let force_status = if soft_limit_force_finish && errs.iter().any(|e| e.contains("max turns") || e.contains("force-finish")) {
                            "partial_success"
                        } else {
                            "success"
                        };
                        return Ok(TaskResult {
                            task_iri: ctx.task_iri,
                            status: force_status.to_string(),
                            summary: final_summary,
                            output: Some(output_value),
                            jsonld_output,
                            artifacts: vec![],
                            errors: errs,
                            turn_count: turn,
                            tool_call_count: tc,
                            five_w2h_updates: None,
                            tracked_actions: action_tracker.actions,
                            archive_iri: intercept_archive,
                        });
                    }

                    if let Some(calls) = &choice.message.tool_calls {
                        let tool_names: Vec<&str> =
                            calls.iter().map(|c| c.function.name.as_str()).collect();
                        debug!("[tool_calls] {} → {:?}", calls.len(), tool_names);

                        // 🔴 PA role forbidden from calling write tools, but read-only tools allowed
                        if agent.role == AgentRole::Plan {
                            let write_tools: Vec<&str> = calls
                                .iter()
                                .map(|c| c.function.name.as_str())
                                .filter(|name| !ToolExecutor::is_pa_readonly_tool(name))
                                .collect();

                            let force_finish = if let Some(ref tc) = self.tool_controller {
                                let tool_calls: Vec<(String, Value)> = calls.iter()
                                    .map(|c| (c.function.name.clone(), serde_json::from_str(&c.function.arguments).unwrap_or_default()))
                                    .collect();
                                tc.should_force_finish(&tool_calls, &agent.role)
                            } else {
                                !write_tools.is_empty()
                            };

                            if force_finish {
                                warn!(
                                    "[PA] detected write tool call: {:?}, forcing finish",
                                    write_tools
                                );
                                info!("[ReAct] PA Agent force-ended (write operations prohibited)");

                                let (final_summary, output_value) =
                                    if !parsed.content.trim().is_empty() {
                                        (parsed.summary.clone()
                                            .unwrap_or_else(|| "PA has formulated a plan".to_string()),
                                         Value::String(parsed.content.clone()))
                                    } else if let Some((agg_summary, agg_content)) =
                                        self.aggregate_tool_results(&messages, agent, &ctx).await
                                    {
                                        (agg_summary, Value::String(agg_content))
                                    } else {
                                        (parsed.summary.clone()
                                            .unwrap_or_else(|| "PA has formulated a plan".to_string()),
                                         Value::String(parsed.content.clone()))
                                    };
                                let jsonld_output = self.apply_output_mapping(
                                    &output_value,
                                    &agent.role,
                                    &ctx.task_iri,
                                );

                                let pa_archive_iri = if !best_content_iri.is_empty() {
                                    Some(best_content_iri.clone())
                                } else {
                                    Some(node_iri.clone())
                                };
                                return Ok(TaskResult {
                                    task_iri: ctx.task_iri,
                                    status: "success".to_string(),
                                    summary: final_summary,
                                    output: Some(output_value),
                                    jsonld_output,
                                    artifacts: vec![],
                                    errors: errs,
                                    turn_count: turn,
                                    tool_call_count: tc,
                                    five_w2h_updates: None,
                                    tracked_actions: Vec::new(),
                                    archive_iri: pa_archive_iri,
                                });
                            }
                        }

                        let asst_summary = parsed.summary.clone()
                            .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: asst_summary,
                            name: None,
                            tool_calls: Some(
                                calls
                                    .iter()
                                    .map(|c| crate::gateway::unified_gateway::ToolCallPayload {
                                        id: c.id.clone(),
                                        call_type: c.call_type.clone(),
                                        function:
                                            crate::gateway::unified_gateway::ToolCallFunction {
                                                name: c.function.name.clone(),
                                                arguments: c.function.arguments.clone(),
                                            },
                                    })
                                    .collect(),
                            ),
                            tool_call_id: None,
                            reasoning_content: reasoning_content.clone(),
                        });

                        for c in calls {
                            tc += 1;
                            let name = &c.function.name;
                            let args_raw = &c.function.arguments;
                            let args: Value = serde_json::from_str(args_raw).unwrap_or_default();
                            debug!(
                                "  [tool] {} args={}",
                                name,
                                &args_raw.chars().take(100).collect::<String>()
                            );

                            // Emit tool_call event for TUI display
                            if let Some(ref event_bus) = self.event_bus {
                                let tce = ExecutionEvent {
                                    event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                                    task_iri: ctx.task_iri.clone(),
                                    timestamp: chrono::Utc::now().timestamp_millis(),
                                    event: ExecutionEventKind::ToolCall(crate::core::execution_event::ToolCall {
                                        call_id: c.id.clone(),
                                        tool_name: name.clone(),
                                        arguments_json: args_raw.clone(),
                                        agent_id: agent.agent_id.clone(),
                                        sequence: tc,
                                    }),
                                };
                                let _ = event_bus.emit(
                                    &ctx.task_iri,
                                    "TOOL_CALL",
                                    &agent.agent_id,
                                    &serde_json::to_string(&tce).unwrap_or_default(),
                                ).await;
                            }

                            {
                                let mut hook_ctx = HookContext::new(
                                    HookPoint::SkillBefore,
                                    &agent.agent_id,
                                    &agent.role.to_string(),
                                )
                                .with_task(&ctx.task_iri, &ctx.task_iri)
                                .with_data("tool_name", Value::String(name.clone()));
                                self.hook_manager
                                    .execute(HookPoint::SkillBefore, &mut hook_ctx)
                                    .await;
                                // Capture ToolGuard pre-injections for next LLM call
                                if let Some(injections) = hook_ctx.metadata.remove("guard_pre_injections") {
                                    if let Value::Array(arr) = injections {
                                        for v in arr {
                                            if let Some(s) = v.as_str() {
                                                guard_pending_pre_injections.push(s.to_string());
                                            }
                                        }
                                    }
                                }
                            }

                            let handler = {
                                let executor = self.tool_executor.read();
                                executor.try_get_handler(name)
                            };
                            let started_at = std::time::Instant::now();
                            let args_clone = args.clone();
                            let result = match handler {
                                Some(f) => f(args).await.unwrap_or_else(|e| json!({"error": e})),
                                None => json!({"error": format!("Tool not found: {}", name)}),
                            };
                            action_tracker.record(name, &args_clone, &result, started_at.elapsed().as_secs_f64());
                            let raw_result_str = serde_json::to_string(&result).unwrap_or_default();

                            let mut result_str = self.route_tool_result(
                                &raw_result_str,
                                name,
                                &c.id,
                            ).await;

                            debug!("  [tool] {} result: {} bytes (raw: {} bytes)", name, result_str.len(), raw_result_str.len());

                            // Emit tool_result event for TUI display
                            if let Some(ref event_bus) = self.event_bus {
                                let tre = ExecutionEvent {
                                    event_id: format!("evt_{}", uuid::Uuid::new_v4().hyphenated()),
                                    task_iri: ctx.task_iri.clone(),
                                    timestamp: chrono::Utc::now().timestamp_millis(),
                                    event: ExecutionEventKind::ToolResult(crate::core::execution_event::ToolResult {
                                        call_id: c.id.clone(),
                                        tool_name: name.clone(),
                                        result: result_str.clone(),
                                        success: result.get("error").is_none(),
                                        result_size_bytes: result_str.len() as u32,
                                        duration_ms: 0,
                                        agent_id: agent.agent_id.clone(),
                                    }),
                                };
                                let _ = event_bus.emit(
                                    &ctx.task_iri,
                                    "TOOL_RESULT",
                                    &agent.agent_id,
                                    &serde_json::to_string(&tre).unwrap_or_default(),
                                ).await;
                            }

                            if let Some(ref compressor_lock) = self.tool_result_compressor {
                                if let Ok(mut compressor) = compressor_lock.lock() {
                                    compressor.add_result(turn, name, &c.id, &result_str);
                                    compressor.compress_tool_messages(&mut messages);
                                }
                            }
                            self.compress_tool_results_with_microtools(&mut messages);

                            // Cross-turn aging: compress old tool results by staleness
                            if let Some(ref aging) = self.tool_result_aging {
                                aging.age_tool_results(&mut messages, &self.tool_executor);
                            }

                            if let Some(err) = result.get("error") {
                                let err_msg = err.as_str().unwrap_or("");
                                let is_tool_not_found = err_msg.starts_with("Tool not found: ");
                                warn!("[tool] {} failed: {}", name, err);
                                errs.push(format!("{}: {}", name, err));

                                if is_tool_not_found {
                                    // Micro-tool registration and handler mismatch causes "tool not found".
                                    // This is not an LLM error -- the tool list was provided by the system. Don't count as consecutive failure.
                                    // try_get_handler already attempted fallback paths; if still not found, it means
                                    // the micro-tool's validity has expired or data has been cleaned. LLM should use original tools (bash/grep etc.)
                                    // with more precise parameters to obtain needed data.
                                    // Additionally, inject prompt into tool message to guide LLM.
                                    info!("[tool_error] {} tool not found (micro-tool fallback also failed), not counting as consecutive failure", name);
                                    // Inject guidance prompt into tool message, helping LLM switch to original tools
                                    result_str = format!(
                                        "{}\n\nTip: Tool {} is currently unavailable. Please use the original tools (e.g. bash, grep_search) with more precise parameters to directly obtain the data. Do not call this micro-tool again.",
                                        result_str, name
                                    );
                                } else {
                                    // Tool execution errors don't count toward consecutive_failures.
                                    // consecutive_failures only tracks LLM-level failures (JSON parse failures, etc.).
                                    // Tool errors are normal operational feedback -- LLM has received the error and can adjust strategy.
                                    // Repeated failure of the same tool is handled by the independent tool_error_counts counter.
                                    let tool_count = tool_error_counts.entry(name.clone()).or_insert(0);
                                    *tool_count += 1;
                                    debug!("[tool_error] {} failure count: {}/3", name, *tool_count);
                                    if *tool_count >= 3 && !tool_recovery_injected.contains(name) {
                                        warn!("[tool_error] {} failed {} consecutive times, injecting recovery guidance", name, *tool_count);
                                        tool_recovery_injected.insert(name.clone());
                                        result_str = format!(
                                            "{}\n\n[System Prompt] Tool {} failed 3 consecutive times, indicating it is currently unavailable.\
                                             \nPlease use other available tools to complete the current objective (e.g. web_search / bash / grep, etc.).\
                                             \nDo not call {} again.",
                                            result_str, name, name
                                        );
                                    }
                                }
                                if let Some(ref event_bus) = self.event_bus {
                                    let _ = event_bus.emit(&ctx.task_iri, "AGENT_ERROR", &agent.agent_id, &serde_json::json!({"error": err, "tool": name}).to_string()).await;
                                }
                            } else {
                                info!("[tool] {} succeeded", name);
                                if recovery_mode_active {
                                    info!("[consecutive_failures] recovery mode exited successfully");
                                }
                                consecutive_failures = 0;
                                recovery_mode_active = false;
                                // Tool executed successfully, clear its error count and recovery flag
                                tool_error_counts.remove(name);
                                tool_recovery_injected.remove(name);
                            }

                            {
                                let mut hook_ctx = HookContext::new(
                                    HookPoint::SkillAfter,
                                    &agent.agent_id,
                                    &agent.role.to_string(),
                                )
                                .with_task(&ctx.task_iri, &ctx.task_iri)
                                .with_data("tool_name", Value::String(name.clone()))
                                .with_data("tool_result", Value::String(raw_result_str.clone()));
                                let hook_result = self.hook_manager
                                    .execute(HookPoint::SkillAfter, &mut hook_ctx)
                                    .await;

                                if hook_result == HookResult::Abort {
                                    let guard_msg = hook_ctx.error.unwrap_or_else(|| "Tool result rejected by guard".to_string());
                                    warn!("[tool] {} ToolGuard intercepted: {}", name, guard_msg);
                                    messages.push(ChatMessage {
                                        role: "tool".to_string(),
                                        content: format!("[ToolGuard Intercepted] Tool {} result rejected by security system. {}", name, guard_msg),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: Some(c.id.clone()),
                                        reasoning_content: None,
                                    });
                                } else {
                                    messages.push(ChatMessage {
                                        role: "tool".to_string(),
                                        content: result_str,
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: Some(c.id.clone()),
                                        reasoning_content: None,
                                    });
                                }
                            }
                        }

                        // ===== Observation Phase =====
                        info!("[ReAct Turn {}] ===== Observation =====", turn);

                        // CycleEnd hook (tool calls path)
                        {
                            let mut hook_ctx = HookContext::new(
                                HookPoint::CycleEnd,
                                &agent.agent_id,
                                &agent.role.to_string(),
                            )
                            .with_task(&ctx.task_iri, &ctx.task_iri)
                            .with_data("turn", Value::Number(turn.into()))
                            .with_data("had_tool_calls", Value::Bool(true));
                            self.hook_manager
                                .execute(HookPoint::CycleEnd, &mut hook_ctx)
                                .await;
                        }

                        continue;
                    } else {
                        warn!("[ReAct] action=tool_call but no tool_calls, continuing to think");
                        let asst_summary = parsed.summary.clone()
                            .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: asst_summary,
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: reasoning_content.clone(),
                        });

                        // CycleEnd hook
                        {
                            let mut hook_ctx = HookContext::new(
                                HookPoint::CycleEnd,
                                &agent.agent_id,
                                &agent.role.to_string(),
                            )
                            .with_task(&ctx.task_iri, &ctx.task_iri)
                            .with_data("turn", Value::Number(turn.into()))
                            .with_data("had_tool_calls", Value::Bool(false));
                            self.hook_manager
                                .execute(HookPoint::CycleEnd, &mut hook_ctx)
                                .await;
                        }
                    }
                }
                "continue" => {
                    let asst_summary = parsed.summary.clone()
                        .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: asst_summary,
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: reasoning_content.clone(),
                    });

                    // CycleEnd hook
                    {
                        let mut hook_ctx = HookContext::new(
                            HookPoint::CycleEnd,
                            &agent.agent_id,
                            &agent.role.to_string(),
                        )
                        .with_task(&ctx.task_iri, &ctx.task_iri)
                        .with_data("turn", Value::Number(turn.into()))
                        .with_data("had_tool_calls", Value::Bool(false));
                        self.hook_manager
                            .execute(HookPoint::CycleEnd, &mut hook_ctx)
                            .await;
                    }
                }
                _ => {
                    warn!("[ReAct] unknown action: {}, continuing to think", action);
                    let asst_summary = parsed.summary.clone()
                        .unwrap_or_else(|| Self::generate_auto_summary(&parsed.content));
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: asst_summary,
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: reasoning_content.clone(),
                    });

                    // CycleEnd hook
                    {
                        let mut hook_ctx = HookContext::new(
                            HookPoint::CycleEnd,
                            &agent.agent_id,
                            &agent.role.to_string(),
                        )
                        .with_task(&ctx.task_iri, &ctx.task_iri)
                        .with_data("turn", Value::Number(turn.into()))
                        .with_data("had_tool_calls", Value::Bool(false));
                        self.hook_manager
                            .execute(HookPoint::CycleEnd, &mut hook_ctx)
                            .await;
                    }
                }
            }
        }

        warn!("AgentRunner incomplete: {} turns, errors: {:?}", turn, errs);
        // Prefer the best content turn's output (with substantive content) over the last assistant reply's short summary
        let (unfinished_status, unfinished_summary, unfinished_output, unfinished_archive) =
            if !best_content_str.is_empty() {
                ("partial_success".to_string(),
                 Self::generate_auto_summary(&best_content_str),
                 Some(Value::String(best_content_str.clone())),
                 if !best_content_iri.is_empty() { Some(best_content_iri.clone()) } else { None })
            } else if let Some((agg_summary, agg_content)) =
                self.aggregate_tool_results(&messages, agent, &ctx).await
            {
                ("partial_success".to_string(), agg_summary, Some(Value::String(agg_content)),
                 if !best_content_iri.is_empty() { Some(best_content_iri.clone()) } else { None })
            } else if let Some(last) = messages.iter().rev().find(|m| m.role == "assistant") {
                ("partial_success".to_string(),
                 Self::generate_auto_summary(&last.content),
                 Some(Value::String(last.content.clone())), None)
            } else if tc > 0 {
                ("partial_success".to_string(),
                 format!("Task partially completed. Executed {} turns, {} tool calls, {} remaining. Errors: {}.", turn, tc, effective_max_turns.saturating_sub(turn), errs.len()),
                 None, None)
            } else {
                ("failed".to_string(), String::new(), None, None)
            };
        Ok(TaskResult {
            task_iri: ctx.task_iri,
            status: unfinished_status,
            summary: unfinished_summary,
            output: unfinished_output,
            jsonld_output: None,
            artifacts: vec![],
            errors: errs,
            turn_count: turn,
            tool_call_count: tc,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: unfinished_archive,
        })
    }

    /// Build knowledge graph context string from the unified Oxigraph store.
    /// Queries for entities (subjects with rdf:type) that have labels or names,
    /// returning them as a structured context block the LLM can use to ground its reasoning.
    fn build_kg_context(store: &oxigraph::store::Store, objective: &str) -> String {
        // Extract meaningful keywords from the task objective (words >= 3 chars, no punctuation)
        let keywords: Vec<&str> = objective
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .collect();

        let dynamic_query;
        let sparql: &str = if keywords.is_empty() {
            "\
                SELECT DISTINCT ?s ?label ?type WHERE {\
                    ?s a ?type .\
                    OPTIONAL { ?s rdfs:label ?label }\
                    OPTIONAL { ?s <http://schema.org/name> ?label }\
                } ORDER BY DESC(?label) LIMIT 10\
            "
        } else {
            let keyword_filters: String = keywords
                .iter()
                .map(|k| {
                    format!(
                        "{{ ?s <http://schema.org/name> ?name . FILTER(CONTAINS(LCASE(STR(?name)), LCASE(\"{}\"))) }} \
                         UNION {{ ?s rdfs:label ?label . FILTER(CONTAINS(LCASE(STR(?label)), LCASE(\"{}\"))) }} \
                         UNION {{ FILTER(CONTAINS(LCASE(STR(?s)), LCASE(\"{}\"))) }}",
                        k, k, k
                    )
                })
                .collect::<Vec<_>>()
                .join(" UNION ");

            dynamic_query = format!(
                "\
                    SELECT DISTINCT ?s ?label ?type WHERE {{\
                        {{ {} }} \
                        OPTIONAL {{ ?s rdfs:label ?label }} \
                        OPTIONAL {{ ?s a ?type }} \
                    }} LIMIT 10\
                ",
                keyword_filters
            );
            dynamic_query.as_str()
        };

        use oxigraph::sparql::QueryResults as Qr;
        use oxigraph::sparql::QuerySolution;
        let solutions: Vec<QuerySolution> = match store.query(sparql) {
            Ok(Qr::Solutions(it)) => it.filter_map(Result::ok).collect(),
            _ => return String::new(),
        };

        let mut lines: Vec<String> = Vec::new();
        for solution in &solutions {
            let s = solution.get("s").map(|v| format!("{}", v)).unwrap_or_default();
            let label = solution.get("label").map(|v| format!("{}", v)).unwrap_or_default();
            let type_ = solution.get("type").map(|v| format!("{}", v)).unwrap_or_default();
            if s.is_empty() {
                continue;
            }
            let name = if !label.is_empty() { label } else { s.clone() };
            let entity = if !type_.is_empty() {
                format!("- **{}** ({})", name, type_)
            } else {
                format!("- **{}**", name)
            };
            if !lines.contains(&entity) {
                lines.push(entity);
            }
        }

        if lines.is_empty() {
            return String::new();
        }

        let mut result = format!(
            "The following entities are available in the knowledge graph (task context: {}):\n\n",
            objective.chars().take(80).collect::<String>()
        );
        result.push_str(&lines.join("\n"));
        result
    }
}
