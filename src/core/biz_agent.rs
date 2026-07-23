use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::core::agent_instance::{AgentInstance, AgentRole, AgentStatus};
use crate::core::agent_runner::{AgentRunner, TaskContext, TaskResult};
use crate::memory::l1_session::L1Session;
use crate::CoreError;

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum sub-agent count
    pub max_sub_agents: usize,
    /// Maximum LLM call iterations
    pub max_iterations: u32,
    /// Whether to use orchestrator mode (decomposition + sub-agents)
    pub orchestrator_mode: bool,
    /// Whether sub-agents execute in parallel
    pub parallel_sub_agents: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_sub_agents: 5,
            max_iterations: 10,
            orchestrator_mode: true,
            parallel_sub_agents: true,
        }
    }
}

/// Unified BizAgent — PA/DA/CA/AA share the same Agent class
///
/// Architecture:
/// - SA creates agent.md (prompt) and starts a BizAgent
/// - BizAgent.execute() runs in one of two modes:
///   - MONO: direct LLM call + tools
///   - ORCHESTRATOR: decompose → spawn sub BizAgents (same role) → aggregate
/// - Each role has different decompose/aggregate logic
/// - Sub-agent count is limited by AgentConfig.max_sub_agents
pub struct BizAgent {
    pub instance: AgentInstance,
    pub agent_md: String,
    pub config: AgentConfig,
    runner: Arc<AgentRunner>,
    #[allow(dead_code)]
    session: Option<L1Session>,
    #[allow(dead_code)]
    sub_agents: Vec<BizAgent>,
    sub_results: Vec<TaskResult>,
}

impl BizAgent {
    pub fn new(
        agent_id: String,
        role: AgentRole,
        agent_md: &str,
        runner: Arc<AgentRunner>,
        config: AgentConfig,
    ) -> Self {
        Self {
            instance: AgentInstance::new(agent_id, role),
            agent_md: agent_md.to_string(),
            config,
            runner,
            session: None,
            sub_agents: Vec::new(),
            sub_results: Vec::new(),
        }
    }

    pub fn agent_id(&self) -> &str {
        &self.instance.agent_id
    }
    pub fn role(&self) -> AgentRole {
        self.instance.role
    }
    pub fn status(&self) -> &AgentStatus {
        &self.instance.status
    }

    /// Main entry: execute task.
    /// In orchestrator mode, delegates to decompose→sub-agents→aggregate.
    pub async fn execute(&mut self, context: TaskContext) -> TaskResult {
        self.instance.status = AgentStatus::Running;
        info!(agent = %self.agent_id(), role = %self.role(), "BizAgent start");

        let task_iri = context.task_iri.clone();
        let mut session = {
            let mut mm = self.runner.memory_manager.lock().await;
            mm.create_session(self.agent_id(), &self.role().to_string(), &context.task_iri)
        };

        let result = if self.config.orchestrator_mode && self.should_decompose(&context) {
            self.execute_orchestrator(context, &mut session).await
        } else {
            self.execute_mono(context).await
        };

        {
            let mut mm = self.runner.memory_manager.lock().await;
            let _ = mm.finalize_session(session, &task_iri);
        }

        self.instance.status = AgentStatus::Completed;
        result
    }

    // ========== MONO mode ==========

    /// Single Agent direct execution, delegates to AgentRunner.execute().
    /// AgentRunner.execute() internally creates and manages an L1 session.
    async fn execute_mono(&self, context: TaskContext) -> TaskResult {
        let result: Result<TaskResult, CoreError> = self
            .runner
            .execute(&mut self.instance.clone(), context)
            .await;

        match result {
            Ok(r) => r,
            Err(e) => TaskResult {
                task_iri: String::new(),
                status: "failed".to_string(),
                summary: e.to_string(),
                output: None,
                jsonld_output: None,
                artifacts: Vec::new(),
                errors: vec![e.to_string()],
                turn_count: 0,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
            },
        }
    }

    // ========== ORCHESTRATOR mode ==========

    fn should_decompose(&self, _context: &TaskContext) -> bool {
        if self.config.max_sub_agents == 0 {
            return false;
        }
        if !self.config.orchestrator_mode {
            return false;
        }
        true
    }

    async fn execute_orchestrator(
        &mut self,
        context: TaskContext,
        session: &mut L1Session,
    ) -> TaskResult {
        let sub_tasks = self.decompose(&context).await;
        let sub_count = sub_tasks.len().min(self.config.max_sub_agents);

        if sub_count == 0 {
            return self.execute_mono(context).await;
        }

        info!(agent = %self.agent_id(), sub_count = sub_count, "Decomposing task");

        let sub_contexts: Vec<TaskContext> = sub_tasks
            .into_iter()
            .take(sub_count)
            .enumerate()
            .map(|(i, ctx)| TaskContext {
                objective: format!("[Sub-{}#{}] {}", self.role(), i, ctx.objective),
                ..ctx
            })
            .collect();

        self.sub_results.clear();

        if self.config.parallel_sub_agents {
            let mut handles = Vec::new();

            for (i, sub_ctx) in sub_contexts.into_iter().enumerate() {
                let sub_id = format!("{}_sub_{}", self.agent_id(), i);
                let runner = self.runner.clone();
                let agent_md = self.agent_md.clone();
                let config = AgentConfig {
                    orchestrator_mode: false,
                    ..self.config.clone()
                };
                let role = self.role();

                let handle = tokio::spawn(async move {
                    let sub = BizAgent::new(sub_id, role, &agent_md, runner, config);
                    sub.execute_mono(sub_ctx).await
                });

                handles.push(handle);
            }

            for handle in handles {
                match handle.await {
                    Ok(result) => {
                        session.add_summary(
                            "assistant",
                            &format!("[Subtask] {}", result.summary),
                            None,
                        );
                        self.sub_results.push(result);
                    }
                    Err(e) => {
                        warn!("Sub-agent execution failed: {}", e);
                        self.sub_results.push(TaskResult {
                            task_iri: String::new(),
                            status: "failed".to_string(),
                            summary: format!("Sub-agent execution failed: {}", e),
                            output: None,
                            jsonld_output: None,
                            artifacts: Vec::new(),
                            errors: vec![e.to_string()],
                            turn_count: 0,
                            tool_call_count: 0,
                            five_w2h_updates: None,
                            tracked_actions: Vec::new(),
                            archive_iri: None,
                        });
                    }
                }
            }
        } else {
            for (i, sub_ctx) in sub_contexts.into_iter().enumerate() {
                let sub_id = format!("{}_sub_{}", self.agent_id(), i);
                let sub = BizAgent::new(
                    sub_id,
                    self.role(),
                    &self.agent_md,
                    self.runner.clone(),
                    AgentConfig {
                        orchestrator_mode: false,
                        ..self.config.clone()
                    },
                );
                let result = sub.execute_mono(sub_ctx).await;
                session.add_summary(
                    "assistant",
                    &format!("[Subtask{}] {}", i, result.summary),
                    None,
                );
                self.sub_results.push(result);
            }
        }

        let final_result = self.aggregate(&context).await;
        session.add_summary("assistant", &final_result.summary, None);
        final_result
    }

    // ========== Role decomposition ==========

    async fn decompose(&self, context: &TaskContext) -> Vec<TaskContext> {
        match self.role() {
            AgentRole::Plan => self.decompose_plan_with_llm(context).await,
            AgentRole::Do => self.decompose_do_with_llm(context).await,
            AgentRole::Check => self.decompose_check_with_llm(context).await,
            AgentRole::Act => self.decompose_act_with_llm(context).await,
        }
    }

    async fn decompose_plan_with_llm(&self, context: &TaskContext) -> Vec<TaskContext> {
        self.decompose_with_llm(context, "planning", 
            "Decompose the task into multiple independent planning sub-tasks, each with clear goals and boundaries").await
    }

    async fn decompose_do_with_llm(&self, context: &TaskContext) -> Vec<TaskContext> {
        self.decompose_with_llm(context, "execution",
            "Decompose the execution task into multiple independent implementation units, each completable independently").await
    }

    async fn decompose_check_with_llm(&self, context: &TaskContext) -> Vec<TaskContext> {
        self.decompose_with_llm(context, "verification",
            "Decompose the check task into multiple independent verification dimensions (e.g., functional, performance, security)").await
    }

    async fn decompose_act_with_llm(&self, context: &TaskContext) -> Vec<TaskContext> {
        self.decompose_with_llm(context, "decision",
            "Decompose the decision task into multiple independent decision points, each with clear options and evaluation criteria").await
    }

    /// Generic LLM decomposition method
    async fn decompose_with_llm(
        &self,
        context: &TaskContext,
        phase: &str,
        instruction: &str,
    ) -> Vec<TaskContext> {
        let prompt = format!(
            r#"You are a task decomposition expert. Please decompose the following {} task into multiple independent sub-tasks.

## Original Task
{}

## Decomposition Guidance
{}

## Output Requirements
Output the decomposed sub-task list as a JSON array, each sub-task includes:
- "description": sub-task description (concise and clear)
- "priority": priority (high/medium/low)
- "dependencies": indices of other sub-tasks this depends on (array, 0-based)

Example format:
[
  {{"description": "Sub-task 1 description", "priority": "high", "dependencies": []}},
  {{"description": "Sub-task 2 description", "priority": "medium", "dependencies": [0]}}
]

If the task does not need decomposition (already atomic), return:
[{{"description": "Original task", "priority": "high", "dependencies": []}}]

Output only the JSON array, no other content."#,
            phase, context.objective, instruction
        );

        let messages = vec![crate::gateway::unified_gateway::ChatMessage {
            role: "user".to_string(),
            content: prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let model = self
            .runner
            .gateway
            .get_model(&self.role().to_string().to_lowercase());

        match self
            .runner
            .gateway
            .chat_with_params(&model, messages, None, None, None, None)
            .await
        {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    if let Some(content) = &choice.message.content {
                        return self.parse_decomposition_result(content, context);
                    }
                }
            }
            Err(e) => {
                warn!("LLM decomposition failed: {}, using original task", e);
            }
        }

        vec![context.clone()]
    }

    fn parse_decomposition_result(&self, content: &str, context: &TaskContext) -> Vec<TaskContext> {
        let json_str = if content.starts_with('[') {
            content.to_string()
        } else {
            if let Some(start) = content.find('[') {
                if let Some(end) = content.rfind(']') {
                    content[start..=end].to_string()
                } else {
                    content.to_string()
                }
            } else {
                content.to_string()
            }
        };

        match serde_json::from_str::<Value>(&json_str) {
            Ok(Value::Array(tasks)) => {
                let sub_tasks: Vec<TaskContext> = tasks
                    .iter()
                    .enumerate()
                    .filter_map(|(i, task)| {
                        let desc = task
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or(&context.objective);

                        Some(TaskContext {
                            task_iri: context.task_iri.clone(),
                            objective: format!("[Sub-{}#{}] {}", self.role(), i, desc),
                            ..context.clone()
                        })
                    })
                    .collect();

                if sub_tasks.is_empty() {
                    vec![context.clone()]
                } else {
                    info!("LLM decomposition succeeded: {} sub-tasks", sub_tasks.len());
                    sub_tasks
                }
            }
            _ => {
                warn!("Cannot parse LLM decomposition result, using original task");
                vec![context.clone()]
            }
        }
    }

    // ========== Role aggregation ==========

    async fn aggregate(&self, context: &TaskContext) -> TaskResult {
        match self.role() {
            AgentRole::Plan => self.aggregate_with_llm(context, "planning").await,
            AgentRole::Do => self.aggregate_with_llm(context, "execution").await,
            AgentRole::Check => self.aggregate_with_llm(context, "verification").await,
            AgentRole::Act => self.aggregate_with_llm(context, "decision").await,
        }
    }

    async fn aggregate_with_llm(&self, context: &TaskContext, phase: &str) -> TaskResult {
        let simple_result = self.aggregate_results(phase);

        if self.sub_results.len() <= 1 {
            return simple_result;
        }

        let sub_summaries: Vec<String> = self
            .sub_results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("Sub-task{} [{}]: {}", i + 1, r.status, r.summary))
            .collect();

        let prompt = format!(
            r#"You are a result aggregation expert. Please summarize the results of multiple sub-tasks from the following {} phase.

## Original Task
{}

## Sub-task Results
{}

## Output Requirements
Output the aggregation result as JSON:
{{
  "summary": "Overall result summary (no more than 300 characters)",
  "key_findings": ["Key finding 1", "Key finding 2"],
  "recommendations": ["Recommendation 1", "Recommendation 2"],
  "overall_status": "success/partial/failed"
}}

Output only the JSON, no other content."#,
            phase,
            context.objective,
            sub_summaries.join("\n")
        );

        let messages = vec![crate::gateway::unified_gateway::ChatMessage {
            role: "user".to_string(),
            content: prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let model = self
            .runner
            .gateway
            .get_model(&self.role().to_string().to_lowercase());

        match self
            .runner
            .gateway
            .chat_with_params(&model, messages, None, None, None, None)
            .await
        {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    if let Some(content) = &choice.message.content {
                        return self.parse_aggregation_result(content, &simple_result);
                    }
                }
            }
            Err(e) => {
                warn!("LLM aggregation failed: {}, using simple aggregation", e);
            }
        }

        simple_result
    }

    fn parse_aggregation_result(&self, content: &str, fallback: &TaskResult) -> TaskResult {
        let json_str = if content.starts_with('{') {
            content.to_string()
        } else {
            if let Some(start) = content.find('{') {
                if let Some(end) = content.rfind('}') {
                    content[start..=end].to_string()
                } else {
                    content.to_string()
                }
            } else {
                content.to_string()
            }
        };

        match serde_json::from_str::<Value>(&json_str) {
            Ok(parsed) => {
                let summary = parsed
                    .get("summary")
                    .and_then(|s| s.as_str())
                    .unwrap_or(&fallback.summary)
                    .to_string();

                let status = parsed
                    .get("overall_status")
                    .and_then(|s| s.as_str())
                    .unwrap_or(&fallback.status)
                    .to_string();

                let mut artifacts = Vec::new();
                if let Some(findings) = parsed.get("key_findings").and_then(|f| f.as_array()) {
                    artifacts.push(json!({"type": "key_findings", "items": findings}));
                }
                if let Some(recommendations) =
                    parsed.get("recommendations").and_then(|r| r.as_array())
                {
                    artifacts.push(json!({"type": "recommendations", "items": recommendations}));
                }

                info!("LLM aggregation succeeded");
                TaskResult {
                    task_iri: fallback.task_iri.clone(),
                    status,
                    summary,
                    output: Some(json!({"aggregated": true})),
                    jsonld_output: None,
                    artifacts,
                    errors: fallback.errors.clone(),
                    turn_count: fallback.turn_count,
                    tool_call_count: fallback.tool_call_count,
                    five_w2h_updates: None,
                    tracked_actions: Vec::new(),
                    archive_iri: None,
                }
            }
            _ => {
                warn!("Cannot parse LLM aggregation result, using simple aggregation");
                fallback.clone()
            }
        }
    }

    fn aggregate_results(&self, _phase: &str) -> TaskResult {
        let total = self.sub_results.len();
        let successes = self
            .sub_results
            .iter()
            .filter(|r| r.status == "success")
            .count();
        let mut all_errors = Vec::new();

        let mut summary_parts = Vec::new();
        for (i, r) in self.sub_results.iter().enumerate() {
            summary_parts.push(format!("  [{}] {}: {}", i, r.status, r.summary));
            all_errors.extend(r.errors.clone());
        }

        let summary = format!(
            "Aggregated {} sub-tasks: {}/{} succeeded\n{}",
            total,
            successes,
            total,
            summary_parts.join("\n"),
        );

        TaskResult {
            task_iri: String::new(),
            status: if successes == total {
                "success".to_string()
            } else {
                "partial".to_string()
            },
            summary,
            output: None,
            jsonld_output: None,
            artifacts: Vec::new(),
            errors: all_errors,
            turn_count: 0,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        }
    }
}
