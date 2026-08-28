use std::collections::HashMap;

use serde::Deserialize;
use tracing::{info, warn};

use crate::core::agent_instance::AgentRole;
use crate::CoreError;

use super::actions::parse_or_repair_json;
use super::agent::SupervisorAgent;
use super::types::*;

// Structural protocol minima, not workload limits: standard PDCA requires one
// PA, one DA and the CA/AA terminal gates; emergency mode intentionally omits
// PA. The configurable max_plan_steps remains the workload ceiling above this
// non-negotiable protocol shape.
const STANDARD_PDCA_PROTOCOL_STEPS: usize = 4;
const EMERGENCY_PDCA_PROTOCOL_STEPS: usize = 3;

/// A token budget extracted from an LLM is only a rough estimate. Treating it
/// as a hard task limit makes an otherwise normal task fail when the model
/// happens to emit a small number such as 5000. Only a budget explicitly
/// requested by the user is authoritative.
pub(crate) fn user_explicitly_requested_token_budget(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    let markers = [
        "token budget",
        "token_budget",
        "token_limit",
        "tokenbudget",
        "token limit",
        "tokens limit",
        "令牌预算",
        "token上限",
    ];
    markers.iter().any(|marker| {
        lower.find(marker).is_some_and(|pos| {
            lower[pos + marker.len()..]
                .chars()
                .take(40)
                .any(|ch| ch.is_ascii_digit())
        })
    })
}

#[cfg(test)]
mod token_budget_tests {
    use super::user_explicitly_requested_token_budget;

    #[test]
    fn only_explicit_budget_markers_are_authoritative() {
        assert!(user_explicitly_requested_token_budget(
            "Run the task with token budget 50000"
        ));
        assert!(user_explicitly_requested_token_budget("token_limit: 12000"));
        assert!(!user_explicitly_requested_token_budget(
            "Inspect the code and report the relevant test command"
        ));
        assert!(!user_explicitly_requested_token_budget(
            "The model may estimate token budget"
        ));
    }
}

fn generated_gate_step(role: AgentRole, step_id: &str) -> PlanStep {
    let (objective, expected_output, success_criteria, effect_policy) = match role {
        AgentRole::Plan => (
            "Analyze the authoritative task contract and create the execution plan",
            "Task-scoped execution plan",
            "Plan preserves the original acceptance boundary",
            crate::core::effect::EffectPolicy::EvidenceOnly,
        ),
        AgentRole::Do => (
            "Execute every requirement in the authoritative task contract",
            "Completed task deliverable and execution evidence",
            "Original task requirements are completed",
            crate::core::effect::EffectPolicy::None,
        ),
        AgentRole::Check => (
            "Independently audit the execution against the authoritative task contract",
            "Structured audit verdict with task-relevant evidence",
            "Every original success criterion is independently verified",
            crate::core::effect::EffectPolicy::EvidenceOnly,
        ),
        AgentRole::Act => (
            "Make the terminal business decision from the latest CA audit",
            "Structured final decision and user-facing summary",
            "Decision follows the latest CA evidence without adding requirements",
            crate::core::effect::EffectPolicy::DecisionOnly,
        ),
    };
    PlanStep {
        step_id: step_id.to_string(),
        role,
        objective: objective.to_string(),
        expected_output: expected_output.to_string(),
        dependencies: Vec::new(),
        tools_allowed: Vec::new(),
        success_criteria: success_criteria.to_string(),
        branch_on_failure: false,
        branch_fallback: None,
        retry_count: 0,
        retry_delay_secs: 0,
        effect_policy,
    }
}

/// Validate only model-generated PDCA plans. Explicit JSON-LD DAGs retain
/// their authored topology and bypass this function.
fn normalize_generated_pdca_steps(
    steps: Vec<PlanStep>,
    complexity: TaskComplexity,
    configured_max_steps: usize,
) -> Vec<PlanStep> {
    if matches!(complexity, TaskComplexity::Instant | TaskComplexity::Simple) {
        return steps
            .into_iter()
            .take(configured_max_steps.max(1))
            .collect();
    }

    let requires_plan = complexity != TaskComplexity::Emergency;
    let semantic_minimum = if requires_plan {
        STANDARD_PDCA_PROTOCOL_STEPS
    } else {
        EMERGENCY_PDCA_PROTOCOL_STEPS
    };
    let effective_max = configured_max_steps.max(semantic_minimum);

    let existing_plan = steps
        .iter()
        .find(|step| step.role == AgentRole::Plan)
        .cloned();
    let existing_check = steps
        .iter()
        .rfind(|step| step.role == AgentRole::Check)
        .cloned();
    let existing_act = steps
        .iter()
        .rfind(|step| step.role == AgentRole::Act)
        .cloned();

    let mut core = Vec::new();
    if requires_plan {
        core.push(
            existing_plan
                .unwrap_or_else(|| generated_gate_step(AgentRole::Plan, "step_kernel_plan")),
        );
    }
    core.extend(steps.into_iter().filter(|step| step.role == AgentRole::Do));
    if !core.iter().any(|step| step.role == AgentRole::Do) {
        core.push(generated_gate_step(AgentRole::Do, "step_kernel_do"));
    }

    let core_capacity = effective_max.saturating_sub(2).max(1);
    core.truncate(core_capacity);

    let retained_ids = core
        .iter()
        .map(|step| step.step_id.clone())
        .collect::<std::collections::HashSet<_>>();
    for index in 0..core.len() {
        core[index]
            .dependencies
            .retain(|dependency| retained_ids.contains(dependency));
        if index > 0 && core[index].dependencies.is_empty() {
            core[index].dependencies = vec![core[index - 1].step_id.clone()];
        }
    }

    let mut check = existing_check
        .unwrap_or_else(|| generated_gate_step(AgentRole::Check, "step_kernel_check"));
    let do_dependencies = core
        .iter()
        .filter(|step| step.role == AgentRole::Do)
        .map(|step| step.step_id.clone())
        .collect::<Vec<_>>();
    check.dependencies = if do_dependencies.is_empty() {
        core.last()
            .map(|step| vec![step.step_id.clone()])
            .unwrap_or_default()
    } else {
        do_dependencies
    };

    let mut act =
        existing_act.unwrap_or_else(|| generated_gate_step(AgentRole::Act, "step_kernel_act"));
    act.dependencies = vec![check.step_id.clone()];
    core.push(check);
    core.push(act);
    core
}

#[cfg(test)]
mod generated_plan_gate_tests {
    use super::*;

    #[test]
    fn model_plan_missing_aa_gets_terminal_ca_and_aa() {
        let steps = vec![
            generated_gate_step(AgentRole::Plan, "pa"),
            generated_gate_step(AgentRole::Do, "da"),
            generated_gate_step(AgentRole::Check, "ca"),
        ];
        let normalized = normalize_generated_pdca_steps(steps, TaskComplexity::Standard, 12);
        assert_eq!(
            normalized.iter().map(|step| step.role).collect::<Vec<_>>(),
            vec![
                AgentRole::Plan,
                AgentRole::Do,
                AgentRole::Check,
                AgentRole::Act
            ]
        );
        assert_eq!(normalized.last().unwrap().dependencies, vec!["ca"]);
    }

    #[test]
    fn step_budget_preserves_terminal_gates_instead_of_prefix_truncating_them() {
        let mut steps = vec![generated_gate_step(AgentRole::Plan, "pa")];
        for index in 0..20 {
            steps.push(generated_gate_step(AgentRole::Do, &format!("da_{index}")));
        }
        steps.push(generated_gate_step(AgentRole::Check, "ca"));
        steps.push(generated_gate_step(AgentRole::Act, "aa"));
        let normalized = normalize_generated_pdca_steps(steps, TaskComplexity::Complex, 6);
        assert_eq!(normalized.len(), 6);
        assert_eq!(normalized[4].role, AgentRole::Check);
        assert_eq!(normalized[5].role, AgentRole::Act);
    }

    #[test]
    fn explicit_simple_plan_is_not_upgraded_to_full_pdca() {
        let steps = vec![generated_gate_step(AgentRole::Do, "da")];
        let normalized = normalize_generated_pdca_steps(steps, TaskComplexity::Simple, 12);
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].role, AgentRole::Do);
    }
}

impl SupervisorAgent {
    pub async fn start_cycle(
        &mut self,
        user_input: &str,
        task_iri: &str,
    ) -> Result<String, CoreError> {
        let cycle_id = format!("cycle_{}", uuid::Uuid::new_v4().hyphenated());

        let perception_result = self
            .perception
            .on_task_start_with_history_limit(
                user_input,
                task_iri,
                self.learning_mode.retrieves_history(),
                self.runner
                    .token_optimization
                    .prompt_optimization
                    .max_learning_hints,
            )
            .await?;
        info!(
            cycle_id = %cycle_id,
            task_iri = %task_iri,
            complexity = %perception_result.complexity,
            risks = ?perception_result.risks,
            "Perception analysis complete"
        );

        let observed_experience_hint_count = perception_result.relevant_experience_hints.len();
        let observed_experience_hint_fingerprints = perception_result
            .relevant_experience_hints
            .iter()
            .map(|hint| {
                use sha2::{Digest, Sha256};
                let digest = Sha256::digest(hint.as_bytes());
                format!("sha256:{}", hex::encode(&digest[..8]))
            })
            .collect();
        let now = chrono::Utc::now();
        let cycle = CycleState {
            cycle_id: cycle_id.clone(),
            task_iri: task_iri.to_string(),
            phase: CyclePhase::Analyzing,
            iteration: 0,
            max_iterations: self.max_iterations,
            started_at: now,
            pdca_started_at: now,
            cycle_deadline_at: now
                + chrono::Duration::seconds(self.perception.cycle_timeout_secs().max(1)),
            last_progress_at: now,
            last_timeout_alert_at: None,
            next_timeout_alert_at: None,
            timeout_alert_count: 0,
            outer_cycle_number: 0,
            phase_history: vec!["Created".to_string()],
            task_completed: false,
            observed_experience_hint_count,
            observed_experience_hint_fingerprints,
            experience_hints: if self.learning_mode.injects_history() {
                perception_result.relevant_experience_hints.clone()
            } else {
                Vec::new()
            },
            intervention: InterventionState::default(),
        };

        self.active_cycles.insert(cycle_id.clone(), cycle);

        info!(cycle_id = %cycle_id, task_iri = %task_iri, input = %user_input, "Cycle started");

        self.event_bus
            .emit(
                task_iri,
                "CYCLE_STARTED",
                "SA",
                &serde_json::json!({
                    "cycle_id": &cycle_id,
                    "user_input": user_input,
                })
                .to_string(),
            )
            .await;

        Ok(cycle_id)
    }

    /// Classify the task and build the matching complexity-based execution plan.
    pub fn analyze_task(&self, user_input: &str) -> ExecutionPlan {
        let complexity = self.classify_complexity(user_input);
        self.build_plan_from_complexity(complexity)
    }

    pub(super) fn build_plan_from_complexity(&self, complexity: TaskComplexity) -> ExecutionPlan {
        let (agent_sequence, parallel_groups, description) = match &complexity {
            TaskComplexity::Instant => (
                vec![AgentRole::Do],
                vec![],
                "Instant query: single DA agent".to_string(),
            ),
            TaskComplexity::Simple => (
                vec![AgentRole::Do],
                vec![],
                "Simple query: single DA agent".to_string(),
            ),
            TaskComplexity::Standard => (
                vec![
                    AgentRole::Plan,
                    AgentRole::Do,
                    AgentRole::Check,
                    AgentRole::Act,
                ],
                vec![],
                "Standard task: PA → DA → CA → AA".to_string(),
            ),
            TaskComplexity::Complex => (
                vec![
                    AgentRole::Plan,
                    AgentRole::Do,
                    AgentRole::Check,
                    AgentRole::Act,
                ],
                vec![],
                "Complex task: PA → DA → CA → AA with full validation".to_string(),
            ),
            TaskComplexity::Exploratory => (
                vec![
                    AgentRole::Plan,
                    AgentRole::Do,
                    AgentRole::Do,
                    AgentRole::Do,
                    AgentRole::Check,
                    AgentRole::Act,
                ],
                vec![vec![AgentRole::Do, AgentRole::Do, AgentRole::Do]],
                "Exploratory: PA → [DA1, DA2, DA3] → CA → AA".to_string(),
            ),
            TaskComplexity::Emergency => (
                vec![AgentRole::Do, AgentRole::Check, AgentRole::Act],
                vec![],
                "Emergency: DA → CA → AA (skip PA)".to_string(),
            ),
            TaskComplexity::Recursive => {
                // Recursive: only 1 round of SA-level PDCA (same as Standard).
                // DA internally executes micro-recursion via execute_recursive_sub_cycle
                // Sub-task decomposition, no SA-level multi-round replay needed.
                let seq = vec![
                    AgentRole::Plan,
                    AgentRole::Do,
                    AgentRole::Check,
                    AgentRole::Act,
                ];
                (
                    seq,
                    vec![],
                    "Recursive: 1 PDCA with DA-internal sub-cycles".to_string(),
                )
            }
        };

        let steps = self.generate_default_steps(&agent_sequence);

        let max_recursion_depth = match &complexity {
            TaskComplexity::Recursive => 3,
            TaskComplexity::Complex => 2,
            _ => 0,
        };

        ExecutionPlan {
            plan_id: format!("plan_{}", uuid::Uuid::new_v4().hyphenated()),
            agent_sequence,
            parallel_groups,
            task_complexity: complexity,
            description,
            steps,
            context_requirements: HashMap::new(),
            success_metrics: vec!["Task completed".to_string()],
            max_recursion_depth,
            sub_tasks: vec![],
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: vec![],
        }
    }

    /// Build resume mode execution plan: standard PDCA sequence
    /// execute_plan will skip completed phases based on resumed_messages
    pub(super) fn build_resume_plan(&self) -> ExecutionPlan {
        let agent_sequence = vec![
            AgentRole::Plan,
            AgentRole::Do,
            AgentRole::Check,
            AgentRole::Act,
        ];
        let steps = self.generate_default_steps(&agent_sequence);
        ExecutionPlan {
            plan_id: format!("plan_resume_{}", uuid::Uuid::new_v4().hyphenated()),
            agent_sequence,
            parallel_groups: vec![],
            task_complexity: TaskComplexity::Standard,
            description: "Resume: continue from checkpoint".to_string(),
            steps,
            context_requirements: HashMap::new(),
            success_metrics: vec!["Task completed".to_string()],
            max_recursion_depth: 0,
            sub_tasks: vec![],
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: vec![],
        }
    }

    fn generate_default_steps(&self, agent_sequence: &[AgentRole]) -> Vec<PlanStep> {
        agent_sequence
            .iter()
            .enumerate()
            .map(|(i, role)| {
                let (objective, expected_output, success_criteria) = match role {
                    AgentRole::Plan => (
                        "Analyze task requirements, create detailed execution plan".to_string(),
                        "JSON-formatted plan with steps, dependencies, resource requirements"
                            .to_string(),
                        "Plan is clear, steps complete, dependencies explicit".to_string(),
                    ),
                    AgentRole::Do => (
                        "Execute the task according to the plan".to_string(),
                        "Execution results, generated files or data".to_string(),
                        "Task completed per plan, output matches expectations".to_string(),
                    ),
                    AgentRole::Check => (
                        "Verify the quality and correctness of execution results".to_string(),
                        "Inspection report with issue list and recommendations".to_string(),
                        "Verification passed or issues identified".to_string(),
                    ),
                    AgentRole::Act => (
                        "Summarize results and make final decision".to_string(),
                        "Final decision and summary report".to_string(),
                        "Decision clear, summary complete".to_string(),
                    ),
                };

                PlanStep {
                    step_id: format!("step_{}", i + 1),
                    role: *role,
                    objective,
                    expected_output,
                    dependencies: if i > 0 {
                        vec![format!("step_{}", i)]
                    } else {
                        vec![]
                    },
                    tools_allowed: vec![],
                    success_criteria,
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: match role {
                        AgentRole::Plan | AgentRole::Check => {
                            crate::core::effect::EffectPolicy::EvidenceOnly
                        }
                        AgentRole::Act => crate::core::effect::EffectPolicy::DecisionOnly,
                        AgentRole::Do => crate::core::effect::EffectPolicy::None,
                    },
                }
            })
            .collect()
    }

    pub(super) async fn extract_5w2h_from_input(
        &self,
        task_iri: &str,
        user_input: &str,
        prefer_exact_contract: bool,
    ) -> crate::core::five_w2h::Task5W2H {
        use crate::core::five_w2h::*;

        // When the caller has already declared that an observable effect is
        // required, the user's text is the authoritative acceptance contract.
        // An additional model call adds latency and can silently paraphrase or
        // invent quantities. PA can still enrich the remaining 5W2H dimensions
        // later through `five_w2h_updates`.
        if prefer_exact_contract {
            let mut w2h = Task5W2H::new(
                user_input,
                "Fulfil the explicit user contract and verify the required effect.",
            );
            w2h.why.success_criteria = vec![user_input.to_string()];
            return w2h;
        }

        if user_input.len() < 20 && !user_input.contains(' ') {
            let mut w2h = Task5W2H::new(user_input, "User task");
            w2h.why.priority = Priority::Low;
            return w2h;
        }

        let prompt = format!(
            r#"Analyze the user task below and extract the 5W2H metadata set.

User task: {}

Output in JSON format (all fields optional except what/why):
{{
  "what": "Core description of the task goal (one sentence)",
  "why_description": "Task intent/value description",
  "success_criteria": ["verifiable condition 1", "condition 2"],
  "priority": "high|medium|low",
  "deadline": "ISO8601 deadline (optional)",
  "estimated_duration": "e.g. 30min, 2h, 3d (optional, guess from task scope)",
  "required_role": "Plan|Do|Check|Act (optional, who should do this)",
  "data_sources": ["file paths or data sources relevant to the task (optional)"],
  "preferred_skills": ["tools or skills likely needed, e.g. file_write, bash (optional)"],
  "token_budget": 100000 (optional, estimated token cost)
}}

Output only JSON, no other content."#,
            user_input
        );

        let model = self.runner.gateway.get_model("default");
        let messages = vec![crate::gateway::unified_gateway::ChatMessage {
            role: "user".to_string(),
            content: prompt,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        match self
            .chat_sa_streaming(
                task_iri,
                "5w2h_extraction",
                &model,
                messages,
                Some(0.3),
                Some(4096),
            )
            .await
        {
            Ok(response) => {
                if let Some(content) = response
                    .choices
                    .first()
                    .and_then(|c| c.message.content.clone())
                {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) {
                        let what = parsed
                            .get("what")
                            .and_then(|v| v.as_str())
                            .unwrap_or(user_input)
                            .to_string();
                        let why_desc = parsed
                            .get("why_description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("User task")
                            .to_string();
                        let success_criteria = parsed
                            .get("success_criteria")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        let priority = match parsed
                            .get("priority")
                            .and_then(|v| v.as_str())
                            .unwrap_or("medium")
                        {
                            "high" => Priority::High,
                            "low" => Priority::Low,
                            _ => Priority::Medium,
                        };

                        let mut w2h = Task5W2H::new(&what, &why_desc);
                        w2h.why.success_criteria = success_criteria;
                        w2h.why.priority = priority;

                        let deadline = parsed
                            .get("deadline")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok());
                        let estimated_duration = parsed
                            .get("estimated_duration")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        if deadline.is_some() || estimated_duration.is_some() {
                            w2h = w2h.with_when(WhenDetail {
                                deadline,
                                start_after: None,
                                estimated_duration,
                                timezone: None,
                                reminder_before: None,
                            });
                        }

                        // ── who ──
                        if let Some(role_str) = parsed.get("required_role").and_then(|v| v.as_str())
                        {
                            w2h = w2h.with_who(WhoDetail {
                                requestor: None,
                                assignees: vec![],
                                stakeholders: vec![],
                                required_role: Some(role_str.to_string()),
                                access_level: None,
                            });
                        }

                        // ── where (data_sources) ──
                        let data_sources: Vec<String> = parsed
                            .get("data_sources")
                            .and_then(|v| v.as_array())
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        if !data_sources.is_empty() {
                            w2h = w2h.with_where(WhereDetail {
                                data_sources,
                                execution_environment: None,
                                target_repository: None,
                                target_branch: None,
                            });
                        }

                        // ── how (preferred_skills) ──
                        let preferred_skills: Vec<String> = parsed
                            .get("preferred_skills")
                            .and_then(|v| v.as_array())
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        if !preferred_skills.is_empty() {
                            w2h = w2h.with_how(HowDetail {
                                plan_iri: None,
                                preferred_skills,
                                forbidden_tools: vec![],
                                required_steps: None,
                                dependencies: vec![],
                            });
                        }

                        // ── how_much (token_budget) ──
                        if user_explicitly_requested_token_budget(user_input) {
                            if let Some(budget) =
                                parsed.get("token_budget").and_then(|v| v.as_u64())
                            {
                                w2h = w2h.with_how_much(HowMuchDetail {
                                    token_budget: Some(budget),
                                    max_sub_agents: None,
                                    max_pdca_cycles: None,
                                    expected_quality: None,
                                    actual_cost: None,
                                });
                            }
                        }

                        return w2h;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("5W2H extraction failed: {}, using defaults", e);
            }
        }

        Task5W2H::new(user_input, "User task")
    }

    pub async fn analyze_task_with_llm(
        &self,
        task_iri: &str,
        user_input: &str,
        five_w2h: &crate::core::five_w2h::Task5W2H,
        experience_hints: &[String],
        task_constraints: &HashMap<String, String>,
    ) -> ExecutionPlan {
        // Keyword verdict takes precedence over the priority mapping for structural
        // complexity (Recursive/Exploratory/Emergency); the LLM refines the plan.
        let keyword_complexity = self.classify_complexity(user_input);

        let delivery_contract =
            crate::core::agent_runner::direct_response_delivery_contract(task_constraints)
                .map(|contract| format!("\n\n## Delivery Contract\n{contract}"))
                .unwrap_or_default();
        let capability_contract =
            crate::core::agent_runner::required_capability_contract(task_constraints)
                .map(|contract| format!("\n\n## Evidence Capability Contract\n{contract}"))
                .unwrap_or_default();
        let enhanced_input = if experience_hints.is_empty() {
            format!("{user_input}{delivery_contract}{capability_contract}")
        } else {
            format!(
                "## Historical Experience Reference\n{}\n\n## Current Task\n{}{}{}",
                experience_hints
                    .iter()
                    .map(|h| format!("- {}", h))
                    .collect::<Vec<_>>()
                    .join("\n"),
                user_input,
                delivery_contract,
                capability_contract,
            )
        };

        let complexity = match keyword_complexity {
            TaskComplexity::Recursive | TaskComplexity::Exploratory | TaskComplexity::Emergency => {
                keyword_complexity
            }
            _ => match five_w2h.why.priority {
                crate::core::five_w2h::Priority::High => TaskComplexity::Complex,
                crate::core::five_w2h::Priority::Medium => TaskComplexity::Standard,
                crate::core::five_w2h::Priority::Low => TaskComplexity::Simple,
            },
        };

        match self
            .generate_detailed_plan_with_llm(task_iri, &enhanced_input, five_w2h)
            .await
        {
            Ok(mut plan) => {
                info!(plan_id = %plan.plan_id, steps = plan.steps.len(), "LLM generated detailed plan successfully");
                // Keyword verdict wins for Recursive/Complex: force the LLM
                // plan's complexity when it disagrees.
                let forced = match keyword_complexity {
                    TaskComplexity::Complex | TaskComplexity::Recursive => keyword_complexity,
                    _ => plan.task_complexity,
                };
                if forced != plan.task_complexity {
                    plan.task_complexity = forced;
                    plan.max_recursion_depth = match forced {
                        TaskComplexity::Recursive => 3,
                        TaskComplexity::Complex => 2,
                        _ => 0,
                    };
                }
                return plan;
            }
            Err(e) => {
                warn!(
                    "LLM failed to generate detailed plan: {}, using default plan",
                    e
                );
            }
        }

        self.build_plan_from_complexity(complexity)
    }

    async fn generate_detailed_plan_with_llm(
        &self,
        task_iri: &str,
        user_input: &str,
        five_w2h: &crate::core::five_w2h::Task5W2H,
    ) -> Result<ExecutionPlan, CoreError> {
        let mut w2h_section = String::new();

        if let Some(ref who) = five_w2h.who {
            if let Some(ref role) = who.required_role {
                w2h_section.push_str(&format!("\n- Required Role: {}", role));
            }
        }

        if let Some(ref when) = five_w2h.when {
            if let Some(ref deadline) = when.deadline {
                w2h_section.push_str(&format!("\n- Deadline: {}", deadline.to_rfc3339()));
            }
            if let Some(ref dur) = when.estimated_duration {
                w2h_section.push_str(&format!("\n- Estimated Duration: {}", dur));
            }
        }

        if let Some(ref where_) = five_w2h.where_ {
            if !where_.data_sources.is_empty() {
                w2h_section.push_str(&format!(
                    "\n- Data Sources: {}",
                    where_.data_sources.join(", ")
                ));
            }
            if let Some(ref env) = where_.execution_environment {
                w2h_section.push_str(&format!("\n- Execution Environment: {}", env));
            }
        }

        if let Some(ref how) = five_w2h.how {
            if !how.preferred_skills.is_empty() {
                w2h_section.push_str(&format!(
                    "\n- Preferred Skills: {}",
                    how.preferred_skills.join(", ")
                ));
            }
            if !how.forbidden_tools.is_empty() {
                w2h_section.push_str(&format!(
                    "\n- Forbidden Tools: {}",
                    how.forbidden_tools.join(", ")
                ));
            }
        }

        if let Some(ref how_much) = five_w2h.how_much {
            if let Some(budget) = how_much.token_budget {
                w2h_section.push_str(&format!("\n- Token Budget: {}", budget));
            }
            if let Some(cycles) = how_much.max_pdca_cycles {
                w2h_section.push_str(&format!("\n- Max PDCA Cycles: {}", cycles));
            }
        }

        if !five_w2h.why.success_criteria.is_empty() {
            w2h_section.push_str(&format!(
                "\n- Success Criteria: {}",
                five_w2h.why.success_criteria.join(", ")
            ));
        }

        w2h_section.push_str(&format!("\n- Priority: {:?}", five_w2h.why.priority));

        let w2h_block = if w2h_section.is_empty() {
            String::new()
        } else {
            format!("\n## 5W2H Constraint Info{}", w2h_section)
        };

        let sa_constitution_prompt = {
            use crate::core::constitution::{ConstitutionRegistry, ConstitutionRole};
            let registry = ConstitutionRegistry::new();
            let constitution_text = registry.build_prompt_for_role(ConstitutionRole::Supervisor);
            // Inject methodology layer discipline (includes auto-trigger protocol, always-active methodology)
            let methodology_text =
                crate::methodology::integration::MethodologyPromptInjector::build_for_sa();
            format!("{}\n{}", constitution_text, methodology_text)
        };

        let prompt = format!(
            r#"You are a task planning expert. Analyze the following task and generate a concise and efficient execution plan.

## Task Description
{}{}
## Output Requirements
Output the plan in JSON format with the following fields:

```json
{{
  "complexity": "simple|standard|complex|exploratory|emergency",
  "description": "Task description",
  "steps": [
    {{
      "step_id": "step_1",
      "role": "Plan|Do|Check|Act",
      "objective": "Specific goal of this step",
      "expected_output": "Expected output",
      "dependencies": [],
      "tools_allowed": ["file_read", "file_write", "grep_search", "glob_search", "web_search", "web_fetch", "bash"],
      "success_criteria": "Success criteria"
    }}
  ],
  "success_metrics": ["Success metric 1", "Success metric 2"]
}}
```

## Role Descriptions
- **Plan (PA)**: Analyze tasks, create plans, decompose sub-tasks
- **Do (DA)**: Execute specific tasks, create artifacts (one DA step should complete multiple related operations)
- **Check (CA)**: Verify results, check quality
- **Act (AA)**: Summarize decisions, final summary

## Complexity Definitions
- **simple**: Simple query, single step (DA only)
- **standard**: Standard task, requires PA→DA→CA→AA flow
- **complex**: Complex task, requires full PA→DA→CA→AA validation, DA internally triggers sub-cycle optimization
- **exploratory**: Exploratory task, requires multiple parallel DAs
- **emergency**: Emergency fix, skip PA, DA→CA→AA

## Important Constraints
1. **Step count limit**: Total steps not to exceed {} (including PA and CA/AA)
2. **DA step merging**: Merge multiple related operations into one DA step. For example, creating multiple files should be done in one DA step, not one step per file
3. **Recommended pattern**: PA(1step) → DA(1-3steps) → CA(1step) → AA(1step)
4. Each DA step's objective should describe a group of related operations to complete, not a single atomic operation

## Code of Conduct
As the Supervisor Agent, you must follow these guidelines:

{}

Output only JSON, no other content."#,
            user_input,
            w2h_block,
            sa_constitution_prompt,
            self.runner.agent_settings.execution_budget.max_plan_steps
        );

        let model = self.runner.gateway.get_model("default");
        let messages = vec![crate::gateway::unified_gateway::ChatMessage {
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
                "plan_generation",
                &model,
                messages,
                Some(0.3),
                Some(8192),
            )
            .await?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| CoreError::Internal {
                message: "No response content".to_string(),
            })?;

        self.parse_llm_plan(&content)
    }

    pub(super) fn parse_llm_plan(&self, content: &str) -> Result<ExecutionPlan, CoreError> {
        let trimmed = content.trim();
        let json_str = if trimmed.starts_with('{') {
            trimmed.to_string()
        } else if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                trimmed[start..=end].to_string()
            } else {
                trimmed.to_string()
            }
        } else {
            return Err(CoreError::Internal {
                message: "No JSON found in LLM plan response".to_string(),
            });
        };

        #[derive(Deserialize)]
        struct LlmPlanResponse {
            complexity: String,
            description: String,
            steps: Vec<LlmPlanStep>,
            success_metrics: Vec<String>,
        }

        #[derive(Deserialize)]
        struct LlmPlanStep {
            step_id: String,
            role: String,
            objective: String,
            expected_output: String,
            dependencies: Vec<String>,
            success_criteria: String,
        }

        let parsed: LlmPlanResponse =
            parse_or_repair_json(&json_str).map_err(|e| CoreError::Internal {
                message: format!("JSON parse error after repair attempt: {}", e),
            })?;

        let complexity = match parsed.complexity.as_str() {
            "simple" => TaskComplexity::Simple,
            "complex" => TaskComplexity::Complex,
            "recursive" => TaskComplexity::Recursive,
            "exploratory" => TaskComplexity::Exploratory,
            "emergency" => TaskComplexity::Emergency,
            _ => TaskComplexity::Standard,
        };

        let steps: Vec<PlanStep> = parsed
            .steps
            .into_iter()
            .map(|s| {
                let role = match s.role.as_str() {
                    "Plan" => AgentRole::Plan,
                    "Do" => AgentRole::Do,
                    "Check" => AgentRole::Check,
                    "Act" => AgentRole::Act,
                    _ => AgentRole::Do,
                };
                PlanStep {
                    step_id: s.step_id,
                    role,
                    objective: s.objective,
                    expected_output: s.expected_output,
                    dependencies: s.dependencies,
                    tools_allowed: vec![],
                    success_criteria: s.success_criteria,
                    branch_on_failure: false,
                    branch_fallback: None,
                    retry_count: 0,
                    retry_delay_secs: 0,
                    effect_policy: match role {
                        AgentRole::Plan | AgentRole::Check => {
                            crate::core::effect::EffectPolicy::EvidenceOnly
                        }
                        AgentRole::Act => crate::core::effect::EffectPolicy::DecisionOnly,
                        AgentRole::Do => crate::core::effect::EffectPolicy::None,
                    },
                }
            })
            .collect();

        let max_plan_steps = self.runner.agent_settings.execution_budget.max_plan_steps;
        let original_step_count = steps.len();
        let steps = normalize_generated_pdca_steps(steps, complexity, max_plan_steps);
        if original_step_count > max_plan_steps || steps.len() != original_step_count {
            warn!(
                original_steps = original_step_count,
                normalized_steps = steps.len(),
                complexity = ?complexity,
                "Model-generated PDCA plan normalized to preserve required BizAgent gates (configured max {})",
                max_plan_steps,
            );
        }

        let agent_sequence: Vec<AgentRole> = steps.iter().map(|s| s.role).collect();

        // Exploratory plans express parallelism as multiple Do steps; expose
        // them as a parallel group so execute_plan dispatches them concurrently.
        let parallel_groups = if complexity == TaskComplexity::Exploratory {
            let do_count = steps.iter().filter(|s| s.role == AgentRole::Do).count();
            if do_count > 1 {
                vec![vec![AgentRole::Do; do_count]]
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let max_recursion_depth = match &complexity {
            TaskComplexity::Recursive => 3,
            TaskComplexity::Complex => 2,
            _ => 0,
        };

        Ok(ExecutionPlan {
            plan_id: format!("plan_{}", uuid::Uuid::new_v4().hyphenated()),
            agent_sequence,
            parallel_groups,
            task_complexity: complexity,
            description: parsed.description,
            steps,
            context_requirements: HashMap::new(),
            success_metrics: parsed.success_metrics,
            max_recursion_depth,
            sub_tasks: vec![],
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: vec![],
        })
    }

    #[allow(dead_code)]
    async fn classify_with_llm(&self, user_input: &str) -> Result<TaskComplexity, CoreError> {
        let prompt = format!(
            r#"Analyze the complexity of the following task, return JSON format result.

Task: {}

Analyze the task:
1. Does it require multi-step execution?
2. Does it require a planning phase?
3. Does it require a verification phase?
4. Does it require multiple parallel explorations?

Return JSON:
{{"complexity": "simple|standard|complex|exploratory|emergency", "reason": "Brief reason"}}

Complexity definitions:
- simple: Simple query, single step
- standard: Standard task, requires plan→execute→check→decide flow
- complex: Complex task, requires multi-step execution and verification
- exploratory: Exploratory task, requires multiple parallel explorations
- emergency: Emergency fix task, skip planning and execute directly"#,
            user_input
        );

        let model = self.runner.gateway.get_model("default");
        let messages = vec![crate::gateway::unified_gateway::ChatMessage {
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
            .chat_with_params(&model, messages, Some(0.3), Some(4096), None, None)
            .await?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        // Parse LLM response
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(complexity_str) = parsed.get("complexity").and_then(|c| c.as_str()) {
                let complexity = match complexity_str {
                    "simple" => TaskComplexity::Simple,
                    "complex" => TaskComplexity::Complex,
                    "exploratory" => TaskComplexity::Exploratory,
                    "emergency" => TaskComplexity::Emergency,
                    _ => TaskComplexity::Standard,
                };
                info!(complexity = ?complexity, reason = ?parsed.get("reason"), "LLM classification result");
                return Ok(complexity);
            }
        }

        // Try to extract from text
        let lower = content.to_lowercase();
        if lower.contains("simple") {
            return Ok(TaskComplexity::Simple);
        } else if lower.contains("exploratory") {
            return Ok(TaskComplexity::Exploratory);
        } else if lower.contains("emergency") {
            return Ok(TaskComplexity::Emergency);
        }

        Err(CoreError::Internal {
            message: "Failed to parse LLM classification".to_string(),
        })
    }

    pub(super) fn classify_complexity(&self, user_input: &str) -> TaskComplexity {
        let lower = user_input.to_lowercase();

        // Instant: very short input (e.g., greetings)
        if user_input.len() < 15 && !user_input.contains(' ') {
            return TaskComplexity::Instant;
        }

        // Emergency: emergency fix category. Strong verbs (fix/bug/urgent/repair)
        // always qualify; weak symptoms (error/crash/broken/fault) only qualify
        // when reinforced by urgency cues or a short input — long pasted logs
        // containing "error" must not be misclassified as emergency.
        let emergency_strong = ["fix", "bug", "urgent", "repair"];
        let emergency_weak = ["error", "crash", "broken", "fault"];
        let emergency_reinforced = [
            "critical",
            "security",
            "production",
            "immediately",
            "outage",
        ];
        let has_strong = emergency_strong.iter().any(|k| lower.contains(k));
        let has_weak = emergency_weak.iter().any(|k| lower.contains(k));
        let reinforced =
            user_input.len() < 200 || emergency_reinforced.iter().any(|k| lower.contains(k));
        if has_strong || (has_weak && reinforced) {
            return TaskComplexity::Emergency;
        }

        // Recursive decomposition: complex multi-step tasks, requires DA internal micro PDCA sub-cycles
        let recursive_keywords = [
            "refactor",
            "rewrite",
            "migrate",
            "split into",
            "decompose",
            "multi-phase",
            "end-to-end",
            "full-stack",
            // Project building category
            "develop",
            "create",
            "implement",
            "building",
            "build",
            "program",
            "project",
            "app",
            "website",
            "system",
            "platform",
            "generate",
        ];
        if recursive_keywords.iter().any(|k| lower.contains(k)) {
            return TaskComplexity::Recursive;
        }

        // Exploratory task → Exploratory (prioritized over research_patterns)
        let exploratory_keywords = ["explore", "investigate"];
        if exploratory_keywords.iter().any(|k| lower.contains(k)) {
            return TaskComplexity::Exploratory;
        }

        let compare_keywords = ["compare"];
        if compare_keywords.iter().any(|k| lower.contains(k)) {
            let multi_patterns = ["different", "various", "multiple", "several"];
            if multi_patterns.iter().any(|p| lower.contains(p)) {
                return TaskComplexity::Exploratory;
            }
            return TaskComplexity::Complex;
        }

        // Research/analysis questions → Standard or Complex
        let research_patterns = ["research", "analysis", "survey", "study"];
        if research_patterns.iter().any(|p| lower.contains(p)) {
            let deep_patterns = [
                "deep",
                "thorough",
                "comprehensive",
                "systematic",
                "in-depth",
            ];
            if deep_patterns.iter().any(|p| lower.contains(p)) {
                return TaskComplexity::Complex;
            }
            return TaskComplexity::Standard;
        }

        // Simple: simple fact query, answerable in one sentence
        let simple_query_patterns = ["what is", "who is", "how to", "explain", "define"];
        let is_simple_query = user_input.len() < 50
            && simple_query_patterns.iter().any(|p| lower.contains(p))
            && !lower.contains("application")
            && !lower.contains("scenario")
            && !lower.contains("analysis")
            && !lower.contains("implementation")
            && !lower.contains("design");

        if is_simple_query {
            return TaskComplexity::Simple;
        }

        // English simple query
        if user_input.len() < 50
            && (lower.starts_with("what is")
                || lower.starts_with("who is")
                || lower.starts_with("where is")
                || lower.starts_with("when is"))
        {
            return TaskComplexity::Simple;
        }

        // Default: Standard
        TaskComplexity::Standard
    }
}
