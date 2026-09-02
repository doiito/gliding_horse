use sha2::{Digest, Sha256};
use tracing::{info, instrument, warn};

use crate::core::agent_instance::AgentRole;
use crate::core::agent_runner::{TaskContext, TaskResult};
use crate::core::policy_learning::{
    learning_families_compatible, learning_task_context, LearningTaskContext,
};
use crate::CoreError;

use super::agent::SupervisorAgent;
use super::types::*;

#[derive(Debug, Clone, serde::Serialize)]
struct PolicyRewardBreakdown {
    status_reward: f32,
    failed_action_penalty: f32,
    excess_turn_penalty: f32,
    recovery_penalty: f32,
    error_penalty: f32,
    prompt_token_penalty: f32,
    latency_penalty: f32,
    late_substantive_action_penalty: f32,
    redundant_read_ratio: f32,
    redundant_read_penalty: f32,
    no_effect_tail: usize,
    no_effect_tail_penalty: f32,
    first_substantive_action_ordinal: Option<usize>,
    total: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct LearningTreatmentMetrics {
    perception_hints_observed: usize,
    experience_hint_fingerprints: Vec<String>,
    skills_observed: usize,
    skill_iris_observed: Vec<String>,
    knowledge_fragments_observed: usize,
    knowledge_fragment_iris_observed: Vec<String>,
    hints_injected: usize,
    hint_chars_injected: usize,
    task_family_raw_features: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experiment_pair_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experiment_seed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experiment_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experiment_config_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_fingerprint: Option<String>,
    objective_fingerprint: String,
    orchestration_mode: String,
}

/// Deduplicate hints preserving first-seen order, then truncate to `cap`.
fn dedup_hints(
    hints: Vec<String>,
    cap: usize,
    max_hint_chars: usize,
    max_total_chars: usize,
) -> Vec<String> {
    let mut seen = std::collections::HashSet::with_capacity(cap.min(hints.len()));
    let mut out = Vec::with_capacity(cap.min(hints.len()));
    let mut total_chars = 0usize;
    for h in hints {
        if out.len() >= cap || total_chars >= max_total_chars {
            break;
        }
        let mut bounded = h.chars().take(max_hint_chars).collect::<String>();
        if h.chars().count() > max_hint_chars {
            bounded.push_str("…");
        }
        let remaining = max_total_chars.saturating_sub(total_chars);
        bounded = bounded.chars().take(remaining).collect();
        if !bounded.is_empty() && seen.insert(bounded.clone()) {
            total_chars += bounded.chars().count();
            out.push(bounded);
        }
    }
    out
}

fn rank_knowledge_fragments(
    fragments: Vec<crate::skill_graph::types::KnowledgeFragment>,
    task_context: &LearningTaskContext,
    max_fragments: usize,
) -> Vec<crate::skill_graph::types::KnowledgeFragment> {
    let current_features = task_context
        .raw_features
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let applicability = |fragment: &crate::skill_graph::types::KnowledgeFragment| {
        let Some(stored_family) = fragment.task_family.as_deref() else {
            return true;
        };
        if stored_family == task_context.family {
            return true;
        }
        if !learning_families_compatible(stored_family, &task_context.family) {
            return false;
        }
        // A coarse policy family is safe for choosing an evidence ordering,
        // but task-specific procedures need lexical applicability as a second
        // gate. Knowledge descriptions retain the original audited objective.
        let historical = learning_task_context(&fragment.description);
        let overlap = historical
            .raw_features
            .iter()
            .filter(|feature| current_features.contains(feature.as_str()))
            .count();
        let denominator = historical
            .raw_features
            .len()
            .min(task_context.raw_features.len())
            .max(1);
        overlap >= 2 && overlap * 5 >= denominator
    };
    let mut fragments = fragments
        .into_iter()
        // A structured fragment is safe only inside its declared family.
        // Legacy failure fragments have no family and remain available at a
        // lower rank for backward compatibility.
        .filter(applicability)
        .collect::<Vec<_>>();
    fragments.sort_by(|left, right| {
        let score = |fragment: &crate::skill_graph::types::KnowledgeFragment| {
            let exact_family =
                u32::from(fragment.task_family.as_deref() == Some(task_context.family.as_str()));
            let passed = u32::from(fragment.ca_verdict.as_deref() == Some("pass"));
            let evidence = fragment.evidence_count.min(100);
            (exact_family, passed, evidence, fragment.last_verified_at)
        };
        score(right).cmp(&score(left))
    });
    fragments.truncate(max_fragments);
    fragments
}

fn render_knowledge_hint(fragment: &crate::skill_graph::types::KnowledgeFragment) -> String {
    if fragment.kind != "ca_validated_task_knowledge" {
        return format!(
            "[Knowledge source={} kind=legacy_failure] problem={} mitigation={}",
            fragment.fragment_iri, fragment.problem, fragment.recommendation
        );
    }
    format!(
        "[Knowledge source={} task={} family={} ca={} evidence={}/{}] applicability={}; procedure={}; checks={}; boundary={}",
        fragment.fragment_iri,
        fragment.source_task_iri.as_deref().unwrap_or("unknown"),
        fragment.task_family.as_deref().unwrap_or("unknown"),
        fragment.ca_verdict.as_deref().unwrap_or("unknown"),
        fragment.success_count,
        fragment.evidence_count,
        fragment.problem,
        if fragment.procedure.is_empty() { "not recorded".to_string() } else { fragment.procedure.join(" | ") },
        if fragment.successful_checks.is_empty() { "not recorded".to_string() } else { fragment.successful_checks.join(" | ") },
        if fragment.counterexamples.is_empty() { "none recorded".to_string() } else { fragment.counterexamples.join(" | ") },
    )
}

fn eligible_policy_candidates(
    skill_count: usize,
    knowledge_count: usize,
    experience_count: usize,
) -> Vec<String> {
    crate::core::retrieval_policy::RetrievalPolicyArm::candidate_names(
        skill_count,
        knowledge_count,
        experience_count,
    )
}

/// Materialize the treatment selected by the constrained policy. `baseline`
/// is a real ablation (no durable history), while each learned arm receives
/// the same bounded evidence in a different source-priority order.
fn policy_treatment_hints(
    action: &str,
    experience: &[String],
    skills: &[String],
    knowledge: &[String],
    max_hints: usize,
    max_hint_chars: usize,
    max_total_chars: usize,
) -> Vec<String> {
    let arm = crate::core::retrieval_policy::RetrievalPolicyArm::parse(action)
        .unwrap_or(crate::core::retrieval_policy::RetrievalPolicyArm::Baseline);
    let hints = arm.order_hints(experience, skills, knowledge);
    dedup_hints(hints, max_hints, max_hint_chars, max_total_chars)
}

/// Existing-workspace tasks can often be accepted by CA→AA without any
/// implementation.  Detailed fallback planning is intentionally deferred
/// until that verification fails, avoiding a full planning LLM call whose
/// result would otherwise never be executed.
fn should_defer_fallback_planning(ctx: &TaskContext, complexity: TaskComplexity) -> bool {
    ctx.workspace_file_summary.is_some()
        && ctx.resumed_messages.is_none()
        && ctx.workflow_jsonld.is_none()
        && !ctx.effective_effect_policy().requires_workspace_mutation()
        && !matches!(complexity, TaskComplexity::Instant | TaskComplexity::Simple)
}

/// Build a PDCA delta plan beginning at the failed step. Completed upstream
/// nodes are retained as evidence in cycle feedback instead of being executed
/// again. External DAG workflows keep their own retry/branch topology and are
/// therefore never rewritten here.
pub(super) fn scoped_recovery_plan(
    plan: &ExecutionPlan,
    failed_step_id: &str,
    revision: u32,
) -> Option<ExecutionPlan> {
    if plan.dag_jsonld.is_some() {
        return None;
    }
    let failed_index = plan
        .steps
        .iter()
        .position(|step| step.step_id == failed_step_id)?;
    let mut scoped = plan.clone();
    scoped.plan_id = format!("{}_delta_{}", plan.plan_id, revision);
    scoped.steps = plan.steps[failed_index..].to_vec();
    let retained = scoped
        .steps
        .iter()
        .map(|step| step.step_id.clone())
        .collect::<std::collections::HashSet<_>>();
    for step in &mut scoped.steps {
        step.dependencies
            .retain(|dependency| retained.contains(dependency));
    }
    scoped.agent_sequence = scoped.steps.iter().map(|step| step.role).collect();
    scoped.parallel_groups.clear();
    scoped.verify_first = false;
    scoped.fallback_steps.clear();
    scoped.description = format!(
        "Scoped recovery from failed step {}; completed predecessors preserved",
        failed_step_id
    );
    Some(scoped)
}

fn recovery_failed_step(summary: &str) -> Option<&str> {
    let marker = "failed_step=";
    let start = summary.find(marker)? + marker.len();
    let tail = &summary[start..];
    Some(
        tail.split(|character: char| character.is_whitespace())
            .next()
            .unwrap_or(tail),
    )
}

fn policy_reward_breakdown(
    result: &TaskResult,
    prompt_tokens: u64,
    elapsed_ms: u64,
    workspace_mutation_required: bool,
) -> PolicyRewardBreakdown {
    let status_reward = match result.status.as_str() {
        "success" | "completed" if !result.summary.contains("[Recovery] scope=Task") => 1.0,
        "partial_success" => 0.25,
        "failed" | "timeout" => -1.0,
        _ => -0.5,
    };
    let failed_actions = result
        .tracked_actions
        .iter()
        .filter(|action| {
            matches!(
                action.status,
                crate::core::tracked_action::ActionStatus::Failed
                    | crate::core::tracked_action::ActionStatus::Retried
            )
        })
        .count() as f32;
    let failed_action_penalty = if result.tracked_actions.is_empty() {
        0.0
    } else {
        (failed_actions / result.tracked_actions.len() as f32 * 0.35).min(0.35)
    };
    // A tool call normally needs one reasoning turn. Penalize only turns that
    // exceed that work plus a small PA/CA/AA coordination allowance.
    let efficient_turn_ceiling = result.tool_call_count.saturating_add(4);
    let excess_turns = result.turn_count.saturating_sub(efficient_turn_ceiling);
    let excess_turn_penalty = (excess_turns as f32 * 0.025).min(0.2);
    let recovery_penalty = if result.summary.contains("[Recovery] scope=Task") {
        0.15
    } else {
        0.0
    };
    let error_penalty = (result.errors.len() as f32 * 0.04).min(0.16);
    let prompt_token_penalty = if prompt_tokens > 120_000 {
        (((prompt_tokens - 120_000) as f32 / 40_000.0) * 0.03).min(0.15)
    } else {
        0.0
    };
    let latency_penalty = if elapsed_ms > 120_000 {
        (((elapsed_ms - 120_000) as f32 / 60_000.0) * 0.03).min(0.12)
    } else {
        0.0
    };
    let first_substantive_action_ordinal = result
        .tracked_actions
        .iter()
        .position(|action| action.substantive_effect)
        .map(|index| index + 1);
    let late_substantive_action_penalty = first_substantive_action_ordinal
        .map(|ordinal| (ordinal.saturating_sub(8) as f32 * 0.02).min(0.1))
        .unwrap_or(0.0);
    let mut evidence_keys = std::collections::HashSet::new();
    let mut evidence_count = 0usize;
    let mut duplicate_evidence_count = 0usize;
    for action in &result.tracked_actions {
        if matches!(
            action.tool_name.as_str(),
            "file_read" | "file_list" | "glob_search" | "grep_search" | "workspace_status"
        ) {
            evidence_count += 1;
            let key = format!(
                "{}:{}",
                action.tool_name,
                serde_json::to_string(&action.tool_args).unwrap_or_default()
            );
            if !evidence_keys.insert(key) {
                duplicate_evidence_count += 1;
            }
        }
    }
    let redundant_read_ratio = if evidence_count == 0 {
        0.0
    } else {
        duplicate_evidence_count as f32 / evidence_count as f32
    };
    let redundant_read_penalty = (redundant_read_ratio * 0.16).min(0.16);
    let no_effect_tail = result
        .tracked_actions
        .iter()
        .rev()
        .take_while(|action| !action.substantive_effect)
        .count();
    let no_effect_tail_penalty =
        if !workspace_mutation_required || result.tracked_actions.is_empty() {
            0.0
        } else {
            (no_effect_tail as f32 / result.tracked_actions.len() as f32 * 0.12).min(0.12)
        };
    let total = (status_reward
        - failed_action_penalty
        - excess_turn_penalty
        - recovery_penalty
        - error_penalty
        - prompt_token_penalty
        - latency_penalty
        - late_substantive_action_penalty
        - redundant_read_penalty
        - no_effect_tail_penalty)
        .clamp(-1.0, 1.0);
    PolicyRewardBreakdown {
        status_reward,
        failed_action_penalty,
        excess_turn_penalty,
        recovery_penalty,
        error_penalty,
        prompt_token_penalty,
        latency_penalty,
        late_substantive_action_penalty,
        first_substantive_action_ordinal,
        redundant_read_ratio,
        redundant_read_penalty,
        no_effect_tail,
        no_effect_tail_penalty,
        total,
    }
}

/// The CA/AA evidence writer uses this same deterministic task key. Keeping
/// the derivation here avoids putting execution payloads into learning records
/// merely to discover whether independent verification was completed.
fn task_audit_evidence_iri(task_iri: &str) -> String {
    format!(
        "{}{}",
        crate::core::policy_learning::AUDIT_EVIDENCE_PREFIX,
        hex::encode(Sha256::digest(task_iri.as_bytes()))
    )
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
        let task_started_at = std::time::Instant::now();

        // AgentRunner counters are intentionally process-wide so the UI can
        // display cumulative usage. A task budget, however, is task-scoped;
        // never charge this task for tokens consumed by an earlier task on
        // the same SupervisorAgent instance.
        let task_prompt_tokens_start = self
            .runner
            .total_prompt_tokens
            .load(std::sync::atomic::Ordering::Relaxed);
        let task_completion_tokens_start = self
            .runner
            .total_completion_tokens
            .load(std::sync::atomic::Ordering::Relaxed);

        let declared_effect_execution = ctx
            .constraints
            .get("required_effect")
            .is_some_and(|value| value == "workspace_mutation");
        let extraction_started_at = std::time::Instant::now();
        let mut five_w2h = self
            .extract_5w2h_from_input(task_iri, user_input, declared_effect_execution)
            .await;
        tracing::info!(
            task_iri = %task_iri,
            elapsed_ms = extraction_started_at.elapsed().as_millis() as u64,
            "SA 5W2H extraction completed"
        );
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
            let what_sha256 = format!(
                "sha256:{}",
                hex::encode(&Sha256::digest(five_w2h.what.as_bytes())[..12])
            );
            tracing::info!(
                task_iri = %task_iri,
                what_chars = five_w2h.what.chars().count(),
                %what_sha256,
                "5W2H initialization complete"
            );
        }

        // start_cycle already performed perception once. Reusing its result
        // avoids a duplicate semantic lookup and gives baseline/shadow/active
        // one unambiguous treatment assignment for the whole task.
        let perception_hints = self
            .active_cycles
            .get(&cycle_id)
            .map(|cycle| cycle.experience_hints.clone())
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, hint)| format!("[Experience source=perception:{}] {}", index + 1, hint))
            .collect::<Vec<_>>();
        let perceived_hint_count = self
            .active_cycles
            .get(&cycle_id)
            .map(|cycle| cycle.observed_experience_hint_count)
            .unwrap_or(perception_hints.len());
        // Preserve the exact experience treatment before skill and knowledge
        // enrichment so the constrained policy can choose a genuinely
        // different, measurable ordering arm.
        // Enrich with skill discovery results if a discovery engine is available.
        // Keep sources separate until after policy selection so every action
        // corresponds to the treatment recorded in the learning evaluation.
        let mut skill_hints = Vec::new();
        let mut knowledge_hints = Vec::new();
        // The original objective is deterministic for matched tasks. Using
        // model-generated 5W2H text here fragments identical observations
        // because the extraction paraphrases `what` on every call.
        let task_context = learning_task_context(user_input);
        let policy_context = task_context.family.clone();
        let mut discovered_skill_count = 0usize;
        let mut discovered_knowledge_count = 0usize;
        let mut observed_skill_iris = Vec::new();
        let mut observed_knowledge_fragment_iris = Vec::new();
        let prompt_settings = &self.runner.token_optimization.prompt_optimization;
        if self.learning_mode.retrieves_history() {
            if let Some(ref de) = self.discovery_engine {
                let mut discovery_constraints = five_w2h.why.success_criteria.clone();
                discovery_constraints.extend(
                    ctx.constraints
                        .iter()
                        .map(|(key, value)| format!("{key}={value}")),
                );
                let disc_task = crate::skill_graph::discovery::Task5W2H {
                    what: user_input.to_string(),
                    why: five_w2h.why.description.clone(),
                    who: five_w2h.who.as_ref().and_then(|w| w.required_role.clone()),
                    when_phase: five_w2h.when.as_ref().map(|w| format!("{:?}", w)),
                    where_context: five_w2h.where_.as_ref().map(|w| format!("{:?}", w)),
                    how_approach: five_w2h.how.as_ref().and_then(|h| h.required_steps.clone()),
                    constraints: discovery_constraints,
                };
                let matches = de.discover_for_task(&disc_task).await;
                skill_hints = matches
                    .iter()
                    .filter_map(|m| {
                        let name = if !m.skill.name.is_empty() {
                            m.skill.name.clone()
                        } else {
                            m.skill.skill_iri.rsplit('/').next()?.to_string()
                        };
                        Some(format!(
                            "[Skill source={} relevance={:.2}] {}",
                            m.skill.skill_iri, m.relevance_score, name
                        ))
                    })
                    .take(prompt_settings.max_discovered_skill_hints)
                    .collect();
                observed_skill_iris = matches
                    .iter()
                    .take(prompt_settings.max_discovered_skill_hints)
                    .map(|matched| matched.skill.skill_iri.clone())
                    .collect();
                knowledge_hints = if let Some(graph) = self.runner.skill_graph_store.as_ref() {
                    let fragments = rank_knowledge_fragments(
                        matches
                            .iter()
                            .flat_map(|m| graph.get_fragments_for_skill(&m.skill.skill_iri))
                            .collect::<Vec<_>>(),
                        &task_context,
                        prompt_settings.max_knowledge_fragments,
                    );
                    observed_knowledge_fragment_iris = fragments
                        .iter()
                        .map(|fragment| fragment.fragment_iri.clone())
                        .collect();
                    fragments.iter().map(render_knowledge_hint).collect()
                } else {
                    Vec::new()
                };
                discovered_skill_count = skill_hints.len();
                discovered_knowledge_count = knowledge_hints.len();
            }
        }
        let policy_candidates = eligible_policy_candidates(
            discovered_skill_count,
            discovered_knowledge_count,
            perceived_hint_count,
        );
        let policy_choice = if self.learning_mode.injects_history() {
            self.policy_learning
                .choose(&policy_context, &policy_candidates, "baseline")
        } else {
            crate::core::policy_learning::PolicyChoice {
                context: policy_context.clone(),
                action: "baseline".to_string(),
                used_fallback: true,
                confidence: 0.0,
                explored: false,
                candidates: policy_candidates,
            }
        };
        let all_hints = if self.learning_mode.injects_history() {
            policy_treatment_hints(
                &policy_choice.action,
                &perception_hints,
                &skill_hints,
                &knowledge_hints,
                prompt_settings.max_learning_hints,
                prompt_settings.max_learning_hint_chars,
                prompt_settings.max_learning_hint_total_chars,
            )
        } else {
            Vec::new()
        };
        if !all_hints.is_empty() {
            tracing::info!(
                task_iri = %task_iri,
                policy_action = %policy_choice.action,
                hints = all_hints.len(),
                "Selected learning treatment enriched planning"
            );
        }
        // start_cycle captures observed history before the policy exists. From
        // this point onward BizAgents must see only the selected treatment.
        if let Some(cycle) = self.active_cycles.get_mut(&cycle_id) {
            cycle.experience_hints = all_hints.clone();
        }
        let learning_treatment = LearningTreatmentMetrics {
            perception_hints_observed: perceived_hint_count,
            experience_hint_fingerprints: self
                .active_cycles
                .get(&cycle_id)
                .map(|cycle| cycle.observed_experience_hint_fingerprints.clone())
                .unwrap_or_default(),
            skills_observed: discovered_skill_count,
            skill_iris_observed: observed_skill_iris,
            knowledge_fragments_observed: discovered_knowledge_count,
            knowledge_fragment_iris_observed: observed_knowledge_fragment_iris,
            hints_injected: all_hints.len(),
            hint_chars_injected: all_hints.iter().map(|hint| hint.chars().count()).sum(),
            task_family_raw_features: task_context.raw_features,
            experiment_pair_id: ctx.constraints.get("learning_pair_id").cloned(),
            experiment_seed: ctx.constraints.get("learning_seed").cloned(),
            experiment_model: ctx.constraints.get("learning_model").cloned(),
            experiment_config_fingerprint: ctx
                .constraints
                .get("learning_experiment_config_fingerprint")
                .cloned(),
            workspace_fingerprint: ctx
                .constraints
                .get("learning_workspace_fingerprint")
                .cloned(),
            objective_fingerprint: {
                use sha2::{Digest, Sha256};
                format!(
                    "sha256:{}",
                    hex::encode(&Sha256::digest(user_input.as_bytes())[..12])
                )
            },
            orchestration_mode: if ctx.workflow_jsonld.is_some() {
                "dag".to_string()
            } else {
                "pdca".to_string()
            },
        };
        self.event_bus
            .emit(
                task_iri,
                "LEARNING_TREATMENT",
                "SA",
                &serde_json::json!({
                    "mode": self.learning_mode,
                    "policy_context": policy_context,
                    "policy_action": policy_choice.action,
                    "treatment": learning_treatment,
                    "model_version": self.policy_learning.model_version(),
                })
                .to_string(),
            )
            .await;

        let initial_complexity = self.classify_complexity(user_input);
        let defer_fallback_planning = should_defer_fallback_planning(&ctx, initial_complexity);
        // Unified execution path: build ExecutionPlan from JSON-LD workflow or LLM.
        // For verify-first candidates, use a cheap structural fallback now and
        // generate the detailed fallback only if CA→AA says execution is needed.
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
        } else if defer_fallback_planning {
            info!(task_iri = %task_iri, complexity = ?initial_complexity, "Deferring detailed fallback planning until verify-first fails");
            self.build_plan_from_complexity(initial_complexity)
        } else if declared_effect_execution {
            // SA owns orchestration; PA owns detailed planning. Asking SA for
            // a complete model-generated plan and then dispatching PA to plan
            // it again duplicates latency, completion tokens, and inspection.
            // The structural plan preserves PDCA/DAG semantics while leaving
            // domain planning to the BizAgent abstraction designed for it.
            info!(task_iri = %task_iri, complexity = ?initial_complexity, "Using structural SA plan; detailed planning delegated once to PA");
            self.build_plan_from_complexity(initial_complexity)
        } else {
            self.analyze_task_with_llm(
                task_iri,
                user_input,
                &five_w2h,
                &all_hints,
                &ctx.constraints,
            )
            .await
        };
        tracing::info!(
            task_iri = %task_iri,
            elapsed_ms = task_started_at.elapsed().as_millis() as u64,
            steps = plan.steps.len(),
            "SA planning completed"
        );

        // ── Verify-first optimization ──
        // When workspace has existing files and plan is non-trivial:
        // prepend CA→AA to check existing code first, store original as fallback_steps.
        // execute_plan returns "failed" if verify CA fails → retry loop uses fallback_steps.
        if !plan.verify_first
            && ctx.workspace_file_summary.is_some()
            && ctx.resumed_messages.is_none()
            && ctx.workflow_jsonld.is_none()
            && !ctx
                .constraints
                .get("required_effect")
                .is_some_and(|value| value == "workspace_mutation")
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
                effect_policy: crate::core::effect::EffectPolicy::EvidenceOnly,
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
                effect_policy: crate::core::effect::EffectPolicy::DecisionOnly,
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
        // execute_plan aggregates every BizAgent inside one orchestration
        // cycle.  This second-level accumulator preserves facts across outer
        // SA PDCA retries so the terminal TaskResult describes the whole user
        // task rather than only its last cycle.
        let mut task_execution_facts = super::execution::TaskExecutionFacts::default();
        let mut plan_revision = 1u32;
        let mut fallback_plan_generated = !defer_fallback_planning;
        let mut next_scoped_plan: Option<ExecutionPlan> = None;
        let mut task_scope_replans_used = 0u32;
        let token_budget = five_w2h.how_much.as_ref().and_then(|h| h.token_budget);
        tracing::info!(
            task_iri = %task_iri,
            token_budget = ?token_budget,
            prompt_tokens_start = task_prompt_tokens_start,
            completion_tokens_start = task_completion_tokens_start,
            "SA task token budget initialized"
        );

        for cycle_num in 0..max_cycles {
            // Reset only the current PDCA attempt clock. The task lifetime
            // `started_at` remains unchanged for end-to-end SLO metrics.
            let now = chrono::Utc::now();
            let cycle_timeout = self.perception.cycle_timeout_secs().max(1);
            if let Some(cycle) = self
                .active_cycles
                .values_mut()
                .find(|cycle| cycle.task_iri == task_iri)
            {
                cycle.pdca_started_at = now;
                cycle.cycle_deadline_at = now + chrono::Duration::seconds(cycle_timeout);
                cycle.last_progress_at = now;
                cycle.last_timeout_alert_at = None;
                cycle.next_timeout_alert_at = None;
                cycle.timeout_alert_count = 0;
                cycle.outer_cycle_number = cycle_num + 1;
            }
            if let Some(limit) = token_budget {
                let cumulative_used = self
                    .runner
                    .total_prompt_tokens
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .saturating_add(
                        self.runner
                            .total_completion_tokens
                            .load(std::sync::atomic::Ordering::Relaxed),
                    );
                let baseline_used =
                    task_prompt_tokens_start.saturating_add(task_completion_tokens_start);
                let used = cumulative_used.saturating_sub(baseline_used);
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
                    blocked_result
                        .errors
                        .push("token budget exhausted".to_string());
                    final_result = Some(blocked_result);
                    break;
                }
            }
            let resumed = if cycle_num == 0 {
                ctx.resumed_messages.clone()
            } else {
                None
            };
            let resumed_state = if cycle_num == 0 {
                ctx.resumed_state.clone()
            } else {
                None
            };

            // Generate a detailed fallback only after verify-first actually
            // fails. This preserves full PA→DA quality while removing the
            // dominant latency/token cost from already-complete tasks.
            if cycle_num >= 1 && plan.verify_first && !fallback_plan_generated {
                let detailed = self
                    .analyze_task_with_llm(
                        task_iri,
                        user_input,
                        &five_w2h,
                        &all_hints,
                        &ctx.constraints,
                    )
                    .await;
                plan.fallback_steps = detailed.steps;
                plan.parallel_groups = detailed.parallel_groups;
                plan.task_complexity = detailed.task_complexity;
                plan.context_requirements = detailed.context_requirements;
                plan.success_metrics = detailed.success_metrics;
                plan.max_recursion_depth = detailed.max_recursion_depth;
                plan.sub_tasks = detailed.sub_tasks;
                plan.description = format!(
                    "[Verify-first] Check existing workspace before full PDCA. Fallback: {}",
                    detailed.description
                );
                fallback_plan_generated = true;
                info!(task_iri = %task_iri, fallback_steps = plan.fallback_steps.len(), "Verify-first failed; detailed fallback plan generated lazily");
            }

            // On retry after verify-first failed, switch to fallback_steps (full PDCA)
            let current_plan = if let Some(scoped) = next_scoped_plan.take() {
                plan_revision = plan_revision.saturating_add(1);
                scoped
            } else if cycle_num >= 1 && plan.verify_first && !plan.fallback_steps.is_empty() {
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
                        "⚠️ PDCA cycle #{} did not pass its latest quality gate — restarting with targeted feedback",
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
            let executed_plan = current_plan.clone();
            let mut result = self
                .execute_plan(
                    current_plan,
                    task_iri,
                    user_input,
                    five_w2h.clone(),
                    &five_w2h_iri,
                    resumed,
                    resumed_state,
                    cycle_feedback.clone(),
                    ctx.effective_effect_policy(),
                    ctx.constraints.clone(),
                )
                .await?;
            task_execution_facts.record(&result);

            // Verify-first: the AA's finish action hardcodes status "success" even when it
            // concluded full execution is needed, so that status is unusable here. Only treat
            // a verify-first cycle as complete when the AA verdict explicitly confirms the
            // task is already done; otherwise fall through to the retry loop which runs the
            // stored fallback_steps (full PDCA) on the next cycle.
            let needs_execution_after_verify =
                cycle_num == 0 && plan.verify_first && verify_aa_needs_execution(&result);

            let task_scope_failure = result.summary.contains("[Recovery] scope=Task")
                || result.summary.contains("scope=Task");
            let local_failure =
                result.status != "success" && !needs_execution_after_verify && !task_scope_failure;
            let decision_report = if result.status == "success" && !needs_execution_after_verify {
                crate::core::recovery::DecisionReport {
                    mode,
                    directive: crate::core::recovery::RecoveryDirective::Accept,
                    reason: crate::core::recovery::RecoveryReason::Accepted,
                    scope: crate::core::recovery::RepairScope::Task,
                    plan_revision,
                }
            } else if local_failure {
                crate::core::recovery::DecisionReport {
                    mode,
                    directive: crate::core::recovery::RecoveryDirective::RetryDa,
                    reason: crate::core::recovery::RecoveryReason::LocalExecutionGap,
                    scope: crate::core::recovery::RepairScope::Step,
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

            if result.status == "success" && !needs_execution_after_verify {
                task_execution_facts.apply_to(&mut result);
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
                self.record_final_learning_outcome(
                    task_iri,
                    &result,
                    &policy_choice,
                    &learning_treatment,
                    task_started_at,
                    task_prompt_tokens_start,
                    task_completion_tokens_start,
                    ctx.effective_effect_policy().requires_workspace_mutation(),
                )
                .await;
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

            if local_failure {
                if mode == crate::core::recovery::OrchestrationMode::Dag {
                    // DAG node-level retry/branch semantics are authoritative;
                    // rewriting and replaying the external graph would violate
                    // its topology and duplicate completed side effects.
                    final_result = Some(result);
                    break;
                }
                if let Some(failed_step) = recovery_failed_step(&result.summary) {
                    next_scoped_plan = scoped_recovery_plan(
                        &executed_plan,
                        failed_step,
                        plan_revision.saturating_add(1),
                    );
                }
            }
            if !local_failure && !needs_execution_after_verify {
                let max_plan_revisions = self
                    .runner
                    .agent_settings
                    .execution_budget
                    .max_plan_revisions;
                if task_scope_replans_used >= max_plan_revisions {
                    result.errors.push(format!(
                        "task-scope PA replan budget exhausted ({})",
                        max_plan_revisions
                    ));
                    final_result = Some(result);
                    break;
                }
                task_scope_replans_used = task_scope_replans_used.saturating_add(1);
            }

            // Build targeted feedback from the latest quality gate. The
            // failing gate can be CA (before AA is allowed to run) or AA; do
            // not misreport every retry as an AA rejection.
            // The CA→DA correction loop (in execute_plan) already handles in-cycle fixes;
            // this SA-level feedback addresses persistent failures requiring plan adjustment.
            let task_level_audit = result.summary.contains("[Recovery] scope=Task");
            let da_fixes = !task_level_audit
                && (result.summary.contains("Dimension Audit")
                    || result.summary.contains("execution failed")
                    || result.summary.contains("not completed"));

            let authoritative_contract = super::execution::authoritative_task_contract(
                user_input,
                &five_w2h,
                &ctx.constraints,
            );
            cycle_feedback = Some(if da_fixes {
                format!(
                    "{}\n\nPDCA Cycle #{} (plan revision {}): the latest quality gate identified execution-level issues.\n\
                     Status: {}\n\n\
                     Quality-gate evidence:\n{}\n\n\
                     ---\n\
                     PRESERVE your previous plan structure. Focus ONLY on:\n\
                     1. Which specific execution steps failed or were incomplete\n\
                     2. What DA needs to do differently (more detail, different approach)\n\
                     3. Do NOT create a brand new plan — refine the existing one",
                    authoritative_contract,
                    cycle_num + 1,
                    plan_revision,
                    result.status,
                    result.summary
                )
            } else {
                format!(
                    "{}\n\nPDCA Cycle #{} (plan revision {}): result\nStatus: {}\nRecovery directive: replan_pa\n\n\
                     Quality-gate evidence:\n{}\n\n\
                     ---\n\
                     The previous plan is not sufficient. Re-plan the task at PA,\
                     preserve verified evidence where possible, and create an improved approach.",
                    authoritative_contract,
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
        let mut final_result = final_result.unwrap_or_else(|| TaskResult {
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
        task_execution_facts.apply_to(&mut final_result);
        self.record_final_learning_outcome(
            task_iri,
            &final_result,
            &policy_choice,
            &learning_treatment,
            task_started_at,
            task_prompt_tokens_start,
            task_completion_tokens_start,
            ctx.effective_effect_policy().requires_workspace_mutation(),
        )
        .await;
        Ok(final_result)
    }

    async fn record_final_learning_outcome(
        &mut self,
        task_iri: &str,
        result: &TaskResult,
        policy_choice: &crate::core::policy_learning::PolicyChoice,
        treatment: &LearningTreatmentMetrics,
        task_started_at: std::time::Instant,
        prompt_tokens_start: u64,
        completion_tokens_start: u64,
        workspace_mutation_required: bool,
    ) {
        let prompt_tokens = self
            .runner
            .total_prompt_tokens
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(prompt_tokens_start);
        let completion_tokens = self
            .runner
            .total_completion_tokens
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(completion_tokens_start);
        let elapsed_ms = task_started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let reward = policy_reward_breakdown(
            result,
            prompt_tokens,
            elapsed_ms,
            workspace_mutation_required,
        );
        // A terminal status asserted by an LLM is not learning evidence on its
        // own.  The execution path stores this compact CA/AA record before it
        // returns here; if it is absent or not reusable, a claimed successful
        // task cannot promote a retrieval treatment.
        let audit_iri = task_audit_evidence_iri(task_iri);
        let independent_ca_aa_pass = self
            .runner
            .l0_store
            .retrieve(&audit_iri)
            .ok()
            .flatten()
            .and_then(|entry| {
                serde_json::from_str::<crate::core::policy_learning::TaskAuditKnowledgeEvidence>(
                    &entry.content,
                )
                .ok()
            })
            .is_some_and(|audit| audit.reusable_success());
        let terminal_success = matches!(result.status.as_str(), "success" | "completed");
        let successful_learning_evidence = !terminal_success || independent_ca_aa_pass;
        let policy_model_version_before = self.policy_learning.model_version();
        let mut evaluation = None;
        let evidence_iri = format!("iri://learning/evaluations/{}", uuid::Uuid::new_v4());
        let observation_evidence = crate::core::policy_learning::PolicyObservationEvidence {
            task_iri: Some(task_iri.to_string()),
            experiment_pair_id: treatment.experiment_pair_id.clone(),
            experiment_seed: treatment.experiment_seed.clone(),
            experiment_model: treatment.experiment_model.clone(),
            experiment_config_fingerprint: treatment.experiment_config_fingerprint.clone(),
            workspace_fingerprint: treatment.workspace_fingerprint.clone(),
            objective_fingerprint: Some(treatment.objective_fingerprint.clone()),
            orchestration_mode: Some(treatment.orchestration_mode.clone()),
        };
        let mut policy_observation_recorded = false;
        if self.learning_mode.updates_learning() && successful_learning_evidence {
            let task_result = serde_json::json!({
                "status": &result.status,
                "summary": &result.summary,
                "turn_count": result.turn_count,
                "tool_call_count": result.tool_call_count,
                "errors": &result.errors,
                "tracked_actions": &result.tracked_actions,
            });
            // A durable experience represents the terminal user task, not an
            // intermediate PA/DA/CA/AA BizAgent invocation.
            self.perception.on_task_end(&task_result, task_iri).await;
            match self.policy_learning.record_reward_gated_with_evidence(
                policy_choice,
                reward.total,
                self.policy_learning.gate(),
                observation_evidence,
            ) {
                Ok(report) => {
                    policy_observation_recorded = true;
                    evaluation = Some(report);
                }
                Err(error) => {
                    tracing::warn!(task_iri = %task_iri, %error, "Policy reward persistence failed")
                }
            }
        } else if matches!(
            self.learning_mode,
            crate::core::policy_learning::LearningMode::Baseline
        ) && successful_learning_evidence
        {
            // A baseline run remains a true behavioral ablation: no history
            // retrieval/injection, perception update, or model training. Its
            // immutable outcome is nevertheless required for a controlled
            // treatment-effect promotion gate.
            match self.policy_learning.record_baseline_evidence(
                policy_choice,
                reward.total,
                observation_evidence,
            ) {
                Ok(recorded) => policy_observation_recorded = recorded,
                Err(error) => tracing::warn!(
                    task_iri = %task_iri,
                    %error,
                    "Controlled baseline evidence persistence failed"
                ),
            }
        } else if terminal_success {
            tracing::info!(
                task_iri = %task_iri,
                "Skipped positive learning observation without independent CA/AA evidence"
            );
        }

        let mut trajectory_evidence_iris = vec![evidence_iri.clone()];
        if independent_ca_aa_pass {
            trajectory_evidence_iris.push(audit_iri.clone());
        }
        let trajectory = crate::core::learning_trajectory::LearningTrajectory {
            schema_version: crate::core::learning_trajectory::LEARNING_TRAJECTORY_SCHEMA_VERSION,
            task_iri: task_iri.to_string(),
            task_family: policy_choice.context.clone(),
            mode: self.learning_mode,
            policy_action: policy_choice.action.clone(),
            policy_candidates: policy_choice.candidates.clone(),
            policy_model_version: self.policy_learning.model_version(),
            policy_explored: policy_choice.explored,
            selected_skill_iris: treatment.skill_iris_observed.clone(),
            selected_knowledge_fragment_iris: treatment.knowledge_fragment_iris_observed.clone(),
            evidence_iris: trajectory_evidence_iris,
            tool_steps: result
                .tracked_actions
                .iter()
                .map(crate::core::learning_trajectory::TrajectoryToolStep::from)
                .collect(),
            outcome: crate::core::learning_trajectory::LearningTrajectoryOutcome {
                terminal_status: result.status.clone(),
                reward: reward.total,
                prompt_tokens,
                completion_tokens,
                elapsed_ms,
                independent_ca_aa_pass,
            },
            created_at: chrono::Utc::now(),
        };
        let trajectory_iri = match self.learning_trajectories.persist(&trajectory) {
            Ok(crate::core::learning_trajectory::TrajectoryPersistResult::Stored { iri })
            | Ok(crate::core::learning_trajectory::TrajectoryPersistResult::AlreadyPresent {
                iri,
            }) => Some(iri),
            Err(error) => {
                tracing::warn!(task_iri = %task_iri, %error, "Learning trajectory persistence failed");
                None
            }
        };

        // The constrained policy remains the authoritative promotion gate.
        // This secondary lifecycle record makes every promoted version
        // inspectable and provides a one-way automatic freeze path.
        let mut evolution_delta_iri = None;
        if independent_ca_aa_pass {
            if let Some(policy_evaluation) = evaluation.as_ref().filter(|report| report.accepted) {
                if let (Some(action), Some(trajectory_iri)) = (
                    policy_evaluation.candidate_action.as_deref(),
                    trajectory_iri.as_deref(),
                ) {
                    if crate::core::retrieval_policy::RetrievalPolicyArm::parse(action).is_some_and(
                        |arm| arm != crate::core::retrieval_policy::RetrievalPolicyArm::Baseline,
                    ) {
                        let candidate_revision = self.policy_learning.model_version();
                        if candidate_revision > policy_model_version_before {
                            let base_revision = candidate_revision.saturating_sub(1);
                            match crate::core::evolution_delta_gate::EvolutionDelta::proposed_policy(
                                task_iri,
                                &policy_choice.context,
                                action,
                                base_revision,
                                candidate_revision,
                                vec![audit_iri.clone(), trajectory_iri.to_string()],
                            ) {
                                Ok(delta) => {
                                    let delta_id = delta.delta_id.clone();
                                    let delta_iri = delta.storage_iri();
                                    let lifecycle = self
                                    .evolution_gate
                                    .propose(&delta)
                                    .and_then(|_| {
                                        self.evolution_gate.transition(
                                            &delta_id,
                                            crate::core::evolution_delta_gate::EvolutionDeltaState::ShadowValidated,
                                            "paired baseline and candidate evidence accepted by policy gate",
                                            false,
                                        )
                                    })
                                    .and_then(|_| {
                                        self.evolution_gate.transition(
                                            &delta_id,
                                            crate::core::evolution_delta_gate::EvolutionDeltaState::Active,
                                            "constrained policy promotion applied the candidate revision",
                                            false,
                                        )
                                    });
                                    match lifecycle {
                                        Ok(_) => evolution_delta_iri = Some(delta_iri),
                                        Err(error) => tracing::warn!(
                                            task_iri = %task_iri,
                                            %error,
                                            "Evolution delta lifecycle persistence failed"
                                        ),
                                    }
                                }
                                Err(error) => tracing::warn!(
                                    task_iri = %task_iri,
                                    %error,
                                    "Refused invalid evolution delta"
                                ),
                            }
                        }
                    }
                }
            }
        }

        let mut health_report = None;
        let mut frozen_delta_ids = Vec::new();
        if self.learning_mode.updates_learning()
            && policy_observation_recorded
            && crate::core::retrieval_policy::RetrievalPolicyArm::parse(&policy_choice.action)
                .is_some_and(|arm| {
                    arm != crate::core::retrieval_policy::RetrievalPolicyArm::Baseline
                })
        {
            let failed_actions = result
                .tracked_actions
                .iter()
                .filter(|action| {
                    matches!(
                        action.status,
                        crate::core::tracked_action::ActionStatus::Failed
                    )
                })
                .count();
            let tool_failure_rate = if result.tracked_actions.is_empty() {
                0.0
            } else {
                failed_actions as f64 / result.tracked_actions.len() as f64
            };
            let observation = crate::core::learning_health::LearningHealthObservation {
                schema_version: crate::core::learning_health::LEARNING_HEALTH_SCHEMA_VERSION,
                task_iri: task_iri.to_string(),
                task_family: policy_choice.context.clone(),
                policy_action: policy_choice.action.clone(),
                policy_model_version: self.policy_learning.model_version(),
                metrics: vec![
                    crate::core::learning_health::HealthMetricValue {
                        name: "reward".into(),
                        value: reward.total as f64,
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "terminal_success".into(),
                        value: if terminal_success && independent_ca_aa_pass {
                            1.0
                        } else {
                            0.0
                        },
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "verified_evidence".into(),
                        value: if independent_ca_aa_pass { 1.0 } else { 0.0 },
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "tool_failure_rate".into(),
                        value: tool_failure_rate,
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "elapsed_ms".into(),
                        value: elapsed_ms as f64,
                    },
                    crate::core::learning_health::HealthMetricValue {
                        name: "total_tokens".into(),
                        value: prompt_tokens.saturating_add(completion_tokens) as f64,
                    },
                ],
                created_at: chrono::Utc::now(),
            };
            match self.learning_health.record_and_assess(&observation) {
                Ok((_, report)) => {
                    if let Some(reason) = report.freeze_reason() {
                        match self
                            .policy_learning
                            .freeze_context(&policy_choice.context, &reason)
                        {
                            Ok(_) => {}
                            Err(error) => tracing::warn!(
                                task_iri = %task_iri,
                                %error,
                                "Policy safety freeze persistence failed"
                            ),
                        }
                        match self
                            .evolution_gate
                            .freeze_active_retrieval_family(&policy_choice.context, &reason)
                        {
                            Ok(deltas) => {
                                frozen_delta_ids =
                                    deltas.into_iter().map(|delta| delta.delta_id).collect()
                            }
                            Err(error) => tracing::warn!(
                                task_iri = %task_iri,
                                %error,
                                "Evolution delta freeze persistence failed"
                            ),
                        }
                    }
                    health_report = Some(report);
                }
                Err(error) => tracing::warn!(
                    task_iri = %task_iri,
                    %error,
                    "Learning health observation persistence failed"
                ),
            }
        }

        let evidence = serde_json::json!({
            "@id": format!("{}#learning-evaluation", task_iri),
            "@type": "LearningEvaluation",
            "task_iri": task_iri,
            "mode": self.learning_mode,
            "policy_context": policy_choice.context,
            "policy_action": policy_choice.action,
            "policy_explored": policy_choice.explored,
            "policy_deployment": if policy_choice.explored {
                "candidate_exploration"
            } else if self.policy_learning.model_version() > 0 && !policy_choice.used_fallback {
                "promoted_model"
            } else {
                "rule_baseline"
            },
            "treatment": treatment,
            "status": result.status,
            "turn_count": result.turn_count,
            "tool_call_count": result.tool_call_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_ms": elapsed_ms,
            "reward": reward,
            "policy_evaluation": evaluation,
            "policy_observation_recorded": policy_observation_recorded,
            "independent_ca_aa_pass": independent_ca_aa_pass,
            "learning_trajectory_iri": trajectory_iri,
            "evolution_delta_iri": evolution_delta_iri,
            "learning_health": health_report,
            "frozen_evolution_delta_ids": frozen_delta_ids,
            "policy_gate": self.policy_learning.gate(),
            "candidate_trial_min_baseline_samples": self.policy_learning.min_observations(),
            "model_version": self.policy_learning.model_version(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        if let Err(error) = self
            .runner
            .l0_store
            .store(&evidence_iri, &evidence.to_string())
        {
            tracing::warn!(task_iri = %task_iri, %error, "Learning evaluation persistence failed");
        }
        self.event_bus
            .emit(task_iri, "LEARNING_OUTCOME", "SA", &evidence.to_string())
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_contract_preserves_exact_user_quantity_and_boundary() {
        let original = "Create at least 10 test cases; do not require 10 test files.";
        let mut five_w2h = crate::core::five_w2h::Task5W2H::new(original, "verify behavior");
        five_w2h.why.success_criteria = vec!["at least 10 test cases pass".to_string()];

        let contract = crate::core::sa::execution::authoritative_task_contract(
            original,
            &five_w2h,
            &std::collections::HashMap::new(),
        );
        assert!(contract.contains(original));
        assert!(contract.contains("at least 10 test cases pass"));
        assert!(contract.contains("must not add, remove, strengthen, weaken, or reinterpret"));
    }

    #[test]
    fn direct_response_contract_does_not_require_a_file_or_graph_node() {
        let original = "输出一份 Markdown 调研报告";
        let five_w2h = crate::core::five_w2h::Task5W2H::new(original, "供用户阅读");
        let constraints = std::collections::HashMap::from([(
            crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT.to_string(),
            crate::core::agent_runner::DELIVERY_MODE_DIRECT_RESPONSE.to_string(),
        )]);
        let contract = crate::core::sa::execution::authoritative_task_contract(
            original,
            &five_w2h,
            &constraints,
        );
        assert!(contract.contains("direct_response"));
        assert!(contract.contains("filesystem path"));
        assert!(contract.contains("invented graph IRI"));
    }

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
        let deduped = dedup_hints(hints, 10, 700, 6_000);
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
        let deduped = dedup_hints(hints, 3, 700, 6_000);
        // Then: only the first 3 distinct hints remain
        assert_eq!(
            deduped,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_dedup_hints_empty_input() {
        // Given: empty hints
        let hints: Vec<String> = Vec::new();
        // When: deduplicated
        let deduped = dedup_hints(hints, 5, 700, 6_000);
        // Then: empty output
        assert!(deduped.is_empty());
    }

    #[test]
    fn policy_arms_require_real_treatment_material() {
        assert_eq!(eligible_policy_candidates(0, 0, 0), vec!["baseline"]);
        assert_eq!(
            eligible_policy_candidates(0, 0, 2),
            vec!["baseline", "experience_first"]
        );
        assert_eq!(
            eligible_policy_candidates(2, 0, 0),
            vec!["baseline", "skill_first"]
        );
        assert_eq!(
            eligible_policy_candidates(0, 1, 0),
            vec!["baseline", "knowledge_first"]
        );
        assert_eq!(
            eligible_policy_candidates(1, 1, 1),
            vec![
                "baseline",
                "knowledge_first",
                "experience_first",
                "skill_first"
            ]
        );
    }

    #[test]
    fn policy_treatments_are_distinct_and_baseline_is_a_true_ablation() {
        let experience = vec!["experience".to_string()];
        let skills = vec!["skill".to_string()];
        let knowledge = vec!["knowledge".to_string()];
        assert!(policy_treatment_hints(
            "baseline",
            &experience,
            &skills,
            &knowledge,
            20,
            700,
            6_000
        )
        .is_empty());
        assert_eq!(
            policy_treatment_hints(
                "experience_first",
                &experience,
                &skills,
                &knowledge,
                20,
                700,
                6_000,
            ),
            vec!["experience", "knowledge", "skill"]
        );
        assert_eq!(
            policy_treatment_hints(
                "knowledge_first",
                &experience,
                &skills,
                &knowledge,
                20,
                700,
                6_000,
            ),
            vec!["knowledge", "experience", "skill"]
        );
        assert_eq!(
            policy_treatment_hints(
                "skill_first",
                &experience,
                &skills,
                &knowledge,
                20,
                700,
                6_000,
            ),
            vec!["skill", "experience", "knowledge"]
        );
    }

    #[test]
    fn verify_first_defers_detailed_planning_only_for_existing_nontrivial_tasks() {
        let mut ctx = TaskContext::new("iri://task/lazy-plan", "verify", 20);
        ctx.workspace_file_summary = Some("12 files".to_string());
        assert!(should_defer_fallback_planning(
            &ctx,
            TaskComplexity::Complex
        ));
        assert!(!should_defer_fallback_planning(
            &ctx,
            TaskComplexity::Simple
        ));

        ctx.constraints.insert(
            "required_effect".to_string(),
            "workspace_mutation".to_string(),
        );
        assert!(!should_defer_fallback_planning(
            &ctx,
            TaskComplexity::Complex
        ));
        ctx.constraints.clear();

        ctx.workflow_jsonld = Some("{}".to_string());
        assert!(!should_defer_fallback_planning(
            &ctx,
            TaskComplexity::Complex
        ));
    }

    #[test]
    fn policy_context_ignores_order_and_volatile_numbers() {
        assert_eq!(
            crate::core::policy_learning::learning_policy_context(
                "Create report 20260825 from JSON data"
            ),
            crate::core::policy_learning::learning_policy_context(
                "JSON data create report 99117 from"
            )
        );
    }

    #[test]
    fn policy_context_is_stable_for_numbered_matched_artifacts() {
        assert_eq!(
            crate::core::policy_learning::learning_policy_context(
                "Create probe/1.txt with exact bytes"
            ),
            crate::core::policy_learning::learning_policy_context(
                "Create probe/987.txt with exact bytes"
            )
        );
    }

    #[test]
    fn reward_distinguishes_clean_success_from_costly_rework() {
        let clean = TaskResult {
            task_iri: "iri://task/clean".into(),
            status: "success".into(),
            verdict: None,
            summary: "accepted".into(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 5,
            tool_call_count: 2,
            five_w2h_updates: None,
            tracked_actions: vec![],
            archive_iri: None,
        };
        let mut costly = clean.clone();
        costly.turn_count = 20;
        costly.errors = vec!["retry one".into(), "retry two".into()];
        costly.summary = "[Recovery] scope=Task; eventually accepted".into();

        let clean_reward = policy_reward_breakdown(&clean, 10_000, 10_000, true);
        let costly_reward = policy_reward_breakdown(&costly, 180_000, 240_000, true);
        assert_eq!(clean_reward.total, 1.0);
        assert!(costly_reward.total < clean_reward.total);
        assert!(costly_reward.excess_turn_penalty > 0.0);
        assert!(costly_reward.recovery_penalty > 0.0);
        assert!(costly_reward.prompt_token_penalty > 0.0);
        assert!(costly_reward.latency_penalty > 0.0);
    }

    #[test]
    fn evidence_only_success_is_not_penalized_for_having_no_mutation_tail() {
        let mut tracker =
            crate::core::tracked_action::ActionTracker::new("iri://task/read-only", "CA");
        tracker.record(
            "file_read",
            &serde_json::json!({"path": "fixture.txt"}),
            &serde_json::json!({"success": true, "content": "ANSWER=helios-731"}),
            0.01,
        );
        let result = TaskResult {
            task_iri: "iri://task/read-only".into(),
            status: "success".into(),
            verdict: None,
            summary: "ANSWER=helios-731".into(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 2,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: tracker.actions,
            archive_iri: None,
        };
        let evidence_only = policy_reward_breakdown(&result, 1_000, 1_000, false);
        let mutation_required = policy_reward_breakdown(&result, 1_000, 1_000, true);
        assert_eq!(evidence_only.no_effect_tail, 1);
        assert_eq!(evidence_only.no_effect_tail_penalty, 0.0);
        assert!(mutation_required.no_effect_tail_penalty > 0.0);
        assert!(evidence_only.total > mutation_required.total);
    }

    #[test]
    fn structured_knowledge_is_family_scoped_and_token_bounded() {
        let mut matched = crate::skill_graph::types::KnowledgeFragment::new(
            "iri://fragment/matched",
            "iri://skill/app",
            "same family",
            "reuse",
        );
        matched.kind = "ca_validated_task_knowledge".into();
        matched.task_family = Some("family:a".into());
        matched.ca_verdict = Some("pass".into());
        matched.evidence_count = 2;
        matched.success_count = 2;
        matched.successful_checks = vec!["test passed".into()];
        let mut unrelated = matched.clone();
        unrelated.fragment_iri = "iri://fragment/unrelated".into();
        unrelated.task_family = Some("family:b".into());

        let task_context = LearningTaskContext {
            family: "family:a".into(),
            operations: vec![],
            modalities: vec![],
            raw_features: vec!["same".into(), "family".into()],
        };
        let ranked = rank_knowledge_fragments(vec![unrelated, matched], &task_context, 8);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].fragment_iri, "iri://fragment/matched");

        let owner_context = learning_task_context(
            "Extend the existing Python task queue with owner filtering and run tests",
        );
        let mut compatible_legacy = crate::skill_graph::types::KnowledgeFragment::new(
            "iri://fragment/legacy-tags",
            "iri://skill/app",
            "Add normalized tags to the existing Python task queue and run tests",
            "reuse with verification",
        );
        compatible_legacy.kind = "ca_validated_task_knowledge".into();
        compatible_legacy.task_family =
            Some("planning:v2:ops=build+operate+test+write;kinds=code+data".into());
        let mut wrong_feature = compatible_legacy.clone();
        wrong_feature.fragment_iri = "iri://fragment/unrelated-software".into();
        wrong_feature.description =
            "Modify a Rust network proxy certificate loader and benchmark TLS".into();
        let ranked =
            rank_knowledge_fragments(vec![wrong_feature, compatible_legacy], &owner_context, 8);
        assert_eq!(
            ranked
                .iter()
                .map(|fragment| fragment.fragment_iri.as_str())
                .collect::<Vec<_>>(),
            vec!["iri://fragment/legacy-tags"]
        );

        let max_hint_chars = 700;
        let max_hints = 20;
        let max_total_chars = 6_000;
        let oversized = vec!["x".repeat(max_hint_chars * 2); max_hints * 2];
        let bounded = dedup_hints(oversized, max_hints, max_hint_chars, max_total_chars);
        assert!(bounded
            .iter()
            .all(|hint| hint.chars().count() <= max_hint_chars + 1));
        assert!(
            bounded
                .iter()
                .map(|hint| hint.chars().count())
                .sum::<usize>()
                <= max_total_chars
        );
    }
}
