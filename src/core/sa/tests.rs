use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent_instance::AgentRole;
    use crate::core::agent_runner::{AgentRunner, TaskResult, TaskVerdict};
    use crate::core::event_bus::EventBus;
    use crate::gateway::unified_gateway::UnifiedGateway;
    use crate::memory::memory_manager::MemoryManager;
    use crate::templates::template_engine::TemplateEngine;
    use crate::tools::skill_registry::SkillRegistry;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_sa_with_tempdir() -> (SupervisorAgent, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let l0 = Arc::new(
            crate::memory::l0_store::L0Store::new(dir.path().join("l0").to_string_lossy().as_ref())
                .unwrap(),
        );
        let l2 = Arc::new(crate::memory::l2_blackboard::Blackboard::new().unwrap());
        let proj = Arc::new(crate::memory::l3_projection::ProjectionEngine::new(
            l2.clone(),
            500,
        ));
        let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
            l0.clone(),
            l2.clone(),
            proj.clone(),
            crate::CoreConfig::default(),
        )));
        let tmpl = Arc::new(TemplateEngine::new(std::path::Path::new("/nonexistent")).unwrap());
        let settings = crate::config::settings::GatewaySettings {
            base_url: "http://localhost:3000".to_string(),
            api_key: "sk-test".to_string(),
            default_model: "deepseek-v4-flash".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: HashMap::new(),
        };
        let gateway = Arc::new(UnifiedGateway::new(&settings).unwrap());
        let skills = Arc::new(SkillRegistry::new());
        let agent_settings = crate::config::settings::AgentSettings::default();
        let runner = Arc::new(AgentRunner::new(
            gateway,
            skills.clone(),
            l2.clone(),
            l0,
            mm,
            tmpl.clone(),
            agent_settings,
        ));
        let sa = SupervisorAgent::new(runner, tmpl, skills, Arc::new(EventBus::new(100)), 10)
            .with_memory(Some(l2), None, None);
        (sa, dir)
    }

    #[test]
    fn test_classify_simple() {
        let (sa, _dir) = make_sa_with_tempdir();
        assert_eq!(
            sa.classify_complexity("What is the weather?"),
            TaskComplexity::Simple
        );
        assert_eq!(
            sa.classify_complexity("Fix this bug in the code"),
            TaskComplexity::Emergency
        );
        assert_eq!(
            sa.classify_complexity("Build a web application with user authentication and database"),
            TaskComplexity::Recursive
        );
    }

    #[test]
    fn test_execution_plan_simple() {
        let (sa, _dir) = make_sa_with_tempdir();
        let plan = sa.analyze_task("Hello");
        assert_eq!(plan.agent_sequence.len(), 1);
        assert_eq!(plan.agent_sequence[0], AgentRole::Do);
    }

    #[test]
    fn test_execution_plan_emergency() {
        let (sa, _dir) = make_sa_with_tempdir();
        let plan = sa.analyze_task("Fix critical security vulnerability");
        assert_eq!(plan.agent_sequence.len(), 3);
        assert_eq!(plan.agent_sequence[0], AgentRole::Do);
        assert!(plan.agent_sequence.contains(&AgentRole::Act));
    }

    #[test]
    fn test_analyze_task_delegates_to_build_plan_from_complexity() {
        let (sa, _dir) = make_sa_with_tempdir();
        let proxied = sa.analyze_task("Fix critical security vulnerability");
        let direct = sa.build_plan_from_complexity(TaskComplexity::Emergency);
        assert_eq!(proxied.agent_sequence, direct.agent_sequence);
        assert_eq!(proxied.task_complexity, direct.task_complexity);
        assert_eq!(proxied.max_recursion_depth, direct.max_recursion_depth);
    }

    #[test]
    fn test_parse_llm_plan_truncates_to_max_steps() {
        let (sa, _dir) = make_sa_with_tempdir();
        let mut steps_json = String::new();
        let configured_limit = sa.runner.agent_settings.execution_budget.max_plan_steps;
        for i in 1..=(configured_limit + 4) {
            steps_json.push_str(&format!(
                r#"{{"step_id":"step_{}","role":"Do","objective":"obj {}","expected_output":"out","dependencies":[],"tools_allowed":[],"success_criteria":"done"}},"#,
                i, i
            ));
        }
        // Loop above leaves a trailing comma on the JSON array — strip it.
        steps_json.pop();
        let content = format!(
            r#"{{"complexity":"standard","description":"test","steps":[{}],"success_metrics":["ok"]}}"#,
            steps_json
        );
        let plan = sa.parse_llm_plan(&content).unwrap();
        assert_eq!(plan.steps.len(), configured_limit);
    }

    #[test]
    fn test_classify_research_deep_is_complex() {
        let (sa, _dir) = make_sa_with_tempdir();
        assert_eq!(
            sa.classify_complexity("Research the market comprehensively"),
            TaskComplexity::Complex
        );
    }

    #[test]
    fn test_classify_simple_patterns_reachable() {
        let (sa, _dir) = make_sa_with_tempdir();
        assert_eq!(
            sa.classify_complexity("What is Rust ownership"),
            TaskComplexity::Simple
        );
    }

    #[test]
    fn test_classify_weak_emergency_word_requires_reinforcement() {
        let (sa, _dir) = make_sa_with_tempdir();
        let long_log = format!(
            "There was an error in the log file at line 3 while processing the request: {}",
            "x".repeat(250)
        );
        assert_ne!(sa.classify_complexity(&long_log), TaskComplexity::Emergency);
        assert_eq!(
            sa.classify_complexity("Production outage: critical error detected"),
            TaskComplexity::Emergency
        );
    }

    #[test]
    fn test_parse_llm_plan_exploratory_builds_parallel_groups() {
        let (sa, _dir) = make_sa_with_tempdir();
        let do_steps: Vec<String> = (1..=3)
            .map(|i| {
                format!(
                    r#"{{"step_id":"step_{}","role":"Do","objective":"obj {}","expected_output":"out","dependencies":[],"tools_allowed":[],"success_criteria":"done"}}"#,
                    i, i
                )
            })
            .collect();
        let content = format!(
            r#"{{"complexity":"exploratory","description":"test","steps":[{}],"success_metrics":["ok"]}}"#,
            do_steps.join(",")
        );
        let plan = sa.parse_llm_plan(&content).unwrap();
        assert_eq!(plan.task_complexity, TaskComplexity::Exploratory);
        assert_eq!(plan.parallel_groups, vec![vec![AgentRole::Do; 3]]);
    }

    #[test]
    fn test_parse_llm_plan_exploratory_single_do_no_parallel_group() {
        let (sa, _dir) = make_sa_with_tempdir();
        let content = r#"{"complexity":"exploratory","description":"test","steps":[{"step_id":"step_1","role":"Do","objective":"obj","expected_output":"out","dependencies":[],"tools_allowed":[],"success_criteria":"done"}],"success_metrics":["ok"]}"#;
        let plan = sa.parse_llm_plan(content).unwrap();
        assert_eq!(plan.parallel_groups, Vec::<Vec<AgentRole>>::new());
    }

    #[test]
    fn test_recursive_plan_depth_unified_to_three() {
        let (sa, _dir) = make_sa_with_tempdir();
        let plan = sa.build_plan_from_complexity(TaskComplexity::Recursive);
        assert_eq!(plan.max_recursion_depth, 3);
    }

    #[tokio::test]
    async fn test_approval_timeout_defaults_to_approved() {
        let (sa, _dir) = make_sa_with_tempdir();
        let sa = sa.with_approval_wait_secs(0);
        let action = InterventionAction::IncreaseBudget {
            additional_tokens: 1000,
            additional_time_secs: 60,
        };
        let approved = sa
            .request_human_approval(&action, "iri://task/approval-timeout")
            .await
            .unwrap();
        assert!(approved, "timeout must default to approved (harmless)");
        let map = sa.pending_approvals.lock().await;
        assert_eq!(map.len(), 1, "pending_approvals must record the request");
        assert!(
            map.values().all(|v| *v),
            "timeout default must be recorded as approved"
        );
    }

    #[tokio::test]
    async fn test_approval_general_timeout_defaults_to_approved() {
        let (sa, _dir) = make_sa_with_tempdir();
        let sa = sa.with_approval_wait_secs(0);
        let result = sa
            .request_human_approval_general("proceed?", "node_1", "iri://task/approval-timeout")
            .await
            .unwrap();
        assert!(result.approved, "timeout must default to approved");
    }

    #[tokio::test]
    async fn test_approval_explicit_deny_respected() {
        let (sa, _dir) = make_sa_with_tempdir();
        let sa = sa.with_approval_wait_secs(5);
        let bus = sa.event_bus.clone();
        let mut rx = bus.subscribe();
        let task_iri = "iri://task/approval-deny";
        let action = InterventionAction::IncreaseBudget {
            additional_tokens: 1000,
            additional_time_secs: 60,
        };
        let task_iri_owned = task_iri.to_string();
        let bus2 = bus.clone();
        let responder = tokio::spawn(async move {
            loop {
                if let Ok(event) = rx.try_recv() {
                    if event.event_type == "HUMAN_APPROVAL_REQUIRED" {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&event.payload) {
                            if let Some(rid) = v.get("request_id").and_then(|r| r.as_str()) {
                                let rid = rid.to_string();
                                for _ in 0..20 {
                                    bus2.emit(
                                        &task_iri_owned,
                                        "HUMAN_APPROVAL_RESULT",
                                        "TEST",
                                        &serde_json::json!({"request_id": rid, "approved": false})
                                            .to_string(),
                                    )
                                    .await;
                                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                                }
                                return;
                            }
                        }
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });
        let approved = sa.request_human_approval(&action, task_iri).await.unwrap();
        responder.await.unwrap();
        assert!(!approved, "explicit deny event must be respected");
    }

    #[test]
    fn test_cleanup_expired_cycles() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        sa.active_cycles.insert(
            "old_cycle".to_string(),
            CycleState {
                cycle_id: "old_cycle".to_string(),
                task_iri: "iri://task/1".to_string(),
                phase: CyclePhase::Completed,
                iteration: 1,
                max_iterations: 10,
                started_at: chrono::Utc::now() - chrono::Duration::hours(2),
                pdca_started_at: chrono::Utc::now() - chrono::Duration::hours(2),
                cycle_deadline_at: chrono::Utc::now() - chrono::Duration::hours(1),
                last_progress_at: chrono::Utc::now() - chrono::Duration::hours(2),
                last_timeout_alert_at: None,
                next_timeout_alert_at: None,
                timeout_alert_count: 0,
                outer_cycle_number: 1,
                phase_history: vec![],
                task_completed: true,
                observed_experience_hint_count: 0,
                observed_experience_hint_fingerprints: vec![],
                experience_hints: vec![],
                intervention: InterventionState::default(),
            },
        );
        sa.cleanup_expired_cycles(3600);
        assert!(sa.active_cycles.is_empty());
    }

    #[test]
    fn test_verify_aa_needs_execution_parses_verdict() {
        fn result_with(summary: &str, verdict: Option<TaskVerdict>) -> TaskResult {
            TaskResult {
                task_iri: "iri://task/verify".to_string(),
                status: "success".to_string(),
                verdict,
                summary: summary.to_string(),
                output: None,
                jsonld_output: None,
                artifacts: vec![],
                errors: vec![],
                turn_count: 1,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
            }
        }

        // Verify-first AA concluded full execution is needed (the regression:
        // the agent_runner's finish action hardcodes status "success", so this
        // verdict must be recovered from the summary to trigger fallback_steps).
        assert!(
            verify_aa_needs_execution(&result_with(
                "Final verdict: needs full execution. Existing workspace has no calculator.py — deliverable is absent.",
                None
            )),
            "explicit needs-execution verdict must require execution"
        );
        assert!(
            verify_aa_needs_execution(&result_with(
                "The existing code does NOT satisfy the task requirements. Missing: calculator.py, test_calculator.py.",
                None
            )),
            "missing deliverables must require execution"
        );
        assert!(
            verify_aa_needs_execution(&result_with("", None)),
            "empty verdict must conservatively require execution"
        );
        // Verify-first AA confirmed the task is already done — must NOT require execution.
        assert!(
            !verify_aa_needs_execution(&result_with(
                "Final verdict: task already done. Existing calculator.py passes all test cases.",
                None
            )),
            "task-already-done verdict must not require execution"
        );
        assert!(
            !verify_aa_needs_execution(&result_with(
                "VERIFIED-PASS: existing code satisfies the task requirements.",
                None
            )),
            "VERIFIED-PASS must not require execution"
        );
        assert!(
            !verify_aa_needs_execution(&result_with(
                "SUCCESS: verified 19/19, single test file",
                Some(TaskVerdict::Success)
            )),
            "the required AA SUCCESS contract must terminate verify-first"
        );
        assert!(
            !verify_aa_needs_execution(&result_with(
                "PASS: ANSWER=helios-731 at line 3, file intact",
                Some(TaskVerdict::Success)
            )),
            "an AA-accepted CA deliverable restored for the user must still terminate verify-first"
        );
    }

    #[test]
    fn test_verify_aa_needs_execution_structured_verdict_priority() {
        fn result_with(summary: &str, verdict: Option<TaskVerdict>) -> TaskResult {
            TaskResult {
                task_iri: "iri://task/verify".to_string(),
                status: "success".to_string(),
                verdict,
                summary: summary.to_string(),
                output: None,
                jsonld_output: None,
                artifacts: vec![],
                errors: vec![],
                turn_count: 1,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
            }
        }

        // Structured verdict takes priority over summary text.
        assert!(
            verify_aa_needs_execution(&result_with("", Some(TaskVerdict::Blocked))),
            "Blocked verdict must require execution"
        );
        assert!(
            verify_aa_needs_execution(&result_with("task already done", Some(TaskVerdict::Failed))),
            "Failed verdict must override a completion-looking summary"
        );
        assert!(
            verify_aa_needs_execution(&result_with("", Some(TaskVerdict::Timeout))),
            "Timeout verdict must require execution"
        );
        // Success/PartialSuccess still consult the summary as a secondary check.
        assert!(
            !verify_aa_needs_execution(&result_with(
                "VERIFIED-PASS: task already complete",
                Some(TaskVerdict::Success)
            )),
            "Success verdict + completion summary must not require execution"
        );
        assert!(
            verify_aa_needs_execution(&result_with(
                "deliverable is absent",
                Some(TaskVerdict::Success)
            )),
            "Success verdict + ambiguous summary must conservatively require execution"
        );
    }

    #[test]
    fn test_finish_verdict_chinese_blocker_not_flatlined_to_success() {
        // The finish action historically flattened any verdict into status "success"
        // when detect_blocker_verdict missed the marker (e.g. a Chinese blocker
        // phrase). The structured channel must preserve the honest intent.
        fn result_with(summary: &str, verdict: Option<TaskVerdict>) -> TaskResult {
            TaskResult {
                task_iri: "iri://task/verify".to_string(),
                status: "success".to_string(),
                verdict,
                summary: summary.to_string(),
                output: None,
                jsonld_output: None,
                artifacts: vec![],
                errors: vec![],
                turn_count: 1,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                archive_iri: None,
            }
        }

        // Scenario: detect_blocker_verdict returned None (no English marker), so the
        // runner set status "success" — but the AA actually concluded it could not
        // proceed. With the structured channel, the SA still requires execution.
        let blocked = result_with("无法继续：缺少任务规格，零交付物", None);
        assert!(
            verify_aa_needs_execution(&blocked),
            "Chinese blocker summary must not be treated as verified-pass"
        );
    }

    #[tokio::test]
    async fn test_intervention_handlers_write_cycle_state() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let mut cycle = CycleState {
            cycle_id: "c1".to_string(),
            task_iri: "iri://task/1".to_string(),
            phase: CyclePhase::Executing,
            iteration: 1,
            max_iterations: 10,
            started_at: chrono::Utc::now(),
            pdca_started_at: chrono::Utc::now(),
            cycle_deadline_at: chrono::Utc::now() + chrono::Duration::minutes(5),
            last_progress_at: chrono::Utc::now(),
            last_timeout_alert_at: None,
            next_timeout_alert_at: None,
            timeout_alert_count: 0,
            outer_cycle_number: 1,
            phase_history: vec![],
            task_completed: false,
            observed_experience_hint_count: 0,
            observed_experience_hint_fingerprints: vec![],
            experience_hints: vec![],
            intervention: InterventionState::default(),
        };
        let task_iri = "iri://task/1";

        let timeout_handler =
            super::actions::get_action_handler(&InterventionAction::IncreaseTimeout {
                additional_seconds: 60,
            })
            .unwrap();
        let params = ActionParams {
            additional_seconds: Some(60),
            ..Default::default()
        };
        timeout_handler(&mut sa, &mut cycle, params, task_iri)
            .await
            .unwrap();
        assert_eq!(cycle.intervention.timeout_delta_secs, 60);

        let retry_handler =
            super::actions::get_action_handler(&InterventionAction::IncreaseRetry {
                additional_retries: 3,
            })
            .unwrap();
        let params = ActionParams {
            additional_retries: Some(3),
            ..Default::default()
        };
        retry_handler(&mut sa, &mut cycle, params, task_iri)
            .await
            .unwrap();
        assert_eq!(cycle.intervention.max_iterations_delta, 3);

        let restrict_handler =
            super::actions::get_action_handler(&InterventionAction::RestrictTools {
                allowed_tools: vec!["file_read".to_string()],
            })
            .unwrap();
        let params = ActionParams {
            allowed_tools: Some(vec!["file_read".to_string()]),
            ..Default::default()
        };
        restrict_handler(&mut sa, &mut cycle, params, task_iri)
            .await
            .unwrap();
        assert_eq!(
            cycle.intervention.tool_allowlist_override,
            Some(vec!["file_read".to_string()])
        );

        let monitor_handler =
            super::actions::get_action_handler(&InterventionAction::ContinueWithMonitor).unwrap();
        monitor_handler(&mut sa, &mut cycle, ActionParams::default(), task_iri)
            .await
            .unwrap();
        assert!(cycle.intervention.monitor);
    }

    #[test]
    fn test_effective_intervention_arithmetic() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let cycle_id = "c_eff".to_string();
        sa.active_cycles.insert(
            cycle_id.clone(),
            CycleState {
                cycle_id: cycle_id.clone(),
                task_iri: "iri://task/eff".to_string(),
                phase: CyclePhase::Executing,
                iteration: 1,
                max_iterations: 10,
                started_at: chrono::Utc::now(),
                pdca_started_at: chrono::Utc::now(),
                cycle_deadline_at: chrono::Utc::now() + chrono::Duration::minutes(5),
                last_progress_at: chrono::Utc::now(),
                last_timeout_alert_at: None,
                next_timeout_alert_at: None,
                timeout_alert_count: 0,
                outer_cycle_number: 1,
                phase_history: vec![],
                task_completed: false,
                observed_experience_hint_count: 0,
                observed_experience_hint_fingerprints: vec![],
                experience_hints: vec![],
                intervention: InterventionState {
                    max_iterations_delta: 5,
                    timeout_delta_secs: 60,
                    ..Default::default()
                },
            },
        );

        assert_eq!(
            sa.effective_max_iterations(&cycle_id),
            sa.max_iterations + 5
        );
        assert_eq!(sa.effective_timeout_secs(&cycle_id, 30), 90);

        sa.active_cycles
            .get_mut(&cycle_id)
            .unwrap()
            .intervention
            .max_iterations_delta = -100;
        sa.active_cycles
            .get_mut(&cycle_id)
            .unwrap()
            .intervention
            .timeout_delta_secs = -100;
        assert_eq!(sa.effective_max_iterations(&cycle_id), 1);
        assert_eq!(sa.effective_timeout_secs(&cycle_id, 30), 1);

        assert_eq!(sa.effective_max_iterations("missing"), sa.max_iterations);
        assert_eq!(sa.effective_timeout_secs("missing", 30), 30);
    }

    #[test]
    fn timeout_is_edge_triggered_and_recent_progress_extends_same_deadline() {
        let now = chrono::Utc::now();
        let mut cycle = CycleState {
            cycle_id: "timeout-edge".to_string(),
            task_iri: "iri://task/timeout-edge".to_string(),
            phase: CyclePhase::Executing,
            iteration: 1,
            max_iterations: 10,
            started_at: now - chrono::Duration::minutes(10),
            pdca_started_at: now - chrono::Duration::minutes(6),
            cycle_deadline_at: now - chrono::Duration::seconds(1),
            last_progress_at: now - chrono::Duration::seconds(5),
            last_timeout_alert_at: None,
            next_timeout_alert_at: None,
            timeout_alert_count: 0,
            outer_cycle_number: 1,
            phase_history: vec![],
            task_completed: false,
            observed_experience_hint_count: 0,
            observed_experience_hint_fingerprints: vec![],
            experience_hints: vec![],
            intervention: InterventionState::default(),
        };
        assert_eq!(
            super::execution::evaluate_cycle_timeout(&mut cycle, now, 300, 60),
            super::execution::TimeoutDecision::ExtendedWithProgress
        );
        assert!(cycle.cycle_deadline_at > now);
        assert_eq!(cycle.timeout_alert_count, 1);
        assert_eq!(
            super::execution::evaluate_cycle_timeout(
                &mut cycle,
                now + chrono::Duration::seconds(1),
                300,
                60,
            ),
            super::execution::TimeoutDecision::None
        );
        assert_eq!(cycle.timeout_alert_count, 1);
    }

    fn branch_fixture() -> (
        crate::core::workflow::loader::WorkflowDag,
        Vec<petgraph::graph::NodeIndex>,
    ) {
        let json = r#"{
            "@id": "wf:branch4",
            "name": "Branch4",
            "description": "branch test",
            "version": "1.0",
            "entry_node": "step_1",
            "nodes": [
                {"@id": "step_1", "@type": "AgentNode", "agent_role": "Do", "objective": "A", "next": "step_2"},
                {"@id": "step_2", "@type": "AgentNode", "agent_role": "Do", "objective": "B",
                 "branch_on_failure": {"condition": "$.result.status == 'failed'", "target": "step_4"},
                 "next": "step_3"},
                {"@id": "step_3", "@type": "AgentNode", "agent_role": "Do", "objective": "C", "next": "step_4"},
                {"@id": "step_4", "@type": "AgentNode", "agent_role": "Act", "objective": "D"}
            ]
        }"#;
        let def = crate::core::workflow::loader::load_workflow_jsonld(json).unwrap();
        let dag = crate::core::workflow::loader::build_dag(&def).unwrap();
        let order = crate::core::workflow::loader::topological_order(&dag).unwrap();
        (dag, order)
    }

    fn failed_result() -> TaskResult {
        TaskResult {
            task_iri: "iri://task/branch-test".to_string(),
            status: "failed".to_string(),
            verdict: Some(TaskVerdict::Failed),
            summary: "agent failed".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec!["boom".to_string()],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        }
    }

    #[tokio::test]
    async fn branch_on_failure_skips_intermediates_to_fallback() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let (dag, order) = branch_fixture();
        let step_2_idx = *dag.node_index.get("step_2").unwrap();
        let step = crate::core::workflow::adapter::node_to_planstep(&dag.graph[step_2_idx].def);
        assert!(step.branch_on_failure);
        assert_eq!(step.branch_fallback.as_deref(), Some("step_4"));

        let mut prev_summary = None;
        let mut da_output = None;
        let mut latest_da_result = None;
        let mut latest_ca_result = None;
        let mut latest_ca_report = None;
        let mut previous_ca_signature = None;
        let mut repeated_ca_failures = 0;
        let mut last_result = None;
        let mut execution_facts = super::execution::TaskExecutionFacts::default();
        let mut completed_node_results = std::collections::HashMap::new();
        let mut skip_nodes = std::collections::HashSet::new();
        let mut five_w2h = crate::core::five_w2h::Task5W2H::default();
        let plan = ExecutionPlan {
            plan_id: "branch4".to_string(),
            agent_sequence: vec![AgentRole::Do, AgentRole::Act],
            parallel_groups: vec![],
            task_complexity: TaskComplexity::Standard,
            description: "branch".to_string(),
            steps: vec![],
            context_requirements: Default::default(),
            success_metrics: vec![],
            max_recursion_depth: 0,
            sub_tasks: vec![],
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: vec![],
        };
        let step_2_wave = order.iter().position(|idx| *idx == step_2_idx).unwrap();
        let task_constraints = std::collections::HashMap::new();
        let mut recursive_budget = super::execution::RecursiveExecutionBudget::new(1, 1);

        let outcome = sa
            .handle_step_result(
                failed_result(),
                step,
                step_2_idx,
                step_2_wave,
                &mut prev_summary,
                &mut da_output,
                &mut latest_da_result,
                &mut latest_ca_result,
                &mut latest_ca_report,
                &mut previous_ca_signature,
                &mut repeated_ca_failures,
                &mut last_result,
                &mut execution_facts,
                &mut completed_node_results,
                &mut skip_nodes,
                &mut five_w2h,
                "iri://task/branch-test",
                "cycle-branch",
                &plan,
                &dag,
                &order,
                "iri://task/branch-test/5w2h",
                &crate::core::effect::EffectPolicy::None,
                &task_constraints,
                &mut recursive_budget,
            )
            .await
            .unwrap();

        assert!(
            outcome.is_none(),
            "branch must continue execution, not abort"
        );
        assert!(
            skip_nodes.contains("step_3"),
            "intermediate step_3 must be skipped"
        );
        assert!(
            !skip_nodes.contains("step_4"),
            "branch fallback step_4 must NOT be skipped"
        );
        assert!(
            last_result.is_some(),
            "failed result must still be recorded"
        );
    }

    #[tokio::test]
    async fn failed_without_branch_aborts_plan() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let (dag, order) = branch_fixture();
        let step_1_idx = *dag.node_index.get("step_1").unwrap();
        let step = crate::core::workflow::adapter::node_to_planstep(&dag.graph[step_1_idx].def);
        assert!(!step.branch_on_failure);

        let mut prev_summary = None;
        let mut da_output = None;
        let mut latest_da_result = None;
        let mut latest_ca_result = None;
        let mut latest_ca_report = None;
        let mut previous_ca_signature = None;
        let mut repeated_ca_failures = 0;
        let mut last_result = None;
        let mut execution_facts = super::execution::TaskExecutionFacts::default();
        let mut completed_node_results = std::collections::HashMap::new();
        let mut skip_nodes = std::collections::HashSet::new();
        let mut five_w2h = crate::core::five_w2h::Task5W2H::default();
        let plan = ExecutionPlan {
            plan_id: "branch4".to_string(),
            agent_sequence: vec![AgentRole::Do, AgentRole::Act],
            parallel_groups: vec![],
            task_complexity: TaskComplexity::Standard,
            description: "branch".to_string(),
            steps: vec![],
            context_requirements: Default::default(),
            success_metrics: vec![],
            max_recursion_depth: 0,
            sub_tasks: vec![],
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: vec![],
        };
        let step_1_wave = order.iter().position(|idx| *idx == step_1_idx).unwrap();
        let task_constraints = std::collections::HashMap::new();
        let mut recursive_budget = super::execution::RecursiveExecutionBudget::new(1, 1);

        let outcome = sa
            .handle_step_result(
                failed_result(),
                step,
                step_1_idx,
                step_1_wave,
                &mut prev_summary,
                &mut da_output,
                &mut latest_da_result,
                &mut latest_ca_result,
                &mut latest_ca_report,
                &mut previous_ca_signature,
                &mut repeated_ca_failures,
                &mut last_result,
                &mut execution_facts,
                &mut completed_node_results,
                &mut skip_nodes,
                &mut five_w2h,
                "iri://task/branch-test",
                "cycle-branch",
                &plan,
                &dag,
                &order,
                "iri://task/branch-test/5w2h",
                &crate::core::effect::EffectPolicy::None,
                &task_constraints,
                &mut recursive_budget,
            )
            .await
            .unwrap();

        assert!(
            outcome.is_some(),
            "failure without branch must abort the plan"
        );
        assert!(skip_nodes.is_empty(), "no nodes may be skipped on abort");
    }

    #[tokio::test]
    async fn dispatch_with_retry_retries_failed_results() {
        let mut calls = 0u32;
        let result = super::execution::dispatch_with_retry(2, 0, || {
            calls += 1;
            async move {
                Ok(TaskResult {
                    task_iri: "iri://task/retry".to_string(),
                    status: "failed".to_string(),
                    verdict: Some(TaskVerdict::Failed),
                    summary: "fail".to_string(),
                    output: None,
                    jsonld_output: None,
                    artifacts: vec![],
                    errors: vec!["boom".to_string()],
                    turn_count: 1,
                    tool_call_count: 0,
                    five_w2h_updates: None,
                    tracked_actions: Vec::new(),
                    archive_iri: None,
                })
            }
        })
        .await
        .unwrap();
        assert_eq!(calls, 3, "initial + 2 retries");
        assert_eq!(result.status, "failed");
        assert_eq!(result.turn_count, 3, "all retry attempts remain observable");
    }

    #[tokio::test]
    async fn dispatch_with_retry_stops_on_success() {
        let mut calls = 0u32;
        let result = super::execution::dispatch_with_retry(2, 0, || {
            calls += 1;
            let attempt = calls;
            async move {
                let status = if attempt == 2 { "success" } else { "failed" };
                Ok(TaskResult {
                    task_iri: "iri://task/retry".to_string(),
                    status: status.to_string(),
                    verdict: Some(TaskVerdict::Failed),
                    summary: "attempt".to_string(),
                    output: None,
                    jsonld_output: None,
                    artifacts: vec![],
                    errors: vec![],
                    turn_count: 1,
                    tool_call_count: 0,
                    five_w2h_updates: None,
                    tracked_actions: Vec::new(),
                    archive_iri: None,
                })
            }
        })
        .await
        .unwrap();
        assert_eq!(calls, 2, "second attempt succeeds, no third dispatch");
        assert_eq!(result.status, "success");
        assert_eq!(result.turn_count, 2, "successful retries retain prior work");
    }

    #[test]
    fn ca_handoff_inlines_evidence_for_toolless_aa() {
        let result = TaskResult {
            task_iri: "iri://task/ca-handoff".to_string(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "PASS: all criteria verified".to_string(),
            output: Some(serde_json::Value::String(
                "criterion A: PASS; command result: 19 passed".to_string(),
            )),
            jsonld_output: None,
            artifacts: vec![serde_json::json!({"path": "report.json"})],
            errors: vec![],
            turn_count: 3,
            tool_call_count: 3,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: Some("iri://task/ca-handoff/turn_3".to_string()),
        };

        let handoff = super::execution::result_handoff(&result, AgentRole::Check, 6_000);
        assert!(handoff.contains("Detailed CA Evidence (directly supplied)"));
        assert!(handoff.contains("19 passed"));
        assert!(handoff.contains("verification tool calls: 3"));
        assert!(handoff.contains("Trace Reference (not required for this decision)"));
        assert!(!handoff.contains("use read_agent_output"));
    }

    #[test]
    fn ca_handoff_bounds_large_unicode_evidence() {
        let result = TaskResult {
            task_iri: "iri://task/ca-handoff-large".to_string(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "PASS".to_string(),
            output: Some(serde_json::Value::String("证".repeat(7_000))),
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        };

        let handoff = super::execution::result_handoff(&result, AgentRole::Check, 6_000);
        assert!(handoff.contains("CA evidence truncated"));
        assert!(handoff.chars().count() < 6_500);
    }

    #[test]
    fn da_handoff_is_summary_bounded_and_keeps_on_demand_archive_reference() {
        let result = TaskResult {
            task_iri: "iri://task/da-handoff-large".to_string(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "implementation-detail-".repeat(1_000),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: Some("iri://task/da-handoff-large/turn_1".to_string()),
        };

        let handoff = super::execution::result_handoff(&result, AgentRole::Do, 2_000);
        assert!(handoff.contains("read_agent_output"));
        assert!(handoff.contains("iri://task/da-handoff-large/turn_1"));
        assert!(handoff.chars().count() < 2_200);
    }

    #[test]
    fn accepted_direct_response_returns_da_deliverable_not_aa_disposition() {
        let mut aa_result = TaskResult {
            task_iri: "iri://task/direct-response".to_string(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "SUCCESS: accept the report".to_string(),
            output: Some(serde_json::Value::String(
                "SUCCESS: accept the report".to_string(),
            )),
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 2,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: Some("iri://task/direct-response/session/aa/turn_2".to_string()),
        };
        let da_result = TaskResult {
            task_iri: aa_result.task_iri.clone(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "完整调研报告".to_string(),
            output: Some(serde_json::Value::String(
                "# AI Agent 调研报告\n\n```mermaid\ngraph TD\nA-->B\n```".to_string(),
            )),
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 8,
            tool_call_count: 4,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: Some("iri://task/direct-response/session/da/turn_8".to_string()),
        };
        let constraints = std::collections::HashMap::from([(
            crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT.to_string(),
            crate::core::agent_runner::DELIVERY_MODE_DIRECT_RESPONSE.to_string(),
        )]);

        super::execution::restore_accepted_deliverable(
            &mut aa_result,
            Some(&da_result),
            None,
            &constraints,
            &crate::core::effect::EffectPolicy::EvidenceOnly,
            false,
        );

        assert_eq!(aa_result.status, "success");
        assert_eq!(aa_result.summary, "完整调研报告");
        assert!(aa_result
            .output
            .as_ref()
            .and_then(serde_json::Value::as_str)
            .unwrap()
            .contains("```mermaid"));
        assert_eq!(aa_result.archive_iri, da_result.archive_iri);
    }

    #[test]
    fn verify_first_evidence_task_returns_ca_business_evidence_after_aa_accepts() {
        let mut final_result = TaskResult {
            task_iri: "iri://task/evidence".to_string(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "SUCCESS: task already done".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        };
        let ca_result = TaskResult {
            task_iri: final_result.task_iri.clone(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "PASS: ANSWER=helios-731 at fixture.txt:3 verified".to_string(),
            output: Some(serde_json::Value::String(
                "ANSWER=helios-731 — evidence: fixture.txt:3".to_string(),
            )),
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 2,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: Some("iri://task/evidence/session/ca/turn_2".to_string()),
        };

        super::execution::restore_accepted_deliverable(
            &mut final_result,
            None,
            Some(&ca_result),
            &std::collections::HashMap::new(),
            &crate::core::effect::EffectPolicy::EvidenceOnly,
            true,
        );

        assert_eq!(final_result.summary, ca_result.summary);
        assert_eq!(final_result.output, ca_result.output);
        assert_eq!(final_result.archive_iri, ca_result.archive_iri);
    }

    #[test]
    fn direct_response_recheck_uses_only_the_exact_agent_output_reader() {
        let constraints = std::collections::HashMap::from([
            (
                crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT.to_string(),
                crate::core::agent_runner::DELIVERY_MODE_DIRECT_RESPONSE.to_string(),
            ),
            (
                crate::core::agent_runner::WORKSPACE_CONTEXT_SCOPE_CONSTRAINT.to_string(),
                crate::core::agent_runner::WORKSPACE_CONTEXT_DISABLED.to_string(),
            ),
        ]);
        assert_eq!(
            super::execution::direct_response_recheck_tools(&constraints),
            Some(vec!["read_agent_output".to_string()])
        );

        let workspace_task = std::collections::HashMap::from([(
            crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT.to_string(),
            crate::core::agent_runner::DELIVERY_MODE_DIRECT_RESPONSE.to_string(),
        )]);
        assert_eq!(
            super::execution::direct_response_recheck_tools(&workspace_task),
            None,
            "workspace-backed CA must retain independent file/command verification"
        );
    }

    fn recovery_step(id: &str, role: AgentRole, dependencies: &[&str]) -> PlanStep {
        PlanStep {
            step_id: id.to_string(),
            role,
            objective: id.to_string(),
            expected_output: String::new(),
            dependencies: dependencies.iter().map(|value| value.to_string()).collect(),
            tools_allowed: Vec::new(),
            success_criteria: String::new(),
            branch_on_failure: false,
            branch_fallback: None,
            retry_count: 0,
            retry_delay_secs: 0,
            effect_policy: crate::core::effect::EffectPolicy::None,
        }
    }

    #[test]
    fn pdca_scoped_recovery_keeps_only_failed_step_and_successors() {
        let plan = ExecutionPlan {
            plan_id: "p".to_string(),
            agent_sequence: vec![
                AgentRole::Plan,
                AgentRole::Do,
                AgentRole::Check,
                AgentRole::Act,
            ],
            parallel_groups: Vec::new(),
            task_complexity: TaskComplexity::Complex,
            description: String::new(),
            steps: vec![
                recovery_step("pa", AgentRole::Plan, &[]),
                recovery_step("da", AgentRole::Do, &["pa"]),
                recovery_step("ca", AgentRole::Check, &["da"]),
                recovery_step("aa", AgentRole::Act, &["ca"]),
            ],
            context_requirements: HashMap::new(),
            success_metrics: Vec::new(),
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: None,
            verify_first: true,
            fallback_steps: vec![recovery_step("fallback", AgentRole::Do, &[])],
        };

        let delta = super::process::scoped_recovery_plan(&plan, "da", 2).unwrap();
        assert_eq!(
            delta
                .steps
                .iter()
                .map(|step| step.step_id.as_str())
                .collect::<Vec<_>>(),
            vec!["da", "ca", "aa"]
        );
        assert!(delta.steps[0].dependencies.is_empty());
        assert_eq!(delta.steps[1].dependencies, vec!["da"]);
        assert!(!delta.verify_first);
        assert!(delta.fallback_steps.is_empty());
    }

    #[test]
    fn external_dag_topology_is_not_rewritten_by_pdca_recovery() {
        let plan = ExecutionPlan {
            plan_id: "dag".to_string(),
            agent_sequence: vec![AgentRole::Do],
            parallel_groups: Vec::new(),
            task_complexity: TaskComplexity::Complex,
            description: String::new(),
            steps: vec![recovery_step("da", AgentRole::Do, &[])],
            context_requirements: HashMap::new(),
            success_metrics: Vec::new(),
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: Some("{}".to_string()),
            verify_first: false,
            fallback_steps: Vec::new(),
        };
        assert!(super::process::scoped_recovery_plan(&plan, "da", 2).is_none());
    }

    #[test]
    fn recoverable_tool_error_is_not_misclassified_as_agent_blocked() {
        assert!(!super::event_requires_blocked_intervention("AGENT_ERROR"));
        assert!(super::event_requires_blocked_intervention("AGENT_BLOCKED"));
    }
}
