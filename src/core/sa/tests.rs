use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent_instance::AgentRole;
    use crate::core::agent_runner::{AgentRunner, TaskContext, TaskResult, TaskVerdict};
    use crate::core::event_bus::EventBus;
    use crate::gateway::unified_gateway::UnifiedGateway;
    use crate::memory::memory_manager::MemoryManager;
    use crate::templates::template_engine::TemplateEngine;
    use crate::tools::skill_registry::SkillRegistry;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn make_sa_with_tempdir() -> (SupervisorAgent, tempfile::TempDir) {
        make_sa_with_tempdir_at_with_retries("http://localhost:3000", 3)
    }

    fn make_sa_with_tempdir_at(base_url: &str) -> (SupervisorAgent, tempfile::TempDir) {
        make_sa_with_tempdir_at_with_retries(base_url, 0)
    }

    fn make_sa_with_tempdir_at_with_retries(
        base_url: &str,
        max_retries: u32,
    ) -> (SupervisorAgent, tempfile::TempDir) {
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
            base_url: base_url.to_string(),
            api_key: "sk-test".to_string(),
            default_model: "deepseek-v4-flash".to_string(),
            timeout_seconds: 30,
            max_retries,
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

    async fn sa_response_server(
        responses: Vec<(u16, &'static str, String)>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind SA test server");
        let address = listener.local_addr().expect("SA test server address");
        let handle = tokio::spawn(async move {
            for (status, content_type, body) in responses {
                let (mut socket, _) = listener.accept().await.expect("accept SA request");
                let mut request = vec![0_u8; 16 * 1024];
                let _ = socket.read(&mut request).await;
                let reason = if status == 200 {
                    "OK"
                } else {
                    "Internal Server Error"
                };
                let header = format!(
                    "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(header.as_bytes()).await.expect("header");
                socket.write_all(body.as_bytes()).await.expect("body");
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{address}"), handle)
    }

    async fn sa_capturing_response_server(
        responses: Vec<(u16, &'static str, String)>,
    ) -> (
        String,
        tokio::task::JoinHandle<()>,
        Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind capturing SA test server");
        let address = listener.local_addr().expect("SA test server address");
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_by_server = captured.clone();
        let handle = tokio::spawn(async move {
            for (status, content_type, body) in responses {
                let (mut socket, _) = listener.accept().await.expect("accept SA request");
                let mut request = Vec::new();
                let mut chunk = [0_u8; 8 * 1024];
                let (header_end, content_length) = loop {
                    let read = socket.read(&mut chunk).await.expect("read request");
                    assert!(read > 0, "request ended before its headers");
                    request.extend_from_slice(&chunk[..read]);
                    if let Some(offset) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                        let header_end = offset + 4;
                        let headers = String::from_utf8_lossy(&request[..header_end]);
                        let content_length = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .expect("request must declare content-length");
                        break (header_end, content_length);
                    }
                };
                while request.len() < header_end + content_length {
                    let read = socket.read(&mut chunk).await.expect("read request body");
                    assert!(read > 0, "request ended before its complete body");
                    request.extend_from_slice(&chunk[..read]);
                }
                captured_by_server.lock().unwrap().push(
                    String::from_utf8(request[header_end..header_end + content_length].to_vec())
                        .expect("request body must be UTF-8 JSON"),
                );

                let reason = if status == 200 {
                    "OK"
                } else {
                    "Internal Server Error"
                };
                let header = format!(
                    "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(header.as_bytes()).await.expect("header");
                socket.write_all(body.as_bytes()).await.expect("body");
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{address}"), handle, captured)
    }

    fn completed_agent_response(summary: &str) -> String {
        let content = serde_json::json!({
            "content": "phase deliverable",
            "summary": summary,
            "action": "finish",
            "emphasis": []
        })
        .to_string();
        serde_json::json!({
            "id": "provider-phase-test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        })
        .to_string()
    }

    fn completed_stream_response(provider_id: &str, content: &str) -> String {
        let identity = serde_json::json!({
            "id": provider_id,
            "model": "test-model"
        });
        let content = serde_json::json!({
            "choices": [{"index": 0, "delta": {"content": content}}]
        });
        let finish = serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        });
        format!("data: {identity}\n\ndata: {content}\n\ndata: {finish}\n\ndata: [DONE]\n\n")
    }

    fn completed_responses_stream(provider_id: &str, content: &str) -> String {
        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": provider_id,
                "model": "test-model",
                "status": "in_progress"
            }
        });
        let delta = serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": content
        });
        let completed = serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": provider_id,
                "model": "test-model",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}]
                }],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
            }
        });
        format!("data: {created}\n\ndata: {delta}\n\ndata: {completed}\n\n")
    }

    fn typed_testing_plan(verification_kind: &str) -> String {
        serde_json::json!({
            "complexity": "standard",
            "description": "deliver and test one bounded result",
            "steps": [
                {
                    "step_id": "pa",
                    "role": "Plan",
                    "objective": "plan the bounded delivery",
                    "expected_output": "an actionable plan",
                    "dependencies": [],
                    "success_criteria": "the plan is actionable"
                },
                {
                    "step_id": "do_parent",
                    "role": "Do",
                    "objective": "deliver and test the bounded result",
                    "expected_output": "project/result.txt and passing test results",
                    "dependencies": ["pa"],
                    "success_criteria": "the artifact is delivered and tests pass",
                    "work_packages": [
                        {
                            "id": "implementation",
                            "objective": "implement the bounded result",
                            "expected_output": "project/result.txt",
                            "success_criteria": "project/result.txt exists",
                            "evidence_requirements": [{
                                "type": "artifact_delivery",
                                "paths": ["project/result.txt"],
                                "min_paths": 1
                            }],
                            "dependencies": []
                        },
                        {
                            "id": "testing",
                            "objective": "test the bounded result",
                            "expected_output": "passing test results",
                            "success_criteria": "the deterministic test passes",
                            "evidence_requirements": [{
                                "type": "verification",
                                "kind": verification_kind,
                                "min_count": 1
                            }],
                            "dependencies": ["implementation"]
                        }
                    ]
                },
                {
                    "step_id": "ca",
                    "role": "Check",
                    "objective": "verify the delivered result",
                    "expected_output": "criterion-linked audit",
                    "dependencies": ["do_parent"],
                    "success_criteria": "the result is independently verified"
                },
                {
                    "step_id": "aa",
                    "role": "Act",
                    "objective": "decide from verified evidence",
                    "expected_output": "terminal decision",
                    "dependencies": ["ca"],
                    "success_criteria": "the decision follows the audit"
                }
            ],
            "success_metrics": ["artifact delivered", "tests pass"]
        })
        .to_string()
    }

    fn phase_test_context(task_iri: &str) -> TaskContext {
        TaskContext::new(task_iri, "perform one phase", 2)
            .with_original_task("perform one phase")
            .with_constraint(
                crate::core::biz_agent::BIZ_AGENT_ORCHESTRATION_CONSTRAINT,
                crate::core::biz_agent::BIZ_AGENT_ORCHESTRATION_DISABLED,
            )
            .with_effect_policy(crate::core::effect::EffectPolicy::EvidenceOnly)
    }

    fn phase_test_materialization(
        role: AgentRole,
    ) -> (
        crate::core::sa::PlanStep,
        crate::core::context_model::AgentSpecSourceRecord,
    ) {
        let step_id = match role {
            AgentRole::Plan => "PLAN",
            AgentRole::Do => "DO",
            AgentRole::Check => "CHECK",
            AgentRole::Act => "ACT",
        };
        let step = crate::core::sa::PlanStep {
            step_id: step_id.to_string(),
            role,
            objective: "perform one isolated phase".to_string(),
            expected_output: "phase result".to_string(),
            dependencies: Vec::new(),
            tools_allowed: Vec::new(),
            success_criteria: "phase completes".to_string(),
            work_packages: Vec::new(),
            branch_on_failure: false,
            branch_fallback: None,
            retry_count: 0,
            retry_delay_secs: 0,
            effect_policy: crate::core::effect::EffectPolicy::EvidenceOnly,
        };
        let source = crate::core::context_model::AgentSpecSourceRecord::new(
            crate::core::context_model::AgentSpecSourceKind::LlmGeneratedPlan,
        )
        .with_source_ref(format!("iri://interaction/phase-test/{role}"))
        .with_producer("SupervisorAgent.test_planner")
        .with_model("test-model")
        .with_interaction_id(format!("llm-phase-test-{role}"));
        (step, source)
    }

    #[test]
    fn sa_creates_a_fresh_agent_identity_for_every_role_dispatch() {
        let (sa, _dir) = make_sa_with_tempdir();
        let mut ids = std::collections::HashSet::new();
        for role in [
            AgentRole::Plan,
            AgentRole::Do,
            AgentRole::Check,
            AgentRole::Act,
            AgentRole::Do,
        ] {
            let agent = sa.create_agent(role, "cycle-isolation");
            assert_eq!(agent.role, role);
            assert!(agent
                .agent_id
                .starts_with(&format!("cycle-isolation_{role}_")));
            assert!(
                ids.insert(agent.agent_id),
                "every dispatch needs a fresh Agent ID"
            );
        }
    }

    #[tokio::test]
    async fn repeated_sa_dispatches_isolate_agent_l1_and_dynamic_agent_md() {
        let (base_url, server, captured_requests) = sa_capturing_response_server(vec![
            (
                200,
                "application/json",
                completed_agent_response("SUCCESS: first isolated phase completed"),
            ),
            (
                200,
                "application/json",
                completed_agent_response("SUCCESS: second isolated phase completed"),
            ),
        ])
        .await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut lifecycle_events = sa.event_bus.subscribe();
        let mut interaction_events = sa.runner.llm_interactions.subscribe();
        let task_iri = "iri://task/repeated-sa-dispatch-isolation";
        let cycle_id = "cycle-repeated-sa-dispatch-isolation";
        let context = phase_test_context(task_iri).with_allowed_tools(Vec::new());

        let (mut first_step, mut first_source) = phase_test_materialization(AgentRole::Do);
        first_step.step_id = "DO_FIRST".to_string();
        first_step.objective = "produce the first independently specified result".to_string();
        first_step.expected_output = "first isolated result".to_string();
        first_source.source_ref = Some(format!("{task_iri}#DO_FIRST"));
        first_source.interaction_id = Some("llm-sa-plan-first".to_string());

        let (mut second_step, mut second_source) = phase_test_materialization(AgentRole::Do);
        second_step.step_id = "DO_SECOND".to_string();
        second_step.objective = "produce the second independently specified result".to_string();
        second_step.expected_output = "second isolated result".to_string();
        second_source.source_ref = Some(format!("{task_iri}#DO_SECOND"));
        second_source.interaction_id = Some("llm-sa-plan-second".to_string());

        // Compute the exact non-authoritative plan-message receipts expected
        // at the provider boundary. The real dispatch below must carry each
        // freshly compiled PlanStep, rather than a shared generic role prompt.
        let first_compiled = sa
            .runner
            .compile_biz_agent_prompt(
                AgentRole::Do,
                &context,
                Some(&first_step),
                Some(first_source.clone()),
            )
            .await;
        let second_compiled = sa
            .runner
            .compile_biz_agent_prompt(
                AgentRole::Do,
                &context,
                Some(&second_step),
                Some(second_source.clone()),
            )
            .await;
        assert_ne!(
            first_compiled.spec.agent_md_sha256,
            second_compiled.spec.agent_md_sha256
        );

        for (step, source) in [(first_step, first_source), (second_step, second_source)] {
            let result = sa
                .dispatch_agent(
                    AgentRole::Do,
                    context.clone(),
                    cycle_id,
                    Some(step),
                    Some(source),
                    0,
                )
                .await
                .expect("each isolated SA dispatch must complete");
            assert_eq!(result.status, "success");
        }
        server.await.unwrap();

        let assembled = std::iter::from_fn(|| interaction_events.try_recv().ok())
            .filter(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Assembled
                    && event.scope.stage == "agent_react"
                    && event.scope.task_iri.as_deref() == Some(task_iri)
            })
            .collect::<Vec<_>>();
        assert_eq!(assembled.len(), 2);
        let agent_ids = assembled
            .iter()
            .map(|event| {
                event
                    .scope
                    .agent_id
                    .clone()
                    .expect("every provider dispatch must retain its Agent identity")
            })
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(agent_ids.len(), 2, "SA must not reuse a BizAgent instance");

        let requests = captured_requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let request_messages = requests
            .iter()
            .map(|body| {
                let request = serde_json::from_str::<serde_json::Value>(body).unwrap();
                request["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|message| message["content"].as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(request_messages[0]
            .iter()
            .any(|content| content.contains("produce the first independently specified result")));
        assert!(!request_messages[0]
            .iter()
            .any(|content| content.contains("produce the second independently specified result")));
        assert!(request_messages[1]
            .iter()
            .any(|content| content.contains("produce the second independently specified result")));
        assert!(!request_messages[1]
            .iter()
            .any(|content| content.contains("produce the first independently specified result")));
        assert!(request_messages[0]
            .iter()
            .any(|content| content.contains(first_compiled.text())));
        assert!(request_messages[1]
            .iter()
            .any(|content| content.contains(second_compiled.text())));
        drop(requests);
        assert_ne!(
            assembled[0].request_hash, assembled[1].request_hash,
            "fresh agent.md definitions must reach distinct provider requests"
        );

        let started_agent_ids = std::iter::from_fn(|| lifecycle_events.try_recv().ok())
            .filter(|event| event.event_type == "Do_STARTED" && event.task_iri == task_iri)
            .map(|event| event.source_agent_iri)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(started_agent_ids, agent_ids);

        let archived_sessions = sa
            .runner
            .l0_store
            .scan_iri_prefix("iri://archive/session/", 32)
            .unwrap()
            .into_iter()
            .filter_map(|entry| serde_json::from_str::<serde_json::Value>(&entry.content).ok())
            .filter(|summary| summary["task_iri"].as_str() == Some(task_iri))
            .collect::<Vec<_>>();
        assert_eq!(archived_sessions.len(), 2);
        assert_eq!(
            archived_sessions
                .iter()
                .filter_map(|summary| summary["session_id"].as_str())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            2,
            "each fresh Agent must own a fresh L1 session"
        );
        assert_eq!(
            archived_sessions
                .iter()
                .filter_map(|summary| summary["agent_id"].as_str())
                .collect::<std::collections::HashSet<_>>(),
            agent_ids.iter().map(String::as_str).collect(),
            "archived L1 sessions must remain bound to their originating Agents"
        );
        assert_eq!(sa.runner.memory_manager.lock().await.l1_session_count(), 0);
    }

    #[tokio::test]
    async fn phase_hooks_wrap_the_real_sa_dispatch_in_order_with_one_trace() {
        use crate::tools::hooks::{FunctionHook, HookPoint, HookResult};

        let (base_url, server) = sa_response_server(vec![(
            200,
            "application/json",
            completed_agent_response("SUCCESS: phase completed"),
        )])
        .await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let observed = Arc::new(std::sync::Mutex::new(Vec::<(
            HookPoint,
            String,
            String,
            String,
        )>::new()));
        let observed_for_hook = observed.clone();
        sa.runner.hook_manager.register(Box::new(FunctionHook::new(
            "phase-order-recorder",
            vec![HookPoint::PhaseStart, HookPoint::PhaseEnd],
            -100,
            move |context| {
                observed_for_hook.lock().unwrap().push((
                    context.hook_point,
                    context
                        .data
                        .get("phase")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    context
                        .data
                        .get("stage_id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    context.trace_id.clone(),
                ));
                HookResult::Continue
            },
        )));

        let (step, source) = phase_test_materialization(AgentRole::Plan);
        let result = sa
            .dispatch_agent(
                AgentRole::Plan,
                phase_test_context("iri://task/phase-order"),
                "cycle-phase-order",
                Some(step),
                Some(source),
                0,
            )
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(result.status, "success");
        let observed = observed.lock().unwrap();
        assert_eq!(observed.len(), 2);
        assert_eq!(observed[0].0, HookPoint::PhaseStart);
        assert_eq!(observed[1].0, HookPoint::PhaseEnd);
        assert_eq!(observed[0].1, "PLAN");
        assert_eq!(observed[1].1, "PLAN");
        assert_eq!(observed[0].2, "PLAN");
        assert_eq!(observed[1].2, "PLAN");
        assert_eq!(observed[0].3, observed[1].3);
    }

    #[tokio::test]
    async fn phase_start_retry_is_bounded_and_skip_never_dispatches_the_agent() {
        use crate::tools::hooks::{FunctionHook, HookPoint, HookResult};

        let (sa, _dir) = make_sa_with_tempdir_at("http://127.0.0.1:9");
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_hook = attempts.clone();
        sa.runner.hook_manager.register(Box::new(FunctionHook::new(
            "retry-then-skip-phase",
            vec![HookPoint::PhaseStart],
            -100,
            move |_| {
                let attempt = attempts_for_hook.fetch_add(1, Ordering::SeqCst);
                if attempt < 2 {
                    HookResult::Retry
                } else {
                    HookResult::Skip
                }
            },
        )));

        let result = sa
            .dispatch_agent(
                AgentRole::Do,
                phase_test_context("iri://task/phase-skip"),
                "cycle-phase-skip",
                None,
                None,
                0,
            )
            .await
            .unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(result.status, "skipped");
        assert_eq!(result.turn_count, 0);
        assert_eq!(result.tool_call_count, 0);
    }

    #[tokio::test]
    async fn phase_abort_and_exhausted_retry_reject_before_dispatch() {
        use crate::tools::hooks::{FunctionHook, HookPoint, HookResult};

        for (name, hook_result, expected_attempts) in [
            ("abort-phase", HookResult::Abort, 1usize),
            ("retry-phase", HookResult::Retry, 3usize),
        ] {
            let (sa, _dir) = make_sa_with_tempdir_at("http://127.0.0.1:9");
            let attempts = Arc::new(AtomicUsize::new(0));
            let attempts_for_hook = attempts.clone();
            sa.runner.hook_manager.register(Box::new(FunctionHook::new(
                name,
                vec![HookPoint::PhaseStart],
                -100,
                move |_| {
                    attempts_for_hook.fetch_add(1, Ordering::SeqCst);
                    hook_result
                },
            )));

            let error = sa
                .dispatch_agent(
                    AgentRole::Do,
                    phase_test_context("iri://task/phase-reject"),
                    "cycle-phase-reject",
                    None,
                    None,
                    0,
                )
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                crate::CoreError::InteractionRejected { ref stage, .. }
                    if stage.starts_with("phase_start:")
            ));
            assert_eq!(attempts.load(Ordering::SeqCst), expected_attempts);
        }
    }

    #[tokio::test]
    async fn zero_node_timeout_inherits_agent_dispatch_default_and_is_not_replayed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let server_requests = requests.clone();
        let server = tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                server_requests.fetch_add(1, Ordering::SeqCst);
                let mut request = vec![0_u8; 16 * 1024];
                let _ = socket.read(&mut request).await;
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        });
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let sa = sa.with_agent_dispatch_timeout(1);
        let mut events = sa.event_bus.subscribe();
        let started = std::time::Instant::now();

        let (step, source) = phase_test_materialization(AgentRole::Do);
        let result = sa
            .dispatch_agent(
                AgentRole::Do,
                phase_test_context("iri://task/agent-dispatch-timeout"),
                "cycle-agent-dispatch-timeout",
                Some(step),
                Some(source),
                0,
            )
            .await
            .unwrap();

        assert!(started.elapsed() < std::time::Duration::from_secs(3));
        assert_eq!(result.status, "timeout");
        assert_eq!(result.verdict, Some(TaskVerdict::Timeout));
        assert_eq!(result.errors.len(), 1);
        assert!(result.summary.contains("not automatically replayed"));
        assert_eq!(result.turn_count, 1);
        assert_eq!(result.tool_call_count, 0);
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        let mut saw_timeout_event = false;
        while let Ok(event) = events.try_recv() {
            if event.event_type == "AGENT_TIMEOUT" {
                saw_timeout_event = true;
                let payload: serde_json::Value = serde_json::from_str(&event.payload).unwrap();
                assert_eq!(payload["timeout_seconds"], 1);
                assert_eq!(payload["turn_count"], 1);
                assert_eq!(payload["tool_call_count"], 0);
                assert_eq!(payload["automatic_retry"], false);
            }
        }
        assert!(saw_timeout_event);
        assert_eq!(
            sa.runner.memory_manager.lock().await.l1_session_count(),
            0,
            "cancelling the timed-out ReAct future must synchronously release its L1 lease"
        );
        server.abort();
    }

    #[tokio::test]
    async fn workflow_retry_does_not_replay_timeout_result() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_dispatch = attempts.clone();
        let result = super::execution::dispatch_with_retry(4, 0, move || {
            attempts_for_dispatch.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Ok(TaskResult {
                task_iri: "iri://task/no-timeout-replay".to_string(),
                status: "timeout".to_string(),
                summary: "timed out with uncertain side-effect state".to_string(),
                output: None,
                jsonld_output: None,
                artifacts: Vec::new(),
                errors: vec!["timeout".to_string()],
                turn_count: 0,
                tool_call_count: 0,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                verdict: Some(TaskVerdict::Timeout),
                archive_iri: None,
            }))
        })
        .await
        .unwrap();

        assert_eq!(result.status, "timeout");
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(super::execution::requires_safe_plan_stop(&result));
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
    fn test_parse_llm_plan_normalizes_to_one_parent_per_role_within_limit() {
        let (sa, _dir) = make_sa_with_tempdir();
        let configured_limit = sa.runner.agent_settings.execution_budget.max_plan_steps;
        let mut steps = Vec::new();
        let mut plan_objectives = Vec::new();
        for i in 1..=(configured_limit + 1) {
            let objective = format!("plan objective {i}");
            plan_objectives.push(objective.clone());
            steps.push(serde_json::json!({
                "step_id": format!("plan_{i}"),
                "role": "Plan",
                "objective": objective,
                "expected_output": "bounded execution plan",
                "dependencies": [],
                "success_criteria": "plan is actionable"
            }));
        }
        steps.extend([
            serde_json::json!({
                "step_id": "do_parent",
                "role": "Do",
                "objective": "deliver the requested project result",
                "expected_output": "project/result.txt",
                "dependencies": ["plan_1"],
                "success_criteria": "project/result.txt exists",
                "work_packages": [{
                    "id": "deliver_result",
                    "objective": "deliver the requested project result",
                    "expected_output": "project/result.txt",
                    "success_criteria": "project/result.txt exists",
                    "evidence_requirements": [{
                        "type": "artifact_delivery",
                        "paths": ["project/result.txt"],
                        "min_paths": 1
                    }],
                    "dependencies": []
                }]
            }),
            serde_json::json!({
                "step_id": "check",
                "role": "Check",
                "objective": "verify the delivered result",
                "expected_output": "criterion-linked audit",
                "dependencies": ["do_parent"],
                "success_criteria": "delivery is verified"
            }),
            serde_json::json!({
                "step_id": "act",
                "role": "Act",
                "objective": "decide from the audit",
                "expected_output": "terminal decision",
                "dependencies": ["check"],
                "success_criteria": "decision follows verified evidence"
            }),
        ]);
        let content = serde_json::json!({
            "complexity": "standard",
            "description": "test",
            "steps": steps,
            "success_metrics": ["ok"]
        })
        .to_string();
        let plan = sa.parse_llm_plan(&content).unwrap();
        assert!(plan.steps.len() <= configured_limit);
        assert_eq!(
            plan.steps.iter().map(|step| step.role).collect::<Vec<_>>(),
            vec![
                AgentRole::Plan,
                AgentRole::Do,
                AgentRole::Check,
                AgentRole::Act
            ]
        );
        let merged_plan = plan
            .steps
            .iter()
            .find(|step| step.role == AgentRole::Plan)
            .unwrap();
        assert!(merged_plan.objective.contains(&plan_objectives[0]));
        assert!(merged_plan
            .objective
            .contains(plan_objectives.last().unwrap()));
        let do_parent = plan
            .steps
            .iter()
            .find(|step| step.role == AgentRole::Do)
            .unwrap();
        assert_eq!(do_parent.work_packages.len(), 1);
        assert!(!do_parent.work_packages[0].evidence_requirements.is_empty());
    }

    #[test]
    fn structural_plan_builder_records_dynamic_agent_spec_provenance() {
        let (sa, _dir) = make_sa_with_tempdir();

        let structural = sa.build_plan_from_complexity(TaskComplexity::Standard);
        let structural_source = structural
            .agent_spec_source_for_step(&structural.steps[0].step_id)
            .unwrap()
            .unwrap();
        assert_eq!(
            structural_source.kind,
            crate::core::context_model::AgentSpecSourceKind::KernelGeneratedPlan
        );
    }

    #[test]
    fn detailed_plan_governance_keeps_numeric_limit_and_constitution_in_their_sections() {
        let constitution = "CONSTITUTION_SENTINEL\n- preserve the user contract";
        let rendered = super::planning::render_sa_plan_governance(17, constitution);

        assert!(rendered.contains("not to exceed 17"));
        assert!(rendered.contains("## Code of Conduct"));
        assert!(rendered.contains(constitution));
        let limit_line = rendered
            .lines()
            .find(|line| line.contains("Step count limit"))
            .expect("numeric step-limit line");
        assert!(!limit_line.contains("CONSTITUTION_SENTINEL"));
        assert!(
            rendered.find("not to exceed 17").unwrap()
                < rendered.find("CONSTITUTION_SENTINEL").unwrap(),
            "constitution belongs after the structural planning limits"
        );
    }

    #[test]
    fn test_parse_llm_plan_requires_caller_to_attach_source() {
        let (sa, _dir) = make_sa_with_tempdir();
        let content = r#"{"complexity":"simple","description":"test","steps":[{"step_id":"llm_do","role":"Do","objective":"implement","expected_output":"out","dependencies":[],"tools_allowed":[],"success_criteria":"done"}],"success_metrics":["ok"]}"#;
        let plan = sa.parse_llm_plan(content).unwrap();
        assert!(plan.agent_spec_provenance.is_none());
        assert!(plan.steps.iter().any(|step| step.step_id == "llm_do"));
        assert!(plan
            .steps
            .iter()
            .all(|step| !step.step_id.starts_with("step_kernel_")));
    }

    #[test]
    fn parse_llm_plan_rejects_empty_role_definition_fields() {
        let (sa, _dir) = make_sa_with_tempdir();
        let base = serde_json::json!({
            "complexity": "standard",
            "description": "complete role plan",
            "steps": [
                {"step_id":"pa","role":"Plan","objective":"plan","expected_output":"plan","dependencies":[],"success_criteria":"planned"},
                {"step_id":"da","role":"Do","objective":"execute","expected_output":"result","dependencies":["pa"],"success_criteria":"done"},
                {"step_id":"ca","role":"Check","objective":"verify","expected_output":"audit","dependencies":["da"],"success_criteria":"verified"},
                {"step_id":"aa","role":"Act","objective":"decide","expected_output":"decision","dependencies":["ca"],"success_criteria":"decided"}
            ],
            "success_metrics": ["complete"]
        });

        for field in [
            "step_id",
            "objective",
            "expected_output",
            "success_criteria",
        ] {
            let mut invalid = base.clone();
            invalid["steps"][1][field] = serde_json::Value::String("   ".to_string());
            let error = sa
                .parse_llm_plan(&invalid.to_string())
                .expect_err("empty role definition fields must fail planning");
            assert!(
                error.to_string().contains(field),
                "missing field {field} not reported: {error}"
            );
        }
    }

    #[test]
    fn parse_llm_plan_preserves_and_validates_nested_work_package_dag() {
        let (sa, _dir) = make_sa_with_tempdir();
        let content = r#"{
          "complexity":"standard",
          "description":"ordered work",
          "steps":[
            {"step_id":"pa","role":"Plan","objective":"plan","expected_output":"plan","dependencies":[],"success_criteria":"planned"},
            {
              "step_id":"do_parent","role":"Do","objective":"execute","expected_output":"result","dependencies":["pa"],"success_criteria":"done",
              "work_packages":[
                {"id":"a","objective":"produce A","expected_output":"project/A.md","success_criteria":"A exists","evidence_requirements":[{"type":"artifact_delivery","paths":["project/A.md"],"min_paths":1}],"dependencies":[]},
                {"id":"b","objective":"produce B","expected_output":"project/B.md","success_criteria":"B uses A","evidence_requirements":[{"type":"artifact_delivery","paths":["project/B.md"],"min_paths":1}],"dependencies":["a"]}
              ]
            },
            {"step_id":"ca","role":"Check","objective":"check","expected_output":"audit","dependencies":["do_parent"],"success_criteria":"verified"},
            {"step_id":"aa","role":"Act","objective":"decide","expected_output":"decision","dependencies":["ca"],"success_criteria":"decided"}
          ],
          "success_metrics":["ok"]
        }"#;
        let plan = sa.parse_llm_plan(content).unwrap();
        let do_step = plan
            .steps
            .iter()
            .find(|step| step.role == AgentRole::Do)
            .unwrap();
        assert_eq!(do_step.work_packages.len(), 2);
        assert_eq!(do_step.work_packages[1].dependencies, vec!["a"]);

        let cyclic = content.replace(
            "\"min_paths\":1}],\"dependencies\":[]",
            "\"min_paths\":1}],\"dependencies\":[\"b\"]",
        );
        assert!(sa
            .parse_llm_plan(&cyclic)
            .unwrap_err()
            .to_string()
            .contains("cycle"));
    }

    #[test]
    fn parse_llm_plan_accepts_downstream_check_test_receipt_and_localizes_its_dag() {
        let (sa, _dir) = make_sa_with_tempdir();
        let content = r#"{
          "complexity":"standard",
          "description":"calculator with a final independent test receipt",
          "steps":[
            {"step_id":"pa","role":"Plan","objective":"plan the calculator","expected_output":"execution plan","dependencies":[],"success_criteria":"planned"},
            {
              "step_id":"da","role":"Do","objective":"deliver calculator artifacts","expected_output":"calculator project","dependencies":["pa"],"success_criteria":"artifacts delivered",
              "work_packages":[
                {"id":"implementation","objective":"implement the calculator","expected_output":"calculator/calculator.py","success_criteria":"implementation complete","evidence_requirements":[{"type":"artifact_delivery","paths":["calculator/calculator.py"],"min_paths":1}],"dependencies":[]},
                {"id":"test_writer","objective":"write pytest test cases","expected_output":"calculator/test_calculator.py","success_criteria":"test source complete","evidence_requirements":[{"type":"artifact_delivery","paths":["calculator/test_calculator.py"],"min_paths":1}],"dependencies":["implementation"]},
                {"id":"documentation","objective":"write README usage documentation from the delivered pytest artifact","expected_output":"calculator/README.md documents the command derived from calculator/test_calculator.py","success_criteria":"README derives its pytest command from the delivered test artifact","evidence_requirements":[{"type":"artifact_delivery","paths":["calculator/README.md"],"min_paths":1}],"dependencies":["test_writer"]}
              ]
            },
            {
              "step_id":"ca","role":"Check","objective":"independently execute final tests","expected_output":"fresh final test receipt","dependencies":["da"],"success_criteria":"all tests pass",
              "work_packages":[
                {"id":"final_verification","objective":"run the delivered pytest suite in the final workspace state","expected_output":"passing pytest results","success_criteria":"pytest exits successfully","evidence_requirements":[{"type":"verification","kind":"test_execution","min_count":1}],"dependencies":["documentation"]}
              ]
            },
            {"step_id":"aa","role":"Act","objective":"decide from verified evidence","expected_output":"decision","dependencies":["ca"],"success_criteria":"decided"}
          ],
          "success_metrics":["calculator delivered and independently tested"]
        }"#;

        let plan = sa.parse_llm_plan(content).unwrap();
        let do_step = plan
            .steps
            .iter()
            .find(|step| step.role == AgentRole::Do)
            .unwrap();
        assert_eq!(do_step.work_packages[2].dependencies, vec!["test_writer"]);
        let check_step = plan
            .steps
            .iter()
            .find(|step| step.role == AgentRole::Check)
            .unwrap();
        assert_eq!(check_step.dependencies, vec![do_step.step_id.clone()]);
        assert!(check_step.work_packages[0].dependencies.is_empty());
        assert_eq!(
            check_step.work_packages[0].evidence_requirements,
            vec![
                WorkPackageEvidenceRequirement::Verification {
                    kind: crate::core::tracked_action::VerificationKind::TestExecution,
                    min_count: 1,
                },
                WorkPackageEvidenceRequirement::TestArtifactExecutionScope {
                    paths: vec!["calculator/test_calculator.py".to_string()],
                },
            ]
        );
        assert!(!check_step.work_packages[0]
            .evidence_requirements
            .iter()
            .any(|requirement| matches!(
                requirement,
                WorkPackageEvidenceRequirement::TestArtifactExecutionScope { paths }
                    if paths.iter().any(|path| path.ends_with("README.md"))
            )));
        validate_plan_work_package_dag(&check_step.work_packages).unwrap();
    }

    #[tokio::test]
    async fn generated_plan_provenance_correlates_exact_completed_interaction() {
        let plan_json = r#"{"complexity":"standard","description":"generated","steps":[{"step_id":"llm_pa","role":"Plan","objective":"plan","expected_output":"plan","dependencies":[],"success_criteria":"planned"},{"step_id":"llm_do","role":"Do","objective":"implement","expected_output":"out","dependencies":["llm_pa"],"success_criteria":"done"},{"step_id":"llm_ca","role":"Check","objective":"verify","expected_output":"audit","dependencies":["llm_do"],"success_criteria":"verified"},{"step_id":"llm_aa","role":"Act","objective":"decide","expected_output":"decision","dependencies":["llm_ca"],"success_criteria":"decided"}],"success_metrics":["ok"]}"#;
        let content_chunk = serde_json::json!({
            "choices": [{"index": 0, "delta": {"content": plan_json}}]
        });
        let body = format!(
            "data: {{\"id\":\"provider-plan-1\",\"model\":\"test-model\"}}\n\ndata: {}\n\ndata: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
            content_chunk
        );
        let (base_url, server) = sa_response_server(vec![(200, "text/event-stream", body)]).await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut events = sa.runner.llm_interactions.subscribe();
        let task_iri = "iri://task/generated-plan-provenance";

        let plan = sa
            .analyze_task_with_llm(
                task_iri,
                "Implement the requested change",
                &crate::core::five_w2h::Task5W2H::default(),
                &[],
                &HashMap::new(),
            )
            .await
            .unwrap();
        server.await.unwrap();

        let mut received = Vec::new();
        while let Ok(event) = events.try_recv() {
            received.push(event);
        }
        let completed = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation"
            })
            .expect("completed plan-generation interaction");
        let assembled = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Assembled
                    && event.scope.stage == "plan_generation"
            })
            .expect("assembled plan-generation interaction");
        assert_eq!(assembled.request_message_receipts.len(), 2);
        assert_eq!(
            assembled.reasoning_effort.as_deref(),
            Some("disabled"),
            "schema-constrained SA planning must not inherit an unbounded provider thinking default"
        );
        assert_eq!(assembled.request_message_receipts[0].role, "system");
        assert_eq!(
            assembled.request_message_receipts[0].kind,
            crate::core::context_model::ContextFragmentKind::AuthoritativeInstruction
        );
        assert_eq!(assembled.request_message_receipts[1].role, "user");
        assert_eq!(
            assembled.request_message_receipts[1].kind,
            crate::core::context_model::ContextFragmentKind::UserInput
        );
        let source = plan
            .agent_spec_source_for_step(&format!("wf:{}/llm_do", plan.plan_id))
            .unwrap()
            .unwrap();
        assert_eq!(
            source.kind,
            crate::core::context_model::AgentSpecSourceKind::LlmGeneratedPlan
        );
        assert_eq!(
            source.interaction_id.as_deref(),
            Some(completed.scope.interaction_id.as_str())
        );
        assert!(plan.context_requirements.is_empty());
    }

    #[tokio::test]
    async fn length_cutoff_without_visible_plan_gets_one_linked_low_latency_retry() {
        let truncated = concat!(
            "data: {\"id\":\"provider-plan-truncated\",\"model\":\"test-model\"}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"planning until the output budget is exhausted\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let valid_plan = r#"{"complexity":"standard","description":"bounded retry plan","steps":[{"step_id":"pa","role":"Plan","objective":"plan","expected_output":"plan","dependencies":[],"success_criteria":"planned"},{"step_id":"da","role":"Do","objective":"implement","expected_output":"result","dependencies":["pa"],"success_criteria":"implemented"},{"step_id":"ca","role":"Check","objective":"verify","expected_output":"audit","dependencies":["da"],"success_criteria":"verified"},{"step_id":"aa","role":"Act","objective":"decide","expected_output":"decision","dependencies":["ca"],"success_criteria":"decided"}],"success_metrics":["done"]}"#;
        let (base_url, server) = sa_response_server(vec![
            (200, "text/event-stream", truncated),
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-plan-retry", valid_plan),
            ),
        ])
        .await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut events = sa.runner.llm_interactions.subscribe();

        let plan = sa
            .analyze_task_with_llm(
                "iri://task/plan-visible-contract-retry",
                "Implement the requested change",
                &crate::core::five_w2h::Task5W2H::default(),
                &[],
                &HashMap::new(),
            )
            .await
            .expect("one bounded retry should recover a length-truncated control response");
        server.await.unwrap();

        let received = std::iter::from_fn(|| events.try_recv().ok()).collect::<Vec<_>>();
        let initial = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation"
            })
            .expect("transport-complete truncated interaction remains auditable");
        let retry = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation_contract_retry"
            })
            .expect("one semantic contract retry");
        assert_eq!(
            retry.scope.parent_interaction_id.as_deref(),
            Some(initial.scope.interaction_id.as_str())
        );
        assert_eq!(initial.reasoning_effort.as_deref(), Some("disabled"));
        assert_eq!(retry.reasoning_effort.as_deref(), Some("disabled"));
        let retry_assembled = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Assembled
                    && event.scope.stage == "plan_generation_contract_retry"
            })
            .expect("unusable-response retry assembly");
        assert_eq!(
            retry_assembled
                .request_message_receipts
                .iter()
                .map(|receipt| receipt.kind)
                .collect::<Vec<_>>(),
            vec![
                crate::core::context_model::ContextFragmentKind::AuthoritativeInstruction,
                crate::core::context_model::ContextFragmentKind::UserInput,
            ],
            "the historical unusable-response retry remains the original two-message request"
        );
        assert!(retry_assembled.advertised_tool_names.is_empty());
        assert_eq!(
            received
                .iter()
                .filter(|event| {
                    event.phase == crate::llm::LlmInteractionPhase::Assembled
                        && event.scope.stage.starts_with("plan_generation")
                })
                .count(),
            2,
            "semantic recovery is strictly bounded to one retry"
        );
        let source = plan
            .agent_spec_source_for_step(&format!("wf:{}/da", plan.plan_id))
            .unwrap()
            .unwrap();
        assert_eq!(
            source.interaction_id.as_deref(),
            Some(retry.scope.interaction_id.as_str())
        );
    }

    #[tokio::test]
    async fn responses_output_limit_uses_the_same_bounded_plan_recovery() {
        let incomplete = serde_json::json!({
            "type": "response.incomplete",
            "response": {
                "id": "provider-plan-output-limit",
                "model": "test-model",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 4096, "total_tokens": 4106}
            }
        });
        let valid_plan = r#"{"complexity":"standard","description":"responses recovery plan","steps":[{"step_id":"pa","role":"Plan","objective":"plan","expected_output":"plan","dependencies":[],"success_criteria":"planned"},{"step_id":"da","role":"Do","objective":"implement","expected_output":"result","dependencies":["pa"],"success_criteria":"implemented"},{"step_id":"ca","role":"Check","objective":"verify","expected_output":"audit","dependencies":["da"],"success_criteria":"verified"},{"step_id":"aa","role":"Act","objective":"decide","expected_output":"decision","dependencies":["ca"],"success_criteria":"decided"}],"success_metrics":["done"]}"#;
        let (base_url, server) = sa_response_server(vec![
            (200, "text/event-stream", format!("data: {incomplete}\n\n")),
            (
                200,
                "text/event-stream",
                completed_responses_stream("provider-plan-recovered", valid_plan),
            ),
        ])
        .await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        sa.runner.gateway.set_use_responses_api(true);
        let mut events = sa.runner.llm_interactions.subscribe();

        let plan = sa
            .analyze_task_with_llm(
                "iri://task/responses-output-limit-recovery",
                "Implement the requested change",
                &crate::core::five_w2h::Task5W2H::default(),
                &[],
                &HashMap::new(),
            )
            .await
            .expect("Responses output cutoff should consume the bounded semantic retry");
        server.await.unwrap();

        let received = std::iter::from_fn(|| events.try_recv().ok()).collect::<Vec<_>>();
        let initial = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Failed
                    && event.scope.stage == "plan_generation"
            })
            .expect("the incomplete provider response remains a failed interaction");
        assert_eq!(initial.error_class.as_deref(), Some("output_token_limit"));
        assert_eq!(initial.retryable, Some(false));
        let retry = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation_contract_retry"
            })
            .expect("the existing bounded plan recovery must complete");
        assert_eq!(
            retry.scope.parent_interaction_id.as_deref(),
            Some(initial.scope.interaction_id.as_str())
        );
        assert!(
            received
                .iter()
                .all(|event| !event.scope.stage.ends_with("_fallback")),
            "a deterministic output cutoff must not be treated as a transport fallback"
        );
        let source = plan
            .agent_spec_source_for_step(&format!("wf:{}/da", plan.plan_id))
            .unwrap()
            .unwrap();
        assert_eq!(
            source.interaction_id.as_deref(),
            Some(retry.scope.interaction_id.as_str())
        );
    }

    #[tokio::test]
    async fn invalid_typed_evidence_gets_one_causal_plan_contract_correction() {
        let invalid_plan = typed_testing_plan("build");
        let corrected_plan = typed_testing_plan("test_execution");
        let (base_url, server, requests) = sa_capturing_response_server(vec![
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-plan-invalid-evidence", &invalid_plan),
            ),
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-plan-corrected", &corrected_plan),
            ),
        ])
        .await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut events = sa.runner.llm_interactions.subscribe();

        let plan = sa
            .analyze_task_with_llm(
                "iri://task/typed-plan-contract-correction",
                "Produce a bounded result with verification",
                &crate::core::five_w2h::Task5W2H::default(),
                &[],
                &HashMap::new(),
            )
            .await
            .expect("the one causal correction should repair typed evidence");
        server.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2, "only one semantic retry is allowed");
        let retry_request: serde_json::Value = serde_json::from_str(&requests[1]).unwrap();
        assert!(retry_request
            .get("tools")
            .is_none_or(serde_json::Value::is_null));
        let retry_messages = retry_request["messages"].as_array().unwrap();
        assert_eq!(retry_messages.len(), 4);
        assert_eq!(retry_messages[1]["role"], "user");
        assert_eq!(retry_messages[1]["name"], "context_user_input");
        assert_eq!(retry_messages[2]["role"], "assistant");
        assert_eq!(retry_messages[2]["name"], "context_model_generated_plan");
        assert_eq!(retry_messages[2]["content"], invalid_plan);
        assert!(retry_messages[2]
            .get("tool_calls")
            .is_none_or(serde_json::Value::is_null));
        assert_eq!(retry_messages[3]["role"], "system");
        assert_eq!(
            retry_messages[3]["name"],
            "context_authoritative_instruction"
        );
        let correction = retry_messages[3]["content"].as_str().unwrap();
        assert!(correction.contains("sa_plan_evidence_contract"));
        assert!(correction.contains("test_execution"));
        drop(requests);

        let received = std::iter::from_fn(|| events.try_recv().ok()).collect::<Vec<_>>();
        let initial = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation"
            })
            .expect("initial completed candidate");
        let retry = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation_contract_retry"
            })
            .expect("completed correction candidate");
        assert_ne!(retry.scope.interaction_id, initial.scope.interaction_id);
        assert_eq!(
            retry.scope.parent_interaction_id.as_deref(),
            Some(initial.scope.interaction_id.as_str())
        );
        let assembled = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Assembled
                    && event.scope.stage == "plan_generation_contract_retry"
            })
            .expect("typed correction assembly");
        assert!(assembled.advertised_tool_names.is_empty());
        assert_eq!(
            assembled
                .request_message_receipts
                .iter()
                .map(|receipt| receipt.kind)
                .collect::<Vec<_>>(),
            vec![
                crate::core::context_model::ContextFragmentKind::AuthoritativeInstruction,
                crate::core::context_model::ContextFragmentKind::UserInput,
                crate::core::context_model::ContextFragmentKind::ModelHistory,
                crate::core::context_model::ContextFragmentKind::AuthoritativeInstruction,
            ]
        );
        let source = plan
            .agent_spec_source_for_step(&format!("wf:{}/do_parent", plan.plan_id))
            .unwrap()
            .unwrap();
        assert_eq!(
            source.interaction_id.as_deref(),
            Some(retry.scope.interaction_id.as_str())
        );
    }

    #[tokio::test]
    async fn two_invalid_typed_plan_candidates_block_before_dispatch_after_two_calls() {
        let first_invalid = typed_testing_plan("build");
        let second_invalid = typed_testing_plan("syntax");
        let (base_url, server, requests) = sa_capturing_response_server(vec![
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-plan-invalid-first", &first_invalid),
            ),
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-plan-invalid-second", &second_invalid),
            ),
        ])
        .await;
        let (mut sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut role_events = sa.event_bus.subscribe();
        let mut interaction_events = sa.runner.llm_interactions.subscribe();
        let task_iri = "iri://task/typed-plan-contract-exhausted";
        let user_input = "Produce a bounded result with verification";
        let context = TaskContext::new(task_iri, user_input, 2)
            .with_original_task(user_input)
            .with_constraint("required_effect", "workspace_mutation")
            .with_effect_policy(crate::core::effect::EffectPolicy::required_workspace_mutation());

        let result = sa
            .process_task_with_context(user_input, task_iri, context)
            .await
            .expect("invalid planning candidates fail closed as a blocked result");
        server.await.unwrap();

        assert_eq!(requests.lock().unwrap().len(), 2);
        assert_eq!(result.status, "blocked");
        assert_eq!(result.verdict, Some(TaskVerdict::Blocked));
        assert_eq!(result.turn_count, 0);
        assert_eq!(result.tool_call_count, 0);
        assert!(result.tracked_actions.is_empty());
        let diagnostic = result.errors.join("\n");
        assert!(diagnostic.contains("sa_plan_evidence_contract"));
        assert!(diagnostic.contains("single bounded causal correction"));
        assert!(result.summary.contains("No generic PA/DA/CA/AA fallback"));
        assert!(result.summary.contains("configured LLM responded"));
        assert!(!result.summary.contains("LLM becomes available"));

        let emitted = std::iter::from_fn(|| role_events.try_recv().ok()).collect::<Vec<_>>();
        for forbidden in ["PLAN_STARTED", "DO_STARTED", "CHECK_STARTED", "ACT_STARTED"] {
            assert!(
                emitted
                    .iter()
                    .all(|event| event.event_type.to_ascii_uppercase() != forbidden),
                "contract exhaustion must not emit {forbidden}: {emitted:?}"
            );
        }
        let blocked = emitted
            .iter()
            .find(|event| event.event_type == "RECOVERY_BLOCKED")
            .expect("typed planning contract blocker event");
        let blocked_payload: serde_json::Value =
            serde_json::from_str(&blocked.payload).expect("blocker payload JSON");
        assert_eq!(blocked_payload["reason"], "sa_plan_contract_rejected");
        assert_eq!(
            blocked_payload["required_action"],
            "inspect the planning diagnostic and retry with a contract-valid plan"
        );
        let interactions =
            std::iter::from_fn(|| interaction_events.try_recv().ok()).collect::<Vec<_>>();
        let plan_assemblies = interactions
            .iter()
            .filter(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Assembled
                    && matches!(
                        event.scope.stage.as_str(),
                        "plan_generation" | "plan_generation_contract_retry"
                    )
            })
            .collect::<Vec<_>>();
        assert_eq!(plan_assemblies.len(), 2);
        assert!(plan_assemblies
            .iter()
            .all(|event| event.advertised_tool_names.is_empty()));
        let initial = plan_assemblies
            .iter()
            .find(|event| event.scope.stage == "plan_generation")
            .unwrap();
        let retry = plan_assemblies
            .iter()
            .find(|event| event.scope.stage == "plan_generation_contract_retry")
            .unwrap();
        assert_eq!(
            retry.scope.parent_interaction_id.as_deref(),
            Some(initial.scope.interaction_id.as_str())
        );
    }

    #[test]
    fn empty_choice_set_is_an_unusable_sa_plan_completion() {
        let response = crate::gateway::unified_gateway::ChatCompletionResponse {
            id: Some("provider-empty-choice".to_string()),
            choices: Vec::new(),
            usage: None,
        };

        assert_eq!(
            crate::core::sa::planning::unusable_sa_plan_completion(&response),
            Some("provider returned no assistant choice")
        );
    }

    #[tokio::test]
    async fn explicit_order_rejects_single_unconstrained_do_plan() {
        let plan_json = r#"{"complexity":"standard","description":"missing order","steps":[{"step_id":"pa","role":"Plan","objective":"plan","expected_output":"plan","dependencies":[],"success_criteria":"planned"},{"step_id":"do","role":"Do","objective":"perform all work","expected_output":"result","dependencies":["pa"],"success_criteria":"done"},{"step_id":"ca","role":"Check","objective":"verify","expected_output":"audit","dependencies":["do"],"success_criteria":"verified"},{"step_id":"aa","role":"Act","objective":"decide","expected_output":"decision","dependencies":["ca"],"success_criteria":"decided"}],"success_metrics":["ok"]}"#;
        let (base_url, server, requests) = sa_capturing_response_server(vec![
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-order-plan-initial", plan_json),
            ),
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-order-plan-retry", plan_json),
            ),
        ])
        .await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let error = sa
            .analyze_task_with_llm(
                "iri://task/order-plan-rejected",
                "First produce A, then produce B",
                &crate::core::five_w2h::Task5W2H::default(),
                &[],
                &HashMap::new(),
            )
            .await
            .unwrap_err();
        server.await.unwrap();
        assert_eq!(requests.lock().unwrap().len(), 2);
        let diagnostic = error.to_string();
        assert_eq!(diagnostic.matches("canonical work-package").count(), 2);
        assert!(diagnostic.contains("single bounded causal correction"));
    }

    #[tokio::test]
    async fn model_plan_missing_required_roles_blocks_before_any_role_dispatch() {
        let five_w2h = serde_json::json!({
            "what": "Explain addition",
            "why_description": "User requested a verified explanation",
            "success_criteria": ["clear and verified"],
            "priority": "medium"
        })
        .to_string();
        let incomplete_plan = r#"{"complexity":"standard","description":"invalid downgrade","steps":[{"step_id":"only_do","role":"Do","objective":"explain addition","expected_output":"explanation","dependencies":[],"success_criteria":"clear"}],"success_metrics":["clear"]}"#;
        let (base_url, server, requests) = sa_capturing_response_server(vec![
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-5w2h-missing-role", &five_w2h),
            ),
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-plan-missing-role-initial", incomplete_plan),
            ),
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-plan-missing-role-retry", incomplete_plan),
            ),
        ])
        .await;
        let (mut sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut role_events = sa.event_bus.subscribe();
        let mut interaction_events = sa.runner.llm_interactions.subscribe();
        let task_iri = "iri://task/missing-llm-role-blocked";
        let context = TaskContext::new(
            task_iri,
            "Explain addition with independent verification",
            2,
        )
        .with_original_task("Explain addition with independent verification")
        .with_constraint(
            crate::core::biz_agent::BIZ_AGENT_ORCHESTRATION_CONSTRAINT,
            crate::core::biz_agent::BIZ_AGENT_ORCHESTRATION_DISABLED,
        );

        let result = sa
            .process_task_with_context(
                "Explain addition with independent verification",
                task_iri,
                context,
            )
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(result.status, "blocked");
        assert_eq!(result.verdict, Some(TaskVerdict::Blocked));
        assert_eq!(result.turn_count, 0);
        assert_eq!(result.tool_call_count, 0);
        assert!(result.tracked_actions.is_empty());
        assert_eq!(
            requests.lock().unwrap().len(),
            3,
            "5W2H plus two plan calls"
        );
        let diagnostic = result.errors.join("\n");
        assert_eq!(diagnostic.matches("missing required").count(), 2);
        assert!(diagnostic.contains("single bounded causal correction"));
        assert!(!diagnostic.contains("step_kernel_"));
        let emitted = std::iter::from_fn(|| role_events.try_recv().ok()).collect::<Vec<_>>();
        for forbidden in ["PLAN_STARTED", "DO_STARTED", "CHECK_STARTED", "ACT_STARTED"] {
            assert!(
                emitted
                    .iter()
                    .all(|event| event.event_type.to_ascii_uppercase() != forbidden),
                "invalid model plan must not emit {forbidden}: {emitted:?}"
            );
        }
        let interactions =
            std::iter::from_fn(|| interaction_events.try_recv().ok()).collect::<Vec<_>>();
        assert_eq!(
            interactions
                .iter()
                .filter(|event| {
                    event.phase == crate::llm::LlmInteractionPhase::Assembled
                        && matches!(
                            event.scope.stage.as_str(),
                            "plan_generation" | "plan_generation_contract_retry"
                        )
                })
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn chinese_calculator_keyword_standard_rejects_model_simple_downgrade() {
        let exact_task = "使用python语言开发计算器程序，需要先进行设计，使用markdown语言，涉及图形使用mermaid格式输出，然后进行测试和文档编写，完成整个工程。注意：必须新创建一个目录把项目相关内容都创建到该目录下。";
        let downgraded_plan = r#"{
          "complexity":"simple",
          "description":"incorrect single-role plan",
          "steps":[{
            "step_id":"calculator_do",
            "role":"Do",
            "objective":"design and implement calculator",
            "expected_output":"calculator project",
            "dependencies":[],
            "success_criteria":"project complete",
            "work_packages":[
              {"id":"design","objective":"design calculator","expected_output":"calculator/DESIGN.md","success_criteria":"design complete","evidence_requirements":[{"type":"artifact_delivery","paths":["calculator/DESIGN.md"],"min_paths":1}],"dependencies":[]},
              {"id":"implementation","objective":"implement calculator","expected_output":"calculator/calculator.py","success_criteria":"implementation complete","evidence_requirements":[{"type":"artifact_delivery","paths":["calculator/calculator.py"],"min_paths":1}],"dependencies":["design"]}
            ]
          }],
          "success_metrics":["complete"]
        }"#;
        let (base_url, server, requests) = sa_capturing_response_server(vec![
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-calculator-downgrade-initial", downgraded_plan),
            ),
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-calculator-downgrade-retry", downgraded_plan),
            ),
        ])
        .await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        assert_eq!(sa.classify_complexity(exact_task), TaskComplexity::Standard);

        let error = sa
            .analyze_task_with_llm(
                "iri://task/chinese-calculator-complexity-floor",
                exact_task,
                &crate::core::five_w2h::Task5W2H::default(),
                &[],
                &HashMap::new(),
            )
            .await
            .expect_err("keyword Standard must not be downgraded to a DA-only Simple plan");
        server.await.unwrap();

        assert_eq!(requests.lock().unwrap().len(), 2);
        let diagnostic = error.to_string();
        assert_eq!(diagnostic.matches("effective Standard").count(), 2);
        assert!(diagnostic.contains("PA"));
        assert!(diagnostic.contains("CA"));
        assert!(diagnostic.contains("AA"));
        assert!(diagnostic.contains("single bounded causal correction"));
        assert!(!diagnostic.contains("step_kernel_"));
    }

    #[tokio::test]
    async fn primary_and_fallback_planner_failure_blocks_before_any_role_dispatch() {
        let five_w2h = serde_json::json!({
            "what": "Explain addition",
            "why_description": "User requested an explanation",
            "success_criteria": ["clear answer"],
            "priority": "medium"
        })
        .to_string();
        let (base_url, server) = sa_response_server(vec![
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-5w2h", &five_w2h),
            ),
            (500, "text/plain", "primary planner unavailable".to_string()),
            (
                500,
                "text/plain",
                "fallback planner unavailable".to_string(),
            ),
        ])
        .await;
        let (mut sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut role_events = sa.event_bus.subscribe();
        let mut interaction_events = sa.runner.llm_interactions.subscribe();
        let task_iri = "iri://task/planning-fails-closed";
        let context = TaskContext::new(task_iri, "Explain addition", 2)
            .with_original_task("Explain addition")
            .with_constraint(
                crate::core::biz_agent::BIZ_AGENT_ORCHESTRATION_CONSTRAINT,
                crate::core::biz_agent::BIZ_AGENT_ORCHESTRATION_DISABLED,
            );

        let result = sa
            .process_task_with_context("Explain addition", task_iri, context)
            .await
            .expect("planner outage is an explicit blocked result, not a generic plan");
        server.await.unwrap();

        assert_eq!(result.status, "blocked");
        assert_eq!(result.verdict, Some(TaskVerdict::Blocked));
        assert_eq!(result.turn_count, 0);
        assert_eq!(result.tool_call_count, 0);
        assert!(result.tracked_actions.is_empty());
        assert!(result.summary.contains("No generic PA/DA/CA/AA fallback"));
        let diagnostic = result.errors.join("\n");
        assert!(diagnostic.contains("primary streaming request failed"));
        assert!(diagnostic.contains("non-streaming LLM fallback failed"));

        let emitted = std::iter::from_fn(|| role_events.try_recv().ok()).collect::<Vec<_>>();
        for forbidden in ["PLAN_STARTED", "DO_STARTED", "CHECK_STARTED", "ACT_STARTED"] {
            assert!(
                emitted
                    .iter()
                    .all(|event| event.event_type.to_ascii_uppercase() != forbidden),
                "planner failure must not emit {forbidden}: {emitted:?}"
            );
        }
        let blocked = emitted
            .iter()
            .find(|event| event.event_type == "RECOVERY_BLOCKED")
            .expect("typed planning blocker event");
        let blocked_payload: serde_json::Value =
            serde_json::from_str(&blocked.payload).expect("blocker payload JSON");
        assert_eq!(blocked_payload["reason"], "sa_planning_unavailable");
        assert_eq!(blocked_payload["automatic_role_dispatch"], false);
        assert_eq!(blocked_payload["generic_plan_fallback"], false);
        assert_eq!(blocked_payload["resume_safe"], true);

        let interactions =
            std::iter::from_fn(|| interaction_events.try_recv().ok()).collect::<Vec<_>>();
        assert!(interactions.iter().any(|event| {
            event.phase == crate::llm::LlmInteractionPhase::Failed
                && event.scope.stage == "plan_generation"
        }));
        assert!(interactions.iter().any(|event| {
            event.phase == crate::llm::LlmInteractionPhase::Failed
                && event.scope.stage == "plan_generation_fallback"
        }));
        assert!(interactions
            .iter()
            .all(|event| { event.scope.role.as_deref().is_none_or(|role| role == "SA") }));

        let checkpoint_manager = crate::core::checkpoint::CheckpointManager::with_persistence(
            sa.runner.l0_store.clone(),
        );
        assert!(checkpoint_manager
            .load_task_contract(task_iri)
            .unwrap()
            .is_none());
        assert!(checkpoint_manager.restore_task(task_iri).unwrap().is_none());
    }

    #[test]
    fn fallback_llm_plan_executes_with_its_exact_model_and_interaction_provenance() {
        std::thread::Builder::new()
            .name("sa-fallback-provenance-test".to_string())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(fallback_llm_plan_executes_with_provenance_inner())
            })
            .unwrap()
            .join()
            .expect("large-stack SA test thread must not panic");
    }

    async fn fallback_llm_plan_executes_with_provenance_inner() {
        let five_w2h = serde_json::json!({
            "what": "Explain addition",
            "why_description": "User requested an explanation",
            "success_criteria": ["clear answer"],
            "priority": "medium"
        })
        .to_string();
        let plan_json = serde_json::json!({
            "complexity": "simple",
            "description": "Explain addition using one isolated DA",
            "steps": [{
                "step_id": "explain_addition",
                "role": "Do",
                "objective": "Explain how addition combines quantities",
                "expected_output": "A concise explanation",
                "dependencies": [],
                "work_packages": [],
                "tools_allowed": [],
                "success_criteria": "The explanation is correct and clear"
            }],
            "success_metrics": ["correct", "clear"]
        })
        .to_string();
        let fallback_plan = serde_json::json!({
            "id": "provider-plan-fallback",
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": plan_json},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        })
        .to_string();
        let (base_url, server) = sa_response_server(vec![
            (
                200,
                "text/event-stream",
                completed_stream_response("provider-5w2h", &five_w2h),
            ),
            (500, "text/plain", "primary planner unavailable".to_string()),
            (200, "application/json", fallback_plan),
            (
                200,
                "application/json",
                completed_agent_response("SUCCESS: addition explained"),
            ),
        ])
        .await;
        let (mut sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut interaction_events = sa.runner.llm_interactions.subscribe();
        let task_iri = "iri://task/planning-fallback-executes";
        let context = TaskContext::new(task_iri, "Explain addition", 2)
            .with_original_task("Explain addition")
            .with_constraint(
                crate::core::biz_agent::BIZ_AGENT_ORCHESTRATION_CONSTRAINT,
                crate::core::biz_agent::BIZ_AGENT_ORCHESTRATION_DISABLED,
            );

        // This integration-shaped SA future carries all PDCA/recovery branch
        // state. The wrapper mirrors glidingcode's explicitly sized TUI
        // runtime stack instead of depending on RUST_MIN_STACK.
        let result = Box::pin(sa.process_task_with_context("Explain addition", task_iri, context))
            .await
            .expect("the second LLM planner call supplies an executable plan");
        server.await.unwrap();
        assert_eq!(result.status, "success");
        assert!(result.turn_count > 0, "the DA plan must actually execute");

        let interactions =
            std::iter::from_fn(|| interaction_events.try_recv().ok()).collect::<Vec<_>>();
        let fallback_completion = interactions
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation_fallback"
            })
            .expect("completed fallback planning interaction");
        assert!(interactions.iter().any(|event| {
            event.phase == crate::llm::LlmInteractionPhase::Failed
                && event.scope.stage == "plan_generation"
        }));
        assert!(interactions.iter().any(|event| {
            event.scope.role.as_deref() == Some("DA")
                && event.phase == crate::llm::LlmInteractionPhase::Completed
        }));

        let contract = crate::core::checkpoint::CheckpointManager::with_persistence(
            sa.runner.l0_store.clone(),
        )
        .load_task_contract(task_iri)
        .unwrap()
        .expect("the exact executable LLM plan is registered before role dispatch");
        let source = contract
            .execution_plan
            .agent_spec_source_for_step(&contract.execution_plan.steps[0].step_id)
            .unwrap()
            .unwrap();
        assert_eq!(
            source.kind,
            crate::core::context_model::AgentSpecSourceKind::LlmGeneratedPlan
        );
        assert_eq!(source.model.as_deref(), Some("deepseek-v4-flash"));
        assert_eq!(
            source.interaction_id.as_deref(),
            Some(fallback_completion.scope.interaction_id.as_str())
        );
        assert_eq!(
            fallback_completion.scope.parent_interaction_id.as_deref(),
            interactions
                .iter()
                .find(|event| {
                    event.phase == crate::llm::LlmInteractionPhase::Failed
                        && event.scope.stage == "plan_generation"
                })
                .map(|event| event.scope.interaction_id.as_str())
        );
    }

    #[tokio::test]
    async fn traced_sa_stream_returns_the_completed_stream_interaction_id() {
        let body = concat!(
            "data: {\"id\":\"provider-stream-1\",\"model\":\"test-model\"}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let (base_url, server) = sa_response_server(vec![(200, "text/event-stream", body)]).await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut events = sa.runner.llm_interactions.subscribe();
        let messages = vec![crate::gateway::unified_gateway::ChatMessage {
            role: "user".to_string(),
            content: "private plan request".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let traced = sa
            .chat_sa_streaming_traced(
                "iri://task/traced-stream",
                "plan_generation",
                "test-model",
                messages,
                None,
                Some(64),
            )
            .await
            .unwrap();
        server.await.unwrap();

        let mut received = Vec::new();
        while let Ok(event) = events.try_recv() {
            received.push(event);
        }
        let terminal = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation"
            })
            .expect("stream completion event");
        assert_eq!(traced.interaction_id, terminal.scope.interaction_id);
        assert_eq!(
            traced.response.choices[0].message.content.as_deref(),
            Some("done")
        );
    }

    #[tokio::test]
    async fn traced_sa_stream_returns_actual_fallback_interaction_id() {
        let fallback_body = serde_json::json!({
            "id": "provider-fallback-1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "fallback done"},
                "finish_reason": "stop"
            }]
        })
        .to_string();
        let (base_url, server) = sa_response_server(vec![
            (500, "text/plain", "stream failed".to_string()),
            (200, "application/json", fallback_body),
        ])
        .await;
        let (sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let mut events = sa.runner.llm_interactions.subscribe();
        let messages = vec![crate::gateway::unified_gateway::ChatMessage {
            role: "user".to_string(),
            content: "private fallback request".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let traced = sa
            .chat_sa_streaming_traced(
                "iri://task/traced-fallback",
                "plan_generation",
                "test-model",
                messages,
                None,
                Some(64),
            )
            .await
            .unwrap();
        server.await.unwrap();

        let mut received = Vec::new();
        while let Ok(event) = events.try_recv() {
            received.push(event);
        }
        let failed_stream = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Failed
                    && event.scope.stage == "plan_generation"
            })
            .expect("failed stream event");
        let completed_fallback = received
            .iter()
            .find(|event| {
                event.phase == crate::llm::LlmInteractionPhase::Completed
                    && event.scope.stage == "plan_generation_fallback"
            })
            .expect("fallback completion event");
        assert_ne!(traced.interaction_id, failed_stream.scope.interaction_id);
        assert_eq!(
            completed_fallback.scope.parent_interaction_id.as_deref(),
            Some(failed_stream.scope.interaction_id.as_str())
        );
        assert_eq!(
            traced.interaction_id,
            completed_fallback.scope.interaction_id
        );
        assert_eq!(
            traced.response.choices[0].message.content.as_deref(),
            Some("fallback done")
        );
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
        let content = r#"{
          "complexity":"exploratory",
          "description":"test",
          "steps":[
            {"step_id":"pa","role":"Plan","objective":"plan","expected_output":"plan","dependencies":[],"success_criteria":"planned"},
            {
              "step_id":"do_parent","role":"Do","objective":"coordinate three independent outcomes","expected_output":"three delivered results","dependencies":["pa"],"success_criteria":"all outcomes are delivered",
              "work_packages":[
                {"id":"package_1","objective":"obj 1","expected_output":"project/result_1.txt","success_criteria":"project/result_1.txt exists","evidence_requirements":[{"type":"artifact_delivery","paths":["project/result_1.txt"],"min_paths":1}],"dependencies":[]},
                {"id":"package_2","objective":"obj 2","expected_output":"project/result_2.txt","success_criteria":"project/result_2.txt exists","evidence_requirements":[{"type":"artifact_delivery","paths":["project/result_2.txt"],"min_paths":1}],"dependencies":[]},
                {"id":"package_3","objective":"obj 3","expected_output":"project/result_3.txt","success_criteria":"project/result_3.txt exists","evidence_requirements":[{"type":"artifact_delivery","paths":["project/result_3.txt"],"min_paths":1}],"dependencies":[]}
              ]
            },
            {"step_id":"ca","role":"Check","objective":"verify","expected_output":"audit","dependencies":["do_parent"],"success_criteria":"verified"},
            {"step_id":"aa","role":"Act","objective":"decide","expected_output":"decision","dependencies":["ca"],"success_criteria":"decided"}
          ],
          "success_metrics":["ok"]
        }"#;
        let plan = sa.parse_llm_plan(&content).unwrap();
        assert_eq!(plan.task_complexity, TaskComplexity::Exploratory);
        assert_eq!(plan.parallel_groups, Vec::<Vec<AgentRole>>::new());
        assert_eq!(
            plan.steps
                .iter()
                .filter(|step| step.role == AgentRole::Do)
                .count(),
            1,
            "SA must create one Do parent; same-role fan-out belongs to BizAgent"
        );
        let do_parent = plan
            .steps
            .iter()
            .find(|step| step.role == AgentRole::Do)
            .unwrap();
        assert_eq!(do_parent.work_packages.len(), 3);
        assert!(do_parent
            .work_packages
            .iter()
            .all(|package| package.dependencies.is_empty()));
        assert!(do_parent.work_packages[2].objective.contains("obj 3"));
    }

    #[test]
    fn test_parse_llm_plan_exploratory_single_do_no_parallel_group() {
        let (sa, _dir) = make_sa_with_tempdir();
        let content = r#"{"complexity":"exploratory","description":"test","steps":[{"step_id":"pa","role":"Plan","objective":"plan","expected_output":"plan","dependencies":[],"success_criteria":"planned"},{"step_id":"step_1","role":"Do","objective":"obj","expected_output":"out","dependencies":["pa"],"tools_allowed":[],"success_criteria":"done"},{"step_id":"ca","role":"Check","objective":"verify","expected_output":"audit","dependencies":["step_1"],"success_criteria":"verified"},{"step_id":"aa","role":"Act","objective":"decide","expected_output":"decision","dependencies":["ca"],"success_criteria":"decided"}],"success_metrics":["ok"]}"#;
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
    async fn test_approval_timeout_defaults_to_rejected() {
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
        assert!(!approved, "timeout must fail closed");
        let map = sa.pending_approvals.lock().await;
        assert!(
            map.is_empty(),
            "terminal approvals must not leak in the pending registry"
        );
    }

    #[tokio::test]
    async fn test_approval_general_timeout_defaults_to_rejected() {
        let (sa, _dir) = make_sa_with_tempdir();
        let sa = sa.with_approval_wait_secs(0);
        let result = sa
            .request_human_approval_general("proceed?", "node_1", "iri://task/approval-timeout")
            .await
            .unwrap();
        assert!(!result.approved, "timeout must fail closed");
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
                if let Ok(event) = rx.recv().await {
                    if event.event_type == "HUMAN_APPROVAL_REQUIRED" {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&event.payload) {
                            if let Some(rid) = v.get("request_id").and_then(|r| r.as_str()) {
                                bus2.emit(
                                    &task_iri_owned,
                                    "HUMAN_APPROVAL_RESULT",
                                    "TEST",
                                    &serde_json::json!({"request_id": rid, "approved": false})
                                        .to_string(),
                                )
                                .await;
                                return;
                            }
                        }
                    }
                }
            }
        });
        let approved = sa.request_human_approval(&action, task_iri).await.unwrap();
        responder.await.unwrap();
        assert!(!approved, "explicit deny event must be respected");
    }

    #[tokio::test]
    async fn test_approval_single_immediate_response_is_not_lost() {
        let (sa, _dir) = make_sa_with_tempdir();
        let sa = sa.with_approval_wait_secs(2);
        let bus = sa.event_bus.clone();
        let mut rx = bus.subscribe();
        let task_iri = "iri://task/approval-immediate";
        let responder_bus = bus.clone();
        let responder = tokio::spawn(async move {
            loop {
                let event = rx.recv().await.expect("approval request event");
                if event.event_type != "HUMAN_APPROVAL_REQUIRED" {
                    continue;
                }
                let payload: serde_json::Value =
                    serde_json::from_str(&event.payload).expect("approval request JSON");
                let request_id = payload["request_id"]
                    .as_str()
                    .expect("request id")
                    .to_string();
                responder_bus
                    .emit(
                        task_iri,
                        "HUMAN_APPROVAL_RESULT",
                        "TEST",
                        &serde_json::json!({"request_id": request_id, "approved": true})
                            .to_string(),
                    )
                    .await;
                break;
            }
        });

        let result = sa
            .request_human_approval_general("proceed?", "node-fast", task_iri)
            .await
            .unwrap();
        responder.await.unwrap();
        assert!(result.approved);
        assert!(sa.pending_approvals.lock().await.is_empty());
    }

    #[tokio::test]
    async fn cancelling_intervention_keeps_authoritative_cycle_resident() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut request = vec![0_u8; 4096];
                let _ = socket.read(&mut request).await;
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        });
        let (mut sa, _dir) = make_sa_with_tempdir_at(&base_url);
        let task_iri = "iri://task/intervention-cancel";
        let cycle_id = "cycle-intervention-cancel";
        let now = chrono::Utc::now();
        sa.active_cycles.insert(
            cycle_id.to_string(),
            CycleState {
                cycle_id: cycle_id.to_string(),
                task_iri: task_iri.to_string(),
                phase: CyclePhase::Executing,
                iteration: 1,
                max_iterations: 10,
                started_at: now,
                pdca_started_at: now,
                cycle_deadline_at: now + chrono::Duration::minutes(5),
                last_progress_at: now,
                last_timeout_alert_at: None,
                next_timeout_alert_at: None,
                timeout_alert_count: 0,
                outer_cycle_number: 1,
                phase_history: vec!["before".to_string()],
                task_completed: false,
                observed_experience_hint_count: 0,
                observed_experience_hint_fingerprints: Vec::new(),
                experience_hints: Vec::new(),
                intervention: InterventionState::default(),
            },
        );
        let plan = crate::perception::proactive_engine::InterventionPlan {
            anomaly_id: "cancel-test".to_string(),
            diagnosis: "wait for model".to_string(),
            actions: vec!["ContinueWithMonitor".to_string()],
            priority: "high".to_string(),
            should_interrupt: true,
        };

        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(50),
                sa.execute_intervention_for_cycle(plan, task_iri),
            )
            .await
            .is_err(),
            "outer task cancellation must interrupt the in-flight model request"
        );
        let cycle = sa
            .active_cycles
            .get(cycle_id)
            .expect("authoritative cycle must remain resident");
        assert_eq!(cycle.phase_history, vec!["before".to_string()]);
        server.abort();
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
        assert_eq!(
            sa.effective_timeout_secs(&cycle_id, 0),
            sa.agent_dispatch_timeout_secs + 60,
            "an unspecified node timeout must inherit the configured agent default"
        );

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
        assert_eq!(
            sa.effective_timeout_secs(&cycle_id, 0),
            sa.agent_dispatch_timeout_secs - 100
        );

        assert_eq!(sa.effective_max_iterations("missing"), sa.max_iterations);
        assert_eq!(sa.effective_timeout_secs("missing", 30), 30);
        assert_eq!(
            sa.effective_timeout_secs("missing", 0),
            sa.agent_dispatch_timeout_secs
        );

        let (unbounded, _dir) = make_sa_with_tempdir();
        let unbounded = unbounded.with_agent_dispatch_timeout(0);
        assert_eq!(unbounded.effective_timeout_secs("missing", 0), 0);
        assert_eq!(
            unbounded.effective_timeout_secs("missing", 17),
            17,
            "a positive workflow-node timeout must override an unbounded default"
        );
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
        ExecutionPlan,
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
        let mut plan = crate::core::workflow::adapter::dag_to_execution_plan(
            &dag,
            &def,
            "iri://task/branch-test",
        );
        plan.dag_jsonld = Some(json.to_string());
        (dag, order, plan)
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

    async fn route_check_result_for_test(
        result: TaskResult,
    ) -> (
        bool,
        Option<crate::core::recovery::RecoveryDirective>,
        bool,
        u32,
    ) {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let (dag, order, mut plan) = branch_fixture();
        // This helper exercises the generated PDCA behavior, not an external
        // workflow's explicit branch policy. Once the workflow payload is
        // removed, its provenance must likewise describe the LLM-generated
        // plan that the production recovery path accepts.
        plan.dag_jsonld = None;
        for plan_step in &mut plan.steps {
            if plan_step.expected_output.is_empty() {
                plan_step.expected_output = format!("{} result", plan_step.step_id);
            }
            if plan_step.success_criteria.is_empty() {
                plan_step.success_criteria = format!("{} completes", plan_step.step_id);
            }
        }
        plan.set_agent_spec_provenance(crate::core::context_model::ExecutionPlanProvenance::new(
            crate::core::context_model::AgentSpecSourceRecord::new(
                crate::core::context_model::AgentSpecSourceKind::LlmGeneratedPlan,
            )
            .with_source_ref("iri://task/branch-test#ca-recovery-plan")
            .with_producer("SupervisorAgent.plan_generation")
            .with_model("test-model")
            .with_interaction_id("interaction-ca-recovery-plan"),
        ))
        .expect("generated PDCA fixture provenance is valid");
        let step_index = *dag.node_index.get("step_1").unwrap();
        let mut step = crate::core::workflow::adapter::node_to_planstep(&dag.graph[step_index].def);
        step.role = AgentRole::Check;
        step.branch_on_failure = false;
        if let Some(plan_step) = plan
            .steps
            .iter_mut()
            .find(|plan_step| plan_step.step_id == step.step_id)
        {
            plan_step.role = AgentRole::Check;
            plan_step.branch_on_failure = false;
        }
        plan.agent_sequence = plan.steps.iter().map(|plan_step| plan_step.role).collect();

        let mut prev_summary = None;
        let mut latest_pa_handoff = None;
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
        let mut five_w2h = crate::core::five_w2h::Task5W2H::new(
            "deliver calculator project",
            "all requested artifacts must be present",
        );
        five_w2h.why.success_criteria = vec!["docs/README.md exists".to_string()];
        let step_wave = order.iter().position(|index| *index == step_index).unwrap();
        let mut task_constraints = std::collections::HashMap::new();
        let mut conformance_contract = None;
        let mut recursive_budget = super::execution::RecursiveExecutionBudget::new(1, 1);

        let outcome = sa
            .handle_step_result(
                result,
                step,
                step_index,
                step_wave,
                &mut prev_summary,
                &mut latest_pa_handoff,
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
                "build calculator project",
                "cycle-check-routing",
                &plan,
                &dag,
                &order,
                "iri://task/branch-test/5w2h",
                &crate::core::effect::EffectPolicy::None,
                &mut task_constraints,
                &mut conformance_contract,
                &mut recursive_budget,
            )
            .await
            .unwrap();

        let directive = latest_ca_report
            .as_ref()
            .map(|report| crate::core::recovery::select_directive(report, 0, 3));
        (
            outcome.is_some(),
            directive,
            latest_ca_result.is_some(),
            repeated_ca_failures,
        )
    }

    #[tokio::test]
    async fn untyped_structured_ca_fail_rechecks_ca_without_granting_mutation() {
        let result = TaskResult {
            task_iri: "iri://task/branch-test".to_string(),
            status: "failed".to_string(),
            verdict: Some(TaskVerdict::Failed),
            summary: "FAIL: docs/README.md is missing".to_string(),
            output: Some(serde_json::Value::String(
                "Overall verdict: FAIL\nFailed criterion: docs/README.md exists".to_string(),
            )),
            jsonld_output: None,
            artifacts: vec![],
            errors: vec![],
            turn_count: 2,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        };

        let (terminated, directive, ca_recorded, repeats) =
            route_check_result_for_test(result).await;
        assert!(
            !terminated,
            "a completed negative audit must enter recovery"
        );
        assert_eq!(
            directive,
            Some(crate::core::recovery::RecoveryDirective::RetryCa)
        );
        assert!(ca_recorded, "typed CA evidence must be retained");
        assert_eq!(repeats, 1);
    }

    #[tokio::test]
    async fn unstructured_ca_runtime_failure_retries_isolated_ca_not_da() {
        let result = TaskResult {
            task_iri: "iri://task/branch-test".to_string(),
            status: "failed".to_string(),
            verdict: Some(TaskVerdict::Failed),
            summary: "provider transport failed before CA produced a verdict".to_string(),
            output: None,
            jsonld_output: None,
            artifacts: vec![],
            errors: vec!["connection reset".to_string()],
            turn_count: 1,
            tool_call_count: 0,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: None,
        };

        let (terminated, directive, ca_recorded, repeats) =
            route_check_result_for_test(result).await;
        assert!(!terminated, "generated PDCA must retain CA-owned recovery");
        assert_eq!(
            directive,
            Some(crate::core::recovery::RecoveryDirective::RetryCa)
        );
        assert!(
            ca_recorded,
            "the failed verification receipt must be retained"
        );
        assert_eq!(repeats, 1);
    }

    #[tokio::test]
    async fn branch_on_failure_skips_intermediates_to_fallback() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let (dag, order, plan) = branch_fixture();
        let step_2_idx = *dag.node_index.get("step_2").unwrap();
        let step = crate::core::workflow::adapter::node_to_planstep(&dag.graph[step_2_idx].def);
        assert!(step.branch_on_failure);
        assert_eq!(step.branch_fallback.as_deref(), Some("step_4"));

        let mut prev_summary = None;
        let mut latest_pa_handoff = None;
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
        let step_2_wave = order.iter().position(|idx| *idx == step_2_idx).unwrap();
        let mut task_constraints = std::collections::HashMap::new();
        let mut conformance_contract = None;
        let mut recursive_budget = super::execution::RecursiveExecutionBudget::new(1, 1);

        let outcome = sa
            .handle_step_result(
                failed_result(),
                step,
                step_2_idx,
                step_2_wave,
                &mut prev_summary,
                &mut latest_pa_handoff,
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
                "branch test",
                "cycle-branch",
                &plan,
                &dag,
                &order,
                "iri://task/branch-test/5w2h",
                &crate::core::effect::EffectPolicy::None,
                &mut task_constraints,
                &mut conformance_contract,
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
        let restored = crate::core::checkpoint::CheckpointManager::with_persistence(
            sa.runner.l0_store.clone(),
        )
        .restore_task("iri://task/branch-test")
        .unwrap()
        .expect("branch decision must have a durable SA boundary");
        assert!(restored.messages.is_empty());
        assert!(restored.state.active_continuation.is_none());
        assert!(restored.state.completed_nodes.contains_key("step_2"));
        assert!(restored.state.skipped_nodes.contains("step_3"));
        assert!(!restored.state.skipped_nodes.contains("step_4"));
    }

    #[tokio::test]
    async fn failed_without_branch_aborts_plan() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let (dag, order, plan) = branch_fixture();
        let step_1_idx = *dag.node_index.get("step_1").unwrap();
        let step = crate::core::workflow::adapter::node_to_planstep(&dag.graph[step_1_idx].def);
        assert!(!step.branch_on_failure);

        let mut prev_summary = None;
        let mut latest_pa_handoff = None;
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
        let step_1_wave = order.iter().position(|idx| *idx == step_1_idx).unwrap();
        let mut task_constraints = std::collections::HashMap::new();
        let mut conformance_contract = None;
        let mut recursive_budget = super::execution::RecursiveExecutionBudget::new(1, 1);

        let outcome = sa
            .handle_step_result(
                failed_result(),
                step,
                step_1_idx,
                step_1_wave,
                &mut prev_summary,
                &mut latest_pa_handoff,
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
                "branch test",
                "cycle-branch",
                &plan,
                &dag,
                &order,
                "iri://task/branch-test/5w2h",
                &crate::core::effect::EffectPolicy::None,
                &mut task_constraints,
                &mut conformance_contract,
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
    fn ca_review_subject_contains_da_deliverable_but_not_da_success_claims() {
        let aggregate_iri = "iri://task/direct-review/session/l1_da_aggregate/turn_2".to_string();
        let result = TaskResult {
            task_iri: "iri://task/direct-review".to_string(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "SUCCESS_HINT_SHOULD_NOT_REACH_CA".to_string(),
            output: Some(serde_json::Value::String(
                "PASS_WORD_IS_PART_OF_DELIVERABLE_CONTENT".to_string(),
            )),
            jsonld_output: None,
            artifacts: vec![serde_json::json!({"path": "report.md"})],
            errors: vec![],
            turn_count: 2,
            tool_call_count: 1,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: Some(aggregate_iri.clone()),
        };

        let subject = super::execution::execution_subject_handoff(&result, 6_000)
            .expect("reviewable DA output");
        assert!(subject.contains("PASS_WORD_IS_PART_OF_DELIVERABLE_CONTENT"));
        assert!(subject.contains("report.md"));
        assert!(subject.contains(&aggregate_iri));
        assert!(subject.starts_with("## Stable DA Aggregate Output Capability"));
        assert!(!subject.contains("SUCCESS_HINT_SHOULD_NOT_REACH_CA"));
        assert!(!subject.contains("status: success"));
    }

    #[test]
    fn bounded_ca_handoff_keeps_only_the_typed_parent_aggregate_read_capability() {
        let parent_aggregate =
            "iri://task/parallel-da/session/l1_da_parent_aggregate/turn_7".to_string();
        let child_turn =
            "iri://task/parallel-da/subtask/research/session/l1_child/turn_3".to_string();
        let child_task = "iri://task/parallel-da/subtask/research".to_string();
        let result = TaskResult {
            task_iri: "iri://task/parallel-da".to_string(),
            status: "success".to_string(),
            verdict: Some(TaskVerdict::Success),
            summary: "model success claim".to_string(),
            output: Some(serde_json::Value::String(
                "large-deliverable-".repeat(2_000),
            )),
            jsonld_output: None,
            artifacts: vec![serde_json::json!({
                "type": "biz_agent_work_package_order_receipt",
                "child_task_iri": child_task,
                "child_archive_iri": child_turn,
                "executions": [{
                    "child_task_iri": "iri://task/parallel-da/subtask/research",
                    "status": "success"
                }]
            })],
            errors: vec![],
            turn_count: 7,
            tool_call_count: 2,
            five_w2h_updates: None,
            tracked_actions: Vec::new(),
            archive_iri: Some(parent_aggregate.clone()),
        };

        let handoff = super::execution::execution_subject_handoff(&result, 120)
            .expect("the kernel aggregate is reviewable");
        assert!(handoff.starts_with("## Stable DA Aggregate Output Capability"));
        assert!(handoff.contains(&parent_aggregate));
        assert!(handoff.contains("only cross-agent AgentTurn read target"));
        assert!(handoff.contains("not readable AgentTurn targets"));
        assert!(handoff.contains("## Kernel Work-Package Order Receipt (complete)"));
        assert!(!handoff.contains("large-deliverable-"));

        let context = TaskContext::new("iri://task/parallel-da", "verify", 4)
            .with_execution_handoff(handoff, parent_aggregate.clone());
        let security = context.tool_security_context("ca-child", "CA", "l1_fresh_ca");
        assert!(security.permits_agent_turn_read(&parent_aggregate));
        assert!(!security.permits_agent_turn_read(&child_turn));
        assert!(!security.permits_agent_turn_read(&child_task));
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
    fn direct_response_recheck_uses_the_agent_output_reader_and_required_live_retrieval() {
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

        let mut research_constraints = constraints.clone();
        research_constraints.insert(
            crate::core::agent_runner::REQUIRED_CAPABILITY_CONSTRAINT.to_string(),
            crate::core::agent_runner::REQUIRED_CAPABILITY_WEB_RESEARCH.to_string(),
        );
        assert_eq!(
            super::execution::direct_response_recheck_tools(&research_constraints),
            Some(vec![
                "read_agent_output".to_string(),
                "web_search".to_string(),
                "web_fetch".to_string(),
            ])
        );

        research_constraints.insert(
            crate::core::agent_runner::REQUIRED_VALIDATION_CONSTRAINT.to_string(),
            crate::core::agent_runner::REQUIRED_VALIDATION_MERMAID.to_string(),
        );
        assert_eq!(
            super::execution::direct_response_recheck_tools(&research_constraints),
            Some(vec![
                "read_agent_output".to_string(),
                "web_search".to_string(),
                "web_fetch".to_string(),
                "mermaid_validate".to_string(),
            ])
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
            work_packages: Vec::new(),
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
            agent_spec_provenance: None,
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
    fn retry_da_without_failed_step_never_restarts_completed_pa() {
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
            agent_spec_provenance: None,
            context_requirements: HashMap::new(),
            success_metrics: Vec::new(),
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: Vec::new(),
        };
        let decision = crate::core::recovery::DecisionReport {
            mode: crate::core::recovery::OrchestrationMode::Pdca,
            directive: crate::core::recovery::RecoveryDirective::RetryDa,
            reason: crate::core::recovery::RecoveryReason::LocalExecutionGap,
            scope: crate::core::recovery::RepairScope::Step,
            plan_revision: 1,
        };

        let delta = super::process::scoped_retry_plan_for_decision(&plan, &decision, None).unwrap();

        assert_eq!(
            delta
                .steps
                .iter()
                .map(|step| (step.step_id.as_str(), step.role))
                .collect::<Vec<_>>(),
            vec![
                ("da", AgentRole::Do),
                ("ca", AgentRole::Check),
                ("aa", AgentRole::Act),
            ]
        );
        assert!(delta.steps.iter().all(|step| step.role != AgentRole::Plan));
    }

    #[test]
    fn retry_da_for_failed_check_reenters_nearest_preceding_da() {
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
            agent_spec_provenance: None,
            context_requirements: HashMap::new(),
            success_metrics: Vec::new(),
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: Vec::new(),
        };
        let decision = crate::core::recovery::DecisionReport {
            mode: crate::core::recovery::OrchestrationMode::Pdca,
            directive: crate::core::recovery::RecoveryDirective::RetryDa,
            reason: crate::core::recovery::RecoveryReason::LocalExecutionGap,
            scope: crate::core::recovery::RepairScope::Step,
            plan_revision: 3,
        };

        let delta =
            super::process::scoped_retry_plan_for_decision(&plan, &decision, Some("ca")).unwrap();

        assert_eq!(delta.steps[0].role, AgentRole::Do);
        assert_eq!(delta.steps[0].step_id, "da");
        assert_eq!(delta.plan_id, "p_delta_4");
    }

    #[test]
    fn retry_ca_for_missing_evidence_preserves_completed_implementation() {
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
            agent_spec_provenance: None,
            context_requirements: HashMap::new(),
            success_metrics: Vec::new(),
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: Vec::new(),
        };
        let decision = crate::core::recovery::DecisionReport {
            mode: crate::core::recovery::OrchestrationMode::Pdca,
            directive: crate::core::recovery::RecoveryDirective::RetryCa,
            reason: crate::core::recovery::RecoveryReason::EvidenceMissing,
            scope: crate::core::recovery::RepairScope::Phase,
            plan_revision: 4,
        };

        let delta =
            super::process::scoped_retry_plan_for_decision(&plan, &decision, Some("ca")).unwrap();

        assert_eq!(
            delta
                .steps
                .iter()
                .map(|step| (step.step_id.as_str(), step.role))
                .collect::<Vec<_>>(),
            vec![("ca", AgentRole::Check), ("aa", AgentRole::Act)]
        );
        assert!(delta.steps.iter().all(|step| step.role != AgentRole::Do));
        assert_eq!(delta.plan_id, "p_delta_5");
    }

    #[test]
    fn retry_ca_rejects_a_downstream_mutation_boundary() {
        let plan = ExecutionPlan {
            plan_id: "mixed".to_string(),
            agent_sequence: vec![
                AgentRole::Plan,
                AgentRole::Do,
                AgentRole::Check,
                AgentRole::Do,
                AgentRole::Check,
                AgentRole::Act,
            ],
            parallel_groups: Vec::new(),
            task_complexity: TaskComplexity::Simple,
            description: String::new(),
            steps: vec![
                recovery_step("pa", AgentRole::Plan, &[]),
                recovery_step("da1", AgentRole::Do, &["pa"]),
                recovery_step("ca1", AgentRole::Check, &["da1"]),
                recovery_step("da2", AgentRole::Do, &["ca1"]),
                recovery_step("ca2", AgentRole::Check, &["da2"]),
                recovery_step("aa", AgentRole::Act, &["ca2"]),
            ],
            agent_spec_provenance: None,
            context_requirements: HashMap::new(),
            success_metrics: Vec::new(),
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: Vec::new(),
        };
        let decision = crate::core::recovery::DecisionReport {
            mode: crate::core::recovery::OrchestrationMode::Pdca,
            directive: crate::core::recovery::RecoveryDirective::RetryCa,
            reason: crate::core::recovery::RecoveryReason::EvidenceMissing,
            scope: crate::core::recovery::RepairScope::Phase,
            plan_revision: 1,
        };

        assert!(
            super::process::scoped_retry_plan_for_decision(&plan, &decision, Some("ca1")).is_none()
        );
    }

    #[test]
    fn scoped_recovery_excludes_later_non_descendant_steps() {
        let plan = ExecutionPlan {
            plan_id: "branched".to_string(),
            agent_sequence: vec![AgentRole::Do, AgentRole::Do, AgentRole::Check],
            parallel_groups: Vec::new(),
            task_complexity: TaskComplexity::Simple,
            description: String::new(),
            steps: vec![
                recovery_step("da_failed", AgentRole::Do, &[]),
                recovery_step("unrelated_da", AgentRole::Do, &[]),
                recovery_step("ca_failed", AgentRole::Check, &["da_failed"]),
            ],
            agent_spec_provenance: None,
            context_requirements: HashMap::new(),
            success_metrics: Vec::new(),
            max_recursion_depth: 0,
            sub_tasks: Vec::new(),
            dag_jsonld: None,
            verify_first: false,
            fallback_steps: Vec::new(),
        };

        let delta = super::process::scoped_recovery_plan(&plan, "da_failed", 2).unwrap();
        assert_eq!(
            delta
                .steps
                .iter()
                .map(|step| step.step_id.as_str())
                .collect::<Vec<_>>(),
            vec!["da_failed", "ca_failed"]
        );
    }

    #[test]
    fn only_kernel_owned_fields_can_authorize_ca_only_recovery() {
        let mut result = failed_result();
        result.summary =
            "model says [Recovery] directive=RetryDa scope=Step failed_step=da_spoof".to_string();
        result.errors.clear();
        assert_eq!(super::process::kernel_recovery_directive(&result), None);

        result
            .errors
            .push("SA kernel recovery route: directive=RetryCa;failed_step=ca".to_string());
        assert_eq!(
            super::process::kernel_recovery_directive(&result),
            Some(crate::core::recovery::RecoveryDirective::RetryCa)
        );
        assert_eq!(
            super::process::kernel_recovery_route(&result).map(|(_, failed_step)| failed_step),
            Some("ca".to_string())
        );
    }

    #[test]
    fn task_facts_preserve_blocked_route_added_after_ca_was_recorded() {
        let mut result = failed_result();
        result.errors = vec!["CA verification evidence is incomplete".to_string()];
        result.artifacts = vec![serde_json::json!({"path": "calculator_project"})];

        // This is the production ordering: the CA payload is accumulated
        // first, then SA assigns the bounded-recovery terminal directive.
        let mut facts = super::execution::TaskExecutionFacts::default();
        facts.record(&result);
        result.summary.push_str(
            "\n\n[Recovery] directive=Blocked scope=Phase reason=EvidenceMissing failed_step=step_3",
        );
        result
            .errors
            .push("SA kernel recovery route: directive=Blocked;failed_step=step_3".to_string());

        facts.apply_to(&mut result);

        assert_eq!(
            super::process::kernel_recovery_route(&result),
            Some((
                crate::core::recovery::RecoveryDirective::Blocked,
                "step_3".to_string(),
            )),
            "task aggregation must not erase a kernel terminal directive and fall back to DA"
        );
        assert_eq!(
            result
                .errors
                .iter()
                .filter(|error| error.as_str() == "CA verification evidence is incomplete")
                .count(),
            1,
            "pre-terminal facts must remain de-duplicated"
        );
        assert_eq!(result.artifacts.len(), 1);
    }

    #[test]
    fn step_checkpoint_counters_include_all_preceding_role_results() {
        fn counted_result(turn_count: u32, tool_call_count: u32) -> TaskResult {
            TaskResult {
                task_iri: "iri://task/checkpoint-counts".to_string(),
                status: "success".to_string(),
                summary: String::new(),
                output: None,
                jsonld_output: None,
                artifacts: Vec::new(),
                errors: Vec::new(),
                turn_count,
                tool_call_count,
                five_w2h_updates: None,
                tracked_actions: Vec::new(),
                verdict: Some(TaskVerdict::Success),
                archive_iri: None,
            }
        }

        let mut facts = super::execution::TaskExecutionFacts::default();
        facts.record(&counted_result(4, 2));
        facts.record(&counted_result(7, 5));
        let state: serde_json::Value =
            serde_json::from_str(&facts.checkpoint_agent_state_json(101, 37)).unwrap();

        assert_eq!(state["turn"], 11);
        assert_eq!(state["tc"], 7);
        assert_eq!(state["prompt_tokens"], 101);
        assert_eq!(state["completion_tokens"], 37);
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
            agent_spec_provenance: None,
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

    #[test]
    fn supplementary_workspace_delivery_replaces_direct_response_contract() {
        let mut constraints = HashMap::from([
            (
                crate::core::agent_runner::WORKSPACE_CONTEXT_SCOPE_CONSTRAINT.to_string(),
                crate::core::agent_runner::WORKSPACE_CONTEXT_DISABLED.to_string(),
            ),
            (
                crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT.to_string(),
                crate::core::agent_runner::DELIVERY_MODE_DIRECT_RESPONSE.to_string(),
            ),
        ]);
        let mut effect_policy = crate::core::effect::EffectPolicy::EvidenceOnly;

        super::execution::apply_workspace_delivery_contract(
            &mut constraints,
            &mut effect_policy,
            "AI_Agent_Research_Report.md",
        );

        assert!(!constraints
            .contains_key(crate::core::agent_runner::WORKSPACE_CONTEXT_SCOPE_CONSTRAINT));
        assert_eq!(
            constraints
                .get(crate::core::agent_runner::DELIVERY_MODE_CONSTRAINT)
                .map(String::as_str),
            Some(crate::core::agent_runner::DELIVERY_MODE_WORKSPACE_ARTIFACT)
        );
        assert_eq!(
            constraints
                .get(crate::core::agent_runner::DELIVERY_TARGET_PATH_CONSTRAINT)
                .map(String::as_str),
            Some("AI_Agent_Research_Report.md")
        );
        assert_eq!(
            effect_policy,
            crate::core::effect::EffectPolicy::required_workspace_mutation()
        );
    }

    #[test]
    fn workspace_delivery_input_is_detected_without_an_llm_classifier() {
        assert_eq!(
            super::intervention::workspace_delivery_target_from_supplement("文件输出到当前工作区")
                .as_deref(),
            Some(super::intervention::DEFAULT_WORKSPACE_DELIVERY_PATH)
        );
        assert_eq!(
            super::intervention::workspace_delivery_target_from_supplement(
                "请输出到当前工作区的 final-report.md"
            )
            .as_deref(),
            Some("final-report.md")
        );
        assert!(
            super::intervention::workspace_delivery_target_from_supplement("继续检索最新的趋势")
                .is_none()
        );
    }

    #[tokio::test]
    async fn planning_time_event_drain_preserves_workspace_delivery_command() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let task_iri = "iri://task/planning-supplement";
        sa.event_bus
            .emit(
                task_iri,
                "USER_SUPPLEMENTARY_INPUT",
                "code_cli",
                "文件输出到当前工作区",
            )
            .await;

        // This is the exact boundary that previously swallowed the command
        // immediately after plan creation.
        sa.drain_and_route_runtime_events(task_iri).await;
        assert_eq!(
            sa.supplementary_inputs
                .get(task_iri)
                .map(Vec::len)
                .unwrap_or_default(),
            1,
            "reliable inbox and broadcast copy must not enqueue twice"
        );
        let outcome = sa
            .check_and_process_supplementary_inputs(
                task_iri,
                &AgentRole::Plan,
                "planning has completed",
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.workspace_delivery_target.as_deref(),
            Some(super::intervention::DEFAULT_WORKSPACE_DELIVERY_PATH)
        );
        assert!(sa
            .supplement_store
            .snapshot_pending(task_iri)
            .iter()
            .any(|entry| entry.content.contains("Mandatory delivery update")));
        sa.drain_and_route_runtime_events(task_iri).await;
        assert!(
            !sa.supplementary_inputs.contains_key(task_iri),
            "processed plaintext must be removed from the transient queue"
        );
        assert_eq!(
            sa.supplement_store.snapshot_pending(task_iri).len(),
            1,
            "a second drain must not duplicate the durable supplement"
        );
    }

    #[tokio::test]
    async fn paused_cycle_resumes_from_one_deterministic_user_command() {
        let (mut sa, _dir) = make_sa_with_tempdir();
        let task_iri = "iri://task/paused-resume";
        let cycle_id = "cycle-paused-resume";
        let now = chrono::Utc::now();
        sa.active_cycles.insert(
            cycle_id.to_string(),
            CycleState {
                cycle_id: cycle_id.to_string(),
                task_iri: task_iri.to_string(),
                phase: CyclePhase::Idle,
                iteration: 1,
                max_iterations: 10,
                started_at: now,
                pdca_started_at: now,
                cycle_deadline_at: now + chrono::Duration::minutes(5),
                last_progress_at: now,
                last_timeout_alert_at: None,
                next_timeout_alert_at: None,
                timeout_alert_count: 0,
                outer_cycle_number: 1,
                phase_history: Vec::new(),
                task_completed: false,
                observed_experience_hint_count: 0,
                observed_experience_hint_fingerprints: Vec::new(),
                experience_hints: Vec::new(),
                intervention: InterventionState::default(),
            },
        );
        sa.event_bus
            .emit(task_iri, "USER_SUPPLEMENTARY_INPUT", "code_cli", "继续")
            .await;

        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            sa.check_and_process_supplementary_inputs(
                task_iri,
                &AgentRole::Do,
                "waiting for resume",
            ),
        )
        .await
        .expect("deterministic resume must not wait for an LLM")
        .unwrap();
        assert!(matches!(
            sa.active_cycles.get(cycle_id).map(|cycle| &cycle.phase),
            Some(CyclePhase::Executing)
        ));
        assert!(super::intervention::deterministic_execution_control("继续深入分析").is_none());
    }
}
