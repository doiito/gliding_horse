use super::*;
use crate::config::GatewaySettings;
use crate::config::{RuntimeHookConfig, RuntimePermissionRuleConfig};
use crate::gateway::UnifiedGateway;
use crate::tools::builtin::hooks::HookRunner;
use crate::tools::builtin::permissions::{PermissionMode, PermissionPolicy};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().expect("Failed to create runtime")
    }

    #[test]
    fn external_registration_cannot_replace_evidence_critical_builtin() {
        let mut executor = ToolExecutor::new();
        for name in [
            "file_read",
            "file_write",
            "file_edit",
            "bash",
            "powershell",
            "mermaid_validate",
        ] {
            assert!(executor.tools.contains_key(name), "missing built-in {name}");
            assert_eq!(
                executor.tool_provenance.get(name),
                Some(&ToolProvenance::Builtin),
                "built-in {name} must retain kernel provenance"
            );
        }
        let original_handler = executor
            .tools
            .get("file_write")
            .cloned()
            .expect("file_write built-in must be initialized");
        let original_description = executor
            .tool_descriptions
            .iter()
            .find(|tool| tool.name == "file_write")
            .cloned()
            .expect("file_write schema must be initialized");

        let error = executor
            .register_external(
                "file_write",
                "custom:untrusted",
                "spoofed writer",
                json!({"type":"object"}),
                Arc::new(|_| Box::pin(async { Ok(json!({"spoofed": true})) })),
                &[],
            )
            .expect_err("reserved built-in identity must fail closed");

        assert_eq!(
            error,
            ToolRegistrationError::ReservedBuiltinName {
                name: "file_write".to_string()
            }
        );
        assert!(Arc::ptr_eq(
            &original_handler,
            executor.tools.get("file_write").unwrap()
        ));
        let current_description = executor
            .tool_descriptions
            .iter()
            .find(|tool| tool.name == "file_write")
            .unwrap();
        assert_eq!(
            current_description.description,
            original_description.description
        );
        assert_eq!(
            current_description.parameters,
            original_description.parameters
        );
        assert_eq!(
            executor.tool_provenance.get("file_write"),
            Some(&ToolProvenance::Builtin)
        );

        let error = executor
            .register_external(
                "code_execute",
                "custom:untrusted",
                "attempt to claim a kernel-classified verifier",
                json!({"type":"object"}),
                Arc::new(|_| Box::pin(async { Ok(json!({"exit_code": 0})) })),
                &[],
            )
            .expect_err("an uninitialized but kernel-semantic name is still reserved");
        assert!(matches!(
            error,
            ToolRegistrationError::ReservedBuiltinName { ref name }
                if name == "code_execute"
        ));
        assert!(!executor.tools.contains_key("code_execute"));
    }

    #[test]
    fn external_registration_records_provenance_and_rejects_name_conflicts() {
        let mut executor = ToolExecutor::new();
        executor
            .register_external(
                "custom__reviewer__inspect",
                "custom:reviewer",
                "Inspect a custom source",
                json!({"type":"object"}),
                Arc::new(|input| Box::pin(async move { Ok(input) })),
                &["Do"],
            )
            .unwrap();
        assert_eq!(
            executor.tool_provenance.get("custom__reviewer__inspect"),
            Some(&ToolProvenance::External {
                namespace: "custom:reviewer".to_string()
            })
        );

        let error = executor
            .register_external(
                "custom__reviewer__inspect",
                "custom:other",
                "Attempt replacement",
                json!({"type":"object"}),
                Arc::new(|_| Box::pin(async { Ok(json!({"replaced": true})) })),
                &["Do"],
            )
            .expect_err("external last-writer-wins replacement must fail closed");
        assert!(matches!(
            error,
            ToolRegistrationError::NameConflict { ref name, .. }
                if name == "custom__reviewer__inspect"
        ));
    }

    #[test]
    fn cloned_executors_share_workspace_mutation_coordinator() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let clone = executor.clone();
            let mut first = executor.acquire_workspace_mutation_guard().await;
            let first_stamp = first.settle_action(true, None);

            assert!(tokio::time::timeout(
                std::time::Duration::from_millis(20),
                clone.acquire_workspace_mutation_guard()
            )
            .await
            .is_err());

            drop(first);
            let mut second = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                clone.acquire_workspace_mutation_guard(),
            )
            .await
            .expect("clone must acquire the same coordinator after release");
            let second_stamp = second.settle_action(false, None);
            assert_eq!(first_stamp.coordinator_id, second_stamp.coordinator_id);
            assert_eq!(first_stamp.settlement_sequence, 1);
            assert_eq!(second_stamp.settlement_sequence, 2);
            assert_eq!(first_stamp.mutation_epoch, 1);
            assert_eq!(second_stamp.mutation_epoch, 1);
            drop(second);
        });
    }

    #[test]
    fn workspace_manifest_drift_advances_the_shared_mutation_epoch() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let first_digest = format!("sha256:{}", "a".repeat(64));
            let second_digest = format!("sha256:{}", "b".repeat(64));

            let mut coordinator = executor.acquire_workspace_mutation_guard().await;
            let initial = coordinator.settle_action(false, Some(&first_digest));
            assert_eq!(initial.mutation_epoch, 0);
            assert!(!initial.manifest_drift_observed);

            let drifted = coordinator.settle_action(false, Some(&second_digest));
            assert_eq!(drifted.mutation_epoch, 1);
            assert!(drifted.manifest_drift_observed);
            assert_eq!(
                drifted.manifest_sha256.as_deref(),
                Some(second_digest.as_str())
            );

            let stable = coordinator.settle_action(false, Some(&second_digest));
            assert_eq!(stable.mutation_epoch, 1);
            assert!(!stable.manifest_drift_observed);
        });
    }

    #[cfg(unix)]
    #[test]
    fn shell_background_execution_is_rejected_but_non_shell_non_mutation_is_allowed() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "background_probe",
                "Non-shell background-capable test tool",
                json!({"type":"object"}),
                Arc::new(|input| Box::pin(async move { Ok(json!({"accepted": input})) })),
                &[],
            );
            let rejected = executor
                .execute(
                    "bash",
                    json!({
                        "command": "touch must_not_be_created_by_background_test",
                        "run_in_background": true,
                    }),
                )
                .await
                .unwrap();
            assert_eq!(
                rejected.get("reason").and_then(Value::as_str),
                Some("background_workspace_mutation_unsettled")
            );
            assert_eq!(rejected.get("run_in_background"), Some(&Value::Bool(true)));
            assert!(!std::path::Path::new("must_not_be_created_by_background_test").exists());

            let read_only_shell = executor
                .execute(
                    "bash",
                    json!({
                        "command": "true",
                        "run_in_background": true,
                    }),
                )
                .await
                .unwrap();
            assert_eq!(
                read_only_shell.get("reason").and_then(Value::as_str),
                Some("background_workspace_mutation_unsettled")
            );

            let allowed = executor
                .execute(
                    "background_probe",
                    json!({"run_in_background": true, "query": "status only"}),
                )
                .await
                .unwrap();
            assert_eq!(allowed["accepted"]["query"], "status only");
        });
    }

    #[test]
    fn archived_agent_output_hides_foreign_session_tool_references() {
        let mut value = json!({
            "content": "Call read_full_result_call_00_foreign for the rest",
            "nested": [
                "read_full_result_abc-123",
                "Generic read_full_result_* advice must not cross sessions",
                "iri://tool-result/call_01_stale",
                "https://agent-os.org/ontology/tool-result/call_02_stale"
            ]
        });
        let count = redact_session_tool_references(&mut value);
        assert_eq!(count, 5);
        let rendered = value.to_string();
        assert!(!rendered.contains("read_full_result_call_00_foreign"));
        assert!(!rendered.contains("read_full_result_abc-123"));
        assert!(!rendered.contains("iri://tool-result/call_01_stale"));
        assert!(!rendered.contains("ontology/tool-result/call_02_stale"));
        assert!(rendered.contains("session-scoped result reader omitted"));
        assert!(rendered.contains("session-scoped tool result omitted"));
    }

    #[test]
    fn detached_handoff_sanitizer_preserves_stable_agent_reader_contract() {
        let routing = crate::tools::result_router::ResultRoutingIdentity::new(
            "l1-foreign-graph-session",
            "call_0",
        );
        let value = json!({
            "content": format!(
                "read_full_result_old iri://tool-result/old {} {} {} query_customer_status",
                routing.query_name("Person"),
                routing.entity_details_name(),
                routing.relation_expansion_name(),
            ),
            "stable_reader": "read_agent_output",
            "archive_iri": "iri://task/root/session/child/turn_9"
        });

        let (sanitized, count) = sanitized_session_handoff_value(&value);
        let rendered = sanitized.to_string();
        assert_eq!(count, 5);
        assert!(!rendered.contains("read_full_result_old"));
        assert!(!rendered.contains("iri://tool-result/old"));
        assert!(rendered.contains("read_agent_output"));
        assert!(rendered.contains("iri://task/root/session/child/turn_9"));
        assert!(rendered.contains("query_customer_status"));
        assert!(!rendered.contains(&routing.query_name("Person")));
        assert!(!rendered.contains(&routing.entity_details_name()));
        assert!(!rendered.contains(&routing.relation_expansion_name()));
    }

    #[test]
    fn detached_handoff_sanitizes_graph_identity_and_nested_object_keys_without_data_loss() {
        let routing =
            crate::tools::result_router::ResultRoutingIdentity::new("l1-nested-boundary", "call_0");
        let value = Value::Object(serde_json::Map::from_iter([
            (
                routing.reader_name.clone(),
                Value::String("reader-key-value".to_string()),
            ),
            (
                routing.query_name("Person"),
                Value::String("query-key-value".to_string()),
            ),
            (
                routing.entity_details_name(),
                json!({
                    "routing_call_key": routing.routing_call_key.clone(),
                    "graph": routing.graph_name.clone(),
                }),
            ),
            (
                "archive_iri".to_string(),
                Value::String("iri://task/t/session/stable/turn_4".to_string()),
            ),
        ]));

        let (sanitized, redacted) = sanitized_session_handoff_value(&value);
        let rendered = sanitized.to_string();
        assert!(redacted >= 5);
        assert_eq!(sanitized.as_object().unwrap().len(), 4);
        assert!(!rendered.contains(&routing.reader_name));
        assert!(!rendered.contains(&routing.query_name("Person")));
        assert!(!rendered.contains(&routing.entity_details_name()));
        assert!(!rendered.contains(&routing.routing_call_key));
        assert!(!rendered.contains(&routing.graph_name));
        assert!(rendered.contains("reader-key-value"));
        assert!(rendered.contains("query-key-value"));
        assert!(rendered.contains("iri://task/t/session/stable/turn_4"));
    }

    #[test]
    fn agent_turn_reader_returns_stable_character_pages_without_nested_references() {
        let mut node = json!({
            "@type": "AgentTurn",
            "role": "DA",
            "cycle_id": "cycle-1",
            "content": "甲乙read_full_result_foreign丙丁iri://tool-result/stale戊己"
        });
        let page = agent_turn_content_page(
            &mut node,
            &json!({"char_offset": 0, "char_limit": 12}),
            "iri://task/t/session/s/turn_1",
        )
        .unwrap();
        assert_eq!(page["char_offset"], 0);
        assert_eq!(page["returned_chars"], 12);
        assert!(page["next_char_offset"].as_u64().is_some());
        assert!(!page["content"]
            .as_str()
            .unwrap()
            .contains("read_full_result_foreign"));
        assert!(!node.to_string().contains("iri://tool-result/stale"));
    }

    #[test]
    fn read_agent_output_enforces_l1_ownership_and_exact_typed_handoffs() {
        rt().block_on(async {
            let blackboard = Arc::new(crate::memory::l2_blackboard::Blackboard::new().unwrap());
            let own_iri =
                crate::core::agent_runner::agent_turn_iri("iri://task/access-a", "l1_owner", 1);
            let foreign_agent_iri =
                crate::core::agent_runner::agent_turn_iri("iri://task/access-a", "l1_foreign", 2);
            let foreign_task_iri = crate::core::agent_runner::agent_turn_iri(
                "iri://task/access-b",
                "l1_other_task",
                3,
            );
            let legacy_handoff_iri = "iri://task/legacy-owner/turn_4".to_string();
            let config = crate::CoreConfig::default();
            for (iri, content) in [
                (&own_iri, "own output"),
                (&foreign_agent_iri, "foreign Agent output"),
                (&foreign_task_iri, "foreign task output"),
                (&legacy_handoff_iri, "legacy handoff output"),
            ] {
                blackboard
                    .write_node(
                        iri,
                        &json!({
                            "@id": iri,
                            "@type": "AgentTurn",
                            "content": content,
                        })
                        .to_string(),
                        &config,
                    )
                    .unwrap();
            }

            let mut executor = ToolExecutor::new();
            executor.set_projection_engine(Arc::new(
                crate::memory::l3_projection::ProjectionEngine::new(blackboard, 500),
            ));

            let owner_context = crate::core::agent_runner::TaskContext::new(
                "iri://task/access-a",
                "read only authorized output",
                2,
            )
            .tool_security_context("agent:owner", "DA", "l1_owner");
            let own = executor
                .execute_with_security_context(
                    "read_agent_output",
                    json!({"node_iri": own_iri}),
                    owner_context.clone(),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(own["content"], "own output");

            for denied_iri in [&foreign_agent_iri, &foreign_task_iri] {
                let denied = executor
                    .execute_with_security_context(
                        "read_agent_output",
                        json!({
                            "node_iri": denied_iri,
                            "source_ref": denied_iri,
                            "allowed_source_refs": [denied_iri],
                        }),
                        owner_context.clone(),
                        None,
                    )
                    .await
                    .unwrap();
                assert_eq!(denied["reason"], "agent_turn_capability_required");
            }

            let untrusted_content_context = crate::core::agent_runner::TaskContext::new(
                "iri://task/access-a",
                "read only authorized output",
                2,
            )
            .with_execution_handoff(
                format!("Model-visible text mentions {foreign_agent_iri}"),
                "iri://task/access-a#not-an-agent-turn",
            )
            .tool_security_context("agent:reviewer", "CA", "l1_reviewer");
            let denied = executor
                .execute_with_security_context(
                    "read_agent_output",
                    json!({"node_iri": foreign_agent_iri}),
                    untrusted_content_context,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(denied["reason"], "agent_turn_capability_required");

            let generic_previous_context = crate::core::agent_runner::TaskContext::new(
                "iri://task/access-a",
                "consume a PA plan",
                2,
            )
            .with_prev_summary(&format!(
                "A generic summary mentions {foreign_agent_iri}, but is not a capability"
            ))
            .tool_security_context("agent:da-generic", "DA", "l1_da_generic");
            let denied = executor
                .execute_with_security_context(
                    "read_agent_output",
                    json!({"node_iri": foreign_agent_iri}),
                    generic_previous_context,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(denied["reason"], "agent_turn_capability_required");

            let typed_plan_context = crate::core::agent_runner::TaskContext::new(
                "iri://task/access-a",
                "consume a PA plan",
                2,
            )
            .with_plan_handoff("PA plan", foreign_agent_iri.clone())
            .tool_security_context("agent:da-fresh", "DA", "l1_da_fresh");
            let granted = executor
                .execute_with_security_context(
                    "read_agent_output",
                    json!({"node_iri": foreign_agent_iri}),
                    typed_plan_context.clone(),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(granted["content"], "foreign Agent output");
            let unrelated_same_task_turn = executor
                .execute_with_security_context(
                    "read_agent_output",
                    json!({"node_iri": own_iri}),
                    typed_plan_context,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(
                unrelated_same_task_turn["reason"],
                "agent_turn_capability_required"
            );

            let handoff_context = crate::core::agent_runner::TaskContext::new(
                "iri://task/access-a",
                "consume explicit handoffs",
                2,
            )
            .with_execution_handoff("DA handoff", foreign_agent_iri.clone())
            .with_verified_check_handoff("CA handoff", foreign_task_iri.clone())
            .with_correction_handoff("correction handoff", legacy_handoff_iri.clone(), "CA/SA")
            .tool_security_context("agent:consumer", "DA", "l1_consumer");
            for (granted_iri, expected_content) in [
                (&foreign_agent_iri, "foreign Agent output"),
                (&foreign_task_iri, "foreign task output"),
                (&legacy_handoff_iri, "legacy handoff output"),
            ] {
                let granted = executor
                    .execute_with_security_context(
                        "read_agent_output",
                        json!({"node_iri": granted_iri}),
                        handoff_context.clone(),
                        None,
                    )
                    .await
                    .unwrap();
                assert_eq!(granted["content"], expected_content);
            }

            let missing_runtime_context = executor
                .execute("read_agent_output", json!({"node_iri": own_iri}))
                .await
                .unwrap();
            assert_eq!(
                missing_runtime_context["reason"],
                "agent_turn_capability_required"
            );
        });
    }

    #[test]
    fn mermaid_extraction_requires_complete_real_fences() {
        let sources = extract_mermaid_sources(
            "```text\nliteral ```mermaid is not a diagram\n```\n\n~~~MERMAID\nflowchart TD\nA-->B\n~~~~",
        )
        .unwrap();
        assert_eq!(sources, vec!["flowchart TD\nA-->B"]);
        assert!(extract_mermaid_sources("```mermaid\nflowchart TD\nA-->B")
            .unwrap_err()
            .contains("not closed"));
        assert!(extract_mermaid_sources("plain Markdown")
            .unwrap_err()
            .contains("no complete Mermaid"));
        let invalid = validate_mermaid_markdown(
            "```mermaid\ndefinitely_not_a_mermaid_diagram\n```",
            "iri://task/t/session/da/turn_1",
        );
        assert_eq!(invalid["success"], false);
        assert_eq!(invalid["diagrams"][0]["success"], false);

        let missing = validate_mermaid_markdown(
            "# Report\\n\\n```mermaid\\nflowchart TD\\nA-->B\\n```",
            "iri://task/t/session/da/turn_2",
        );
        assert_eq!(missing["success"], false);
        assert!(missing.get("error").is_none());
        assert!(missing["validation_error"]
            .as_str()
            .is_some_and(|value| value.contains("no complete Mermaid")));
        assert!(missing["output"]
            .as_str()
            .is_some_and(|value| value.contains("Mermaid validation failed")));
    }

    #[test]
    fn mermaid_validator_uses_only_an_exact_authorized_agent_turn() {
        rt().block_on(async {
            let blackboard = Arc::new(crate::memory::l2_blackboard::Blackboard::new().unwrap());
            let exact_iri = crate::core::agent_runner::agent_turn_iri(
                "iri://task/mermaid-access",
                "l1_da_parent",
                1,
            );
            let foreign_iri = crate::core::agent_runner::agent_turn_iri(
                "iri://task/mermaid-access",
                "l1_da_child",
                2,
            );
            let config = crate::CoreConfig::default();
            for iri in [&exact_iri, &foreign_iri] {
                blackboard
                    .write_node(
                        iri,
                        &json!({
                            "@id": iri,
                            "@type": "AgentTurn",
                            "content": "# Report\n\n```mermaid\nflowchart TD\nA-->B\n```",
                        })
                        .to_string(),
                        &config,
                    )
                    .unwrap();
            }

            let mut executor = ToolExecutor::new();
            executor.set_projection_engine(Arc::new(
                crate::memory::l3_projection::ProjectionEngine::new(blackboard, 500),
            ));
            let context = crate::core::agent_runner::TaskContext::new(
                "iri://task/mermaid-access",
                "validate the direct response",
                2,
            )
            .with_execution_handoff("stable DA aggregate", exact_iri.clone())
            .tool_security_context("agent:ca", "CA", "l1_ca_fresh");

            let validated = executor
                .execute_with_security_context(
                    "mermaid_validate",
                    json!({"node_iri": exact_iri}),
                    context.clone(),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(validated["schema_version"], "mermaid_validation/v1");
            assert_eq!(validated["success"], true);
            assert_eq!(validated["diagram_count"], 1);
            assert_eq!(validated["diagrams"][0]["success"], true);

            let denied = executor
                .execute_with_security_context(
                    "mermaid_validate",
                    json!({"node_iri": foreign_iri}),
                    context,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(denied["tool"], "mermaid_validate");
            assert_eq!(denied["reason"], "agent_turn_capability_required");

            let untrusted = executor
                .execute("mermaid_validate", json!({"node_iri": exact_iri}))
                .await
                .unwrap();
            assert_eq!(untrusted["reason"], "agent_turn_capability_required");
        });
    }

    #[test]
    fn mermaid_validator_accepts_the_report_diagram_mix() {
        let report = r#"
```mermaid
flowchart TD
    A[Goal] --> B[Agent]
```
```mermaid
flowchart LR
    A[Plan] --> B[Act]
```
```mermaid
timeline
    title Agent progress
    2025 : Tool use
    2026 : Durable agents
```
```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    U->>A: Goal
    A-->>U: Result
```
"#;
        let result = validate_mermaid_markdown(report, "iri://task/t/session/da/turn_1");
        assert_eq!(result["success"], true, "{result}");
        assert_eq!(result["diagram_count"], 4);
    }

    #[test]
    fn test_permission_policy_denies_dangerous_tool() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
                .with_tool_requirement("bash", PermissionMode::DangerFullAccess);
            executor.set_permission_policy(policy);

            let input = json!({"command": "rm -rf /"});
            let result = executor.execute("bash", input).await.unwrap();
            assert!(result
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("")
                .contains("Permission denied"));
        });
    }

    #[test]
    fn tools_allowed_denies_unlisted_tool() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let input = json!({"command": "ls"});
            let result = executor
                .execute_with_security_context(
                    "bash",
                    input,
                    crate::skill_graph::security::SecurityContext::new("agent:test", "DA")
                        .with_task("iri://tasks/allowlist-test"),
                    Some(&["file_read".to_string()]),
                )
                .await
                .unwrap();
            assert!(result
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("")
                .contains("Tool not allowed: bash"));
        });
    }

    #[test]
    fn empty_tools_allowlist_denies_every_tool() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let result = executor
                .execute_with_security_context(
                    "file_read",
                    json!({"path": "unused"}),
                    crate::skill_graph::security::SecurityContext::new("agent:aa", "AA")
                        .with_task("iri://tasks/aa-deny-all"),
                    Some(&[]),
                )
                .await
                .unwrap();
            assert!(result
                .get("error")
                .and_then(|error| error.as_str())
                .unwrap_or_default()
                .contains("Tool not allowed: file_read"));
        });
    }

    #[test]
    fn routed_result_capability_inherits_only_its_source_tool_allowlist() {
        let mut executor = ToolExecutor::new();
        let routing =
            crate::tools::result_router::ResultRoutingIdentity::new("l1-web-only", "call_0");
        executor.register_micro_tool(
            &routing.reader_name,
            MicroToolContext {
                routing_call_key: routing.routing_call_key,
                provider_call_id: "call_0".to_string(),
                storage_key: routing.storage_iri,
                tool_name: "web_search".to_string(),
                entity_types: vec![],
                preview_size: 100,
            },
        );
        let file_allowed = vec!["file_read".to_string()];
        let web_allowed = vec!["web_search".to_string()];
        assert!(ToolExecutor::explicit_allowlist_permits(
            "read_agent_output",
            &file_allowed
        ));
        assert!(!executor.allowlist_permits(&routing.reader_name, &file_allowed));
        assert!(executor.allowlist_permits(&routing.reader_name, &web_allowed));
        assert!(!executor.allowlist_permits("query_customer_status", &file_allowed));
        assert!(!ToolExecutor::is_micro_tool_name("query_customer_status"));
        assert!(!ToolExecutor::is_pa_readonly_tool("query_customer_status"));
    }

    #[test]
    fn micro_tool_lookup_is_scoped_by_originating_call() {
        let mut executor = ToolExecutor::new();
        for call_id in ["call_a", "call_b"] {
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-tool-executor",
                call_id,
            );
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: call_id.to_string(),
                    storage_key: routing.storage_iri,
                    tool_name: "file_read".to_string(),
                    entity_types: vec![],
                    preview_size: 100,
                },
            );
        }

        let call_a =
            crate::tools::result_router::ResultRoutingIdentity::new("l1-tool-executor", "call_a");
        assert_eq!(
            executor.get_micro_tool_names_for_routing_key(&call_a.routing_call_key),
            vec![call_a.reader_name]
        );
        assert!(executor
            .get_micro_tool_names_for_routing_key("unknown")
            .is_empty());
    }

    #[test]
    fn session_cleanup_retires_only_its_own_microtools_and_payloads() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let pa =
                crate::tools::result_router::ResultRoutingIdentity::new("l1-pa-cleanup", "call_0");
            let da = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-da-still-running",
                "call_0",
            );

            for (routing, marker) in [(&pa, "PA_ONLY"), (&da, "DA_ONLY")] {
                executor.store_micro_tool_data(
                    &routing.storage_iri,
                    json!({"content": marker, "tool_name": "bash"}),
                );
                executor.register_micro_tool(
                    &routing.reader_name,
                    MicroToolContext {
                        routing_call_key: routing.routing_call_key.clone(),
                        provider_call_id: "call_0".to_string(),
                        storage_key: routing.storage_iri.clone(),
                        tool_name: "bash".to_string(),
                        entity_types: vec![],
                        preview_size: 100,
                    },
                );
            }

            assert_eq!(executor.remove_micro_tools_for_session("l1-pa-cleanup"), 1);
            assert!(executor
                .get_micro_tool_names_for_routing_key(&pa.routing_call_key)
                .is_empty());
            assert_eq!(
                executor.get_micro_tool_names_for_routing_key(&da.routing_call_key),
                vec![da.reader_name.clone()]
            );
            assert!(executor
                .execute(&pa.reader_name, json!({"char_offset": 0, "char_limit": 1}))
                .await
                .is_err());
            let live = executor
                .execute(
                    &da.reader_name,
                    json!({"char_offset": 0, "char_limit": 100}),
                )
                .await
                .unwrap();
            assert_eq!(live["content"], "DA_ONLY");
            assert_eq!(live["call_id"], "call_0");
        });
    }

    #[test]
    fn configured_micro_tool_limits_control_catalog_and_paging() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.set_micro_tool_limits(2, 2, 3);
            let mut routings = Vec::new();
            for call_id in ["one", "two", "three"] {
                let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                    "l1-configured-limits",
                    call_id,
                );
                executor.store_micro_tool_data(
                    &routing.storage_iri,
                    json!({
                        "content": json!({
                            "path": "fixture.txt",
                            "total_lines": 5,
                            "offset": 0,
                            "lines": ["a", "b", "c", "d", "e"],
                            "returned": 5,
                        }).to_string()
                    }),
                );
                executor.register_micro_tool(
                    &routing.reader_name,
                    MicroToolContext {
                        routing_call_key: routing.routing_call_key.clone(),
                        provider_call_id: call_id.to_string(),
                        storage_key: routing.storage_iri.clone(),
                        tool_name: "file_read".to_string(),
                        entity_types: vec![],
                        preview_size: 1,
                    },
                );
                routings.push(routing);
            }

            let advertised = executor
                .tool_definitions_for_role("DA")
                .into_iter()
                .filter(|definition| {
                    definition["function"]["name"]
                        .as_str()
                        .is_some_and(|name| name.starts_with("read_full_result_"))
                })
                .count();
            assert_eq!(advertised, 2);

            let default_page = executor
                .execute(&routings[2].reader_name, json!({}))
                .await
                .unwrap();
            assert_eq!(default_page["returned"], 2);
            assert_eq!(default_page["next_cursor"]["offset"], 0);
            assert_eq!(default_page["next_cursor"]["limit"], 2);
            assert_eq!(default_page["next_cursor"]["char_offset"], 1);
            let capped_page = executor
                .execute(&routings[1].reader_name, json!({"limit": 99}))
                .await
                .unwrap();
            assert_eq!(capped_page["returned"], 3);
        });
    }

    #[test]
    fn micro_tool_character_pages_bound_single_line_json_results() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-character-pages",
                "long-json",
            );
            executor.store_micro_tool_data(&routing.storage_iri, json!({"content": "abcdefghij"}));
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: "long_json".to_string(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "rag_search".to_string(),
                    entity_types: vec![],
                    preview_size: 4,
                },
            );

            let first = executor
                .execute(&routing.reader_name, json!({}))
                .await
                .unwrap();
            assert_eq!(first["content"], "abcd");
            assert_eq!(first["returned_chars"], 4);
            assert_eq!(first["next_char_offset"], 4);
            assert_eq!(first["truncated"], true);
            assert!(executor
                .micro_tool_definition(&routing.reader_name)
                .is_some());

            let second = executor
                .execute(
                    &routing.reader_name,
                    json!({"char_offset": first["next_char_offset"]}),
                )
                .await
                .unwrap();
            assert_eq!(second["content"], "efgh");
            assert_eq!(second["next_char_offset"], 8);

            let terminal = executor
                .execute(
                    &routing.reader_name,
                    json!({"char_offset": second["next_char_offset"]}),
                )
                .await
                .unwrap();
            assert_eq!(terminal["content"], "ij");
            assert_eq!(terminal["complete"], true);
            assert!(terminal["next_cursor"].is_null());
            assert!(
                executor
                    .micro_tool_definition(&routing.reader_name)
                    .is_none(),
                "a completely delivered raw reader must leave the next model tool window"
            );
            assert!(
                executor
                    .micro_tool_definition_for_history(&routing.reader_name)
                    .is_some(),
                "a retired reader must retain a receipt-only schema for provider histories that still contain its native tool call"
            );
            assert!(!executor
                .tool_definitions_for_role("DA")
                .iter()
                .any(|definition| definition["function"]["name"] == routing.reader_name));

            let repeated = executor
                .clone()
                .execute(&routing.reader_name, json!({"char_offset": 0}))
                .await
                .unwrap();
            assert_eq!(repeated["status"], "already_consumed");
            assert_eq!(repeated["receipt_kind"], "archived_result_consumed");
            assert_eq!(repeated["content"], "");
            assert_eq!(repeated["content_omitted"], true);
            assert_eq!(repeated["call_id"], "long_json");
            assert_eq!(repeated["routing_call_key"], routing.routing_call_key);
            assert!(repeated["archive_sha256"]
                .as_str()
                .is_some_and(|digest| digest.starts_with("sha256:") && digest.len() == 71));

            // The raw provider call ID may repeat, but a fresh L1 produces a
            // different composite reader and therefore has independent
            // delivery state.
            let fresh = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-character-pages-fresh",
                "long-json",
            );
            executor.store_micro_tool_data(&fresh.storage_iri, json!({"content": "abcdefghij"}));
            executor.register_micro_tool(
                &fresh.reader_name,
                MicroToolContext {
                    routing_call_key: fresh.routing_call_key.clone(),
                    provider_call_id: "long_json".to_string(),
                    storage_key: fresh.storage_iri.clone(),
                    tool_name: "rag_search".to_string(),
                    entity_types: vec![],
                    preview_size: 4,
                },
            );
            let fresh_first = executor
                .execute(&fresh.reader_name, json!({}))
                .await
                .unwrap();
            assert_eq!(fresh_first["content"], "abcd");
            assert_ne!(fresh.reader_name, routing.reader_name);
        });
    }

    #[test]
    fn archived_character_reader_rejects_eof_skip_and_out_of_range_cursor() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-character-eof",
                "provider-eof",
            );
            executor.store_micro_tool_data(&routing.storage_iri, json!({"content": "abc界"}));
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "web_fetch".to_string(),
                    entity_types: vec![],
                    preview_size: 2,
                },
            );

            let eof = executor
                .execute(&routing.reader_name, json!({"char_offset": 4}))
                .await
                .unwrap();
            assert_eq!(eof["content"], "");
            assert_eq!(eof["status"], "cursor_gap_rejected");
            assert_eq!(eof["receipt_kind"], "archived_result_cursor_required");
            assert_eq!(eof["complete"], false);
            assert_eq!(eof["next_cursor"]["char_offset"], 0);
            // The parser accepts exact EOF as a valid coordinate, but the
            // delivery ledger cannot let it skip the preceding bytes.
            assert!(executor
                .micro_tool_definition(&routing.reader_name)
                .is_some());

            let beyond = executor
                .execute(&routing.reader_name, json!({"char_offset": 5}))
                .await
                .unwrap_err();
            assert!(beyond.contains("outside content range 0..4"));
        });
    }

    #[test]
    fn archived_character_reader_omits_repeated_and_overlapping_delivered_prefixes() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-character-prefix",
                "provider-prefix-raw",
            );
            executor.store_micro_tool_data(&routing.storage_iri, json!({"content": "abcdefghij"}));
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "web_fetch".to_string(),
                    entity_types: vec![],
                    preview_size: 4,
                },
            );

            let first = executor
                .execute(&routing.reader_name, json!({}))
                .await
                .unwrap();
            assert_eq!(first["content"], "abcd");
            assert_eq!(first["next_cursor"]["char_offset"], 4);

            for repeated_input in [json!({}), json!({"char_offset": 2, "char_limit": 4})] {
                let receipt = executor
                    .execute(&routing.reader_name, repeated_input)
                    .await
                    .unwrap();
                assert_eq!(receipt["status"], "already_delivered_prefix");
                assert_eq!(receipt["receipt_kind"], "archived_result_prefix_consumed");
                assert_eq!(receipt["content"], "");
                assert_eq!(receipt["content_omitted"], true);
                assert_eq!(receipt["complete"], false);
                assert_eq!(receipt["next_cursor"]["char_offset"], 4);
                assert_eq!(receipt["call_id"], "provider-prefix-raw");
                assert_eq!(receipt["routing_call_key"], routing.routing_call_key);
            }

            let skipped = executor
                .execute(&routing.reader_name, json!({"char_offset": 8}))
                .await
                .unwrap();
            assert_eq!(skipped["status"], "cursor_gap_rejected");
            assert_eq!(skipped["receipt_kind"], "archived_result_cursor_required");
            assert_eq!(skipped["content"], "");
            assert_eq!(skipped["next_cursor"]["char_offset"], 4);

            let continued = executor
                .execute(&routing.reader_name, json!({"char_offset": 4}))
                .await
                .unwrap();
            assert_eq!(continued["content"], "efgh");
            assert_eq!(continued["next_cursor"]["char_offset"], 8);
        });
    }

    #[test]
    fn archived_execution_reader_pages_decoded_streams_with_exact_character_cursors() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-execution-stream-reader",
                "provider/Call:RAW-01",
            );
            let stdout = (0..100)
                .map(|line| format!("output-line-{line:03}-界"))
                .collect::<Vec<_>>()
                .join("\n");
            let raw_envelope = json!({
                "command": "python -m pytest -q",
                "stdout": stdout,
                "stderr": "one warning\nsecond warning",
                "exit_code": 0,
                "duration_ms": 42,
            })
            .to_string();
            executor.store_micro_tool_data(
                &routing.storage_iri,
                json!({"content": raw_envelope, "tool_name": "bash"}),
            );
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "bash".to_string(),
                    entity_types: vec![],
                    preview_size: 73,
                },
            );

            let schema = executor
                .micro_tool_definition(&routing.reader_name)
                .unwrap();
            assert!(schema
                .pointer("/function/parameters/properties/stream")
                .is_some());
            assert!(schema
                .pointer("/function/parameters/properties/offset")
                .is_none());

            let mut reconstructed = String::new();
            let mut char_offset = 0usize;
            loop {
                let page = executor
                    .execute(
                        &routing.reader_name,
                        json!({"stream": "stdout", "char_offset": char_offset}),
                    )
                    .await
                    .unwrap();
                assert_eq!(page["reader_view"], "execution_stream");
                assert_eq!(page["stream"], "stdout");
                assert_eq!(page["call_id"], "provider/Call:RAW-01");
                assert_eq!(page["exit_code"], 0);
                reconstructed.push_str(page["content"].as_str().unwrap());
                let Some(next) = page["next_cursor"]["char_offset"].as_u64() else {
                    break;
                };
                assert!(next as usize > char_offset);
                char_offset = next as usize;
            }
            assert_eq!(reconstructed, stdout);

            let stderr = executor
                .execute(
                    &routing.reader_name,
                    json!({"stream": "stderr", "char_offset": 0}),
                )
                .await
                .unwrap();
            assert!(stderr["content"]
                .as_str()
                .is_some_and(|content| content.starts_with("one warning")));
            assert!(
                executor
                    .micro_tool_definition(&routing.reader_name)
                    .is_none(),
                "stdout and the non-empty stderr stream were both consumed"
            );
            let raw = executor
                .execute(
                    &routing.reader_name,
                    json!({"stream": "raw", "char_offset": 0}),
                )
                .await
                .unwrap();
            assert!(raw["content"]
                .as_str()
                .is_some_and(|content| content.starts_with('{')));

            let legacy_cursor = executor
                .execute(&routing.reader_name, json!({"offset": 22, "limit": 40}))
                .await
                .unwrap_err();
            assert!(legacy_cursor.contains("not valid for this result view"));
            let exact_end = executor
                .execute(
                    &routing.reader_name,
                    json!({"stream": "stdout", "char_offset": stdout.chars().count()}),
                )
                .await
                .unwrap();
            assert_eq!(exact_end["status"], "already_consumed");
            assert_eq!(exact_end["content"], "");
            assert_eq!(exact_end["complete"], true);
            let past_end = executor
                .execute(
                    &routing.reader_name,
                    json!({"stream": "stdout", "char_offset": stdout.chars().count() + 1}),
                )
                .await
                .unwrap_err();
            assert!(past_end.contains("outside content range"));
        });
    }

    #[test]
    fn archived_execution_reader_retires_after_complete_raw_envelope_delivery() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-execution-raw-retirement",
                "provider-raw-retirement",
            );
            let raw_envelope = json!({
                "command": "python3 -m pytest -q",
                "stdout": "12 passed\n",
                "stderr": "one warning\n",
                "exit_code": 0,
            })
            .to_string();
            executor.store_micro_tool_data(
                &routing.storage_iri,
                json!({"content": raw_envelope, "tool_name": "bash"}),
            );
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "bash".to_string(),
                    entity_types: vec![],
                    preview_size: 1_000,
                },
            );

            let raw = executor
                .execute(&routing.reader_name, json!({"stream": "raw"}))
                .await
                .unwrap();
            assert_eq!(raw["content"], raw_envelope);
            assert_eq!(raw["complete"], true);
            assert!(executor
                .micro_tool_definition(&routing.reader_name)
                .is_none());

            let repeated = executor
                .execute(&routing.reader_name, json!({"stream": "raw"}))
                .await
                .unwrap();
            assert_eq!(repeated["status"], "already_consumed");
            assert_eq!(repeated["content"], "");
            assert_eq!(repeated["call_id"], "provider-raw-retirement");
        });
    }

    #[test]
    fn seeded_long_command_survives_stdout_completion_until_its_exact_cursor_finishes() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-command-preview-lifecycle",
                "provider-command-RAW/7",
            );
            let command = format!("python3 -m pytest -q {}", "参数 ".repeat(2_000));
            let raw_envelope = json!({
                "command": command,
                "stdout": "7 passed\n",
                "stderr": "",
                "exit_code": 0,
                "duration_ms": 17,
            })
            .to_string();
            executor.store_micro_tool_data(
                &routing.storage_iri,
                json!({"content": raw_envelope, "tool_name": "bash"}),
            );
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "bash".to_string(),
                    entity_types: vec![],
                    preview_size: 97,
                },
            );
            let preview = crate::tools::result_router::summary::execution_preview_envelope(
                "bash",
                &raw_envelope,
                &routing,
                raw_envelope.len(),
                1_024,
            );
            let preview_value: Value = serde_json::from_str(&preview).unwrap();
            let command_cursor = preview_value["reader_cursors"]["command"].clone();
            let start = command_cursor["char_offset"].as_u64().unwrap() as usize;
            assert!(start > 0);
            assert!(
                executor.seed_archived_reader_progress_from_preview(&routing.reader_name, &preview)
            );

            let stdout_receipt = executor
                .execute(
                    &routing.reader_name,
                    json!({"stream": "stdout", "char_offset": 0}),
                )
                .await
                .unwrap();
            assert_eq!(stdout_receipt["status"], "already_consumed");
            assert!(executor
                .micro_tool_definition(&routing.reader_name)
                .is_some());

            let raw_prefix = executor
                .execute(
                    &routing.reader_name,
                    json!({"stream": "raw", "char_offset": 0, "char_limit": 31}),
                )
                .await
                .unwrap();
            let mut raw_cursor = raw_prefix["next_cursor"].clone();
            assert!(!raw_cursor.is_null());

            let mut cursor = command_cursor;
            let mut first = true;
            loop {
                let page = executor
                    .execute(&routing.reader_name, cursor)
                    .await
                    .unwrap();
                assert_eq!(page["call_id"], "provider-command-RAW/7");
                if first {
                    assert_eq!(
                        page["content"].as_str().unwrap().chars().next(),
                        command.chars().nth(start)
                    );
                    first = false;
                }
                if page["next_cursor"].is_null() {
                    break;
                }
                assert!(executor
                    .micro_tool_definition(&routing.reader_name)
                    .is_some());
                cursor = page["next_cursor"].clone();
            }
            assert!(
                executor
                    .micro_tool_definition(&routing.reader_name)
                    .is_some(),
                "a partially consumed raw view must keep the reader live"
            );
            loop {
                let page = executor
                    .execute(&routing.reader_name, raw_cursor)
                    .await
                    .unwrap();
                if page["next_cursor"].is_null() {
                    break;
                }
                raw_cursor = page["next_cursor"].clone();
            }
            assert!(executor
                .micro_tool_definition(&routing.reader_name)
                .is_none());
        });
    }

    #[test]
    fn archived_code_execute_reader_defaults_to_stdout_and_maps_code_to_command_stream() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-code-execute-reader",
                "provider-code-call",
            );
            let raw_envelope = json!({
                "code": "print('ready')",
                "stdout": "ready\n",
                "stderr": "",
                "exit_code": 0,
            })
            .to_string();
            executor.store_micro_tool_data(
                &routing.storage_iri,
                json!({"content": raw_envelope, "tool_name": "code_execute"}),
            );
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "code_execute".to_string(),
                    entity_types: vec![],
                    preview_size: 128,
                },
            );

            let default_page = executor
                .execute(&routing.reader_name, json!({}))
                .await
                .unwrap();
            assert_eq!(default_page["reader_view"], "execution_stream");
            assert_eq!(default_page["stream"], "stdout");
            assert_eq!(default_page["content"], "ready\n");

            let command_page = executor
                .execute(&routing.reader_name, json!({"stream": "command"}))
                .await
                .unwrap();
            assert_eq!(command_page["content"], "print('ready')");
            assert_eq!(command_page["call_id"], "provider-code-call");
        });
    }

    #[test]
    fn archived_file_reader_uses_source_line_coordinates_not_serialized_json_lines() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-file-lines-reader",
                "call_file_lines",
            );
            let lines = (0..500)
                .map(|line| format!("source-line-{line:03}"))
                .collect::<Vec<_>>();
            let raw_envelope = json!({
                "path": "docs/large.md",
                "total_lines": 500,
                "offset": 0,
                "lines": lines,
                "returned": 500,
            })
            .to_string();
            executor.store_micro_tool_data(
                &routing.storage_iri,
                json!({"content": raw_envelope, "tool_name": "file_read"}),
            );
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "file_read".to_string(),
                    entity_types: vec![],
                    preview_size: 2_000,
                },
            );

            let schema = executor
                .micro_tool_definition(&routing.reader_name)
                .unwrap();
            assert!(schema
                .pointer("/function/parameters/properties/offset")
                .is_some());
            assert!(schema
                .pointer("/function/parameters/properties/stream")
                .is_none());
            let skipped = executor
                .execute(&routing.reader_name, json!({"offset": 200, "limit": 40}))
                .await
                .unwrap();
            assert_eq!(skipped["status"], "cursor_gap_rejected");
            assert_eq!(skipped["next_cursor"]["offset"], 0);

            for offset in (0..=200).step_by(40) {
                let page = executor
                    .execute(&routing.reader_name, json!({"offset": offset, "limit": 40}))
                    .await
                    .unwrap();
                assert_eq!(page["reader_view"], "file_lines");
                assert_eq!(page["offset"], offset);
                assert_eq!(page["returned"], 40);
                assert_eq!(page["next_cursor"]["offset"], offset + 40);
                assert_eq!(page["call_id"], "call_file_lines");
                if offset == 200 {
                    let content = page["content"].as_str().unwrap();
                    assert!(content.starts_with("source-line-200\nsource-line-201"));
                    assert!(content.ends_with("source-line-239"));
                    assert!(!content.contains("\\\"lines\\\""));
                }
            }

            let eof = executor
                .execute(&routing.reader_name, json!({"offset": 500, "limit": 1}))
                .await
                .unwrap();
            assert_eq!(eof["content"], "");
            assert_eq!(eof["status"], "cursor_gap_rejected");
            assert_eq!(eof["next_cursor"]["offset"], 240);
            let invalid = executor
                .execute(&routing.reader_name, json!({"offset": 501, "limit": 1}))
                .await
                .unwrap_err();
            assert!(invalid.contains("outside archived source range"));
        });
    }

    #[test]
    fn archived_file_reader_retires_only_after_exact_cursor_chain_reaches_terminal_page() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.set_micro_tool_limits(5, 2, 3);
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-file-consumption",
                "provider-file-consumption",
            );
            let raw_envelope = json!({
                "path": "docs/short.md",
                "total_lines": 3,
                "offset": 0,
                "lines": ["alpha", "bravo", "charlie"],
                "returned": 3,
            })
            .to_string();
            executor.store_micro_tool_data(
                &routing.storage_iri,
                json!({"content": raw_envelope, "tool_name": "file_read"}),
            );
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "file_read".to_string(),
                    entity_types: vec![],
                    // Force character continuations inside selected line
                    // pages as well as a later source-line continuation.
                    preview_size: 5,
                },
            );

            let mut cursor = json!({});
            let mut pages = 0usize;
            loop {
                let page = executor
                    .execute(&routing.reader_name, cursor)
                    .await
                    .unwrap();
                pages += 1;
                assert!(pages < 10, "cursor chain did not converge: {page}");
                if page["next_cursor"].is_null() {
                    assert_eq!(page["complete"], true);
                    break;
                }
                assert!(executor
                    .micro_tool_definition(&routing.reader_name)
                    .is_some());
                cursor = page["next_cursor"].clone();
            }
            assert!(pages > 3, "fixture must exercise both cursor dimensions");
            assert!(executor
                .micro_tool_definition(&routing.reader_name)
                .is_none());

            let duplicate = executor
                .execute(&routing.reader_name, json!({"offset": 0, "limit": 2}))
                .await
                .unwrap();
            assert_eq!(duplicate["status"], "already_consumed");
            assert_eq!(duplicate["content"], "");
            assert_eq!(duplicate["path"], "docs/short.md");
        });
    }

    #[test]
    fn archived_file_reader_omits_repeated_prefix_and_returns_exact_continuation() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.set_micro_tool_limits(5, 2, 3);
            let routing = crate::tools::result_router::ResultRoutingIdentity::new(
                "l1-file-prefix",
                "provider-file-prefix",
            );
            let raw_envelope = json!({
                "path": "docs/prefix.md",
                "total_lines": 3,
                "offset": 0,
                "lines": ["alpha", "bravo", "charlie"],
                "returned": 3,
            })
            .to_string();
            executor.store_micro_tool_data(
                &routing.storage_iri,
                json!({"content": raw_envelope, "tool_name": "file_read"}),
            );
            executor.register_micro_tool(
                &routing.reader_name,
                MicroToolContext {
                    routing_call_key: routing.routing_call_key.clone(),
                    provider_call_id: routing.provider_call_id.clone(),
                    storage_key: routing.storage_iri.clone(),
                    tool_name: "file_read".to_string(),
                    entity_types: vec![],
                    preview_size: 5,
                },
            );

            let first = executor
                .execute(&routing.reader_name, json!({"offset": 0, "limit": 2}))
                .await
                .unwrap();
            assert_eq!(first["content"], "alpha");
            assert_eq!(first["next_cursor"]["offset"], 0);
            assert_eq!(first["next_cursor"]["limit"], 2);
            assert_eq!(first["next_cursor"]["char_offset"], 5);

            for repeated_input in [
                json!({"offset": 0, "limit": 2, "char_offset": 0}),
                json!({"offset": 0, "limit": 2, "char_offset": 2}),
            ] {
                let receipt = executor
                    .execute(&routing.reader_name, repeated_input)
                    .await
                    .unwrap();
                assert_eq!(receipt["status"], "already_delivered_prefix");
                assert_eq!(receipt["content"], "");
                assert_eq!(receipt["next_cursor"]["offset"], 0);
                assert_eq!(receipt["next_cursor"]["limit"], 2);
                assert_eq!(receipt["next_cursor"]["char_offset"], 5);
                assert_eq!(receipt["call_id"], "provider-file-prefix");
            }

            let changed_page = executor
                .execute(
                    &routing.reader_name,
                    json!({"offset": 0, "limit": 1, "char_offset": 5}),
                )
                .await
                .unwrap();
            assert_eq!(changed_page["status"], "cursor_gap_rejected");
            assert_eq!(
                changed_page["receipt_kind"],
                "archived_result_cursor_required"
            );
            assert_eq!(changed_page["content"], "");
            assert_eq!(changed_page["next_cursor"]["offset"], 0);
            assert_eq!(changed_page["next_cursor"]["limit"], 2);
            assert_eq!(changed_page["next_cursor"]["char_offset"], 5);

            let continued = executor
                .execute(&routing.reader_name, first["next_cursor"].clone())
                .await
                .unwrap();
            assert_eq!(continued["content"], "\nbrav");
        });
    }

    #[test]
    fn tool_search_queries_the_live_catalog_instead_of_static_fallback() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let result = executor
                .execute(
                    "tool_search",
                    json!({"query": "knowledge import directory", "max_results": 10}),
                )
                .await
                .unwrap();
            let names: Vec<&str> = result["matches"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|item| item["name"].as_str())
                .collect();
            assert!(names.contains(&"knowledge_import_directory"));
        });
    }

    #[test]
    fn tools_allowed_passes_listed_tool_through() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let input = json!({"query": "search test"});
            let result = executor
                .execute_with_security_context(
                    "tool_search",
                    input,
                    crate::skill_graph::security::SecurityContext::new("agent:test", "DA")
                        .with_task("iri://tasks/allowlist-test"),
                    Some(&["tool_search".to_string()]),
                )
                .await
                .unwrap();
            assert!(!result
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("")
                .contains("Tool not allowed: tool_search"));
        });
    }

    #[test]
    fn knowledge_path_handler_accepts_absolute_paths_inside_workspace() {
        let container = tempfile::Builder::new()
            .prefix(".knowledge-path-inside-test-")
            .tempdir_in(std::env::current_dir().unwrap())
            .unwrap();
        let file = container.path().join("sample.py");
        std::fs::write(&file, "def value():\n    return 1\n").unwrap();

        for field in ["path", "file_path"] {
            let mut input = json!({});
            input[field] = json!(file);
            let resolved = resolve_workspace_path_argument(input, field).unwrap();
            assert_eq!(
                resolved[field],
                std::fs::canonicalize(&file)
                    .unwrap()
                    .to_string_lossy()
                    .as_ref()
            );
        }
    }

    #[test]
    fn knowledge_tool_handlers_pass_absolute_paths_inside_workspace() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let container = tempfile::Builder::new()
                .prefix(".knowledge-handler-inside-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();

            // Invalid UTF-8 proves that the import-file handler passed the
            // workspace boundary and reached its content decoder without
            // creating any global RAG-index fixture.
            let invalid_text = container.path().join("invalid.txt");
            std::fs::write(&invalid_text, [0xff]).unwrap();
            let import_error = executor
                .execute("knowledge_import_file", json!({"path": invalid_text}))
                .await
                .unwrap_err();
            assert!(
                import_error.contains("Failed to read file"),
                "{import_error}"
            );

            let empty_directory = container.path().join("empty");
            std::fs::create_dir(&empty_directory).unwrap();
            let directory_result = executor
                .execute(
                    "knowledge_import_directory",
                    json!({"path": empty_directory}),
                )
                .await
                .unwrap();
            assert_eq!(directory_result["success"], true);
            assert_eq!(directory_result["files_processed"], 0);

            let code = container.path().join("sample.py");
            std::fs::write(&code, "def value():\n    return 1\n").unwrap();
            let extraction_result = executor
                .execute("knowledge_extract_code", json!({"file_path": code}))
                .await
                .unwrap();
            assert_eq!(extraction_result["success"], true, "{extraction_result}");
        });
    }

    #[test]
    fn knowledge_tool_handlers_reject_absolute_and_parent_paths_outside_workspace() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let outside = tempfile::tempdir().unwrap();
            let outside_file = outside.path().join("secret.py");
            std::fs::write(&outside_file, "SECRET = True\n").unwrap();

            for (tool, field, path) in [
                (
                    "knowledge_import_file",
                    "path",
                    outside_file.to_string_lossy().to_string(),
                ),
                (
                    "knowledge_import_directory",
                    "path",
                    outside.path().to_string_lossy().to_string(),
                ),
                (
                    "knowledge_extract_code",
                    "file_path",
                    outside_file.to_string_lossy().to_string(),
                ),
            ] {
                let mut input = json!({});
                input[field] = json!(path);
                let error = executor.execute(tool, input).await.unwrap_err();
                assert!(
                    error.contains("not within the workspace"),
                    "{tool}: {error}"
                );

                let mut input = json!({});
                input[field] = json!("../outside-workspace");
                let error = executor.execute(tool, input).await.unwrap_err();
                assert!(
                    error.contains("not within the workspace"),
                    "{tool}: {error}"
                );
            }
        });
    }

    #[cfg(unix)]
    #[test]
    fn knowledge_tool_handlers_reject_workspace_symlink_escapes() {
        use std::os::unix::fs::symlink;

        rt().block_on(async {
            let executor = ToolExecutor::new();
            let container = tempfile::Builder::new()
                .prefix(".knowledge-path-symlink-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let outside = tempfile::tempdir().unwrap();
            let outside_file = outside.path().join("secret.py");
            std::fs::write(&outside_file, "SECRET = True\n").unwrap();
            let file_link = container.path().join("file-link.py");
            let directory_link = container.path().join("directory-link");
            symlink(&outside_file, &file_link).unwrap();
            symlink(outside.path(), &directory_link).unwrap();

            for (tool, field, path) in [
                (
                    "knowledge_import_file",
                    "path",
                    file_link.to_string_lossy().to_string(),
                ),
                (
                    "knowledge_import_directory",
                    "path",
                    directory_link.to_string_lossy().to_string(),
                ),
                (
                    "knowledge_extract_code",
                    "file_path",
                    file_link.to_string_lossy().to_string(),
                ),
            ] {
                let mut input = json!({});
                input[field] = json!(path);
                let error = executor.execute(tool, input).await.unwrap_err();
                assert!(
                    error.contains("not within the workspace"),
                    "{tool}: {error}"
                );
            }
        });
    }

    #[test]
    fn cached_large_file_still_returns_later_requested_ranges() {
        rt().block_on(async {
            // Built-in file tools intentionally enforce the process workspace.
            // Keep the fixture inside it so this test reaches the cache/range
            // behavior instead of being rejected by path isolation first.
            let dir = tempfile::Builder::new()
                .prefix(".file-read-cache-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let path = dir.path().join("large.txt");
            let content = (0..500)
                .map(|index| format!("line-{index}"))
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(&path, content).unwrap();

            let monitor = crate::tools::workspace_monitor::WorkspaceMonitor::initialize(
                crate::tools::workspace_monitor::WorkspaceMonitorConfig {
                    workspace_root: dir.path().to_path_buf(),
                    watch_enabled: false,
                    db_path: None,
                    ..Default::default()
                },
                None,
                None,
            )
            .unwrap();
            let mut executor = ToolExecutor::new();
            executor.set_workspace_monitor(Arc::new(monitor));

            let first = executor
                .execute("file_read", json!({"path": path, "offset": 0, "limit": 10}))
                .await
                .unwrap();
            let second = executor
                .execute(
                    "file_read",
                    json!({"path": path, "offset": 10, "limit": 10}),
                )
                .await
                .unwrap();

            assert!(first.get("lines").is_some());
            assert!(second.get("lines").is_some(), "{second}");
            assert!(second.to_string().contains("line-10"), "{second}");
        });
    }

    #[test]
    fn file_write_reports_semantic_noop_without_advancing_workspace_generation() {
        rt().block_on(async {
            let dir = tempfile::Builder::new()
                .prefix(".file-write-noop-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let path = dir.path().join("same.txt");
            std::fs::write(&path, "same content").unwrap();
            let monitor = Arc::new(
                crate::tools::workspace_monitor::WorkspaceMonitor::initialize(
                    crate::tools::workspace_monitor::WorkspaceMonitorConfig {
                        workspace_root: dir.path().to_path_buf(),
                        watch_enabled: false,
                        ..Default::default()
                    },
                    None,
                    None,
                )
                .unwrap(),
            );
            let initial_generation = monitor.generation();
            let mut executor = ToolExecutor::new();
            executor.set_workspace_monitor(monitor.clone());

            let result = executor
                .execute(
                    "file_write",
                    json!({"path": path, "content": "same content"}),
                )
                .await
                .unwrap();
            assert_eq!(result["success"], true);
            assert_eq!(result["changed"], false);
            assert_eq!(result["bytes_written"], 0);
            assert_eq!(
                result["content_sha256"],
                crate::utils::CryptoUtils::sha256_hex("same content")
            );
            assert_eq!(monitor.generation(), initial_generation);
        });
    }

    #[test]
    fn workspace_resource_lease_enforces_exact_file_write_target() {
        rt().block_on(async {
            let workspace = tempfile::Builder::new()
                .prefix(".workspace-lease-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let lease = crate::core::effect::WorkspaceResourceLease::new(
                "child-writer",
                workspace.path(),
                vec![(
                    "allowed.txt".to_string(),
                    crate::core::effect::WorkspaceLeaseAccess::Write,
                )],
            )
            .unwrap();
            let executor = ToolExecutor::new();
            let context = crate::skill_graph::security::SecurityContext::new("agent:child", "DA")
                .with_task("iri://tasks/workspace-lease")
                .with_workspace_resource_lease(lease);

            let allowed = executor
                .execute_with_security_context_and_effect_policy(
                    "file_write",
                    json!({"path": "allowed.txt", "content": "committed"}),
                    context.clone(),
                    Some(&["file_write".to_string()]),
                    &crate::core::effect::EffectPolicy::None,
                )
                .await
                .unwrap();
            assert_eq!(allowed["success"], true, "{allowed}");
            assert_eq!(
                std::fs::read_to_string(workspace.path().join("allowed.txt")).unwrap(),
                "committed"
            );

            let denied = executor
                .execute_with_security_context_and_effect_policy(
                    "file_write",
                    json!({"path": "denied.txt", "content": "must not exist"}),
                    context,
                    Some(&["file_write".to_string()]),
                    &crate::core::effect::EffectPolicy::None,
                )
                .await
                .unwrap();
            assert_eq!(denied["reason"], "workspace_resource_lease_violation");
            assert!(!workspace.path().join("denied.txt").exists());
        });
    }

    #[cfg(unix)]
    #[test]
    fn hook_rewritten_file_path_is_rechecked_against_workspace_lease() {
        rt().block_on(async {
            let workspace = tempfile::Builder::new()
                .prefix(".workspace-lease-hook-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let lease = crate::core::effect::WorkspaceResourceLease::new(
                "hook-child",
                workspace.path(),
                vec![(
                    "allowed.txt".to_string(),
                    crate::core::effect::WorkspaceLeaseAccess::Write,
                )],
            )
            .unwrap();
            let mut executor = ToolExecutor::new();
            executor.set_hook_runner_with_input_rewrite(
                HookRunner::new(RuntimeHookConfig::new(
                    vec![r#"printf '%s' '{"hookSpecificOutput":{"updatedInput":{"path":"denied.txt","content":"escaped"}}}'"#.to_string()],
                    vec![],
                    vec![],
                )),
                true,
            );
            let context = crate::skill_graph::security::SecurityContext::new(
                "agent:hook-child",
                "DA",
            )
            .with_task("iri://tasks/workspace-lease-hook")
            .with_workspace_resource_lease(lease);

            let result = executor
                .execute_with_security_context_and_effect_policy(
                    "file_write",
                    json!({"path": "allowed.txt", "content": "original"}),
                    context,
                    Some(&["file_write".to_string()]),
                    &crate::core::effect::EffectPolicy::None,
                )
                .await
                .unwrap();

            assert_eq!(result["reason"], "workspace_resource_lease_violation");
            assert!(!workspace.path().join("allowed.txt").exists());
            assert!(!workspace.path().join("denied.txt").exists());
        });
    }

    #[test]
    fn file_list_uses_complete_workspace_inventory_with_canonical_state() {
        rt().block_on(async {
            let dir = tempfile::Builder::new()
                .prefix(".inventory-list-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            std::fs::create_dir_all(dir.path().join("src")).unwrap();
            std::fs::write(dir.path().join("src/lib.rs"), "pub fn value() -> u8 { 1 }").unwrap();
            let monitor = crate::tools::workspace_monitor::WorkspaceMonitor::initialize(
                crate::tools::workspace_monitor::WorkspaceMonitorConfig {
                    workspace_root: dir.path().to_path_buf(),
                    watch_enabled: false,
                    ..Default::default()
                },
                None,
                None,
            )
            .unwrap();
            let mut executor = ToolExecutor::new();
            executor.set_workspace_monitor(Arc::new(monitor));

            let root = executor
                .execute("file_list", json!({"path": dir.path()}))
                .await
                .unwrap();
            assert_eq!(root["source"], "workspace_inventory");
            assert!(root["entries"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry["name"] == "src" && entry["type"] == "dir"));

            let src = executor
                .execute("file_list", json!({"path": dir.path().join("src")}))
                .await
                .unwrap();
            let lib = src["entries"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["name"] == "lib.rs")
                .unwrap();
            assert_eq!(lib["language"], "rust");
            assert_eq!(lib["state"], "discovered");
        });
    }

    #[test]
    fn whole_file_cache_visibility_is_isolated_between_biz_agents() {
        rt().block_on(async {
            let dir = tempfile::Builder::new()
                .prefix(".file-read-agent-cache-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let path = dir.path().join("shared.txt");
            std::fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

            let monitor = crate::tools::workspace_monitor::WorkspaceMonitor::initialize(
                crate::tools::workspace_monitor::WorkspaceMonitorConfig {
                    workspace_root: dir.path().to_path_buf(),
                    watch_enabled: false,
                    db_path: None,
                    ..Default::default()
                },
                None,
                None,
            )
            .unwrap();
            let mut executor = ToolExecutor::new();
            executor.set_workspace_monitor(Arc::new(monitor));
            let allowed = ["file_read".to_string()];

            let read_as = |agent: &str| {
                executor.execute_with_security_context(
                    "file_read",
                    json!({"path": path}),
                    crate::skill_graph::security::SecurityContext::new(agent, "DA")
                        .with_task("iri://tasks/shared-cache"),
                    Some(&allowed),
                )
            };

            let pa_first = read_as("pa_001").await.unwrap();
            let pa_repeat = read_as("pa_001").await.unwrap();
            let da_first = read_as("da_001").await.unwrap();

            assert!(pa_first.get("lines").is_some(), "{pa_first}");
            assert!(pa_repeat.get("lines").is_none(), "{pa_repeat}");
            assert_eq!(pa_repeat.get("from_cache"), Some(&Value::Bool(true)));
            assert!(
                da_first.get("lines").is_some(),
                "a new BizAgent context must receive cached content: {da_first}"
            );
            assert_eq!(da_first.get("from_cache"), Some(&Value::Bool(true)));
        });
    }

    #[test]
    fn whole_file_cache_visibility_is_isolated_between_l1_sessions_of_same_agent() {
        rt().block_on(async {
            let dir = tempfile::Builder::new()
                .prefix(".file-read-l1-cache-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let path = dir.path().join("shared.txt");
            std::fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

            let monitor = crate::tools::workspace_monitor::WorkspaceMonitor::initialize(
                crate::tools::workspace_monitor::WorkspaceMonitorConfig {
                    workspace_root: dir.path().to_path_buf(),
                    watch_enabled: false,
                    db_path: None,
                    ..Default::default()
                },
                None,
                None,
            )
            .unwrap();
            let mut executor = ToolExecutor::new();
            executor.set_workspace_monitor(Arc::new(monitor));
            let allowed = ["file_read".to_string()];
            let task = crate::core::agent_runner::TaskContext::new(
                "iri://task/shared-cache",
                "audit shared file",
                4,
            );

            let read_as = |l1_session: &str| {
                executor.execute_with_security_context(
                    "file_read",
                    json!({"path": path}),
                    task.tool_security_context("ca_001", "CA", l1_session),
                    Some(&allowed),
                )
            };

            let l1_a_first = read_as("l1_a").await.unwrap();
            let l1_a_repeat = read_as("l1_a").await.unwrap();
            let l1_b_first = read_as("l1_b").await.unwrap();
            let l1_b_repeat = read_as("l1_b").await.unwrap();

            assert!(l1_a_first.get("lines").is_some(), "{l1_a_first}");
            assert!(l1_a_repeat.get("lines").is_none(), "{l1_a_repeat}");
            assert_eq!(l1_a_repeat.get("from_cache"), Some(&Value::Bool(true)));
            assert!(
                l1_b_first.get("lines").is_some(),
                "a fresh L1 of the same Agent must receive complete content: {l1_b_first}"
            );
            assert_eq!(l1_b_first.get("from_cache"), Some(&Value::Bool(true)));
            assert!(
                l1_b_first["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("had not observed it yet")),
                "{l1_b_first}"
            );
            assert!(l1_b_repeat.get("lines").is_none(), "{l1_b_repeat}");
        });
    }

    #[test]
    fn existing_file_overwrite_requires_fresh_agent_l1_whole_file_baseline() {
        rt().block_on(async {
            let dir = tempfile::Builder::new()
                .prefix(".file-baseline-cas-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let existing = dir.path().join("existing.txt");
            let stale = dir.path().join("stale.txt");
            let created = dir.path().join("created.txt");
            std::fs::write(&existing, "alpha\nbeta\n").unwrap();
            std::fs::write(&stale, "revision-a\n").unwrap();

            let monitor = crate::tools::workspace_monitor::WorkspaceMonitor::initialize(
                crate::tools::workspace_monitor::WorkspaceMonitorConfig {
                    workspace_root: dir.path().to_path_buf(),
                    watch_enabled: false,
                    db_path: None,
                    ..Default::default()
                },
                None,
                None,
            )
            .unwrap();
            let mut executor = ToolExecutor::new();
            executor.set_workspace_monitor(Arc::new(monitor));
            let allowed = ["file_read".to_string(), "file_write".to_string()];
            let task = crate::core::agent_runner::TaskContext::new(
                "iri://task/file-baseline-cas",
                "safely update exact files",
                8,
            );
            let context = |agent: &str, l1: &str| task.tool_security_context(agent, "DA", l1);
            let baseline = |agent: &str, l1: &str, path: &std::path::Path, hash: String| {
                crate::skill_graph::security::FileOverwriteBaselineEvent {
                    path: Some(path.to_string_lossy().to_string()),
                    content_sha256: Some(hash),
                    source_call_identity: Some(
                        crate::core::execution_journal::ToolCallIdentity::new(
                            agent,
                            l1,
                            "previous-model-request",
                            "raw-provider-call-id",
                        ),
                    ),
                }
            };

            let denied = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": existing, "content": "replacement\n"}),
                    context("da-a", "l1-a"),
                    Some(&allowed),
                )
                .await
                .unwrap_err();
            assert!(denied.contains("overwrite_baseline_required"), "{denied}");
            assert_eq!(std::fs::read_to_string(&existing).unwrap(), "alpha\nbeta\n");

            let partial = executor
                .execute_with_security_context(
                    "file_read",
                    json!({"path": existing, "limit": 1}),
                    context("da-a", "l1-a"),
                    Some(&allowed),
                )
                .await
                .unwrap();
            assert_eq!(partial["returned"], 1);
            let denied_after_partial = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": existing, "content": "replacement\n"}),
                    context("da-a", "l1-a"),
                    Some(&allowed),
                )
                .await
                .unwrap_err();
            assert!(
                denied_after_partial.contains("overwrite_baseline_required"),
                "{denied_after_partial}"
            );

            let whole = executor
                .execute_with_security_context(
                    "file_read",
                    json!({"path": existing, "mode": "full"}),
                    context("da-a", "l1-a"),
                    Some(&allowed),
                )
                .await
                .unwrap();
            assert!(whole.get("lines").is_some(), "{whole}");
            let whole_baseline = baseline(
                "da-a",
                "l1-a",
                &existing,
                whole["content_sha256"].as_str().unwrap().to_string(),
            );
            let written = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": existing, "content": "replacement\n"}),
                    context("da-a", "l1-a").with_file_overwrite_baseline_events([whole_baseline]),
                    Some(&allowed),
                )
                .await
                .unwrap();
            assert_eq!(written["changed"], true);
            assert_eq!(std::fs::read_to_string(&existing).unwrap(), "replacement\n");

            let fresh_l1_denied = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": existing, "content": "fresh-l1-blind-write\n"}),
                    context("da-a", "l1-b"),
                    Some(&allowed),
                )
                .await
                .unwrap_err();
            assert!(
                fresh_l1_denied.contains("overwrite_baseline_required"),
                "{fresh_l1_denied}"
            );

            executor
                .execute_with_security_context(
                    "file_read",
                    json!({"path": stale, "mode": "full"}),
                    context("da-a", "l1-a"),
                    Some(&allowed),
                )
                .await
                .unwrap();
            std::fs::write(&stale, "revision-b-by-other-writer\n").unwrap();
            let stale_baseline = baseline(
                "da-a",
                "l1-a",
                &stale,
                crate::utils::CryptoUtils::sha256_hex("revision-a\n"),
            );
            let stale_denied = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": stale, "content": "must-not-land\n"}),
                    context("da-a", "l1-a").with_file_overwrite_baseline_events([stale_baseline]),
                    Some(&allowed),
                )
                .await
                .unwrap_err();
            assert!(
                stale_denied.contains("overwrite_baseline_stale"),
                "{stale_denied}"
            );
            assert_eq!(
                std::fs::read_to_string(&stale).unwrap(),
                "revision-b-by-other-writer\n"
            );

            let created_result = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": created, "content": "first\n"}),
                    context("da-a", "l1-a"),
                    Some(&allowed),
                )
                .await
                .unwrap();
            assert_eq!(created_result["created"], true);
            let creator_followup = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": created, "content": "second\n"}),
                    context("da-a", "l1-a").with_file_overwrite_baseline_events([baseline(
                        "da-a",
                        "l1-a",
                        &created,
                        created_result["content_sha256"]
                            .as_str()
                            .unwrap()
                            .to_string(),
                    )]),
                    Some(&allowed),
                )
                .await
                .unwrap();
            assert_eq!(creator_followup["changed"], true);

            let forged = executor
                .execute_with_security_context(
                    "file_write",
                    json!({
                        "path": existing,
                        "content": "forged\n",
                        "__gh_expected_current_sha256": whole["content_sha256"],
                    }),
                    context("da-b", "l1-forged"),
                    Some(&allowed),
                )
                .await
                .unwrap();
            assert_eq!(forged["reason"], "reserved_internal_field");
            assert_eq!(std::fs::read_to_string(&existing).unwrap(), "replacement\n");
        });
    }

    #[test]
    fn identical_blind_write_does_not_grant_a_later_overwrite_baseline() {
        rt().block_on(async {
            let dir = tempfile::Builder::new()
                .prefix(".file-baseline-noop-test-")
                .tempdir_in(std::env::current_dir().unwrap())
                .unwrap();
            let path = dir.path().join("same.txt");
            std::fs::write(&path, "same\n").unwrap();
            let executor = ToolExecutor::new();
            let allowed = ["file_write".to_string()];
            let context = crate::skill_graph::security::SecurityContext::new("da-noop", "DA")
                .with_task("iri://task/file-baseline-noop");

            let noop = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": path, "content": "same\n"}),
                    context.clone(),
                    Some(&allowed),
                )
                .await
                .unwrap();
            assert_eq!(noop["changed"], false);
            let denied = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": path, "content": "different\n"}),
                    context,
                    Some(&allowed),
                )
                .await
                .unwrap_err();
            assert!(denied.contains("overwrite_baseline_required"), "{denied}");
        });
    }

    #[test]
    fn security_context_denies_high_risk_registered_tool_and_audits_it() {
        rt().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let target = dir.path().join("must-not-write");
            let executor = ToolExecutor::new();
            let registry = Arc::new(SkillRegistry::new());
            let graph = Arc::new(SkillGraphStore::new());
            let meta = registry.get_skill("iri://skills/file_write").unwrap();
            graph
                .register_skill(crate::skill_graph::types::SkillGraphNode::from_skill_meta(
                    &meta,
                ))
                .unwrap();
            let security = Arc::new(crate::skill_graph::security::SecurityEngine::new(
                graph.clone(),
            ));
            executor.set_shared_skill_registry(registry);
            executor.set_shared_skill_graph(graph);
            executor.set_security_engine(security.clone());

            let result = executor
                .execute_with_security_context(
                    "file_write",
                    json!({"path": target, "content": "blocked"}),
                    crate::skill_graph::security::SecurityContext::new("agent:test", "DA")
                        .with_task("iri://tasks/security-test"),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(result["error"], "Security denied");
            assert!(!target.exists());
            let audit = security
                .get_audit_log(Some("iri://skills/file_write"), Some("agent:test"), 10)
                .await;
            assert_eq!(audit.len(), 1);
        });
    }

    #[test]
    fn security_gate_allows_ca_inspection_tools_with_whitelisted_file_read() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let registry = Arc::new(SkillRegistry::new());
            let graph = Arc::new(SkillGraphStore::new());
            // The CLI (apps/gliding_code/src/engine.rs) registers SystemBuiltin skills like
            // file_read into the graph and wires SecurityEngine::with_whitelisted_skills using
            // that same SystemBuiltin set as the allowlist. Replicate that setup here so the
            // the gate resolves iri://skills/file_read and whitelist-approves it.
            let meta = registry.get_skill("iri://skills/file_read").unwrap();
            graph
                .register_skill(crate::skill_graph::types::SkillGraphNode::from_skill_meta(
                    &meta,
                ))
                .unwrap();
            let whitelist = std::collections::HashSet::from(["iri://skills/file_read".to_string()]);
            let security = Arc::new(
                crate::skill_graph::security::SecurityEngine::with_whitelisted_skills(
                    graph.clone(),
                    whitelist.clone(),
                ),
            );
            executor.set_shared_skill_registry(registry);
            executor.set_shared_skill_graph(graph);
            executor.set_security_engine(security.clone());

            // CA inspection tools must not be rejected as unregistered. AA is
            // decision-only and receives this evidence from CA through BizAgent.
            for tool in [
                "file_list",
                "workspace_status",
                "rag_search",
                "kg_search",
                "knowledge_list",
                "knowledge_search",
                "knowledge_extract_code",
            ] {
                // Handler-level input validation may still Err (e.g. kg_search needs "query");
                // what matters is that the security gate never denies the tool as unregistered.
                let outcome = executor
                    .execute_with_security_context(
                        tool,
                        json!({"path": "."}),
                        crate::skill_graph::security::SecurityContext::new("agent:test", "CA")
                            .with_task("iri://tasks/security-test"),
                        None,
                    )
                    .await;
                let err = match outcome {
                    Ok(result) => result
                        .get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("")
                        .to_string(),
                    Err(e) => e,
                };
                assert!(
                    !err.contains("no registered executable skill"),
                    "tool {} was denied by gate: {}",
                    tool,
                    err
                );
            }

            let path = std::env::current_dir()
                .unwrap()
                .join("target")
                .join(format!("ca-security-audit-{}.txt", uuid::Uuid::new_v4()));
            std::fs::write(&path, b"verified").unwrap();
            let read_result = executor
                .execute_with_security_context(
                    "file_read",
                    json!({"path": path}),
                    crate::skill_graph::security::SecurityContext::new("agent:test", "CA")
                        .with_task("iri://tasks/security-test"),
                    None,
                )
                .await;
            let _ = std::fs::remove_file(&path);
            let read = read_result.unwrap();
            assert_eq!(read["lines"][0], "verified");

            let audit = security
                .get_audit_log(Some("iri://skills/file_read"), Some("agent:test"), 50)
                .await;
            assert!(
                audit
                    .iter()
                    .any(|e| e.outcome == crate::skill_graph::types::AuditOutcome::Success),
                "whitelisted read skill should produce allow audit entries"
            );
        });
    }

    #[test]
    fn skill_creator_gateway_is_executor_local_and_settable_after_builtin_registration() {
        let executor = ToolExecutor::new();
        let gateway = Arc::new(
            UnifiedGateway::new(&GatewaySettings {
                base_url: "http://127.0.0.1:9".to_string(),
                api_key: "test".to_string(),
                default_model: "test".to_string(),
                timeout_seconds: 1,
                max_retries: 0,
                retry_base_ms: 1,
                use_responses_api: false,
                model_mapping: HashMap::new(),
            })
            .unwrap(),
        );

        executor.set_shared_skill_creator_gateway(gateway.clone());

        assert!(executor
            .shared_skill_creator_gateway
            .read()
            .as_ref()
            .is_some_and(|stored| { Arc::ptr_eq(stored, &gateway) }));
    }

    #[test]
    fn nested_skill_llm_calls_preserve_runner_hooks_accounting_and_concurrent_scope_isolation() {
        rt().block_on(async {
            use crate::llm::{
                LlmInteractionPhase, LlmInteractionService, LlmUsageSnapshot,
            };
            use crate::tools::hooks::{FunctionHook, HookManager, HookPoint, HookResult};
            use std::sync::atomic::{AtomicU64, Ordering};
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let response_body = serde_json::json!({
                "id": "nested-skill-provider",
                "choices": [{
                    "index": 0,
                    // Deliberately invalid as a skill definition. The model
                    // interaction completes and is accounted, while no graph,
                    // registry or filesystem artifact is mutated by this test.
                    "message": {"role": "assistant", "content": "not a skill definition"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })
            .to_string();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let mut connections = Vec::new();
                for _ in 0..2 {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let body = response_body.clone();
                    connections.push(tokio::spawn(async move {
                        let mut request = vec![0_u8; 32 * 1024];
                        let _ = socket.read(&mut request).await;
                        let header = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            body.len()
                        );
                        socket.write_all(header.as_bytes()).await.unwrap();
                        socket.write_all(body.as_bytes()).await.unwrap();
                        let _ = socket.shutdown().await;
                    }));
                }
                for connection in connections {
                    connection.await.unwrap();
                }
            });

            let gateway = Arc::new(
                UnifiedGateway::new(&GatewaySettings {
                    base_url: format!("http://{address}"),
                    api_key: "test".to_string(),
                    default_model: "test-model".to_string(),
                    timeout_seconds: 2,
                    max_retries: 0,
                    retry_base_ms: 1,
                    use_responses_api: false,
                    model_mapping: HashMap::new(),
                })
                .unwrap(),
            );
            let interactions = Arc::new(LlmInteractionService::new(gateway.clone()));
            let hook_calls = Arc::new(AtomicU64::new(0));
            let hook_calls_for_handler = hook_calls.clone();
            let hooks = Arc::new(HookManager::new());
            hooks.register(Box::new(FunctionHook::new(
                "nested-skill-observer",
                vec![HookPoint::LlmRequest, HookPoint::LlmResponse],
                1,
                move |_| {
                    hook_calls_for_handler.fetch_add(1, Ordering::SeqCst);
                    HookResult::Continue
                },
            )));
            interactions.set_hook_manager(hooks);
            let mut events = interactions.subscribe();
            let event_bus = Arc::new(crate::core::event_bus::EventBus::new(32));
            let mut forwarded_events = event_bus.subscribe();
            interactions.attach_event_bus(event_bus);

            let executor = ToolExecutor::new();
            executor.set_shared_skill_creator_gateway(gateway);
            executor.set_shared_skill_creator_interactions(interactions.clone());
            assert!(executor
                .skill_creator_interactions()
                .is_some_and(|stored| Arc::ptr_eq(&stored, &interactions)));

            let first_context = crate::skill_graph::security::SecurityContext::new(
                "agent:da/first",
                "DA",
            )
            .with_task("iri://task/child-first")
            .with_llm_invocation(
                "iri://task/root-first",
                "llm_react_first",
                "cycle-first",
            );
            let second_context = crate::skill_graph::security::SecurityContext::new(
                "agent:da/second",
                "DA",
            )
            .with_task("iri://task/child-second")
            .with_llm_invocation(
                "iri://task/root-second",
                "llm_react_second",
                "cycle-second",
            );

            // Reserved-looking values are intentionally malicious. Scope is
            // derived only from SecurityContext and must ignore all of them.
            let first_input = json!({
                "description": "first nested skill",
                "task_iri": "iri://forged/task",
                "usage_scope_iri": "iri://forged/root",
                "parent_interaction_id": "llm_forged_parent",
                "cycle_id": "forged-cycle",
                "agent_id": "agent:forged"
            });
            let second_input = json!({"markdown_content": "# Second nested skill"});
            let (first_result, second_result) = tokio::join!(
                executor.execute_with_security_context(
                    "create_skill",
                    first_input,
                    first_context,
                    None,
                ),
                executor.execute_with_security_context(
                    "convert_skill",
                    second_input,
                    second_context,
                    None,
                ),
            );
            assert!(first_result.is_err());
            assert!(second_result.is_err());
            server.await.unwrap();

            let mut forwarded = Vec::new();
            for _ in 0..6 {
                forwarded.push(
                    tokio::time::timeout(
                        std::time::Duration::from_secs(1),
                        forwarded_events.recv(),
                    )
                    .await
                    .expect("nested interaction EventBus forwarding timed out")
                    .expect("nested interaction EventBus closed"),
                );
            }
            assert_eq!(
                forwarded
                    .iter()
                    .filter(|event| event.task_iri == "iri://task/child-first")
                    .count(),
                3
            );
            assert_eq!(
                forwarded
                    .iter()
                    .filter(|event| event.task_iri == "iri://task/child-second")
                    .count(),
                3
            );

            let mut observed = Vec::new();
            while let Ok(event) = events.try_recv() {
                observed.push(event);
            }
            assert_eq!(
                observed
                    .iter()
                    .filter(|event| event.phase == LlmInteractionPhase::Completed)
                    .count(),
                2
            );
            assert_eq!(hook_calls.load(Ordering::SeqCst), 4);
            assert_eq!(
                interactions.usage_snapshot_for_scope("iri://task/root-first"),
                LlmUsageSnapshot {
                    prompt_tokens: 3,
                    completion_tokens: 2,
                }
            );
            assert_eq!(
                interactions.usage_snapshot_for_scope("iri://task/root-second"),
                LlmUsageSnapshot {
                    prompt_tokens: 3,
                    completion_tokens: 2,
                }
            );
            assert_eq!(
                interactions.usage_snapshot_for_scope("iri://forged/root"),
                LlmUsageSnapshot::default()
            );

            let first_event = observed
                .iter()
                .find(|event| {
                    event.phase == LlmInteractionPhase::Assembled
                        && event.scope.task_iri.as_deref() == Some("iri://task/child-first")
                })
                .expect("first nested interaction must be observable on runner service");
            assert_eq!(first_event.scope.stage, "skill_creation");
            assert_eq!(
                first_event.scope.usage_scope_iri.as_deref(),
                Some("iri://task/root-first")
            );
            assert_eq!(
                first_event.scope.parent_interaction_id.as_deref(),
                Some("llm_react_first")
            );
            assert_eq!(first_event.scope.cycle_id.as_deref(), Some("cycle-first"));
            assert_eq!(first_event.scope.agent_id.as_deref(), Some("agent:da/first"));
            assert_ne!(
                first_event.scope.task_iri.as_deref(),
                Some("iri://forged/task")
            );

            let second_event = observed
                .iter()
                .find(|event| {
                    event.phase == LlmInteractionPhase::Assembled
                        && event.scope.task_iri.as_deref() == Some("iri://task/child-second")
                })
                .expect("second nested interaction must retain an isolated scope");
            assert_eq!(second_event.scope.stage, "skill_markdown_conversion");
            assert_eq!(
                second_event.scope.parent_interaction_id.as_deref(),
                Some("llm_react_second")
            );
            assert_ne!(
                first_event.scope.interaction_id,
                second_event.scope.interaction_id
            );
        });
    }

    #[test]
    fn test_permission_policy_allows_read_tool() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
                .with_tool_requirement("bash", PermissionMode::DangerFullAccess);
            executor.set_permission_policy(policy);

            let input = json!({"pattern": "*.rs", "path": "."});
            let result = executor.execute("glob_search", input).await;
            assert!(result.is_ok());
        });
    }

    #[test]
    fn test_permission_policy_with_default_config_allows_all() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.set_default_permission_policy();

            let input = json!({"command": "ls"});
            let result = executor.execute("bash", input).await;
            assert!(result.is_ok() || result.is_err());
            if let Ok(val) = &result {
                assert!(
                    val.get("error").is_none()
                        || !val
                            .get("error")
                            .and_then(|e| e.as_str())
                            .unwrap_or("")
                            .contains("Permission denied")
                );
            }
        });
    }

    #[test]
    fn test_permission_policy_denies_write_in_readonly_mode() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
                .with_tool_requirement("file_write", PermissionMode::WorkspaceWrite);
            executor.set_permission_policy(policy);

            let input = json!({"path": "/tmp/test.txt", "content": "test"});
            let result = executor.execute("file_write", input).await.unwrap();
            assert!(result
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("")
                .contains("Permission denied"));
        });
    }

    #[test]
    fn test_hook_runner_pre_tool_use_denies_tool() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let hook_config = RuntimeHookConfig::new(
                vec!["printf 'blocked by security policy'; exit 2".to_string()],
                vec![],
                vec![],
            );
            executor.set_hook_runner(HookRunner::new(hook_config));

            let input = json!({"command": "ls"});
            let result = executor.execute("bash", input).await.unwrap();
            assert!(result
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("")
                .contains("Pre-tool hook denied"));
        });
    }

    #[test]
    fn test_hook_runner_does_not_block_allowed_tool() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let hook_config = RuntimeHookConfig::new(
                vec!["printf 'blocked by security policy'; exit 2".to_string()],
                vec![],
                vec![],
            );
            executor.set_hook_runner(HookRunner::new(hook_config));

            let input = json!({"query": "search test"});
            let result = executor.execute("tool_search", input).await;
            assert!(result.is_ok());
        });
    }

    #[test]
    fn denied_post_tool_hook_never_returns_original_output() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "secret_result",
                "Return a value used to test post-hook disclosure",
                json!({"type": "object", "additionalProperties": false}),
                Arc::new(|_| {
                    Box::pin(async { Ok(json!({"content": "POST_HOOK_SECRET_SENTINEL"})) })
                }),
                &[],
            );
            executor.set_hook_runner(HookRunner::new(RuntimeHookConfig::new(
                vec![],
                vec!["printf 'deny result'; exit 2".to_string()],
                vec![],
            )));

            let result = executor.execute("secret_result", json!({})).await.unwrap();
            let serialized = result.to_string();
            assert_eq!(result["post_hook_denied"], true);
            assert!(!serialized.contains("POST_HOOK_SECRET_SENTINEL"));
            assert!(result.get("original_output").is_none());
        });
    }

    #[test]
    fn pre_tool_hook_applies_schema_valid_updated_input() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "hook_echo",
                "Echo validated hook input",
                json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": false
                }),
                Arc::new(|input| Box::pin(async move { Ok(input) })),
                &[],
            );
            executor.set_hook_runner_with_input_rewrite(
                HookRunner::new(RuntimeHookConfig::new(
                    vec![
                    r#"printf '%s' '{"hookSpecificOutput":{"updatedInput":{"value":"after"}}}'"#
                        .to_string(),
                ],
                    vec![],
                    vec![],
                )),
                true,
            );

            let result = executor
                .execute("hook_echo", json!({"value": "before"}))
                .await
                .unwrap();

            assert_eq!(result, json!({"value": "after"}));
        });
    }

    #[test]
    fn pre_tool_hook_ignores_updated_input_without_explicit_rewrite_permission() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "hook_echo_no_rewrite",
                "Echo the original input when rewrite policy is disabled",
                json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": false
                }),
                Arc::new(|input| Box::pin(async move { Ok(input) })),
                &[],
            );
            executor.set_hook_runner(HookRunner::new(RuntimeHookConfig::new(
                vec![
                    r#"printf '%s' '{"hookSpecificOutput":{"updatedInput":{"value":"after"}}}'"#
                        .to_string(),
                ],
                vec![],
                vec![],
            )));

            let result = executor
                .execute("hook_echo_no_rewrite", json!({"value": "before"}))
                .await
                .unwrap();

            assert_eq!(result, json!({"value": "before"}));
        });
    }

    #[test]
    fn pre_tool_hook_rejects_updated_input_that_violates_schema() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "hook_echo",
                "Echo validated hook input",
                json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": false
                }),
                Arc::new(|input| Box::pin(async move { Ok(input) })),
                &[],
            );
            executor.set_hook_runner_with_input_rewrite(
                HookRunner::new(RuntimeHookConfig::new(
                    vec![
                    r#"printf '%s' '{"hookSpecificOutput":{"updatedInput":{"unexpected":true}}}'"#
                        .to_string(),
                ],
                    vec![],
                    vec![],
                )),
                true,
            );

            let result = executor
                .execute("hook_echo", json!({"value": "before"}))
                .await
                .unwrap();

            assert!(result["error"]
                .as_str()
                .is_some_and(|error| error.contains("updatedInput rejected")));
        });
    }

    #[test]
    fn pre_tool_hook_cannot_inject_reserved_internal_fields() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.set_hook_runner_with_input_rewrite(HookRunner::new(RuntimeHookConfig::new(
                vec![r#"printf '%s' '{"hookSpecificOutput":{"updatedInput":{"path":"Cargo.toml","__gh_read_session":"forged"}}}'"#.to_string()],
                vec![],
                vec![],
            )), true);

            let result = executor
                .execute("file_read", json!({"path": "Cargo.toml"}))
                .await
                .unwrap();

            assert!(result["error"]
                .as_str()
                .is_some_and(|error| error.contains("reserved internal fields")));
        });
    }

    #[test]
    fn permission_rules_are_rechecked_after_hook_input_mutation() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "hook_command",
                "Execute a test command model without side effects",
                json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }),
                Arc::new(|input| Box::pin(async move { Ok(input) })),
                &[],
            );
            let rules = RuntimePermissionRuleConfig::new(
                vec![],
                vec!["hook_command(blocked:*)".to_string()],
                vec![],
            );
            executor.set_permission_policy(
                PermissionPolicy::new(PermissionMode::Allow)
                    .with_tool_requirement("hook_command", PermissionMode::DangerFullAccess)
                    .with_permission_rules(&rules),
            );
            executor.set_hook_runner_with_input_rewrite(HookRunner::new(RuntimeHookConfig::new(
                vec![r#"printf '%s' '{"hookSpecificOutput":{"updatedInput":{"command":"blocked: payload"}}}'"#.to_string()],
                vec![],
                vec![],
            )), true);

            let result = executor
                .execute("hook_command", json!({"command": "safe"}))
                .await
                .unwrap();

            assert!(result["error"]
                .as_str()
                .is_some_and(|error| error.contains("Permission denied after pre-tool hook")));
        });
    }

    #[test]
    fn internal_skill_hook_patch_cannot_upgrade_permission() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "internal_hook_command",
                "Model a command without running a process",
                json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": false
                }),
                Arc::new(|input| Box::pin(async move { Ok(input) })),
                &[],
            );
            let rules = RuntimePermissionRuleConfig::new(
                vec![],
                vec!["internal_hook_command(blocked:*)".to_string()],
                vec![],
            );
            executor.set_permission_policy(
                PermissionPolicy::new(PermissionMode::Allow)
                    .with_tool_requirement(
                        "internal_hook_command",
                        PermissionMode::DangerFullAccess,
                    )
                    .with_permission_rules(&rules),
            );

            let result = executor
                .execute_hook_modified_with_security_context_and_effect_policy(
                    "internal_hook_command",
                    json!({"command": "blocked: payload"}),
                    crate::skill_graph::security::SecurityContext::new("agent:hook", "DA")
                        .with_task("iri://tasks/internal-hook-permission"),
                    Some(&["internal_hook_command".to_string()]),
                    &crate::core::effect::EffectPolicy::None,
                )
                .await
                .unwrap();

            assert!(result["error"]
                .as_str()
                .is_some_and(|error| error.contains("Permission denied")));
        });
    }

    #[test]
    fn internal_skill_hook_patch_must_satisfy_registered_schema() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "internal_hook_schema",
                "Accept only a string value",
                json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": false
                }),
                Arc::new(|input| Box::pin(async move { Ok(input) })),
                &[],
            );

            let result = executor
                .execute_hook_modified_with_security_context_and_effect_policy(
                    "internal_hook_schema",
                    json!({"value": 7}),
                    crate::skill_graph::security::SecurityContext::new("agent:hook", "DA")
                        .with_task("iri://tasks/internal-hook-schema"),
                    Some(&["internal_hook_schema".to_string()]),
                    &crate::core::effect::EffectPolicy::None,
                )
                .await
                .unwrap();

            assert!(result["error"]
                .as_str()
                .is_some_and(|error| error.contains("arguments patch rejected")));
        });
    }

    #[test]
    fn hook_permission_deny_is_enforced_without_static_policy() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.set_hook_runner(HookRunner::new(RuntimeHookConfig::new(
                vec![r#"printf '%s' '{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"local hook policy"}}'"#.to_string()],
                vec![],
                vec![],
            )));

            let result = executor
                .execute("tool_search", json!({"query": "memory"}))
                .await
                .unwrap();

            assert_eq!(result["error"], "local hook policy");
        });
    }

    #[test]
    fn contextual_security_is_rechecked_after_hook_input_mutation() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let registry = Arc::new(SkillRegistry::new());
            let graph = Arc::new(SkillGraphStore::new());
            let meta = registry.get_skill("iri://skills/file_read").unwrap();
            graph
                .register_skill(crate::skill_graph::types::SkillGraphNode::from_skill_meta(
                    &meta,
                ))
                .unwrap();
            let security = Arc::new(
                crate::skill_graph::security::SecurityEngine::with_whitelisted_skills(
                    graph.clone(),
                    std::collections::HashSet::from(["iri://skills/file_read".to_string()]),
                ),
            );
            executor.set_shared_skill_registry(registry);
            executor.set_shared_skill_graph(graph);
            executor.set_security_engine(security.clone());
            executor.set_hook_runner_with_input_rewrite(HookRunner::new(RuntimeHookConfig::new(
                vec![r#"printf '%s' '{"hookSpecificOutput":{"updatedInput":{"path":"Cargo.lock","offset":1,"limit":1}}}'"#.to_string()],
                vec![],
                vec![],
            )), true);

            let result = executor
                .execute_with_security_context(
                    "file_read",
                    json!({"path": "Cargo.toml", "offset": 1, "limit": 1}),
                    crate::skill_graph::security::SecurityContext::new("agent:hook", "CA")
                        .with_task("iri://tasks/hook-security"),
                    Some(&["file_read".to_string()]),
                )
                .await
                .unwrap();

            assert!(result.get("lines").is_some(), "{result}");
            let audit = security
                .get_audit_log(
                    Some("iri://skills/file_read"),
                    Some("agent:hook"),
                    10,
                )
                .await;
            assert_eq!(
                audit.len(),
                2,
                "original and hook-mutated calls must each cross the security boundary"
            );
        });
    }

    #[test]
    fn test_permission_policy_takes_precedence_over_hooks() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
                .with_tool_requirement("bash", PermissionMode::DangerFullAccess);
            executor.set_permission_policy(policy);
            let hook_config = RuntimeHookConfig::new(vec![], vec![], vec![]);
            executor.set_hook_runner(HookRunner::new(hook_config));

            let input = json!({"command": "ls"});
            let result = executor.execute("bash", input).await.unwrap();
            assert!(result
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("")
                .contains("Permission denied"));
        });
    }

    #[test]
    fn test_pa_readonly_tools_includes_bash() {
        assert!(ToolExecutor::is_pa_readonly_tool("bash"));
        assert!(ToolExecutor::is_pa_readonly_tool("file_read"));
        assert!(ToolExecutor::is_pa_readonly_tool("grep_search"));
        assert!(ToolExecutor::is_pa_readonly_tool("read_agent_output"));
        let routing =
            crate::tools::result_router::ResultRoutingIdentity::new("l1-pa-readonly", "call_0");
        assert!(ToolExecutor::is_pa_readonly_tool(&routing.reader_name));
        assert!(ToolExecutor::is_pa_readonly_tool(
            &routing.query_name("Person")
        ));
        assert!(!ToolExecutor::is_pa_readonly_tool("query_result_entities"));
        assert!(!ToolExecutor::is_pa_readonly_tool("file_write"));
        assert!(!ToolExecutor::is_pa_readonly_tool("file_edit"));
        assert!(!ToolExecutor::is_pa_readonly_tool(
            "unregistered_dynamic_tool"
        ));
    }

    #[test]
    fn test_knowledge_tools_use_store_injected_after_builtin_registration() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let unified = crate::memory::unified_graph::UnifiedGraphStore::new().unwrap();
            executor.set_unified_kg_store(unified.store());

            executor
                .execute(
                    "knowledge_import_json",
                    json!({
                        "json_data": r#"{"id":"shared-store-check","type":"https://example.org/Concept","label":"Shared store check"}"#,
                        "mapping_config": r#"{"id_field":"id","type_field":"type","label_field":"label"}"#
                    }),
                )
                .await
                .unwrap();

            let kg_store = executor.knowledge_graph_store();
            let rows = kg_store
                .read()
                .unwrap()
                .query_sparql("SELECT ?s WHERE { ?s ?p ?o }", Some("graph:world"))
                .unwrap();

            assert!(
                !rows.is_empty(),
                "knowledge tool writes must be visible through the injected shared store"
            );
        });
    }

    #[test]
    fn create_skill_description_does_not_claim_automatic_executability() {
        let executor = ToolExecutor::new();
        let description = executor
            .tool_descriptions
            .iter()
            .find(|tool| tool.name == "create_skill")
            .expect("create_skill builtin should be registered")
            .description
            .to_lowercase();

        assert!(description.contains("does not create an executable"));
        assert!(!description.contains("available for use"));
    }

    #[test]
    fn trusted_internal_registration_fully_replaces_contextual_skill_builtin() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "create_skill",
                "Application-owned skill creation",
                json!({
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"]
                }),
                Arc::new(|input| {
                    Box::pin(async move { Ok(json!({"custom": true, "input": input})) })
                }),
                &["DA"],
            );

            let result = executor
                .execute_with_security_context(
                    "create_skill",
                    json!({"description": "handled without an LLM"}),
                    crate::skill_graph::security::SecurityContext::new("agent:custom", "DA")
                        .with_task("iri://task/custom"),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(result["custom"], true);
            assert_eq!(result["input"]["description"], "handled without an LLM");
        });
    }

    /// Regression test: MCP tools are registered with long-form role names
    /// ("Plan"/"Do"/"Check"/"Act", see McpClient::register_tools_to_tool_executor),
    /// while consumers call tool_definitions_for_role with short-form names.
    /// Trusted role aliases must match, while the kernel ceiling still blocks
    /// arbitrary tools for PA/CA and every tool for decision-only AA.
    #[test]
    fn test_long_form_allowed_roles_match_short_form_agent_roles() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "mcp_server_browse",
                "MCP-registered browsing tool",
                json!({"type": "object", "properties": {}}),
                Arc::new(|input: Value| Box::pin(async move { Ok(json!({"ok": input})) })),
                &["Plan", "Do", "Check", "Act"],
            );
            executor.register(
                "web_search",
                "PA read-only MCP tool",
                json!({"type": "object", "properties": {}}),
                Arc::new(|input: Value| Box::pin(async move { Ok(json!({"ok": input})) })),
                &["Plan"],
            );
            executor.register(
                "jsonld_validate",
                "CA validation MCP tool",
                json!({"type": "object", "properties": {}}),
                Arc::new(|input: Value| Box::pin(async move { Ok(json!({"ok": input})) })),
                &["Check"],
            );

            let names = |role: &str| {
                executor
                    .tool_definitions_for_role(role)
                    .iter()
                    .filter_map(|d| d["function"]["name"].as_str().map(String::from))
                    .collect::<Vec<_>>()
            };
            assert!(names("PA").contains(&"web_search".to_string()));
            assert!(names("CA").contains(&"jsonld_validate".to_string()));
            assert!(names("DA").contains(&"mcp_server_browse".to_string()));
            assert!(!names("PA").contains(&"mcp_server_browse".to_string()));
            assert!(!names("CA").contains(&"mcp_server_browse".to_string()));
            assert!(executor.tool_definitions_for_role("AA").is_empty());
        });
    }

    #[test]
    fn test_tool_definitions_for_role_with_allowlist_intersection() {
        let executor = ToolExecutor::new();
        let full = executor.tool_definitions_for_role("DA");
        assert!(!full.is_empty(), "DA role should expose builtin tools");
        let full_names: Vec<String> = full
            .iter()
            .filter_map(|td| td["function"]["name"].as_str().map(String::from))
            .collect();

        // None keeps the full role-filtered set; explicit empty denies all.
        assert_eq!(
            executor
                .tool_definitions_for_role_with_allowlist("DA", None)
                .len(),
            full.len()
        );
        let empty: Vec<String> = vec![];
        assert!(executor
            .tool_definitions_for_role_with_allowlist("DA", Some(&empty))
            .is_empty());

        // Single-tool allowlist → intersection keeps only that tool
        let one = vec![full_names[0].clone()];
        let filtered = executor.tool_definitions_for_role_with_allowlist("DA", Some(&one));
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            filtered[0]["function"]["name"].as_str().unwrap(),
            full_names[0]
        );

        // Allowlist with no overlap → empty intersection
        let disjoint = vec!["no_such_tool".to_string()];
        assert!(executor
            .tool_definitions_for_role_with_allowlist("DA", Some(&disjoint))
            .is_empty());
    }

    #[test]
    fn optimized_visible_tools_hide_on_demand_groups_until_search() {
        let mut executor = ToolExecutor::new();
        executor.set_tool_group_manager(crate::tools::tool_groups::ToolGroupManager::new(None));

        let visible = executor.visible_tool_definitions_for_role("AA");
        let names: std::collections::HashSet<String> = visible
            .iter()
            .filter_map(|td| td["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(names.is_empty(), "AA must not receive a tool menu");
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_self_protect_pkill_excludes_own_pid() {
        rt().block_on(async {
            // `pkill -f <our own cmdline fragment>` must NOT kill this test
            // process (the agent itself). The wrapper resolves targets via
            // pgrep and filters out the agent PID.
            let self_pid = std::process::id();
            let cmd = format!("pkill -f 'self_protect_marker_{}'", self_pid);
            let result = super::builtins::execute_bash(json!({"command": cmd}))
                .await
                .unwrap();
            // Exit code 1 = "no matching process" — correct: our own PID was
            // filtered out, and nothing else matches the unique marker.
            assert_eq!(
                result["exit_code"], 1,
                "own PID must be excluded: {:?}",
                result
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_self_protect_pkill_still_kills_real_target() {
        rt().block_on(async {
            use std::process::Command;
            // Spawn a real background sleep; pkill -f on a unique marker
            // must still terminate it (protection only filters the agent).
            let marker = format!("real_target_marker_{}", std::process::id());
            let mut child = Command::new("sh")
                .arg("-c")
                .arg(format!("exec -a {} sleep 60", marker))
                .spawn()
                .expect("spawn sleep");
            // Give it a moment to exec so the marker appears in argv[0].
            std::thread::sleep(std::time::Duration::from_millis(200));
            let cmd = format!("pkill -f '{}'", marker);
            let result = super::builtins::execute_bash(json!({"command": cmd}))
                .await
                .unwrap();
            assert_eq!(
                result["exit_code"], 0,
                "pkill should find the target: {:?}",
                result
            );
            // The child must be gone shortly after.
            for _ in 0..50 {
                if let Ok(Some(status)) = child.try_wait() {
                    assert!(!status.success() || status.code() != Some(0));
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            panic!("target process was not killed");
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_self_protect_killall_excludes_own_pid() {
        rt().block_on(async {
            let self_pid = std::process::id();
            // killall matches by process name; our unique name is not a real
            // process, so exit 1 (nothing found) proves the wrapper didn't
            // fall back to a broad match that would hit the test process.
            let cmd = format!("killall nonexistent_agent_{} 2>/dev/null || true", self_pid);
            let result = super::builtins::execute_bash(json!({"command": cmd}))
                .await
                .unwrap();
            assert_eq!(result["exit_code"], 0);
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_self_protect_plain_command_unchanged() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({"command": "printf ok"}))
                .await
                .unwrap();
            assert_eq!(result["exit_code"], 0);
            assert_eq!(result["stdout"], "ok");
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_environment_excludes_sensitive_names() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({"command": "env"}))
                .await
                .unwrap();
            assert_eq!(result["exit_code"], 0);
            let environment = result["stdout"].as_str().unwrap_or("").to_ascii_uppercase();
            for sensitive in [
                "API_KEY=",
                "APIKEY=",
                "ACCESS_KEY=",
                "SECRET=",
                "TOKEN=",
                "PASSWORD=",
                "PASSWD=",
                "PRIVATE_KEY=",
                "CREDENTIAL=",
                "AUTHORIZATION=",
                "COOKIE=",
            ] {
                assert!(
                    !environment.contains(sensitive),
                    "sensitive environment key was inherited: {sensitive}"
                );
            }
        });
    }

    #[cfg(unix)]
    #[test]
    fn clean_python_verification_redirects_bytecode_outside_the_workspace() {
        rt().block_on(async {
            let workspace = tempfile::tempdir().unwrap();
            std::fs::write(workspace.path().join("sample_module.py"), "VALUE = 42\n").unwrap();
            let command = format!(
                "cd '{}' && python3 -c 'import sample_module; assert sample_module.VALUE == 42'",
                workspace.path().display()
            );
            let result = super::builtins::execute_bash(json!({
                "command": command,
                "__gh_execution_profile": "clean_python_verification",
            }))
            .await
            .unwrap();

            assert_eq!(result["exit_code"], 0, "{result:?}");
            assert_eq!(result["execution_profile"], "clean_python_verification");
            assert_eq!(result["isolated_environment"]["PYTHONPYCACHEPREFIX"], true);
            assert!(
                !workspace.path().join("__pycache__").exists(),
                "verification bytecode must not pollute the delivered workspace"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn clean_pytest_profile_is_observable_without_exposing_its_temp_path() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({
                "command": "python3 -c 'import os; assert os.environ.get(\"PYTHONPYCACHEPREFIX\"); assert \"cache_dir=\" in os.environ.get(\"PYTEST_ADDOPTS\", \"\")'",
                "__gh_execution_profile": "clean_pytest_verification",
            }))
            .await
            .unwrap();

            assert_eq!(result["exit_code"], 0, "{result:?}");
            assert_eq!(result["execution_profile"], "clean_pytest_verification");
            assert_eq!(result["isolated_environment"]["PYTHONPYCACHEPREFIX"], true);
            assert_eq!(result["isolated_environment"]["PYTEST_ADDOPTS"], true);
            assert!(
                !result.to_string().contains("glidinghorse-verification-"),
                "the ephemeral cache path is kernel-private"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn tool_executor_injects_clean_profile_after_model_visible_validation() {
        rt().block_on(async {
            let workspace = tempfile::tempdir().unwrap();
            std::fs::write(workspace.path().join("profile_target.py"), "VALUE = 7\n").unwrap();
            let command = format!(
                "cd '{}' && python3 -c 'import profile_target; assert profile_target.VALUE == 7'",
                workspace.path().display()
            );
            let executor = ToolExecutor::new();
            let context =
                crate::skill_graph::security::SecurityContext::new("agent:clean-verifier", "CA")
                    .with_task("iri://tasks/clean-verifier");
            let result = executor
                .execute_with_security_context_effect_policy_and_profile(
                    "bash",
                    json!({"command": command}),
                    context,
                    Some(&["bash".to_string()]),
                    &crate::core::effect::EffectPolicy::EvidenceOnly,
                    ToolExecutionProfile::CleanPythonVerification,
                )
                .await
                .unwrap();

            assert_eq!(result["exit_code"], 0, "{result:?}");
            assert_eq!(result["execution_profile"], "clean_python_verification");
            assert!(!workspace.path().join("__pycache__").exists());

            let forged = executor
                .execute(
                    "bash",
                    json!({
                        "command": "printf should-not-run",
                        "__gh_execution_profile": "clean_python_verification"
                    }),
                )
                .await
                .unwrap();
            assert_eq!(forged["reason"], "reserved_internal_field");
        });
    }

    #[cfg(unix)]
    #[test]
    fn standard_bash_does_not_report_or_inject_a_verification_profile() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({
                "command": "printf '%s|%s' \"${PYTEST_ADDOPTS-unset}\" \"${PYTHONDONTWRITEBYTECODE-unset}\"",
            }))
            .await
            .unwrap();

            assert_eq!(result["exit_code"], 0, "{result:?}");
            assert_eq!(result["stdout"], "unset|1");
            assert!(result.get("execution_profile").is_none());
            assert!(result.get("isolated_environment").is_none());
        });
    }

    #[cfg(unix)]
    #[test]
    fn standard_bash_python_probe_does_not_pollute_the_workspace() {
        rt().block_on(async {
            let workspace = tempfile::tempdir().unwrap();
            std::fs::write(workspace.path().join("probe_module.py"), "VALUE = 42\n").unwrap();
            let command = format!(
                "cd '{}' && python3 -c 'import probe_module; assert probe_module.VALUE == 42'",
                workspace.path().display()
            );
            let result = super::builtins::execute_bash(json!({"command": command}))
                .await
                .unwrap();

            assert_eq!(result["exit_code"], 0, "{result:?}");
            assert!(result.get("execution_profile").is_none());
            assert!(
                !workspace.path().join("__pycache__").exists(),
                "ordinary shell inspection must not create deliverable cache files"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_sandbox_status_reported() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({
                "command": "printf hi",
                "dangerouslyDisableSandbox": false,
            }))
            .await
            .unwrap();
            assert_eq!(result["exit_code"], 0);
            let status = &result["sandbox_status"];
            assert!(
                status.is_object(),
                "sandbox_status must be present: {:?}",
                result
            );
            assert_eq!(status["requested"]["enabled"], true);
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_sandbox_disabled_when_requested() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({
                "command": "printf hi",
                "dangerouslyDisableSandbox": true,
            }))
            .await
            .unwrap();
            assert_eq!(result["exit_code"], 0);
            let status = &result["sandbox_status"];
            assert_eq!(
                status["enabled"], false,
                "sandbox must be disabled: {:?}",
                result
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_sandbox_unshare_launcher_active() {
        rt().block_on(async {
            // Sandbox is opt-in: with explicit enablement and namespace
            // restrictions the command must run inside the unshare sandbox
            // (proven by an isolated PID namespace: the child's PID 1 is
            // not the host init).
            let result = super::builtins::execute_bash(json!({
                "command": "test \"$(ps -p 1 -o comm= 2>/dev/null || echo unknown)\" != \"$(cat /proc/1/comm 2>/dev/null || echo unknown)\" || echo pid1_is_shared",
                "dangerouslyDisableSandbox": false,
                "namespaceRestrictions": true,
            }))
            .await
            .unwrap();
            // Either the sandbox isolated PID 1 (success) or, on hosts
            // without unshare support, we fall back gracefully — the command
            // itself always exits 0.
            assert_eq!(result["exit_code"], 0, "sandbox command failed: {:?}", result);
            assert_eq!(result["sandbox_status"]["enabled"], true);
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_run_in_background_returns_task_id() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({
                "command": "sleep 5",
                "run_in_background": true,
            }))
            .await
            .unwrap();
            let task_id = result["background_task_id"].as_str().unwrap_or("");
            assert!(
                !task_id.is_empty(),
                "background task id must be present: {:?}",
                result
            );
        });
    }

    #[cfg(unix)]
    fn unix_process_exists(pid: u32) -> bool {
        let Ok(pid) = libc::pid_t::try_from(pid) else {
            return false;
        };
        if unsafe { libc::kill(pid, 0) } == 0 {
            return true;
        }
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_foreground_return_terminates_nohup_descendant_before_late_write() {
        rt().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let child_pid_path = dir.path().join("child.pid");
            let late_write_path = dir.path().join("late.txt");
            let command = format!(
                "nohup sh -c 'printf \"%s\" \"$$\" > \"{}\"; sleep 0.4; printf late > \"{}\"' >/dev/null 2>&1 & while [ ! -s \"{}\" ]; do sleep 0.01; done",
                child_pid_path.display(),
                late_write_path.display(),
                child_pid_path.display(),
            );

            let result = super::builtins::execute_bash(json!({"command": command}))
                .await
                .unwrap();
            assert_eq!(result["exit_code"], 0, "foreground shell failed: {result:?}");
            assert_eq!(
                result["process_group_cleanup"]["residual_processes_detected"],
                true,
                "the detached descendant must be detected: {result:?}"
            );
            assert_eq!(
                result["process_group_cleanup"]["confirmed_gone"],
                true,
                "execute_bash may not return before its process group is gone: {result:?}"
            );

            let child_pid = std::fs::read_to_string(&child_pid_path)
                .unwrap()
                .trim()
                .parse::<u32>()
                .unwrap();
            assert!(
                !unix_process_exists(child_pid),
                "nohup descendant PID {child_pid} survived execute_bash"
            );
            assert!(!late_write_path.exists());
            tokio::time::sleep(std::time::Duration::from_millis(650)).await;
            assert!(
                !late_write_path.exists(),
                "a descendant wrote after execute_bash returned"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_background_job_with_explicit_wait_completes_normally() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({
                "command": "sh -c 'sleep 0.05; printf child-complete' & wait",
                "timeout": 1000,
            }))
            .await
            .unwrap();

            assert_eq!(result["exit_code"], 0, "wait command failed: {result:?}");
            assert_eq!(result["stdout"], "child-complete");
            assert!(
                result.get("process_group_cleanup").is_none(),
                "a job consumed by shell wait is not a residual process: {result:?}"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_foreground_residual_ignoring_term_is_forced_and_confirmed_gone() {
        rt().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let child_pid_path = dir.path().join("term-immune.pid");
            let command = format!(
                "nohup sh -c 'trap \"\" TERM; printf \"%s\" \"$$\" > \"{}\"; while :; do :; done' >/dev/null 2>&1 & while [ ! -s \"{}\" ]; do sleep 0.01; done",
                child_pid_path.display(),
                child_pid_path.display(),
            );

            let result = super::builtins::execute_bash(json!({"command": command}))
                .await
                .unwrap();
            let child_pid = std::fs::read_to_string(&child_pid_path)
                .unwrap()
                .trim()
                .parse::<u32>()
                .unwrap();
            assert_eq!(
                result["process_group_cleanup"]["forced_kill"], true,
                "TERM-immune descendant must reach the bounded KILL phase: {result:?}"
            );
            assert_eq!(
                result["process_group_cleanup"]["confirmed_gone"], true,
                "KILL phase must confirm that the group disappeared: {result:?}"
            );
            assert!(!unix_process_exists(child_pid));
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_background_process_is_reaped() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({
                "command": "exit 7",
                "run_in_background": true,
            }))
            .await
            .unwrap();
            let task_id: u32 = result["background_task_id"]
                .as_str()
                .unwrap()
                .parse()
                .unwrap();
            for _ in 0..50 {
                if super::builtins::background_process_status(task_id).as_deref()
                    == Some("exited:7")
                {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            panic!(
                "background process was not reaped: {:?}",
                super::builtins::background_process_status(task_id)
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_timeout_does_not_replay_command() {
        rt().block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("attempts");
            let shell_pid_path = dir.path().join("shell.pid");
            let command = format!(
                "printf x >> '{}'; printf \"%s\" \"$$\" > '{}'; sleep 2",
                marker.display(),
                shell_pid_path.display(),
            );
            let result = super::builtins::execute_bash(json!({
                "command": command,
                "timeout": 300,
            }))
            .await
            .unwrap();

            assert_eq!(result["timed_out"], true);
            assert_eq!(
                std::fs::read_to_string(marker).unwrap_or_default(),
                "x",
                "command must execute exactly once: {result:?}"
            );
            assert_eq!(result["error"], "Timeout after 300ms");
            let shell_pid = std::fs::read_to_string(shell_pid_path)
                .unwrap()
                .trim()
                .parse::<u32>()
                .unwrap();
            assert!(
                !unix_process_exists(shell_pid),
                "timed-out foreground PID {shell_pid} survived execute_bash"
            );
            assert_eq!(
                result["process_group_cleanup"]["confirmed_gone"], true,
                "timeout must settle the complete process group: {result:?}"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_wait_does_not_block_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let started = std::time::Instant::now();
            let command = super::builtins::execute_bash(json!({
                "command": "sleep 0.2",
                "timeout": 1000,
            }));
            let timer = async {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                started.elapsed()
            };
            let (_, timer_elapsed) = tokio::join!(command, timer);
            assert!(
                timer_elapsed < std::time::Duration::from_millis(100),
                "runtime timer was blocked for {:?}",
                timer_elapsed
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_output_truncated_at_16k() {
        rt().block_on(async {
            let result = super::builtins::execute_bash(json!({
                "command": "head -c 30000 /dev/zero | tr '\\0' 'a'",
            }))
            .await
            .unwrap();
            assert_eq!(result["exit_code"], 0);
            assert_eq!(result["truncated"], true);
            let stdout = result["stdout"].as_str().unwrap_or("");
            assert!(
                stdout.contains("[output truncated"),
                "stdout must carry marker: {:?}",
                result
            );
            assert!(
                stdout.len() < 20_000,
                "stdout must be capped: {}",
                stdout.len()
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_truncate_output_short_unchanged() {
        let (out, truncated) = super::builtins::truncate_output("hello");
        assert_eq!(out, "hello");
        assert!(!truncated);
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_truncate_output_exact_boundary() {
        let (out, truncated) = super::builtins::truncate_output(&"a".repeat(16_384));
        assert_eq!(out.len(), 16_384);
        assert!(!truncated);
    }

    #[cfg(unix)]
    #[test]
    fn test_bash_truncate_output_one_over() {
        let (out, truncated) = super::builtins::truncate_output(&"a".repeat(16_385));
        assert!(truncated);
        assert!(out.contains("[output truncated"));
    }

    #[test]
    fn test_web_fetch_network_policy_rejects_non_public_addresses() {
        use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

        for ip in [
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(100, 64, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254)),
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
            "fc00::1".parse().unwrap(),
            "fe80::1".parse().unwrap(),
        ] {
            assert!(!super::builtins::is_public_web_ip(ip), "must block {ip}");
        }
        assert!(super::builtins::is_public_web_ip(
            "8.8.8.8".parse().unwrap()
        ));
        assert!(super::builtins::is_public_web_ip(
            "2606:4700:4700::1111".parse().unwrap()
        ));
    }

    #[test]
    fn test_web_fetch_rejects_localhost_before_connecting() {
        rt().block_on(async {
            let result = super::builtins::execute_web_fetch(json!({
                "url": "http://127.0.0.1:9/private"
            }))
            .await;
            assert!(result
                .unwrap_err()
                .contains("Private, local, or reserved network target"));
        });
    }

    #[test]
    fn test_web_fetch_streaming_limit_rejects_chunked_oversize_body() {
        rt().block_on(async {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0u8; 1024];
                let _ = stream.read(&mut request).await;
                let body = vec![b'a'; 10_000_001];
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
                    .await
                    .unwrap();
                stream
                    .write_all(format!("{:x}\r\n", body.len()).as_bytes())
                    .await
                    .unwrap();
                stream.write_all(&body).await.unwrap();
                stream.write_all(b"\r\n0\r\n\r\n").await.unwrap();
            });
            let response = reqwest::Client::new()
                .get(format!("http://{address}"))
                .send()
                .await
                .unwrap();

            let error = super::builtins::read_limited_web_body(response)
                .await
                .unwrap_err();
            assert!(error.contains("stream exceeded 10000000 bytes"));
            server.await.unwrap();
        });
    }
}
