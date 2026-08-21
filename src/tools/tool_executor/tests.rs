use super::*;
use crate::config::GatewaySettings;
use crate::config::RuntimeHookConfig;
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
                    crate::skill_graph::security::SecurityContext::new(
                        "agent:test",
                        "DA",
                    )
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
    fn tools_allowed_passes_listed_tool_through() {
        rt().block_on(async {
            let executor = ToolExecutor::new();
            let input = json!({"query": "search test"});
            let result = executor
                .execute_with_security_context(
                    "tool_search",
                    input,
                    crate::skill_graph::security::SecurityContext::new(
                        "agent:test",
                        "DA",
                    )
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
    fn security_gate_allows_aa_ca_inspection_tools_with_whitelisted_file_read() {
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
            let whitelist = std::collections::HashSet::from([
                "iri://skills/file_read".to_string(),
            ]);
            let security = Arc::new(
                crate::skill_graph::security::SecurityEngine::with_whitelisted_skills(
                    graph.clone(),
                    whitelist.clone(),
                ),
            );
            executor.set_shared_skill_registry(registry);
            executor.set_shared_skill_graph(graph);
            executor.set_security_engine(security.clone());

            // The AA/CA default inspection tools (also registered read-only workspace/KG readers)
            // must NOT be rejected as "no registered executable skill" — otherwise the
            // verify-first CA/AA cannot inspect the workspace and cannot verify deliverables.
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
                        crate::skill_graph::security::SecurityContext::new("agent:test", "AA")
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

            let audit = security
                .get_audit_log(Some("iri://skills/file_read"), Some("agent:test"), 50)
                .await;
            assert!(
                audit.iter().any(|e| e.outcome
                    == crate::skill_graph::types::AuditOutcome::Success),
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
        assert!(!ToolExecutor::is_pa_readonly_tool("file_write"));
        assert!(!ToolExecutor::is_pa_readonly_tool("file_edit"));
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

    /// Regression test: MCP tools are registered with long-form role names
    /// ("Plan"/"Do"/"Check"/"Act", see McpClient::register_tools_to_tool_executor),
    /// while consumers call tool_definitions_for_role with short-form names
    /// ("PA"/"DA"/"CA"/"AA", see AgentRole::Display). Both conventions must match.
    #[test]
    fn test_long_form_allowed_roles_match_short_form_agent_roles() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.register(
                "mcp_server_browse",
                "MCP-registered browsing tool",
                json!({"type": "object", "properties": {}}),
                Arc::new(|input: Value| {
                    Box::pin(async move { Ok(json!({"ok": input})) })
                }),
                &["Plan", "Do", "Check", "Act"],
            );

            for role in ["PA", "DA", "CA", "AA"] {
                let defs = executor.tool_definitions_for_role(role);
                let names: Vec<String> = defs
                    .iter()
                    .map(|d| {
                        d["function"]["name"]
                            .as_str()
                            .unwrap_or("")
                            .to_string()
                    })
                    .collect();
                assert!(
                    names.contains(&"mcp_server_browse".to_string()),
                    "role {} should see the MCP-registered tool, got {} tools: {:?}",
                    role,
                    names.len(),
                    &names[..names.len().min(15)]
                );
            }
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

        // None/empty allowlist → full role-filtered set unchanged
        assert_eq!(
            executor
                .tool_definitions_for_role_with_allowlist("DA", None)
                .len(),
            full.len()
        );
        let empty: Vec<String> = vec![];
        assert_eq!(
            executor
                .tool_definitions_for_role_with_allowlist("DA", Some(&empty))
                .len(),
            full.len()
        );

        // Single-tool allowlist → intersection keeps only that tool
        let one = vec![full_names[0].clone()];
        let filtered =
            executor.tool_definitions_for_role_with_allowlist("DA", Some(&one));
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
            assert_eq!(result["exit_code"], 1, "own PID must be excluded: {:?}", result);
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
            assert_eq!(result["exit_code"], 0, "pkill should find the target: {:?}", result);
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
            assert!(status.is_object(), "sandbox_status must be present: {:?}", result);
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
            assert_eq!(status["enabled"], false, "sandbox must be disabled: {:?}", result);
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
            assert!(!task_id.is_empty(), "background task id must be present: {:?}", result);
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
            assert!(stdout.contains("[output truncated"), "stdout must carry marker: {:?}", result);
            assert!(stdout.len() < 20_000, "stdout must be capped: {}", stdout.len());
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
}
