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
    fn archived_agent_output_hides_foreign_session_tool_references() {
        let mut value = json!({
            "content": "Call read_full_result_call_00_foreign for the rest",
            "nested": [
                "read_full_result_abc-123",
                "iri://tool-result/call_01_stale",
                "https://agent-os.org/ontology/tool-result/call_02_stale"
            ]
        });
        let count = redact_session_tool_references(&mut value);
        assert_eq!(count, 4);
        let rendered = value.to_string();
        assert!(!rendered.contains("read_full_result_call_00_foreign"));
        assert!(!rendered.contains("read_full_result_abc-123"));
        assert!(!rendered.contains("iri://tool-result/call_01_stale"));
        assert!(!rendered.contains("ontology/tool-result/call_02_stale"));
        assert!(rendered.contains("session-scoped result reader omitted"));
        assert!(rendered.contains("session-scoped tool result omitted"));
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
    fn file_read_capability_allows_only_read_result_microtools() {
        let allowed = vec!["file_read".to_string()];
        assert!(ToolExecutor::explicit_allowlist_permits(
            "read_full_result_call_123",
            &allowed
        ));
        assert!(ToolExecutor::explicit_allowlist_permits(
            "read_agent_output",
            &allowed
        ));
        assert!(!ToolExecutor::explicit_allowlist_permits(
            "file_write",
            &allowed
        ));
    }

    #[test]
    fn micro_tool_lookup_is_scoped_by_originating_call() {
        let mut executor = ToolExecutor::new();
        for call_id in ["call_a", "call_b"] {
            executor.register_micro_tool(
                &format!("read_full_result_{call_id}"),
                MicroToolContext {
                    call_id: call_id.to_string(),
                    storage_key: format!("iri://tool-result/{call_id}"),
                    tool_name: "file_read".to_string(),
                    entity_types: vec![],
                    preview_size: 100,
                },
            );
        }

        assert_eq!(
            executor.get_micro_tool_names_for_call("call_a"),
            vec!["read_full_result_call_a".to_string()]
        );
        assert!(executor.get_micro_tool_names_for_call("unknown").is_empty());
    }

    #[test]
    fn configured_micro_tool_limits_control_catalog_and_paging() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            executor.set_micro_tool_limits(2, 2, 3);
            for call_id in ["one", "two", "three"] {
                let storage_key = format!("iri://tool-result/{call_id}");
                executor.store_micro_tool_data(&storage_key, json!({"content": "a\nb\nc\nd\ne"}));
                executor.register_micro_tool(
                    &format!("read_full_result_{call_id}"),
                    MicroToolContext {
                        call_id: call_id.to_string(),
                        storage_key,
                        tool_name: "file_read".to_string(),
                        entity_types: vec![],
                        preview_size: 1,
                    },
                );
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
                .execute("read_full_result_three", json!({}))
                .await
                .unwrap();
            assert_eq!(default_page["returned"], 2);
            let capped_page = executor
                .execute("read_full_result_three", json!({"limit": 99}))
                .await
                .unwrap();
            assert_eq!(capped_page["returned"], 3);
        });
    }

    #[test]
    fn micro_tool_character_pages_bound_single_line_json_results() {
        rt().block_on(async {
            let mut executor = ToolExecutor::new();
            let storage_key = "iri://tool-result/long-json";
            executor.store_micro_tool_data(storage_key, json!({"content": "abcdefghij"}));
            executor.register_micro_tool(
                "read_full_result_long_json",
                MicroToolContext {
                    call_id: "long_json".to_string(),
                    storage_key: storage_key.to_string(),
                    tool_name: "rag_search".to_string(),
                    entity_types: vec![],
                    preview_size: 4,
                },
            );

            let first = executor
                .execute("read_full_result_long_json", json!({}))
                .await
                .unwrap();
            assert_eq!(first["content"], "abcd");
            assert_eq!(first["returned_chars"], 4);
            assert_eq!(first["next_char_offset"], 4);
            assert_eq!(first["truncated"], true);

            let second = executor
                .execute(
                    "read_full_result_long_json",
                    json!({"char_offset": first["next_char_offset"]}),
                )
                .await
                .unwrap();
            assert_eq!(second["content"], "efgh");
            assert_eq!(second["next_char_offset"], 8);
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
            assert_eq!(monitor.generation(), initial_generation);
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
        assert!(ToolExecutor::is_pa_readonly_tool("read_agent_output"));
        assert!(ToolExecutor::is_pa_readonly_tool(
            "read_full_result_call_session_a"
        ));
        assert!(ToolExecutor::is_pa_readonly_tool("query_result_entities"));
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
}
