# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Opt-in unshare sandbox for the bash tool**: the registered `ToolExecutor` bash now supports per-command sandboxing via `dangerouslyDisableSandbox` / `namespaceRestrictions` / `isolateNetwork` / `filesystemMode` / `allowedMounts` overrides. When enabled (opt-in, disabled by default to keep pkill/pgrep able to manage host processes across the namespace boundary), commands run under `unshare` (user/mount/pid/ipc/uts/net namespaces) with sandboxed `HOME`/`TMPDIR`. The effective `SandboxStatus` is returned in the `sandbox_status` field. Also migrated in: 16KB output truncation (`truncated` / `original_size` fields) and background execution (`run_in_background` → immediate `background_task_id`).
  - `src/tools/tool_executor/builtins.rs` (+139), `src/tools/tool_executor/mod.rs` (bash schema), `src/tools/builtin/sandbox.rs` (+11, `ensure_sandbox_dirs`), `src/tools/tool_executor/tests.rs` (+112, 8 new tests)

- **Bash pkill/killall self-protection**: commands mentioning `pkill`/`killall` are wrapped with shell-function overrides that resolve targets via `pgrep` and exclude the agent's own PID and the wrapper shell PID before signaling — the DA can clean up spawned processes without killing the agent OS process itself (whose command line embeds the task prompt, including the same file names it pkill -f's).
  - `src/tools/tool_executor/builtins.rs` (4 unit tests)

### Changed

- **Unified bash implementation**: the sandbox-capable orphan `src/tools/builtin/bash.rs` (352 lines, zero callers) was merged into the registered `tool_executor` bash — the single bash now carries sandboxing, truncation, background execution, pkill self-protection, timeout retry, process-group management, permission policy/hooks integration, and Windows PowerShell delegation together, eliminating the previous split where security lived on the live path and isolation on a dead one.
  - `src/tools/tool_executor/builtins.rs`

### Removed

- **`src/tools/builtin/bash.rs`**: deleted after its capabilities were migrated into `tool_executor`; `pub mod bash` removed from `src/tools/builtin/mod.rs`. `src/tools/builtin/sandbox.rs` remains as a pure logic library (container detection, status resolution, unshare launcher construction, `ensure_sandbox_dirs`).

- **Edge daemon Docker sandbox repositioned (not removed)**: `SandboxManager` (container lifecycle: create/exec/destroy/list) is retained and documented as the reserved heavy-weight container isolation layer — complementary to the process-level unshare sandbox, not a replacement. Wired into nothing yet; enabling requires instantiating it in the daemon agent execution path.
  - `apps/software_engineering_team/edge/daemon/src/sandbox/mod.rs` (module doc), `apps/software_engineering_team/README.md`

### Added

- **Methodology usage-window settlement**: `MethodologyGate` now captures a settled `MethodologyUsageRecord` (methodology → task → agent → success/duration/error) for every activation window opened at `SkillBefore` and closed at `SkillAfter` (success) or `TaskError` (failure). Settled methodologies leave the active set so the next qualifying tool call re-activates them. `usage_history()` / `usage_count()` expose the records.
  - `src/methodology/gate.rs` (+443), `tests/verify_gate_activation.rs` (new, runtime activation verification)

- **Governed methodology evolution**: `suggest_methodology_adjustments()` recommends a `Methodology`-typed `EvolutionSuggestion` when a methodology's settled success rate falls below a threshold (min samples / min success rate). New `EvolutionPatch::Methodology` + `EvolutionSuggestionType::Methodology` flow through the shared `EvolutionProposalStore` approval/commit lifecycle via synthetic IRIs (`iri://methodology/<id>`), never touching the skill graph — methodology proposals are committed directly on approval.
  - `src/methodology/gate.rs`, `src/skill_graph/evolution.rs` (+174), `src/methodology/evolution.rs`

- **Methodology cold-archiving**: `archive_cold_methodologies()` / `archive_methodologies()` exclude archived IDs from activation entirely; `find_cold_methodologies()` in the evolution engine detects methodologies unused since a cutoff. Manual `activate()` also respects the archive.
  - `src/methodology/gate.rs`, `src/methodology/evolution.rs`

- **JSON-LD methodology nodes**: `MethodologyRegistry::load_from_jsonld_dir` / `load_from_jsonld_file` / `load_bundled_nodes` load methodology definitions from `src/methodology/nodes/*.jsonld` (7 bundled superpowers-style definitions: brainstorming, executing-plans, dispatching-parallel-agents, writing-plans, requesting/receiving-code-review, finishing-a-development-branch, subagent-driven-development). Same-ID entries replace builtins; missing dir falls back gracefully with a warning.
  - `src/methodology/mod.rs` (+651), `src/methodology/nodes/` (new, 7 JSON-LD files), `src/core/agent_runner/mod.rs`

- **Constitution hook-point rename**: methodology bindings now use the real hook point names (`skill_before`, `skill_after`, `phase_start`, `task_start`) instead of legacy aliases (`PreToolCall`, `PostToolCall`, `PrePlanCreation`, `TaskProgress`, `PreDestructiveAction`) — making the triggers actually fire.
  - `src/core/constitution.rs` (18 changed)

- **External skill directory loading**: `SkillRegistry::load_from_jsonld_dir` merges skills from a `skills/*/skill.jsonld` layout; new `--skill-dir <DIR>` CLI flag loads external skills into the skill graph before the bootstrap loop.
  - `src/tools/skill_registry.rs` (+85), `apps/gliding_code/src/main.rs`, `apps/gliding_code/src/config.rs`, `apps/gliding_code/src/engine.rs`

- **Real MCP tool dispatch into ToolExecutor**: `McpClient::register_tools_to_tool_executor` registers every connected server tool as a real `ToolExecutor` tool (Plan/Do/Check/Act) dispatching through the shared lazy `Arc<Mutex<Option<McpClient>>>` handle; `SupervisorAgent::tool_executor()` exposes the executor for runtime registration. The simulated-result fallbacks in HTTP/stdio `call_tool` were removed — transport failures now surface as real errors. New integration tests lock real dispatch (no simulated results).
  - `src/tools/mcp_client.rs` (+64), `src/core/sa/agent.rs`, `apps/gliding_code/src/engine.rs`, `tests/mcp_integration_test.rs` (+88)

- **DA workspace file manifest injection**: the DA system prompt gains a `Workspace Files` section (path + bytes + cached line count, sorted) so the executor no longer needs `file_list` + per-file reads on turn one. `build_workspace_file_manifest()` reads the workspace monitor inventory + content cache.
  - `src/core/agent_runner/prompt.rs` (+60)

- **`file_read` result preview routing**: new `RouteDecision::FileReadPreview` — small files (≤300 lines AND ≤4KB) pass through fully; larger files return the JSON skeleton (path/total_lines) + first 200 lines inline with a `read_full_result_<call_id>` micro-tool for the rest. The 32KB/1000-line exemption gap (a 32KB file with <1000 lines previously entered context whole) is closed.
  - `src/tools/result_router/mod.rs`, `src/tools/result_router/router.rs` (+83), `src/core/agent_runner/utils.rs` (+57)

- **CJK-aware token estimation**: `estimate_text_tokens()` counts CJK characters as 1 token each (real tokenizers cost them ~1 token/char) and other bytes at 4-bytes-per-token, replacing the raw `len()/4` that undervalued Chinese/Japanese/Korean content. `ContextWindowManager` and tool-result summaries now budget CJK correctly.
  - `src/core/context_compressor.rs` (+151)

- **Checkpoint retention cap**: `MAX_CHECKPOINTS_PER_TASK = 20` — `prune_oldest()` evicts the oldest checkpoint (L0 physical delete) on create, bounding L0 growth for long multi-turn tasks. Regression test locks retention + restorability.
  - `src/core/checkpoint.rs` (+60)

- **Batch trigger custom-event enqueue**: custom event-bus events are now pushed into the sliding window as real `WindowEntry`s (with event_type/task_iri/source metadata) instead of a no-op wakeup.
  - `src/batch/trigger.rs` (+57)

- **Perception task-key reclamation**: a task key is removed from the pending perception map once all its entries are consumed, bounding memory for long-running tasks.
  - `src/core/perception_store.rs` (+30)

- **Embedding failure warn-once**: OneAPI/Ollama embedding services log the failure warning exactly once per process (`AtomicBool` gate) instead of spamming on every search/upsert when the backend is unreachable.
  - `src/memory/embedding_service.rs` (+56)

- **KG context LIMIT 50 → 500**: the L0 entity-injection SPARQL query widened so task-relevant entities are not truncated out of context; regression test proves an entity sorting beyond label top-50 is still injected.
  - `src/core/agent_runner/execution.rs`

- **Tool-summary entry cap**: force-finish summary aggregates at most the 20 most recent assistant→tool pairs (`MAX_AGGREGATE_TOOL_ENTRIES`), bounding the summary prompt for long tasks.
  - `src/core/agent_runner/execution.rs`

- **Workspace perception lists paths**: `generate_perception_text` now names up to 10 newly discovered unread files instead of just a count.
  - `src/tools/workspace_monitor/mod.rs` (+40)

- **Experience-hint merge**: `ProactiveEngine` merges semantic experience hints into (instead of overwriting) perception/workspace hints already present — workspace events survive alongside skill-discovered scenarios.
  - `src/perception/proactive_engine.rs` (+30)

- **SA prompt dedup + cap**: all experience hints (perception + skill discovery) are deduplicated preserving first-seen order and capped at `MAX_ALL_HINTS = 20` before injection into the supervisor prompt.
  - `src/core/sa/process.rs` (+74)

- **Explicit Responses-API incomplete error**: a DeepSeek Responses response whose reasoning consumed the whole `max_output_tokens` budget (no `message` block) now surfaces as a real error instead of a misleading empty `content: None`.
  - `src/gateway/unified_gateway.rs` (+21)

### Changed

- **SA LLM output budgets raised**: recursive decomposition 1000→8192, intervention 1000→4096, planning 500→4096 / 2000→8192, complexity 200→4096 tokens — SA stages no longer truncate long JSON plans.
  - `src/core/sa/execution.rs`, `src/core/sa/intervention.rs`, `src/core/sa/planning.rs`

- **SSE tool-call parsing**: stream parsing now handles the first tool call of each `tool_calls` delta instead of iterating a usually-single-element array.
  - `src/llm/sse.rs`

### Fixed

- **Workspace-monitor feedback loop (P0)**: the watcher's own graph writes re-triggered the watcher via mirrored debug logs, causing an unbounded self-sustaining loop (951 graph writes / ~6 min, continuing after task end). Per-path cooldown (`MIN_EVENT_INTERVAL = 1000ms`, silent skip within window) in native + polling paths breaks the loop at the source; writes now stop at task completion.
  - `src/tools/workspace_monitor/watch_engine.rs` (+161)

- **Dead event emit (P0)**: `let _ = event_bus.emit(...)` on the async `emit` dropped the future without polling — `sender.send` never ran, so native watcher events were never delivered (consumer evidence was 0). Native path now spawns via a captured runtime handle; polling path awaits directly. Perception/consistency consumers fire again.
  - `src/tools/workspace_monitor/watch_engine.rs`

- **TemplateEngine symlink ELOOP**: recursion used `path.is_dir()` (follows symlinks); a self-referential symlink chain in `/tmp` caused infinite recursion / `os error 40`, failing `AgentOsWorker::new` and two worker tests. All three scan sites now use `entry.file_type()` (no symlink following); regression test `test_new_skips_symlink_loop`.
  - `src/templates/template_engine.rs` (+62)

- **Knowledge-graph tools falsely denied**: `knowledge_query` / `knowledge_neighbors` (read-only SPARQL/neighbor queries) were missing from `builtin_security_skill_iri`, so the fail-closed gate rejected them with "Security denied: tool has no registered executable skill" while their siblings (`kg_search`, `knowledge_search`) were whitelisted. Added to the table; the agent's KG queries now succeed (0 denials in verification runs).
  - `src/tools/tool_executor/mod.rs` (+1)

- **Shutdown log mislabeled as ERROR**: the workspace-monitor event-bus closure (normal shutdown path) logged `error!`; now `debug!`. Final runs show 0 real ERROR lines.
  - `src/tools/workspace_monitor/mod.rs` (+2)

- **`fnv` test weakened**: the old assertion `assert_ne!(fnv_hash("a"), fnv_hash("b"))` was a coincidental hash collision assumption; replaced with a warn-once behavioral test.

### Added

- **Responses API native support**: new `use_responses_api` config option (default on for `deepseek-v4-flash`, other models fall back to chat completions). Includes full request/response conversion (messages → input items, tool_choice parsing, tool_call → function_call / custom_tool_call mapping, reasoning content extraction), OpenAI Responses semantic stream event parsing (`response.created` / `output_text.delta` / `reasoning_text.delta` / `function_call_arguments.delta` / `completed|incomplete|failed`, including streams without `data:[DONE]` termination), and shared retry logic across chat completions and responses paths. New `USE_RESPONSES_API` env var (compatible with `AGENT_OS_GATEWAY_USE_RESPONSES_API`).
  - `src/gateway/unified_gateway.rs` (+678), `src/llm/sse.rs` (+323), `src/config/settings.rs`, `apps/gliding_code/src/config.rs`, `config.yaml`

- **Memory WriteThrough consistency hook**: `Blackboard::set_write_hook` — `write_node_to_graph` now triggers the consistency engine carrying node tags; key-tagged nodes are flushed to L0 immediately and L3 projection cache invalidated (WriteThrough semantics). The redundant `on_l2_write` in `scheduler.on_task_complete` was removed (regression test locks the unique L3 invalidation path).
  - `src/memory/consistency_engine.rs` (+85), `src/memory/l2_blackboard.rs` (+163), `src/memory/scheduler.rs` (−3)

- **Memory prefetch event bus wiring**: `PrefetchEngine::spawn_consumer` now subscribes to `MEMORY_PREFETCH` / `PREFETCH_REQUEST` events, refreshes the entity knowledge graph, and drains the queue. 6 new PrefetchEngine unit tests.
  - `src/memory/prefetch_engine.rs` (+163), `src/worker/agent_os_worker.rs` (+16)

- **L2 eviction over gRPC**: gRPC layer adds/updates L2 eviction fields and handling; obsolete proto fields marked `deprecated`. Remote management can now trigger L2 eviction.
  - `proto/pdca_core.proto` (+7), `src/api/grpc/server.rs`

- **Live-test stream retry**: new `collect_stream_retrying()` test helper — 2xx streams cut mid-flight (rate limit / network jitter) retry with exponential backoff (400ms × 2^n, max 3 retries). CI/live test suites no longer fail randomly on transient network jitter.
  - `src/gateway/unified_gateway.rs`

- **Single-task real-time log mirror**: `LogBuffer` gains `set_mirror_to_stderr` / `mirrors_to_stderr`; single-task (`--prompt`) mode without a TUI log panel writes every log line to stderr in real time — long tasks no longer appear hung with zero output.
  - `apps/gliding_code/src/log_buffer.rs` (+49), `apps/gliding_code/src/main.rs` (+16)

- **Workspace monitor wired to L2**: `WorkspaceMonitor::initialize` now injects `Some(l2/blackboard)` so file events (create/modify/delete) can be written to the L2 blackboard directly, closing the causal-observation ↔ file-monitoring loop.
  - `apps/gliding_code/src/engine.rs`, `src/worker/agent_os_worker.rs`

- **Scheduler full-pipeline e2e test**: in-memory scheduler end-to-end coverage to prevent scheduling regressions.
  - `tests/e2e_scheduler_pipeline_test.rs` (+209)

### Changed

- Integration tests adapted for the `use_responses_api` field so the suite stays stable across model switching.
  - `tests/` (`116c9b8`)

### Fixed

- **SA verify-first gate**: the verify-first PDCA cycle now runs the full PDCA (stored `fallback_steps`) unless the AA verdict explicitly confirms the task is already done. Previously the AA's `finish` action hardcoded `status: "success"`, which falsely short-circuited execution for tasks that genuinely needed it. Recovery is conservative by design: any missing/ambiguous/negative verdict means full execution.
  - `src/core/sa/types.rs` — new `verify_aa_needs_execution()` verdict parser (8 completion markers)
  - `src/core/sa/process.rs` — cycle-0 gate: skip completion only when `status == "success" && !needs_execution_after_verify`
  - `src/core/sa/tests.rs` — `test_verify_aa_needs_execution_parses_verdict`

- **`finish` reports real verdict**: the `finish` action now uses `detect_blocker_verdict()` (conservative match of explicit blocker wording such as `no task spec` / `zero deliverables` / `cannot proceed` / `missing task spec`) instead of a hardcoded `"success"`. Tasks judged blocked / zero-deliverable are honestly reported `failed`, so the SA PDCA retry loop actually triggers and the CLI no longer shows "fake success with no output".
  - `src/core/agent_runner/utils.rs`, `src/core/agent_runner/execution.rs`, `src/core/agent_runner/tests.rs` (4-assertion unit test)

- **Tool executor security IRI coverage**: the built-in security skill IRI table now whitelists the AA/CA read-only inspection tools (`file_list`, `workspace_status`, `rag_search`, `kg_search`, `codebase_search`, `knowledge_list`, `knowledge_search`, `knowledge_extract_code`). Previously these were rejected as "no registered executable skill", blocking verify-first CA/AA from inspecting the workspace and verifying deliverables.
  - `src/tools/tool_executor/mod.rs` — extended `builtin_security_skill_iri()`
  - `src/tools/tool_executor/tests.rs` — `security_gate_allows_aa_ca_inspection_tools_with_whitelisted_file_read`

- **L1 session turn embedding setter**: turn embedding writes no longer go through the deprecated assignment path, avoiding state inconsistency.
  - `src/memory/l1_session.rs` (+10/−1)

- **P2 thread leak in sync reaping**: `reap_finished_syncs` keeps pending oxigraph sync handles bounded, and `clear()` flushes oxigraph before clearing — long-running processes no longer accumulate oxigraph background threads.
  - `src/memory/l2_blackboard.rs`

- **P0 orphaned event bus**: the worker previously used a private EventBus, so prefetch/consistency events were consumed by nobody. It now shares the main EventBus with upper layers.
  - `src/worker/agent_os_worker.rs`

## [0.1.4.preview] - 2026

### Added

- SA module monolith decomposition into 8 focused modules (`types`, `planning`, `execution`, `intervention`, `agent`, `process`, `stats`, `actions`)
- Unified Timeline System (`TimeRange` + `TimelineEntry`) with exponential time-decay reranking
- 5W2H dimension-level audit with `AuditStatus` (Pass/Warning/Fail) and causal chain linkage
- Knowledge graph context injection into agent system prompts
- Time-aware system prompt (`SystemPromptRegion::TimeAwareness`)
- Hyperspace-integrated proactive perception with time decay (λ=0.5)
- Causal-integrated workspace monitor (`CausalObservation` recording)
- LLRU cold archive (Skill Graph) with `storage_tier = L0Permanent`
- TimelineStore mutation tracking (`GraphMutation`)

### Fixed

- PDCA P0: pre-check runtime error (`Error("expect L0-3")`) — graceful fallback to L0
- PDCA P1: PA creation failure when DA/TL sets `PauseOnError`
- PDCA P2: silent output loss on L0 fallback
- TL: `pend` always 0 — corrected `pend_sum` aggregation
- HNSW lock-free safety: `visited_gen` switched to `Vec<AtomicUsize>`
- Removed two `.expect("RwLock poisoned")` calls in `execution.rs`

### Removed

- Old petgraph-based `DagEngine` (replaced by PDCA 7-level adaptive execution)
