# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
