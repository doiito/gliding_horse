# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **SA verify-first gate**: the verify-first PDCA cycle now runs the full PDCA (stored `fallback_steps`) unless the AA verdict explicitly confirms the task is already done. Previously the AA's `finish` action hardcoded `status: "success"`, which falsely short-circuited execution for tasks that genuinely needed it. Recovery is conservative by design: any missing/ambiguous/negative verdict means full execution.
  - `src/core/sa/types.rs` — new `verify_aa_needs_execution()` verdict parser (8 completion markers)
  - `src/core/sa/process.rs` — cycle-0 gate: skip completion only when `status == "success" && !needs_execution_after_verify`
  - `src/core/sa/tests.rs` — `test_verify_aa_needs_execution_parses_verdict`

- **Tool executor security IRI coverage**: the built-in security skill IRI table now whitelists the AA/CA read-only inspection tools (`file_list`, `workspace_status`, `rag_search`, `kg_search`, `codebase_search`, `knowledge_list`, `knowledge_search`, `knowledge_extract_code`). Previously these were rejected as "no registered executable skill", blocking verify-first CA/AA from inspecting the workspace and verifying deliverables.
  - `src/tools/tool_executor/mod.rs` — extended `builtin_security_skill_iri()`
  - `src/tools/tool_executor/tests.rs` — `security_gate_allows_aa_ca_inspection_tools_with_whitelisted_file_read`

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
