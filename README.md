# Gliding Horse Agent OS 
<div align="center">

![Gliding Horse Logo](assets/logo.jpg)

**An Industrial-Grade AI Agent Operating System Built in Rust**  [![Star on GitHub](https://img.shields.io/github/stars/doiito/gliding_horse?style=flat)](https://github.com/doiito/gliding_horse)

*Inspired by Zhuge Liang's Wooden Ox and Gliding Horse — Ancient Ingenuity Meets Modern AI*

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![gRPC](https://img.shields.io/badge/gRPC-Protocol-green.svg)](https://grpc.io/)
[![Knowledge Graph](https://img.shields.io/badge/Knowledge%20Graph-Oxigraph-purple.svg)](https://oxigraph.org/)
[![Release](https://img.shields.io/badge/release-v0.1.4-blue)](https://github.com/doiito/gliding_horse/releases)

---

[**English**](README.md) · [**中文**](README.zh.md) · [**Design Detail →**](docs/DESIGN_DETAIL.md)
[**medium URL**](https://medium.com/@doiito-sun)
[**中文稀土掘金**](https://juejin.cn/column/7647868075887165450)
[**中文博客园**](https://www.cnblogs.com/doiito)
[**中文CSDN博客**](https://blog.csdn.net/2604_96270735)
[**B站播客**](https://space.bilibili.com/1547455799/lists)

</div>

---

## 🎉 v0.1.4.preview Release

We are proud to announce the **v0.1.4.preview release** of Gliding Horse Agent OS — a major stability and intelligence release with significant architectural refactoring and 85 file changes (+6134 / −4127 lines).

**Highlights:**

| Area | Description |
|------|-------------|
| **SA Module Monolith Decomposition** | Split the 3408-line `sa/mod.rs` into 8 focused modules (`types`, `planning`, `execution`, `intervention`, `agent`, `process`, `stats`, `actions`) — each with a single responsibility. The largest structural refactor in the codebase. |
| **Unified Timeline System** | New `TimeRange` + `TimelineEntry` framework for cross-subsystem temporal queries. Includes exponential time-decay reranking (`apply_time_decay()`) for memory recall — older entries gracefully decay in relevance. |
| **5W2H Dimension Audit** | Formalized dimension-level audit with `AuditStatus` (Pass/Warning/Fail) and automatic causal chain linkage. Failed dimensions now feed into `CausalEngine` for root-cause analysis — no more black-box PASS/FAIL. |
| **Knowledge Graph Context Injection** | Agent system prompts now auto-inject relevant KG entities before each execution turn, ensuring SA interventions and new knowledge are immediately visible to all agents. |
| **Time-Aware System Prompt** | Agents now receive current time and session context via `SystemPromptRegion::TimeAwareness`, enabling time-sensitive reasoning and checkpoint-consistent recovery. |
| **Hyperspace-Integrated Proactive Perception** | `ProactiveEngine` now uses HyperspaceStore semantic search (with time decay λ=0.5) for experience retrieval — replacing L0 tag-substring matching. Graceful fallback to legacy path. |
| **Causal-Integrated Workspace Monitor** | File events (create/modify/delete) now record `CausalObservation` for root-cause traceability. Pre-task snapshots + objective-aware file inventory injection. |
| **LLRU Cold Archive (Skill Graph)** | `SkillGraphStore` auto-archives cold skills (`last_used_at < cutoff`) to L0 with `storage_tier = L0Permanent`. `find_stale_skills()` triggers re-indexing of outdated entries. |
| **TimelineStore Mutation Tracking** | Every `SkillGraphStore` structural change (register/update/remove/link/MOC) now records a `GraphMutation` — making `pending_mutations()` and `snapshot_count()` accurately reflect real graph activity. |
| **Old DAG Workflow Engine Removed** | Removed the 211-line petgraph-based `DagEngine` — PDCA's 7-level adaptive execution now fully replaces the legacy DAG orchestration. |
| **HNSW Lock-Free Safety** | `visited_gen` changed from `Vec<usize>` to `Vec<AtomicUsize>` in `IncrementalHNSW`, eliminating concurrent search data races while preserving lock-free throughput. |
| **PDCA P0: Pre-check Runtime Error** | Fixed crash (`Error("expect L0-3")`) when TL fails to match a skill — gracefully falls back to L0 execution instead of aborting the workflow. |
| **PDCA P1: PA Uncreatable** | Fixed PA creation failure when DA/TL sets `PauseOnError` — missing `execute` field now properly populated. |
| **PDCA P2: No Output on L0 Fallback** | Fixed silent output loss when metrics unavailable — default metrics ensure user always receives a response. |
| **TL: pend always 0** | Fixed `pend_sum` calculation in TL aggregation — was incorrectly using `sum` instead of actual pend values from sub-tasks. |
| **Error Handling Cleanup** | Removed two `.expect("RwLock poisoned")` calls in `execution.rs` — graceful degradation over panic. |
| **Skill Graph Security Enhancements** | New access-control checkpoints in skill registration and query paths, plus MCP tool call security filtering. |

---

## 🎉 v0.1.3 Release

We are proud to announce the **v0.1.3 release** of Gliding Horse Agent OS.

**What's new in v0.1.3:**

| Feature | Description |
|---------|-------------|
| **Causal Engine** | New standalone causal analysis subsystem with `CausalEngine`, `FusionEngine`, `CausalStore`, and typed `CausalFactor`. Enables causal reasoning across agent operations with fused multi-factor analysis — identifies root causes, propagates failure chains, and computes causal graphs for agent decisions. |
| **Unified Graph Backend** | Consolidated `GraphBackend` (~1,200 LOC) replacing fragmented graph storage — provides a single, optimized interface for node/edge CRUD with batch operations, subgraph extraction, and path-finding across all knowledge layers. |
| **Graph Features Computation** | New `graph_features` module computing structural feature vectors (degree centrality, clustering coefficient, PageRank, betweenness) and graph similarity scoring via feature-distance comparison. Enables quantitative graph analysis and comparison across cognitive snapshots. |
| **Snapshot Timeline** | Skill-graph snapshots with durable post-snapshot mutation records, point-in-time restoration, and diff support. It is an experimental graph timeline, not a complete session-history or crash-recovery system. |
| **Self-Awareness (SA) Overhaul** | Major rewrite of the self-awareness module (+410 lines) with enhanced monitoring of agent state, environment perception, and adaptive behavior. Integrated with the causal engine for self-diagnosis. |
| **5W2H Dimension Audit Enhancement** | Expanded dimension-level audit in `src/core/five_w2h.rs` with deeper causal attribution per dimension. What/Why failures now chain into the causal engine for automated root-cause analysis. |
| **Advanced Features Design** | Comprehensive [`ADVANCED_FEATURES_DESIGN.md`](docs/ADVANCED_FEATURES_DESIGN.md) document covering graph backend architecture, causal reasoning design, timeline-based snapshot semantics, and performance benchmarks. |
| **Graph Backend Benchmarks** | New benchmark suite (`benches/bench_graph_backend.rs`) measuring node/edge R/W, subgraph extraction, and path-finding throughput. |
| **Gliding Code TUI Refinements** | Engine and TUI improvements in the terminal client — better Markdown rendering, enhanced MCP server lifecycle, and internal refactoring for maintainability. |
| **Bug Fixes** | Fixed duplicate secondary-index update in L2 `write_node` that caused inconsistent indexing under concurrent writes. |

---

## 🎉 v0.1.2 Release

We are proud to announce the **v0.1.2 release** of Gliding Horse Agent OS.

**What's new in v0.1.2:**

| Feature | Description |
|---------|-------------|
| **HyperspaceEngine** | Production-grade vector embedding engine with HNSW ANN search, Write-Ahead Log (WAL), tangent-space pruning, and runtime-switchable metrics (Poincaré, Cosine, Euclidean, Lorentz). |
| **Skill Graph Cognitive Network** | Hypergraph composition, Poincaré structural embeddings, PageRank/betweenness/community detection algorithms, causal failure analysis, experimental temporal snapshots/rollback, formal invariant verification (6 checks), hybrid text×structural search. |
| **Semantic Skill Discovery** | Vector-store integration in `SkillDiscoveryEngine` — finds semantically related skills via HyperspaceStore cosine search, replaces Jaccard-only `suggest_links()` |
| **Oxigraph SPARQL Bridge** | Skill-graph-to-Oxigraph RDF projection via SPARQL INSERT/DELETE with named-graph isolation; reverse RDF-to-skill synchronization is not implemented. |
| **L2 Blackboard Memory** | Typed document store with JSON-LD threading, projections, message packs, and LRU eviction for long-term agent context |
| **Workspace Monitor** | Real-time file system perception engine with 10 event triggers, anomaly deduplication, and 5W2H constraint checking |
| **Batch Agent Manager** | Sliding-window batch components with configurable triggers, event-bus integration, and business-domain isolation. The root gRPC service wires opt-in custom-event and cron/window consumption, including streaming requests through shared service state; its L0 event journal deduplicates, acknowledges on successful execution, and replays pending input at startup. `KnowledgePersister` is not wired to this execution path; graph-changing handlers are disabled unless `apply_graph_mutations` is explicitly enabled, and that opt-in path still lacks event-id idempotency and approval/transaction gates. |
| **Gliding Code TUI** | Interactive terminal UI (ratatui v0.28) with Markdown rendering, MCP server support, checkpoint/resume, and multi-model backends |

---

## What Is Gliding Horse?

An **AI agent operating system** built in Rust that orchestrates multiple agents via the PDCA cycle, enabling coordinated, auditable, and self-improving systems. — much like how Zhuge Liang's **Wooden Ox and Gliding Horse** revolutionized logistics by harnessing mechanical power across treacherous terrain.

> "We don't just build agents; we build the **infrastructure that harnesses their collective intelligence**."

### Core Architecture

| Layer | Technology | Role |
|-------|-----------|------|
| **Core Coordination** (Rust) | `PDCA cycle` · `5W2H ontology` · `EventBus` | Agent orchestration & lifecycle |
| **Skill Graph** | `RDF` · `6 link types` · `18 modules` | Dynamic cognitive network |
| **Memory System** | `L0 Sled` · `L1 Session` · `L2 Blackboard` · `L3 Projection` · `MESI coherence` | Hierarchical memory with prefetch |
| **Knowledge Graph** | `Oxigraph RDF` · `SPARQL 1.1` · `Code AST` · `Named Graphs` | Cross-subsystem unified store |
| **HyperspaceEngine** | `HNSW ANN` · `WAL` · `Poincaré/Cosine/Euclidean` · `Hybrid search` | Embedded vector embeddings |
| **Gliding Code TUI** | `ratatui` · `crossterm` · `MCP` · `checkpoint/resume` | Terminal AI coding assistant |
| **Data Bus** | `JSON-LD 1.1` · `@id/@type/@context` · `Named Graphs` | Universal interoperability |
| **Gateway** | `gRPC` · `HTTP (OpenAI-compatible)` · `MCP` | Production interface |
| **Perception Engine** | `10 triggers` · `Anomaly dedup` · `5W2H constraint check` | Proactive monitoring |
| **Agent Workflow** | `PA/DA/CA` · `Tool system` · `Checkpoint` · `Tracked actions` | Multi-agent execution |

---

## 📖 The Story: From Ancient Wisdom to Modern Intelligence

In the turbulent era of the Three Kingdoms (220–280 AD), the legendary strategist **Zhuge Liang** (诸葛亮), chancellor of the Shu Han state, faced a critical challenge: how to transport supplies efficiently through the treacherous mountain paths of Sichuan during his Northern Expeditions. Traditional wheeled carts struggled on narrow trails; human porters exhausted quickly.

His solution — the **Wooden Ox (木牛)** and **Gliding Horse (流马)** — were autonomous transport devices that could navigate difficult terrain with minimal human guidance. These mechanical wonders were not merely tools; they represented a paradigm shift — **autonomous systems that extended human capability**.

### Bridging Past and Present

Just as the Gliding Horse served as an **intelligent harness** for transporting supplies across impossible terrain, **Gliding Horse Agent OS** serves as an **intelligent harness for AI agents**:

| Ancient Innovation | Modern Implementation |
|-------------------|----------------------|
| **Autonomous Transport** | Self-directing agent workflows |
| **Terrain Adaptation** | Dynamic complexity handling (7 levels) |
| **Load Distribution** | Parallel agent execution |
| **Minimal Guidance** | Proactive anomaly detection |
| **Mechanical Reliability** | Rust's memory safety guarantees |

> *"The wise adapt their methods to circumstances, just as water shapes its course according to the ground over which it flows."*  
> — **Zhuge Liang**

This ancient wisdom guides our design: **flexible orchestration that adapts to task complexity**, rather than rigid frameworks that force tasks into predefined molds.

---

## 🔧 Key Highlights

### 1. HyperspaceEngine — Embedded Vector Engine
Production-grade spatial memory engine with **runtime-switchable metrics** (Poincaré, Cosine, Euclidean, Lorentz). Features **HNSW approximate nearest neighbor search**, CRC32-verified **Write-Ahead Log (WAL)** with 3 sync modes, **tangent-space pruning** for Poincaré ball search, JSON-LD metadata index with RoaringBitmap filters, and dual-space **hybrid search** (text × structural). A self-contained crate with zero external vector database dependencies.

### 2. Skill Graph Cognitive Network
Dynamic in-memory cognitive network with **6 semantic link types** (Prerequisite, Composition, Related, Alternative, Extends, Generalization). Includes **Poincaré structural embedding** computation from graph topology (prerequisite depth, tag fingerprinting), **hypergraph composition** with first-class `Hyperedge` and `CompositionType` (Sequential, Parallel, Conditional, Optional, Fallback), **graph algorithms** (PageRank, betweenness centrality, label-propagation community detection, DFS prerequisite chains, Tarjan SCC cycle detection), **causal failure analysis** with root cause inference, **formal invariant verification** (6 checks: acyclicity, link existence, composite reachability, no deprecated prereqs, valid 5W2H, valid security levels), and **temporal versioning** with snapshot/rollback.

### 3. Generalized PDCA — 7-Level Adaptive Execution
Dynamically selects from 7 complexity levels (L0 instant → L5 recursive → L6 emergency) via 5W2H metadata. One engine handles everything from instant queries to multi-week projects — no rigid workflows. **PA/DA/CA agent roles** with template-driven prompt construction.

### 4. CPU Cache-Inspired Memory — 4 Layers + MESI Coherence
**L0** Sled disk storage → **L1** session context → **L2** Oxigraph RDF + Blackboard → **L3** SPARQL projection cache. The repository implements cache-inspired coordination and prefetch components; no end-to-end latency or multi-agent consistency benchmark is currently published.

### 5. JSON-LD Universal Data Bus — Internal Interoperability Subset
The internal JSON-LD utilities support `@context`, `@id`, `@graph`, framing, validation and routing used by this repository. They are not a claim of complete JSON-LD 1.1, SHACL, or general RDF interoperability.

### 6. Self-Evolving Skill Graph — Autonomous Learning
AA agents record knowledge fragments, links, and evolution suggestions after task completion. `/learn` and `/reduce` provide explicit acquisition/consolidation operations; suggestions are not automatically applied because typed-patch verification, security, and conflict gates are still pending. `BootstrapEngine` ingests markdown skills from the filesystem.

### 7. Universal Knowledge Graph — Unified Cognitive Backbone
Skills, memories, tasks, and code knowledge can use the shared **Oxigraph RDF store** through named graphs, enabling scoped SPARQL joins where producers are wired to that store. Code ASTs parsed by tree-sitter are converted to RDF triples. `SkillGraphStore` projects its changes into the semantic store; reverse RDF-to-skill synchronization is not implemented.

### 8. Semantic Skill Discovery Engine
`SkillDiscoveryEngine` wraps `HyperspaceStore` for vector-based semantic search across skills. `suggest_links()` falls back from Jaccard tag overlap to cosine similarity via embedding vectors. Includes BFS path finding (`find_skill_chain()`), composition tree construction (`get_skill_tree()`), and conflict detection.

### 9. 5W2H Dimension-Level Audit — Precision Rollback
CA audits each of the 7 dimensions independently. What/Why fail → re-analyze. How/Where fail → re-plan. When/HowMuch fail → conditional pass. No more black-box "PASS/FAIL" — you know exactly what went wrong.

### 10. Proactive Perception Engine
10 execution triggers with 60-second anomaly deduplication. Monitors deadline violations, budget overruns (>80% tokens), role mismatches, environment conflicts. **Workspace Monitor** detects file creations/modifications/deletions in real-time. Auto-escalates to human when needed.

### 11. Micro-Tool System — Tame Large Outputs
Results >8KB auto-generate conversational micro-tools (e.g., "search_in_results"). Transforms unwieldy 50KB+ outputs into interactive, queryable artifacts within the LLM context.

### 12. MCP Integration — One Protocol to Connect Them All
Standard **Model Context Protocol** connects GitHub, Slack, Jira, and any MCP-compatible server. Dynamic tool discovery at runtime. Supports both HTTP SSE and stdio transport modes with repeatable `--mcp-server` CLI flags.

### 13. Checkpoint & Recovery — Explicit Session Management
Session checkpoints and `--resume <task_iri>` / `--list-checkpoints` support explicit session management. Crash recovery and complete long-running-task replay require dedicated fault-injection and end-to-end validation before being claimed.

### 14. Center + Edge Federation — Local Autonomy, Global Orchestration
Go Center handles workflow orchestration (Temporal), project management, agent registry. Rust Edge runs local LLM execution with Docker sandbox. VS Code Plugin provides real-time developer awareness. No single point of failure.

---

## 🖥️ Gliding Code — The Terminal AI Assistant

**Gliding Code** is a terminal-based AI coding assistant (`ratatui` TUI) that brings the power of Gliding Horse's knowledge graph and agent orchestration directly into your command line — no IDE required.

**Features:**
- Interactive TUI with **Markdown rendering** (`tui-markdown`) and **mermaid diagram** support
- **MCP server integration** via `--mcp-server` and `--mcp-server-stdio` flags
- **Checkpoint/resume** with `--resume <task_iri>` and `--list-checkpoints`
- **Multi-model backends**: DeepSeek, OpenAI-compatible APIs
- **PDCA and JSON-LD DAG workflow execution** through the same SA → BizAgent runtime
- **Auditable continuous learning** with CA-validated, task-family-scoped knowledge and guarded policy promotion
- **Configurable** workspace, max iterations, max PDCA cycles, verbosity

![Gliding Code Demo](assets/screenshot.gif)

![Knowledge Graph in Action](assets/gliding_code_kg.JPG)
*Knowledge graph visualization — real-time entity relationships, code structure understanding, and cross-subsystem awareness powered by Oxigraph RDF*

![Completed Programming Task](assets/gliding_code.JPG)
*Task completion interface — AI agent successfully analyzing and solving a programming task with full traceability*

---

## 🚀 Quick Start

### Download & Run — Gliding Code

No dependencies required. Just download, extract, and run:

| Platform | Download |
|----------|----------|
| Linux (x86_64, musl) | [`glidingcode-x86_64-unknown-linux-musl.tar.gz`](https://github.com/doiito/gliding_horse/releases) (~15 MB) |
| Linux (aarch64, musl) | [`glidingcode-aarch64-unknown-linux-musl.tar.gz`](https://github.com/doiito/gliding_horse/releases) (~14 MB) |
| macOS (Apple Silicon) | [`glidingcode-aarch64-apple-darwin.tar.gz`](https://github.com/doiito/gliding_horse/releases) (~13 MB) |
| Windows (x86_64) | [`glidingcode-x86_64-pc-windows-msvc.zip`](https://github.com/doiito/gliding_horse/releases) (~12 MB) |

```bash
# Linux / macOS
tar xzf glidingcode-*.tar.gz
./glidingcode --help

# Windows (PowerShell)
Expand-Archive glidingcode-x86_64-pc-windows-msvc.zip .
.\glidingcode.exe --help
```

> All Linux builds are **fully statically linked** (musl) — no runtime dependencies required.

Set your API key and start using it:

```bash
export DEEPSEEK_API_KEY="sk-..."        # Linux / macOS
# or
set DEEPSEEK_API_KEY="sk-..."            # Windows (cmd)
# or
$env:DEEPSEEK_API_KEY="sk-..."           # Windows (PowerShell)

# Alternatively, use any OpenAI-compatible provider:
export AGENT_OS_GATEWAY_API_KEY="sk-..."
export AGENT_OS_GATEWAY_API_URL="https://your-endpoint/v1"

# Web search tool (powered by Exa):
# Get your free API key at https://exa.ai/docs/reference/team-management/get-api-key
# Falls back to DuckDuckGo (unreliable in China, not recommended for Chinese users)
export EXA_API_KEY="your-exa-api-key"

# Run an interactive session
./glidingcode

# Or run a one-shot task
./glidingcode "Explain how Rust's borrow checker works"

# With MCP server attached
./glidingcode --mcp-server chrome=http://localhost:3000/sse

# Optional: use Parallel Search MCP (no account or API key required)
# Chosen queries and requested URLs are sent to Parallel.
./glidingcode --mcp-server parallel-search=https://search.parallel.ai/mcp

# Privacy note: third-party MCP servers (e.g. chrome, parallel-search)
# receive the queries, URLs, and prompts you send through them. Review
# each server's privacy policy before enabling.

# Resume from checkpoint
./glidingcode --resume task:abc123

# Execute an explicit JSON-LD DAG instead of the default PDCA-generated plan
./glidingcode --workflow ./workflow.jsonld "Run the workflow"

# Inspect durable task-level learning evidence without starting the full TUI engine
./glidingcode --list-learning-evaluations
./glidingcode --summarize-learning-evaluations

# Controlled baseline/shadow/active replay labels. Reuse the pair ID, model,
# seed, objective, workspace snapshot, and orchestration mode across all arms.
./glidingcode --learning-mode baseline --learning-pair-id replay-001 --learning-seed 42 "Task"
./glidingcode --learning-mode shadow   --learning-pair-id replay-001 --learning-seed 42 "Task"
./glidingcode --learning-mode active   --learning-pair-id replay-001 --learning-seed 42 "Task"
```

Active learning never bypasses the current task's CA audit. A learned policy
remains a bounded candidate (or shadow observation) until the same normalized
task family has at least five independent baseline and five candidate samples
and passes the promotion gate. The summary command reports observed sample
counts, success rates, P50/P95 reward, latency, prompt tokens, turns, tool calls,
and whether replay arms are actually comparable; it does not synthesize missing
counterfactual results.

### Build from Source

```bash
git clone https://github.com/doiito/gliding_horse.git
cd gliding_horse

# Build the glidingcode binary (release, ~51 MB)
cargo build -p code_cli --release
./target/release/glidingcode --help
```

---

## 🗺️ Roadmap

**v0.1.x Release Series** (stabilization):
- Binary distribution for Linux/macOS/Windows via GitHub Releases
- Pre-built musl static builds for Linux (zero-dependency)
- MCP tool ecosystem expansion and documentation
- Checkpoint/resume polish and testing

**v0.2.x Release Series** (planned):
- Native web dashboard for agent monitoring and task management
- Python/TypeScript SDK for easier integration
- Skill marketplace prototype with community plugin registry
- Multi-model routing with cost-aware scheduling

**v0.3.x+ Release Series** (future):
- Kubernetes deployment operator for production scaling
- Distributed agent mesh across Edge nodes
- Multi-modal agent support (vision, audio)
- Multi-turn conversation memory compression

---

## 📊 Performance Targets

| Operation | Latency | Throughput |
|-----------|---------|-----------|
| L2 Node Write (Oxigraph) | ~2ms | 500 ops/sec |
| L3 SPARQL Projection | ~15ms | 66 ops/sec |
| L0 redb KV Read | ~1ms | 1000 ops/sec |
| Hyperspace HNSW Search (10K vectors) | ~1ms | 1000 qps |
| Poincaré Embedding (4D) | ~50µs | — |
| Agent ReAct Turn | 1-5s | 0.2-1 turns/sec |
| Idle Memory | ~200MB | scales with tasks |

The deterministic local targets (L0/L2/L3/HNSW/Poincaré) can be measured in
release mode with `cargo run --release --example readme_performance`. Agent
turn latency and idle memory are environment/model-level measurements and must
be verified from a real provider run and the glidingcode process respectively.

---

## 📚 Documentation

- **Design Detail** → [`docs/DESIGN_DETAIL.md`](docs/DESIGN_DETAIL.md) · [`docs/DESIGN_DETAIL.zh.md`](docs/DESIGN_DETAIL.zh.md) (中文)
- **Core Design Philosophy** → [`docs/CORE_DESIGN_PHILOSOPHY.md`](docs/CORE_DESIGN_PHILOSOPHY.md) · [`docs/CORE_DESIGN_PHILOSOPHY.zh.md`](docs/CORE_DESIGN_PHILOSOPHY.zh.md) (中文)
- **Ontology Namespace Migration** → [`docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md`](docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md)
- **gRPC Proto** → [`proto/pdca_core.proto`](proto/pdca_core.proto)

---

## 🤝 Contributing

We welcome contributions from the community!

- **🐛 Report bugs**: [GitHub Issues](https://github.com/doiito/gliding_horse/issues)
- **💡 Propose ideas**: [GitHub Discussions](https://github.com/doiito/gliding_horse/discussions)
- **🔀 Submit PRs**: Fork → feature branch → PR against `main`

```bash
git checkout -b feat/my-feature
# Make your changes
cargo fmt && cargo clippy  # Keep code clean
cargo test                 # Ensure nothing breaks
git commit -am 'Add my feature'
git push origin feat/my-feature
```

All contributors are expected to adhere to our [Code of Conduct](docs/CODE_OF_CONDUCT.md).

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">

Star ⭐ if you find this useful — join us in building the infrastructure for tomorrow's AI.

[![GitHub stars](https://img.shields.io/github/stars/doiito/gliding_horse.svg?style=social&label=Star)](https://github.com/doiito/gliding_horse)

*"Wisdom is not inherited; it is built upon the shoulders of those who came before."*

</div>


<a href="https://www.star-history.com/?repos=doiito%2Fgliding_horse&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=doiito/gliding_horse&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=doiito/gliding_horse&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=doiito/gliding_horse&type=date&legend=top-left" />
 </picture>
</a>
