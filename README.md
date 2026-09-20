# Gliding Horse Agent OS 
<div align="center">

![Gliding Horse Logo](assets/logo.jpg)

**An Industrial-Grade AI Agent Operating System Built in Rust**  [![Star on GitHub](https://img.shields.io/github/stars/doiito/gliding_horse?style=flat)](https://github.com/doiito/gliding_horse)

*Inspired by Zhuge Liang's Wooden Ox and Gliding Horse — Ancient Ingenuity Meets Modern AI*

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![gRPC](https://img.shields.io/badge/gRPC-Protocol-green.svg)](https://grpc.io/)
[![Knowledge Graph](https://img.shields.io/badge/Knowledge%20Graph-Oxigraph-purple.svg)](https://oxigraph.org/)
[![Release](https://img.shields.io/github/v/release/doiito/gliding_horse?include_prereleases&label=release)](https://github.com/doiito/gliding_horse/releases)

---

[**English**](README.md) · [**中文**](README.zh.md) · [**Design Detail →**](docs/DESIGN_DETAIL.md) · [**Changelog →**](CHANGELOG.md)
[**medium URL**](https://medium.com/@doiito-sun)
[**中文稀土掘金**](https://juejin.cn/column/7647868075887165450)
[**中文博客园**](https://www.cnblogs.com/doiito)
[**中文CSDN博客**](https://blog.csdn.net/2604_96270735)
[**B站播客**](https://space.bilibili.com/1547455799/lists)

</div>

---

Release notes and version history are tracked in [CHANGELOG.md](CHANGELOG.md).

---

## What Is Gliding Horse?

An **AI agent operating system** built in Rust that orchestrates multiple agents via the PDCA cycle, enabling coordinated, auditable, and self-improving systems. — much like how Zhuge Liang's **Wooden Ox and Gliding Horse** revolutionized logistics by harnessing mechanical power across treacherous terrain.

> "We don't just build agents; we build the **infrastructure that harnesses their collective intelligence**."

### Core Architecture

| Layer | Technology | Role |
|-------|-----------|------|
| **Core Coordination** (Rust) | `PDCA cycle` · `5W2H ontology` · `EventBus` | Agent orchestration & lifecycle |
| **Skill Graph** | `RDF` · `6 link types` · `15 modules` | Dynamic cognitive network |
| **Memory System** | `L0 redb` · `L1 Session` · `L2 Oxigraph + Blackboard` · `L3 Projection` · `MESI coherence` | Hierarchical memory with prefetch |
| **Knowledge Graph** | `Oxigraph RDF` · `SPARQL 1.1` · `Code AST` · `Named Graphs` | Cross-subsystem unified store |
| **HyperspaceEngine** | `HNSW ANN` · `WAL` · `Poincaré/Cosine/Euclidean/Lorentz` · `Hybrid search` | Embedded vector embeddings |
| **Gliding Code TUI** | `ratatui` · `crossterm` · `MCP` · `checkpoint/resume` | Terminal AI coding assistant |
| **Data Bus** | `JSON-LD subset` · `@id/@type/@context` · `Named Graphs` | Internal interoperability |
| **Gateway** | `gRPC` · `HTTP (axum REST)` · `MCP` | Service interface |
| **Perception Engine** | `10 triggers` · `Anomaly dedup` · `5W2H constraint check` | Proactive monitoring |
| **Agent Workflow** | `SA/PA/DA/CA/AA` · `Tool system` · `Checkpoint` · `Tracked actions` | Multi-agent execution |

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
Self-contained vector engine with **runtime-switchable metrics** (Cosine, Poincaré, Lorentz, Euclidean). Features **HNSW approximate nearest neighbor search**, CRC32-verified **Write-Ahead Log (WAL)** with 3 sync modes, **tangent-space pruning** for Poincaré ball search, a JSON-LD metadata index with RoaringBitmap filters, and dual-space **hybrid search** (text × structural). The `crates/hyperspace-engine` crate has no external vector database dependency.

### 2. Skill Graph Cognitive Network
Dynamic in-memory cognitive network with **6 semantic link types** (Prerequisite, Composition, Related, Alternative, Extends, Generalization). Includes **Poincaré structural embedding** computation from graph topology (prerequisite depth, tag fingerprinting), **hypergraph composition** with first-class `Hyperedge` and `CompositionType` (Conjunction, Disjunction, Exactly(n), AtLeast(n), Pipeline), **graph algorithms** (PageRank, betweenness centrality, community detection, prerequisite-chain traversal, Tarjan SCC cycle detection), **causal failure analysis** with root cause inference, **formal invariant verification** (6 checks: acyclicity, link target existence, composite reachability, no deprecated prerequisites, valid 5W2H, valid security levels), and **temporal versioning** with snapshot/rollback.

### 3. Generalized PDCA — 7-Level Adaptive Execution
Dynamically selects one of 7 complexity levels from 5W2H metadata: `Instant` → `Simple` → `Standard` → `Complex` → `Exploratory` → `Emergency` → `Recursive`. One engine handles everything from instant queries to multi-week projects — no rigid workflows. **SA/PA/DA/CA/AA agent roles** with template-driven prompt construction.

### 4. CPU Cache-Inspired Memory — 4 Layers + MESI Coherence
**L0** redb disk storage → **L1** session context → **L2** Oxigraph-backed blackboard → **L3** projection cache. The repository implements cache-inspired coordination and prefetch components; no end-to-end latency or multi-agent consistency benchmark is currently published.

### 5. JSON-LD Data Bus — Internal Interoperability Subset
The internal JSON-LD utilities support `@context`, `@id`, `@graph`, framing, validation and routing used by this repository. They are not a claim of complete JSON-LD 1.1, SHACL, or general RDF interoperability.

### 6. Self-Evolving Skill Graph — Autonomous Learning
AA agents record knowledge fragments, links, and evolution proposals after task completion. `BootstrapEngine` exposes explicit learn/reduce operations and ingests Markdown skills from the filesystem; evolution proposals require approval, validation, and commit, so suggestions are not applied automatically.

### 7. Universal Knowledge Graph — Unified Cognitive Backbone
Skills, memories, tasks, and code knowledge can use the shared **Oxigraph RDF store** through named graphs, enabling scoped SPARQL joins where producers are wired to that store. Code ASTs parsed by tree-sitter are converted to RDF triples. `SkillGraphStore` projects its changes into the semantic store; reverse RDF-to-skill synchronization is not implemented.

### 8. Semantic Skill Discovery Engine
`SkillDiscoveryEngine` wraps `HyperspaceStore` for vector-based semantic search across skills. `suggest_links()` prefers cosine similarity via embedding vectors and falls back to Jaccard tag overlap when embeddings are unavailable. Includes BFS path finding (`find_skill_chain()`), composition tree construction (`get_skill_tree()`), and conflict detection.

### 9. 5W2H Dimension-Level Audit — Precision Rollback
CA audits all 7 dimensions (`what`, `why`, `who`, `when`, `where`, `how`, `how_much`) independently. What/Why fail → re-analyze. How/Where fail → re-plan. When/HowMuch fail → conditional pass. No more black-box "PASS/FAIL" — you know exactly what went wrong.

### 10. Proactive Perception Engine
10 execution triggers (`TaskStart`, `PlanCompleted`, `ProgressAnomaly`, `CheckCompleted`, `TaskEnd`, `CycleTimeout`, `AgentBlocked`, `ResourceConflict`, `QualityDegradation`, `UserFeedback`) with a 60-second anomaly deduplication window. Monitors deadline violations, budget overruns (>80% tokens), role mismatches, and environment conflicts. **Workspace Monitor** detects file creations/modifications/deletions in real-time. Auto-escalates to human when needed.

### 11. Micro-Tool System — Tame Large Outputs
Results at or above 16 KB (16,384 bytes) auto-generate conversational micro-tools (e.g., "search_in_results"). Transforms unwieldy large outputs into interactive, queryable artifacts within the LLM context.

### 12. MCP Integration — One Protocol to Connect Them All
Standard **Model Context Protocol** connects GitHub, Slack, Jira, and any MCP-compatible server. Dynamic tool discovery at runtime. Supports both HTTP SSE and stdio transport modes with repeatable `--mcp-server` / `--mcp-server-stdio` CLI flags.

### 13. Checkpoint & Recovery — Explicit Session Management
Session checkpoints and `--resume <task_iri>` / `--list-checkpoints` support explicit session management. Crash recovery and complete long-running-task replay require dedicated fault-injection and end-to-end validation before being claimed.

### 14. Center + Edge Federation — Local Autonomy, Global Orchestration
The [`apps/software_engineering_team`](apps/software_engineering_team/README.md) prototype splits the system across three tiers: a Go **Center** (Gin + Temporal + gRPC) owns workflow orchestration, project management, and agent registration; a Rust **Edge daemon** (axum + async-openai) runs local LLM execution, caches graph data, and communicates with the IDE; a TypeScript **VS Code plugin** provides chat, task, and graph views over WebSocket/REST. The Docker sandbox for heavy isolation is reserved; the repository's `unshare` process sandbox is the default lightweight path.

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

Prebuilt binaries for Linux (x86_64 / aarch64, fully static musl), macOS (Apple Silicon), and Windows (x86_64) are published on the **[Releases](https://github.com/doiito/gliding_horse/releases)** page. Download the archive for your platform, then:

```bash
# Linux / macOS
tar xzf glidingcode-*.tar.gz
./glidingcode --help

# Windows (PowerShell)
Expand-Archive glidingcode-*.zip .
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
and passes the configurable positive-improvement promotion gate. Controlled
pairs must also match seed, model, application/workflow/skill-catalog
configuration, workspace snapshot, objective, and orchestration mode;
repeating one pair ID does not increase the independent sample count. The
summary command reports observed sample
counts, success rates, P50/P95 reward, latency, prompt tokens, turns, tool calls,
and whether replay arms are actually comparable; it does not synthesize missing
counterfactual results.

### Build from Source

```bash
git clone https://github.com/doiito/gliding_horse.git
cd gliding_horse

# Build the glidingcode binary in release mode
cargo build -p code_cli --release
./target/release/glidingcode --help
```

---

## 🗺️ Roadmap

**v0.1.x series — released** (current: `v0.1.7.preview`)
- Prebuilt binaries for Linux (x86_64 / aarch64, fully static musl), macOS (Apple Silicon), and Windows (x86_64), published on the Releases page
- MCP integration over HTTP SSE and stdio with repeatable `--mcp-server` / `--mcp-server-stdio` flags
- Checkpoint/resume, explicit JSON-LD DAG workflow execution, and durable continuous-learning audit surfaces
- Reproducible L0 / L2 / L3 / HNSW / Poincaré benchmarks via `examples/readme_performance.rs`

**v0.2.x series — in progress / planned**
- Harden the Center + Edge federation prototype (`apps/software_engineering_team`), including the Docker sandbox for the Edge daemon
- Native web dashboard for agent monitoring and task management
- Python/TypeScript SDK for easier integration
- Skill marketplace prototype with a community plugin registry
- Multi-model routing with cost-aware scheduling

**v0.3.x+ series — future**
- Kubernetes deployment operator for production scaling
- Distributed agent mesh across Edge nodes
- Multi-modal agent support (vision, audio)
- Multi-turn conversation memory compression

---

## 📊 Performance Targets

| Operation | Target latency | Target throughput |
|-----------|---------------|-------------------|
| L2 durable node write (Oxigraph-backed blackboard) | ~2ms | 500 ops/sec |
| L3 cold projection | ~15ms | 66 ops/sec |
| L0 redb KV read | ~1ms | 1000 ops/sec |
| HNSW search (10K vectors) | ~1ms | 1000 qps |
| Poincaré 4D vector construction | ~50µs | — |
| Agent ReAct turn | 1–5s | environment/model dependent |
| Idle memory | ~200MB | scales with tasks |

These are **targets**, not published benchmarks. The first five are reproducible
in release mode with `cargo run --release --example readme_performance`, which
prints the actual value next to each target and a pass/fail status. Agent turn
latency and idle memory are environment/model-level measurements and must be
verified from a real provider run and the glidingcode process respectively.

---

## 📚 Documentation

- **Design Detail** → [`docs/DESIGN_DETAIL.md`](docs/DESIGN_DETAIL.md) · [`docs/DESIGN_DETAIL.zh.md`](docs/DESIGN_DETAIL.zh.md) (中文)
- **Core Design Philosophy** → [`docs/CORE_DESIGN_PHILOSOPHY.md`](docs/CORE_DESIGN_PHILOSOPHY.md) · [`docs/CORE_DESIGN_PHILOSOPHY.zh.md`](docs/CORE_DESIGN_PHILOSOPHY.zh.md) (中文)
- **Ontology Namespace Migration** → [`docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md`](docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md)
- **Changelog** → [`CHANGELOG.md`](CHANGELOG.md)
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
