# 流马智能体操作系统
<div align="center">

![Gliding Horse Logo](assets/logo.jpg)

**工业级 AI 智能体操作系统 · Rust 构建**  [![Star on GitHub](https://img.shields.io/github/stars/doiito/gliding_horse?style=flat)](https://github.com/doiito/gliding_horse)

*受诸葛亮木牛流马启发 — 古老智慧与现代 AI 的融合*

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![gRPC](https://img.shields.io/badge/gRPC-Protocol-green.svg)](https://grpc.io/)
[![Knowledge Graph](https://img.shields.io/badge/Knowledge%20Graph-Oxigraph-purple.svg)](https://oxigraph.org/)
[![Release](https://img.shields.io/github/v/release/doiito/gliding_horse?include_prereleases&label=release)](https://github.com/doiito/gliding_horse/releases)

---

[**中文**] · [**English**](README.md) · [**设计细节 →**](docs/DESIGN_DETAIL.zh.md) · [**变更日志 →**](CHANGELOG.md)
[**medium URL**](https://medium.com/@doiito-sun)
[**中文稀土掘金**](https://juejin.cn/column/7647868075887165450)
[**中文博客园**](https://www.cnblogs.com/doiito)
[**中文CSDN博客**](https://blog.csdn.net/2604_96270735)
[**B站播客**](https://space.bilibili.com/1547455799/lists)

</div>

---

版本变化与发布记录统一记录在 [CHANGELOG.md](CHANGELOG.md)。

---

## 什么是 Gliding Horse？

一个 **基于 Rust 构建的 AI 智能体操作系统**，通过 PDCA 循环编排多智能体，实现协调、可审计和自我改进的系统。——正如诸葛亮当年用木牛流马在险峻山路上革新了后勤运输。

> "我们不只构建智能体；我们构建**驾驭集体智能的基础设施**。"

### 核心技术栈

| 层级 | 技术 | 职责 |
|------|------|------|
| **核心编排** (Rust) | `PDCA 循环` · `5W2H 本体` · `事件总线` | 智能体编排与生命周期管理 |
| **技能图谱** | `RDF` · `6 种链接类型` · `15 模块` | 动态认知网络 |
| **记忆系统** | `L0 redb` · `L1 Session` · `L2 Oxigraph + Blackboard` · `L3 Projection` · `MESI 一致性` | 带预取的分层记忆 |
| **知识图谱** | `Oxigraph RDF` · `SPARQL 1.1` · `代码 AST` · `命名图` | 跨子系统统一存储 |
| **HyperspaceEngine** | `HNSW ANN` · `WAL` · `Poincaré/Cosine/Euclidean/Lorentz` · `混合搜索` | 嵌入式向量嵌入引擎 |
| **Gliding Code TUI** | `ratatui` · `crossterm` · `MCP` · `断点恢复` | 终端 AI 编程助手 |
| **数据总线** | `JSON-LD 子集` · `@id/@type/@context` · `命名图` | 内部互操作层 |
| **网关** | `gRPC` · `HTTP (axum REST)` · `MCP` | 服务接口 |
| **感知引擎** | `10 种触发器` · `异常去重` · `5W2H 约束检查` | 主动监控 |
| **智能体工作流** | `SA/PA/DA/CA/AA` · `工具系统` · `检查点` · `追踪操作` | 多智能体执行 |

---

## 📖 故事：从古老智慧到现代智能

三国时期（220–280年），传奇战略家**诸葛亮**（蜀汉丞相）面临一项严峻挑战：如何在北伐中通过四川险峻的山路高效运输补给。传统轮车在狭窄陡峭的小路上举步维艰；人力搬运工负重有限，很快便精疲力竭。

他的解决方案——**木牛流马**——是能够以最少人力引导在复杂地形中行驶的自动运输装置。这些机械奇迹不仅仅是工具；它们代表了一种范式转变——**延伸人类能力的自主系统**。

### 连接古今：Agent Harness

正如流马作为穿越天险运输补给的**智能鞍具**，**Gliding Horse Agent OS** 充当了 AI 智能体的**智能驾驭层**：

| 古代创新 | 现代实现 |
|---------|---------|
| **自主运输** | 自驱动智能体工作流 |
| **地形适应** | 动态复杂度处理（7 级） |
| **负载分配** | 并行智能体执行 |
| **最小引导** | 主动异常检测 |
| **机械可靠性** | Rust 内存安全保障 |

> *"善战者因其势而利导之，譬如以水投水。"*  
> — **诸葛亮**

这一古老智慧指导着我们的设计：**适应任务复杂度的灵活编排**，而非将任务强行塞入预定模具的僵化框架。

---

## 🔧 亮点速览

### 1. HyperspaceEngine — 嵌入式向量引擎
自包含向量引擎，支持 **运行时可选度量空间**（Cosine、Poincaré、Lorentz、Euclidean）。内置 **HNSW 近似最近邻搜索**、CRC32 校验的**预写日志（WAL）**（3 种同步模式）、**切线空间剪枝**（优化 Poincaré 球搜索）、JSON-LD 元数据索引（RoaringBitmap 位图过滤器）以及双空间**混合搜索**（文本 × 结构）。`crates/hyperspace-engine` 独立 crate，不依赖任何外部向量数据库。

### 2. 技能图谱认知网络
动态内存认知网络，**6 种语义链接类型**（前置依赖、组合、关联、替代、扩展、泛化）。核心能力包括：基于图谱拓扑的 **Poincaré 结构嵌入**（前置依赖深度 + 标签域指纹）；**超图组合**——一等公民 `Hyperedge` 与 `CompositionType`（Conjunction、Disjunction、Exactly(n)、AtLeast(n)、Pipeline）；**图算法**（PageRank、介数中心性、社区发现、前置链遍历、Tarjan SCC 环检测）；**因果故障分析**与根因推断；**形式化不变式验证**（6 项检查：无环、链接目标存在、组合可达、无废弃前置依赖、5W2H 有效、安全等级有效）；**时序版本管理**与快照回滚。

### 3. 泛化 PDCA — 7 级自适应执行
通过 5W2H 元数据在 7 个复杂度等级中动态选择：`Instant` → `Simple` → `Standard` → `Complex` → `Exploratory` → `Emergency` → `Recursive`。同一引擎同时处理即时查询与数周工程项目——无需僵硬的固定流程。**SA/PA/DA/CA/AA 智能体角色**，基于模板的提示词构建。

### 4. CPU 缓存记忆 — 4 层结构 + MESI 一致性
**L0** redb 磁盘存储 → **L1** 会话上下文 → **L2** Oxigraph 支撑的 Blackboard → **L3** 投影缓存。仓库实现了借鉴缓存一致性的协调与预取组件；目前没有已发布的端到端延迟或多智能体一致性基准。

### 5. JSON-LD 数据总线 — 内部互操作子集
内部 JSON-LD 工具支持本仓库使用的 `@context`、`@id`、`@graph`、framing、校验和路由；这不是完整 JSON-LD 1.1、SHACL 或通用 RDF 互操作性声明。

### 6. 自进化技能图谱 — 自主学习
AA 智能体在任务完成后记录知识片段、链接和演化提案。`BootstrapEngine` 提供显式的 learn/reduce 操作，并从文件系统摄取 Markdown 格式技能；演化提案需要审批、验证与提交后才会生效，因此建议不会自动应用。

### 7. 通用知识图谱 — 统一认知骨干
技能、记忆、任务和代码知识可通过命名图使用共享 **Oxigraph RDF 存储**；已接线的生产者可进行受范围约束的 SPARQL 联合查询。tree-sitter 解析的代码 AST 会转为 RDF 三元组。`SkillGraphStore` 将变更投影到语义存储；尚未实现 RDF 到技能图的反向同步。

### 8. 语义技能发现引擎
`SkillDiscoveryEngine` 包装 `HyperspaceStore` 实现基于向量的语义技能搜索。`suggest_links()` 优先使用嵌入向量的余弦相似度，在嵌入不可用时回退到 Jaccard 标签重叠。内置 BFS 路径发现（`find_skill_chain()`）、组合树构建（`get_skill_tree()`）和冲突检测。

### 9. 5W2H 维度级审计 — 精准回滚
CA 独立审计全部 7 个维度（`what`、`why`、`who`、`when`、`where`、`how`、`how_much`）。What/Why 失败 → 重新分析。How/Where 失败 → 重新规划。When/HowMuch 失败 → 条件通过。告别黑盒"通过/不通过"——精确定位问题根因。

### 10. 主动感知引擎 — 防患于未然
10 种执行触发器（`TaskStart`、`PlanCompleted`、`ProgressAnomaly`、`CheckCompleted`、`TaskEnd`、`CycleTimeout`、`AgentBlocked`、`ResourceConflict`、`QualityDegradation`、`UserFeedback`），异常去重窗口 60 秒。监控截止时间违规、预算超支（>80% Token）、角色不匹配、环境冲突。**工作区监控器**实时检测文件创建/修改/删除。必要时自动升级到人工处理。

### 11. 微工具系统 — 驾驭大型输出
结果达到或超过 16 KB（16,384 字节）时自动生成可对话的微工具（如"search_in_results"）。将笨重的大型输出转变为 LLM 上下文中可交互、可查询的产物。

### 12. MCP 集成 — 一个协议连接一切
标准 **Model Context Protocol** 连接 GitHub、Slack、Jira 等任意 MCP 兼容服务器。运行时动态发现工具。支持 HTTP SSE 和 stdio 两种传输模式，通过可重复 `--mcp-server` / `--mcp-server-stdio` CLI 标志配置。

### 13. 检查点与恢复 — 显式会话管理
关键执行点会保存会话检查点，`--resume <task_iri>` 和 `--list-checkpoints` 提供显式会话管理。崩溃恢复和完整长任务回放仍需故障注入与端到端验证后才能作为能力宣称。

### 14. Center + Edge 联邦 — 本地自治，全局编排
[`apps/software_engineering_team`](apps/software_engineering_team/README.md) 原型将系统分为三层：Go **Center**（Gin + Temporal + gRPC）负责工作流编排、项目管理与智能体注册；Rust **Edge Daemon**（axum + async-openai）负责本地 LLM 执行、图数据缓存，并与 IDE 通信；TypeScript **VS Code 插件**通过 WebSocket/REST 提供对话、任务与图视图。重型隔离用的 Docker 沙箱为预留能力；主仓的 `unshare` 进程级沙箱是默认的轻量路径。

---

## 🖥️ Gliding Code — 终端 AI 编程助手

**Gliding Code** 是一款基于终端的 AI 编程助手（`ratatui` TUI），将流马智能体操作系统的知识图谱与智能体编排能力直接带入命令行——无需 IDE。

**功能特性：**
- 交互式 TUI，支持 **Markdown 渲染**（`tui-markdown`）和 **Mermaid 图表**
- **MCP 服务器集成**，通过 `--mcp-server` 和 `--mcp-server-stdio` 标志
- **检查点恢复**：`--resume <task_iri>` 和 `--list-checkpoints`
- **多模型后端**：DeepSeek、兼容 OpenAI 的 API
- **PDCA 与 JSON-LD DAG 工作流**均可通过同一 SA → BizAgent 运行时执行
- **可审计的持续学习**：CA 校验、任务族范围的知识与受门禁控制的策略提升
- **可配置**：工作区、最大迭代次数、最大 PDCA 周期、日志级别

![Gliding Code 演示](assets/screenshot.gif)

![知识图谱实战](assets/gliding_code_kg.JPG)
*知识图谱可视化——实时实体关系、代码结构理解、基于 Oxigraph RDF 的跨子系统感知*

![编程任务完成](assets/gliding_code.JPG)
*任务完成界面——AI 智能体成功分析并解决编程任务，全程可追溯*

---

## 🚀 快速开始

### 直接下载 — Gliding Code

适用于 Linux（x86_64 / aarch64，musl 全静态）、macOS（Apple Silicon）和 Windows（x86_64）的预编译二进制发布在 **[Releases](https://github.com/doiito/gliding_horse/releases)** 页面。下载对应平台的压缩包后：

```bash
# Linux / macOS
tar xzf glidingcode-*.tar.gz
./glidingcode --help

# Windows (PowerShell)
Expand-Archive glidingcode-*.zip .
.\glidingcode.exe --help
```

> 所有 Linux 版本均为**全静态链接**（musl），无需任何运行时依赖。

设置 API 密钥后即可使用：

```bash
export DEEPSEEK_API_KEY="sk-..."        # Linux / macOS
# 或
set DEEPSEEK_API_KEY="sk-..."           # Windows (cmd)
# 或
$env:DEEPSEEK_API_KEY="sk-..."          # Windows (PowerShell)

# 也可使用任意兼容 OpenAI 的服务：
export AGENT_OS_GATEWAY_API_KEY="sk-..."
export AGENT_OS_GATEWAY_API_URL="https://your-endpoint/v1"

# Web search 工具（基于 Exa 搜索引擎）：
# 从 https://exa.ai/docs/reference/team-management/get-api-key 免费获取 API Key
# 未设置时自动降级为 DuckDuckGo 模式，但国内 DuckDuckGo 不好用，不推荐国内使用
export EXA_API_KEY="your-exa-api-key"

# 启动交互式会话
./glidingcode

# 或单次执行任务
./glidingcode "解释 Rust 的借用检查器工作原理"

# 附接 MCP 服务器
./glidingcode --mcp-server chrome=http://localhost:3000/sse

# 可选：使用 Parallel Search MCP（无需账号或 API Key）
# 所选择的查询与请求的 URL 会发送给 Parallel。
./glidingcode --mcp-server parallel-search=https://search.parallel.ai/mcp

# 隐私提示：第三方 MCP 服务器（如 chrome、parallel-search）会收到你通过它们
# 发送的查询、URL 与提示词，启用前请阅读对应服务器的隐私政策。

# 从检查点恢复
./glidingcode --resume task:abc123

# 使用显式 JSON-LD DAG 取代默认的 PDCA 生成计划
./glidingcode --workflow ./workflow.jsonld "Run the workflow"

# 无需启动完整 TUI 引擎即可查看持久化的任务级学习证据
./glidingcode --list-learning-evaluations
./glidingcode --summarize-learning-evaluations

# 受控的 baseline/shadow/active 回放标签。各实验臂需复用同一 pair ID、模型、
# 随机种子、目标、工作区快照与编排模式。
./glidingcode --learning-mode baseline --learning-pair-id replay-001 --learning-seed 42 "Task"
./glidingcode --learning-mode shadow   --learning-pair-id replay-001 --learning-seed 42 "Task"
./glidingcode --learning-mode active   --learning-pair-id replay-001 --learning-seed 42 "Task"
```

主动学习永远不会绕过当前任务的 CA 审计。在同一归一化任务族累积到至少 5 个独立的 baseline 样本与 5 个候选样本，并通过可配置的正向提升门禁之前，学习到的策略始终只是有界候选（或 shadow 观测）。受控配对还需匹配随机种子、模型、应用/工作流/技能目录配置、工作区快照、目标与编排模式；重复使用同一个 pair ID 不会增加独立样本数。汇总命令会报告实际观测到的样本数、成功率、P50/P95 奖励、延迟、prompt token、轮次、工具调用，以及各回放臂是否真正可比；它不会凭空合成缺失的反事实结果。

### 从源码构建

```bash
git clone https://github.com/doiito/gliding_horse.git
cd gliding_horse

# 编译 glidingcode 二进制（release）
cargo build -p code_cli --release
./target/release/glidingcode --help
```

---

## 🗺️ 路线图

**v0.1.x 系列 — 已发布**（当前：`v0.1.7.preview`）
- Linux（x86_64 / aarch64，musl 全静态）、macOS（Apple Silicon）、Windows（x86_64）预编译二进制，发布在 Releases 页面
- 支持 HTTP SSE 与 stdio 的 MCP 集成，通过可重复 `--mcp-server` / `--mcp-server-stdio` 标志配置
- 检查点恢复、显式 JSON-LD DAG 工作流执行，以及持久化的持续学习审计入口
- 可复现的 L0 / L2 / L3 / HNSW / Poincaré 基准测试（`examples/readme_performance.rs`）

**v0.2.x 系列 — 进行中 / 规划中**
- 完善 Center + Edge 联邦原型（`apps/software_engineering_team`），包括 Edge Daemon 的 Docker 沙箱
- 原生 Web 仪表盘（智能体监控与任务管理）
- Python/TypeScript SDK 简化集成
- 技能市场原型与社区插件注册表
- 多模型路由与成本感知调度

**v0.3.x+ 系列 — 未来**
- Kubernetes 部署算子，生产级弹性伸缩
- 跨 Edge 节点的分布式智能体网格
- 多模态智能体支持（视觉、音频）
- 多轮对话记忆压缩

---

## 📊 性能目标

| 操作 | 目标延迟 | 目标吞吐量 |
|------|---------|-----------|
| L2 持久化节点写入（Oxigraph 支撑的 Blackboard） | ~2ms | 500 ops/sec |
| L3 冷投影 | ~15ms | 66 ops/sec |
| L0 redb KV 读取 | ~1ms | 1000 ops/sec |
| HNSW 搜索（万级向量） | ~1ms | 1000 qps |
| Poincaré 4D 向量构造 | ~50µs | — |
| Agent ReAct 单轮 | 1–5s | 依环境/模型而定 |
| 空闲内存 | ~200MB | 随任务扩展 |

以上为**目标值**，并非已发布的基准测试结果。前五项可在 release 模式下通过
`cargo run --release --example readme_performance` 复现，命令会打印实测值、目标值
及通过/未通过状态。Agent 单轮延迟与空闲内存属于环境/模型级指标，需分别通过真实
provider 运行与 glidingcode 进程实测验证。

---

## 📚 文档

- **设计细节** → [`docs/DESIGN_DETAIL.zh.md`](docs/DESIGN_DETAIL.zh.md) · [`docs/DESIGN_DETAIL.md`](docs/DESIGN_DETAIL.md) (English)
- **核心设计理念** → [`docs/CORE_DESIGN_PHILOSOPHY.zh.md`](docs/CORE_DESIGN_PHILOSOPHY.zh.md) · [`docs/CORE_DESIGN_PHILOSOPHY.md`](docs/CORE_DESIGN_PHILOSOPHY.md) (English)
- **本体命名空间迁移** → [`docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md`](docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md) (English)
- **变更日志** → [`CHANGELOG.md`](CHANGELOG.md)
- **gRPC Proto** → [`proto/pdca_core.proto`](proto/pdca_core.proto)

---

## 🤝 参与贡献

欢迎社区贡献！

- **🐛 报告 Bug**：[GitHub Issues](https://github.com/doiito/gliding_horse/issues)
- **💡 提出想法**：[GitHub Discussions](https://github.com/doiito/gliding_horse/discussions)
- **🔀 提交 PR**：Fork → 功能分支 → PR 至 `main`

```bash
git checkout -b feat/my-feature
# 进行你的修改
cargo fmt && cargo clippy  # 保持代码整洁
cargo test                 # 确保一切正常
git commit -am '添加我的功能'
git push origin feat/my-feature
```

所有贡献者应遵守我们的[行为准则](docs/CODE_OF_CONDUCT.zh.md)。

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE)。

---

<div align="center">

觉得有用就点个 ⭐ —— 和我们一起构建未来 AI 的基础设施。

[![GitHub stars](https://img.shields.io/github/stars/doiito/gliding_horse.svg?style=social&label=Star)](https://github.com/doiito/gliding_horse)

*"智慧并非继承而来；它建立在先辈的肩膀之上。"*

</div>

<a href="https://www.star-history.com/?repos=doiito%2Fgliding_horse&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=doiito/gliding_horse&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=doiito/gliding_horse&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=doiito/gliding_horse&type=date&legend=top-left" />
 </picture>
</a>
