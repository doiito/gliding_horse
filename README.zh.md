# 流马智能体操作系统
<div align="center">

![Gliding Horse Logo](assets/logo.jpg)

**工业级 AI 智能体操作系统 · Rust 构建**  [![Star on GitHub](https://img.shields.io/github/stars/doiito/gliding_horse?style=flat)](https://github.com/doiito/gliding_horse)

*受诸葛亮木牛流马启发 — 古老智慧与现代 AI 的融合*

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![gRPC](https://img.shields.io/badge/gRPC-Protocol-green.svg)](https://grpc.io/)
[![Knowledge Graph](https://img.shields.io/badge/Knowledge%20Graph-Oxigraph-purple.svg)](https://oxigraph.org/)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue)](https://github.com/doiito/gliding_horse/releases)

---

[**中文**] · [**English**](README.md) · [**设计细节 →**](docs/DESIGN_DETAIL.zh.md)
[**medium URL**](https://medium.com/@doiito-sun)
[**中文稀土掘金**](https://juejin.cn/column/7647868075887165450)
[**中文博客园**](https://www.cnblogs.com/doiito)
[**中文CSDN博客**](https://blog.csdn.net/2604_96270735)
[**B站播客**](https://space.bilibili.com/1547455799/lists)

</div>

---

## 🎉 v0.1.4.preview 发布

我们自豪地宣布 **流马智能体操作系统 v0.1.4.preview 发布** — 本次发布是一次重大的稳定性与智能感知升级，涉及 85 个文件的架构重构与功能增强（+6134 / −4127 行）。

**发布亮点：**

| 领域 | 说明 |
|------|------|
| **SA 模块单体拆分重构** | 将 3408 行的 `sa/mod.rs` 拆分为 8 个专注模块（`types`、`planning`、`execution`、`intervention`、`agent`、`process`、`stats`、`actions`）——代码库中最大的结构性重构。 |
| **统一时间线系统** | 新增 `TimeRange` + `TimelineEntry` 跨子系统时间线查询框架。包含指数时间衰减重排序（`apply_time_decay()`）——旧记忆优雅降权，确保最新相关记忆优先。 |
| **5W2H 维度审计** | 正式化的维度级审计，含 `AuditStatus`（通过/警告/失败）三级状态，失败维度自动连锁到因果引擎进行根因分析——告别黑盒"通过/不通过"。 |
| **知识图谱上下文注入** | Agent 系统提示现自动注入相关 KG 实体，SA 干预后的新知识立即可见。 |
| **时间感知系统提示** | Agent 现接收当前时间与会话上下文，支持时间敏感推理和检查点一致性恢复。 |
| **Hyperspace 集成主动感知** | 经验查询优先使用 HyperspaceStore 语义搜索（时间衰减 λ=0.5），替代 L0 标签子串匹配，优雅降级至原路径。 |
| **因果集成工作区监控** | 文件创建/修改/删除事件记录 `CausalObservation`，支持根因追溯。任务前快照 + 目标感知文件清单注入。 |
| **LLRU 冷归档（技能图谱）** | 自动将冷数据技能归档至 L0（`storage_tier = L0Permanent`），`find_stale_skills()` 触发过期技能重新索引。 |
| **TimelineStore 突变追踪** | 每次技能图谱结构变更（注册/更新/删除/链接/MOC）记录 `GraphMutation`，确保 `pending_mutations()` 反映真实图活动。 |
| **旧 DAG 工作流引擎移除** | 移除基于 petgraph 的 211 行 `DagEngine`——PDCA 7 级自适应执行已完全取代旧有 DAG 编排。 |
| **HNSW 无锁并发安全** | `IncrementalHNSW` 的 `visited_gen` 从 `Vec<usize>` 改为 `Vec<AtomicUsize>`，消除并发搜索数据竞争，保留无锁路径高吞吐。 |
| **PDCA P0: 预检运行时错误** | 修复 TL 无法匹配技能时的崩溃——优雅降级至 L0 而非中止工作流。 |
| **PDCA P1: PA 无法创建** | 修复 `PauseOnError` 时 PA 创建失败——补全 `execute` 字段。 |
| **PDCA P2: L0 降级无输出** | 修复指标不可用时静默输出丢失——默认指标确保始终产出可读响应。 |
| **TL: pend 始终为 0** | 修复 TL 聚合中 `pend_sum` 误用 `sum` 而非子任务实际 pend 值的问题。 |
| **错误处理规范化** | 移除 `execution.rs` 中两处 `.expect("RwLock poisoned")` panic ——优雅降级优于崩溃。 |
| **技能图谱安全增强** | 新增技能注册/查询路径的访问控制检查点，及 MCP 工具调用安全过滤。 |

---

## 🎉 v0.1.3 正式发布

我们自豪地宣布 **流马智能体操作系统 v0.1.3 正式发布**。

**v0.1.3 新增核心特性：**

| 特性 | 说明 |
|------|------|
| **因果引擎 (Causal Engine)** | 全新独立因果分析子系统，包含 `CausalEngine`、`FusionEngine`、`CausalStore` 和类型化 `CausalFactor`。支持跨智能体操作的因果推理，融合多因子分析——识别根因、传播故障链、计算智能体决策的因果图。 |
| **统一图后端 (Graph Backend)** | 整合的 `GraphBackend`（约 1200 行）替代碎片化的图存储——提供统一的节点/边 CRUD 优化接口，支持批量操作、子图提取和跨知识层的路径查找。 |
| **图特征计算 (Graph Features)** | 新增 `graph_features` 模块，计算结构特征向量（度中心性、聚类系数、PageRank、介数中心性），并通过特征距离比较实现图相似度评分。支持跨认知快照的定量图分析。 |
| **快照时间线 (Snapshot Timeline)** | 技能图快照与持久化的快照后 mutation 记录，支持时间点恢复和差异查询。它仍是实验性图时间线，并非完整会话历史或防崩溃恢复系统。 |
| **自我意识模块重构** | 自我意识（SA）模块重大重写（+410 行），增强智能体状态监控、环境感知和自适应行为。与因果引擎集成实现自动自我诊断。 |
| **5W2H 维度审计增强** | 扩展了维度级审计功能，每个维度增加了更深层的因果归因。What/Why 失败现可链入因果引擎进行自动根因分析。 |
| **高级特性设计文档** | 新增全面的 [`ADVANCED_FEATURES_DESIGN.md`](docs/ADVANCED_FEATURES_DESIGN.md)，涵盖图后端架构、因果推理设计、时间线快照语义和性能基准。 |
| **图后端基准测试** | 新增基准测试套件（`benches/bench_graph_backend.rs`），涵盖节点/边读写、子图提取和路径查找吞吐量。 |
| **Gliding Code TUI 优化** | 终端客户端的引擎和 TUI 改进——更好的 Markdown 渲染、增强的 MCP 服务器生命周期管理、内部重构以提升可维护性。 |
| **Bug 修复** | 修复了 L2 `write_node` 中的重复二级索引更新问题，该问题在并发写入时会导致索引不一致。 |

---

## 🎉 v0.1.2 正式发布

| 特性 | 说明 |
|------|------|
| **HyperspaceEngine 向量引擎** | 生产级嵌入式向量引擎，支持 HNSW ANN 搜索、预写日志（WAL）、切线空间剪枝及运行时可选度量空间（Poincaré、Cosine、Euclidean、Lorentz）。 |
| **技能图谱认知网络** | 超图组合、Poincaré 结构嵌入、PageRank/Betweenness/社区发现算法、因果故障分析、实验性的时序快照/回滚、6 项形式化不变式检查、混合文本×结构搜索。 |
| **语义技能发现引擎** | `SkillDiscoveryEngine` 集成 HyperspaceStore 向量搜索，用余弦相似度替代纯 Jaccard 标签重叠的 `suggest_links()`，支持 BFS 路径发现、组合树构建和冲突检测。 |
| **Oxigraph SPARQL 桥接** | 技能图谱通过 SPARQL INSERT/DELETE 投影到 Oxigraph RDF 存储，并使用命名图隔离；尚未实现 RDF 到技能图的反向同步。 |
| **L2 Blackboard 记忆系统** | 带 JSON-LD 线程、投影、消息包的类型化文档存储，LRU 淘汰策略，支撑长期智能体上下文。 |
| **工作区监控器** | 实时文件系统感知引擎，10 种事件触发器，60 秒异常去重，5W2H 约束检查。 |
| **批处理智能体管理器** | 基于滑动窗口的批处理组件，支持可配置触发器、事件总线集成和业务域隔离；根 gRPC 服务已接线 opt-in 自定义事件及 cron/window 消费，流式请求复用共享服务状态；事件持久化重放仍待完成。 |
| **Gliding Code TUI 终端助手** | 交互式终端 UI（ratatui v0.28），支持 Markdown 渲染、Mermaid 图表、MCP 服务器集成、断点恢复、多模型后端。 |

---

## 什么是 Gliding Horse？

一个 **基于 Rust 构建的 AI 智能体操作系统**，通过 PDCA 循环编排多智能体，实现协调、可审计和自我改进的系统。——正如诸葛亮当年用木牛流马在险峻山路上革新了后勤运输。

> "我们不只构建智能体；我们构建**驾驭集体智能的基础设施**。"

### 核心技术栈

| 层级 | 技术 | 职责 |
|------|------|------|------|
| **核心编排** (Rust) | `PDCA 循环` · `5W2H 本体` · `事件总线` | 智能体编排与生命周期管理 |
| **技能图谱** | `RDF` · `6 种链接类型` · `18 模块` | 动态认知网络 |
| **记忆系统** | `L0 Sled` · `L1 Session` · `L2 Blackboard` · `L3 Projection` · `MESI 一致性` | 带预取的分层记忆 |
| **知识图谱** | `Oxigraph RDF` · `SPARQL 1.1` · `代码 AST` · `命名图` | 跨子系统统一存储 |
| **HyperspaceEngine** | `HNSW ANN` · `WAL` · `Poincaré/Cosine/Euclidean` · `混合搜索` | 嵌入式向量嵌入引擎 |
| **Gliding Code TUI** | `ratatui` · `crossterm` · `MCP` · `断点恢复` | 终端 AI 编程助手 |
| **数据总线** | `JSON-LD 1.1` · `@id/@type/@context` · `命名图` | 通用互操作层 |
| **网关** | `gRPC` · `HTTP (兼容 OpenAI)` · `MCP` | 生产级接口 |
| **感知引擎** | `10 种触发器` · `异常去重` · `5W2H 约束检查` | 主动监控 |
| **智能体工作流** | `PA/DA/CA` · `工具系统` · `检查点` · `追踪操作` | 多智能体执行 |

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
生产级空间记忆引擎，支持 **运行时可选度量空间**（Poincaré、Cosine、Euclidean、Lorentz）。内置 **HNSW 近似最近邻搜索**、CRC32 校验的**预写日志（WAL）**（3 种同步模式）、**切线空间剪枝**（优化 Poincaré 球搜索）、JSON-LD 元数据索引（RoaringBitmap 位图过滤器）以及双空间**混合搜索**（文本 × 结构）。独立 crate，零外部向量数据库依赖。

### 2. 技能图谱认知网络
动态内存认知网络，**6 种语义链接类型**（前置依赖、组合、关联、替代、扩展、泛化）。核心能力包括：基于图谱拓扑的 **Poincaré 结构嵌入**（前置依赖深度 + 标签域指纹）；**超图组合**——一等公民 `Hyperedge` 与 `CompositionType`（顺序、并行、条件、可选、回退）；**图算法**（PageRank、介数中心性、标签传播社区发现、DFS 前置链、Tarjan SCC 环检测）；**因果故障分析**与根因推断；**形式化不变式验证**（6 项检查：无环、链接可达、组合可达、无废弃前置依赖、5W2H 有效、安全等级有效）；**时序版本管理**与快照回滚。

### 3. 泛化 PDCA — 7 级自适应执行
通过 5W2H 元数据动态选择 7 级复杂度（L0 即时 → L5 递归 → L6 应急）。同一引擎同时处理即时查询与数周工程项目——无需僵硬的固定流程。**PA/DA/CA 智能体角色**，基于模板的提示词构建。

### 4. 语义技能发现引擎
`SkillDiscoveryEngine` 包装 `HyperspaceStore` 实现基于向量的语义技能搜索。`suggest_links()` 从 Jaccard 标签重叠优雅降级到余弦相似度搜索。内置 BFS 路径发现（`find_skill_chain()`）、组合树构建（`get_skill_tree()`）和冲突检测。

### 5. CPU 缓存记忆 — 4 层结构 + MESI 一致性
**L0** Sled 磁盘存储 → **L1** 会话上下文 → **L2** Oxigraph RDF + Blackboard → **L3** SPARQL 投影缓存。仓库实现了借鉴缓存一致性的协调与预取组件；目前没有已发布的端到端延迟或多智能体一致性基准。

### 6. JSON-LD 通用数据总线 — 内部互操作子集
内部 JSON-LD 工具支持本仓库使用的 `@context`、`@id`、`@graph`、framing、校验和路由；这不是完整 JSON-LD 1.1、SHACL 或通用 RDF 互操作性声明。

### 7. 自进化技能图谱 — 自主学习
AA 智能体在任务完成后记录知识片段、链接和演化建议。`/learn`/`/reduce` 提供显式的技能获取与归并操作；建议不会自动应用，因为 typed patch 的验证、安全与冲突门禁尚待实现。`BootstrapEngine` 从文件系统摄取 Markdown 格式技能。

### 8. 通用知识图谱 — 统一认知骨干
技能、记忆、任务和代码知识可通过命名图使用共享 **Oxigraph RDF 存储**；已接线的生产者可进行受范围约束的 SPARQL 联合查询。tree-sitter 解析的代码 AST 会转为 RDF 三元组。`SkillGraphStore` 将变更投影到语义存储；尚未实现 RDF 到技能图的反向同步。

### 9. 5W2H 维度级审计 — 精准回滚
CA 独立审计 7 个维度。What/Why 失败 → 重新分析。How/Where 失败 → 重新规划。When/HowMuch 失败 → 条件通过。告别黑盒"通过/不通过"——精确定位问题根因。

### 10. 主动感知引擎 — 防患于未然
10 种执行触发器，60 秒异常去重窗口。监控截止时间违规、预算超支（>80% Token）、角色不匹配、环境冲突。**工作区监控器**实时检测文件创建/修改/删除。必要时自动升级到人工处理。

### 11. 微工具系统 — 驾驭大型输出
结果 >8KB 时自动生成可对话的微工具（如"search_in_results"）。将 50KB+ 的笨重输出转变为 LLM 上下文中可交互、可查询的产物。

### 12. MCP 集成 — 一个协议连接一切
标准 **Model Context Protocol** 连接 GitHub、Slack、Jira 等任意 MCP 兼容服务器。运行时动态发现工具。支持 HTTP SSE 和 stdio 两种传输模式，通过可重复 `--mcp-server` CLI 标志配置。

### 13. 检查点与恢复 — 显式会话管理
关键执行点会保存会话检查点，`--resume <task_iri>` 和 `--list-checkpoints` 提供显式会话管理。崩溃恢复和完整长任务回放仍需故障注入与端到端验证后才能作为能力宣称。

### 14. Center + Edge 联邦 — 本地自治，全局编排
Go Center 负责工作流编排（Temporal）、项目管理、智能体注册。Rust Edge 运行本地 LLM 执行与 Docker 沙箱。VS Code 插件提供实时开发者感知。无单点故障。

---

## 🖥️ Gliding Code — 终端 AI 编程助手

**Gliding Code** 是一款基于终端的 AI 编程助手（`ratatui` TUI），将流马智能体操作系统的知识图谱与智能体编排能力直接带入命令行——无需 IDE。

**功能特性：**
- 交互式 TUI，支持 **Markdown 渲染**（`tui-markdown`）和 **Mermaid 图表**
- **MCP 服务器集成**，通过 `--mcp-server` 和 `--mcp-server-stdio` 标志
- **检查点恢复**：`--resume <task_iri>` 和 `--list-checkpoints`
- **多模型后端**：DeepSeek、兼容 OpenAI 的 API
- **PDCA 工作流执行**：规划/执行/检查/行动完整周期
- **可配置**：工作区、最大迭代次数、最大 PDCA 周期、日志级别

![Gliding Code 演示](assets/screenshot.gif)

![知识图谱实战](assets/gliding_code_kg.JPG)
*知识图谱可视化——实时实体关系、代码结构理解、基于 Oxigraph RDF 的跨子系统感知*

![编程任务完成](assets/gliding_code.JPG)
*任务完成界面——AI 智能体成功分析并解决编程任务，全程可追溯*

---

## 🚀 快速开始

### 直接下载 — Gliding Code

无需任何依赖。下载、解压、直接运行：

| 平台 | 下载 |
|------|------|
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

# 从检查点恢复
./glidingcode --resume task:abc123
```

### 从源码构建

```bash
git clone https://github.com/doiito/gliding_horse.git
cd gliding_horse

# 编译 glidingcode 二进制（release，约 51 MB）
cargo build -p code_cli --release
./target/release/glidingcode --help
```

---

## 🗺️ 路线图

**v0.1.x 发布系列**（稳定化）：
- Linux/macOS/Windows 多平台二进制分发
- Linux musl 全静态编译（零依赖）
- MCP 工具生态扩展与文档完善
- 检查点恢复功能的测试与打磨

**v0.2.x 发布系列**（规划中）：
- 原生 Web 仪表盘（智能体监控与任务管理）
- Python/TypeScript SDK 简化集成
- 技能市场原型与社区插件注册表
- 多模型路由与成本感知调度

**v0.3.x+ 发布系列**（未来）：
- Kubernetes 部署算子，生产级弹性伸缩
- 跨 Edge 节点的分布式智能体网格
- 多模态智能体支持（视觉、音频）
- 多轮对话记忆压缩

---

## 📊 性能目标

| 操作 | 延迟 | 吞吐量 |
|------|------|--------|
| L2 节点写入 (Oxigraph) | ~2ms | 500 ops/sec |
| L3 SPARQL 投影 | ~15ms | 66 ops/sec |
| L0 Sled KV 读取 | ~1ms | 1000 ops/sec |
| Hyperspace HNSW 搜索（万级向量） | ~1ms | 1000 qps |
| Poincaré 嵌入（4 维） | ~50µs | — |
| Agent ReAct 单轮 | 1-5s | 0.2-1 turns/sec |
| 空闲内存 | ~200MB | 随任务扩展 |

---

## 📚 文档

- **设计细节** → [`docs/DESIGN_DETAIL.zh.md`](docs/DESIGN_DETAIL.zh.md) · [`docs/DESIGN_DETAIL.md`](docs/DESIGN_DETAIL.md) (English)
- **核心设计理念** → [`docs/CORE_DESIGN_PHILOSOPHY.zh.md`](docs/CORE_DESIGN_PHILOSOPHY.zh.md) · [`docs/CORE_DESIGN_PHILOSOPHY.md`](docs/CORE_DESIGN_PHILOSOPHY.md) (English)
- **本体命名空间迁移** → [`docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md`](docs/16-ONTOLOGY_NAMESPACE_MIGRATION.md) (English)
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
