# RuVector 与 Gliding Horse 单机融合分析报告

- 日期：2026-09-01
- 范围：仅分析 PR-res/ruvector 的本地材料，并对照当前 Gliding Horse 核心与 Gliding Code。
- 明确排除：集群、Raft、复制、CRDT 共识、联邦学习、分布式 Swarm、自动分片、云部署与多节点 PostgreSQL 扩展。

## 结论

不建议把 RuVector 作为 Gliding Horse 的第二套向量库、图数据库或 Agent 编排运行时整体引入。Gliding Horse 已有单机 HNSW 持久化、带过滤和时间衰减的检索、Skill Graph、因果分析、时间线快照、执行日志，以及带基线/影子/主动模式和晋升门的策略学习。直接替换会制造双写、双索引和两套学习状态，反而削弱可解释性与可回滚性。

建议融合 RuVector 中经过重新约束后仍有价值的四类思想：

1. 受验证结果约束的轨迹学习；
2. 只在安全参数集合中选择的检索策略自适应；
3. 按任务族度量的学习漂移/退化监控；
4. 任何学习或技能演化均以快照、增量、独立验证和回滚为边界。

本路线采用“参考机制、原生重写”的融合方式：首期及后续生产路线均不复制 RuVector 代码、不把 RuVector crate 作为运行时依赖，也不把其示例改名后并入工程。每一项能力都必须围绕 Gliding Horse 既有的 TaskContext、EffectPolicy、L0、HyperspaceStore、Skill Graph、TimelineStore、CA/AA 和 PolicyGate 重新设计、实现、测试和运维。

## 非复制融合与生产级硬性原则

RuVector 材料只提供问题分解、候选算法和反例。Gliding Horse 的实现必须满足下列硬性原则：

1. **领域自洽**：新类型、IRI、状态机和错误语义以 Gliding Horse 的任务、技能、策略和证据模型为唯一事实来源；不得维护平行的 RuVector memory、graph、DAG 或 reward store。
2. **不复制实现**：不复制 crate 源码、接口或示例代码；只可在设计文档中引用其机制名称和本地材料路径。算法重新推导为最小需求，并用本项目的命名、数据模型和测试实现。
3. **生产完整性**：任何新模块必须提供 schema version/migration、持久化和恢复、幂等性、并发边界、超时/错误语义、指标、告警事件、权限边界、回滚和保留/GC 策略；没有这些内容不得进入 Active 路径。
4. **证据优先**：只有独立 CA/AA 证据可提升经验、策略或技能候选；模型自评、点击式信号或单次成功只能作为候选特征，不能直接改变生产决策。
5. **渐进发布**：Baseline → Shadow → 受控配对 → Active 是唯一晋升路径。任何退化先冻结和回退，绝不在生产任务中自行扩大搜索范围、工具权限、文件写入权限或模型权限。
6. **单机约束**：只支持本地进程和当前持久化后端；不实现或预留 Raft、复制、CRDT、远程同步、分片与节点协调接口。

### 生产级交付清单

| 维度 | 合格标准 |
|---|---|
| API 与数据 | 稳定公共接口；schema 有版本；迁移可重复、可中断恢复；旧数据无静默丢失。 |
| 一致性 | 单写入者或明确锁顺序；重复事件幂等；崩溃后从 L0/Timeline 恢复到可验证状态。 |
| 安全与隐私 | 默认不保存原始 LLM/工具载荷；按现有调试开关显式捕获；所有外部副作用仍受 EffectPolicy 和工具门控制。 |
| 可观测性 | 每次策略选择、证据、晋升、冻结、回滚均有结构化事件、相关 ID、耗时和原因码。 |
| 质量门 | 单元、并发、持久化恢复、迁移、故障注入、属性/模糊、端到端和性能回归测试齐备。 |
| 运维 | 配置有安全默认值、上限和文档；存储保留/压缩/GC 可预测；指标异常不会自动执行不可逆操作。 |

## 调研事实与可信度边界

本地目录含 3,275 个文件、约 86 MB，含 ruvector-core、ruvector-dag、ruvector-delta-core、ruvector-gnn-rerank 等源码及测试。候选 crate 的源码/测试规模并不相同，例如 core 为 60 个 Rust 文件和 446 个测试标记，dag 为 79 个 Rust 文件和 158 个测试标记。

但该材料不是可直接复现的完整 Rust workspace：

- 该材料的根 Cargo.toml、Cargo.lock 和根许可证文件缺失；
- ruvector-core 等 manifest 使用 workspace 继承；
- 对本地副本执行 cargo metadata 失败，无法在不补齐上游 workspace 的情况下完成构建与依赖审计；
- README 中的性能与提升幅度属于材料作者声明，尚未在 Gliding Horse 的数据、模型、硬件和任务集上独立复现。

因此，报告中的收益均为待验证假设，而非承诺。任何第三方依赖进入主工程前，应固定上游版本/提交、补齐许可证证据、生成锁文件、运行 cargo deny/SBOM/安全审计，并在隔离 feature 下完成基准与回归。

## 参考实现的可改进点与原生修正

以下问题来自对本地材料的静态审查。它们不是要求修改或修补 PR-res/ruvector 中的源码，而是 Gliding Horse 在重新设计时必须主动避免的反例；不把这些约束带入原生实现，所谓“自进化”会变成不可控的自调参。

| 参考点 | 观察到的问题 | Gliding Horse 原生修正 |
|---|---|---|
| SONA 温度示例 | `TemperatureTracker` 在计算间隔前更新 `last_access`，随后读取的 elapsed 接近零，访问时的时间衰减实际没有生效。 | 由可注入时钟先读取旧时间戳、计算 elapsed、再原子更新新时间戳；后台衰减使用有上限的批处理任务，而不是每实例无限循环。为零间隔、长空闲、时钟回退和并发访问加入确定性测试。 |
| SONA 反馈重排序 | 正/负反馈以乘数直接放大基础分，既没有上下界，也没有最小样本与证据质量要求，极端反馈可使分数失真甚至为负。 | 以任务族隔离的、校准且有界的收益估计替代裸乘数；仅独立 CA/AA 结论计入晋升证据，记录不确定性、样本量与过期时间，默认不提升。 |
| SONA 检索参数自调节 | 仅以 p99 与 recall 的单次比较调节 `ef_search`，缺少样本下限、迟滞、冷却期、实验归因和回滚语义。 | 不允许在线代码自动改 HNSW 结构参数、重建索引或改写向量；只可从经过配置审查的有限策略臂中选择候选数量、衰减与排序配额，并走 baseline → shadow → 受控配对 → active。 |
| DAG 漂移检测 | 窗口以 `Vec::remove(0)` 淘汰旧值，持续运行时为 O(n)；固定基线且默认“指标升高=改善”，不适用于延迟、成本、错误率。 | 使用有界环形缓冲/`VecDeque` 与运行统计；每项指标声明 `HigherIsBetter`、`LowerIsBetter` 或 `TargetBand`，具备最小样本、鲁棒估计、置信条件、任务族隔离和版本化基线。 |
| DAG 修复策略 | `IndexRebalance`、`PatternReset`、`CacheFlush` 仅短暂 sleep 后即报告成功，属于模拟行为，不能构成生产恢复。 | 退化处理的唯一自动动作是冻结对应 Active 策略、降级 baseline/shadow、保全证据并发出结构化告警。快照恢复必须经过完整性检查、幂等锁和明确批准；不得伪报修复成功。 |
| Delta 向量编码 | `VectorDelta::from_dense` 等路径在维度为零时先做除法，需显式定义零维输入及非有限值的语义。 | EvolutionDeltaGate 只处理带 schema 的技能/策略领域变更，不复用原始向量 delta；所有输入在持久化前拒绝空维度、NaN/Inf、未知 schema 与越权范围，并覆盖序列化、恢复和模糊测试。 |
| 时序内存与 GNN 重排序 | temporal-coherence 的存储说明仍是扁平 O(n) PoC；`ruvector-gnn-rerank` 明确属于研究型重排序，未提供当前任务集的收益证明。 | 继续使用 HyperspaceStore 的持久化 HNSW 作为唯一在线检索路径；任何复杂重排序仅旁路离线回放，必须同时证明质量、延迟、成本和稳定性收益后才可 feature flag。 |
| 材料可复现性与安全边界 | 本地副本缺根 workspace、锁文件与许可证证据，且 DAG 的非 production-crypto 路径明示为占位方案。 | 当前不引入 crate 或安全实现；原生模块沿用本项目已验证的安全边界。若未来复审第三方依赖，须先完成来源、许可证、SBOM、漏洞和密码学专项审计。 |

这些修正把“学习”限定为一条可证明、可关闭、可恢复的决策链：观测先于评分，评分先于候选，候选先于影子验证，验证通过后才允许有限范围的 Active 使用。任何缺少独立证据、统计条件或恢复点的变化均停留在观测或离线实验层。

## 主工程既有参考痕迹与清理要求

静态检索确认，当前 Cargo manifest 和锁文件没有 `ruvector` crate 依赖；但 `hyperspace-engine` 中已有少量面向参考材料的注释：`metric.rs` 标注“ruvector generalization/original”，`hyper_vector.rs` 标注 `fast_acosh` “borrowed from ruvector”，`tangent.rs` 标注 TangentCache pattern。`snapshots` 与 `graph_features` 的模块文档也链接到 PR-res 中的实验材料。

这些注释本身不足以证明代码逐行复制，但由于本地材料缺少可复现上游提交、根许可证和完整 workspace，不能据此建立可审计的来源链。因此“原生重写、无直接代码复用”的目标尚未完全闭环，必须列为 P0 清理项，而不是把既有实现默认为可继续扩展的 RuVector 分支。

1. 将双曲几何代码按公开数学定义独立重新推导，明确输入维度、曲率、有限数、Poincaré 球边界和 Lorentz 双曲面约束；不保留“borrowed/original”表述或 RuVector 专有模式名。
2. 以带错误返回的内部校验 API 取代静默 `zip` 截断、零/负/非有限曲率和非法向量的隐式计算；距离 trait 的兜底行为必须是可监控的失败，而不是生成 NaN 或伪近邻。
3. 将切线缓存明确为 Poincaré 优化组件：构建时验证所有向量的指标和维度一致，增删时维护 ID/向量一致性，候选排序后必须由精确距离复核；不把它扩展为第二套索引或分片缓存。
4. 对重写后的实现补充性质测试：同一向量距离为零、距离非负/有限、维度不匹配拒绝、边界投影、序列化篡改、极小/极大曲率以及 Poincaré/Lorentz 约束；再用固定语料与当前余弦路径做召回/延迟回归。
5. PR-res 的链接只保留在本报告和设计审计记录中，运行时代码的文档改为解释本项目的约束和适用边界。若双曲功能没有可复现的端到端收益，应以 feature flag 关闭，而不是让实验特性进入默认检索路径。

## 现有能力对照

| Gliding Horse 已有能力 | 证据 | 对 RuVector 融合的含义 |
|---|---|---|
| 持久化 HNSW、metadata 过滤、标签提升、时间衰减、混合检索 | src/memory/hyperspace_store.rs | 不替换为 ruvector-core、ruvector-hybrid 或 temporal-coherence 的内存 PoC。 |
| Skill Graph、语义技能发现和依赖展开 | src/skill_graph/discovery.rs、src/skill_graph/graph_store.rs | 图检索可以改进排序/证据，但不再引入第二图存储。 |
| 图变更时间线、增量 mutation、快照与回滚 | src/snapshots/timeline.rs | 可承载 Delta Behavior 的演化审计与回退，无需 delta-consensus。 |
| 因果观测与根因推断 | src/causal/engine.rs、src/causal/store.rs | 可与漂移信号结合定位退化来源。 |
| 隐私优先的 LLM/工具执行日志 | src/core/execution_journal.rs | 可生成学习特征；默认只保留大小、摘要和哈希，不能把原始载荷默认送入学习库。 |
| 受约束策略学习、baseline/shadow/active、奖励门与回滚 | src/core/policy_learning.rs、src/core/sa/process.rs | 已具备安全在线学习骨架；应扩展候选策略，不接管为黑盒自优化。 |

## 推荐融合项

### P0-1：验证驱动的 Learning Trajectory

RuVector 的 ReasoningBank/Agentic-Jujutsu 示例把操作序列、成功评分和复盘沉淀为可复用轨迹。对应材料为 examples/agentic-jujutsu/learning-workflow.ts。

Gliding Horse 应原生新增轻量 LearningTrajectory 记录，而不是引入 Agentic-Jujutsu 或其版本控制包装。该模块以本项目的 L0、Journal、TaskAuditKnowledgeEvidence 和 Timeline 为基础重新实现：

- 输入：任务族、策略臂、检索到的经验/技能 IRI、计划 DAG 摘要、工具名序列、耗时、token、失败分类；
- 结果：CA 独立核验、AA 终态、奖励、文件/命令证据的引用；
- 隐私：沿用 TaskExecutionJournal 默认哈希引用；原始 prompt、LLM 响应、工具参数和结果仅在显式调试模式可用；
- 入库：仅 AA 成功且 CA 证据完整的轨迹可成为经验候选；失败轨迹只用于负例、根因和退化检测；
- 复用：按现有任务族检索，先在 Shadow 记录命中效果，满足晋升门后再进入 Active 提示注入。

收益假设是减少重复试错、提高同类任务首轮计划质量，并使自提升有可审计训练样本。它直接复用现有执行日志、TaskAuditKnowledgeEvidence、TaskFinalizer 和策略学习，不扩大工具权限。

### P0-2：检索策略的安全自适应

RuVector SONA ADR 描述了热度、查询模式、反馈重排序和检索参数调节。对应材料为 docs/architecture/decisions/ADR-006-sona-self-optimization.md。

建议原生实现受约束的反馈闭环，且把可学习范围限制在不改变外部副作用的检索参数：

| 可控策略臂 | 现有承载点 | 约束 |
|---|---|---|
| 经验优先、技能优先、知识优先、基线 | SupervisorAgent 现有 policy candidates | 保持既有 baseline/shadow/active 与晋升门。 |
| 候选数量、语义/结构候选配额 | HyperspaceStore + SkillDiscoveryEngine | 仅改变候选排序和上下文长度；不改变工具权限或 effect policy。 |
| 记忆时间衰减系数 | HyperspaceStore.search_with_time_decay | 按任务族试验，限制在预定义区间。 |
| 高质量经验的轻量 boost | 经验卡元数据 | 只能由 CA/AA 证据赋分；不能由模型自评直接提升。 |

禁止直接让 SONA 自动改 HNSW 参数、重建索引、修改向量、写入技能图或训练/切换模型。SONA 材料中的热点缓存、查询聚类和自调参可作为后续优化灵感，但在单机任务量不足时容易过拟合或形成冷启动噪声。

### P1-1：Learning Health Monitor 与自动降级

ruvector-dag 含 LearningDriftDetector、异常检测和修复策略。其漂移检测的滑动窗口结构可借鉴，但不应直接复用：

- 该实现默认把指标升高判断为 Improving；这不适用于延迟、错误率、token 或成本；
- repair strategies 中存在模拟睡眠式修复，不是可执行的生产修复；
- 完整 ruvector-dag 还包含 QuDAG、加密、治理和同步代码，默认功能复杂度远超需求，且未启用 production-crypto 时明示使用占位加密。

建议在 Gliding Horse 原生实现小型、方向感知的 LearningHealthMonitor；它不是 ruvector-dag 的封装或裁剪版本：

1. 每个任务族分开维护 30 至 100 个近期样本和冻结基线；
2. 观察 CA 通过率、AA 成功率、重复工具率、重试数、p95 总耗时、token、无证据完成率和检索命中后成功增益；
3. 采用最小样本数、置信区间/非参数检验和指标方向，而非单一平均值；
4. 触发退化时：停止该任务族的 Active 注入和探索，切换为 Shadow 或 baseline，写入事件并要求人工批准后才晋升；
5. 将异常关联到策略版本、提示版本、技能快照、模型与嵌入版本，再交给已有 CausalEngine 给出候选根因。

这比“自动修复”更安全：系统先停止扩大影响、保留证据和回滚条件，而不是在错误归因时自行重建状态。

### P1-2：以 Delta Gate 管控技能与学习变更

Delta Behavior 的核心价值是小步变更、全局一致性、节流/阻断和稳定态偏好。对应材料为 examples/delta-behavior/README.md 与 crates/ruvector-delta-core。

Gliding Horse 已有 TimelineStore 的快照与 mutation log，推荐原生新增领域层 EvolutionDeltaGate，而非引入或改写向量增量编码库：

1. 变更前记录技能图/策略版本快照和任务族基线；
2. 变更只以候选形式写入，附带来源轨迹、CA 证据、预期收益和允许影响范围；
3. 在影子或受控配对任务中验证；通过晋升门才合并；
4. 任一硬条件失败立即回滚到最近已验证快照，并冻结相应策略臂；
5. 将变更、评估、晋升/回滚写入 L0 与 timeline，供审计和因果分析。

硬条件至少包括：不可降低独立 CA/AA 成功率、不可扩大工具/effect 权限、不可绕过人工批准、不可覆盖原始证据、不可在没有可恢复快照时应用。

### P2：离线候选重排序与行为边界研究

ruvector-gnn-rerank 提供 ANN 候选集上的扩散/MinCut 风格重排序；其源码明确标记为研究背景。它可作为离线实验，但不是当前生产依赖：

- HyperspaceStore 目前已经通过 HNSW 取候选；可在旁路收集同一候选集，比较 baseline、简单 metadata rerank 与该重排序；
- 只有在人工/CA 标注的 Recall@k、证据覆盖率和端到端成功率同时改善，且 p95 延迟满足预算时才尝试 feature flag；
- 不使用它的“精确 L2 oracle”作为生产排序，也不把 MinCut 分数当作真实性或安全性证明。

temporal-attractor-discovery 和 boundary-discovery 示例展示图结构变化点检测。可在 P2 用于识别模型升级、提示改版或技能演化后的行为断点；先离线回放 TimelineStore/执行日志，确认假阳性率后再用作 Health Monitor 的辅助信号。

## 单机目标架构

~~~mermaid
flowchart LR
    A[任务与执行日志] --> B[隐私过滤的 LearningTrajectory]
    B --> C[CA/AA 证据门]
    C -->|通过| D[经验卡与候选策略证据]
    C -->|失败或不完整| E[负例/因果观测]
    D --> F[baseline / shadow / active]
    F --> G[检索与计划执行]
    G --> H[质量、成本、延迟指标]
    H --> I[LearningHealthMonitor]
    I -->|健康| C
    I -->|退化| J[冻结 Active、回退基线]
    J --> K[Timeline 快照/回滚与根因分析]
~~~

所有节点均在单机进程与现有 L0、Hyperspace、Oxigraph/Skill Graph、TimelineStore 内运行；没有网络共识、分片、远程副本或集群控制面。

## 不建议引入或明确排除的部分

| 项目 | 决定 | 原因 |
|---|---|---|
| ruvector-core、ruvector-hybrid、ruvector-hyperbolic-hnsw | 不引入 | 与现有 Hyperspace HNSW/持久化/过滤/衰减高度重叠；core 默认特性还带存储、并行和 API embeddings，迁移风险大。 |
| ruvector-graph、PostgreSQL 扩展、Graph RAG 存储 | 不引入 | 已有 Skill Graph、统一 Oxigraph 图与本体桥；避免双图与一致性问题。 |
| 完整 ruvector-dag/SONA | 不引入 | 功能面过宽，含同步、治理和占位加密路径；仅借鉴漂移与策略选择概念。 |
| LoRA/本地 LLM/注意力/量化/稀疏推理 | 暂缓 | 当前目标是可靠地提升执行与记忆，而非新增推理运行时；需要独立模型评测、硬件和数据治理方案。 |
| GNN rerank、超曲率 HNSW、MinCut attention | 仅离线研究 | 缺少 Gliding Horse 标注集和可复现收益，复杂度高。 |
| cluster、raft、replication、delta-consensus、federation、edge-net、swarm、自动分片 | 明确排除 | 不符合当前无集群需求的边界。 |

## 分阶段实施与验收

### 阶段 0：核心来源与数值边界清理

- 对现有 `hyperspace-engine` 中带 RuVector 参考痕迹的双曲几何/切线缓存代码进行独立重写和来源审计，不引入任何 RuVector crate；
- 收紧向量维度、有限数、曲率和几何流形约束，拒绝错误输入，消除静默截断或非有限距离；
- 将 Poincaré 切线缓存保持为可关闭的单机候选优化，精确距离始终负责最终排序；
- 验收：没有运行时 RuVector 依赖或“borrowed/original”来源表述；数值性质、损坏序列化、并发索引更新和固定语料回归测试全部通过；未证明收益的实验特性默认关闭。

### 阶段 A：可观测轨迹与实验基线

- 新增 LearningTrajectory 和 LearningHealthMetric 的 L0 schema；
- 将 Journal、CA/AA、策略选择和 Timeline 版本关联，但不改变运行时决策；
- 补齐每任务族的受控 baseline/shadow 数据集；
- 验收：敏感原文默认不入库；每条轨迹可追溯到 task、证据、策略、快照；基线运行不读取/不写学习状态。

### 阶段 B：受限检索策略学习

- 把候选数量、时间衰减和经验/技能优先级注册为白名单策略臂；
- 仅在具备同任务族基线证据时探索，沿用现有 PolicyGate；
- 验收：独立 CA/AA 成功率不下降；P95 耗时和 token 预算不恶化超过预设阈值；每次晋升均可回放和回滚。

### 阶段 C：健康监控与演化门

- 引入指标方向、最小样本和退化冻结；
- 为技能/策略候选加 EvolutionDeltaGate 和快照关联；
- 验收：注入故意退化的策略后，系统能停止 Active、记录因果候选并恢复到已验证版本；不触发工具权限或工作区写入的自动扩大。

### 阶段 D：离线研究项

- 构建标注检索集与任务回放集；
- 对 GNN rerank、图变化点检测进行与 baseline 的盲测；
- 验收：只有统计上稳定的端到端收益才进入小流量 feature flag；否则保留实验报告，不进入主路径。

## 实施结果（2026-09-01）

本节记录本轮按“确认 → 设计 → 开发 → 测试”完成的交付。所有实现均为 Gliding Horse 原生 Rust 模块；未复制 `PR-res/ruvector` 中的源码、接口、示例或 crate 依赖。由于用户已明确不保留不兼容快照，本轮对不满足新数值/序列化约束的旧 Hyperspace 快照采取拒绝并删除的安全策略，而不进行静默迁移。

| 项目 | 确认与设计结论 | 原生实现 | 测试结果 |
|---|---|---|---|
| 阶段 0：来源与双曲边界 | 发现源码注释中的参考痕迹，以及维度截断、非法度量/流形输入可能穿透边界。 | 重写 `hyperspace-engine` 的 `hyper_vector`、`metric`、`tangent` 与恢复校验；向量必须维度、度量、有限数和 Poincaré/Lorentz 几何一致。切线候选优化默认不进入线上路径，最终仍精确复核。移除主工程中所有 RuVector/`borrowed from` 文字痕迹。 | `cargo test -p hyperspace-engine`：85 单测、15 集成测试全部通过。覆盖错误维度、错误度量、非法球/双曲面向量、损坏快照和并发检索。 |
| 阶段 A：LearningTrajectory | 轨迹应是可审计的终态证据，而不是第二执行日志；原始 LLM/工具载荷不能进入学习存储。 | 新增 `core::learning_trajectory`：版本化 L0 schema、任务幂等键、稳定 IRI 引用、上限/唯一性校验和隐私最小化 `TrajectoryToolStep`。仅独立 CA/AA 通过的 Active 轨迹为可复用候选。 | 覆盖幂等持久化、未知策略拒绝、非 IRI 证据拒绝、元数据长度/泄露边界。 |
| 阶段 B：受限检索学习 | 不允许在线调 HNSW、修改向量或扩大权限；只能选择上下文来源排序。 | 新增 `core::retrieval_policy`，唯一策略臂为 `baseline`、`experience_first`、`knowledge_first`、`skill_first`；接入 SA 的候选集与提示排序。成功结果缺少 CA/AA 证据时不会写入正向策略样本。 | 覆盖白名单、真正 baseline 消融和已有 PolicyGate/持久化冻结回归。 |
| 阶段 C：健康与演化门 | 退化自动动作只能是冻结，恢复必须可审计且需人工批准；指标方向不能默认“越高越好”。 | 新增 `core::learning_health` 与 `core::evolution_delta_gate`，接入 `SupervisorAgent` 终态学习闭环。健康记录按任务族/策略臂分区，支持高优、低优和目标区间指标；退化会持久化冻结上下文和 Active delta。Delta 仅允许 `Proposed → ShadowValidated → Active → Frozen → RolledBack`，最后一步需人工批准。增加 `learning_health` 配置，并在 gRPC 与 Gliding Code 的 SA 构建路径应用。 | 覆盖指标方向、最小样本、幂等记录、跨任务族隔离、伪造 Active 状态拒绝、冻结幂等和审批回滚。 |
| 阶段 D：离线研究准入 | GNN/复杂重排序没有当前任务集收益证据，不能接入线上。 | 新增 `core::offline_retrieval_eval`。它只接收 IRI 级独立相关性标注和已生成的候选排序，计算 Recall@k、MRR@k、NDCG@k、p95 延迟；结果带语料摘要、不可覆盖的 L0 键和准入原因。`admitted` 只是离线结论，绝不自动改变线上检索。 | 覆盖正向准入、召回/NDCG/延迟拒绝、原始文本或重复标识拒绝、同 ID 重放幂等。 |

### 集成中发现并修复的关联问题

全库回归时，原有 Ontology 双空间 `cross_search` 在“文本 Cosine 向量”和“结构 Poincaré 向量”维度恰好相同时直接复用坐标。新的核心度量校验正确拒绝了这一错误。已将其修正为：跨空间检索始终要求显式、版本化的 `CrossSpaceProjection`，并在目标空间重新构造目标度量向量；没有投影时明确失败。该规则防止语义不相容的坐标产生伪近邻，并有“无投影拒绝/有投影成功”的回归测试。

### 运行闭环补充复核

对 Gliding Code 实际入口复核后，发现 `learning_health` 配置此前只在 gRPC 的 SA 构建路径生效；CLI 路径没有显式应用它。现已统一接入，避免同一配置在不同入口产生不一致的自进化保护行为。

同时补齐本地单机运维闭环：

- `--list-learning-health`：以 JSON 输出按任务族隔离的健康报告；
- `--list-learning-deltas`：以 JSON 输出可审计的策略 Delta 生命周期；
- `--rollback-learning-delta <ID> --delta-approver <IDENTITY>`：仅允许对已冻结 Delta 记录人工批准的回滚。该操作**不会**重新启用策略臂，策略族仍冻结并保持 baseline，后续恢复必须是独立的、重新验证的晋升流程。

复测发现非交互式管理命令原先会执行终端恢复，向标准输出写入备用屏幕转义序列，破坏 JSON 和日志采集。已把恢复操作限制为真正启动 TUI 时执行；release 版 `--list-learning-health` 和 `--list-learning-deltas` 在空工作区均精确输出 `[]`，错误的回滚调用只输出清晰错误并返回非零状态。

### P2 补充实施：原生离线候选图重排

对 `ruvector-gnn`、`ruvector-gnn-rerank` 和 `ruvector-cnn` 进一步确认后，本轮只实施有明确单机价值、可独立验证的候选图重排实验闭环：

- 不引入第三方 crate，也不复制其接口或源码；新增 `core::offline_graph_rerank`，只在同一、显式为 Cosine 或 Euclidean 的候选集上计算瞬态邻接图。混合空间、Poincaré 空间、维度不一致、非有限数、未归一化的一阶段分数、超限候选和超限图传播均拒绝；
- 每个实验必同时生成“一阶段原始排序、按原度量精确排序、受限图扩散排序”三组 IRI 排名。图扩散以 50% 至 90% 的原始分数为主，最多 3 轮，默认 1 轮；候选数默认上限 128、绝对上限 256，避免 O(n²×d) 计算无边界扩大；
- 图扩散的准入基线是**精确重排**，而不是弱的一阶段近似排序。只有独立标签下 Recall 不退化、NDCG 增益达到既有阈值且 p95 延迟受限时，`graph-vs-exact` 评估才会 `admitted`；
- 实验输入可含向量，但不持久化。L0 仅保存原有 `OfflineRetrievalEvaluation` 的 IRI、指标、摘要和拒绝原因；新增查询命令用于审计，不会输出原始向量；
- 通过评估不等于上线。只有被准入的 `graph-vs-exact` 评估才能创建 `RetrievalRerankerCandidate` Delta，初始状态固定为 `Proposed`；该 Delta 没有任何线上读取路径。未来即使进入 Active 生命周期，也会随任务族健康退化与检索策略一起冻结。

命令行入口如下，均为非交互式纯 JSON/状态输出：

```text
glidingcode --evaluate-candidate-graph-rerank <experiment.json>
glidingcode --list-offline-retrieval-evaluations
glidingcode --propose-candidate-graph-rerank <proposal.json>
```

其中 `experiment.json` 是一次性本地输入，包含版本、候选算法标识、受限配置及独立标注案例；`proposal.json` 只包含已准入评估 ID、候选 ID、来源任务族和显式版本号。常规生产阈值仍要求至少 20 个独立标注案例。两条合成案例的命令行回放中，图扩散相对精确参考的 NDCG 提升约 0.131，但因样本数不足被正确拒绝，随后的提议命令返回非零；20 条案例的应用级测试则确认准入后仅产生 `Proposed` Delta。

`ruvector-gnn` 的通用图学习、重放/EWC 等概念目前仍没有 Gliding Horse 独立图学习标签和可复现收益，故未接入在线自进化；`ruvector-cnn` 默认实现没有可验证、固定版本的预训练视觉权重，且当前没有图像检索主链路，故明确不采用。两者均保留为未来独立立项的研究边界，而非伪生产功能。

### 最终验证

| 检查 | 结果 |
|---|---|
| `rg -i 'ruvector|borrowed from|ruvector original'`（排除 `PR-res` 和 `target`） | 无匹配；Cargo manifest/lock 亦无 RuVector 依赖。 |
| `cargo test -p hyperspace-engine` | 85 单测 + 15 集成测试通过。 |
| `cargo test -p glidinghorse --lib` | 1,564 项通过（包含候选图重排、离线评估加载/枚举和检索 Delta 健康冻结回归）。 |
| `cargo build -p code_cli --release` | 通过；正确包名为 `code_cli`，产物为 `target/release/glidingcode`。 |
| `cargo test -p code_cli` | 13 个库测试 + 8 个命令行测试通过；覆盖 20 条标注案例准入后仅创建 `Proposed` Delta。 |
| `target/release/glidingcode --help` | 通过，学习健康、Delta 审计、离线检索评估、候选图重排与人工回滚控制入口可正常加载。 |
| `target/release/glidingcode --workspace /tmp/gliding-horse-cli-empty --list-learning-health` | 通过，标准输出精确为 `[]`，没有终端转义序列。 |
| release 版 `--list-offline-retrieval-evaluations` | 通过，返回纯 JSON 审计记录；两条合成案例因低于 20 条最小样本要求被正确拒绝。 |
| `git diff --check` | 通过。 |

完整回归仍显示 5 条与本改动无关的既有警告：`SkillCausalModel` 弃用、一个测试中的未使用变量和一个未使用测试辅助函数；本轮未为消警扩大改动范围。

## 采纳门槛

本路线不采纳 RuVector crate 的直接依赖。若未来出现重新评估的提案，也必须先由架构评审明确改变这一决定，并同时满足：

1. 上游 workspace、锁文件、许可证和来源提交完整可复现；
2. SBOM、漏洞、unsafe 与供应链审计完成；
3. 相对于原生实现，具有可重复且显著的单机端到端收益；
4. 通过故障/回滚测试：失败、超时、错误标签、损坏候选和进程重启均不能污染已晋升知识；
5. 以可关闭 feature flag 隔离，且不替代当前稳定路径。

## 本地材料索引

- PR-res/ruvector/README.md：能力清单、SONA、DAG、Delta Behavior 与性能声明。
- PR-res/ruvector/docs/architecture/decisions/ADR-006-sona-self-optimization.md：温度、反馈、检索参数调节和性能/成本声明。
- PR-res/ruvector/examples/agentic-jujutsu/learning-workflow.ts：轨迹、评分、复盘和建议流程示例。
- PR-res/ruvector/crates/ruvector-dag/src/healing/drift_detector.rs：滑动窗口漂移检测实现。
- PR-res/ruvector/crates/ruvector-dag/src/healing/strategies.rs：包含模拟式修复，不能直接作为生产修复。
- PR-res/ruvector/crates/ruvector-dag/src/qudag/crypto/security_notice.rs：默认占位加密的显式风险说明。
- PR-res/ruvector/examples/delta-behavior/README.md、crates/ruvector-delta-core：受约束增量变更理念与编码实现。
- PR-res/ruvector/crates/ruvector-gnn-rerank/src/lib.rs：研究型候选重排序实现。
- PR-res/ruvector/examples/temporal-attractor-discovery/src/main.rs、examples/boundary-discovery/src/main.rs：时序图结构变化点实验。

## 最终建议

先完成阶段 0，再做阶段 A，并以阶段 B 的小范围、可回滚检索策略学习作为下一步。阶段 C 是防止自进化退化的必要配套，不能在没有监控与回滚的情况下跳过。所有阶段均应以生产模块标准原生实现，而不是 demo、示例移植或第三方代码拼接。RuVector 对 Gliding Horse 的最大价值是提供可验证的设计参照，而不是作为一个整体平台替换现有核心；集群相关能力在当前阶段全部不纳入路线图。
