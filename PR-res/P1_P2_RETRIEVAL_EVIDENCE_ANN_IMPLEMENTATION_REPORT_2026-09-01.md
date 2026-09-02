# P1 / P2 检索、证据与 ANN 健康能力实施报告

日期：2026-09-01  
范围：Gliding Horse 核心库、`hyperspace-engine` 与 `gliding_code`；不涉及集群能力，也不涉及 `software_engineering_single`、`software_engineering_team`。

## 结论

P1 与 P2 项目已经按“确认、设计、开发、测试”完成。实现为本项目原生代码：没有引入外部项目源码、运行时依赖或命名痕迹。

其中，P1-A 同时修复了已确认的语义检索问题：不再把随机 `task_iri` 当作向量查询文本。P1-B 为任务执行轨迹提供可验证、可封存的本地证据链。P2 提供只读 ANN 健康检查与维护建议，绝不自动修改索引。

## P1-A：受控融合召回与语义查询修正

### 确认

原有部分调用路径把任务 IRI 直接传入语义检索。任务 IRI 通常是随机或结构化标识，不具有稳定的语义，因此会造成无关向量召回。该问题属实。

同时，密集检索对精确代码标识、路径、中文短词的覆盖不稳定，适合补充受权限约束的词法召回；但不应在没有离线评估和人工准入时直接替代现有召回结果。

### 设计

- 新增版本化 `ContextRecallQuery`，显式分离 `task_iri`（身份、作用域）与 `semantic_text`（仅由任务目标、原始任务、快照的 what/why、期望输出、验收标准组成）。
- 对字段和总查询长度做边界控制、空白归一化；旧接口退化为无语义文本，杜绝回退到 IRI 向量化。
- 在 `hyperspace-engine` 新增进程内、可从持久化 payload 重建的 BM25 风格词法索引。只有带 `_gh_lexical_recall: true` 的显式批准文档入索引。
- 密集与词法候选以确定性 RRF（`K=60`）融合，严格保留原有过滤条件与时间衰减规则。
- 默认处于 shadow：会计算并记录融合候选数量，但运行时仍返回现有语义基线。只有独立离线评估、演化闸门和人工准入完成后，才能显式开启融合结果。

### 开发

- 新增 `src/memory/context_recall.rs`，并将任务上下文、调度器、SA 执行路径接入结构化查询。
- 新增 `crates/hyperspace-engine/src/lexical.rs`，在插入、更新、删除、恢复后维护或重建词法索引。
- `HyperspaceStore` 新增受控的可召回写入接口与 `search_fused_recall`；普通写入、遥测及未授权内容不会被词法召回索引。
- 技能发现和经验写入按不同保留策略标注，检索衰减语义保持一致。

### 测试

- 不透明任务 IRI 不进入语义查询；查询归一化和长度边界通过。
- 词法索引覆盖标识符/CJK 分词、硬过滤、更新删除后无陈旧 posting。
- 融合召回只包含显式批准内容、严格遵守过滤条件，并可在 checkpoint 后从持久化 payload 重建。

## P1-B：任务执行证据账本

### 确认

仅依赖普通 L0 日志条目无法为并发追加、顺序完整性、终态封存与被覆盖后的检测提供强保证。任务执行记录需要一个以任务为边界的原子账本。

### 设计

- 在本地 L0（redb）中增加帧、头指针、封存记录三个表；不引入集群协调。
- 每帧包含连续序号、上帧哈希、事件哈希和帧哈希；头指针与帧在同一事务内追加。
- 封存时原子写入帧数、Merkle 风格链根（末帧哈希）和终态，之后拒绝追加。
- 普通 L0 日志保留为便于查询的投影，但校验以账本帧为准，并检测投影替换。
- 默认不保存原始 payload；捕获任务载荷必须显式 opt-in，避免无意扩大敏感数据留存。

### 开发

- 扩展 `L0Store` 的事务性 `try_append_task_evidence`、`try_seal_task_evidence`、读取与封存查询接口。
- 重构 `TaskExecutionJournal`：分配序号、哈希链、冲突重试、封存和验证均基于耐久账本。
- `TaskFinalizer` 在任务终态时封存证据链。
- `glidingcode --verify-task-evidence <TASK_IRI>` 可在不启动 LLM、图数据库、监控器或 TUI 的情况下读取并验证已封存证据。

### 测试

- 跨重启的顺序、哈希链与封存校验通过。
- 8 个独立 journal 实例并发追加同一任务，得到连续的 `0..7` 序列并成功封存。
- 封存后追加被拒绝；手动替换 L0 投影会被验证器报告。
- 默认无原始载荷与显式载荷捕获两条路径均通过。

## P2：ANN 健康检查

### 确认

ANN 索引会受候选不足、删除墓碑、WAL 积压等因素影响。仅使用 ANN 自身结果无法判断召回质量，且在线自动重建会给生产任务带来额外风险。因此需要有精确基线的、只读诊断。

### 设计

- 固定种子从当前活跃条目确定性抽样；对每个样本比较 ANN 与精确全量检索的 top-k 交集。
- 报告平均 recall、候选不足率、ANN/精确查询 p50/p95、活跃/已分配/墓碑数和 WAL 大小。
- 根据阈值给出 `Healthy`、`CheckpointRecommended`、`MetadataVacuumRecommended`、`ReindexRequired` 等建议；不执行任何维护动作。
- 报告只保存聚合指标，不保存探针文本或嵌入向量。

### 开发

- `hyperspace-engine` 增加原始 ANN 与精确检索的受控探针接口及索引统计。
- `HyperspaceStore` 增加配置校验、报告与建议生成。
- `AnnHealthMonitor` 将聚合诊断作为 `AuditEvidence` 写入 L0。
- `glidingcode --inspect-ann-health` 由操作员显式触发；结果是诊断和建议，不会修改索引。

### 测试

- ANN 结果和精确基线均参与报告；recall 边界有效。
- 探针前后索引条目数不变，确认检查无写入式维护副作用。

## 验证记录

以下命令均在隔离构建目录中成功完成：

```text
cargo test -p hyperspace-engine lexical --lib
cargo test -p glidinghorse execution_journal::tests --lib
cargo test -p glidinghorse fused_recall_indexes_only_explicitly_approved_documents_and_honors_filters --lib
cargo test -p glidinghorse lexical_recall_rebuilds_from_durable_payload_after_checkpoint --lib
cargo test -p glidinghorse ann_health_probe_uses_exact_baseline_and_never_mutates_index --lib
cargo test -p glidinghorse context_recall --lib
cargo test -p glidinghorse memory::scheduler::tests --lib
cargo test -p glidinghorse task_finalizer::tests --lib
cargo test -p code_cli --bin glidingcode
cargo check -p hyperspace-engine -p glidinghorse -p code_cli
cargo fmt --all -- --check
git diff --check
```

构建与测试过程中仅出现项目既有的 deprecated/unused 测试告警；没有新增编译错误或测试失败。

## 运行约束与后续操作

- 融合召回当前默认 shadow，不能仅凭本次实现直接提升为主路径。应先为目标任务族收集独立标注集，完成离线基线比较，再经已有演化治理流程人工批准。
- ANN 健康建议需要由操作员在低峰窗口执行相应 checkpoint、清理或重建；本实现故意不自动执行。
- 证据验证要求任务已经进入终态并封存。未封存任务返回不可验证状态是预期行为。

## 应用级端到端验证补充

本节使用新编译的 release `glidingcode`，所有工作区、L0、图数据库、向量索引和测试夹具均位于 `/tmp`。项目工作树没有被任务修改。

### P1-A：实际调用了结构化查询与 shadow 融合

真实模型任务成功结束（1 个 ReAct turn、0 次工具调用、退出码 0）。随后在带有默认技能索引的真实任务上启用 scheduler debug 日志，得到如下运行证据：

```text
Fused retrieval evaluated in shadow mode
query_version=1
field_sources=[objective, original_task, five_w2h_what, five_w2h_why,
               expected_output, success_criteria]
fused_candidate_count=10
```

这证明应用运行时使用的是版本化语义字段，而不是任务 IRI，并且融合候选确实被计算。日志没有记录原始查询文本，保持了调试可观测性与最小暴露之间的边界。由于处于 shadow，返回路径仍是基线，符合准入前不改变线上行为的设计。

### P1-B：真实任务证据链可在无密钥时验证

对同一真实任务执行以下命令，并显式移除 `DEEPSEEK_API_KEY` 与 `AGENT_OS_GATEWAY_API_KEY`：

```text
glidingcode --verify-task-evidence iri://task/e53a22c6-59e7-4ea2-9e1d-dea450edc1ce
```

结果为 `frame_count=15`、`sealed=true`、`valid=true`、`failures=[]`。这既证明真实终态已封存，也验证了本地管理命令不再错误依赖模型凭据。

### P2：真实 ANN/精确基线探针与无副作用复测

默认安装没有真实语义嵌入服务时，`--inspect-ann-health` 现在明确以非零退出码提示配置 `ollama` 或 `oneapi`；它不会以 hash fallback 伪造质量指标。

为覆盖完整探针路径，端到端测试提供了一个仅监听 `127.0.0.1` 的确定性 Ollama 协议夹具，真实任务先持久化 44 个技能向量。两次连续探针均得到：

```text
samples_evaluated=32
top_k=10
mean_recall_at_k=0.715625
candidate_shortfall_rate=0.0
active_vectors=44
allocated_slots=45
tombstone_slots=1
active_wal_bytes=0
recommendation=reindex_required
```

低 recall 是 4 维确定性测试向量造成的预期受控退化，验证了 `reindex_required` 阈值分支。两次探针前后的 `active_vectors` 均为 44，说明探针只写入 L0 审计证据、没有自动 checkpoint、清理或重建 ANN 索引。

### 端到端发现并修复的入口问题

`--verify-task-evidence`、`--inspect-ann-health` 等命令虽不需要调用 LLM，但此前会在配置初始化阶段要求 API Key。现已将本地管理命令改为可创建无凭据 gateway 配置；单次 Agent 任务、恢复和交互模式仍严格要求凭据。ANN fallback 的内部错误也改为可操作的配置说明。

### 非 P1/P2 的观察

一次“单句概述工作流”的模型任务在系统状态上成功，但模型最终摘要偏向“输出单句工作流摘要”的元描述，而非完整用户所要内容。该现象属于通用终态质量/验收语义门问题，不是 P1/P2 的召回、证据或 ANN 实现缺陷；本次不改变其产品策略。对于生产任务，建议后续以独立 CA/AA 证据或显式输出契约校验作为单独优化项。
