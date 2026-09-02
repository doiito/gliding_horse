# Snapshot、恢复与 LLM 调用追踪专项审计

> 审计日期：2026-08-31  
> 范围：`glidinghorse` 核心、`crates/hyperspace-engine`、`apps/gliding_code`；不含 `software_engineering_single` 与 `software_engineering_team`。  
> 方法：静态源码追踪、恢复路径比对、事件链路比对与既有单元测试覆盖核对；第 9 节记录本轮实现后的复验结果。

## 1. 结论

需要优化，但不是把所有 snapshot 一律做得更细。

| 子系统 | 当前判断 | 主要问题 | 优先级 |
|---|---|---|---|
| 任务 checkpoint / 恢复 | **需要重构恢复协议** | 会话全文重复保存，但恢复状态不完整；checkpoint 间隔以 5 turn 为单位，未与外部副作用提交对齐 | P0 |
| 工作区 snapshot / 回滚 | **当前不可作为可靠恢复能力** | 元数据仅进程内保存；历史内容定位错误；无法处理快照后新增文件 | P0 |
| LLM / 工具调用追踪 | **有局部日志，不能支持完整调试复盘** | 无统一 trace、无请求—响应—工具因果关系，事件定义和实际发射/传输脱节 | P0 |
| 技能图 Timeline | **分块快照 + mutation log 的粒度合理** | 失败被吞掉、并发快照与 mutation 序列未原子绑定、rollback 后时间线不收敛 | P1 |
| Hyperspace snapshot + WAL | **核心恢复模型合理** | 单代 snapshot、无格式版本/校验/兼容检查；快照损坏而 WAL 已清理时没有回退副本 | P2 |

“粒度”要分三种看：

1. **存储粒度**：任务 checkpoint 反而过重——每隔数 turn 复制完整消息；
2. **恢复粒度**：又偏粗——不能定位到工具调用前后、不能可靠恢复在途状态；
3. **语义粒度**：字段虽多，但真正参与恢复的字段很少。

因此建议将任务恢复改为“里程碑快照 + 追加执行日志 + 外部副作用提交记录”，而不是单纯缩短完整会话快照周期。

## 2. 当前快照与恢复链路

### 2.1 任务 checkpoint：数据重、恢复语义轻

`CheckpointData` 保存完整 `session_messages_json`，并预留角色、cycle state、已完成节点、审批、工具错误、动作记录等 JSON 字段；每个角色开始、每 5 turn、强制结束和角色完成都会创建 checkpoint。见 `src/core/checkpoint.rs` 与 `src/core/agent_runner/execution.rs`。

这提供了基本的“从历史对话继续”能力，但存在以下确定性缺口：

- CLI `resume_task()` 读取并解析 `agent_state_json`，却没有将该状态应用到新的 `TaskContext`；它只用 checkpoint 名生成一段自然语言继续提示。见 `apps/gliding_code/src/engine.rs`。
- TUI 路径会恢复消息并交给 SA；SA 再以 checkpoint 名推断要跳过的 PDCA 阶段。两条恢复入口的恢复语义不同。
- `current_role`、`cycle_state_json`、`completed_nodes_json`、`pending_approvals_json` 等字段大多没有形成“读取—校验—恢复—确认”的完整协议；它们不能保证精确续跑。
- 周期 checkpoint 在新 turn 的 LLM 调用之前创建，保存的是此前消息，计数却写入新 turn；恢复后又按消息重新计数，语义存在偏移。
- 外部工具副作用、结果写入和 checkpoint 之间没有同一份 durable journal。进程在工具已修改工作区、但 checkpoint 尚未落盘时崩溃，恢复后的模型可能重复执行同一副作用。
- 每任务最多保留最近 20 个 checkpoint。简单按时间淘汰会同时淘汰重要里程碑和普通 turn 快照；长 PDCA / 多 cycle 任务的可恢复窗口不稳定。
- checkpoint 没有显式 `schema_version`、内容摘要、迁移器或“最新损坏则回退前一有效 checkpoint”的策略。

现有 SA 对“从哪个 PDCA 阶段继续”的推断是有价值的过渡实现；但它应被视为 best-effort resume，不应被表述为精确恢复。

### 2.2 工作区 snapshot：设计意图正确，实现尚不能兑现

`SnapshotManager` 的目标是以 `path -> content hash` 保存工作区清单，内容由 `ContentStore` 的版本库复用，理论上是合理的内容寻址设计。实际链路存在 P0 问题：

- `WorkspaceMonitor::initialize()` 为 `SnapshotManager` 创建的是 `InMemoryBackend`；快照元数据在进程退出后消失，无法用于崩溃恢复。见 `src/tools/workspace_monitor/mod.rs`。
- `rollback_to()` 试图按 hash 找历史内容，但 `find_content_by_hash()` 扫描的是 snapshot 元数据表，而非 `ContentStore` 的 `version_store` 表。找不到时会退化为该路径的 version 0 内容，不能保证是目标 snapshot 的内容。见 `src/tools/workspace_monitor/snapshot.rs` 与 `content_store.rs`。
- inventory 为降低扫描成本可保留空 `content_hash`；未在创建 snapshot 前强制 materialize / hash 全部文件，manifest 可能没有可恢复的内容指针。
- 回滚写文件后才判断文件是否存在，所以 `files_created` 实际不会正确统计；快照后新建、但不属于目标 snapshot 的文件不会删除，`files_deleted` 始终为 0。
- snapshot 在每个非 CA AgentRunner 开始前创建，但未与单个工具调用或任务提交绑定；频率高而复原点不精确。目前未接入自动回滚，因而尚未扩大影响。

结论：工作区 snapshot 的目标粒度并不粗，但当前保存和回滚实现不能作为可靠承诺。修复前应只把它用于非恢复性的运行期观测，不能作为自动回滚依据。

### 2.3 技能图 Timeline：粒度合理，事务边界不足

`TimelineStore` 使用“每 N 次 mutation 一份全量图快照 + 中间 mutation log”的结构，默认 N=100。这是图演化的合适折中：读取时从最近全量快照回放增量，避免每次 mutation 复制完整图，也避免无限长 replay。持久化重建和 compensation rollback 已具备基础。见 `src/snapshots/timeline.rs`。

需要补强的不是将 N 改小，而是持久化边界：

- mutation 或 full snapshot 写入失败仅记录 warning，调用方无法得知失败；随后仍可能清理已持久 mutation，形成耐久性空洞；
- `backend.snapshot()` 与 `mutation_count` 读取、mutation 日志清理之间没有共用提交屏障；并发 mutation 时，快照包含的状态与标记 mutation count 可能不一致；
- rollback 只修改 backend。它没有提交 rollback mutation / 新快照并收敛内存中的 mutation timeline，重启后的 `reconstruct_latest()` 不能把“已经回滚”作为权威状态；
- 未知 node 类型和反序列化失败会被跳过，可能生成部分快照而不报错。

### 2.4 Hyperspace：WAL 细粒度正确，快照韧性还可提升

`hyperspace-engine` 用 WAL first、CRC、回放截断、快照临时文件 + `fsync` + rename、写屏障和共同逻辑时钟协调 checkpoint。它是本项目最成熟的恢复实现之一，快照 + WAL 的粒度不粗：全量 snapshot 用于缩短恢复时间，WAL 用于保留每个 mutation。

仍建议补强：

- snapshot 使用单个 `index.snapshot`，没有 format/schema version、校验摘要或兼容性校验（dimension、metric、配置）；
- checkpoint 成功后会清理 frozen WAL。若唯一 snapshot 发生静默损坏、二进制格式不兼容或被错误替换，已没有上一代完整副本可回退；
- 加载 snapshot 时应校验维度、metric 和存储格式，错误应明确降级到上一代可验证 snapshot；
- 需补 fault-injection 测试：snapshot 损坏、两代 snapshot 切换、旧版本格式、checkpoint 任意阶段掉电。

## 3. LLM / 工具追踪现状

### 3.1 已具备的有效信息

- Gateway 成功调用会记录 model 和 usage；失败会记录状态、是否可重试以及解析失败时的响应长度。`UnifiedGateway` 已避免把完整请求正文写入通用日志，这一隐私方向正确。
- AgentRunner 会记录 turn、消息数、工具数、解析后的 action、工具名、参数前缀和工具结果长度；TUI 可实时显示 thought、工具调用和工具结果。
- `ExecutionEvent` 类型已经覆盖 `LlmContent`、`ToolCall`、`ToolResult`、`TokenUsage`、错误和完成事件，模型层面具备扩展位置。
- checkpoint 的消息历史间接保留了部分请求上下文、模型回答和工具消息。

### 3.2 为什么当前不足以调试

1. **没有一次 LLM 调用的统一身份。** Gateway 返回的 response id 没有作为 trace id / request id 向上传播；任务、cycle、agent、turn、provider response、tool call 之间没有稳定关联键。
2. **请求不可复盘。** 实际发送给 Gateway 的 `request_messages` 会经过压缩、恢复、动态 guard 注入、阶段性提示和动态工具窗口，但该最终请求没有摘要、hash、大小、tool schema hash 或可控 payload 捕获记录。
3. **响应不可完整归因。** Gateway 没有记录耗时、端点类型、retry attempt、cache hit、finish reason、provider request id；AgentRunner 仅记录内容长度和解析后的结果。
4. **工具链只覆盖一部分运行期事件。** `ToolCall` 在执行前、`ToolResult` 在执行后发出，但两者没有 LLM request id；正常工具的 `duration_ms` 固定写为 0，即使实际执行时间已由 `Instant` 测得。
5. **定义与实际事件路径脱节。** `ExecutionEventEmitter` 定义了 LLM content 和 TokenUsage 发射方法，但 AgentRunner 实际没有调用它们。实际 `THOUGHT` / `TOOL_CALL` / `TOOL_RESULT` 使用 EventBus 的自定义字符串；gRPC 的 `convert_event_bus_to_grpc()` 不识别这些 Custom event，因此会丢弃它们。`include_thought` / `include_tool_calls` 仅影响单独的 emitter，不约束 AgentRunner 手工发出的事件，配置语义不成立。
6. **EventBus 不是 durable trace store。** 它是 broadcast 加进程内有界 history；订阅者 lag 会丢失，进程重启后无历史。它适合实时 UI，不适合故障复盘。
7. **敏感信息治理不一致。** 通用 Gateway 日志避免记录完整请求是正确的，但 EventBus payload、checkpoint 消息、tool arguments/result、thought 与 `reasoning_content` 可保留原文；已有 `sanitize_sensitive_fields()` 工具函数未接到实际输出链路。不能用“日志未打印 body”来推断数据未持久化。

结论：当前可辅助观察“系统是否还在工作”，但无法可靠回答一次异常调用“最终发给模型什么、供应商返回什么、模型为何选择工具、工具是否真的执行、恢复是否会重复执行”。

## 4. 建议目标架构

### 4.1 统一的 Task Execution Journal

为每项任务生成 `trace_id`，每个 agent turn 生成 `turn_id`，每次 LLM 请求生成 `llm_request_id`。所有记录统一携带：

```text
trace_id / task_iri / cycle_id / agent_id / role / turn_id / llm_request_id / timestamp / sequence
```

将以下事件追加到 durable journal（可按任务分段、按 sequence 排序）：

```text
CheckpointCommitted
LlmRequestPrepared -> LlmAttemptStarted -> LlmAttemptFinished
LlmResponseCompleted
ToolCallProposed -> ToolExecutionStarted -> ToolExecutionFinished
WorkspaceMutationCommitted
PhaseCommitted / TaskCompleted
```

写入顺序要表达外部副作用边界：先 durable 记录 `ToolExecutionStarted` 和 idempotency key；工具成功且工作区变更已观察到后记录 `ToolExecutionFinished` / `WorkspaceMutationCommitted`；之后再把结果加入会话并提交 checkpoint。恢复时：

- finished 的幂等工具可直接复用结果；
- started 但未 finished 的非幂等工具进入“需要核验”状态，禁止盲目重放；
- 可重试工具按显式 retry/idempotency policy 执行；
- checkpoint 只作为 journal 的压缩锚点，不再是唯一事实来源。

### 4.2 两层 LLM 可观测性

**默认元数据层（始终开启、低风险）**：不保存原文，只保存 model、endpoint kind、provider response id、attempt、cache hit、消息数/字节数/token estimate、tool 名单与 schema hash、finish reason、usage、TTFT/总耗时、错误分类、HTTP status、tool call ID/名称/耗时/结果 hash。

**受控 payload 层（调试时显式开启）**：保存最终请求消息、完整响应、工具参数与结果，但必须具备：

- 独立 trace store，不写入普通 `tracing` 日志；
- schema-aware secret redaction、路径/内容规则和不可绕过的 API key/header 脱敏；
- 单项与单任务大小上限、采样、TTL、加密/权限隔离；
- payload hash、redaction version、截断标记和保留原因；
- 导出前再次脱敏，TUI 仅按权限显示可展开内容。

流式调用不应逐 token 持久化全部 delta；记录 TTFT、chunk 数、完成状态和最终聚合内容即可，必要时才按受控模式保留 sampled deltas。

### 4.3 恢复粒度设计

| 层次 | 建议单位 | 保存内容 | 恢复用途 |
|---|---|---|---|
| 里程碑 checkpoint | phase start/end、审批、任务完成 | 紧凑 `TaskRuntimeState` + journal cursor + 会话摘要/引用 | 快速定位恢复起点 |
| turn journal | 单次 LLM 请求/响应 | 请求与响应的元数据，必要时 payload 引用 | 精确诊断与继续推理 |
| tool journal | 单次工具调用 | 调用 ID、参数 hash/受控 payload、耗时、结果 hash、幂等性、effect 证据 | 防止重复副作用 |
| workspace manifest | 每个可变更 step 前后 | 内容寻址 blob、路径、mode、hash、根身份、变更集 | 精确 diff / 手动或受控回滚 |
| 长期图/索引 snapshot | 图 mutation 批次、向量 checkpoint | 当前 snapshot + WAL/增量链 + 校验 | 快速重启与灾难恢复 |

建议保留策略：所有 phase milestone 保留；turn journal 按 TTL/容量分层；完整 payload 采用短 TTL；工作区快照保留任务基线、每个成功 mutation step 和任务完成点；不要仅保留“最近 N 个”。

## 5. 实施计划

### P0：先让恢复与事件语义真实可信

1. **冻结工作区自动 rollback 承诺并修复 SnapshotManager。** 将 manifest DB 改为持久路径；ContentStore 增加按 hash 的真实读取接口；创建 snapshot 时强制 materialize 并校验 blob；修正创建/删除计数；加入 dry-run、冲突检测、恢复前备份与事务日志。未完成前不允许自动调用 `rollback_to()`。
2. **统一 TaskResumeState。** 以有版本的结构体替代 JSON 字符串拼接；让 CLI/TUI 使用同一 `restore_task()`；恢复 `role/cycle/completed steps/pending approvals/counters/journal cursor`，并在兼容失败时回退前一有效 checkpoint。
3. **建立 Task Execution Journal。** 先接入 LLM 请求开始/完成、tool start/finish、checkpoint committed 四类事件；把现有 EventBus 仅保留为 journal 的实时投影。
4. **修正 ExecutionEvent 管线。** 明确注册事件类型或让 converter 解析标准 envelope；把 include_thought/include_tool_calls 传给 AgentRunner；真实填充 tool duration；接入 LLM content/usage 或移除未接线接口，避免虚假 API。
5. **安全默认值。** trace 默认仅元数据；payload capture 默认关闭；统一接入脱敏器并将 checkpoint/trace store 的权限和 TTL 写入配置。

验收：模拟 crash 分别发生在 LLM 前后、工具开始/完成前后、checkpoint 提交前后；恢复后无重复非幂等副作用；同一任务在 CLI/TUI 得到相同恢复状态；关闭 detail 选项时 gRPC/TUI 不泄漏 thought/tool payload。

### P1：图时间线一致性与调试体验

1. Timeline 的 `record mutation + persist + full snapshot + compact` 使用共同提交屏障，所有持久化失败向上传递 `Result`；snapshot 成功后才删除已覆盖 mutation。
2. rollback 后提交一条 durable `RollbackApplied` 事件或创建新的 full snapshot，使重启状态与当前 backend 一致。
3. 提供 `trace list/show/export`：默认显示时间线、耗时、模型、工具、状态和 hash；仅经显式授权显示脱敏 payload。
4. TUI 展示以 `turn_id` 为层级：请求摘要 -> 响应 -> 工具调用 -> 结果 -> checkpoint，支持查看截断、脱敏和恢复决策。

验收：并发 mutation + snapshot 的压力测试、持久化失败注入、rollback 后重启；一次失败 LLM 调用可以在不看普通日志的情况下定位 request id、最终工具窗口、重试和工具结果。

### P2：Hyperspace 与容量治理

1. Snapshot 加 `format_version`、content hash、dimension/metric/config compatibility 校验。
2. 保留至少“当前 + 上一有效代” snapshot；写新代、校验、原子切换 manifest 后再回收旧代。
3. 注入文件损坏、旧格式、任意 checkpoint 阶段掉电的恢复测试。
4. 为 journal、payload、workspace blob、Timeline mutation 设置配额、压缩、TTL、告警和可观测 metrics。

## 6. 建议测试矩阵

| 场景 | 必须证明的结果 |
|---|---|
| 工具已写文件，进程在结果入会话前退出 | 恢复识别在途调用；不自动重复非幂等写入；能核验工作区 effect |
| checkpoint JSON 损坏 | 自动选择上一份校验通过的 checkpoint，并记录恢复降级原因 |
| TUI 与 CLI 恢复同一 task | role、cycle、批准、计数、已完成 step、下一个操作完全一致 |
| workspace snapshot 后新增/删除/修改文件 | dry-run 给出准确 diff；执行后内容、路径、统计均与 manifest 一致 |
| LLM 429 后重试成功 | 单一 trace 关联所有 attempts、Retry-After、provider response、usage 与总耗时 |
| 工具被拒绝、未广告工具、真正执行失败 | 三种状态可区分；call_id、错误分类、duration、结果 hash 完整 |
| `include_tool_calls=false` | gRPC、TUI 投影和持久 trace 均不输出工具 payload，仅保留授权的元数据 |
| Hyperspace 当前 snapshot 损坏 | 从上一有效代 + WAL 恢复；不能静默以空索引启动 |

## 7. 不建议的做法

- 不建议把完整 prompt、response、tool output 无条件写入普通应用日志：成本高、噪声大且容易泄露源码和凭据。
- 不建议只把 checkpoint 间隔从 5 turn 改成 1 turn：会增加写放大，仍不能解决外部副作用重放。
- 不建议将 EventBus 当作 durable audit log：它适合实时投影，天然可能 lag/丢失且进程重启后为空。
- 不建议在修正历史内容寻址前开放工作区自动 rollback。

## 8. 最终判断

Gliding Horse 已有多种 snapshot 原语，尤其 Hyperspace 的 WAL + 快照设计可靠性基础较好，SkillGraph Timeline 的分块模型也正确。问题集中在任务/工作区恢复的“事实来源”尚未统一，以及调试信息散落在通用日志、会话消息和进程内 EventBus 中。

下一阶段应优先建设可恢复的执行 journal 和受控 LLM trace，而不是继续增加更多独立 snapshot 类型。完成 P0 后，系统才能同时获得：可解释的 LLM 调试、可证明的工具副作用边界，以及跨入口一致的任务恢复。

## 9. 实施与复验结果（2026-08-31）

本节是第 1～6 节方案的实施状态，范围仍限 `glidinghorse`、`hyperspace-engine` 和 `apps/gliding_code`。

| 项目 | 状态 | 已实施结果 |
|---|---|---|
| P0-1 工作区快照 | 完成 | manifest 与内容 blob 持久化；快照强制物化内容并按 hash 恢复；支持 dry-run、冲突检查、恢复前安全快照和可选删除目标快照外文件。 |
| P0-2 任务恢复 | 完成 | 引入版本化 `TaskResumeState`/`RestoredTask`；CLI 与 TUI 均经 `restore_task()` 恢复；损坏或未知新版本 checkpoint 自动跳过并回退至前一有效记录。 |
| P0-3 执行 Journal | 完成 | 新增按任务持久化、顺序化的 LLM 请求/响应、工具开始/结束、checkpoint 与工作区变更事件；默认只写摘要、长度和 hash，payload capture 必须显式开启。 |
| P0-4 实时投影 | 完成 | EventBus 自定义执行事件被 gRPC 标准投影识别；thought/tool 过滤生效；LLM content、usage 与真实工具耗时均已发射。 |
| P1 Timeline | 完成 | mutation、持久化、快照、回滚共用提交屏障；持久化失败向上传播；rollback 写入收敛快照；并发 mutation 保持唯一、连续的 durable sequence。 |
| P2 Hyperspace | 完成 | 快照改为带 magic、版本、metric、payload 长度与 SHA-256 的 envelope；保留 current + previous 两代，并保留上一代所需 frozen WAL。 |

### 9.1 最新确认：不做旧快照兼容

按确认，**不兼容的快照不迁移、不降级读取**。引擎打开时按 `current -> previous -> WAL` 处理：

1. 旧 bincode 格式、未知版本、截断、长度或校验和错误的快照会删除；
2. 维度、度量、节点向量或邻接关系不符合当前引擎的快照也会删除；
3. 删除 current 后尝试 previous；没有有效快照时只回放 WAL，不会因不兼容快照阻断启动；
4. checkpoint 只清理由上一有效代已覆盖的 frozen WAL，因此 current 损坏时仍可由 previous + retained WAL 恢复最新状态。

这比保留隐式兼容分支更容易审计：磁盘上的 snapshot 只有当前 envelope 这一种可接受格式。

### 9.2 本轮验证

| 命令 | 结果 |
|---|---|
| `cargo test -p hyperspace-engine --offline` | 91 个单元测试、15 个集成测试通过；覆盖 checksum、旧格式删除、metric 不匹配删除、current 损坏后的 previous + WAL 恢复。 |
| `cargo check -p code_cli --locked` | 通过。 |
| `cargo test -p glidinghorse --lib --locked` | 1,544 个测试通过。两项回环 HTTP 测试在受限沙箱中无法 bind 临时端口；在允许回环绑定的复验环境中全部通过。 |

现阶段的 Journal 已提供可靠的默认元数据追踪和受控 payload 开关。面向运维的 trace 查询/导出界面、payload 的长期保留策略和权限管理，仍应作为独立产品接口按部署要求配置，不能以普通应用日志替代。
