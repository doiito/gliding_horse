# Gliding Horse 核心与 Gliding Code 专项审计报告

> 审计日期：2026-08-31  
> 报告版本：P1/P2 实施复验版  
> 代码基线：`main@0c559fb63e04`  
> 审计方式：源码走查、模块依赖追踪、离线全量测试、格式/命名空间/静态质量检查  
> 审计范围：根 `glidinghorse`、`hyperspace-engine`、`ontologies`、`apps/gliding_code`，以及直接影响这些模块的配置、文档、测试和 CI。其他应用层项目及其问题不在本报告范围内。

## 0. P1/P2 实施结果（当前状态）

本报告第 5、6 节保留了实施前的原始证据、风险解释和建议，便于审计追溯；各问题的当前状态以本节为准。`software_engineering_single` 与 `software_engineering_team` 未纳入实施或验收。

所有 P1、P2 项均按“确认—设计—开发—测试”完成推进；其中 P2-3 采用保持公开 API 不变的渐进式模块化，已先抽离 TUI Markdown/样式桥接这一独立高风险职责，未在安全加固提交中对其余大型状态机做高风险一次性搬迁。

| 项目 | 当前状态 | 实施结果 |
|---|---|---|
| P1-1 | 完成 | 删除跨 crate `unsafe transmute`，逐字段转换颜色/Modifier，并抽到 `tui/markdown.rs`；3 个 TUI 测试通过 |
| P1-2 | 完成 | `archive_to_l0` 在 `await` 前释放同步读锁；并发写锁与 `Send` 回归测试通过；周边审批/测试锁作用域同步修复 |
| P1-3 | 完成 | CLI 凭据不再写入进程环境；Shell/MCP `env_clear` 后仅传安全白名单或显式 MCP 环境；泄漏回归测试通过 |
| P1-4 | 完成 | `web_fetch` 限 HTTP(S)，逐跳 DNS/IP 校验并固定解析地址，拒绝本机/私网/链路本地/保留地址；10 MB 流式硬上限测试通过 |
| P1-5 | 完成 | Shell/PowerShell 改为 Tokio 子进程和有界并发读取；超时不重放，进程组异步终止，后台进程统一回收；16 项测试通过 |
| P1-6 | 完成 | MCP 全请求超时、JSON-RPC ID 匹配、notification 跳过、HTTP headers、非 2xx、kill/wait/关闭路径均已接线；相关测试通过 |
| P1-7 | 完成 | Worker 使用 durable inflight claim、结果按 task 隔离、启动恢复、真实并发隔离执行；持久任务不再序列化 API key；审批按 request ID 分发 |
| P1-8 | 完成 | 新增 Core Quality PR 工作流：fmt、版本一致性、关键异步 Clippy、workspace 全测和 Gliding Code smoke |
| P1-9 | 完成 | HTTP/gRPC 默认回环；远程监听必须声明外部 TLS 终止并配置 token；HTTP 非健康路由和全部 gRPC Bearer 鉴权；假状态改 501/NOT_FOUND，假成功 RPC 改 UNIMPLEMENTED |
| P2-1 | 完成 | 并发测试由墙钟阈值改为 Barrier 契约；真实供应商测试严格置于 `live-tests`；确认所谓慢文档测试实际为冷编译，CI 用缓存处理 |
| P2-2 | 完成 | 新增 ontology crate 直接契约测试：语法错误、diff、SHACL minCount、RDFS materialization，共 4 项 |
| P2-3 | 完成（渐进式） | 抽离 TUI Markdown 解析和跨版本样式桥接，保持调用 API 不变；其余大状态机保留原位以避免把纯结构搬迁混入安全修复 |
| P2-4 | 完成 | Gateway 限流进入主请求链；响应缓存为显式 opt-in 且工具请求禁用；custom approval 使用确定性 DSL；文档明确可选沙箱和哈希嵌入降级语义 |
| P2-5 | 完成 | `[workspace.package].version` 成为唯一版本源，所有 crate 继承；CLI `--version`、README badge 和 CI 脚本一致 |
| P2-6 | 完成 | EventBus 改为动态 bitset，130 种类型回归测试通过；订阅数直接取 Tokio receiver 实际数量，drop 后归零 |
| P2-7 | 完成 | 408/429/5xx 分类重试、`Retry-After`、有界指数退避+jitter；成功响应读取/解析失败不重放；请求正文不再进入日志/错误；Gateway Debug 密钥脱敏 |
| P2-8 | 完成 | TUI 有界事件通道 256、每帧 128 条；listener 显式 abort+join；UI 500 条/2 MiB、恢复历史 200 条/1 MiB并做 UTF-8 安全截断 |
| P2-9 | 完成 | 查询按 UTF-8 bytes 做 form 编码；allowed/blocked domain 按 hostname 精确/子域匹配，并统一应用 Exa 与 DDG 路径 |

### 0.1 实施后验证证据

| 检查 | 结果 |
|---|---|
| `cargo fmt --all -- --check` | 通过 |
| `cargo clippy --workspace --all-targets --locked -- -A warnings -D clippy::await_holding_lock -D clippy::future_not_send` | 通过 |
| `cargo test --workspace --locked --no-fail-fast` | 全部通过，0 失败 |
| Gliding Code | 库 12、二进制 5，全部通过 |
| Gliding Horse 核心库 | 1,535 项（含最后新增的 API token 脱敏测试），全部通过 |
| 核心集成套件 | 97；MCP 11、技能演化 1、模板 19、验证门 9，全部通过 |
| Hyperspace Engine | 单元 91、集成 15，全部通过；快照 envelope、双代恢复、旧/不兼容快照删除与 WAL 回放已复验 |
| Ontologies | 新增直接契约 4 项，全部通过 |
| 文档测试 | Gliding Horse 8 项通过；Hyperspace/ontologies 无文档测试 |
| `bash scripts/check_namespace.sh` | 通过 |
| `bash scripts/check_version_consistency.sh` | 通过，workspace `0.1.0` |
| `glidingcode --version` | `glidingcode 0.1.0` |
| `git diff --check` | 通过 |

## 1. 结论摘要

### 1.1 总体结论

**核心 Gliding Horse 与 Gliding Code 的设计方向合理，主要执行链路完整；本轮 P1/P2 实施后，离线全量测试和关键异步静态门禁全部通过。**

审计确认的子进程密钥泄漏、`web_fetch` SSRF/无界读取、Shell 阻塞与重放、MCP 生命周期、daemon 提前确认、EventBus 类型上限、Gateway 隐私重试和 TUI 生命周期等问题均已修复并加入回归测试。真实供应商兼容性、外部 TLS 反向代理配置和不同平台 namespace 能力仍属于部署验收边界，不由离线测试替代。

结论需按运行形态区分：

| 运行形态 | 结论 | 说明 |
|---|---|---|
| Gliding Code 本地单次命令 | 通过 | 进程内组装核心；子进程环境、网络访问和 Shell/MCP 生命周期已加固 |
| Gliding Code 交互式 TUI | 通过 | 安全样式转换、有界通道、监听取消和历史预算均已实现并测试 |
| Gliding Code daemon / Worker | 有条件通过 | durable claim、恢复、隔离结果与真并发已实现；生产部署仍需按业务定义重复交付幂等语义 |
| 核心 Rust 库 | 通过 | 主要模块实现充分，测试覆盖很强；已修复并验证同步锁跨 `await` 问题 |
| 根网络服务对外部署 | 有条件通过 | 默认仅回环；远程监听必须在可信 TLS 反向代理后显式启用并配置 Bearer token；未实现接口明确报错 |

### 1.2 原始审计验证结果

以下表格保留实施前审计记录；实施后的权威结果见 0.1。

| 检查项 | 结果 |
|---|---|
| `cargo fmt --all -- --check` | 通过 |
| `bash scripts/check_namespace.sh` | 通过 |
| `cargo test --workspace --no-fail-fast --offline` | 通过，0 失败 |
| Gliding Code 测试 | 14 通过（库 9、二进制 5） |
| Gliding Horse 核心库测试 | 1,505 通过（含新增锁释放并发回归测试） |
| 核心集成测试 | 97 通过；MCP 11、技能演化 1、模板 19、验证门 9 均通过 |
| Hyperspace Engine | 单元 85、集成 15，全部通过 |
| Rust 文档测试 | 8 通过；本次复测约 1.25 秒，曾出现约 77 秒的环境波动 |
| 严格全量 Clippy | 仍会被大量存量样式/测试告警阻断；`await_holding_lock` 专项门禁已对全部 glidinghorse targets 通过 |
| 定向 Clippy 深入检查 | 核心和 Gliding Code 的 `future_not_send` 通过；Gliding Code `large_futures` 通过；核心生产 lib `unwrap_used`/`expect_used` 通过 |

全量测试不访问外部模型服务，因此只证明本地确定性逻辑和模拟链路正常；真实模型供应商、网络故障恢复、远端 MCP 兼容性仍需要带凭据的单独验收。

## 2. 项目内容与目的确认

项目的实际目标与 README 描述基本一致：它不是单一聊天程序，而是一套 Rust Agent OS 内核，并提供 Gliding Code 作为面向代码工作区的终端产品入口。

核心能力包括：

- 以 PA/DA/CA 角色和 PDCA 周期组织复杂任务；
- 以 5W2H 元数据进行任务分级、审计、回滚和干预；
- 提供 L0-L3 分层记忆、黑板、RDF/Oxigraph 语义图和投影缓存；
- 使用 Hyperspace/HNSW 支持向量与结构化混合检索；
- 维护技能图、因果模型、演化建议、审批与恢复记录；
- 提供文件读写、Shell、MCP、工作区监控、检查点和事件总线；
- 由 Gliding Code 将这些能力组合成单次命令、交互式 TUI、检查点恢复和后台 worker 入口。

Gliding Code 的主要真实调用链为：

```text
CLI / TUI
  -> CodeCliEngine
  -> SupervisorAgent
  -> AgentRunner + ToolExecutor
  -> Gateway / Memory / SkillGraph / Hyperspace / WorkspaceMonitor
  -> 结果审计、事件记录、检查点与演化记录
```

这一链路在进程内直接构造核心组件。根 HTTP/gRPC 服务是另一种部署适配器，并非 Gliding Code 本地模式的前置依赖。这一点使本地产品路径与服务端尚未完成的兼容接口实现保持了解耦，整体边界是合理的。

## 3. 模块实现评估

| 模块 | 完整度 | 合理性 | 审计判断 |
|---|---:|---:|---|
| PDCA / PA-DA-CA / 5W2H | 高 | 高 | 状态流转、干预、审计、恢复和证据链均有大量测试，属于项目最成熟部分 |
| Agent Runner / Supervisor Agent | 高 | 高 | 计划、执行、工具调用、失败识别、上下文注入与检查点已形成闭环 |
| L0-L3 记忆与一致性 | 中高 | 高 | 分层职责清楚、降级策略完整；同步读锁跨异步等待已修复并有并发回归测试 |
| Blackboard / RDF / Projection | 高 | 高 | 命名图隔离、投影、SPARQL、消息包和一致性均有集成测试 |
| Hyperspace Engine | 高 | 高 | HNSW、WAL、过滤、混合检索和恢复测试充分，可独立使用 |
| Skill Graph / 因果 / 演化 | 中高 | 高 | 功能广、审批和持久化链路较完整；文件过大，维护成本偏高 |
| Tool Executor / Workspace Monitor | 中高 | 中高 | 工作区扫描、工具路由、结果压缩、沙箱能力均已实现；不应默认视为强隔离环境 |
| Unified Gateway | 中高 | 中高 | 请求构造、SSE 解析、重试和兼容逻辑有测试；真实供应商链路未在本轮验证 |
| `ontologies` crate | 中 | 中高 | SHACL、推理和差异接口存在，但该 crate 自身为 0 个直接测试，应补独立契约测试 |
| Gliding Code Engine | 中高 | 高 | 初始化、持久化隔离、任务分类、扫描、执行、恢复和学习记录衔接完整 |
| Gliding Code TUI | 中高 | 中 | 交互能力较完整；一处跨 crate 类型布局假设不满足 Rust 安全保证 |
| 根 HTTP/gRPC 适配器 | 中 | 中 | 可用于内网开发联调，但安全边界和若干兼容接口还未达到公网生产要求 |

## 4. 已确认合理、完整的实现

### 4.1 Gliding Code 初始化链路

`CodeCliEngine` 会解析并规范化工作区，构造统一 Gateway、临时或持久 L0、共享 Oxigraph、Blackboard、Hyperspace、Ontology Bridge、Projection、Memory Manager、模板、技能注册表、时间线和演化恢复组件。关键存储创建失败会显式返回错误；技能水合或向量服务异常则记录告警并降级，不会造成无提示崩溃。

持久化目录按规范化工作区路径做哈希命名空间隔离，可以避免多个仓库共享同一数据根时相互污染。未找到持久目录时使用由结构体持有的 `TempDir`，生命周期管理正确。

需要注意，初始化时调用 `std::env::set_current_dir` 修改进程级工作目录。这对当前单工作区 CLI 是合理的，但如果以后把 Engine 嵌入同一进程的多租户服务，就必须改为所有工具显式携带 workspace root。

### 4.2 任务执行闭环

任务开始前会进行嵌入健康探测、技能索引、Workspace Monitor 初次扫描及任务类型判断；代码工作区任务注入文件清单和代码知识，研究型任务避免无意义扫描。随后构造 `TaskContext`，由 Supervisor Agent 驱动 PDCA 和角色协作，结束后写入事件、效果证据与演化记录。

恢复、检查点、学习模式、演化提案审批/验证/提交均有独立入口。对于预览版本而言，主执行闭环是完整的，而不是只有接口或静态演示。

### 4.3 核心认知与记忆系统

项目对失败、回滚和长期状态的处理明显强于普通 Agent CLI：

- 5W2H 维度审计可以把 What/Why/How/Where/When/Who/HowMuch 的失败映射到不同恢复动作；
- 检查点和事件总线支持任务恢复、阶段追踪和效果证据记录；
- L0-L3、Blackboard、Projection 与预取组件有大量正常、降级和一致性测试；
- Skill Graph 不仅保存技能，还覆盖发现、组合、依赖、因果、版本、演化建议和审批；
- Hyperspace 具备 WAL、HNSW、过滤和恢复测试，并非对外部向量库的空包装。

### 4.4 降级与可观测性

嵌入不可用时可退化到哈希嵌入或非语义路径，Gliding Code 会保留降级状态。日志被接入 TUI 面板，单次命令模式则镜像到标准错误，避免长任务看似“卡死”。终端启动前也做了上次异常退出后的 raw mode/alternate screen 恢复，产品化考虑合理。

## 5. 原始 P1 发现与建议（当前状态见第 0 节）

### P1-1：TUI 使用 `unsafe transmute` 转换两个不同 crate 的 `Style`

位置：`apps/gliding_code/src/tui.rs` 的 `markdown_to_owned_lines`。

当前实现假设 `ratatui_core::Style` 与 `ratatui::Style` 字段和内存布局一致，再通过 `std::mem::transmute` 转换。即使当前依赖版本和 feature 组合下测试通过，Rust 也没有为两个不同类型提供稳定 ABI/布局承诺；依赖升级可能引入未定义行为。

建议：统一 `ratatui`/`ratatui-core` 依赖版本与类型来源，或逐字段安全映射颜色、背景和 modifier。修复后增加 Markdown 多样式渲染测试，并将该处从发布阻断清单移除。

### P1-2（已完成）：内存调度器持有同步读锁跨越 `await`

位置：`src/memory/scheduler.rs` 的 `archive_to_l0`。

原实现取得 `self.sessions.read()` 后，通过借用的 session 完成归档，随后在读锁仍存活时调用 `self.consistency.on_l0_update(&iri).await`。问题属实；不过当前 EventBus 发送 Future 恰好不会挂起，因此旧实现不一定在运行时稳定死锁，其确定性影响是归档 Future 非 `Send`，并为以后会挂起的一致性通知留下写锁阻塞风险。

已完成以下优化：

- 将同步 session 访问限定在独立作用域，确保 guard 在一致性通知 `await` 前释放；
- 增加可控暂停的一致性更新测试：通知 Future 挂起期间，另一线程必须能取得 `sessions` 写锁并移除 session；
- 测试通过 `tokio::spawn` 同时约束归档 Future 为 `Send`；
- 扫描并修复同类实现：`ChannelApprovalNotifier::submit_response` 在异步通道发送前释放 `pending` 写锁；Agent Runner 测试中的临时读 guard 也在 `await` 前释放；
- `cargo clippy -p glidinghorse --all-targets --offline -- -A warnings -D clippy::await_holding_lock` 已通过。

### P1-3：Bash 与 stdio MCP 子进程继承完整父进程环境，可泄露模型密钥

位置：`apps/gliding_code/src/main.rs`、`src/tools/tool_executor/builtins.rs`、`src/tools/mcp_client.rs`。

Gliding Code 会将 CLI `--api-key` 写入 `DEEPSEEK_API_KEY` 进程环境。Bash 工具未调用 `env_clear` 或移除敏感变量；stdio MCP 更是显式执行 `cmd.envs(std::env::vars())`。因此 Agent 执行的 Shell 命令或第三方 MCP 进程可直接读取模型 API key，以及同进程中的 GitHub、AWS 等其他令牌。Shell namespace 沙箱也不会自动清理环境变量。

建议：CLI key 直接传入 Gateway 配置，不写入进程环境；Shell 只继承 `PATH`/locale/终端等最小安全白名单；MCP 仅获得其配置显式声明的变量。增加回归测试，确认密钥不会出现在 Shell/MCP 环境。

### P1-4：`web_fetch` 存在 SSRF 和分块响应无界内存风险

位置：`src/tools/tool_executor/builtins.rs` 的 `execute_web_fetch`。

当前接受任意 URL，没有限制 scheme，也没有拒绝 loopback、RFC1918、link-local 或云元数据地址；跳转仅限次数，不会重新校验目标。这使 Agent 可被外部内容诱导访问本机/内网服务。另外，代码只在读取前检查 `Content-Length`，对无长度的 chunked 响应会先用 `resp.bytes().await` 把整个 body 读入内存，超过 10 MB 后才拒绝，实际上没有读取期间硬上限。

建议：仅允许 HTTP/HTTPS，对初始 URL、DNS 解析结果及每次跳转都拒绝私有/本机/链路本地地址，除非用户显式授权；使用流式读取累计字节，到达上限立即中断。补私有 IP、DNS/跳转到私有 IP、无 `Content-Length` 超限响应测试。

### P1-5：Shell 执行器阻塞 Tokio，输出无界，且自动重放超时命令

位置：`src/tools/tool_executor/builtins.rs` 的 `execute_bash`/对应 PowerShell 路径。

- `async fn` 内使用 `std::process::Command`、`try_wait` 轮询和 `std::thread::sleep`，长命令会占用 Tokio worker 线程；
- stdout/stderr 读取线程先 `read_to_string` 到无界 `String`，16 KB 截断在进程结束后才发生，大量输出仍可导致 OOM；Windows 路径不及时排空 pipe，还可因 pipe 写满误判超时；
- 首次超时后会无条件重新执行同一条命令，可重复数据迁移、发布、写文件等非幂等副作用；
- 后台命令返回 PID 后丢弃 `Child`，缺少统一的状态、回收和终止生命周期。

建议：改为 `tokio::process` + `tokio::time::timeout`，或将整个同步流程放入 `spawn_blocking`；读取时就使用有界 ring buffer/临时文件；默认不重放超时命令，仅对明确幂等操作开放重试；使用可管理的后台子进程注册表。

### P1-6：stdio/HTTP MCP 的配置契约与生命周期未完整实现

位置：`src/tools/mcp_client.rs`、`src/config/runtime.rs`、`apps/gliding_code/src/engine.rs`。

- `McpStdioServerConfig.tool_call_timeout_ms` 已被解析，但 stdio 的连接与 `tools/call` 都直接等待 `read_line`，配置未进入调用链，静默的 MCP 可无限期卡住任务；
- `send_request` 假定下一行就是当前请求响应，不校验 response `id`，服务端 notification 或异步响应会导致误配；
- HTTP MCP 配置有 `headers`，但 `register_from_config` 只传 URL，认证 header 被丢弃；
- `McpClient` 提供 `kill_all_processes`，注释声称引擎会显式调用，但 Gliding Code 只找到连接/注册路径，未找到关闭调用；`Command` 也未设 `kill_on_drop(true)`，注释中“drop 会发 SIGKILL”与 Tokio 默认行为不符。

建议：为连接、读写和工具调用统一应用可配置超时，超时后终止/重建进程；按 JSON-RPC `id` 分派响应；保留 HTTP headers；在 Engine 关闭路径显式 kill + wait，并将 `kill_on_drop(true)` 作为异常退出兜底。

### P1-7：Daemon/Worker 队列的可靠性、并发和任务契约不完整

位置：`src/worker/task_queue.rs`、`src/worker/agent_os_worker.rs`、`src/tools/hooks.rs`、`apps/gliding_code/src/main.rs`。

- Worker 在 `recv_task` 刚反序列化后就 `commit`，执行期崩溃或结果发送失败会永久丢任务；
- `concurrency` 配置仅被打印，主循环仍是 `recv -> execute.await -> send`串行；CLI 帮助和注释声称 Unix socket，实际主路径使用 yaque 文件队列；
- `AgentOsTask.context` 中的 `project_dir`、前置输出和 LLM 配置未被 `execute_task` 使用，其中 API key 却可被明文序列化到持久队列；
- 多客户结果队列会消费并确认他人的结果，再放入本进程无界 `pending_results`，正确客户端可永久收不到结果；
- `ChannelApprovalNotifier` 所有审批共享一个只能被一个 waiter 取走的 receiver，未来开启并发后会误投、丢弃或提前默认决策；
- daemon 顶层只打印 `run_worker` 错误仍返回 `Ok(())`，监控系统会看到成功退出码。

建议：引入 claim/lease/ack 或 inflight 队列，在结果持久化后才确认任务，并支持超时回收；按 task/client 路由结果；并发路径使用 semaphore/`JoinSet` 且为每任务隔离执行状态；用每请求 oneshot 管理审批；真正应用经验证的 task context 或删除误导字段，密钥改为凭据引用；致命错误必须非零退出。

### P1-8：PR 缺少核心编译与测试门禁

当前 PR CI 只检查 ontology namespace；发布工作流负责多平台构建 Gliding Code，但普通 PR 没有自动执行 `cargo fmt`、核心测试或最小 Clippy。

建议至少新增：

1. `cargo fmt --all -- --check`；
2. `cargo test --workspace --no-fail-fast --locked`；
3. 对生产目标执行渐进式 Clippy 门禁；
4. Gliding Code `--help`、无效配置和单次命令模拟 Gateway 的 smoke test。

现有约 244 条 `-D warnings` 结果包含大量测试代码机械告警，不适合一次性全部设为阻断。可以先修生产代码风险，再用 Clippy baseline 或按 crate 分批收紧。

### P1-9：根网络服务的安全边界不适合直接公网部署

根服务默认 gRPC `0.0.0.0:50051`、HTTP `0.0.0.0:8080`，路由层未看到入站认证/TLS 中间件；HTTP 任务状态/详情仍含静态示例值，流式任务接口只订阅已有事件而不启动任务；部分兼容 RPC 固定返回成功或空对象。

这不影响 Gliding Code 本地执行路径，但如果核心服务需要对外部署，应先：默认回环地址、增加认证授权和 TLS、实现真实任务状态源、让占位接口返回明确的 `UNIMPLEMENTED`，并补跨协议端到端测试。

## 6. 原始 P2 发现与建议（当前状态见第 0 节）

### P2-1：测试稳定性与执行时间

首次审计时两个冷启动性能断言曾在约 1 秒阈值附近失败，本次复测通过，说明结果受机器负载、debug 构建或冷缓存影响。另有 8 个 Hyperspace 文档测试耗时约 77 秒，明显拖慢反馈。

建议将性能回归改为 release/benchmark 环境、增加预热和统计分位数；普通单测只验证正确性与宽松上界。文档示例可复用一次初始化、缩小数据集，或拆为普通集成测试。

### P2-2：`ontologies` 缺少 crate 级直接测试

根项目对 ontology bridge 和 namespace 有间接验证，但 `ontologies` crate 本身显示 0 个测试。SHACL 校验、推理规则、diff/merge 和错误输入应建立独立契约测试，避免根项目测试通过却遗漏 crate API 回归。

### P2-3：关键文件体积过大

当前大文件包括：

| 文件 | 行数 |
|---|---:|
| `src/skill_graph/evolution.rs` | 3,784 |
| `src/core/sa/execution.rs` | 3,754 |
| `src/core/agent_runner/execution.rs` | 3,499 |
| `apps/gliding_code/src/engine.rs` | 3,309 |
| `apps/gliding_code/src/tui.rs` | 2,923 |

功能本身能够运行，但审查、冲突处理和局部推理成本已经偏高。建议按“初始化/执行/恢复/演化记录”和“TUI 状态/渲染/输入/Markdown”拆分，并保持现有公开 API 不变。

### P2-4：存量未接线或未完成能力需要明确标注

- Gateway 导出了 `RateLimiter` 和 `ResponseCache`，但主 `UnifiedGateway` 调用链未见实际接线，不应在文档中暗示请求天然具备限流和缓存；
- `HumanApprovalHook` 的 custom condition 仍为 TODO；
- 沙箱/namespace 隔离是可选能力，且受平台权限影响，不应把默认工具执行视为处理不可信代码的强安全边界；
- 哈希嵌入是可用性降级方案，不是语义嵌入的等价替代，应在状态和评估中区分。

### P2-5：版本信息漂移

根 `Cargo.toml` 为 `0.1.0`，README 徽章和发布说明为 `0.1.4.preview`，CHANGELOG 顶部又以 `v0.1.6.preview` 为基线。建议建立单一版本源，并在发布 CI 中校验 Cargo、二进制 `--version`、README 和标签一致性。

### P2-6：EventBus 存在 64 类型崩溃上限和订阅者计数失真

`TypeMask` 用一个 `u64` 为事件类型分配 bit，第 65 个不同类型会直接 `panic!`。`emit` 会为动态事件名调用该分配器，因此长时间运行的服务或插件生成足够多 `Custom` 类型后可确定性崩溃。另外，`subscriber_count` 只在 subscribe 时增加，receiver drop 时不减，对外指标会单调虚高。

建议改用动态 bitset/字符串集合或可安全降级的 overflow bucket，任何用户可控事件名都不应触发 panic；对 receiver 包装 Drop 计数，或直接使用 broadcast 实际 receiver 数。

### P2-7：Gateway 重试策略会漏掉 408/429，并可在日志/错误中泄露请求内容

`UnifiedGateway::send_with_retry` 对所有 4xx 立即停止，因而 408 和 429 不会重试，也不识别 `Retry-After`；反之，HTTP 已成功但 body 读取/JSON 解析/转换失败时又会重放整个模型请求，可产生额外计费或重复副作用。指数退避没有 jitter/上限。4xx 时还会把最多 8,000 字符的请求预览同时写日志和错误对象，其中可包含用户提示、源码、工具输出和其他秘密。

建议使用按状态码分类的重试策略，支持 408/429 和 `Retry-After`，加 jitter/上限；对成功响应解析失败默认不重放，除非供应商支持幂等 key；请求正文诊断改为显式 opt-in 且先脱敏。

### P2-8：Gliding Code TUI 的事件监听与会话内存需要显式预算

每个任务启动一个 EventBus listener，任务结束时只 drop mpsc receiver；listener 若正阻塞在 broadcast `recv`，要到下一个事件发送失败才退出。状态通道是无界的，UI 每帧会一次排空；流式 delta 已有合并，但大工具 payload 仍可引起峰值。`status_events` 和 log 已有数量上限，但会话 `messages`/恢复历史未见字节或 token 上限，长会话会持有完整 payload。

建议保留 listener task handle/cancellation token，任务结束时 abort + join；使用有界通道与每帧排空预算；为会话按字节/token/条数设置上限，旧记录压缩或归档。

### P2-9：Web Search 对 Unicode 查询编码错误

自定义 `urlencode` 按 Rust `char` 遍历后将非 ASCII 字符直接 `as u8`，中文等查询会被截断为错误百分号编码。应换成基于 UTF-8 bytes 的标准 form/percent encoder，并增加中文、emoji 和混合查询测试。同时 allowed/blocked domain 应解析 hostname 精确匹配，并在所有搜索供应商路径一致应用。

## 7. 测试覆盖边界

本次测试证明：

- 核心数据结构、状态机、PDCA、5W2H、记忆、知识图、技能图和 Hyperspace 在本机离线环境下正常；
- Gliding Code 的配置、任务分类、引擎辅助逻辑和拓扑规则正常；
- 主要集成测试和文档示例能够完成。

本次测试未证明：

- 默认模型名在每个真实供应商账户上均可用；
- 网络超时、限流、断流和供应商协议变化下的完整恢复效果；
- 所有远端/stdio MCP 服务实现均兼容；
- TUI 在全部终端、窗口尺寸和 Unicode 组合下都无渲染问题；
- 多进程同时访问同一持久数据目录时具有完整事务隔离；
- 根网络服务已经满足公网安全要求。

因此，“正常”应理解为：**仓库内确定性核心与本地产品主路径正常；外部依赖和公网部署需要独立验收。**

## 8. Git 与正式文件管理

根 `.gitignore` 已覆盖：

- Rust 各层 `target`；
- 本地 Agent、分析器和沙箱状态；
- `.gliding_horse` 运行数据、日志、数据库；
- Python/Node 缓存、覆盖率、IDE 和系统元数据；
- 嵌套误生成的 `Cargo.lock`，同时保留根 lockfile；
- `PR-res` 中的临时材料，仅放行本正式审计报告。

仓库历史中仍有两个已跟踪的 Gliding Code 工作区监控运行文件：

- `apps/gliding_code/.gliding_horse/ws_monitor/content`
- `apps/gliding_code/.gliding_horse/ws_monitor/metadata`

`.gitignore` 只能阻止新文件被跟踪，不能自动移除历史已跟踪文件。建议在确认它们不是测试夹具后，用一次独立提交从 Git 索引移除；不要在本审计报告提交中混入无关源码、用户已有 CHANGELOG 修改或历史文件删除。

## 9. 建议执行顺序

原建议序列中的 P1/P2 工作均已按第 0 节完成。后续不再遗留代码级 P1/P2 阻断项，转入部署验收：真实供应商协议与配额、TLS 反向代理、平台 namespace 权限、daemon 重复交付时的业务幂等性，以及后续独立 PR 对其余大型状态机继续做低风险模块化。

## 10. 最终确认

在本轮限定范围内：

- **项目目的清晰，核心架构与产品入口匹配；**
- **核心模块实现总体合理、主链路完整；**
- **P1/P2 优化均已实施，关键安全、并发、生命周期和资源预算问题有回归测试；**
- **离线全量测试、关键 Clippy、格式、命名空间、版本和 CLI smoke 全部通过；**
- **Gliding Code 本地单进程、TUI 与 durable worker 主路径正常；**
- **没有发现需要重构整个系统的证据；**
- **外部模型、第三方 MCP、公网 TLS 代理和不同宿主隔离能力仍须在目标部署环境单独验收。**

综合判断：**核心与 Gliding Code 已完成本报告定义的 P1/P2 工程优化，可进入下一阶段真实环境验收。根网络服务默认安全，但非回环部署必须置于可信 TLS 终止层后并配置 Bearer token；不能把离线验证解释为第三方服务或宿主隔离的生产认证。**
