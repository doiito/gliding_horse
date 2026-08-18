# 3. 记忆系统

## 3.1 模块概览

记忆系统采用四层分层架构，从永久图记忆到按需投影，实现 Token 预算的精细控制。所有层通过统一 Oxigraph 存储共享底层 RDF 图数据，通过命名图实现隔离。

```mermaid
graph TB
    subgraph 记忆层次
        L0["L0 Store<br/>redb 永久图记忆<br/>标签二级索引"]
        L1["L1 Session<br/>Agent 短期会话<br/>Vec&lt;L1Turn&gt;"]
        L2["L2 Blackboard<br/>Oxigraph Store + DashMap<br/>共享黑板缓存"]
        L3["L3 Projection<br/>SPARQL CONSTRUCT<br/>按需投影引擎<br/>物化视图缓存"]
    end

    subgraph 管理组件
        MM["MemoryManager<br/>统一管理入口"]
        MB["MemoryBus<br/>内存事件总线"]
        CE["ConsistencyEngine<br/>MESI 一致性"]
        SCHED["MemoryScheduler<br/>缓存调度"]
        PF["PrefetchEngine<br/>预取引擎"]
        HE["HyperspaceEngine<br/>向量检索"]
    end

    subgraph 统一存储
        UGS["UnifiedGraphStore<br/>Arc&lt;Store&gt;<br/>系统唯一 Oxigraph 实例"]
    end

    L0 <-->|归档/检索| L1
    L1 <-->|读写| L2
    L2 <-->|投影| L3
    MM --> L0 & L1 & L2 & L3
    MB --> CE
    CE --> L0 & L1 & L2
    SCHED --> L1
    PF --> L0
    L3 --> HE
    L2 --> UGS
```

## 3.2 各层详细设计

### 3.2.1 L0 Store — 永久图记忆

**文件**: `src/memory/l0_store.rs`  
**实现状态**: ✅ 完整  
**存储引擎**: redb (Rust 原生嵌入式键值库，带标签二级索引和命名图索引)

L0 是系统的永久存储层，存储完整的 JSON-LD 图数据，支持实体对齐和 MESI 一致性状态。

**核心结构体**:

```rust
pub struct L0Entry {
    pub iri: String,
    pub content: String,
    pub importance: f32,
    pub access_count: u32,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub tags: Vec<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub mesi_state: MesiState,
    pub content_hash: String,
    pub named_graph: Option<String>,
    pub jsonld_context: Option<String>,    // JSON-LD @context
    pub jsonld_types: Vec<String>,         // 多类型列表
}
```

**索引结构**:
- **主索引**: IRI → bincode 序列化的 L0Entry
- **标签索引**: `tag:{tag_name}` → 该标签下的 IRI 集合
- **命名图索引**: `graph:{graph_name}` → 该图下的 IRI 集合

**核心方法**:

| 方法 | 功能 |
|------|------|
| `store(iri, content)` | 存储条目 |
| `store_entry(entry)` | 存储完整 L0Entry（含标签索引更新） |
| `retrieve(iri)` | 检索条目 |
| `delete(iri)` | 删除条目和索引 |
| `search(query)` | 搜索条目 |
| `search_by_tags(tags)` | 按标签搜索 |
| `get_by_importance(min)` | 按重要性获取 |
| `query_by_type(type_iri)` | 按类型查询 |
| `store_jsonld_node(node)` | 存储 JSON-LD 节点 |
| `retrieve_jsonld_node(iri)` | 检索 JSON-LD 节点 |
| `merge_entries(existing, new)` | 合并相同 @id 节点 |

### 3.2.2 L1 Session — Agent 短期会话

**文件**: `src/memory/l1_session.rs`  
**实现状态**: ✅ 完整

L1 是 Agent 的短期会话记忆，存储当前对话的轮次信息，支持 Token 预算控制和淘汰策略。

```rust
pub struct L1Turn {
    pub role: String,
    pub summary: String,
    pub timestamp: DateTime<Utc>,
    pub l0_archive_iri: Option<String>,  // L0 归档 IRI
    pub embedding: Option<Vec<f32>>,      // 语义向量
}

pub struct L1Session {
    session_id: String,
    agent_id: String,
    agent_role: String,
    task_iri: String,
    turns: Vec<L1Turn>,
    token_budget: usize,
    current_tokens: usize,
    weak_refs: Vec<String>,              // 淘汰的 IRI 弱引用
    mesi_state: MesiState,               // MESI 缓存一致性状态
}
```

**淘汰策略**:

```
得分 = (1 / 距上次访问秒数) × 0.3 + (1 / 语义相关度) × 0.4 + token_cost × 0.3
得分越低越应被淘汰
```

### 3.2.3 L2 Blackboard — 共享黑板

**文件**: `src/memory/l2_blackboard.rs`  
**实现状态**: ✅ 完整  
**存储引擎**: Oxigraph Store + DashMap 节点缓存

L2 是所有 Agent 共享的黑板，基于统一 Oxigraph 存储，支持 SPARQL 读写和命名图隔离。

**核心结构体**:

```rust
pub struct Node {
    pub iri: String,
    pub json_ld: String,
    pub size: usize,
    pub created_at: DateTime<Utc>,
    pub created_by: Option<String>,
    pub tags: Vec<String>,
    pub node_type: Option<String>,
    pub dirty: bool,
    pub mesi_state: MesiState,
    pub parent_task: Option<String>,
    pub named_graph: Option<String>,
    pub jsonld_types: Vec<String>,
}

pub struct Blackboard {
    store: Arc<Store>,                    // 共享 Oxigraph 存储
    node_cache: DashMap<String, Arc<Node>>,  // 节点缓存
    task_nodes: RwLock<HashMap<String, Vec<String>>>,  // 任务→节点索引
    task_tree: RwLock<HashMap<String, TaskTreeNode>>,  // 任务树
    node_count: AtomicU64,
    total_bytes: AtomicU64,
    permission_matrix: PermissionMatrix,
}
```

**构造方式**:

```rust
// 创建独立存储
pub fn new() -> Result<Self, CoreError>

// 使用共享统一存储（推荐）
pub fn with_store(store: Arc<Store>) -> Result<Self, CoreError>
```

**核心方法**:

| 方法 | 功能 |
|------|------|
| `write_node(node_iri, json_ld, config)` | 写入节点（含权限检查 + 大小校验） |
| `write_node_to_graph(node_iri, json_ld, graph_name, config)` | 写入指定命名图 |
| `read_node(iri)` | 读取节点 |
| `query(sparql)` | SPARQL 查询 |
| `query_graph(graph_name, sparql)` | 查询指定命名图 |
| `query_by_types(types)` | 多类型查询 |
| `query_nodes(task_iri)` | 查询任务的所有节点 |
| `write_batch_to_graphs(nodes)` | 批量写入不同命名图 |
| `gc_completed_tasks()` | 自动垃圾回收 |
| `check_permission(role, graph, perm)` | 权限检查 |

**权限强制执行**:

L2 Blackboard 在 `write_node()` 中强制执行权限检查。`system` 角色拥有完全访问权限，其他角色按权限矩阵配置检查，权限拒绝时返回 `CoreError::PermissionDenied`。

**命名图隔离**:

```mermaid
graph TB
    subgraph 命名图布局
        SHARED["blackboard:shared<br/>公共共享区"]
        TASK1["blackboard:task-001<br/>任务1私有区"]
        TASK2["blackboard:task-002<br/>任务2私有区"]
        PREFETCH["blackboard:prefetch<br/>预取缓冲区"]
    end

    subgraph 系统元数据
        SKILLS["system:skills"]
        MCP["system:mcp-tools"]
        HOOKS["system:hooks"]
        AUDIT["system:audit-log"]
    end

    DA1["DA-1"] -->|写入| TASK1
    DA2["DA-2"] -->|写入| TASK2
    CA["CA"] -->|读取| SHARED
    SA["SA"] -->|监控| AUDIT
```

### 3.2.4 L2 作战地图增强

**实现状态**: ✅ 完整 (v2.1.0)

L2 Blackboard 扩展了作战地图（Battle Map）能力，使 Agent 能感知全局态势、协调资源、跟踪跨任务依赖。

#### Agent 态势感知 (AgentTracker)

Blackboard 新增 `agent_registry` 字段（`RwLock<HashMap<String, AgentStatus>>`），实时跟踪每个 Agent 的状态：

```rust
pub struct AgentStatus {
    pub agent_id: String,
    pub agent_role: String,       // Plan/Do/Check/Act/Supervisor
    pub task_iri: String,
    pub status: AgentActivity,    // Idle/Working/Blocked/Error
    pub started_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub current_operation: Option<String>,  // 当前操作描述
    pub resource_locks: Vec<ResourceLock>,  // 持有的资源锁
}
```

**方法**:

| 方法 | 功能 |
|------|------|
| `register_agent(id, role, task)` | 注册 Agent 到作战地图 |
| `update_agent_status(id, activity, op)` | 更新 Agent 活动状态 |
| `update_agent_heartbeat(id)` | 更新心跳时间戳 |
| `get_agent_status(id)` | 查询单个 Agent 状态 |
| `list_active_agents()` | 列出所有活跃 Agent |
| `unregister_agent(id)` | 从作战地图移除 Agent |
| `detect_stale_agents(max_idle_secs)` | 检测心跳超时的 Agent |

**MemoryManager 委派**: `MemoryManager` 提供上述同名方法，通过 `self.l2` 委派到 Blackboard。

#### 资源锁定 (ResourceLock)

资源锁防止多 Agent 并发访问同一资源导致冲突：

```rust
pub enum LockType { Read, Write, Exclusive }
pub struct ResourceLock {
    pub resource_type: String,  // "file", "db", "api", "graph"
    pub resource_id: String,
    pub acquired_at: DateTime<Utc>,
    pub acquired_by: String,     // agent_id
    pub lock_type: LockType,
}
```

**冲突规则**:
- Exclusive 与所有锁类型互斥
- Write 与其它 Write 互斥
- Read 可共存（多个 Read 允许并发）
- 同一 Agent 对自己持有的锁不冲突

**方法**: `acquire_resource()`, `release_resource()`, `release_agent_resources()`, `list_resource_locks()`, `check_resource_available()`

#### 跨任务依赖 (TaskDAG)

`TaskTreeNode` 新增 `dependencies` / `dependents` 字段，支持跨任务依赖跟踪：

```rust
pub struct TaskTreeNode {
    pub task_iri: String,
    pub parent: Option<String>,
    pub children: Vec<String>,
    pub dependencies: Vec<String>,     // 跨任务依赖 (IRI 列表)
    pub dependents: Vec<String>,       // 反向索引
    pub status: String,
    pub node_iris: Vec<String>,
}
```

**方法**: `add_task_dependency()`, `remove_task_dependency()`, `get_task_dependencies()`, `get_task_dependents()`, `get_task_dag()`

`get_task_dag()` 使用 Kahn 算法计算拓扑层级，返回 `Vec<Vec<String>>`，每层可并行执行：

```
Layer 0: [task_a]
Layer 1: [task_b, task_c]   ← b 和 c 可并行
Layer 2: [task_d]
```

#### blackboard:shared 协调区域 (SharedZone)

多 Agent 通过 `blackboard:shared` 命名图交换协调消息和共享状态：

**协调消息**:

```rust
pub struct CoordinationMessage {
    pub from_agent: String,
    pub msg_type: CoordinationMsgType,  // TaskAnnouncement | ProgressUpdate | ResourceRequest | ConflictWarning | SyncRequest
    pub payload: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}
```

**方法**: `publish_coordination()`, `read_coordination_messages()`, `read_coordination_messages_since(since)`

每条消息写入独立 `iri://coordination/{uuid}` 节点，多 Agent 并行发布不冲突。

**共享状态**:

```rust
pub fn publish_shared_state(&self, task_iri: &str, state: &Value) -> Result<(), CoreError>;
pub fn get_shared_state(&self, task_iri: &str) -> Result<Option<Value>, CoreError>;
```

**Agent 快照同步**: `publish_agent_snapshot_to_shared(agent_id)` 将 Agent 状态快照发布到 `blackboard:shared`，使 SPARQL 可直接查询态势数据：

```sparql
-- 查询所有 Working 的 DA
SELECT ?agent ?task ?operation WHERE {
  GRAPH <blackboard:shared> {
    ?agent a <https://agent-os.org/ontology/core/AgentSnapshot> ;
           <https://agent-os.org/ontology/core/agent_role> "Do" ;
           <https://agent-os.org/ontology/core/status> "Working" ;
           <https://agent-os.org/ontology/core/current_operation> ?operation ;
           <https://agent-os.org/ontology/core/task_iri> ?task .
  }
}
```

### 3.2.5 L3 Projection — 投影引擎

**文件**: `src/memory/l3_projection.rs`  
**实现状态**: ✅ 完整

L3 是按需投影引擎，根据 Agent 角色和 Token 预算生成定制化的上下文视图。

```rust
pub struct ProjectionEngine {
    blackboard: Arc<Blackboard>,
    max_size: usize,
    frames: HashMap<String, ProjectionFrame>,
    materialized_cache: RwLock<HashMap<String, MaterializedView>>,
    hyperspace_store: Option<Arc<HyperspaceStore>>,
}
```

**预定义投影模板**:

| 模板名 | 用途 | 包含属性 |
|--------|------|---------|
| `summary_only` | SA 全局态势感知 | summary, status |
| `pa_init` | PA 初始化 | summary, objective, constraints |
| `da_input` | DA 输入 | plan, subtasks, resources |
| `ca_review` | CA 审查 | results, validation_rules |
| `aa_decision` | AA 决策 | review_results, alternatives |
| `reference_only` | 最小引用 | 仅 @id |

**缓存失效机制**:

| 方法 | 功能 |
|------|------|
| `invalidate_for_node(node_iri)` | 使依赖指定节点的所有缓存视图失效 |
| `invalidate_for_nodes(node_iris)` | 批量失效 |
| `cleanup_invalid()` | 清理已失效的缓存条目 |

失效流程：
1. L2 数据写入时通过 MemoryBus 发布 Invalidate 事件
2. L3 监听事件，调用 `invalidate_for_node()` 标记相关缓存为无效
3. 下次投影请求时自动重新生成物化视图

**Frame 驱动投影**:

```mermaid
sequenceDiagram
    participant SA
    participant L3 as L3 Projection
    participant L2 as L2 Blackboard
    participant L0 as L0 Store
    participant HE as HyperspaceEngine

    SA->>L3: project_with_frame(task_iri, frame)
    L3->>L2: SPARQL CONSTRUCT 查询
    L2-->>L3: 原始图数据
    L3->>HE: 向量检索（语义增强）
    HE-->>L3: 相关 IRI 列表
    L3->>L3: apply_frame() 裁剪
    L3-->>SA: 投影结果
```

## 3.3 UnifiedGraphStore — 统一存储

**文件**: `src/memory/unified_graph.rs`  
**实现状态**: ✅ 完整

系统中唯一的 Oxigraph Store 实例，各模块通过 Arc 共享，通过命名图隔离数据域。

```rust
pub struct UnifiedGraphStore {
    store: Arc<Store>,
}

impl UnifiedGraphStore {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>>;
    pub fn store(&self) -> Arc<Store>;  // 获取底层 Store 的 Arc 引用
    pub fn ref_count(&self) -> usize;   // 引用计数诊断
}
```

**共享模型**:

```mermaid
graph TB
    UGS["UnifiedGraphStore<br/>Arc&lt;Store&gt;"] -->|共享| L2["L2 Blackboard<br/>write_node / SPARQL"]
    UGS -->|共享| KGS["KnowledgeGraphStore<br/>知识图谱"]
    UGS -->|共享| SGS["SkillGraphStore<br/>技能图谱"]
    UGS -->|共享| AR["AgentRunner<br/>unified_graph_store"]
```

## 3.4 辅助组件

### 3.4.1 MemoryBus — 内存事件总线

**文件**: `src/memory/memory_bus.rs`  
**实现状态**: ✅ 完整

**事件类型**:
| 事件 | 触发条件 | 处理动作 |
|------|---------|---------|
| `Invalidate(iri)` | L0 数据被修改 | 使所有 L1 缓存行无效 |
| `WriteBack(iri)` | L1 脏数据需回写 | 将 L1 数据写回 L0 |
| `Evict(iri)` | L1 超出 Token 预算 | 淘汰低优先级缓存行 |
| `Prefetch(iri)` | 预测即将访问 | 提前加载到 L2 |
| `Sync(iri, layer)` | 层间同步请求 | 同步指定层的数据 |

**批量操作**:
| 方法 | 功能 |
|------|------|
| `publish_invalidate(iri, scope)` | 单节点缓存失效 |
| `publish_invalidate_batch(iris, scope)` | 批量缓存失效 |
| `publish_with_priority(iri, scope, priority)` | 带优先级的事件发布 |

### 3.4.2 ConsistencyEngine — MESI 一致性

**文件**: `src/memory/consistency_engine.rs`  
**实现状态**: ✅ 完整

```mermaid
stateDiagram-v2
    [*] --> Invalid
    Invalid --> Shared: Read Hit
    Invalid --> Exclusive: Read Miss (独占加载)
    Shared --> Modified: Write Hit
    Shared --> Invalid: Invalidate
    Exclusive --> Modified: Write Hit
    Exclusive --> Shared: Read by Other
    Modified --> Shared: Write Back + Share
    Modified --> Invalid: Invalidate
```

### 3.4.3 HyperspaceEngine — 超空间向量引擎

**文件**: `src/memory/hyperspace_store.rs`, `src/memory/embedding_service.rs`  
**实现状态**: ✅ 完整

HyperspaceEngine 是自包含的向量嵌入引擎，零外部向量数据库依赖。核心特性：

- **HNSW 近似最近邻搜索**：基于 Hierarchical Navigable Small World 图索引，10K 向量 ~1ms 延迟
- **运行时可切换度量**：Poincaré、Cosine、Euclidean、Lorentz，无需重启
- **Write-Ahead Log (WAL)**：CRC32 校验，支持 3 种同步模式（None/Sync/Full），崩溃安全
- **切线空间剪枝**：Poincaré 球搜索时自动剪枝，提升超双曲空间检索精度
- **JSON-LD 元数据索引**：基于 RoaringBitmap 过滤器的元数据索引
- **混合搜索**：文本向量 × 结构嵌入双空间检索

支持多种 Embedding 服务提供商：

| 提供商 | 配置键 | 说明 |
|--------|--------|------|
| Ollama | `ollama` | 本地 Ollama 服务（默认） |
| OneAPI | `oneapi` | OpenAI 兼容 API |
| Fallback | `fallback` | 随机向量兜底 |

### 3.4.4 PrefetchEngine — 预取引擎

**文件**: `src/memory/prefetch_engine.rs`  
**实现状态**: ✅ 完整

基于访问模式的主动预取，提前加载可能需要的数据到 L2。SA 在执行计划时调用 `prefetch.on_intent_change()` 进行预取。

### 3.4.5 MemoryScheduler — 缓存调度

**文件**: `src/memory/scheduler.rs`  
**实现状态**: ✅ 完整

L1 缓存调度器，管理 Token 预算和淘汰策略。可注入 MemoryManager 实现统一管理。

### 3.4.6 MemoryManager — 统一管理器

**文件**: `src/memory/memory_manager.rs`  
**实现状态**: ✅ 完整

```rust
pub struct MemoryManager {
    l0: Arc<L0Store>,
    l2: Arc<Blackboard>,
    projection: Arc<ProjectionEngine>,
    config: CoreConfig,
    sessions: HashMap<String, L1Session>,
    scheduler: Option<Arc<MemoryScheduler>>,
    l1_active_count: AtomicU64,
}
```

**核心方法**:

| 方法 | 功能 |
|------|------|
| `new(l0, l2, projection, config)` | 创建 MemoryManager |
| `with_scheduler(l0, l2, projection, config, scheduler)` | 创建带 Scheduler 的实例 |
| `create_session(agent_id, role, task_iri)` | 创建 L1 session |
| `track_session(session)` | 注册 session 到管理器 |
| `get_session(session_id)` | 获取 session |
| `projection()` | 获取 ProjectionEngine 引用 |

## 3.5 数据流全景

```mermaid
flowchart TB
    subgraph Agent执行
        AR[AgentRunner]
    end

    subgraph 记忆写入
        AR -->|thought+content| L0_W["L0 归档"]
        AR -->|summary| L1_W["L1 追加"]
        AR -->|中间结果| L2_W["L2 黑板<br/>Oxigraph Store"]
    end

    subgraph 记忆读取
        L3_R["L3 投影"] -->|上下文| AR
        L3_R --> L2_R["L2 查询"]
        L3_R --> HE_R["HyperspaceEngine 向量检索"]
        L2_R --> L0_R["L0 回退"]
    end

    subgraph 一致性
        MB["MemoryBus"]
        CE["ConsistencyEngine"]
        L0_W --> MB
        L2_W --> MB
        MB --> CE
        CE -->|Invalidate| L1_W
    end
```
