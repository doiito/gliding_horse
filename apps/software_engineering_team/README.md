# 中心管控 + 边缘执行 联邦架构

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VS Code Plugin (TypeScript)                  │
│          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│          │  Chat Panel  │  │  Graph View  │  │  Task Panel  │       │
│          └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│                 │                  │                  │              │
│                 └──────────────────┼──────────────────┘              │
│                                    │  WebSocket / REST              │
└────────────────────────────────────┼────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────┐
│                    Edge Daemon (Rust)                               │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  API Server   │  │  Agent Core  │  │  Sandbox (Docker, 预留)   │  │
│  │  (axum)       │──│  (SA/DA)     │──│  - 安全执行环境          │  │
│  │  /api/health  │  │  LLM Client  │  │  - 代码编译/测试         │  │
│  │  /api/chat    │  │  async-openai│  │  - 文件操作              │  │
│  │  /api/ws      │  └──────┬───────┘  └──────────────────────────┘  │
│  └───────┬───────┘         │                                         │
│          │                 │                                         │
│  ┌───────┴─────────────────┴──────────────────────────────────────┐ │
│  │  Sync Layer                        │  Graph Layer               │ │
│  │  - heartbeat (注册/保活)           │  - local_store (sled)     │ │
│  │  - client (与 Center gRPC 通信)    │  - delta (变更同步)        │ │
│  │  - auth (JWT 认证)                │  - sync (图数据同步)       │ │
│  └────────────────────────────────────┴────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ gRPC + REST
┌────────────────────────────────┼────────────────────────────────────┐
│                      Center (Go)                                    │
│  ┌─────────────────┐  ┌───────┴────────┐  ┌──────────────────────┐  │
│  │  HTTP API (gin) │  │  Temporal      │  │  Agent Manager       │  │
│  │  /api/v1/*      │  │  Workflow      │  │  - 注册/心跳         │  │
│  │  /ws            │  │  Orchestrator  │  │  - 任务分配           │  │
│  └────────┬────────┘  └───────┬────────┘  └──────────┬───────────┘  │
│           │                   │                      │              │
│  ┌────────┴───────────────────┴──────────────────────┴───────────┐  │
│  │  Executors                        │  Store (SQLite/gRPC)      │  │
│  │  - requirement / design          │  - meta_store (SQLite)    │  │
│  │  - coding / review / testing     │  - gRPC client → Edge     │  │
│  │  - cicd / deploy / extract       │  - graph sync             │  │
│  └───────────────────────────────────┴───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 技术栈

| 层级 | 语言 | 框架 | 职责 |
|------|------|------|------|
| **Center** | Go 1.25+ | Gin + Temporal + gRPC | 工作流编排、项目管理、Agent 注册、API 网关 |
| **Edge Daemon** | Rust 1.94+ | axum + async-openai + bollard | 本地 LLM 执行、Docker 沙箱（重型隔离，预留）、图数据缓存、VS Code 通信 |
| **VS Code Plugin** | TypeScript | VS Code API + WebSocket | 开发者交互界面、任务视图、实时日志、图可视化 |

## 目录结构

```
software_engineering_golang_team/
├── README.md                    # 本文档
├── .env.example                 # 环境变量模板
├── center/                      # Go 中心管控服务
│   ├── cmd/
│   │   ├── server/main.go       # 中心 API 服务入口
│   │   └── worker/main.go       # Temporal Worker 入口
│   ├── internal/
│   │   ├── agent/               # Agent 管理（注册、心跳、状态）
│   │   │   ├── agent_manager.go
│   │   │   └── agent_store.go
│   │   ├── api/                 # HTTP API 处理器
│   │   │   ├── router.go        # 路由定义（~50 个端点）
│   │   │   ├── service.go       # 服务层依赖注入
│   │   │   ├── agent_handler.go
│   │   │   ├── project_handler.go
│   │   │   ├── review_handler.go
│   │   │   ├── pipeline_handler.go
│   │   │   ├── graph_handler.go
│   │   │   ├── dashboard_handler.go
│   │   │   └── websocket.go
│   │   ├── config/              # 配置加载（viper + godotenv）
│   │   │   └── config.go
│   │   ├── executor/            # SDLC 各阶段执行器
│   │   │   ├── executor.go      # 执行器接口
│   │   │   ├── requirement.go
│   │   │   ├── design.go
│   │   │   ├── coding.go
│   │   │   ├── review.go
│   │   │   ├── testing.go
│   │   │   ├── cicd.go
│   │   │   ├── deploy.go
│   │   │   ├── extract.go
│   │   │   └── builtin.go
│   │   ├── graph/               # 图数据管理器
│   │   │   └── graph_manager.go
│   │   ├── grpc/                # gRPC 客户端（与 Edge 通信）
│   │   │   └── client.go
│   │   ├── store/               # 元数据存储（SQLite）
│   │   │   └── sqlite_store.go
│   │   ├── workflow/            # Temporal 工作流定义
│   │   │   ├── activities.go
│   │   │   ├── sdlc_workflow.go
│   │   │   ├── worker.go
│   │   │   ├── callback.go
│   │   │   ├── human_review.go
│   │   │   └── pipeline/
│   │   │       └── dsl.go
│   │   └── types/               # 共享类型
│   │       ├── types.go
│   │       └── meta.go
│   ├── proto/seapp/             # Protobuf 定义及生成代码
│   │   ├── se_app.proto
│   │   ├── se_app.pb.go
│   │   └── se_app_grpc.pb.go
│   ├── config.yaml              # 中心配置
│   ├── go.mod / go.sum
│   └── server                   # 编译产物
├── edge/
│   ├── daemon/                  # Rust 边缘守护进程
│   │   ├── src/
│   │   │   ├── main.rs          # CLI 入口（daemon start / register）
│   │   │   ├── config.rs        # 配置结构体（YAML 反序列化）
│   │   │   ├── server.rs        # axum HTTP 服务器
│   │   │   ├── agent/           # 本地 Agent 执行
│   │   │   │   ├── mod.rs
│   │   │   │   ├── sa.rs        # Supervisor Agent
│   │   │   │   ├── da.rs        # Do Agent（执行 Agent）
│   │   │   │   └── runner.rs    # Agent 执行器（LLM + 工具循环）
│   │   │   ├── api/             # HTTP API
│   │   │   │   ├── mod.rs       # 路由挂载
│   │   │   │   ├── health.rs    # 健康检查
│   │   │   │   ├── chat.rs      # 聊天接口
│   │   │   │   └── ws.rs        # WebSocket 事件推送
│   │   │   ├── sync/            # 与 Center 同步
│   │   │   │   ├── mod.rs
│   │   │   │   ├── heartbeat.rs # 心跳保活
│   │   │   │   ├── client.rs    # gRPC 客户端
│   │   │   │   └── auth.rs      # JWT 认证
│   │   │   ├── graph/           # 本地图存储
│   │   │   │   ├── mod.rs
│   │   │   │   ├── local_store.rs # sled 持久化
│   │   │   │   ├── delta.rs     # 变更追踪
│   │   │   │   └── sync.rs      # 中心同步
│   │   │   └── sandbox/         # Docker 沙箱（重型隔离，预留；主仓 unshare 沙盒为默认轻量级）
│   │   │       ├── mod.rs
│   │   │       └── manage.rs    # 容器生命周期
│   │   ├── Cargo.toml
│   │   ├── Cargo.lock
│   │   └── config.yaml          # 运行时会从此目录加载
│   └── vscode/                  # VS Code 插件
│       ├── package.json         # 插件清单（命令、视图、配置）
│       ├── src/
│       │   ├── extension.ts     # 插件激活入口
│       │   ├── agentClient.ts   # Daemon HTTP 客户端
│       │   ├── chatPanel.ts     # 聊天面板
│       │   ├── graphPanel.ts    # 图可视化面板
│       │   ├── taskPanel.ts     # 任务列表面板
│       │   └── statusBar.ts     # 状态栏指示器
│       └── webview/
│           └── chatPanel.html   # 内嵌 WebView
└── tests/                       # 集成测试（进行中）
```

## 快速开始

### 前置要求

- Go 1.25+
- Rust 1.94+
- Cargo（Rust 工具链）
- Docker（沙箱执行，重型隔离——预留；默认命令沙盒走主仓 unshare 进程级隔离）
- Temporal Server（工作流引擎）

### 配置

```bash
# 复制环境变量模板
cp .env.example .env
# 编辑 .env 填入实际密钥
vim .env
```

Center 配置文件位于 `center/config.yaml`，Edge Daemon 配置位于 `edge/daemon/config.yaml`。

### 构建

```bash
# 构建 Center（Go）
cd center
go build ./...

# 构建 Edge Daemon（Rust）
cd edge/daemon
cargo build
```

### 运行

```bash
# 启动 Center API 服务
cd center && go run ./cmd/server/...

# 启动 Temporal Worker
cd center && go run ./cmd/worker/...

# 启动 Edge Daemon
cd edge/daemon && cargo run -- daemon start
```

### 测试

```bash
cd center && go test ./...
```

## 配置说明

### Center 配置 (`center/config.yaml`)

```yaml
server:
  host: "0.0.0.0"         # HTTP API 监听地址
  port: 8080               # HTTP API 监听端口

temporal:
  host: "localhost"        # Temporal Server 地址
  port: 7233               # Temporal Server 端口
  task_queue: "se-center-queue"  # 任务队列名

grpc:
  host: "localhost"        # gRPC 监听地址
  port: 50051              # gRPC 监听端口

meta_store:
  driver: "sqlite3"        # 元数据存储驱动
  dsn: "se_center.db"      # SQLite 数据库文件路径

llm:
  api_key: ""              # LLM API 密钥（优先从环境变量读取）
  base_url: "https://api.openai.com/v1"  # LLM API 地址
  model: "gpt-4o"          # 默认模型
  provider: "openai"       # 供应商（openai / anthropic / deepseek 等）
```

### Edge Daemon 配置 (`edge/daemon/config.yaml`)

```yaml
server:
  host: "127.0.0.1"        # Daemon HTTP 监听地址
  port: 7890               # Daemon HTTP 监听端口

center:
  url: "http://localhost:8080"           # Center 服务地址
  auth_token: ""                         # 认证令牌

llm:
  provider: "openai"       # LLM 供应商
  api_key: ""              # LLM API 密钥
  model: "gpt-4o"          # 默认模型
  base_url: "https://api.openai.com/v1"  # API 地址

sandbox:
  docker_socket: "/var/run/docker.sock"  # Docker 套接字路径
  default_image: "ubuntu:22.04"          # 默认沙箱镜像（重型容器级隔离，预留）

graph:
  db_path: "./data/graph"  # 本地图数据库路径
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ONE_API_URL` | `http://localhost:3000` | LLM 统一网关地址 |
| `ONE_API_KEY` | — | LLM 统一网关密钥 |
| `CENTER_URL` | `http://localhost:8080` | Center 服务地址 |
| `DAEMON_URL` | `http://localhost:7890` | Edge Daemon 地址 |
| `DOCKER_SOCKET` | `unix:///var/run/docker.sock` | Docker 套接字 |
| `SE_TEMPORAL_HOST` | `localhost` | Temporal Server 地址 |
| `SE_LLM_API_KEY` | — | Center 端 LLM 密钥 |
| `SE_GRPC_HOST` | `localhost` | gRPC 服务地址 |

## API 端点总览

### Center HTTP API (`/api/v1/*`)

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/projects` | 创建项目 |
| `GET` | `/api/v1/projects` | 项目列表 |
| `GET` | `/api/v1/projects/:id` | 获取项目详情 |
| `PUT` | `/api/v1/projects/:id` | 更新项目 |
| `DELETE` | `/api/v1/projects/:id` | 删除项目 |
| `POST` | `/api/v1/projects/:id/tasks` | 创建任务 |
| `GET` | `/api/v1/projects/:id/tasks` | 任务列表 |
| `GET` | `/api/v1/tasks` | 全部任务 |
| `GET` | `/api/v1/tasks/:id` | 任务详情 |
| `PUT` | `/api/v1/tasks/:id` | 更新任务 |
| `POST` | `/api/v1/tasks/:id/retry` | 重试任务 |
| `POST` | `/api/v1/tasks/:id/rollback` | 回滚任务 |
| `POST` | `/api/v1/pipelines` | 启动流水线 |
| `GET` | `/api/v1/pipelines/:id` | 流水线结果 |
| `GET` | `/api/v1/tasks/:id/stages` | 阶段结果列表 |
| `GET