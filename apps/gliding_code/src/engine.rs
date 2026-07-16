use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use glidinghorse::causal::engine::CausalEngine;
use glidinghorse::causal::fused::FusedRootCauseEngine;
use glidinghorse::causal::store::CausalModelStore;
use glidinghorse::config::{McpServerConfig, McpStdioServerConfig};
use glidinghorse::core::agent_runner::TaskResult;
use glidinghorse::core::event_bus::{EventBus, Event};
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::memory::l1_session::EvictionConfig;
use glidinghorse::gateway::UnifiedGateway;
use glidinghorse::graph_features::features::FeatureExtractor;
use glidinghorse::graph_backend::{GraphBackend, PetgraphBackend, SkillGraphSnapshotBackend};
use glidinghorse::knowledge_graph::store::KnowledgeGraphStore;
use glidinghorse::memory::embedding_service::{create_embedding_service_from_config, FallbackEmbeddingService};
use glidinghorse::memory::hyperspace_store::HyperspaceStore;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::skill_graph::discovery::SkillDiscoveryEngine;
use glidinghorse::skill_graph::evolution::SkillEvolutionEngine;
use glidinghorse::skill_graph::graph_algorithms::SkillGraphAlgorithms;
use glidinghorse::skill_graph::graph_store::SkillGraphStore;
use glidinghorse::snapshots::timeline::TimelineStore;
use glidinghorse::templates::template_engine::TemplateEngine;
use glidinghorse::tools::mcp_client::McpClient;
use glidinghorse::tools::skill_registry::SkillRegistry;
use glidinghorse::tools::workspace_monitor::{WorkspaceMonitor, WorkspaceMonitorConfig};
use glidinghorse::CoreConfig;
use tempfile::TempDir;
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::config::CliConfig;

#[derive(Debug, Clone)]
pub struct AgentEvent {
    pub event_type: String,
    pub source: String,
    pub payload: String,
}

pub struct CodeCliEngine {
    sa: SupervisorAgent,
    event_bus: Arc<EventBus>,
    config: CliConfig,
    _temp_dir: TempDir,
    l2_bb: Arc<Blackboard>,
    proj: Arc<ProjectionEngine>,
    mm: Arc<tokio::sync::Mutex<MemoryManager>>,
    l0: Arc<L0Store>,
    prompt_tokens: Arc<AtomicU64>,
    completion_tokens: Arc<AtomicU64>,
    last_prompt_tokens: Arc<AtomicU64>,
    last_completion_tokens: Arc<AtomicU64>,
    context_limit: u64,
    skills: Arc<SkillRegistry>,
    mcp_client: Option<McpClient>,
    workspace_monitor: Option<Arc<WorkspaceMonitor>>,
    /// Skill Graph Store — cognitive network
    skill_graph: Arc<SkillGraphStore>,
    /// Skill discovery engine (semantic search)
    discovery_engine: Arc<SkillDiscoveryEngine>,
    /// Feature extractor (GNN topological features)
    feature_extractor: Arc<FeatureExtractor>,
    /// Causal engine (Bayesian inference on skill graph)
    causal_engine: Arc<CausalEngine>,
    /// Skill evolution engine (usage tracking & self-improvement)
    evolution_engine: Arc<tokio::sync::Mutex<SkillEvolutionEngine>>,
    /// Timeline store (temporal event recording)
    timeline: Arc<TimelineStore>,
    /// Core config for L2 blackboard writes etc.
    core_config: CoreConfig,
}

impl CodeCliEngine {
    pub fn new(mut config: CliConfig) -> anyhow::Result<Self> {
        // Set the process working directory to the configured workspace so that
        // agent_os tool handlers (execute_file_read/write/edit, execute_bash, …)
        // resolve relative paths against the correct root. Without this they
        // default to std::env::current_dir() which may be anything.
        let workspace_abs = std::path::Path::new(&config.workspace)
            .canonicalize()
            .unwrap_or_else(|_| std::path::PathBuf::from(&config.workspace));
        // Store canonicalized path so engine.workspace() returns the real absolute path
        config.workspace = workspace_abs.to_string_lossy().to_string();
        std::env::set_current_dir(&workspace_abs)
            .map_err(|e| anyhow::anyhow!("无法切换到工作目录 '{}': {}", workspace_abs.display(), e))?;

        let gateway = Arc::new(UnifiedGateway::new(&config.gateway)?);
        let dir = tempfile::TempDir::new()?;

        let l0_path = config.data_dir.as_ref()
            .map(|d| {
                let _ = std::fs::create_dir_all(d);
                d.clone()
            })
            .unwrap_or_else(|| dir.path().join("l0").to_string_lossy().to_string());

        let l0 = Arc::new(
            L0Store::new(&l0_path)
                .map_err(|e| anyhow::anyhow!("L0Store 创建失败: {}", e))?,
        );
        let l2 = Arc::new(
            Blackboard::new()
                .map_err(|e| anyhow::anyhow!("Blackboard 创建失败: {}", e))?,
        );

        // Load agent-os config (config.yaml + AGENT_OS_* env vars) for tunable
        // parameters; fall back to Defaults when no config file is present.
        let loaded_settings = glidinghorse::config::Settings::load().ok();
        let settings = loaded_settings.clone().unwrap_or_default();

        // Initialize HyperspaceEngine-backed vector store for semantic search
        let embed: Arc<dyn glidinghorse::memory::embedding_service::EmbeddingService> =
            match &loaded_settings {
                Some(s) => create_embedding_service_from_config(&s.embedding, s.agents.embedding_timeout_secs),
                None => Arc::new(FallbackEmbeddingService::new()),
            };
        let hyperspace_path = config.data_dir.as_ref()
            .map(|d| format!("{}/hyperspace", d))
            .unwrap_or_else(|| dir.path().join("hyperspace").to_string_lossy().to_string());
        let _ = std::fs::create_dir_all(&hyperspace_path);
        let vector_store = Arc::new(
            HyperspaceStore::open(
                std::path::Path::new(&hyperspace_path),
                embed,
            )
            .map_err(|e| anyhow::anyhow!("HyperspaceStore 初始化失败: {}", e))?,
        );

        let agent_settings = settings.agents.clone();

        let proj = Arc::new(ProjectionEngine::with_vector_store(
            l2.clone(),
            agent_settings.max_projection_size,
            Some(vector_store.clone()),
        ));
        let core_config = CoreConfig {
            max_node_size: settings.memory.l2.max_node_size,
            max_projection_size: agent_settings.max_projection_size,
            l0_storage_path: settings.memory.l0.path.clone(),
            event_buffer_size: settings.agents.event_bus_capacity,
            enable_metrics: true,
            eviction_config: {
                let l1 = &settings.memory.l1;
                if l1.eviction_recency_weight.is_some()
                    || l1.eviction_relevance_weight.is_some()
                    || l1.eviction_cost_weight.is_some()
                {
                    Some(EvictionConfig {
                        recency_weight: l1.eviction_recency_weight.unwrap_or(0.30),
                        relevance_weight: l1.eviction_relevance_weight.unwrap_or(0.40),
                        cost_weight: l1.eviction_cost_weight.unwrap_or(0.30),
                        relevance_threshold: l1.eviction_relevance_threshold.unwrap_or(0.3),
                        safe_window_seconds: l1.eviction_safe_window_seconds.unwrap_or(300),
                        beta: l1.eviction_beta.unwrap_or(0.7),
                    })
                } else {
                    None
                }
            },
        };
        let mm = Arc::new(tokio::sync::Mutex::new(MemoryManager::new(
            l0.clone(),
            l2.clone(),
            proj.clone(),
            core_config.clone(),
        )));
        let mm_for_runner = mm.clone();

        let templates_dir = dir.path().join("templates");
        std::fs::create_dir_all(&templates_dir)?;
        let tmpl = Arc::new(
            TemplateEngine::new(&templates_dir)
                .map_err(|e| anyhow::anyhow!("TemplateEngine 创建失败: {}", e))?,
        );

        let skills = Arc::new(SkillRegistry::new());
        let skills_for_engine = skills.clone();

        let workspace_root = std::path::PathBuf::from(&config.workspace);

        // ── TimelineStore (temporal event recording for graph mutations) ──
        // Created before SkillGraphStore so the store can attach it and record
        // every structural mutation (otherwise TL: pending stays at 0).
        let timeline = Arc::new(TimelineStore::new(
            agent_settings.snapshot_frequency,
            agent_settings.max_full_snapshots,
        ));

        // ── Skill Graph Store — cognitive network ──
        let skill_graph = Arc::new(
            SkillGraphStore::new()
                .with_blackboard(l2.clone())
                .with_l0_store(l0.clone())
                .with_timeline(timeline.clone()),
        );

        // Bootstrap the SkillGraphStore with default skills from SkillRegistry
        // so that SG: N E (node/edge) metrics are non-zero from startup.
        for meta in skills.list_all_skills() {
            if let Err(e) = skill_graph.register_skill(
                glidinghorse::skill_graph::types::SkillGraphNode::from_skill_meta(&meta),
            ) {
                tracing::warn!("Failed to register bootstrap skill {}: {}", meta.name, e);
            }
        }
        let skill_graph_algorithms = Arc::new(SkillGraphAlgorithms::from_store(&skill_graph));

        // ── PetgraphBackend (structural dimension for FusedRootCauseEngine) ──
        let graph_backend: Arc<dyn GraphBackend> = Arc::new(PetgraphBackend::new(skill_graph.clone()));

        // ── AgentRunner (without fused engine — upgraded below after kg_store is available) ──
        let mut runner = glidinghorse::core::agent_runner::AgentRunner::new(
            gateway,
            skills.clone(),
            l2.clone(),
            l0.clone(),
            mm_for_runner,
            tmpl.clone(),
            agent_settings.clone(),
        ).with_prompt_loader(glidinghorse::core::prompt_loader::PromptLoader::new(
            Default::default(),
            tmpl.clone(),
        )).with_workspace_root(workspace_root.clone());

        // Extract the ToolExecutor's KGS, create FusedRootCauseEngine, and upgrade the runner
        let inner_store = {
            let executor = runner.tool_executor.read();
            executor.knowledge_graph_store()
                .read().expect("kg_store RwLock poisoned")
                .store_arc().clone()
        };
        let unified_kg_store = inner_store.clone();
        {
            let fused_kg = Arc::new(
                KnowledgeGraphStore::with_shared_store(inner_store)
                    .expect("Failed to create shared KG Store for FusedRootCauseEngine"),
            );
            let fused_rce = FusedRootCauseEngine::new(
                Some(graph_backend.clone()),
                Some(fused_kg),
            );
            runner = runner.with_fused_root_cause_engine(fused_rce);
        }

        // ── Skill Discovery Engine (semantic skill search via Hyperspace) ──
        let discovery_engine = Arc::new(
            SkillDiscoveryEngine::new(skill_graph.clone())
                .with_vector_store(vector_store.clone()),
        );

        // ── FeatureExtractor (GNN topological features for causal analysis) ──
        use glidinghorse::graph_backend::SkillGraphFeatureGraph;
        let feature_graph = SkillGraphFeatureGraph::new(
            skill_graph.clone(),
            skill_graph_algorithms.clone(),
        );
        let feature_extractor = Arc::new(FeatureExtractor::new(Arc::new(feature_graph)));

        // ── CausalEngine (Bayesian causal inference on skill graph) ──
        let causal_model_store = Arc::new(CausalModelStore::new());
        let causal_engine = Arc::new(CausalEngine::new(
            causal_model_store,
            graph_backend.clone(),
        ));

        // ── SkillEvolutionEngine (usage tracking & self-improvement) ──
        let evolution_engine = Arc::new(tokio::sync::Mutex::new(
            SkillEvolutionEngine::new(skill_graph.clone())
                .with_causal_analysis(5000)
                .with_causal_engine(causal_engine.clone()),
        ));

        let event_bus = Arc::new(EventBus::new(100));

        // TimelineStore EventBus subscription deferred — requires a Tokio runtime.
        // Subscribe via start_async_components() in process_task().

        // 初始化 WorkspaceMonitor — 从 settings.workspace 读取配置
        let workspace_monitor: Option<Arc<WorkspaceMonitor>> = {
            let ws_db_path = {
                let mut p = workspace_root.clone();
                p.push(".gliding_horse/ws_monitor");
                p
            };
            let ws_config = WorkspaceMonitorConfig {
                workspace_root,
                content_store_max_bytes: settings.workspace.content_store_max_bytes,
                content_cache_capacity: settings.workspace.content_cache_capacity,
                watch_enabled: settings.workspace.watch_enabled,
                poll_interval_ms: settings.workspace.poll_interval_ms,
                debounce_ms: settings.workspace.debounce_ms,
                max_debounce_wait_ms: settings.workspace.max_debounce_wait_ms,
                exclude_patterns: settings.workspace.exclude_patterns.clone(),
                db_path: Some(ws_db_path),
                ..Default::default()
            };
            match WorkspaceMonitor::initialize(ws_config, None, Some(event_bus.clone())) {
                Ok(ws) => {
                    ws.register_hooks(&runner.hook_manager);
                    info!(root = %config.workspace, "WorkspaceMonitor 已初始化");
                    Some(Arc::new(ws))
                }
                Err(e) => {
                    warn!("WorkspaceMonitor 初始化失败: {}", e);
                    None
                }
            }
        };

        if let Some(ref wm) = workspace_monitor {
            let mut executor = runner.tool_executor.write();
            executor.set_workspace_monitor(wm.clone());
            wm.set_causal_engine(causal_engine.clone());
        }

        // 注入共享 SkillGraphStore 到 ToolExecutor（create_skill 工具使用）
        {
            let executor = runner.tool_executor.write();
            executor.set_shared_skill_graph(skill_graph.clone());
        }

        // 注入 CausalEngine + SkillGraphStore 到 AgentRunner
        runner = runner
            .with_causal_engine(causal_engine.clone())
            .with_skill_graph_store(skill_graph.clone())
            .with_unified_graph_store(unified_kg_store);

        // 完成 AgentRunner 初始化接线：perception_store → WorkspaceMonitor
        runner.finalize_setup();

        // 保存 runner 的 perception_store 引用，传递给 SA 的 ProactiveEngine
        let runner_perception = runner.perception_store.clone();
        let runner = Arc::new(runner);
        let l2_bb = l2.clone();
        let sa = SupervisorAgent::with_pdca_cycles(
            runner,
            tmpl,
            skills,
            event_bus.clone(),
            config.max_iterations,
            config.max_pdca_cycles,
        )
        .with_memory(Some(l2), None, None)
        .with_execution_timeout(agent_settings.sa_execution_timeout_secs)
        .with_perception_hyperspace(vector_store.clone())
        .with_perception_store(Arc::new(runner_perception))
        .with_discovery_engine(discovery_engine.clone());

        let (prompt_tokens, completion_tokens, last_prompt_tokens, last_completion_tokens) = sa.token_usage_arcs();

        // MCP initialization — register HTTP and stdio servers from config
        let has_mcp = !config.mcp_servers.is_empty() || !config.mcp_stdio_servers.is_empty();
        let mcp_client = if has_mcp {
            let mut client = McpClient::with_timeout(agent_settings.mcp_timeout_secs);
            for server in &config.mcp_servers {
                info!(name = %server.name, url = %server.url, "注册 MCP 服务器 (HTTP)");
                client.register_server(&server.name, &server.url);
            }
            for (name, entry) in &config.mcp_stdio_servers {
                let stdio_config = McpStdioServerConfig {
                    command: entry.command.clone(),
                    args: entry.args.clone(),
                    env: entry.env.clone(),
                    tool_call_timeout_ms: entry.tool_call_timeout_ms,
                };
                let cfg = McpServerConfig::Stdio(stdio_config);
                info!(name = %name, command = %entry.command, "注册 MCP 服务器 (Stdio)");
                client.register_from_config(name, &cfg);
            }
            Some(client)
        } else {
            None
        };

        info!(
            model = %config.model,
            workspace = %config.workspace,
            max_iterations = config.max_iterations,
            mcp_servers = config.mcp_servers.len(),
            "Code CLI 引擎初始化完成"
        );

        let context_limit = Self::resolve_context_limit(&config);

        Ok(Self {
            sa,
            event_bus,
            config,
            _temp_dir: dir,
            l2_bb,
            proj,
            mm,
            l0: l0.clone(),
            prompt_tokens,
            completion_tokens,
            last_prompt_tokens,
            last_completion_tokens,
            context_limit,
            skills: skills_for_engine,
            mcp_client,
            workspace_monitor,
            skill_graph,
            discovery_engine,
            feature_extractor,
            causal_engine,
            evolution_engine,
            timeline,
            core_config,
        })
    }

    pub fn rebuild(&mut self) -> anyhow::Result<()> {
        *self = Self::new(self.config.clone())?;
        Ok(())
    }

    pub fn rebuild_with_model(&mut self, model: String) -> anyhow::Result<()> {
        let model_name = model.clone();
        self.config = self.config.clone_with_model(model);
        // 更新 gateway 的模型配置 + 上下文窗口上限（不重建 Engine，避免 redb 文件锁冲突）
        self.sa.set_model(&model_name);
        self.context_limit = Self::resolve_context_limit(&self.config);
        Ok(())
    }

    pub fn rebuild_with_api_key(&mut self, api_key: String) -> anyhow::Result<()> {
        self.config = self.config.clone_with_api_key(api_key.clone());
        self.sa.set_api_key(&api_key);
        Ok(())
    }

    pub fn rebuild_with_api_url(&mut self, api_url: String) -> anyhow::Result<()> {
        self.config = self.config.clone_with_api_url(api_url.clone());
        self.sa.set_base_url(&api_url);
        Ok(())
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    pub fn api_key(&self) -> &str {
        &self.config.gateway.api_key
    }

    pub fn api_url(&self) -> &str {
        &self.config.gateway.base_url
    }

    pub fn workspace(&self) -> &str {
        &self.config.workspace
    }

    pub fn max_iterations(&self) -> u32 {
        self.config.max_iterations
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.event_bus.subscribe()
    }

    /// Reset workspace perception state for a new task topic.
    /// Clears WorkspaceMonitor inventory and PerceptionStore global entries
    /// to prevent files from previous tasks leaking into the new task's context.
    pub fn reset_perception(&self) {
        if let Some(ref wm) = self.workspace_monitor {
            wm.reset_inventory();
        }
    }

    pub async fn process_task(&mut self, user_input: &str) -> anyhow::Result<(String, TaskResult)> {
        // 首次进入 async 上下文时完成 WorkspaceMonitor 的异步初始化
        if let Some(ref wm) = self.workspace_monitor {
            wm.start_async_components();
            // Trigger rescan when WatchEngine is not active to catch files
            // created between tasks (e.g. by git clones, dependency installs).
            wm.rescan();
        }

        let task_id = uuid::Uuid::new_v4().to_string();
        let task_iri = format!("iri://task/{}", task_id);

        // Collect workspace file summary once for both paths
        let ws_summary = self.workspace_monitor.as_ref()
            .and_then(|wm| wm.get_file_inventory_summary());

        let result = if let Some(ref wf_path) = self.config.workflow_path {
            let wf_jsonld = std::fs::read_to_string(wf_path)
                .map_err(|e| anyhow::anyhow!("读取工作流文件 '{}' 失败: {}", wf_path, e))?;
            let ctx = glidinghorse::core::agent_runner::TaskContext::new(&task_iri, user_input, self.config.max_iterations)
                .with_original_task(user_input)
                .with_workflow(&wf_jsonld);
            let ctx = if let Some(ref summary) = ws_summary {
                ctx.with_workspace_summary(summary)
            } else {
                ctx
            };
            self.sa.process_task_with_context(user_input, &task_iri, ctx).await?
        } else {
            let ctx = glidinghorse::core::agent_runner::TaskContext::new(&task_iri, user_input, self.config.max_iterations)
                .with_original_task(user_input);
            let ctx = if let Some(ref summary) = ws_summary {
                ctx.with_workspace_summary(summary)
            } else {
                ctx
            };
            self.sa.process_task_with_context(user_input, &task_iri, ctx).await?
        };

        info!(
            task_iri = %task_iri,
            status = %result.status,
            turn_count = result.turn_count,
            tool_call_count = result.tool_call_count,
            "任务处理完成"
        );

        // Record post-task metrics for skill evolution + causal analysis
        if let Ok(mut ee) = self.evolution_engine.try_lock() {
            let success = result.status == "completed" || result.status == "success";
            let tool_names: Vec<String> = result.tracked_actions.iter()
                .map(|a| a.tool_name.clone())
                .collect();

            // Track each distinct tool as a skill usage event
            for tool_name in &tool_names {
                let tool_iri = format!("iri://skill/tool/{}", tool_name);
                let _ = ee.record_usage(
                    glidinghorse::skill_graph::evolution::UsageRecord::new(
                        &tool_iri,
                        &task_iri,
                        "system:sa",
                        success,
                    )
                    .with_context_tag(&result.status)
                    .with_tokens(result.turn_count * 100),
                );
            }

            // When task fails, record a failure event that triggers infer_root_cause
            if !success {
                let error_msg = if !result.errors.is_empty() {
                    result.errors.join("; ")
                } else {
                    format!("Task status: {}", result.status)
                };
                let _ = ee.record_usage(
                    glidinghorse::skill_graph::evolution::UsageRecord::new(
                        "iri://skill/task-completion",
                        &task_iri,
                        "system:sa",
                        false,
                    )
                    .with_error(&error_msg)
                    .with_context_tag(&result.status)
                    .with_context_tag(&tool_names.join(",")),
                );

                // Phase 2: Extract causal root cause from evolution engine and
                // publish suggestions to the event bus for observability / downstream use.
                {
                    let suggestions = ee.get_pending_suggestions().to_vec();
                    if !suggestions.is_empty() {
                        let causal_summary = suggestions.iter()
                            .map(|s| format!("{} (conf={:.2}): {}", s.skill_iri, s.confidence, s.description))
                            .collect::<Vec<_>>()
                            .join("; ");
                        info!(
                            task_iri = %task_iri,
                            suggestion_count = suggestions.len(),
                            causal_summary = %causal_summary,
                            "故障因果分析完成，生成了演化建议"
                        );

                        // Publish causal analysis result to event bus
                        let serialized_suggestions: Vec<serde_json::Value> = suggestions.iter()
                            .map(|s| serde_json::json!({
                                "type": format!("{:?}", s.suggestion_type),
                                "skill_iri": s.skill_iri,
                                "description": s.description,
                                "confidence": s.confidence,
                            }))
                            .collect();
                        let _ = self.event_bus.emit(
                            &task_iri,
                            "causal_analysis",
                            "system:evolution_engine",
                            &serde_json::json!({
                                "task_iri": task_iri,
                                "status": result.status,
                                "errors": result.errors,
                                "suggestions": serialized_suggestions,
                            }).to_string(),
                        ).await;
                    }
                }

                // Also collect preventive actions for each affected tool
                for tool_name in &tool_names {
                    let tool_iri = format!("iri://skill/tool/{}", tool_name);
                    let actions = ee.suggest_preventive_action(&tool_iri);
                    for action in &actions {
                        let _ = ee.suggest_link(
                            &tool_iri,
                            "iri://knowledge/causal-analysis",
                            glidinghorse::skill_graph::SkillLinkType::Related,
                            action,
                        );
                    }
                }

                // Persist causal analysis to L2 blackboard for cross-task awareness
                {
                    let suggestions = ee.get_pending_suggestions().to_vec();
                    if !suggestions.is_empty() {
                        let causal_node_iri = format!("{}#causal", task_iri);
                        let all_suggestions: Vec<serde_json::Value> = suggestions.iter()
                            .map(|s| serde_json::json!({
                                "type": format!("{:?}", s.suggestion_type),
                                "skill_iri": s.skill_iri,
                                "description": s.description,
                                "confidence": s.confidence,
                            }))
                            .collect();
                        let causal_json = serde_json::json!({
                            "@id": causal_node_iri,
                            "@type": "CausalAnalysis",
                            "task_iri": task_iri,
                            "status": result.status,
                            "errors": result.errors,
                            "error_message": error_msg,
                            "suggestions": all_suggestions,
                            "timestamp": chrono::Utc::now().to_rfc3339(),
                        });
                        let _ = self.l2_bb.write_node(
                            &causal_node_iri,
                            &causal_json.to_string(),
                            &self.core_config,
                        );
                    }
                }
            }
        }

        // Phase 4: Periodic suggest_improvements & health snapshot after each task
        // (runs unlocked so async calls are safe)
        {
            if let Ok(mut ee) = self.evolution_engine.try_lock() {
                let improvements = ee.suggest_improvements().await;
                if !improvements.is_empty() {
                    let snapshot_json = serde_json::json!({
                        "@id": format!("{}#health-snapshot", task_iri),
                        "@type": "SkillHealthSnapshot",
                        "task_iri": task_iri,
                        "improvements": improvements.iter().map(|s| serde_json::json!({
                            "type": format!("{:?}", s.suggestion_type),
                            "skill_iri": s.skill_iri,
                            "description": s.description,
                            "confidence": s.confidence,
                        })).collect::<Vec<_>>(),
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                    });
                    let snapshot_iri = format!("{}#health-snapshot", task_iri);
                    let _ = self.l2_bb.write_node(
                        &snapshot_iri,
                        &snapshot_json.to_string(),
                        &self.core_config,
                    );
                    info!(
                        task_iri = %task_iri,
                        improvement_count = improvements.len(),
                        "技能图健康快照已保存"
                    );
                }
            }
        }

        Ok((task_iri, result))
    }

    /// Returns a clone of the internal EventBus (for supplementary input / event monitoring).
    pub fn event_bus(&self) -> Arc<EventBus> {
        self.event_bus.clone()
    }

    /// Blackboard reference (lock-free node count reads).
    pub fn l2_bb(&self) -> Arc<Blackboard> {
        self.l2_bb.clone()
    }

    /// ProjectionEngine reference (std RwLock for cache_stats, safe from sync context).
    pub fn proj(&self) -> Arc<ProjectionEngine> {
        self.proj.clone()
    }

    /// MemoryManager Arc (for lock-free L1 session count reads via atomic).
    pub fn mm(&self) -> Arc<tokio::sync::Mutex<MemoryManager>> {
        self.mm.clone()
    }

    /// L0Store reference (for checkpoint loading during resume).
    pub fn l0(&self) -> Arc<L0Store> {
        self.l0.clone()
    }

    /// WorkspaceMonitor reference (for topic shift perception reset).
    pub fn workspace_monitor(&self) -> Option<Arc<WorkspaceMonitor>> {
        self.workspace_monitor.clone()
    }

    /// SkillGraphStore — cognitive network (node/link count, snapshots).
    pub fn skill_graph(&self) -> Arc<SkillGraphStore> {
        self.skill_graph.clone()
    }

    /// SkillDiscoveryEngine — semantic skill search via Hyperspace vectors.
    pub fn discovery_engine(&self) -> Arc<SkillDiscoveryEngine> {
        self.discovery_engine.clone()
    }

    /// FeatureExtractor — GNN topological features for causal analysis.
    pub fn feature_extractor(&self) -> Arc<FeatureExtractor> {
        self.feature_extractor.clone()
    }

    /// CausalEngine — Bayesian causal inference on the skill graph.
    pub fn causal_engine(&self) -> Arc<CausalEngine> {
        self.causal_engine.clone()
    }

    /// SkillEvolutionEngine — usage tracking and self-improvement.
    pub fn evolution_engine(&self) -> Arc<tokio::sync::Mutex<SkillEvolutionEngine>> {
        self.evolution_engine.clone()
    }

    /// TimelineStore — versioned snapshots of skill graph mutations.
    pub fn timeline(&self) -> Arc<TimelineStore> {
        self.timeline.clone()
    }

    /// Token counter Arcs (lock-free reads from TUI).
    /// Returns (total_prompt, total_completion, last_prompt, last_completion).
    pub fn token_arcs(&self) -> (Arc<AtomicU64>, Arc<AtomicU64>, Arc<AtomicU64>, Arc<AtomicU64>) {
        (
            self.prompt_tokens.clone(),
            self.completion_tokens.clone(),
            self.last_prompt_tokens.clone(),
            self.last_completion_tokens.clone(),
        )
    }

    /// 返回模型上下文窗口上限（用于计算 token 占比）。
    pub fn context_limit(&self) -> u64 {
        self.context_limit
    }

    /// 更新模型上下文窗口上限（切换模型时调用）。
    pub fn set_context_limit(&mut self, limit: u64) {
        self.context_limit = limit;
    }

    /// 根据模型名返回上下文窗口上限。
    /// 1. 环境变量 `GLIDING_HORSE_CONTEXT_LIMIT` 优先（所有模型统一覆盖）
    /// 2. 按模型名匹配
    fn model_context_limit(model: &str) -> u64 {
        match model {
            n if n.contains("deepseek-v4") || n.contains("deepseek_v4") => 1_048_576, // 1M
            n if n.contains("deepseek") => 65536,
            n if n.contains("gpt-4") || n.contains("gpt4") => 128000,
            n if n.contains("gpt-3.5") => 16385,
            n if n.contains("gemini") => 1_048_576,
            n if n.contains("llama") || n.contains("qwen") => 128000,
            _ => 128000,
        }
    }

    /// 解析上下文窗口上限。
    /// 优先级：env var > 模型名匹配 > 默认 128K
    fn resolve_context_limit(config: &CliConfig) -> u64 {
        std::env::var("GLIDING_HORSE_CONTEXT_LIMIT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or_else(|| Self::model_context_limit(&config.model))
    }

    /// Query memory subsystem usage counts: (L1_session_count, L2_node_count, L3_projection_count)
    ///
    /// All reads are lock-free or use independent locks (not the engine lock),
    /// so this can be called from the UI thread without blocking.
    pub fn memory_stats(&self) -> (u64, u64, u64) {
        let l2 = self.l2_bb.node_count();
        let l3 = self.proj.cache_stats().total_views as u64;
        let l1 = self.sa.try_l1_session_count().unwrap_or(0);
        (l1, l2, l3)
    }

    pub async fn list_checkpoints(&self) -> anyhow::Result<Vec<glidinghorse::core::checkpoint::CheckpointData>> {
        let prefix = "iri://checkpoint/";
        let entries = self.l0.scan_iri_prefix(prefix, 100)?;
        let mut results: Vec<glidinghorse::core::checkpoint::CheckpointData> = entries
            .iter()
            .filter_map(|e| serde_json::from_str(&e.content).ok())
            .collect();
        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        results.truncate(20);
        Ok(results)
    }

    pub async fn resume_task(&mut self, task_iri: &str) -> anyhow::Result<TaskResult> {
        let cm = glidinghorse::core::checkpoint::CheckpointManager::with_persistence(self.l0.clone());
        let cp = cm.restore_latest(task_iri)?
            .ok_or_else(|| anyhow::anyhow!("没有找到 task_iri={} 的 checkpoint", task_iri))?;

        let _agent_state: serde_json::Value = serde_json::from_str(&cp.agent_state_json)?;

        let resume_input = format!(
            "继续执行之前中断的任务。上次进度: {}\n\n请从上次中断处继续。",
            cp.name
        );
        self.process_task_with_iri(&resume_input, task_iri).await
    }

    /// 从 checkpoint 恢复任务，包含完整的历史上下文消息
    pub async fn resume_task_with_messages(&mut self, task_iri: &str, resumed_messages: Vec<glidinghorse::gateway::unified_gateway::ChatMessage>) -> anyhow::Result<TaskResult> {
        let resume_input = "继续执行之前中断的任务。请从上次中断处继续。".to_string();
        self.process_task_with_iri_and_messages(&resume_input, task_iri, Some(resumed_messages)).await
    }

    /// Process a task with an externally-generated task IRI so the caller
    /// can emit supplementary input events during execution.
    pub async fn process_task_with_iri(&mut self, user_input: &str, task_iri: &str) -> anyhow::Result<TaskResult> {
        self.process_task_with_iri_and_messages(user_input, task_iri, None).await
    }

    /// Process a task with optional resumed messages (for checkpoint resume)
    pub async fn process_task_with_iri_and_messages(
        &mut self,
        user_input: &str,
        task_iri: &str,
        resumed_messages: Option<Vec<glidinghorse::gateway::unified_gateway::ChatMessage>>,
    ) -> anyhow::Result<TaskResult> {
        // Lazy MCP connect — connect to registered servers on first task
        if let Some(ref mut client) = self.mcp_client {
            let needs_connect: Vec<String> = client.list_servers().iter()
                .filter(|s| s.status == "registered")
                .map(|s| s.name.clone())
                .collect();

            for name in &needs_connect {
                info!(server = %name, "连接 MCP 服务器");
                if let Err(e) = client.connect(name).await {
                    warn!("MCP 服务器 '{}' 连接失败: {}", name, e);
                }
            }

            if !needs_connect.is_empty() {
                client.register_tools_to_skill_registry(&self.skills);
            }
        }

        use glidinghorse::core::agent_runner::TaskContext;

        let ws_summary = self.workspace_monitor.as_ref()
            .and_then(|wm| wm.get_file_inventory_summary());

        let ctx = TaskContext::new(task_iri, user_input, self.config.max_iterations)
            .with_original_task(user_input);
        let ctx = if let Some(ref summary) = ws_summary {
            ctx.with_workspace_summary(summary)
        } else {
            ctx
        };
        let ctx = if let Some(ref wf_path) = self.config.workflow_path {
            let wf_jsonld = std::fs::read_to_string(wf_path)
                .map_err(|e| anyhow::anyhow!("读取工作流文件 '{}' 失败: {}", wf_path, e))?;
            ctx.with_workflow(&wf_jsonld)
        } else {
            ctx
        };
        let ctx = if let Some(msgs) = resumed_messages {
            let turn_count = msgs.iter().filter(|m| m.role == "assistant").count() as u32;
            let tool_count = msgs.iter().filter(|m| m.role == "tool" || m.tool_call_id.is_some()).count() as u32;
            ctx.with_resumed_messages(msgs, turn_count, tool_count)
        } else {
            ctx
        };

        let result = self.sa.process_task_with_context(user_input, task_iri, ctx).await?;

        info!(
            task_iri = %task_iri,
            status = %result.status,
            turn_count = result.turn_count,
            tool_call_count = result.tool_call_count,
            "任务处理完成"
        );

        // Snapshot the skill graph to the TimelineStore after each task,
        // enabling temporal rollback and traceability of graph evolution.
        let backend = SkillGraphSnapshotBackend::new(self.skill_graph.clone());
        self.timeline.create_snapshot(&backend, &format!("task:{}", result.status.as_str()));

        Ok(result)
    }
}

