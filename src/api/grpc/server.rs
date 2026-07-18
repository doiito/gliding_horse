use std::sync::Arc;
use std::pin::Pin;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use tonic::{Request, Response, Status};
use tokio_stream::Stream;
use tokio::sync::{mpsc, RwLock};

use crate::batch::manager::BatchAgentManager;
use crate::core::sa::SupervisorAgent;
use crate::core::agent_runner::AgentRunner;
use crate::core::event_bus::EventBus;
use crate::core::checkpoint::CheckpointManager;
use crate::core::execution_event::ExecutionEventEmitter;
use crate::core::execution_event::ExecutionEventKind;
use crate::core::execution_event::ExecutionState;
use crate::gateway::unified_gateway::UnifiedGateway;
use crate::memory::consistency_engine::ConsistencyEngine;
use crate::memory::l0_store::L0Store;
use crate::memory::l2_blackboard::Blackboard;
use crate::memory::l3_projection::ProjectionEngine;
use crate::memory::memory_bus::MemoryBus;
use crate::memory::memory_manager::MemoryManager;
use crate::memory::prefetch_engine::PrefetchEngine;
use crate::memory::scheduler::MemoryScheduler;
use crate::memory::unified_graph::UnifiedGraphStore;
use crate::skill_graph::graph_store::SkillGraphStore;
use crate::templates::template_engine::TemplateEngine;
use crate::tools::skill_registry::SkillRegistry;
use crate::tools::workspace_monitor::{WorkspaceMonitor, WorkspaceMonitorConfig};
use crate::config::settings::Settings;
use crate::CoreConfig;

pub mod seapp {
    tonic::include_proto!("seapp");
}

use seapp::*;

static TASK_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct AgentOSService {
    settings: Settings,
    gateway: Arc<UnifiedGateway>,
    l0: Arc<L0Store>,
    blackboard: Arc<Blackboard>,
    projection: Arc<ProjectionEngine>,
    memory_manager: Arc<tokio::sync::Mutex<MemoryManager>>,
    skills: Arc<SkillRegistry>,
    templates: Arc<TemplateEngine>,
    event_bus: Arc<EventBus>,
    checkpoints: Arc<CheckpointManager>,
    scheduler: Arc<MemoryScheduler>,
    prefetch: Arc<PrefetchEngine>,
    unified_graph: Arc<UnifiedGraphStore>,
    execution_states: Arc<RwLock<HashMap<String, ExecutionState>>>,
    /// Batch Agent manager, post-new async initialization
    batch_manager: tokio::sync::Mutex<Option<BatchAgentManager>>,
    /// Skill graph store for background maintenance (archive + re-index)
    skill_graph: Option<Arc<SkillGraphStore>>,
}

impl AgentOSService {
    pub fn new(settings: Settings) -> Result<Self, String> {
        let gateway = Arc::new(
            UnifiedGateway::new(&settings.gateway)
                .map_err(|e| format!("Gateway init failed: {}", e))?
        );

        let l0 = Arc::new(
            L0Store::new(&settings.memory.l0.path)
                .map_err(|e| format!("L0 init failed: {}", e))?
        );

        let unified_graph = Arc::new(
            UnifiedGraphStore::new().map_err(|e| format!("UnifiedGraph init failed: {}", e))?
        );

        let blackboard = Arc::new(
            Blackboard::with_store(unified_graph.store()).map_err(|e| format!("L2 init failed: {}", e))?
        );
        let projection = Arc::new(ProjectionEngine::new(blackboard.clone(), settings.memory.l3.max_size));
        let skills = Arc::new(SkillRegistry::new());
        let templates_path = settings.agents.template_path
            .as_deref()
            .unwrap_or("src/templates/templates");
        let templates = Arc::new(
            TemplateEngine::new(std::path::Path::new(templates_path))
                .map_err(|e| format!("Template engine init failed (path={}): {}", templates_path, e))?
        );
        let event_bus = Arc::new(EventBus::new(settings.agents.event_bus_capacity));

        let memory_bus = Arc::new(MemoryBus::new(event_bus.clone()));
        let consistency = Arc::new(ConsistencyEngine::new(
            memory_bus.clone(), l0.clone(), blackboard.clone(), projection.clone(),
        ));
        let scheduler = Arc::new(MemoryScheduler::new(
            l0.clone(), blackboard.clone(), projection.clone(), consistency.clone(), memory_bus.clone(),
        ));
        let prefetch = Arc::new(PrefetchEngine::new(
            memory_bus.clone(), blackboard.clone(), projection.clone(),
        ));

        let memory_manager = Arc::new(tokio::sync::Mutex::new(
            MemoryManager::with_scheduler(l0.clone(), blackboard.clone(), projection.clone(), CoreConfig::default(), scheduler.clone()),
        ));

        let checkpoints = Arc::new(CheckpointManager::new());

        let eb_checkpoint = event_bus.clone();
        let cp_clone = checkpoints.clone();
        eb_checkpoint.spawn_consumer(
            vec!["CYCLE_STARTED".to_string(), "CYCLE_COMPLETED".to_string()],
            move |event| {
                let cp = cp_clone.clone();
                async move {
                    match event.event_type.as_str() {
                        "CYCLE_STARTED" => {
                            let id = cp.create(&event.task_iri, &format!("cycle:{}", event.task_iri), "{}", "{}", "{}", &[]);
                            tracing::debug!(checkpoint_id = ?id, "Checkpoint created for cycle start");
                        }
                        "CYCLE_COMPLETED" => {
                            let _ = cp.restore(&event.task_iri);
                            tracing::debug!("Checkpoint restored for cycle completion");
                        }
                        _ => {}
                    }
                }
            },
        );

        let eb_5w2h = event_bus.clone();
        eb_5w2h.spawn_consumer(
            vec!["DEADLINE_APPROACHING".to_string(), "BUDGET_EXCEEDED".to_string()],
            move |event| {
                let et = event.event_type.clone();
                async move {
                    tracing::warn!(
                        event_type = %et,
                        task_iri = %event.task_iri,
                        "5W2H constraint alert consumed: needs attention"
                    );
                }
            },
        );

        let eb_invalidate = event_bus.clone();
        let l0_inv = l0.clone();
        let bb_inv = blackboard.clone();
        eb_invalidate.spawn_consumer(
            vec!["MEMORY_INVALIDATE".to_string(), "CACHE_INVALIDATE".to_string()],
            move |event| {
                let bb = bb_inv.clone();
                let l0 = l0_inv.clone();
                async move {
                    tracing::info!(
                        event_type = %event.event_type,
                        task_iri = %event.task_iri,
                        "Cache invalidation event consumed"
                    );
                    let _ = (l0, bb);
                }
            },
        );

        let eb_prefetch = event_bus.clone();
        let bb_prefetch = blackboard.clone();
        let proj_prefetch = projection.clone();
        eb_prefetch.spawn_consumer(
            vec!["MEMORY_PREFETCH".to_string(), "PREFETCH_REQUEST".to_string()],
            move |event| {
                let bb = bb_prefetch.clone();
                let proj = proj_prefetch.clone();
                async move {
                    tracing::info!(
                        event_type = %event.event_type,
                        task_iri = %event.task_iri,
                        "Prefetch request event consumed"
                    );
                    let _ = (bb, proj);
                }
            },
        );

        let eb_tasks = event_bus.clone();
        eb_tasks.spawn_consumer(
            vec![
                "TASK_STARTED".to_string(),
                "TASK_COMPLETED".to_string(),
                "TASK_FAILED".to_string(),
                "AGENT_ERROR".to_string(),
            ],
            move |event| {
                async move {
                    match event.event_type.as_str() {
                        "TASK_FAILED" | "AGENT_ERROR" => {
                            tracing::warn!(
                                event_type = %event.event_type,
                                task_iri = %event.task_iri,
                                source = %event.source_agent_iri,
                                "Task failure event"
                            );
                        }
                        _ => {
                            tracing::info!(
                                event_type = %event.event_type,
                                task_iri = %event.task_iri,
                                source = %event.source_agent_iri,
                                "Task lifecycle event"
                            );
                        }
                    }
                }
            },
        );

        // ── BatchAgent manager (sync register, async start) ──
        let skill_graph = Arc::new(
            SkillGraphStore::new()
                .with_blackboard(blackboard.clone())
                .with_l0_store(l0.clone())
                .with_oxi_store(unified_graph.store()),
        );
        let batch_mgr = {
            let mut mgr = BatchAgentManager::new()
                .with_event_bus(event_bus.clone())
                .with_graph_store(skill_graph.clone());

            let agent_settings = &settings.batch_agents.agents;
            if !agent_settings.is_empty() {
                let results = mgr.register_maintenance_agents(agent_settings);
                let ok = results.iter().filter(|r| r.is_ok()).count();
                let err = results.len() - ok;
                tracing::info!(
                    "BatchAgent registration complete: {} OK, {} failed, {} total configs",
                    ok, err, results.len()
                );
                for r in results.iter().filter_map(|r| r.as_ref().err()) {
                    tracing::warn!("BatchAgent registration failed: {:?}", r);
                }
            }
            mgr
        };

        let s = Self {
            settings,
            gateway,
            l0,
            blackboard: blackboard.clone(),
            projection,
            memory_manager,
            skills,
            templates,
            event_bus: event_bus.clone(),
            checkpoints,
            scheduler,
            prefetch,
            unified_graph,
            execution_states: Arc::new(RwLock::new(HashMap::new())),
            batch_manager: tokio::sync::Mutex::new(Some(batch_mgr)),
            skill_graph: Some(skill_graph),
        };

        Ok(s)
    }

/// Assemble axum HTTP/SSE routes, reusing the service's runtime shared state (EventBus / Blackboard /
/// SkillRegistry etc.), so HTTP `/api/v1/tasks/stream` and gRPC task execution share the same event bus.
    pub fn build_http_router(&self) -> axum::Router {
        use crate::core::core_types::SemanticCore;
        use crate::core::validation::ValidationEngine;

        let config = CoreConfig::default();
        let core = Arc::new(SemanticCore {
            blackboard: self.blackboard.clone(),
            l0_store: self.l0.clone(),
            projection: self.projection.clone(),
            skills: self.skills.clone(),
            events: self.event_bus.clone(),
            validation: Arc::new(ValidationEngine::new(config.max_node_size)),
            checkpoints: self.checkpoints.clone(),
            config,
        });
        crate::api::http::build_router(core, self.unified_graph.store())
    }

    /// Async start BatchAgent system + background maintenance tasks. Call before gRPC serve.
    pub async fn init_batch_system(&self) {
        let mut guard = self.batch_manager.lock().await;
        if let Some(ref mut mgr) = *guard {
            match mgr.start(None).await {
                Ok(()) => tracing::info!("BatchAgent system started"),
                Err(e) => tracing::warn!("BatchAgent partial startup failure: {:?}", e),
            }
        } else {
            tracing::info!("BatchAgent initialized or disabled");
        }
        drop(guard);

        // ── Background maintenance: archive + re-index every 30 minutes ──
        if let Some(ref sg) = self.skill_graph {
            let sg_clone = sg.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(1800));
                loop {
                    interval.tick().await;

                    // Archive cold skills (L2→L0, last_used > 48 hours ago)
                    let cutoff = chrono::Utc::now() - chrono::Duration::hours(48);
                    match sg_clone.archive_cold_skills(&cutoff) {
                        Ok(archived) => {
                            if archived > 0 {
                                tracing::info!(archived = archived, "Maintenance: archived cold skills");
                            }
                        }
                        Err(e) => tracing::warn!("Maintenance: archive_cold_skills failed: {}", e),
                    }

                    // Re-index stale skills (updated_at > 4 hours ago)
                    let stale_age = chrono::Duration::hours(4);
                    let reindexed = sg_clone.reindex_stale_skills(&stale_age);
                    if reindexed > 0 {
                        tracing::info!(reindexed = reindexed, "Maintenance: re-indexed stale skills");
                    }
                }
            });
            tracing::info!("Background maintenance task spawned (archive=48h, reindex=4h, interval=30min)");
        }
    }

    fn create_sa(&self, settings: &Settings) -> SupervisorAgent {
        // initialize WorkspaceMonitor (if workspace root is configured)
        let workspace_root_path: Option<std::path::PathBuf> = settings.workspace.root.as_ref().map(|s| std::path::PathBuf::from(s));
        let workspace_monitor_opt: Option<Arc<WorkspaceMonitor>> = if let Some(ref ws_root) = workspace_root_path {
            let ws_config = WorkspaceMonitorConfig {
                workspace_root: ws_root.clone(),
                exclude_patterns: settings.workspace.exclude_patterns.clone(),
                watch_enabled: settings.workspace.watch_enabled,
                content_store_max_bytes: settings.workspace.content_store_max_bytes,
                ..Default::default()
            };
            match WorkspaceMonitor::initialize(ws_config, Some(self.blackboard.clone()), Some(self.event_bus.clone())) {
                Ok(ws) => {
                    tracing::info!(root = %ws_root.display(), "WorkspaceMonitor initialized");
                    Some(Arc::new(ws))
                }
                Err(e) => {
                    tracing::warn!("WorkspaceMonitor init failed: {}, using default workspace settings", e);
                    None
                }
            }
        } else {
            None
        };

        let mut runner_builder = AgentRunner::new(
            self.gateway.clone(),
            self.skills.clone(),
            self.blackboard.clone(),
            self.l0.clone(),
            self.memory_manager.clone(),
            self.templates.clone(),
            settings.agents.clone(),
        )
        .with_scheduler(self.scheduler.clone())
        .with_prefetch_engine(self.prefetch.clone())
        .with_unified_graph_store(self.unified_graph.store());
        if let Some(ref sg) = self.skill_graph {
            runner_builder = runner_builder.with_skill_graph_store(sg.clone());
        }
        if let Some(ref ws_root) = workspace_root_path {
            runner_builder = runner_builder.with_workspace_root(ws_root.clone());
        }

        let runner = Arc::new(runner_builder);

        {
            let ug_store = self.unified_graph.store();
            let mut executor = runner.tool_executor.write();
            executor.set_unified_kg_store(ug_store);
            if let Some(ref sg) = self.skill_graph {
                executor.set_shared_skill_graph(sg.clone());
            }
            if let Some(ref wm) = workspace_monitor_opt {
                executor.set_workspace_monitor(wm.clone());
            }
        }

        // register WorkspaceMonitor hooks into AgentRunner's hook_manager
        if let Some(ref wm) = workspace_monitor_opt {
            wm.register_hooks(&runner.hook_manager);
        }

        // complete AgentRunner init wiring: perception_store → WorkspaceMonitor
        runner.finalize_setup();

        let mut sa = SupervisorAgent::with_pdca_cycles(
            runner,
            self.templates.clone(),
            self.skills.clone(),
            self.event_bus.clone(),
            settings.agents.max_iterations,
            settings.agents.max_pdca_cycles,
        );

        sa = sa.with_memory(Some(self.blackboard.clone()), Some(self.prefetch.clone()), Some(self.scheduler.clone()));
        sa
    }

    fn apply_request_settings(&self, req: &impl RequestSettings) -> Settings {
        let mut settings = self.settings.clone();
        req.apply_to(&mut settings);
        settings
    }
}

trait RequestSettings {
    fn apply_to(&self, settings: &mut Settings);
}

impl AgentOSService {
    pub async fn send_supplementary_input(
        &self,
        task_iri: &str,
        content: &str,
    ) {
        tracing::info!(task_iri = %task_iri, "Received user supplementary input");
        self.event_bus.emit(
            task_iri,
            "USER_SUPPLEMENTARY_INPUT",
            "external",
            content,
        ).await;
    }
}

impl RequestSettings for ExecuteStageRequest {
    fn apply_to(&self, settings: &mut Settings) {
        if !self.llm_api_key.is_empty() {
            settings.gateway.api_key = self.llm_api_key.clone();
        }
        if !self.llm_base_url.is_empty() {
            settings.gateway.base_url = self.llm_base_url.clone();
        }
        if !self.llm_model.is_empty() {
            settings.gateway.default_model = self.llm_model.clone();
            settings.gateway.model_mapping.insert("default".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("planning".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("execution".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("analysis".to_string(), self.llm_model.clone());
        }
    }
}

impl RequestSettings for ChatStreamRequest {
    fn apply_to(&self, settings: &mut Settings) {
        if !self.llm_api_key.is_empty() {
            settings.gateway.api_key = self.llm_api_key.clone();
        }
        if !self.llm_base_url.is_empty() {
            settings.gateway.base_url = self.llm_base_url.clone();
        }
        if !self.llm_model.is_empty() {
            settings.gateway.default_model = self.llm_model.clone();
            settings.gateway.model_mapping.insert("default".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("planning".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("execution".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("analysis".to_string(), self.llm_model.clone());
        }
    }
}

impl RequestSettings for ExecuteTaskStreamRequest {
    fn apply_to(&self, settings: &mut Settings) {
        if !self.llm_api_key.is_empty() {
            settings.gateway.api_key = self.llm_api_key.clone();
        }
        if !self.llm_base_url.is_empty() {
            settings.gateway.base_url = self.llm_base_url.clone();
        }
        if !self.llm_model.is_empty() {
            settings.gateway.default_model = self.llm_model.clone();
            settings.gateway.model_mapping.insert("default".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("planning".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("execution".to_string(), self.llm_model.clone());
            settings.gateway.model_mapping.insert("analysis".to_string(), self.llm_model.clone());
        }
    }
}

#[tonic::async_trait]
impl seapp::se_kernel_service_server::SeKernelService for AgentOSService {
    type ChatStreamStream = Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, Status>> + Send>>;
    type ExecuteTaskStreamStream = Pin<Box<dyn Stream<Item = Result<seapp::ExecutionEvent, Status>> + Send>>;

    async fn execute_stage(
        &self,
        request: Request<ExecuteStageRequest>,
    ) -> Result<Response<ExecuteStageResponse>, Status> {
        let req = request.into_inner();
        let settings = self.apply_request_settings(&req);

        let mut sa = self.create_sa(&settings);

        let task_iri = if req.task_iri.is_empty() {
            format!("iri://stage/{}", req.stage_id)
        } else {
            req.task_iri
        };

        let result = sa.process_task(&req.prompt, &task_iri).await
            .map_err(|e| Status::internal(format!("SA execution failed: {}", e)))?;

        let output_bytes = match &result.output {
            Some(v) => serde_json::to_vec(v).unwrap_or_default(),
            None => Vec::new(),
        };

        Ok(Response::new(ExecuteStageResponse {
            status: result.status.clone(),
            summary: result.summary.clone(),
            output_json: output_bytes,
            output_iri: task_iri,
            artifacts: vec![],
            errors: result.errors.clone(),
        }))
    }

    async fn chat_stream(
        &self,
        request: Request<ChatStreamRequest>,
    ) -> Result<Response<Self::ChatStreamStream>, Status> {
        let req = request.into_inner();
        let settings = self.apply_request_settings(&req);

        let (tx, rx) = mpsc::channel::<Result<ChatStreamChunk, Status>>(64);

        let mut sa = self.create_sa(&settings);

        let task_iri = if req.task_iri.is_empty() {
            format!("iri://chat/{}", uuid::Uuid::new_v4().hyphenated())
        } else {
            req.task_iri.clone()
        };

        let _ = tx.send(Ok(ChatStreamChunk {
            content: String::new(),
            done: false,
            status: "processing".to_string(),
        })).await;

        match sa.process_task(&req.prompt, &task_iri).await {
            Ok(result) => {
                let content = extract_content(&result);

                let chunk_size = 20;
                let chars: Vec<char> = content.chars().collect();
                for chunk in chars.chunks(chunk_size) {
                    let chunk_str: String = chunk.iter().collect();
                    if tx.send(Ok(ChatStreamChunk {
                        content: chunk_str,
                        done: false,
                        status: "streaming".to_string(),
                    })).await.is_err() {
                        return Ok(Response::new(Box::pin(
                            tokio_stream::wrappers::ReceiverStream::new(rx)
                        )));
                    }
                }

                let _ = tx.send(Ok(ChatStreamChunk {
                    content: String::new(),
                    done: true,
                    status: result.status.clone(),
                })).await;
            }
            Err(e) => {
                let _ = tx.send(Ok(ChatStreamChunk {
                    content: format!("Error: {}", e),
                    done: true,
                    status: "error".to_string(),
                })).await;
            }
        }

        let output = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output)))
    }

    async fn execute_task_stream(
        &self,
        request: Request<ExecuteTaskStreamRequest>,
    ) -> Result<Response<Self::ExecuteTaskStreamStream>, Status> {
        let req = request.into_inner();
        let settings = self.apply_request_settings(&req);

        let (tx, rx) = mpsc::channel::<Result<seapp::ExecutionEvent, Status>>(256);

        let task_iri = if req.task_iri.is_empty() {
            let seq = TASK_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
            format!("iri://stream/{}", seq)
        } else {
            req.task_iri.clone()
        };

        let include_thought = req.include_thought;
        let include_tool_calls = req.include_tool_calls;

        {
            let mut states = self.execution_states.write().await;
            states.insert(task_iri.clone(), ExecutionState::new());
        }

        let event_bus = self.event_bus.clone();
        let states = self.execution_states.clone();
        let task_iri_clone = task_iri.clone();
        let mut event_rx = event_bus.subscribe();

        let tx_clone = tx.clone();
        let states_clone = states.clone();
        tokio::spawn(async move {
            loop {
                match event_rx.recv().await {
                    Ok(event) => {
                        if event.task_iri != task_iri_clone {
                            continue;
                        }

                        if let Some((core_event, proto_event)) = convert_event_bus_to_grpc(&event) {
                            let mut states = states_clone.write().await;
                            if let Some(state) = states.get_mut(&task_iri_clone) {
                                state.update_from_event(&core_event);
                            }
                            if tx_clone.send(Ok(proto_event)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        });

        let settings_clone = settings.clone();
        let sa_settings = settings.clone();
        let prompt = req.prompt.clone();
        let task_iri_for_task = task_iri.clone();
        let _tx_for_task = tx.clone();
        let event_bus_for_task = self.event_bus.clone();

        tokio::spawn(async move {
            let mut sa = {
                let service = AgentOSService::new(settings_clone).expect("AgentOSService::new failed");
                service.create_sa(&sa_settings)
            };

            let emitter = ExecutionEventEmitter::with_options(
                &task_iri_for_task,
                None,
                Some(event_bus_for_task),
                include_thought,
                include_tool_calls,
            );

            emitter.emit_phase_change("idle", "plan", "PA", "Task started");

            match sa.process_task(&prompt, &task_iri_for_task).await {
                Ok(result) => {
                    emitter.emit_completion(&result.status, &result.summary, result.output.clone());
                }
                Err(e) => {
                    emitter.emit_error("ExecutionError", &e.to_string(), "SA", false);
                    emitter.emit_completion("failed", &e.to_string(), None);
                }
            }

            {
                let mut states = states.write().await;
                states.remove(&task_iri_for_task);
            }
        });

        let output = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output)))
    }

    async fn get_execution_details(
        &self,
        request: Request<GetExecutionDetailsRequest>,
    ) -> Result<Response<ExecutionDetails>, Status> {
        let req = request.into_inner();
        let task_iri = req.task_iri;

        let states = self.execution_states.read().await;
        let state = states.get(&task_iri).cloned().unwrap_or_default();

        let details = ExecutionDetails {
            task_iri: task_iri.clone(),
            status: "running".to_string(),
            current_phase: state.current_phase.clone(),
            plan: None,
            steps: vec![],
            agent_sessions: vec![],
            stats: Some(ExecutionStats {
                total_turns: state.current_turn as i32,
                total_tool_calls: 0,
                total_tokens: 0,
                prompt_tokens: 0,
                completion_tokens: 0,
                error_count: 0,
                retry_count: 0,
            }),
            created_at: String::new(),
            updated_at: String::new(),
            duration_ms: 0,
        };

        Ok(Response::new(details))
    }

    async fn get_realtime_status(
        &self,
        request: Request<GetRealtimeStatusRequest>,
    ) -> Result<Response<RealtimeStatus>, Status> {
        let req = request.into_inner();
        let task_iri = req.task_iri;

        let states = self.execution_states.read().await;
        let state = states.get(&task_iri).cloned().unwrap_or_default();

        let status = RealtimeStatus {
            task_iri: task_iri.clone(),
            status: "running".to_string(),
            current_phase: state.current_phase.clone(),
            current_agent: Some(CurrentAgentInfo {
                id: state.current_agent_id.clone(),
                role: state.current_agent_role.clone(),
                status: "running".to_string(),
                turn: state.current_turn as i32,
            }),
            current_action: state.current_tool.as_ref().map(|t| CurrentActionInfo {
                r#type: "tool_call".to_string(),
                tool_name: t.clone(),
                started_at: String::new(),
            }),
            progress: Some(ExecutionProgress {
                completed_steps: state.completed_steps as i32,
                total_steps: state.total_steps as i32,
                percentage: if state.total_steps > 0 {
                    (state.completed_steps * 100 / state.total_steps) as i32
                } else {
                    0
                },
                estimated_remaining_ms: 0,
            }),
            phase_history: state.phase_history.iter().map(|p| PhaseHistoryEntry {
                phase: p.phase.clone(),
                agent_id: p.agent_id.clone(),
                started_at: p.started_at,
                completed_at: p.completed_at.unwrap_or(0),
                status: p.status.clone(),
            }).collect(),
        };

        Ok(Response::new(status))
    }

    async fn validate_contract(
        &self,
        _request: Request<ValidateContractRequest>,
    ) -> Result<Response<ValidateContractResponse>, Status> {
        Ok(Response::new(ValidateContractResponse {
            valid: true,
            violations: vec![],
        }))
    }

    async fn flatten_to_frontend(
        &self,
        _request: Request<FlattenRequest>,
    ) -> Result<Response<FlattenResponse>, Status> {
        Ok(Response::new(FlattenResponse {
            frontend_json: "{}".to_string(),
        }))
    }

    async fn submit_human_approval(
        &self,
        _request: Request<SubmitApprovalRequest>,
    ) -> Result<Response<SubmitApprovalResponse>, Status> {
        Ok(Response::new(SubmitApprovalResponse {
            success: true,
            message: "ok".to_string(),
        }))
    }
}

fn convert_event_bus_to_grpc(event: &crate::core::event_bus::Event) -> Option<(crate::core::execution_event::ExecutionEvent, seapp::ExecutionEvent)> {
    use crate::core::event_bus::EventType;
    use crate::core::execution_event::ExecutionEvent as CoreExecutionEvent;

    let event_type = EventType::from_str(&event.event_type);
    let timestamp = event.timestamp.timestamp_millis();

    let kind = match event_type {
        EventType::PlanStarted => ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
            from_phase: "idle".to_string(),
            to_phase: "plan".to_string(),
            agent_role: "PA".to_string(),
            reason: "Plan phase started".to_string(),
        }),
        EventType::PlanCompleted => ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
            from_phase: "plan".to_string(),
            to_phase: "do".to_string(),
            agent_role: "PA".to_string(),
            reason: "Plan phase completed".to_string(),
        }),
        EventType::DoStarted => ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
            from_phase: "plan".to_string(),
            to_phase: "do".to_string(),
            agent_role: "DA".to_string(),
            reason: "Do phase started".to_string(),
        }),
        EventType::DoCompleted => ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
            from_phase: "do".to_string(),
            to_phase: "check".to_string(),
            agent_role: "DA".to_string(),
            reason: "Do phase completed".to_string(),
        }),
        EventType::CheckStarted => ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
            from_phase: "do".to_string(),
            to_phase: "check".to_string(),
            agent_role: "CA".to_string(),
            reason: "Check phase started".to_string(),
        }),
        EventType::CheckCompleted => ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
            from_phase: "check".to_string(),
            to_phase: "act".to_string(),
            agent_role: "CA".to_string(),
            reason: "Check phase completed".to_string(),
        }),
        EventType::ActStarted => ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
            from_phase: "check".to_string(),
            to_phase: "act".to_string(),
            agent_role: "AA".to_string(),
            reason: "Act phase started".to_string(),
        }),
        EventType::ActCompleted => ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
            from_phase: "act".to_string(),
            to_phase: "completed".to_string(),
            agent_role: "AA".to_string(),
            reason: "Act phase completed".to_string(),
        }),
        EventType::AgentStarted => ExecutionEventKind::AgentStatus(crate::core::execution_event::AgentStatus {
            agent_id: event.source_agent_iri.clone(),
            role: "unknown".to_string(),
            status: "running".to_string(),
            turn: 0,
            iteration: 0,
            timestamp: None,
        }),
        EventType::AgentCompleted => ExecutionEventKind::AgentStatus(crate::core::execution_event::AgentStatus {
            agent_id: event.source_agent_iri.clone(),
            role: "unknown".to_string(),
            status: "completed".to_string(),
            turn: 0,
            iteration: 0,
            timestamp: None,
        }),
        EventType::AgentError => ExecutionEventKind::Error(crate::core::execution_event::Error {
            error_type: "AgentError".to_string(),
            message: event.payload.clone(),
            agent_id: event.source_agent_iri.clone(),
            recoverable: false,
        }),
        EventType::TaskCompleted => ExecutionEventKind::Completion(crate::core::execution_event::Completion {
            status: "success".to_string(),
            summary: event.payload.clone(),
            total_turns: 0,
            total_tool_calls: 0,
            total_tokens: 0,
            output_json: None,
        }),
        EventType::TaskFailed => ExecutionEventKind::Completion(crate::core::execution_event::Completion {
            status: "failed".to_string(),
            summary: event.payload.clone(),
            total_turns: 0,
            total_tool_calls: 0,
            total_tokens: 0,
            output_json: None,
        }),
        _ => return None,
    };

    let core_event = CoreExecutionEvent {
        event_id: event.event_id.clone(),
        task_iri: event.task_iri.clone(),
        timestamp,
        event: kind.clone(),
    };

    let proto_event = seapp::ExecutionEvent {
        event_id: event.event_id.clone(),
        task_iri: event.task_iri.clone(),
        timestamp,
        event: Some(kind_to_proto_event(kind)),
    };

    Some((core_event, proto_event))
}

fn kind_to_proto_event(kind: ExecutionEventKind) -> seapp::execution_event::Event {
    use seapp::execution_event::Event;

    match kind {
        ExecutionEventKind::PhaseChange(pc) => Event::PhaseChange(PhaseChangeEvent {
            from_phase: pc.from_phase,
            to_phase: pc.to_phase,
            agent_role: pc.agent_role,
            reason: pc.reason,
        }),
        ExecutionEventKind::AgentStatus(as_) => Event::AgentStatus(AgentStatusEvent {
            agent_id: as_.agent_id,
            role: as_.role,
            status: as_.status,
            turn: as_.turn as i32,
            iteration: as_.iteration as i32,
        }),
        ExecutionEventKind::LlmContent(lc) => Event::LlmContent(LlmContentEvent {
            agent_id: lc.agent_id,
            role: lc.role,
            content_delta: lc.content_delta,
            is_reasoning: lc.is_reasoning,
            token_count: lc.token_count as i32,
        }),
        ExecutionEventKind::ToolCall(tc) => Event::ToolCall(ToolCallEvent {
            call_id: tc.call_id,
            tool_name: tc.tool_name,
            arguments_json: tc.arguments_json,
            agent_id: tc.agent_id,
            sequence: tc.sequence as i32,
        }),
        ExecutionEventKind::ToolResult(tr) => Event::ToolResult(ToolResultEvent {
            call_id: tr.call_id,
            tool_name: tr.tool_name,
            result: tr.result,
            success: tr.success,
            result_size_bytes: tr.result_size_bytes as i32,
            duration_ms: tr.duration_ms as i32,
        }),
        ExecutionEventKind::Thought(t) => Event::Thought(ThoughtEvent {
            agent_id: t.agent_id,
            thought: t.thought,
            action: t.action,
            emphasis: t.emphasis,
        }),
        ExecutionEventKind::TokenUsage(tu) => Event::TokenUsage(TokenUsageEvent {
            prompt_tokens: tu.prompt_tokens as i32,
            completion_tokens: tu.completion_tokens as i32,
            total_tokens: tu.total_tokens as i32,
            model: tu.model,
            turn: tu.turn as i32,
        }),
        ExecutionEventKind::Error(e) => Event::Error(ErrorEvent {
            error_type: e.error_type,
            message: e.message,
            agent_id: e.agent_id,
            recoverable: e.recoverable,
        }),
        ExecutionEventKind::Completion(c) => Event::Completion(CompletionEvent {
            status: c.status,
            summary: c.summary,
            total_turns: c.total_turns as i32,
            total_tool_calls: c.total_tool_calls as i32,
            total_tokens: c.total_tokens as i32,
            output_json: c.output_json.map(|v| serde_json::to_string(&v).unwrap_or_default()).unwrap_or_default(),
        }),
    }
}

fn extract_content(result: &crate::core::agent_runner::TaskResult) -> String {
    if let Some(ref output) = result.output {
        match output {
            serde_json::Value::String(s) => {
                let cleaned = clean_content(s);
                if !cleaned.is_empty() {
                    return cleaned;
                }
            }
            serde_json::Value::Object(map) => {
                if let Some(content) = map.get("content").and_then(|v| v.as_str()) {
                    let cleaned = clean_content(content);
                    if !cleaned.is_empty() {
                        return cleaned;
                    }
                }
                if let Some(summary) = map.get("summary").and_then(|v| v.as_str()) {
                    let cleaned = clean_content(summary);
                    if !cleaned.is_empty() {
                        return cleaned;
                    }
                }
            }
            _ => {}
        }
        if let Some(formatted) = serde_json::to_string_pretty(output).ok() {
            return formatted;
        }
    }

    if !result.summary.is_empty() {
        return clean_content(&result.summary);
    }

    "No content returned".to_string()
}

fn clean_content(text: &str) -> String {
    let re = regex::Regex::new(r#"\{[^}]*"thought"[^}]*\}"#).ok();
    let cleaned = re.map(|r| r.replace_all(text, "").to_string()).unwrap_or_else(|| text.to_string());
    let cleaned = cleaned.trim().to_string();
    if cleaned.is_empty() { text.to_string() } else { cleaned }
}
