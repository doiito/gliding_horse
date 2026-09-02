use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::batch::extractor::ExtractorPipeline;
use crate::batch::manager::BatchAgentManager;
use crate::batch::prompt::DynamicPromptEngine;
use crate::batch::types::{BatchMetrics, PromptContext};
use crate::batch::BatchEventJournal;
use crate::config::settings::Settings;
use crate::core::agent_runner::AgentRunner;
use crate::core::checkpoint::CheckpointManager;
use crate::core::event_bus::EventBus;
use crate::core::execution_event::ExecutionEventEmitter;
use crate::core::execution_event::ExecutionEventKind;
use crate::core::execution_event::ExecutionState;
use crate::core::sa::SupervisorAgent;
use crate::core::TaskFinalizer;
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
    batch_manager: Arc<tokio::sync::Mutex<Option<BatchAgentManager>>>,
    /// Skill graph store for background maintenance (archive + re-index)
    skill_graph: Option<Arc<SkillGraphStore>>,
}

/// Bring the graph projection up to the registry's baseline without
/// overwriting an evolved/persisted node with the same IRI.  Keeping this as a
/// helper makes the root gRPC bootstrap contract independently testable.
fn bootstrap_skill_graph_from_registry(
    registry: &SkillRegistry,
    graph: &SkillGraphStore,
) -> Result<usize, crate::CoreError> {
    let mut inserted = 0;
    for meta in registry.list_all_skills() {
        if graph.get_skill(&meta.skill_iri).is_some() {
            continue;
        }
        graph.register_skill(crate::skill_graph::types::SkillGraphNode::from_skill_meta(
            &meta,
        ))?;
        inserted += 1;
    }
    Ok(inserted)
}

/// Keep root-service RDF state beside its configured L0 data.  This avoids a
/// second unrelated process-wide default path and gives L0/UnifiedGraph the
/// same deployment lifetime.
fn unified_graph_path(settings: &Settings) -> PathBuf {
    PathBuf::from(&settings.memory.l0.path).join("unified-graph")
}

fn node_iri_from_invalidate_event(payload: &str, fallback: &str) -> String {
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()
        .and_then(|v| v.get("node_iri").and_then(|n| n.as_str()).map(String::from))
        .unwrap_or_else(|| fallback.to_string())
}

/// Evict a clean L2 cache line so the next read reloads from L0 (MESI-style).
/// Dirty lines are authoritative and must be retained until flushed.
fn evict_stale_l2_line(blackboard: &Blackboard, node_iri: &str) -> bool {
    match blackboard.read_node(node_iri) {
        Ok(Some(node)) if !node.dirty => blackboard.delete_node(node_iri).unwrap_or(false),
        _ => false,
    }
}

async fn execute_batch_agent(
    manager: &mut BatchAgentManager,
    pipeline: &ExtractorPipeline,
    agent_name: &str,
    context: &PromptContext,
) -> bool {
    match manager.execute_ready(agent_name, pipeline, context).await {
        Ok(_) => true,
        Err(error) => {
            tracing::warn!(agent = %agent_name, %error, "Batch execution did not complete");
            false
        }
    }
}

async fn run_batch_agents_for_event(
    batch_manager: &Arc<tokio::sync::Mutex<Option<BatchAgentManager>>>,
    pipeline: &ExtractorPipeline,
    event: &crate::core::event_bus::Event,
    context: &PromptContext,
    journal: &BatchEventJournal,
) {
    let mut guard = batch_manager.lock().await;
    let Some(manager) = guard.as_mut() else {
        return;
    };
    // A CustomEvent is itself a trigger, so execute each explicitly matched
    // agent after enqueueing. Other trigger kinds are evaluated by the tick.
    let names = manager.enqueue_custom_event(event);
    if names.is_empty() {
        return;
    }
    match journal.record(event) {
        Ok(true) => {}
        Ok(false) => {
            tracing::debug!(event_id = %event.event_id, "Skipping duplicate pending batch event");
            return;
        }
        Err(error) => {
            tracing::warn!(%error, event_id = %event.event_id, "Batch event was not journaled; refusing execution");
            return;
        }
    }
    let mut complete = true;
    for name in names {
        complete &= execute_batch_agent(manager, pipeline, &name, context).await;
    }
    if complete {
        if let Err(error) = journal.acknowledge(&event.event_id) {
            tracing::warn!(%error, event_id = %event.event_id, "Batch event completed but journal acknowledgement failed");
        }
    }
}

async fn run_ready_batch_agents(
    batch_manager: &Arc<tokio::sync::Mutex<Option<BatchAgentManager>>>,
    pipeline: &ExtractorPipeline,
    context: &PromptContext,
) {
    let mut guard = batch_manager.lock().await;
    let Some(manager) = guard.as_mut() else {
        return;
    };
    let names: Vec<String> = manager
        .list_agents()
        .into_iter()
        .map(str::to_owned)
        .collect();
    for name in names {
        if !manager.evaluate_triggers(&name).await.is_empty() {
            let _ = execute_batch_agent(manager, pipeline, &name, context).await;
        }
    }
}

async fn replay_pending_batch_events(
    batch_manager: &Arc<tokio::sync::Mutex<Option<BatchAgentManager>>>,
    pipeline: &ExtractorPipeline,
    journal: &BatchEventJournal,
) {
    let pending = match journal.pending(10_000) {
        Ok(events) => events,
        Err(error) => {
            tracing::warn!(%error, "Failed to read pending batch journal events");
            return;
        }
    };
    for envelope in pending {
        let event = envelope.into_event();
        let context = PromptContext {
            context_summary: Some(event.payload.clone()),
            ..Default::default()
        };
        let mut guard = batch_manager.lock().await;
        let Some(manager) = guard.as_mut() else {
            return;
        };
        let names = manager.enqueue_custom_event(&event);
        let mut complete = !names.is_empty();
        for name in names {
            complete &= execute_batch_agent(manager, pipeline, &name, &context).await;
        }
        drop(guard);
        if complete {
            if let Err(error) = journal.acknowledge(&event.event_id) {
                tracing::warn!(%error, event_id = %event.event_id, "Failed to acknowledge replayed batch event");
            }
        }
    }
}

impl AgentOSService {
    pub fn new(settings: Settings) -> Result<Self, String> {
        let gateway = Arc::new(
            UnifiedGateway::new(&settings.gateway)
                .map_err(|e| format!("Gateway init failed: {}", e))?,
        );

        let l0 = Arc::new(
            L0Store::new(&settings.memory.l0.path).map_err(|e| format!("L0 init failed: {}", e))?,
        );

        let unified_graph = Arc::new(
            UnifiedGraphStore::new_persistent(unified_graph_path(&settings))
                .map_err(|e| format!("UnifiedGraph init failed: {}", e))?,
        );

        let blackboard = Arc::new(
            Blackboard::with_store(unified_graph.store())
                .map_err(|e| format!("L2 init failed: {}", e))?,
        );
        let projection = Arc::new(ProjectionEngine::new(
            blackboard.clone(),
            settings.memory.l3.max_size,
        ));
        let skills = Arc::new(SkillRegistry::new());
        let templates_path = settings
            .agents
            .template_path
            .as_deref()
            .unwrap_or("src/templates/templates");
        let templates = Arc::new(
            TemplateEngine::new(std::path::Path::new(templates_path)).map_err(|e| {
                format!(
                    "Template engine init failed (path={}): {}",
                    templates_path, e
                )
            })?,
        );
        let event_bus = Arc::new(EventBus::new(settings.agents.event_bus_capacity));

        let memory_bus = Arc::new(MemoryBus::new(event_bus.clone()));
        let consistency = Arc::new(ConsistencyEngine::new(
            memory_bus.clone(),
            l0.clone(),
            blackboard.clone(),
            projection.clone(),
        ));
        let scheduler = Arc::new(MemoryScheduler::new(
            l0.clone(),
            blackboard.clone(),
            projection.clone(),
            consistency.clone(),
            memory_bus.clone(),
        ));
        let prefetch = Arc::new(PrefetchEngine::new(
            memory_bus.clone(),
            blackboard.clone(),
            projection.clone(),
        ));

        let memory_manager = Arc::new(tokio::sync::Mutex::new(MemoryManager::with_scheduler(
            l0.clone(),
            blackboard.clone(),
            projection.clone(),
            CoreConfig::default(),
            scheduler.clone(),
        )));

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
            vec![
                "DEADLINE_APPROACHING".to_string(),
                "BUDGET_EXCEEDED".to_string(),
            ],
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
        let bb_inv = blackboard.clone();
        eb_invalidate.spawn_consumer(
            vec![
                "MEMORY_INVALIDATE".to_string(),
                "CACHE_INVALIDATE".to_string(),
            ],
            move |event| {
                let bb = bb_inv.clone();
                async move {
                    let node_iri = node_iri_from_invalidate_event(&event.payload, &event.task_iri);
                    if evict_stale_l2_line(&bb, &node_iri) {
                        tracing::debug!(
                            node_iri = %node_iri,
                            event_type = %event.event_type,
                            "Evicted clean L2 cache line on invalidation"
                        );
                    }
                }
            },
        );

        let eb_prefetch = event_bus.clone();
        prefetch.spawn_consumer(eb_prefetch, blackboard.clone());

        let eb_tasks = event_bus.clone();
        eb_tasks.spawn_consumer(
            vec![
                "TASK_STARTED".to_string(),
                "TASK_COMPLETED".to_string(),
                "TASK_FAILED".to_string(),
                "AGENT_ERROR".to_string(),
            ],
            move |event| async move {
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
            },
        );

        // ── BatchAgent manager (sync register, async start) ──
        let skill_graph = Arc::new(
            SkillGraphStore::new()
                .with_blackboard(blackboard.clone())
                .with_l0_store(l0.clone())
                .with_oxi_store(unified_graph.store()),
        );
        if let Err(error) = skill_graph.hydrate_from_l0() {
            tracing::warn!(%error, "Failed to hydrate persisted skill graph; continuing with bootstrap skills");
        }
        match bootstrap_skill_graph_from_registry(skills.as_ref(), skill_graph.as_ref()) {
            Ok(inserted) if inserted > 0 => {
                tracing::info!(inserted, "Bootstrapped root gRPC skill graph from registry")
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(%error, "Failed to bootstrap root gRPC skill graph from registry")
            }
        }
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
                    ok,
                    err,
                    results.len()
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
            batch_manager: Arc::new(tokio::sync::Mutex::new(Some(batch_mgr))),
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
        crate::api::http::build_router(
            core,
            self.unified_graph.store(),
            self.settings.api.auth_token.clone(),
        )
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

        // A batch configuration only receives events it explicitly names via a
        // CustomEvent trigger. The adapter turns such events into window entries,
        // evaluates the configured trigger, and runs exactly one ready batch.
        // A periodic tick performs the same evaluation for cron/window triggers.
        let batch_manager = self.batch_manager.clone();
        let event_bus = self.event_bus.clone();
        let journal = BatchEventJournal::new(self.l0.clone());
        let prompt_engine = Arc::new(DynamicPromptEngine::new(
            self.templates.clone(),
            Some(self.l0.clone()),
        ));
        let pipeline = Arc::new(ExtractorPipeline::new(
            self.gateway.clone(),
            prompt_engine,
            Arc::new(std::sync::Mutex::new(BatchMetrics::default())),
        ));
        replay_pending_batch_events(&batch_manager, pipeline.as_ref(), &journal).await;
        tokio::spawn(async move {
            let mut events = event_bus.subscribe();
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                tokio::select! {
                    received = events.recv() => match received {
                        Ok(event) => {
                            let context = PromptContext {
                                context_summary: Some(event.payload.clone()),
                                ..Default::default()
                            };
                            run_batch_agents_for_event(&batch_manager, &pipeline, &event, &context, &journal).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(lagged = n, "Batch event adapter lagged; missed events are not replayed");
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    },
                    _ = tick.tick() => {
                        run_ready_batch_agents(&batch_manager, &pipeline, &PromptContext::default()).await;
                    }
                }
            }
        });
        tracing::info!("Batch event/cron adapter spawned");

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
                                tracing::info!(
                                    archived = archived,
                                    "Maintenance: archived cold skills"
                                );
                            }
                        }
                        Err(e) => tracing::warn!("Maintenance: archive_cold_skills failed: {}", e),
                    }

                    // Re-index stale skills (updated_at > 4 hours ago)
                    let stale_age = chrono::Duration::hours(4);
                    let reindexed = sg_clone.reindex_stale_skills(&stale_age);
                    if reindexed > 0 {
                        tracing::info!(
                            reindexed = reindexed,
                            "Maintenance: re-indexed stale skills"
                        );
                    }
                }
            });
            tracing::info!(
                "Background maintenance task spawned (archive=48h, reindex=4h, interval=30min)"
            );
        }
    }

    fn create_sa(&self, settings: &Settings) -> SupervisorAgent {
        // initialize WorkspaceMonitor (if workspace root is configured)
        let workspace_root_path: Option<std::path::PathBuf> = settings
            .workspace
            .root
            .as_ref()
            .map(|s| std::path::PathBuf::from(s));
        let workspace_monitor_opt: Option<Arc<WorkspaceMonitor>> =
            if let Some(ref ws_root) = workspace_root_path {
                let ws_config = WorkspaceMonitorConfig {
                    workspace_root: ws_root.clone(),
                    exclude_patterns: settings.workspace.exclude_patterns.clone(),
                    watch_enabled: settings.workspace.watch_enabled,
                    content_store_max_bytes: settings.workspace.content_store_max_bytes,
                    content_cache_capacity: settings.workspace.content_cache_capacity,
                    poll_interval_ms: settings.workspace.poll_interval_ms,
                    debounce_ms: settings.workspace.debounce_ms,
                    max_debounce_wait_ms: settings.workspace.max_debounce_wait_ms,
                    initial_scan_wait_ms: settings.workspace.initial_scan_wait_ms,
                    change_history_capacity: settings.workspace.change_history_capacity,
                    effect_snapshot_max_files: settings.workspace.effect_snapshot_max_files,
                    effect_snapshot_max_bytes: settings.workspace.effect_snapshot_max_bytes,
                    ..Default::default()
                };
                match WorkspaceMonitor::initialize(
                    ws_config,
                    Some(self.blackboard.clone()),
                    Some(self.event_bus.clone()),
                ) {
                    Ok(ws) => {
                        tracing::info!(root = %ws_root.display(), "WorkspaceMonitor initialized");
                        Some(Arc::new(ws))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "WorkspaceMonitor init failed: {}, using default workspace settings",
                            e
                        );
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
            executor.set_shared_skill_registry(self.skills.clone());
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

        sa = sa.with_memory(
            Some(self.blackboard.clone()),
            Some(self.prefetch.clone()),
            Some(self.scheduler.clone()),
        );
        sa = sa
            .with_policy_learning_settings(&settings.policy_learning)
            .with_learning_health_settings(&settings.learning_health);
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
    pub async fn send_supplementary_input(&self, task_iri: &str, content: &str) {
        tracing::info!(task_iri = %task_iri, "Received user supplementary input");
        self.event_bus
            .emit(task_iri, "USER_SUPPLEMENTARY_INPUT", "external", content)
            .await;
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
            settings
                .gateway
                .model_mapping
                .insert("default".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("planning".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("execution".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("analysis".to_string(), self.llm_model.clone());
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
            settings
                .gateway
                .model_mapping
                .insert("default".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("planning".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("execution".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("analysis".to_string(), self.llm_model.clone());
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
            settings
                .gateway
                .model_mapping
                .insert("default".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("planning".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("execution".to_string(), self.llm_model.clone());
            settings
                .gateway
                .model_mapping
                .insert("analysis".to_string(), self.llm_model.clone());
        }
    }
}

#[tonic::async_trait]
impl seapp::se_kernel_service_server::SeKernelService for AgentOSService {
    type ChatStreamStream = Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, Status>> + Send>>;
    type ExecuteTaskStreamStream =
        Pin<Box<dyn Stream<Item = Result<seapp::ExecutionEvent, Status>> + Send>>;

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

        let result = match sa.process_task(&req.prompt, &task_iri).await {
            Ok(result) => result,
            Err(error) => {
                TaskFinalizer::new(self.event_bus.clone())
                    .finalize_error(&task_iri, &error.to_string())
                    .await;
                return Err(Status::internal(format!("SA execution failed: {}", error)));
            }
        };
        TaskFinalizer::new(self.event_bus.clone())
            .finalize(&task_iri, &result)
            .await;

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

        let _ = tx
            .send(Ok(ChatStreamChunk {
                content: String::new(),
                done: false,
                status: "processing".to_string(),
            }))
            .await;

        match sa.process_task(&req.prompt, &task_iri).await {
            Ok(result) => {
                TaskFinalizer::new(self.event_bus.clone())
                    .finalize(&task_iri, &result)
                    .await;
                let content = extract_content(&result);

                let chunk_size = 20;
                let chars: Vec<char> = content.chars().collect();
                for chunk in chars.chunks(chunk_size) {
                    let chunk_str: String = chunk.iter().collect();
                    if tx
                        .send(Ok(ChatStreamChunk {
                            content: chunk_str,
                            done: false,
                            status: "streaming".to_string(),
                        }))
                        .await
                        .is_err()
                    {
                        return Ok(Response::new(Box::pin(
                            tokio_stream::wrappers::ReceiverStream::new(rx),
                        )));
                    }
                }

                let _ = tx
                    .send(Ok(ChatStreamChunk {
                        content: String::new(),
                        done: true,
                        status: result.status.clone(),
                    }))
                    .await;
            }
            Err(e) => {
                TaskFinalizer::new(self.event_bus.clone())
                    .finalize_error(&task_iri, &e.to_string())
                    .await;
                let _ = tx
                    .send(Ok(ChatStreamChunk {
                        content: format!("Error: {}", e),
                        done: true,
                        status: "error".to_string(),
                    }))
                    .await;
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
                            {
                                let mut states = states_clone.write().await;
                                if let Some(state) = states.get_mut(&task_iri_clone) {
                                    state.update_from_event(&core_event);
                                }
                            }
                            if !should_stream_execution_event(
                                &core_event,
                                include_thought,
                                include_tool_calls,
                            ) {
                                continue;
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

        let sa_settings = settings.clone();
        // Build from this service's shared stores before spawning. Creating a
        // new AgentOSService here used a second EventBus/Batch manager and
        // silently bypassed the runtime initialized by `init_batch_system`.
        let mut sa = self.create_sa(&sa_settings);
        let prompt = req.prompt.clone();
        let task_iri_for_task = task_iri.clone();
        let _tx_for_task = tx.clone();
        let event_bus_for_task = self.event_bus.clone();

        tokio::spawn(async move {
            let emitter = ExecutionEventEmitter::with_options(
                &task_iri_for_task,
                None,
                Some(event_bus_for_task.clone()),
                include_thought,
                include_tool_calls,
            );

            emitter.emit_phase_change("idle", "plan", "PA", "Task started");

            match sa.process_task(&prompt, &task_iri_for_task).await {
                Ok(result) => {
                    TaskFinalizer::new(event_bus_for_task.clone())
                        .finalize(&task_iri_for_task, &result)
                        .await;
                    emitter.emit_completion(&result.status, &result.summary, result.output.clone());
                }
                Err(e) => {
                    TaskFinalizer::new(event_bus_for_task.clone())
                        .finalize_error(&task_iri_for_task, &e.to_string())
                        .await;
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
        let state = states
            .get(&task_iri)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("active task not found: {task_iri}")))?;

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
        let state = states
            .get(&task_iri)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("active task not found: {task_iri}")))?;

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
            phase_history: state
                .phase_history
                .iter()
                .map(|p| PhaseHistoryEntry {
                    phase: p.phase.clone(),
                    agent_id: p.agent_id.clone(),
                    started_at: p.started_at,
                    completed_at: p.completed_at.unwrap_or(0),
                    status: p.status.clone(),
                })
                .collect(),
        };

        Ok(Response::new(status))
    }

    async fn validate_contract(
        &self,
        _request: Request<ValidateContractRequest>,
    ) -> Result<Response<ValidateContractResponse>, Status> {
        Err(Status::unimplemented(
            "contract validation is not implemented",
        ))
    }

    async fn flatten_to_frontend(
        &self,
        _request: Request<FlattenRequest>,
    ) -> Result<Response<FlattenResponse>, Status> {
        Err(Status::unimplemented(
            "frontend flattening is not implemented",
        ))
    }

    async fn submit_human_approval(
        &self,
        _request: Request<SubmitApprovalRequest>,
    ) -> Result<Response<SubmitApprovalResponse>, Status> {
        Err(Status::unimplemented(
            "human approval submission is not wired",
        ))
    }
}

fn convert_event_bus_to_grpc(
    event: &crate::core::event_bus::Event,
) -> Option<(
    crate::core::execution_event::ExecutionEvent,
    seapp::ExecutionEvent,
)> {
    use crate::core::event_bus::EventType;
    use crate::core::execution_event::ExecutionEvent as CoreExecutionEvent;

    let event_type = EventType::from_str(&event.event_type);
    let timestamp = event.timestamp.timestamp_millis();

    // Rich execution events are already serialized by AgentRunner and
    // ExecutionEventEmitter.  Preserve that payload rather than collapsing it
    // into the legacy EventType subset, which previously discarded THOUGHT,
    // TOOL_CALL and TOOL_RESULT updates from streaming clients.
    if let EventType::Custom(name) = &event_type {
        if matches!(
            name.as_str(),
            "PHASE_CHANGE"
                | "AGENT_STATUS"
                | "LLM_CONTENT"
                | "TOOL_CALL"
                | "TOOL_RESULT"
                | "THOUGHT"
                | "TOKEN_USAGE"
                | "EXECUTION_ERROR"
                | "COMPLETION"
        ) {
            let core_event = serde_json::from_str::<CoreExecutionEvent>(&event.payload).ok()?;
            // The event bus routing key is authoritative.  Do not allow an
            // embedded payload to project an execution event into another
            // task's stream.
            if core_event.task_iri != event.task_iri {
                return None;
            }
            let proto_event = seapp::ExecutionEvent {
                event_id: core_event.event_id.clone(),
                task_iri: core_event.task_iri.clone(),
                timestamp: core_event.timestamp,
                event: Some(kind_to_proto_event(core_event.event.clone())),
            };
            return Some((core_event, proto_event));
        }
    }

    let kind = match event_type {
        EventType::PlanStarted => {
            ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
                from_phase: "idle".to_string(),
                to_phase: "plan".to_string(),
                agent_role: "PA".to_string(),
                reason: "Plan phase started".to_string(),
            })
        }
        EventType::PlanCompleted => {
            ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
                from_phase: "plan".to_string(),
                to_phase: "do".to_string(),
                agent_role: "PA".to_string(),
                reason: "Plan phase completed".to_string(),
            })
        }
        EventType::DoStarted => {
            ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
                from_phase: "plan".to_string(),
                to_phase: "do".to_string(),
                agent_role: "DA".to_string(),
                reason: "Do phase started".to_string(),
            })
        }
        EventType::DoCompleted => {
            ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
                from_phase: "do".to_string(),
                to_phase: "check".to_string(),
                agent_role: "DA".to_string(),
                reason: "Do phase completed".to_string(),
            })
        }
        EventType::CheckStarted => {
            ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
                from_phase: "do".to_string(),
                to_phase: "check".to_string(),
                agent_role: "CA".to_string(),
                reason: "Check phase started".to_string(),
            })
        }
        EventType::CheckCompleted => {
            ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
                from_phase: "check".to_string(),
                to_phase: "act".to_string(),
                agent_role: "CA".to_string(),
                reason: "Check phase completed".to_string(),
            })
        }
        EventType::ActStarted => {
            ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
                from_phase: "check".to_string(),
                to_phase: "act".to_string(),
                agent_role: "AA".to_string(),
                reason: "Act phase started".to_string(),
            })
        }
        EventType::ActCompleted => {
            ExecutionEventKind::PhaseChange(crate::core::execution_event::PhaseChange {
                from_phase: "act".to_string(),
                to_phase: "completed".to_string(),
                agent_role: "AA".to_string(),
                reason: "Act phase completed".to_string(),
            })
        }
        EventType::AgentStarted => {
            ExecutionEventKind::AgentStatus(crate::core::execution_event::AgentStatus {
                agent_id: event.source_agent_iri.clone(),
                role: "unknown".to_string(),
                status: "running".to_string(),
                turn: 0,
                iteration: 0,
                timestamp: None,
            })
        }
        EventType::AgentCompleted => {
            ExecutionEventKind::AgentStatus(crate::core::execution_event::AgentStatus {
                agent_id: event.source_agent_iri.clone(),
                role: "unknown".to_string(),
                status: "completed".to_string(),
                turn: 0,
                iteration: 0,
                timestamp: None,
            })
        }
        EventType::AgentError => ExecutionEventKind::Error(crate::core::execution_event::Error {
            error_type: "AgentError".to_string(),
            message: event.payload.clone(),
            agent_id: event.source_agent_iri.clone(),
            recoverable: false,
        }),
        EventType::TaskCompleted => {
            ExecutionEventKind::Completion(crate::core::execution_event::Completion {
                status: "success".to_string(),
                summary: event.payload.clone(),
                total_turns: 0,
                total_tool_calls: 0,
                total_tokens: 0,
                output_json: None,
            })
        }
        EventType::TaskFailed => {
            ExecutionEventKind::Completion(crate::core::execution_event::Completion {
                status: "failed".to_string(),
                summary: event.payload.clone(),
                total_turns: 0,
                total_tool_calls: 0,
                total_tokens: 0,
                output_json: None,
            })
        }
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

fn should_stream_execution_event(
    event: &crate::core::execution_event::ExecutionEvent,
    include_thought: bool,
    include_tool_calls: bool,
) -> bool {
    match &event.event {
        ExecutionEventKind::Thought(_) => include_thought,
        ExecutionEventKind::LlmContent(content) if content.is_reasoning => include_thought,
        ExecutionEventKind::ToolCall(_) | ExecutionEventKind::ToolResult(_) => include_tool_calls,
        _ => true,
    }
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
            output_json: c
                .output_json
                .map(|v| serde_json::to_string(&v).unwrap_or_default())
                .unwrap_or_default(),
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
    let cleaned = re
        .map(|r| r.replace_all(text, "").to_string())
        .unwrap_or_else(|| text.to_string());
    let cleaned = cleaned.trim().to_string();
    if cleaned.is_empty() {
        text.to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serialized_execution_bus_event(
        kind: ExecutionEventKind,
        event_type: &str,
    ) -> crate::core::event_bus::Event {
        let execution_event = crate::core::execution_event::ExecutionEvent {
            event_id: "evt-projected".into(),
            task_iri: "iri://task/projected".into(),
            timestamp: 42,
            event: kind,
        };
        let payload = serde_json::to_string(&execution_event).unwrap();
        crate::core::event_bus::Event {
            event_id: "bus-projected".into(),
            task_iri: "iri://task/projected".into(),
            event_type: event_type.into(),
            source_agent_iri: "agent://do".into(),
            payload_json_ld: payload.clone(),
            payload,
            timestamp: chrono::Utc::now(),
            sequence: 1,
            type_mask: Default::default(),
            priority: crate::core::event_bus::EventPriority::Normal,
        }
    }

    #[test]
    fn grpc_projection_preserves_serialized_tool_execution_events() {
        let bus_event = serialized_execution_bus_event(
            ExecutionEventKind::ToolResult(crate::core::execution_event::ToolResult {
                call_id: "call-1".into(),
                tool_name: "file_write".into(),
                result: "{\"changed\":true}".into(),
                success: true,
                result_size_bytes: 16,
                duration_ms: 23,
                agent_id: "DA".into(),
            }),
            "TOOL_RESULT",
        );

        let (core, proto) = convert_event_bus_to_grpc(&bus_event).unwrap();
        assert!(matches!(core.event, ExecutionEventKind::ToolResult(_)));
        assert!(proto.event.is_some());
        assert!(should_stream_execution_event(&core, false, true));
        assert!(!should_stream_execution_event(&core, false, false));
    }

    #[test]
    fn grpc_projection_filters_reasoning_without_dropping_state_events() {
        let bus_event = serialized_execution_bus_event(
            ExecutionEventKind::LlmContent(crate::core::execution_event::LlmContent {
                agent_id: "DA".into(),
                role: "Do".into(),
                content_delta: "private reasoning".into(),
                is_reasoning: true,
                token_count: 3,
            }),
            "LLM_CONTENT",
        );
        let (core, _) = convert_event_bus_to_grpc(&bus_event).unwrap();
        assert!(!should_stream_execution_event(&core, false, true));
        assert!(should_stream_execution_event(&core, true, false));
    }

    #[test]
    fn root_grpc_bootstrap_populates_missing_registry_skills_without_overwrite() {
        let registry = SkillRegistry::new();
        let graph = SkillGraphStore::new();
        let first = registry.list_all_skills().into_iter().next().unwrap();
        let mut evolved = crate::skill_graph::types::SkillGraphNode::from_skill_meta(&first);
        evolved.description = "persisted evolution must survive bootstrap".to_string();
        graph.register_skill(evolved).unwrap();

        let inserted = bootstrap_skill_graph_from_registry(&registry, &graph).unwrap();
        assert_eq!(inserted, registry.list_all_skills().len() - 1);
        assert_eq!(
            graph.list_all_skills().len(),
            registry.list_all_skills().len()
        );
        assert_eq!(
            graph.get_skill(&first.skill_iri).unwrap().description,
            "persisted evolution must survive bootstrap"
        );
    }

    #[test]
    fn root_grpc_unified_graph_path_is_scoped_to_l0_directory() {
        let dir = tempfile::tempdir().unwrap();
        let mut settings = Settings::default();
        settings.memory.l0.path = dir.path().join("l0").to_string_lossy().to_string();
        assert_eq!(
            unified_graph_path(&settings),
            dir.path().join("l0/unified-graph")
        );
    }

    #[test]
    fn evict_stale_l2_line_evicts_clean_but_keeps_dirty() {
        let bb = Blackboard::new().unwrap();
        let config = CoreConfig::default();
        let json_ld = r#"{"@id":"iri://test/1","@type":"Test"}"#;

        // Clean line: first write leaves dirty=false → evictable.
        bb.write_node("iri://test/clean", json_ld, &config).unwrap();
        assert!(evict_stale_l2_line(&bb, "iri://test/clean"));
        assert!(bb.read_node("iri://test/clean").unwrap().is_none());

        // Dirty line: second write marks dirty=true → must be retained.
        bb.write_node("iri://test/dirty", json_ld, &config).unwrap();
        bb.write_node("iri://test/dirty", json_ld, &config).unwrap();
        assert!(!evict_stale_l2_line(&bb, "iri://test/dirty"));
        assert!(bb.read_node("iri://test/dirty").unwrap().is_some());

        // Absent line: no-op, returns false.
        assert!(!evict_stale_l2_line(&bb, "iri://test/absent"));
    }

    #[test]
    fn cache_invalidate_payload_extracts_node_iri_with_fallback() {
        let payload = r#"{"node_iri":"iri://node/42"}"#;
        assert_eq!(
            node_iri_from_invalidate_event(payload, "iri://task/fallback"),
            "iri://node/42".to_string()
        );
        assert_eq!(
            node_iri_from_invalidate_event("not-json", "iri://task/fallback"),
            "iri://task/fallback".to_string()
        );
        assert_eq!(
            node_iri_from_invalidate_event(r#"{"other":"x"}"#, "iri://task/fallback"),
            "iri://task/fallback".to_string()
        );
    }
}
