use std::sync::Arc;
use std::time::Instant;

use tracing::{error, info, warn};

use crate::config::GatewaySettings;
use crate::core::EventBus;
use crate::core::{AgentRunner, CoreConfig, SupervisorAgent};
use crate::gateway::UnifiedGateway;
use crate::memory::consistency_engine::ConsistencyEngine;
use crate::memory::memory_bus::MemoryBus;
use crate::memory::prefetch_engine::PrefetchEngine;
use crate::memory::scheduler::MemoryScheduler;
use crate::memory::{Blackboard, L0Store, MemoryManager, ProjectionEngine};
use crate::templates::TemplateEngine;
use crate::tools::hooks::{
    ApprovalCondition, ApprovalPoint, ChannelApprovalNotifier, HookManager, HumanApprovalConfig,
    HumanApprovalHook,
};
use crate::tools::workspace_monitor::{WorkspaceMonitor, WorkspaceMonitorConfig};
use crate::tools::SkillRegistry;

use super::task_queue::{AgentOsResult, AgentOsTask, QueueError, WorkerQueue};

/// Agent OS Worker Configuration
#[derive(Clone)]
pub struct WorkerConfig {
    /// Queue base path
    pub queue_base_path: String,
    /// L0 storage path
    pub l0_path: String,
    /// Concurrency level
    pub concurrency: usize,
    /// LLM gateway configuration
    pub gateway: Option<GatewaySettings>,
    /// Human Approval configuration
    pub approval_config: Option<HumanApprovalConfig>,
    /// Workspace root directory (optional)
    pub workspace_root: Option<String>,
    /// Event bus capacity
    pub event_bus_capacity: usize,
    /// Causal engine for root-cause analysis (optional — when set, enables dimension audit observations)
    pub causal_engine: Option<std::sync::Arc<crate::causal::CausalEngine>>,
    /// Skill graph store — the cognitive network (optional — when set, wired into tools and runner)
    pub skill_graph_store: Option<std::sync::Arc<crate::skill_graph::graph_store::SkillGraphStore>>,
}

impl std::fmt::Debug for WorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerConfig")
            .field("queue_base_path", &self.queue_base_path)
            .field("l0_path", &self.l0_path)
            .field("concurrency", &self.concurrency)
            .field("gateway", &self.gateway)
            .field("approval_config", &self.approval_config)
            .field("workspace_root", &self.workspace_root)
            .field("event_bus_capacity", &self.event_bus_capacity)
            .field(
                "causal_engine",
                &self.causal_engine.as_ref().map(|_| "Some(...)"),
            )
            .field(
                "skill_graph_store",
                &self.skill_graph_store.as_ref().map(|_| "Some(...)"),
            )
            .finish()
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            queue_base_path: "./data/agent_os_queue".to_string(),
            l0_path: "./data/l0".to_string(),
            concurrency: 4,
            gateway: None,
            approval_config: None,
            workspace_root: None,
            event_bus_capacity: 100,
            causal_engine: None,
            skill_graph_store: None,
        }
    }
}

impl WorkerConfig {
    pub fn from_env() -> Self {
        let gateway = std::env::var("ONE_API_URL").ok().map(|base_url| {
            let api_key = std::env::var("ONE_API_KEY").unwrap_or_default();
            GatewaySettings {
                base_url,
                api_key,
                default_model: "deepseek-v4-flash".to_string(),
                timeout_seconds: 300,
                max_retries: 3,
                retry_base_ms: 500,
                use_responses_api: false,
                model_mapping: Default::default(),
            }
        });

        let approval_config = if std::env::var("AGENT_OS_APPROVAL_ENABLED")
            .ok()
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false)
        {
            Some(HumanApprovalConfig {
                enabled: true,
                approval_points: vec![ApprovalPoint {
                    hook_point: crate::tools::hooks::HookPoint::PhaseEnd,
                    condition: ApprovalCondition::OnStageComplete,
                    message_template: "Phase {stage} completed, please confirm whether to continue"
                        .to_string(),
                    timeout_seconds: std::env::var("AGENT_OS_APPROVAL_TIMEOUT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(3600),
                    default_action: crate::tools::hooks::DefaultAction::Approve,
                    stages: Vec::new(),
                }],
                default_timeout_seconds: 3600,
                default_action: crate::tools::hooks::DefaultAction::Approve,
            })
        } else {
            None
        };

        Self {
            queue_base_path: std::env::var("AGENT_OS_QUEUE_PATH")
                .unwrap_or_else(|_| "./data/agent_os_queue".to_string()),
            l0_path: std::env::var("AGENT_OS_L0_PATH").unwrap_or_else(|_| "./data/l0".to_string()),
            concurrency: std::env::var("AGENT_OS_CONCURRENCY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4),
            workspace_root: std::env::var("AGENT_OS_WORKSPACE_ROOT").ok(),
            gateway,
            approval_config,
            event_bus_capacity: 100,
            causal_engine: None,
            skill_graph_store: None,
        }
    }
}

/// Agent OS Worker
pub struct AgentOsWorker {
    config: WorkerConfig,
    queue: WorkerQueue,
    sa: SupervisorAgent,
    approval_notifier: Option<Arc<ChannelApprovalNotifier>>,
    prefetch_engine: Option<Arc<PrefetchEngine>>,
    blackboard: Arc<Blackboard>,
    /// Event bus for cross-component communication
    pub event_bus: Arc<EventBus>,
}

impl AgentOsWorker {
    /// Create a new Worker
    pub fn new(config: WorkerConfig) -> Result<Self, QueueError> {
        let queue = WorkerQueue::new(&config.queue_base_path)?;

        let l0 = Arc::new(
            L0Store::new(&config.l0_path)
                .map_err(|e| QueueError::Queue(format!("Failed to create L0: {}", e)))?,
        );

        let blackboard = Arc::new(
            Blackboard::new()
                .map_err(|e| QueueError::Queue(format!("Failed to create Blackboard: {}", e)))?,
        );

        let gateway_settings = config.gateway.clone().unwrap_or_else(|| GatewaySettings {
            base_url: std::env::var("DEEPSEEK_API_URL")
                .or_else(|_| std::env::var("ONE_API_URL"))
                .unwrap_or_else(|_| "https://api.deepseek.com".to_string()),
            api_key: std::env::var("DEEPSEEK_API_KEY")
                .or_else(|_| std::env::var("ONE_API_KEY"))
                .unwrap_or_default(),
            default_model: "deepseek-v4-flash".to_string(),
            timeout_seconds: 300,
            max_retries: 3,
            retry_base_ms: 500,
            use_responses_api: false,
            model_mapping: Default::default(),
        });

        let gateway = Arc::new(
            UnifiedGateway::new(&gateway_settings)
                .map_err(|e| QueueError::Queue(format!("Failed to create Gateway: {}", e)))?,
        );

        let templates_dir = std::env::temp_dir();
        let templates_engine =
            Arc::new(TemplateEngine::new(&templates_dir).map_err(|e| {
                QueueError::Queue(format!("Failed to create template engine: {}", e))
            })?);

        let skills = Arc::new(SkillRegistry::new());

        let projection_engine = Arc::new(ProjectionEngine::new(blackboard.clone(), 500));

        // 主事件总线提前创建,让 memory_bus 与上层共享同一总线(修复独立总线导致预取/一致性事件无人消费的缺陷)
        let event_bus = Arc::new(EventBus::new(config.event_bus_capacity));

        let memory_bus = Arc::new(MemoryBus::new(event_bus.clone()));
        let consistency = Arc::new(ConsistencyEngine::new(
            memory_bus.clone(),
            l0.clone(),
            blackboard.clone(),
            projection_engine.clone(),
        ));
        let scheduler = Arc::new(MemoryScheduler::new(
            l0.clone(),
            blackboard.clone(),
            projection_engine.clone(),
            consistency.clone(),
            memory_bus.clone(),
        ));
        let prefetch = Arc::new(PrefetchEngine::new(
            memory_bus.clone(),
            blackboard.clone(),
            projection_engine.clone(),
        ));

        let memory_manager = Arc::new(tokio::sync::Mutex::new(MemoryManager::with_scheduler(
            l0.clone(),
            blackboard.clone(),
            projection_engine,
            CoreConfig::default(),
            scheduler.clone(),
        )));

        let hook_manager = HookManager::new();
        let mut approval_notifier = None;

        if let Some(ref approval_cfg) = config.approval_config {
            if approval_cfg.enabled {
                let (hook, notifier) =
                    HumanApprovalHook::with_channel_notifier(approval_cfg.clone());
                hook_manager.register(hook);
                approval_notifier = Some(notifier);
                info!("HumanApprovalHook registered");
            }
        }

        // Initialize WorkspaceMonitor (if workspace root is configured)
        let workspace_root_path: Option<std::path::PathBuf> = config
            .workspace_root
            .as_ref()
            .map(|s| std::path::PathBuf::from(s));
        let workspace_monitor_opt: Option<Arc<WorkspaceMonitor>> = if let Some(ref ws_root) =
            workspace_root_path
        {
            let ws_config = WorkspaceMonitorConfig {
                workspace_root: ws_root.clone(),
                ..Default::default()
            };
            match WorkspaceMonitor::initialize(ws_config, Some(blackboard.clone()), None) {
                Ok(ws) => {
                    ws.register_hooks(&hook_manager);
                    info!(root = %ws_root.display(), "WorkspaceMonitor initialized");
                    Some(Arc::new(ws))
                }
                Err(e) => {
                    warn!("WorkspaceMonitor initialization failed: {}, using default workspace settings", e);
                    None
                }
            }
        } else {
            None
        };

        let mut runner_builder = AgentRunner::new(
            gateway,
            skills.clone(),
            blackboard.clone(),
            l0,
            memory_manager,
            templates_engine.clone(),
            crate::config::AgentSettings::default(),
        )
        .with_hook_manager(hook_manager);
        if let Some(ref ws_root) = workspace_root_path {
            runner_builder = runner_builder.with_workspace_root(ws_root.clone());
        }
        // Wire optional advanced subsystems BEFORE wrapping in Arc
        if let Some(ref ce) = config.causal_engine {
            runner_builder = runner_builder.with_causal_engine(ce.clone());
        }
        if let Some(ref sg) = config.skill_graph_store {
            runner_builder = runner_builder.with_skill_graph_store(sg.clone());
        }
        let runner = Arc::new(runner_builder);

        // Wire shared SkillGraphStore into ToolExecutor (via Arc interior lock)
        if let Some(ref sg) = config.skill_graph_store {
            runner
                .tool_executor
                .write()
                .set_shared_skill_graph(sg.clone());
        }

        // Set workspace_monitor on ToolExecutor
        if let Some(ref wm) = workspace_monitor_opt {
            runner
                .tool_executor
                .write()
                .set_workspace_monitor(wm.clone());
        }

        // Finalize AgentRunner initialization wiring: perception_store → WorkspaceMonitor
        runner.finalize_setup();

        // Capture the runner's perception store so it can be shared with the SA
        let runner_perception = runner.perception_store.clone();

        let sa = SupervisorAgent::new(runner, templates_engine, skills, event_bus.clone(), 20)
            .with_memory(
                Some(blackboard.clone()),
                Some(prefetch.clone()),
                Some(scheduler),
            )
            .with_perception_store(Arc::new(runner_perception))
            .with_execution_timeout(600);

        Ok(Self {
            config,
            queue,
            sa,
            approval_notifier,
            prefetch_engine: Some(prefetch),
            blackboard,
            event_bus,
        })
    }

    /// Get the approval notifier (for external approval submission)
    pub fn approval_notifier(&self) -> Option<&Arc<ChannelApprovalNotifier>> {
        self.approval_notifier.as_ref()
    }

    /// Run the Worker main loop
    pub async fn run(&mut self) -> Result<(), QueueError> {
        info!(
            queue_path = %self.config.queue_base_path,
            concurrency = self.config.concurrency,
            approval_enabled = self.approval_notifier.is_some(),
            "Agent OS Worker started"
        );

        if let Some(pf) = self.prefetch_engine.clone() {
            pf.spawn_consumer(self.event_bus.clone(), self.blackboard.clone());
        }

        loop {
            match self.queue.recv_task().await {
                Ok(task) => {
                    info!(task_id = %task.task_id, task_iri = %task.task_iri, "Task received");

                    let result = self.execute_task(task).await;

                    if let Err(e) = self.queue.send_result(&result).await {
                        error!(error = %e, "Failed to send result");
                    }
                }
                Err(e) => {
                    error!(error = %e, "Failed to receive task");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    }

    /// Execute a single task
    async fn execute_task(&mut self, task: AgentOsTask) -> AgentOsResult {
        let start = Instant::now();
        let original_task_id = task.task_id.clone();

        info!(task_id = %original_task_id, "Starting task execution");

        match self.sa.process_task(&task.prompt, &task.task_iri).await {
            Ok(task_result) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                info!(
                    task_id = %original_task_id,
                    status = %task_result.status,
                    duration_ms = duration_ms,
                    "Task execution completed"
                );

                let mut result = AgentOsResult::from(task_result);
                result.task_id = original_task_id;
                result.duration_ms = duration_ms;
                result
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                error!(task_id = %original_task_id, error = %e, duration_ms = duration_ms, "Task execution failed");

                AgentOsResult {
                    task_id: original_task_id,
                    status: "failed".to_string(),
                    summary: format!("Task execution failed: {}", e),
                    output: None,
                    jsonld_output: None,
                    artifacts: Vec::new(),
                    errors: vec![e.to_string()],
                    duration_ms,
                    tool_call_count: 0,
                    turn_count: 0,
                }
            }
        }
    }
}

/// Helper function to start a Worker
pub async fn run_worker(config: WorkerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut worker = AgentOsWorker::new(config)?;
    worker.run().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_worker_creation() {
        let temp_dir = TempDir::new().unwrap();

        let config = WorkerConfig {
            queue_base_path: temp_dir.path().join("queue").to_str().unwrap().to_string(),
            l0_path: temp_dir.path().join("l0").to_str().unwrap().to_string(),
            ..Default::default()
        };

        let worker = AgentOsWorker::new(config);
        assert!(worker.is_ok());
    }

    #[test]
    fn test_worker_with_approval() {
        let temp_dir = TempDir::new().unwrap();

        let config = WorkerConfig {
            queue_base_path: temp_dir.path().join("queue").to_str().unwrap().to_string(),
            l0_path: temp_dir.path().join("l0").to_str().unwrap().to_string(),
            approval_config: Some(HumanApprovalConfig {
                enabled: true,
                approval_points: vec![ApprovalPoint::default()],
                default_timeout_seconds: 3600,
                default_action: crate::tools::hooks::DefaultAction::Approve,
            }),
            ..Default::default()
        };

        let worker = AgentOsWorker::new(config);
        assert!(worker.is_ok());
        assert!(worker.unwrap().approval_notifier.is_some());
    }
}
