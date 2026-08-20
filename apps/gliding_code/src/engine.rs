use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use glidinghorse::causal::engine::CausalEngine;
use glidinghorse::causal::fused::FusedRootCauseEngine;
use glidinghorse::causal::store::CausalModelStore;
use glidinghorse::config::{McpServerConfig, McpStdioServerConfig};
use glidinghorse::core::agent_runner::TaskResult;
use glidinghorse::core::event_bus::{Event, EventBus};
use glidinghorse::core::sa::SupervisorAgent;
use glidinghorse::gateway::UnifiedGateway;
use glidinghorse::graph_backend::{GraphBackend, PetgraphBackend, SkillGraphSnapshotBackend};
use glidinghorse::graph_features::features::FeatureExtractor;
use glidinghorse::knowledge_graph::code_ast::CodeAstExtractor;
use glidinghorse::knowledge_graph::store::KnowledgeGraphStore;
use glidinghorse::memory::embedding_service::{
    create_embedding_service_from_config, FallbackEmbeddingService,
};
use glidinghorse::memory::hyperspace_store::HyperspaceStore;
use glidinghorse::memory::l0_store::L0Store;
use glidinghorse::memory::l1_session::EvictionConfig;
use glidinghorse::memory::l2_blackboard::Blackboard;
use glidinghorse::memory::l3_projection::ProjectionEngine;
use glidinghorse::memory::memory_manager::MemoryManager;
use glidinghorse::memory::unified_graph::UnifiedGraphStore;
use glidinghorse::ontology_bridge::{
    FallbackStructuralEmbeddingService, LinearCrossSpaceProjection,
    LinearCrossSpaceProjectionConfig, OntologyBridgeConfig, OntologyBridgeManager,
};
use glidinghorse::skill_graph::discovery::SkillDiscoveryEngine;
use glidinghorse::skill_graph::evolution::{EvolutionProposalStore, SkillEvolutionEngine};
use glidinghorse::skill_graph::graph_algorithms::SkillGraphAlgorithms;
use glidinghorse::skill_graph::graph_store::SkillGraphStore;
use glidinghorse::skill_graph::security::SecurityEngine;
use glidinghorse::skill_graph::{LinkStrength, SkillLinkType};
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
    mcp_client: Option<Arc<tokio::sync::Mutex<Option<McpClient>>>>,
    workspace_monitor: Option<Arc<WorkspaceMonitor>>,
    /// Skill Graph Store — cognitive network
    skill_graph: Arc<SkillGraphStore>,
    /// Skill discovery engine (semantic search)
    discovery_engine: Arc<SkillDiscoveryEngine>,
    /// Set after the persisted skill graph has been indexed into Hyperspace.
    /// Kept false after a failed attempt so the next task can retry.
    skill_vectors_ready: AtomicBool,
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
    /// Unified Oxigraph store reference for auto-code-analysis et al.
    oxi_store: Arc<oxigraph::store::Store>,
    /// Stable code-scan exclusions compiled from built-ins, workspace settings,
    /// and the supported subset of the workspace `.gitignore`.
    code_scan_exclude_patterns: Vec<String>,
}

const DEFAULT_CODE_SCAN_EXCLUSIONS: &[&str] = &[
    ".git/",
    ".hg/",
    ".svn/",
    "target/",
    "node_modules/",
    "dist/",
    "build/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".next/",
    ".gliding_horse/",
];

fn load_code_scan_exclusions(root: &std::path::Path, configured: &[String]) -> Vec<String> {
    let mut patterns: Vec<String> = DEFAULT_CODE_SCAN_EXCLUSIONS
        .iter()
        .map(|pattern| (*pattern).to_string())
        .collect();
    for pattern in configured {
        if !pattern.trim().is_empty() && !patterns.contains(pattern) {
            patterns.push(pattern.clone());
        }
    }

    // This intentionally shares WorkspaceMonitor's documented, small
    // gitignore-compatible subset rather than claiming full gitignore support.
    if let Ok(contents) = std::fs::read_to_string(root.join(".gitignore")) {
        for line in contents.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
                continue;
            }
            let pattern = line.strip_prefix('/').unwrap_or(line);
            let pattern = if !pattern.contains('.') && !pattern.ends_with('/') {
                format!("{pattern}/")
            } else {
                pattern.to_string()
            };
            if !patterns.contains(&pattern) {
                patterns.push(pattern);
            }
        }
    }
    patterns
}

fn code_path_is_excluded(
    relative_path: &std::path::Path,
    is_directory: bool,
    patterns: &[String],
) -> bool {
    let mut path = relative_path.to_string_lossy().replace('\\', "/");
    if is_directory && !path.ends_with('/') {
        path.push('/');
    }
    patterns.iter().any(|pattern| {
        glidinghorse::tools::workspace_monitor::inventory::match_glob_pattern(&path, pattern)
    })
}

fn collect_workspace_code_files(
    root: &std::path::Path,
    exclusion_patterns: &[String],
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let supported = |ext: &str| {
        matches!(
            ext,
            "rs" | "py"
                | "pyi"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "go"
                | "java"
                | "c"
                | "cpp"
                | "h"
                | "hpp"
        )
    };
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        let mut entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries.collect::<Result<Vec<_>, _>>().map_err(|error| {
                anyhow::anyhow!(
                    "Unable to scan workspace directory '{}': {error}",
                    dir.display()
                )
            })?,
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "Unable to scan workspace directory '{}': {error}",
                    dir.display()
                ))
            }
        };
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let file_type = entry.file_type().map_err(|error| {
                anyhow::anyhow!(
                    "Unable to inspect workspace path '{}': {error}",
                    path.display()
                )
            })?;
            if file_type.is_symlink() {
                continue;
            }
            let relative_path = path.strip_prefix(root).unwrap_or(path.as_path());
            if file_type.is_dir() {
                if !code_path_is_excluded(relative_path, true, exclusion_patterns) {
                    pending.push(path);
                }
            } else if file_type.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(supported)
                && !code_path_is_excluded(relative_path, false, exclusion_patterns)
            {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
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
        std::env::set_current_dir(&workspace_abs).map_err(|e| {
            anyhow::anyhow!("无法切换到工作目录 '{}': {}", workspace_abs.display(), e)
        })?;

        let gateway = Arc::new(UnifiedGateway::new(&config.gateway)?);
        let dir = tempfile::TempDir::new()?;

        // A configured data directory is a shared root, not a workspace
        // identity. Keep state from separate repositories isolated below a
        // stable, non-path-leaking hash of the canonical workspace.
        let workspace_namespace = {
            let digest = Sha256::digest(config.workspace.as_bytes());
            format!("workspace-{}", hex::encode(&digest[..12]))
        };
        let persistent_root = config
            .data_dir
            .as_ref()
            .map(|base| {
                let root = std::path::Path::new(base).join(&workspace_namespace);
                std::fs::create_dir_all(&root)?;
                Ok::<_, std::io::Error>(root)
            })
            .transpose()?;

        let l0_path = persistent_root
            .as_ref()
            .map(|d| d.join("l0").to_string_lossy().to_string())
            .unwrap_or_else(|| dir.path().join("l0").to_string_lossy().to_string());

        let l0 = Arc::new(
            L0Store::new(&l0_path).map_err(|e| anyhow::anyhow!("L0Store 创建失败: {}", e))?,
        );

        // ── Unified Oxigraph Store — shared across Blackboard, SkillGraphStore,
        //    ToolExecutor (KnowledgeGraphStore), and KnowledgeBridge so that all
        //    subsystems operate on the same RDF store via named-graph isolation.
        let unified = Arc::new(
            match &persistent_root {
                Some(root) => UnifiedGraphStore::new_persistent(root.join("unified-graph")),
                None => UnifiedGraphStore::new(),
            }
            .map_err(|e| anyhow::anyhow!("UnifiedGraphStore 创建失败: {}", e))?,
        );

        let l2 = Arc::new(
            Blackboard::with_store(unified.store())
                .map_err(|e| anyhow::anyhow!("Blackboard 创建失败: {}", e))?,
        );

        // Load agent-os config (config.yaml + AGENT_OS_* env vars) for tunable
        // parameters; fall back to Defaults when no config file is present.
        let loaded_settings = glidinghorse::config::Settings::load().ok();
        let settings = loaded_settings.clone().unwrap_or_default();

        // Initialize HyperspaceEngine-backed vector store for semantic search
        let embed: Arc<dyn glidinghorse::memory::embedding_service::EmbeddingService> =
            match &loaded_settings {
                Some(s) => create_embedding_service_from_config(
                    &s.embedding,
                    s.agents.embedding_timeout_secs,
                ),
                None => Arc::new(FallbackEmbeddingService::new()),
            };
        let hyperspace_path = persistent_root
            .as_ref()
            .map(|d| d.join("hyperspace").to_string_lossy().to_string())
            .unwrap_or_else(|| dir.path().join("hyperspace").to_string_lossy().to_string());
        let _ = std::fs::create_dir_all(&hyperspace_path);
        let vector_store = Arc::new(
            HyperspaceStore::open(std::path::Path::new(&hyperspace_path), embed)
                .map_err(|e| anyhow::anyhow!("HyperspaceStore 初始化失败: {}", e))?,
        );

        // ── OntologyBridge: dual-space embedding store (text Cosine + struct Poincaré) ──
        // Uses separate data directories alongside the main HyperspaceStore.
        let ontology_base = std::path::Path::new(&hyperspace_path)
            .parent()
            .map(|p| p.join("ontology"))
            .unwrap_or_else(|| {
                let mut p = std::path::PathBuf::from(&hyperspace_path);
                p.pop();
                p.join("ontology")
            });
        let _ = std::fs::create_dir_all(&ontology_base);
        let text_dir = ontology_base.join("text");
        let struct_dir = ontology_base.join("struct");
        // Reuse the same text embedding service; structural uses its own.
        let struct_embed_svc: Arc<dyn glidinghorse::ontology_bridge::StructuralEmbeddingService> =
            Arc::new(FallbackStructuralEmbeddingService::new());
        let mut ontology_bridge = OntologyBridgeManager::open(OntologyBridgeConfig {
            text_dir,
            struct_dir,
            embed: match &loaded_settings {
                Some(s) => create_embedding_service_from_config(
                    &s.embedding,
                    s.agents.embedding_timeout_secs,
                ),
                None => Arc::new(FallbackEmbeddingService::new()),
            },
            struct_embed: struct_embed_svc,
        })
        .map_err(|e| anyhow::anyhow!("OntologyBridge 初始化失败: {}", e))?;
        let projection_path = ontology_base.join("projection.json");
        if projection_path.exists() {
            let raw = std::fs::read_to_string(&projection_path)
                .map_err(|e| anyhow::anyhow!("读取 Ontology projection 配置失败: {}", e))?;
            let config: LinearCrossSpaceProjectionConfig = serde_json::from_str(&raw)
                .map_err(|e| anyhow::anyhow!("解析 Ontology projection 配置失败: {}", e))?;
            let projection = LinearCrossSpaceProjection::from_config(config)
                .map_err(|e| anyhow::anyhow!("Ontology projection 配置无效: {}", e))?;
            ontology_bridge = ontology_bridge.with_cross_space_projection(Arc::new(projection));
            info!(path = %projection_path.display(), "Ontology cross-space projection loaded");
        }
        let ontology_bridge = Arc::new(ontology_bridge);
        info!("OntologyBridge initialised (text + structural engines)");

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
        // Attach OntologyBridge to MemoryManager for dual-space embedding access.
        {
            let mut mm_lock = mm.blocking_lock();
            mm_lock.set_ontology_bridge(ontology_bridge.clone());
        }
        let mm_for_runner = mm.clone();

        let templates_dir = dir.path().join("templates");
        std::fs::create_dir_all(&templates_dir)?;
        let tmpl = Arc::new(
            TemplateEngine::new(&templates_dir)
                .map_err(|e| anyhow::anyhow!("TemplateEngine 创建失败: {}", e))?,
        );

        let skills = Arc::new(SkillRegistry::new());
        let skills_for_engine = skills.clone();

        // Load external skills from --skill-dir (skills/*/skill.jsonld)
        // before the bootstrap loop so they join the skill graph at startup.
        if let Some(skill_dir) = config.skill_dir.as_deref() {
            match skills.load_from_jsonld_dir(skill_dir) {
                Ok(count) => info!(count, path = %skill_dir, "Loaded external skills from --skill-dir"),
                Err(error) => warn!(path = %skill_dir, %error, "Failed to load skills from --skill-dir"),
            }
        }

        let workspace_root = std::path::PathBuf::from(&config.workspace);
        let code_scan_exclude_patterns =
            load_code_scan_exclusions(&workspace_root, &settings.workspace.exclude_patterns);

        // ── TimelineStore (temporal event recording for graph mutations) ──
        // Created before SkillGraphStore so the store can attach it and record
        // every structural mutation (otherwise TL: pending stays at 0).
        let timeline = Arc::new(
            TimelineStore::new(
                agent_settings.snapshot_frequency,
                agent_settings.max_full_snapshots,
            )
            .with_l0_store(l0.clone()),
        );
        if let Err(error) = timeline.load_persisted() {
            tracing::warn!(%error, "Failed to load persisted timeline snapshots");
        }

        // ── Skill Graph Store — cognitive network ──
        let skill_graph = Arc::new(
            SkillGraphStore::new()
                .with_blackboard(l2.clone())
                .with_l0_store(l0.clone())
                .with_oxi_store(unified.store())
                .with_timeline(timeline.clone()),
        );

        if let Err(error) = skill_graph.hydrate_from_l0() {
            tracing::warn!(%error, "Failed to hydrate persisted skill graph; continuing with bootstrap skills");
        }

        // Bootstrap the SkillGraphStore with default skills from SkillRegistry
        // so that SG: N E (node/edge) metrics are non-zero from startup.
        for meta in skills.list_all_skills() {
            if skill_graph.get_skill(&meta.skill_iri).is_some() {
                continue;
            }
            if let Err(e) = skill_graph.register_skill(
                glidinghorse::skill_graph::types::SkillGraphNode::from_skill_meta(&meta),
            ) {
                tracing::warn!("Failed to register bootstrap skill {}: {}", meta.name, e);
            }
        }

        // Resolve a process interruption after a governed AddLink commit only
        // after the persisted graph has been hydrated and bootstrap nodes have
        // been restored. This never creates new proposals or auto-approves
        // them; it only finalizes/compensates a previously durable Applying
        // record.
        match EvolutionProposalStore::new(l0.clone()).recover_inflight(skill_graph.as_ref()) {
            Ok(recovery) if recovery.committed + recovery.rolled_back + recovery.failed > 0 => {
                info!(
                    committed = recovery.committed,
                    rolled_back = recovery.rolled_back,
                    failed = recovery.failed,
                    "Recovered in-flight evolution proposals"
                );
            }
            Ok(_) => {}
            Err(error) => warn!(%error, "Failed to recover in-flight evolution proposals"),
        }

        // ── Auto-create Related links between skills sharing skill_types ──
        // This gives the skill graph non-zero edge count (SG: N E) from startup
        // and makes the cognitive network navigable from the beginning.
        {
            let nodes = skill_graph.list_all_skills();
            let mut link_count = 0usize;
            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    let a = &nodes[i];
                    let b = &nodes[j];
                    let shared: Vec<&str> = a
                        .tags
                        .iter()
                        .filter_map(|t| {
                            if b.tags.contains(t) {
                                Some(t.as_str())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !shared.is_empty() {
                        let strength = if shared.len() >= 2 {
                            LinkStrength::Recommended
                        } else {
                            LinkStrength::Navigation
                        };
                        let desc = format!("Related via: {}", shared.join(", "));
                        let _ = skill_graph.add_link(
                            &a.skill_iri,
                            &b.skill_iri,
                            SkillLinkType::Related,
                            strength,
                            &desc,
                        );
                        link_count += 1;
                    }
                }
            }
            if link_count > 0 {
                info!(
                    link_count = link_count,
                    "Auto-created skill graph edges from shared types"
                );
            }
        }
        let skill_graph_algorithms = Arc::new(SkillGraphAlgorithms::from_store(&skill_graph));

        // ── PetgraphBackend (structural dimension for FusedRootCauseEngine) ──
        let graph_backend: Arc<dyn GraphBackend> =
            Arc::new(PetgraphBackend::new(skill_graph.clone()));

        // ── AgentRunner (without fused engine — upgraded below after kg_store is available) ──
        let skill_creator_gateway = gateway.clone();
        let mut runner = glidinghorse::core::agent_runner::AgentRunner::new(
            gateway,
            skills.clone(),
            l2.clone(),
            l0.clone(),
            mm_for_runner,
            tmpl.clone(),
            agent_settings.clone(),
        )
        .with_prompt_loader(glidinghorse::core::prompt_loader::PromptLoader::new(
            Default::default(),
            tmpl.clone(),
        ))
        .with_workspace_root(workspace_root.clone())
        .with_token_optimization(settings.token_optimization.clone());

        // Create FusedRootCauseEngine backed by the shared unified Oxigraph store
        let unified_kg_store = unified.store();
        {
            let fused_kg = Arc::new(
                KnowledgeGraphStore::with_shared_store(unified_kg_store.clone())
                    .expect("Failed to create shared KG Store for FusedRootCauseEngine"),
            );
            let fused_rce = FusedRootCauseEngine::new(Some(graph_backend.clone()), Some(fused_kg));
            runner = runner.with_fused_root_cause_engine(fused_rce);
        }

        // ── Skill Discovery Engine (semantic skill search via Hyperspace) ──
        let discovery_engine = Arc::new(
            SkillDiscoveryEngine::new(skill_graph.clone()).with_vector_store(vector_store.clone()),
        );

        // ── FeatureExtractor (GNN topological features for causal analysis) ──
        use glidinghorse::graph_backend::SkillGraphFeatureGraph;
        let feature_graph =
            SkillGraphFeatureGraph::new(skill_graph.clone(), skill_graph_algorithms.clone());
        let feature_extractor = Arc::new(FeatureExtractor::new(Arc::new(feature_graph)));

        // ── CausalEngine (Bayesian causal inference on skill graph) ──
        let causal_model_store = Arc::new(CausalModelStore::new());
        let causal_engine = Arc::new(CausalEngine::new(causal_model_store, graph_backend.clone()));

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
            match WorkspaceMonitor::initialize(ws_config, Some(l2.clone()), Some(event_bus.clone())) {
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

        // 注入共享 Oxigraph Store 到 ToolExecutor（替换内部隔离的 KnowledgeGraphStore）
        {
            let mut executor = runner.tool_executor.write();
            executor.set_unified_kg_store(unified.store());
        }

        if let Some(ref wm) = workspace_monitor {
            let mut executor = runner.tool_executor.write();
            executor.set_workspace_monitor(wm.clone());
            wm.set_causal_engine(causal_engine.clone());
        }

        // 注入共享 SkillGraphStore 到 ToolExecutor（create_skill 工具使用）
        {
            let executor = runner.tool_executor.write();
            executor.set_shared_skill_graph(skill_graph.clone());
            executor.set_shared_skill_registry(skills.clone());
            executor.set_shared_skill_vector_store(vector_store.clone());
            executor.set_shared_skill_creator_gateway(skill_creator_gateway.clone());
            let trusted_builtins = skill_graph
                .list_all_skills()
                .into_iter()
                .filter(|skill| {
                    skill.security_info.as_ref().is_some_and(|info| {
                        info.source == glidinghorse::skill_graph::types::SkillSource::SystemBuiltin
                    })
                })
                .map(|skill| skill.skill_iri)
                .collect();
            executor.set_security_engine(Arc::new(SecurityEngine::with_whitelisted_skills(
                skill_graph.clone(),
                trusted_builtins,
            )));
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
        .with_discovery_engine(discovery_engine.clone())
        .with_perception_ontology_bridge(ontology_bridge.clone());

        let (prompt_tokens, completion_tokens, last_prompt_tokens, last_completion_tokens) =
            sa.token_usage_arcs();

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
            Some(Arc::new(tokio::sync::Mutex::new(Some(client))))
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
            skill_vectors_ready: AtomicBool::new(false),
            feature_extractor,
            causal_engine,
            evolution_engine,
            timeline,
            core_config,
            oxi_store: unified.store(),
            code_scan_exclude_patterns,
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

    async fn ensure_skill_vectors_indexed(&self) {
        // The graph is authoritative and may have been hydrated from L0
        // before this process started. Retry on a later task if embedding is
        // temporarily unavailable, regardless of whether the task came from
        // CLI, TUI, or resume.
        if !self.skill_vectors_ready.swap(true, Ordering::AcqRel) {
            match self.discovery_engine.index_all_skills().await {
                Ok(indexed) => info!(indexed, "Skill semantic index synchronized"),
                Err(error) => {
                    self.skill_vectors_ready.store(false, Ordering::Release);
                    warn!(%error, "Skill semantic index synchronization failed; will retry on next task");
                }
            }
        }
    }

    /// Incrementally refresh the shared code KG from workspace source files.
    /// Both one-shot CLI and TUI/resume call this method so entrypoint choice
    /// cannot decide whether a changed nested source file is represented.
    fn scan_workspace_code(&self) -> anyhow::Result<()> {
        let root = std::path::Path::new(&self.config.workspace);
        if !root.is_dir() {
            return Ok(());
        }
        let kg_store =
            KnowledgeGraphStore::with_shared_store(self.oxi_store.clone()).map_err(|error| {
                anyhow::anyhow!("Failed to create KG store for code analysis: {error}")
            })?;
        let files = collect_workspace_code_files(root, &self.code_scan_exclude_patterns)?;
        let mut analyzed = 0u32;
        let mut errors = 0u32;
        for path in files {
            match CodeAstExtractor::extract_incremental(
                &path.to_string_lossy(),
                "graph:code",
                &kg_store,
            ) {
                Ok(_) => analyzed += 1,
                Err(error) => {
                    warn!(path = %path.display(), %error, "Code analysis failed");
                    errors += 1;
                }
            }
        }
        if analyzed > 0 || errors > 0 {
            info!(
                analyzed,
                errors, "Auto code analysis complete on workspace files"
            );
        }
        Ok(())
    }

    pub async fn process_task(&mut self, user_input: &str) -> anyhow::Result<(String, TaskResult)> {
        self.ensure_skill_vectors_indexed().await;
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
        let ws_summary = self
            .workspace_monitor
            .as_ref()
            .and_then(|wm| wm.get_file_inventory_summary());

        self.scan_workspace_code()?;

        let result = if let Some(ref wf_path) = self.config.workflow_path {
            let wf_jsonld = std::fs::read_to_string(wf_path)
                .map_err(|e| anyhow::anyhow!("读取工作流文件 '{}' 失败: {}", wf_path, e))?;
            let ctx = glidinghorse::core::agent_runner::TaskContext::new(
                &task_iri,
                user_input,
                self.config.max_iterations,
            )
            .with_original_task(user_input)
            .with_workflow(&wf_jsonld);
            let ctx = if let Some(ref summary) = ws_summary {
                ctx.with_workspace_summary(summary)
            } else {
                ctx
            };
            self.sa
                .process_task_with_context(user_input, &task_iri, ctx)
                .await?
        } else {
            let ctx = glidinghorse::core::agent_runner::TaskContext::new(
                &task_iri,
                user_input,
                self.config.max_iterations,
            )
            .with_original_task(user_input);
            let ctx = if let Some(ref summary) = ws_summary {
                ctx.with_workspace_summary(summary)
            } else {
                ctx
            };
            self.sa
                .process_task_with_context(user_input, &task_iri, ctx)
                .await?
        };

        info!(
            task_iri = %task_iri,
            status = %result.status,
            turn_count = result.turn_count,
            tool_call_count = result.tool_call_count,
            "任务处理完成"
        );

        // Every interactive entry reaches the same transport-neutral terminal
        // event before product-specific evolution and persistence follow-ups.
        glidinghorse::core::TaskFinalizer::new(self.event_bus.clone())
            .finalize(&task_iri, &result)
            .await;

        // Record post-task metrics for skill evolution + causal analysis
        if let Ok(mut ee) = self.evolution_engine.try_lock() {
            let success = result.status == "completed" || result.status == "success";
            let mut affected_skill_iris = Vec::new();

            // Record actual action outcomes against the canonical SkillRegistry
            // IRI. Unknown tools remain observable but are never turned into a
            // fabricated graph node under a second IRI namespace.
            for action in &result.tracked_actions {
                let Some(skill_iri) = self.skills.skill_iri_for_tool_name(&action.tool_name) else {
                    warn!(
                        task_iri = %task_iri,
                        tool = %action.tool_name,
                        "No registered skill mapping for tracked tool; skipping skill evolution record"
                    );
                    continue;
                };

                let action_success = matches!(
                    action.status,
                    glidinghorse::core::tracked_action::ActionStatus::Success
                );
                let mut usage = glidinghorse::skill_graph::evolution::UsageRecord::new(
                    &skill_iri,
                    &task_iri,
                    &action.agent_role,
                    action_success,
                )
                .with_context_tag(&result.status)
                .with_context_tag(&format!("tool:{}", action.tool_name))
                .with_duration(action.duration_secs.ceil().min(u32::MAX as f64) as u32);

                if let Some(error) = action.error.as_deref() {
                    usage = usage.with_error(error);
                }

                if let Err(error) = ee.record_usage(usage) {
                    warn!(
                        task_iri = %task_iri,
                        skill_iri = %skill_iri,
                        error = %error,
                        "Failed to record skill evolution usage"
                    );
                } else if !affected_skill_iris.contains(&skill_iri) {
                    affected_skill_iris.push(skill_iri);
                }
            }

            // When task fails, record a failure event that triggers infer_root_cause
            if !success {
                let error_msg = if !result.errors.is_empty() {
                    result.errors.join("; ")
                } else {
                    format!("Task status: {}", result.status)
                };
                if affected_skill_iris.is_empty() {
                    warn!(
                        task_iri = %task_iri,
                        error = %error_msg,
                        "Task failed without a mapped tracked skill; no synthetic skill usage will be recorded"
                    );
                }

                // Phase 2: Extract causal root cause from evolution engine and
                // publish suggestions to the event bus for observability / downstream use.
                {
                    let suggestions = ee.get_pending_suggestions().to_vec();
                    if !suggestions.is_empty() {
                        let causal_summary = suggestions
                            .iter()
                            .map(|s| {
                                format!(
                                    "{} (conf={:.2}): {}",
                                    s.skill_iri, s.confidence, s.description
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("; ");
                        info!(
                            task_iri = %task_iri,
                            suggestion_count = suggestions.len(),
                            causal_summary = %causal_summary,
                            "故障因果分析完成，生成了演化建议"
                        );

                        // Publish causal analysis result to event bus
                        let serialized_suggestions: Vec<serde_json::Value> = suggestions
                            .iter()
                            .map(|s| {
                                serde_json::json!({
                                    "type": format!("{:?}", s.suggestion_type),
                                    "skill_iri": s.skill_iri,
                                    "description": s.description,
                                    "confidence": s.confidence,
                                    "approval": s.approval.status(),
                                })
                            })
                            .collect();
                        let _ = self
                            .event_bus
                            .emit(
                                &task_iri,
                                "causal_analysis",
                                "system:evolution_engine",
                                &serde_json::json!({
                                    "task_iri": task_iri,
                                    "status": result.status,
                                    "errors": result.errors,
                                    "suggestions": serialized_suggestions,
                                })
                                .to_string(),
                            )
                            .await;
                    }
                }

                // Preserve preventive actions as causal evidence. They are
                // not SkillLinks: the previous implementation tried to link
                // to a non-skill knowledge IRI and discarded the resulting
                // validation error.
                let mut preventive_actions = Vec::new();
                for skill_iri in &affected_skill_iris {
                    let actions = ee.suggest_preventive_action(skill_iri);
                    for action in &actions {
                        preventive_actions.push(serde_json::json!({
                            "skill_iri": skill_iri,
                            "action": action,
                        }));
                    }
                }
                if !preventive_actions.is_empty() {
                    let preventive_iri = format!("{}#preventive-actions", task_iri);
                    let preventive_json = serde_json::json!({
                        "@id": preventive_iri,
                        "@type": "CausalPreventiveActions",
                        "task_iri": task_iri,
                        "actions": preventive_actions,
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                    });
                    let event_id = self
                        .event_bus
                        .emit(
                            &task_iri,
                            "causal_preventive_actions",
                            "system:evolution_engine",
                            &preventive_json.to_string(),
                        )
                        .await;
                    info!(task_iri = %task_iri, %event_id, "Published causal preventive actions");
                    if let Err(error) = self.l2_bb.write_node(
                        &preventive_iri,
                        &preventive_json.to_string(),
                        &self.core_config,
                    ) {
                        warn!(task_iri = %task_iri, %error, "Failed to write causal preventive actions");
                    }
                }

                // Persist causal analysis to L2 blackboard for cross-task awareness
                {
                    let suggestions = ee.get_pending_suggestions().to_vec();
                    if !suggestions.is_empty() {
                        let causal_node_iri = format!("{}#causal", task_iri);
                        let all_suggestions: Vec<serde_json::Value> = suggestions
                            .iter()
                            .map(|s| {
                                serde_json::json!({
                                    "type": format!("{:?}", s.suggestion_type),
                                    "skill_iri": s.skill_iri,
                                    "description": s.description,
                                    "confidence": s.confidence,
                                    "approval": s.approval.status(),
                                })
                            })
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
                    // Persist only typed proposals. This makes suggestions
                    // reviewable across restart while deliberately avoiding
                    // auto-approval or graph mutation in the task path.
                    let proposal_store = EvolutionProposalStore::new(self.l0.clone());
                    let proposal_ids = improvements.iter().filter_map(|suggestion| {
                        if suggestion.patch.is_none() {
                            return None;
                        }
                        let serialized = serde_json::to_vec(suggestion).ok()?;
                        let proposal_key = format!(
                            "{}:evolution:{}",
                            task_iri,
                            hex::encode(Sha256::digest(serialized)),
                        );
                        match proposal_store.create_or_get(&proposal_key, suggestion.clone(), self.skill_graph.as_ref()) {
                            Ok(proposal) => Some(proposal.proposal_id),
                            Err(error) => {
                                warn!(task_iri = %task_iri, error = %error, "Failed to persist typed evolution proposal");
                                None
                            }
                        }
                    }).collect::<Vec<_>>();
                    let snapshot_json = serde_json::json!({
                        "@id": format!("{}#health-snapshot", task_iri),
                        "@type": "SkillHealthSnapshot",
                        "task_iri": task_iri,
                        "improvements": improvements.iter().map(|s| serde_json::json!({
                            "type": format!("{:?}", s.suggestion_type),
                            "skill_iri": s.skill_iri,
                            "description": s.description,
                            "confidence": s.confidence,
                            "approval": s.approval.status(),
                        })).collect::<Vec<_>>(),
                        "proposal_ids": proposal_ids,
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

        // ── Write task metadata to the unified KG store for knowledge evolution ──
        // This gives the KG auto-growing task records queryable via SPARQL.
        {
            use std::fmt::Write as FmtWrite;
            let mut sparql = String::from("PREFIX task: <https://agent-os.org/ontology/task#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n");
            let escape_sparql = |s: &str| -> String {
                s.replace('\\', "\\\\")
                    .replace('\'', "\\'")
                    .replace('\n', " ")
                    .replace('\r', " ")
            };
            let status_safe = escape_sparql(&result.status);
            let input_safe: String = user_input.chars().take(200).collect();
            let input_safe = escape_sparql(&input_safe);
            let _ = write!(sparql, "INSERT DATA {{ GRAPH <graph:tasks> {{ <{iri}> a task:Task ; task:status '{status}' ; task:input '{input}' ; task:turnCount \"{turns}\"^^xsd:integer . }} }}\n",
                iri = task_iri,
                status = status_safe,
                input = input_safe,
                turns = result.turn_count,
            );
            match self.l2_bb.sparql_update(&sparql) {
                Ok(_) => info!(task_iri = %task_iri, "Task metadata written to KG store"),
                Err(e) => {
                    warn!(task_iri = %task_iri, error = %e, "Failed to write task metadata to KG")
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
    pub fn token_arcs(
        &self,
    ) -> (
        Arc<AtomicU64>,
        Arc<AtomicU64>,
        Arc<AtomicU64>,
        Arc<AtomicU64>,
    ) {
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

    pub async fn list_checkpoints(
        &self,
    ) -> anyhow::Result<Vec<glidinghorse::core::checkpoint::CheckpointData>> {
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

    /// List durable, human-reviewable evolution proposals for this workspace.
    pub fn list_evolution_proposals(
        &self,
    ) -> anyhow::Result<Vec<glidinghorse::skill_graph::EvolutionProposal>> {
        Ok(EvolutionProposalStore::new(self.l0.clone()).list()?)
    }

    /// Record an explicit local-operator review. `approver` is an audit label,
    /// not an authentication mechanism.
    pub fn approve_evolution_proposal(
        &self,
        proposal_id: &str,
        approver: &str,
        comment: Option<String>,
    ) -> anyhow::Result<glidinghorse::skill_graph::EvolutionProposal> {
        Ok(EvolutionProposalStore::new(self.l0.clone()).approve(proposal_id, approver, comment)?)
    }

    pub fn validate_evolution_proposal(
        &self,
        proposal_id: &str,
    ) -> anyhow::Result<glidinghorse::skill_graph::EvolutionProposal> {
        Ok(EvolutionProposalStore::new(self.l0.clone())
            .validate_for_commit(proposal_id, self.skill_graph.as_ref())?)
    }

    /// Commit a governed link patch after durable approval and validation. No
    /// automatic call path invokes this method.
    pub fn commit_evolution_proposal(
        &self,
        proposal_id: &str,
    ) -> anyhow::Result<glidinghorse::skill_graph::EvolutionProposal> {
        Ok(EvolutionProposalStore::new(self.l0.clone())
            .commit_validated_link_patch(proposal_id, self.skill_graph.as_ref())?)
    }

    pub async fn resume_task(&mut self, task_iri: &str) -> anyhow::Result<TaskResult> {
        let cm =
            glidinghorse::core::checkpoint::CheckpointManager::with_persistence(self.l0.clone());
        let cp = cm
            .restore_latest(task_iri)?
            .ok_or_else(|| anyhow::anyhow!("没有找到 task_iri={} 的 checkpoint", task_iri))?;

        let _agent_state: serde_json::Value = serde_json::from_str(&cp.agent_state_json)?;

        let resume_input = format!(
            "继续执行之前中断的任务。上次进度: {}\n\n请从上次中断处继续。",
            cp.name
        );
        self.process_task_with_iri(&resume_input, task_iri).await
    }

    /// 从 checkpoint 恢复任务，包含完整的历史上下文消息
    pub async fn resume_task_with_messages(
        &mut self,
        task_iri: &str,
        resumed_messages: Vec<glidinghorse::gateway::unified_gateway::ChatMessage>,
    ) -> anyhow::Result<TaskResult> {
        let resume_input = "继续执行之前中断的任务。请从上次中断处继续。".to_string();
        self.process_task_with_iri_and_messages(&resume_input, task_iri, Some(resumed_messages))
            .await
    }

    /// Process a task with an externally-generated task IRI so the caller
    /// can emit supplementary input events during execution.
    pub async fn process_task_with_iri(
        &mut self,
        user_input: &str,
        task_iri: &str,
    ) -> anyhow::Result<TaskResult> {
        self.process_task_with_iri_and_messages(user_input, task_iri, None)
            .await
    }

    /// Process a task with optional resumed messages (for checkpoint resume)
    pub async fn process_task_with_iri_and_messages(
        &mut self,
        user_input: &str,
        task_iri: &str,
        resumed_messages: Option<Vec<glidinghorse::gateway::unified_gateway::ChatMessage>>,
    ) -> anyhow::Result<TaskResult> {
        self.ensure_skill_vectors_indexed().await;
        // Keep the code-KG precondition equivalent to the CLI task path.
        // Scan failure is surfaced rather than silently claiming an updated
        // code graph for a TUI/resume task.
        self.scan_workspace_code()?;
        // Lazy MCP connect — connect to registered servers on first task
        if let Some(ref handle) = self.mcp_client {
            let mut guard = handle.lock().await;
            if let Some(client) = guard.as_mut() {
                let needs_connect: Vec<String> = client
                    .list_servers()
                    .iter()
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
                    let tool_executor = self.sa.tool_executor();
                    let mut executor = tool_executor.write();
                    client.register_tools_to_tool_executor(&mut executor, handle.clone());
                }
            }
        }

        use glidinghorse::core::agent_runner::TaskContext;

        let ws_summary = self
            .workspace_monitor
            .as_ref()
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
            let tool_count = msgs
                .iter()
                .filter(|m| m.role == "tool" || m.tool_call_id.is_some())
                .count() as u32;
            ctx.with_resumed_messages(msgs, turn_count, tool_count)
        } else {
            ctx
        };

        let result = self
            .sa
            .process_task_with_context(user_input, task_iri, ctx)
            .await?;

        info!(
            task_iri = %task_iri,
            status = %result.status,
            turn_count = result.turn_count,
            tool_call_count = result.tool_call_count,
            "任务处理完成"
        );

        // TUI/resume uses this entry rather than `process_task`. Keep the
        // transport-neutral terminal event and task-KG record consistent with
        // the CLI path; CLI-specific AST/evolution follow-ups remain owned by
        // `process_task` until they are separately made context-independent.
        glidinghorse::core::TaskFinalizer::new(self.event_bus.clone())
            .finalize(task_iri, &result)
            .await;
        self.record_tui_task_evolution(task_iri, &result).await;
        self.write_task_metadata(task_iri, user_input, &result);

        // Snapshot the skill graph to the TimelineStore after each task,
        // enabling temporal rollback and traceability of graph evolution.
        let backend = SkillGraphSnapshotBackend::new(self.skill_graph.clone());
        self.timeline
            .create_snapshot(&backend, &format!("task:{}", result.status.as_str()));

        Ok(result)
    }

    fn write_task_metadata(&self, task_iri: &str, user_input: &str, result: &TaskResult) {
        use std::fmt::Write as FmtWrite;
        let mut sparql = String::from(
            "PREFIX task: <https://agent-os.org/ontology/task#>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        );
        let escape_sparql = |s: &str| -> String {
            s.replace('\\', "\\\\")
                .replace('\'', "\\'")
                .replace('\n', " ")
                .replace('\r', " ")
        };
        let status_safe = escape_sparql(&result.status);
        let input_safe = escape_sparql(&user_input.chars().take(200).collect::<String>());
        let _ = write!(
            sparql,
            "INSERT DATA {{ GRAPH <graph:tasks> {{ <{task_iri}> a task:Task ; task:status '{status_safe}' ; task:input '{input_safe}' ; task:turnCount \"{}\"^^xsd:integer . }} }}\n",
            result.turn_count,
        );
        match self.l2_bb.sparql_update(&sparql) {
            Ok(_) => info!(task_iri = %task_iri, "Task metadata written to KG store"),
            Err(error) => {
                warn!(task_iri = %task_iri, %error, "Failed to write task metadata to KG")
            }
        }
    }

    /// The TUI/resume path has no CLI-only workspace scan, but it must still
    /// contribute real tool outcomes to the shared skill graph and preserve
    /// typed suggestions for review. This deliberately has no approval or
    /// commit side effect.
    async fn record_tui_task_evolution(&self, task_iri: &str, result: &TaskResult) {
        let Ok(mut evolution) = self.evolution_engine.try_lock() else {
            return;
        };
        let mut affected_skill_iris = Vec::new();
        for action in &result.tracked_actions {
            let Some(skill_iri) = self.skills.skill_iri_for_tool_name(&action.tool_name) else {
                warn!(task_iri = %task_iri, tool = %action.tool_name, "No skill IRI for TUI tracked action");
                continue;
            };
            let succeeded = matches!(
                action.status,
                glidinghorse::core::tracked_action::ActionStatus::Success
            );
            let mut usage = glidinghorse::skill_graph::evolution::UsageRecord::new(
                &skill_iri,
                task_iri,
                &action.agent_role,
                succeeded,
            )
            .with_context_tag(&result.status)
            .with_context_tag(&format!("tool:{}", action.tool_name))
            .with_duration(action.duration_secs.ceil().min(u32::MAX as f64) as u32);
            if let Some(error) = action.error.as_deref() {
                usage = usage.with_error(error);
            }
            if let Err(error) = evolution.record_usage(usage) {
                warn!(task_iri = %task_iri, skill_iri = %skill_iri, %error, "Failed to record TUI skill usage");
            } else if !affected_skill_iris.contains(&skill_iri) {
                affected_skill_iris.push(skill_iri);
            }
        }

        let causal_suggestions = evolution.get_pending_suggestions().to_vec();
        if !causal_suggestions.is_empty() {
            let serialized = causal_suggestions
                .iter()
                .map(|suggestion| {
                    serde_json::json!({
                        "type": format!("{:?}", suggestion.suggestion_type),
                        "skill_iri": suggestion.skill_iri,
                        "description": suggestion.description,
                        "confidence": suggestion.confidence,
                        "approval": suggestion.approval.status(),
                    })
                })
                .collect::<Vec<_>>();
            let payload = serde_json::json!({
                "task_iri": task_iri,
                "status": result.status,
                "errors": result.errors,
                "suggestions": serialized,
                "entrypoint": "tui_or_resume",
            });
            let event_id = self
                .event_bus
                .emit(
                    task_iri,
                    "causal_analysis",
                    "system:evolution_engine",
                    &payload.to_string(),
                )
                .await;
            info!(task_iri = %task_iri, %event_id, "Published TUI causal analysis");
            let causal_iri = format!("{}#causal", task_iri);
            if let Err(error) =
                self.l2_bb
                    .write_node(&causal_iri, &payload.to_string(), &self.core_config)
            {
                warn!(task_iri = %task_iri, %error, "Failed to write TUI causal analysis");
            }
        }

        let preventive_actions = affected_skill_iris
            .iter()
            .flat_map(|skill_iri| {
                evolution.suggest_preventive_action(skill_iri).into_iter().map(move |action| {
                serde_json::json!({ "skill_iri": skill_iri, "action": action })
            })
            })
            .collect::<Vec<_>>();
        if !preventive_actions.is_empty() {
            let preventive_iri = format!("{}#preventive-actions", task_iri);
            let preventive = serde_json::json!({
                "@id": preventive_iri,
                "@type": "CausalPreventiveActions",
                "task_iri": task_iri,
                "actions": preventive_actions,
                "entrypoint": "tui_or_resume",
                "timestamp": chrono::Utc::now().to_rfc3339(),
            });
            let event_id = self
                .event_bus
                .emit(
                    task_iri,
                    "causal_preventive_actions",
                    "system:evolution_engine",
                    &preventive.to_string(),
                )
                .await;
            info!(task_iri = %task_iri, %event_id, "Published TUI causal preventive actions");
            if let Err(error) =
                self.l2_bb
                    .write_node(&preventive_iri, &preventive.to_string(), &self.core_config)
            {
                warn!(task_iri = %task_iri, %error, "Failed to write TUI causal preventive actions");
            }
        }

        let improvements = evolution.suggest_improvements().await;
        if improvements.is_empty() {
            return;
        }
        let proposal_store = EvolutionProposalStore::new(self.l0.clone());
        let proposal_ids = improvements.iter().filter_map(|suggestion| {
            let patch = suggestion.patch.as_ref()?;
            let serialized = serde_json::to_vec(suggestion).ok()?;
            let key = format!("{}:tui-evolution:{}", task_iri, hex::encode(Sha256::digest(serialized)));
            match proposal_store.create_or_get(&key, suggestion.clone(), self.skill_graph.as_ref()) {
                Ok(proposal) => Some(proposal.proposal_id),
                Err(error) => {
                    warn!(task_iri = %task_iri, patch = ?patch, %error, "Failed to persist TUI evolution proposal");
                    None
                }
            }
        }).collect::<Vec<_>>();
        let snapshot_iri = format!("{}#health-snapshot", task_iri);
        let snapshot = serde_json::json!({
            "@id": snapshot_iri,
            "@type": "SkillHealthSnapshot",
            "task_iri": task_iri,
            "proposal_ids": proposal_ids,
            "entrypoint": "tui_or_resume",
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        if let Err(error) =
            self.l2_bb
                .write_node(&snapshot_iri, &snapshot.to_string(), &self.core_config)
        {
            warn!(task_iri = %task_iri, %error, "Failed to write TUI skill health snapshot");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{collect_workspace_code_files, load_code_scan_exclusions};

    #[test]
    fn code_scan_honors_configured_and_gitignore_exclusions_recursively() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        std::fs::create_dir_all(workspace.path().join("src/nested")).unwrap();
        std::fs::create_dir_all(workspace.path().join("generated")).unwrap();
        std::fs::create_dir_all(workspace.path().join("configured_skip")).unwrap();
        std::fs::write(workspace.path().join("src/nested/kept.rs"), "fn kept() {}").unwrap();
        std::fs::write(workspace.path().join("generated/ignored.ts"), "export {};").unwrap();
        std::fs::write(workspace.path().join("configured_skip/ignored.py"), "pass").unwrap();
        std::fs::write(workspace.path().join("ignored.log"), "not source").unwrap();
        std::fs::write(
            workspace.path().join(".gitignore"),
            "generated/\n*.log\n!kept.rs\n",
        )
        .unwrap();

        let exclusions =
            load_code_scan_exclusions(workspace.path(), &["configured_skip/".to_string()]);
        let files = collect_workspace_code_files(workspace.path(), &exclusions)
            .expect("workspace scan should succeed");
        let relative = files
            .iter()
            .map(|path| {
                path.strip_prefix(workspace.path())
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            })
            .collect::<Vec<_>>();

        assert_eq!(relative, vec!["src/nested/kept.rs"]);
    }
}
